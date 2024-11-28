"""VASP calculator interface."""

# Copyright (C) 2011 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import io
import os
import sys
import warnings
import xml.etree.cElementTree as etree
import xml.parsers.expat
from collections import Counter
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np

from phonopy.file_IO import (
    get_io_module_to_decompress,
    write_FORCE_CONSTANTS,
    write_force_constants_to_hdf5,
)
from phonopy.structure.atoms import PhonopyAtoms, atom_data, symbol_map
from phonopy.structure.symmetry import elaborate_borns_and_epsilon
from phonopy.units import VaspToTHz


def check_forces(forces, num_atom, filename, verbose=True):
    """Check a set of forces and show message if it is wrong."""
    if len(forces) != num_atom:
        if verbose:
            print("")
            if isinstance(filename, io.IOBase):
                file_ptr_name = "forces"
            else:
                file_ptr_name = f'"{filename}"'
            stars = "*" * len(file_ptr_name)
            print(f"**************{stars}**************")
            print(f"***** Parsing {file_ptr_name} failed. *****")
            print(f"**************{stars}**************")
        return False
    else:
        return True


def get_drift_forces(forces, filename=None, verbose=True):
    """Calculate drift force and show it."""
    drift_force = np.sum(forces, axis=0) / len(forces)

    if verbose:
        if filename is None:
            print(
                "Drift force: %12.8f %12.8f %12.8f to be subtracted"
                % tuple(drift_force)
            )
        else:
            print('Drift force of "%s" to be subtracted' % filename)
            print("%12.8f %12.8f %12.8f" % tuple(drift_force))
        sys.stdout.flush()

    return drift_force


def get_scaled_positions_lines(scaled_positions):
    """Return text lines of scaled positions."""
    return "\n".join(_get_scaled_positions_lines(scaled_positions))


def sort_positions_by_symbols(symbols: Sequence, positions: np.ndarray):
    """Sort atomic positions by symbols.

    Sort positions by symbols (using the order defined by reduced_symbols)
    using a stable sort algorithm. Written by @ExpHP, refactored by @atztogo.

    symbols = ["A", "B", "A", "B"]
    reduced_symbols = ["A", "B"]
    sort_keys = [0, 1, 0, 1]
    perm = [0, 2, 1, 3]
    counts_dict = {'A': 2, 'B': 2}
    counts_list = [2, 2]

    Parameters
    ----------
    symbols : list[str] or list[int] or np.ndarray[int]
        Sequence of hashable objects. This may be a list of chemical symbols
        or numbers.
    positions : np.ndarray
        Atomic positions.

    Returns
    -------
    sorted_positions = positions[perm]
    For the others, see the example above.

    Functions
    ---------
    _argsort_stable :
        Alternative to `np.argsort(keys)` that uses a stable sorting algorithm
        so that indices tied for the same value are listed in increasing order.

    """

    def _argsort_stable(keys):
        # Python's built-in sort algorithm is a stable sort
        return sorted(range(len(keys)), key=keys.__getitem__)

    # dict in Python 3.7 or later is ordered dict.
    reduced_symbols = list(dict.fromkeys(symbols))
    counts_dict = Counter(symbols)
    # list(counts_dict.values()) may be used...
    counts_list = [counts_dict[s] for s in reduced_symbols]
    sort_keys = [reduced_symbols.index(i) for i in symbols]
    perm = _argsort_stable(sort_keys)
    sorted_positions = positions[perm]

    return counts_list, reduced_symbols, sorted_positions, perm


def _get_forces_points_and_energy(
    fp: io.IOBase,
    use_expat: bool = True,
    filename: Optional[Union[str, os.PathLike]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
    vasprun = Vasprun(fp, use_expat=use_expat)
    try:
        forces = vasprun.read_forces()
        points = vasprun.read_points()
        energy = vasprun.read_energy()
    except (RuntimeError, ValueError, xml.parsers.expat.ExpatError) as err:
        msg = (
            "Probably this vasprun.xml "
            "is broken or some value diverges. Check this "
            "calculation carefully before sending questions to the "
            "phonopy mailing list."
        )
        if filename is not None:
            msg = f'Could not parse "{filename}". ' + msg
        raise RuntimeError(msg) from err
    return forces, points, energy


def parse_set_of_forces(
    num_atoms: int,
    forces_filenames: Sequence[Union[str, bytes, os.PathLike, io.IOBase]],
    use_expat: bool = True,
    verbose: bool = True,
) -> dict:
    """Parse sets of forces of files."""
    if verbose:
        print("counter (file index): ", end="")

    is_parsed = True
    force_sets = []
    point_sets = []
    energy_sets = []

    for i, fp in enumerate(forces_filenames):
        if verbose:
            print(f"{i + 1}", end=" ")
        if isinstance(fp, io.IOBase):
            forces, points, energy = _get_forces_points_and_energy(
                fp, use_expat=use_expat
            )
        else:
            myio = get_io_module_to_decompress(fp)
            with myio.open(fp, "rb") as _fp:
                forces, points, energy = _get_forces_points_and_energy(
                    _fp, use_expat=use_expat, filename=fp
                )
        force_sets.append(forces)
        energy_sets.append(energy)
        point_sets.append(points)

        if not check_forces(force_sets[-1], num_atoms, fp):
            is_parsed = False

    if verbose:
        print("")

    if is_parsed:
        return {
            "forces": force_sets,
            "points": point_sets,
            "supercell_energies": energy_sets,
        }
    else:
        return {}


def create_FORCE_CONSTANTS(filename, is_hdf5, log_level):
    """Parse vasprun.xml and write it into force constants file."""
    force_constants, atom_types = parse_force_constants(filename)

    if force_constants is None:
        print("")
        print("'%s' dones not contain necessary information." % filename)
        return 1

    if is_hdf5:
        try:
            import h5py  # noqa F401
        except ImportError as exc:
            raise ModuleNotFoundError("You need to install python-h5py.") from exc

        write_force_constants_to_hdf5(force_constants)
        if log_level > 0:
            print("force_constants.hdf5 has been created from vasprun.xml.")
    else:
        write_FORCE_CONSTANTS(force_constants)
        if log_level > 0:
            print("FORCE_CONSTANTS has been created from vasprun.xml.")

    if log_level > 0:
        print("Atom types: %s" % (" ".join(atom_types)))
    return 0


def parse_force_constants(filename):
    """Return force constants and chemical elements.

    Parameters
    ----------
    filename : str
       File name.

    Returns
    -------
    tuple :
        force constants and chemical elements

    """
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rb") as f:
        vasprun = Vasprun(f)
        fc = vasprun.read_force_constants()
    return fc


#
# read VASP POSCAR
#
def read_vasp(filename, symbols=None):
    """Parse POSCAR type file."""
    with open(filename) as infile:
        lines = infile.readlines()
    return _get_atoms_from_poscar(lines, symbols)


def read_vasp_from_strings(strings, symbols=None):
    """Parse POSCAR type string."""
    return _get_atoms_from_poscar(io.StringIO(strings).readlines(), symbols)


def _get_atoms_from_poscar(lines, symbols):
    line1 = [x for x in lines[0].split()]
    if _is_exist_symbols(line1):
        symbols = line1

    scale = float(lines[1])

    cell = []
    for i in range(2, 5):
        cell.append([float(x) for x in lines[i].split()[:3]])
    cell = np.array(cell) * scale

    try:
        num_atoms = np.array([int(x) for x in lines[5].split()])
        line_at = 6
    except ValueError:
        symbols = [x for x in lines[5].split()]
        num_atoms = np.array([int(x) for x in lines[6].split()])
        line_at = 7

    expaned_symbols = _expand_symbols(num_atoms, symbols)

    if lines[line_at][0].lower() == "s":
        line_at += 1

    is_scaled = True
    if lines[line_at][0].lower() == "c" or lines[line_at][0].lower() == "k":
        is_scaled = False

    line_at += 1

    positions = []
    for i in range(line_at, line_at + num_atoms.sum()):
        positions.append([float(x) for x in lines[i].split()[:3]])

    if is_scaled:
        atoms = PhonopyAtoms(
            symbols=expaned_symbols, cell=cell, scaled_positions=positions
        )
    else:
        atoms = PhonopyAtoms(symbols=expaned_symbols, cell=cell, positions=positions)

    return atoms


def _is_exist_symbols(symbols):
    for s in symbols:
        if s not in symbol_map:
            return False
    return True


def _expand_symbols(num_atoms, symbols=None):
    expanded_symbols = []
    is_symbols = True
    if symbols is None:
        is_symbols = False
    else:
        if len(symbols) != len(num_atoms):
            is_symbols = False
        else:
            for s in symbols:
                if s not in symbol_map:
                    is_symbols = False
                    break

    if is_symbols:
        for s, num in zip(symbols, num_atoms):
            expanded_symbols += [s] * num
    else:
        for i, num in enumerate(num_atoms):
            expanded_symbols += [atom_data[i + 1][1]] * num

    return expanded_symbols


#
# write vasp POSCAR
#
def write_vasp(filename, cell, direct=True):
    """Write crystal structure to a VASP POSCAR style file.

    Parameters
    ----------
    filename : str
        Filename.
    cell : PhonopyAtoms
        Crystal structure.
    direct : bool, optional
        In 'Direct' or not in VASP POSCAR format. Default is True.

    """
    lines = get_vasp_structure_lines(cell, direct=direct)
    with open(filename, "w") as w:
        w.write("\n".join(lines))


def get_vasp_structure_lines(
    cell: PhonopyAtoms,
    direct: bool = True,
    is_vasp5: bool = True,
    is_vasp4: bool = False,
    first_line_str: Optional[str] = None,
):
    """Generate POSCAR text lines as a list from PhonopyAtoms instance.

    direct : bool
        Dummy argument. This doesn nothing.
    is_vasp5 : bool
        Deprecated. This is replaced by ``is_vasp4 = not is_vasp5``.

    """
    _is_vasp4 = is_vasp4
    if is_vasp5 is False:
        warnings.warn(
            "is_vasp5 parameter is deprecated. "
            "Use is_vasp4=True instead of is_vasp5=False",
            DeprecationWarning,
            stacklevel=2,
        )
        _is_vasp4 = True
    if direct is False:
        warnings.warn(
            "direct=False is not supported. ", DeprecationWarning, stacklevel=2
        )
    lines, scaled_positions = _get_vasp_structure_header_lines(
        cell, is_vasp4=_is_vasp4, first_line_str=first_line_str
    )
    lines.append("Direct")
    lines += _get_scaled_positions_lines(scaled_positions)

    # VASP compiled on some system, ending by \n is necessary to read POSCAR
    # properly.
    lines.append("")

    return lines


def write_supercells_with_displacements(
    supercell, cells_with_displacements, ids, pre_filename="POSCAR", width=3
):
    """Write supercells with displacements to files."""
    write_vasp("S%s" % pre_filename, supercell, direct=True)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}".format(
            i, pre_filename=pre_filename, width=width
        )
        write_vasp(filename, cell, direct=True)


def _get_vasp_structure_header_lines(
    cell: PhonopyAtoms, is_vasp4: bool = False, first_line_str: Optional[str] = None
):
    (num_atoms, symbols, scaled_positions, _) = sort_positions_by_symbols(
        cell.symbols, cell.scaled_positions
    )
    lines = []
    if is_vasp4:
        lines.append(" ".join(["%s" % s for s in symbols]))
    elif first_line_str is None:
        lines.append("generated by phonopy")
    else:
        lines.append(first_line_str)
    lines.append("   1.0")
    for a in cell.cell:
        lines.append("  %21.16f %21.16f %21.16f" % tuple(a))
    if not is_vasp4:
        lines.append(" ".join(["%s" % s for s in symbols]))
    lines.append(" ".join(["%4d" % n for n in num_atoms]))
    return lines, scaled_positions


def _get_scaled_positions_lines(scaled_positions):
    # map into 0 <= x < 1.
    # (the purpose of the second '% 1' is to handle a surprising
    #  edge case for small negative numbers: '-1e-30 % 1 == 1.0')
    unit_positions = scaled_positions % 1 % 1

    return [
        " %19.16f %19.16f %19.16f" % tuple(vec)
        for vec in unit_positions.tolist()  # lists are faster for iteration
    ]


#
# Non-analytical term
#
def get_born_vasprunxml(
    filename="vasprun.xml",
    primitive_matrix=None,
    supercell_matrix=None,
    is_symmetry=True,
    symmetrize_tensors=False,
    symprec=1e-5,
):
    """Parse vasprun.xml to get NAC parameters.

    In phonopy, primitive cell is created through the path of
    unit cell -> supercell -> primitive cell. To trace this path exactly,
    `primitive_matrix` and `supercell_matrix` can be given, but these are
    optional.

    Returns
    -------
    See elaborate_borns_and_epsilon.

    """
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rb") as f:
        vasprun = VasprunxmlExpat(f)
        try:
            vasprun.parse()
        except xml.parsers.expat.ExpatError as exc:
            raise xml.parsers.expat.ExpatError(
                'Could not parse "%s". Please check the content.' % filename
            ) from exc
        except ValueError as exc:
            raise ValueError(
                'Could not parse "%s". Please check the content.' % filename
            ) from exc

    return elaborate_borns_and_epsilon(
        vasprun.cell,
        vasprun.born,
        vasprun.epsilon,
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        is_symmetry=is_symmetry,
        symmetrize_tensors=symmetrize_tensors,
        symprec=symprec,
    )


def get_born_OUTCAR(
    poscar_filename="POSCAR",
    outcar_filename=None,
    primitive_matrix=None,
    supercell_matrix=None,
    is_symmetry=True,
    symmetrize_tensors=False,
    symprec=1e-5,
):
    """Parse OUTCAR to get NAC parameters.

    Returns
    -------
    See elaborate_borns_and_epsilon.

    """
    if outcar_filename is None:
        filename = "OUTCAR"
    else:
        filename = outcar_filename

    ucell = read_vasp(poscar_filename)
    borns, epsilon = _read_born_and_epsilon_from_OUTCAR(filename)
    if len(borns) == 0 or len(epsilon) == 0:
        return None
    else:
        return elaborate_borns_and_epsilon(
            ucell,
            borns,
            epsilon,
            primitive_matrix=primitive_matrix,
            supercell_matrix=supercell_matrix,
            is_symmetry=is_symmetry,
            symmetrize_tensors=symmetrize_tensors,
            symprec=symprec,
        )


def _read_born_and_epsilon_from_OUTCAR(filename):
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, mode="rt") as outcar:
        borns = []
        epsilon = []

        while True:
            line = outcar.readline()
            if not line:
                break

            if "NIONS" in line:
                num_atom = int(line.split()[11])

            if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
                epsilon = []
                outcar.readline()
                epsilon.append([float(x) for x in outcar.readline().split()])
                epsilon.append([float(x) for x in outcar.readline().split()])
                epsilon.append([float(x) for x in outcar.readline().split()])

            if "BORN" in line:
                outcar.readline()
                line = outcar.readline()
                if "ion" in line:
                    for _ in range(num_atom):
                        born = []
                        born.append([float(x) for x in outcar.readline().split()][1:])
                        born.append([float(x) for x in outcar.readline().split()][1:])
                        born.append([float(x) for x in outcar.readline().split()][1:])
                        outcar.readline()
                        borns.append(born)

        borns = np.array(borns, dtype="double")
        epsilon = np.array(epsilon, dtype="double")

    return borns, epsilon


#
# vasprun.xml handling
#
class VasprunWrapper:
    """VasprunWrapper class.

    This is used to fix broken vasprun.xml of VASP 5.2.8 at PRECFOCK.

    """

    def __init__(self, fileptr):
        """Init method."""
        self._fileptr = fileptr

    def read(self, size=None):
        """Replace broken PRECFOCK."""
        element = self._fileptr.next()
        if element.find("PRECFOCK") == -1:
            return element
        else:
            return '<i type="string" name="PRECFOCK"></i>'


class Vasprun:
    """vasprun.xml parser class."""

    def __init__(self, fileptr, use_expat=False):
        """Init method."""
        self._fileptr = fileptr
        self._use_expat = use_expat
        self._vasprun_expat = None

    def read_forces(self) -> np.ndarray:
        """Read forces either using expat or etree."""
        if self._use_expat:
            return self._parse_expat_vasprun_xml(target="forces")
        else:
            vasprun_etree = self._parse_etree_vasprun_xml(tag="varray")
            return self._get_forces(vasprun_etree)

    def read_points(self) -> Optional[np.ndarray]:
        """Read forces either using expat or etree."""
        if self._use_expat:
            return self._parse_expat_vasprun_xml(target="points")
        else:
            return None

    def read_energy(self) -> Optional[float]:
        """Read energy using expat and etree is not supported."""
        if self._use_expat:
            return self._parse_expat_vasprun_xml(target="energy")
        else:
            return None

    def read_force_constants(self):
        """Read force constants using etree."""
        vasprun = self._parse_etree_vasprun_xml()
        return self._get_force_constants(vasprun)

    def _get_forces(self, vasprun_etree):
        """Return forces using etree.

        vasprun_etree = etree.iterparse(fileptr, tag='varray')

        """
        forces = []
        for _, element in vasprun_etree:
            if element.attrib["name"] == "forces":
                for v in element:
                    forces.append([float(x) for x in v.text.split()])
        return np.array(forces)

    def _get_force_constants(
        self, vasprun_etree
    ) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
        """Read hessian and calculate force constants.

        Hessian elements include sqrt(mass_a * mass_b) of two atoms a and b.
        To obtain force constants, atomic masses are parsed from vasprun.xml
        and those sqrt VASP masses are multiplied to the Hessian elements.

        In VASP-6, the unit is transformed to THz^2. To recover the unit of
        eV/Angstrom^2, the unit conversion factor is multiplied.

        """
        fc_tmp = None
        hessian_units = ""
        num_atom = 0
        for _, element in vasprun_etree:
            if num_atom == 0:
                atomtypes = self._get_atomtypes(element)
                if atomtypes:
                    num_atoms, elements, elem_masses = atomtypes[:3]
                    num_atom = np.sum(num_atoms)
                    masses = []
                    for n, m in zip(num_atoms, elem_masses):
                        masses += [m] * n

            # Get dynmat node
            if element.tag == "dynmat":
                # Get Hessian matrix (normalized by masses)
                v_elements = element.findall("./varray[@name='hessian']/v")
                if v_elements is not None:
                    fc_tmp = np.array(
                        [[float(x) for x in v.text.strip().split()] for v in v_elements]
                    )

                # Get physical units of Hessian
                unit_element = element.find("./i[@name='unit']")
                if unit_element is not None:
                    hessian_units = unit_element.text.strip()

            # Stop parsing when we have all the information
            if num_atom > 0 and fc_tmp is not None:
                break

        if fc_tmp is None:
            return None, None
        else:
            if fc_tmp.shape != (num_atom * 3, num_atom * 3):
                return False

            force_constants = np.zeros((num_atom, num_atom, 3, 3), dtype="double")
            for i in range(num_atom):
                for j in range(num_atom):
                    force_constants[i, j] = fc_tmp[
                        i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3
                    ]

            # Inverse normalization by atomic weights
            for i in range(num_atom):
                for j in range(num_atom):
                    force_constants[i, j] *= -np.sqrt(masses[i] * masses[j])

            # Recover the unit of eV/Angstrom^2 for VASP-6.
            if hessian_units == "THz^2":
                force_constants /= VaspToTHz**2

            return force_constants, elements

    def _get_atomtypes(self, element):
        atom_types = []
        masses = []
        valences = []
        num_atoms = []

        if element.tag == "atominfo":
            rc_elements = element.findall("./array[@name='atomtypes']/set/rc")
            if rc_elements is not None:
                for rc in rc_elements:
                    atom_info = [x.text for x in rc.findall("./c")]
                    num_atoms.append(int(atom_info[0]))
                    atom_types.append(atom_info[1].strip())
                    masses.append(float(atom_info[2]))
                    valences.append(float(atom_info[3]))
                return num_atoms, atom_types, masses, valences

        return None

    def _parse_etree_vasprun_xml(self, tag=None):
        if self._is_version528():
            return self._parse_by_etree(VasprunWrapper(self._fileptr), tag=tag)
        else:
            return self._parse_by_etree(self._fileptr, tag=tag)

    def _parse_by_etree(self, fileptr, tag=None):
        for event, elem in etree.iterparse(fileptr):
            if tag is None or elem.tag == tag:
                yield event, elem

    def _parse_expat_vasprun_xml(
        self, target: Literal["forces", "points", "energy"] = "forces"
    ) -> Union[np.ndarray, float]:
        if self._is_version528():
            return self._parse_by_expat(VasprunWrapper(self._fileptr), target=target)
        else:
            return self._parse_by_expat(self._fileptr, target=target)

    def _parse_by_expat(
        self,
        fileptr: io.IOBase,
        target: Literal["forces", "points", "energy"] = "forces",
    ) -> Union[np.ndarray, float]:
        if self._vasprun_expat is None:
            self._vasprun_expat = VasprunxmlExpat(fileptr)
            try:
                self._vasprun_expat.parse()
            except xml.parsers.expat.ExpatError:
                raise
            except ValueError:
                raise
            except Exception:
                raise

        if target == "forces":
            return self._vasprun_expat.forces[-1]
        if target == "points":
            return self._vasprun_expat.points[-1]
        if target == "energy":
            return float(self._vasprun_expat.energies[-1][1])

    def _is_version528(self):
        for line in self._fileptr:
            if '"version"' in str(line):
                self._fileptr.seek(0)
                if "5.2.8" in str(line):
                    sys.stdout.write(
                        "\n"
                        "**********************************************\n"
                        "* A special routine was used for VASP 5.2.8. *\n"
                        "**********************************************\n"
                    )
                    return True
                else:
                    return False


class VasprunxmlExpat:
    """Class to parse vasprun.xml by Expat."""

    def __init__(self, fileptr: io.IOBase):
        """Init method.

        Parameters
        ----------
        fileptr: binary stream
            E.g., by open(filename, "rb")

        """
        import xml.parsers.expat

        self._fileptr = fileptr

        self._is_forces = False
        self._is_stress = False
        self._is_positions = False
        self._is_symbols = False
        self._is_basis = False
        self._is_volume = False
        self._is_energy = False
        self._is_k_weights = False
        self._is_kpoints = False
        self._is_kpointlist = False
        self._is_eigenvalues = False
        self._is_epsilon = False
        self._is_born = False
        self._is_efermi = False
        self._is_generation = False
        self._is_divisions = False
        self._is_NELECT = False
        self._is_NGXYZ = [False, False, False]
        self._is_NGXYZF = [False, False, False]

        self._is_v = False
        self._is_i = False
        self._is_rc = False
        self._is_c = False
        self._is_set = False
        self._is_r = False
        self._is_field = False
        self._is_separator = False
        self._is_grids = False

        self._is_scstep = False
        self._is_structure = False
        self._is_projected = False
        self._is_proj_eig = False
        self._is_field_string = False
        self._is_pseudopotential = False

        self._all_forces = []
        self._all_stress = []
        self._all_points = []
        self._all_lattice = []
        self._all_energies = []
        self._all_volumes = []
        self._born = []
        self._forces = None
        self._stress = None
        self._points = None
        self._lattice = None
        self._energies = None
        self._epsilon = None
        self._born_atom = None
        self._k_weights = None
        self._kpointlist = None
        self._k_mesh = None
        self._eigenvalues = None
        self._eig_state = [0, 0]
        self._projectors = None
        self._proj_state = [0, 0, 0]
        self._field_val = None
        self._pseudopotentials = []
        self._ps_atom = None
        self._fft_grid = [0, 0, 0]
        self._fft_fine_grid = [0, 0, 0]
        self._efermi = None
        self._symbols = None
        self._NELECT = None

        self._p = xml.parsers.expat.ParserCreate()
        self._p.buffer_text = True
        self._p.StartElementHandler = self._start_element
        self._p.EndElementHandler = self._end_element
        self._p.CharacterDataHandler = self._char_data

        self._cbuf = None

    def parse(self):
        """Parse file pointer of vasprun.xml."""
        self._p.ParseFile(self._fileptr)

    @property
    def forces(self):
        """Return forces."""
        return np.array(self._all_forces, dtype="double", order="C")

    def get_forces(self):
        """Return forces."""
        warnings.warn(
            "VasprunxmlExpat.get_forces()) is deprecated. "
            "Use VasprunxmlExpat.forces attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.forces

    @property
    def stress(self):
        """Return stress tensor."""
        return np.array(self._all_stress, dtype="double", order="C")

    def get_stress(self):
        """Return stress tensor."""
        warnings.warn(
            "VasprunxmlExpat.get_stress()) is deprecated. "
            "Use VasprunxmlExpat.stress attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stress

    @property
    def epsilon(self):
        """Return dielectric constant tensor."""
        return np.array(self._epsilon, dtype="double", order="C")

    def get_epsilon(self):
        """Return dielectric constant tensor."""
        warnings.warn(
            "VasprunxmlExpat.get_epsilon()) is deprecated. "
            "Use VasprunxmlExpat.epsilon attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.epsilon

    @property
    def efermi(self):
        """Return Fermi energy."""
        return self._efermi

    def get_efermi(self):
        """Return efermi."""
        warnings.warn(
            "VasprunxmlExpat.get_efermi()) is deprecated. "
            "Use VasprunxmlExpat.efermi attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._efermi

    @property
    def born(self):
        """Return Born effective charges."""
        return np.array(self._born, dtype="double", order="C")

    def get_born(self):
        """Return Born effective charges."""
        return self.born

    @property
    def points(self):
        """Return all atomic positions of structure optimization steps."""
        return np.array(self._all_points, dtype="double", order="C")

    def get_points(self):
        """Return all atomic positions of structure optimization steps."""
        warnings.warn(
            "VasprunxmlExpat.get_points()) is deprecated. "
            "Use VasprunxmlExpat.points attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.points

    @property
    def lattice(self):
        """Return all basis vectors of structure optimization steps.

        Each basis vectors are in row vectors (a, b, c)

        """
        return np.array(self._all_lattice, dtype="double", order="C")

    def get_lattice(self):
        """Return all basis vectors of structure optimization steps."""
        warnings.warn(
            "VasprunxmlExpat.get_lattice()) is deprecated. "
            "Use VasprunxmlExpat.lattice attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.lattice

    @property
    def volume(self):
        """Return all cell volumes of structure optimization steps."""
        return np.array(self._all_volumes, dtype="double")

    @property
    def symbols(self):
        """Return atomic symbols."""
        return self._symbols

    def get_symbols(self):
        """Return atomic symbols."""
        warnings.warn(
            "VasprunxmlExpat.get_symbols()) is deprecated. "
            "Use VasprunxmlExpat.symbols attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._symbols

    @property
    def energies(self):
        """Return energies.

        Returns
        -------
        ndarray
            dtype='double'
            shape=(structure opt. steps, 3)
            [free energy TOTEN, energy(sigma->0), entropy T*S EENTRO]

        """
        return np.array(self._all_energies, dtype="double", order="C")

    def get_energies(self):
        """Return energies."""
        warnings.warn(
            "VasprunxmlExpat.get_energies()) is deprecated. "
            "Use VasprunxmlExpat.energies attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.energies

    @property
    def k_mesh(self):
        """Return k_mesh."""
        return np.array(self._k_mesh, dtype="intc")

    @property
    def kpointlist(self):
        """Return kpoint list."""
        return np.array(self._kpointlist, dtype="double")

    @property
    def k_weights(self):
        """Return k_weights.

        Returns
        -------
        ndarray
            Geometric k-point weights. The sum is normalized to 1, i.e.,
            Number of arms of k-star in BZ divided by number of all k-points.
            dtype='double'
            shape=(irreducible_kpoints,)

        """
        return np.array(self._k_weights, dtype="double")

    def get_k_weights(self):
        """Return k_weights."""
        warnings.warn(
            "VasprunxmlExpat.get_k_weights()) is deprecated. "
            "Use VasprunxmlExpat.k_weights attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.k_weights

    @property
    def k_weights_int(self):
        """Return k_weights in integers.

        Returns
        -------
        ndarray
            Geometric k-point weights (number of arms of k-star in BZ).
            dtype='intc'
            shape=(irreducible_kpoints,)

        """
        nk = np.prod(self.k_mesh)
        _weights = self.k_weights * nk
        weights = np.rint(_weights).astype("intc")
        assert (np.abs(weights - _weights) < 1e-7 * nk).all()
        return np.array(weights, dtype="intc")

    @property
    def eigenvalues(self):
        """Return eigenvalues.

        Returns
        -------
        ndarray
            Eigenvalues and occupations (the last index)
            dtype='double'
            shape=(spin, kpoints, bands, 2)

        """
        return np.array(self._eigenvalues, dtype="double", order="C")

    def get_eigenvalues(self):
        """Return eigenvalues."""
        warnings.warn(
            "VasprunxmlExpat.get_eigenvalues()) is deprecated. "
            "Use VasprunxmlExpat.eigenvalues attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.eigenvalues

    @property
    def projectors(self):
        """Return projectors."""
        return self._projectors

    def get_projectors(self):
        """Return projectors."""
        warnings.warn(
            "VasprunxmlExpat.get_projectors()) is deprecated. "
            "Use VasprunxmlExpat.projectors attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.projectors

    @property
    def pseudopotentials(self):
        """Return pseudo potential information.

        Example:
            [[2, u'N', 14.001, 5.0, u'PAW_PBE N 08Apr2002'],
             [2, u'Ga', 69.723, 13.0, u'PAW_PBE Ga_d 06Jul2010']]

        """
        return self._pseudopotentials

    def get_pseudopotentials(self):
        """Return pseudo potential information."""
        warnings.warn(
            "VasprunxmlExpat.get_pseudopotentials()) is deprecated. "
            "Use VasprunxmlExpat.pseudopotentials attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.pseudopotentials

    @property
    def cell(self):
        """Return cell in PhonopyAtoms."""
        return PhonopyAtoms(
            symbols=self._symbols,
            scaled_positions=self.points[-1],
            cell=self.lattice[-1],
        )

    @property
    def fft_grid(self):
        """Return FFT gird [NGX, NGY, NGZ]."""
        return self._fft_grid

    @property
    def fft_fine_grid(self):
        """Return fine FFT gird [NGXF, NGYF, NGZF]."""
        return self._fft_fine_grid

    @property
    def NELECT(self):
        """Return number of electrons, NELECT."""
        return self._NELECT

    def _start_element(self, name, attrs):
        # Used not to collect energies in <scstep>
        if name == "scstep":
            self._is_scstep = True

        # Used not to collect basis and positions in
        # <structure name="initialpos" >
        # <structure name="finalpos" >
        if name == "structure":
            if "name" in attrs.keys():
                self._is_structure = True

        if (
            self._is_forces
            or self._is_stress
            or self._is_epsilon
            or self._is_born
            or self._is_positions
            or self._is_basis
            or self._is_volume
            or self._is_k_weights
            or self._is_kpointlist
            or self._is_generation
        ):
            if name == "v":
                self._cbuf = ""
                self._is_v = True
                if "name" in attrs.keys():
                    if attrs["name"] == "divisions":
                        self._is_divisions = True

        if name == "varray":
            if "name" in attrs.keys():
                if attrs["name"] == "forces":
                    self._is_forces = True
                    self._forces = []

                if attrs["name"] == "stress":
                    self._is_stress = True
                    self._stress = []

                if attrs["name"] == "weights":
                    self._is_k_weights = True
                    self._k_weights = []

                if attrs["name"] == "kpointlist":
                    self._is_kpointlist = True
                    self._kpointlist = []

                if attrs["name"] == "epsilon" or attrs["name"] == "epsilon_scf":
                    self._is_epsilon = True
                    self._epsilon = []

                if not self._is_structure:
                    if attrs["name"] == "positions":
                        self._is_positions = True
                        self._points = []

                    if attrs["name"] == "basis":
                        self._is_basis = True
                        self._lattice = []

        if name == "kpoints":
            self._is_kpoints = True

        if name == "field":
            if "type" in attrs:
                self._cbuf = ""
                if attrs["type"] == "string":
                    self._is_field_string = True
            else:
                self._is_field = True

        if name == "generation":
            self._is_generation = True

        if name == "i":
            if "name" in attrs.keys():
                self._cbuf = ""
                if attrs["name"] == "efermi":
                    self._is_i = True
                    self._is_efermi = True
                if attrs["name"] == "NELECT":
                    self._is_i = True
                    self._is_NELECT = True
                if not self._is_structure and attrs["name"] == "volume":
                    self._is_i = True
                    self._is_volume = True
                if attrs["name"] == "NGX":
                    self._is_i = True
                    self._is_NGXYZ[0] = True
                if attrs["name"] == "NGY":
                    self._is_i = True
                    self._is_NGXYZ[1] = True
                if attrs["name"] == "NGZ":
                    self._is_i = True
                    self._is_NGXYZ[2] = True
                if attrs["name"] == "NGXF":
                    self._is_i = True
                    self._is_NGXYZF[0] = True
                if attrs["name"] == "NGYF":
                    self._is_i = True
                    self._is_NGXYZF[1] = True
                if attrs["name"] == "NGZF":
                    self._is_i = True
                    self._is_NGXYZF[2] = True

        if self._is_energy and name == "i":
            self._cbuf = ""
            self._is_i = True

        if name == "energy" and (not self._is_scstep):
            self._is_energy = True
            self._energies = []

        if self._is_symbols and name == "rc":
            self._is_rc = True

        if self._is_symbols and self._is_rc and name == "c":
            self._cbuf = ""
            self._is_c = True

        if self._is_born and name == "set":
            self._is_set = True
            self._born_atom = []

        if self._field_val == "pseudopotential":
            if name == "set":
                self._is_set = True
            if name == "rc" and self._is_set:
                self._is_rc = True
                self._ps_atom = []
            if name == "c":
                self._cbuf = ""
                self._is_c = True

        if name == "array":
            if "name" in attrs.keys():
                if attrs["name"] == "atoms":
                    self._is_symbols = True
                    self._symbols = []

                if attrs["name"] == "born_charges":
                    self._is_born = True

        if self._is_projected and not self._is_proj_eig:
            if name == "set":
                if "comment" in attrs.keys():
                    if "spin" in attrs["comment"]:
                        self._projectors.append([])
                        spin_num = int(attrs["comment"].replace("spin", ""))
                        self._proj_state = [spin_num - 1, -1, -1]
                    if "kpoint" in attrs["comment"]:
                        self._projectors[self._proj_state[0]].append([])
                        k_num = int(attrs["comment"].split()[1])
                        self._proj_state[1:3] = k_num - 1, -1
                    if "band" in attrs["comment"]:
                        s, k = self._proj_state[:2]
                        self._projectors[s][k].append([])
                        b_num = int(attrs["comment"].split()[1])
                        self._proj_state[2] = b_num - 1
            if name == "r":
                self._cbuf = ""
                self._is_r = True

        if self._is_eigenvalues:
            if name == "set":
                if "comment" in attrs.keys():
                    if "spin" in attrs["comment"]:
                        self._eigenvalues.append([])
                        spin_num = int(attrs["comment"].split()[1])
                        self._eig_state = [spin_num - 1, -1]
                    if "kpoint" in attrs["comment"]:
                        self._eigenvalues[self._eig_state[0]].append([])
                        k_num = int(attrs["comment"].split()[1])
                        self._eig_state[1] = k_num - 1
            if name == "r":
                self._cbuf = ""
                self._is_r = True

        if name == "projected":
            self._is_projected = True
            self._projectors = []

        if name == "eigenvalues":
            if self._is_projected:
                self._is_proj_eig = True
            else:
                self._is_eigenvalues = True
                self._eigenvalues = []

        if name == "separator":
            self._is_separator = True
            if attrs["name"] == "grids":
                self._is_grids = True

    def _end_element(self, name):
        if name == "scstep":
            self._is_scstep = False

        if name == "structure" and self._is_structure:
            self._is_structure = False

        if name == "varray":
            if self._is_forces:
                self._is_forces = False
                self._all_forces.append(self._forces)

            if self._is_stress:
                self._is_stress = False
                self._all_stress.append(self._stress)

            if self._is_k_weights:
                self._is_k_weights = False

            if self._is_kpointlist:
                self._is_kpointlist = False

            if self._is_positions:
                self._is_positions = False
                self._all_points.append(self._points)

            if self._is_basis:
                self._is_basis = False
                self._all_lattice.append(self._lattice)

            if self._is_epsilon:
                self._is_epsilon = False

        if name == "generation":
            if self._is_generation:
                self._is_generation = False

        if name == "kpoints":
            if self._is_kpoints:
                self._is_kpoints = False

        if name == "array":
            if self._is_symbols:
                self._is_symbols = False

            if self._is_born:
                self._is_born = False

        if name == "energy" and (not self._is_scstep):
            self._is_energy = False
            self._all_energies.append(self._energies)

        if name == "v":
            self._run_v()
            self._is_v = False
            if self._is_divisions:
                self._is_divisions = False

        if name == "i":
            self._run_i()
            self._is_i = False

        if name == "rc":
            self._is_rc = False
            if self._is_symbols:
                self._symbols.pop(-1)

        if name == "c":
            self._run_c()
            self._is_c = False

        if name == "r":
            self._run_r()
            self._is_r = False

        if name == "projected":
            self._is_projected = False

        if name == "eigenvalues":
            self._is_eigenvalues = False
            if self._is_projected:
                self._is_proj_eig = False

        if name == "set":
            self._is_set = False
            if self._is_born:
                self._born.append(self._born_atom)
                self._born_atom = None

        if name == "field":
            self._is_field_string = False
            self._is_field = False

        if self._field_val == "pseudopotential":
            if name == "set":
                self._is_set = False
                self._field_val = None
            if name == "rc" and self._is_set:
                self._is_rc = False
                self._pseudopotentials.append(self._ps_atom)
                self._ps_atom = None
            if name == "c":
                self._is_c = False

        if name == "separator":
            if self._is_grids:
                self._is_grids = False
            self._is_separator = False

    def _run_v(self):
        if self._is_v:
            if self._is_forces:
                self._forces.append([self._to_float(x) for x in self._cbuf.split()])
            if self._is_stress:
                self._stress.append([self._to_float(x) for x in self._cbuf.split()])
            if self._is_epsilon:
                self._epsilon.append([self._to_float(x) for x in self._cbuf.split()])
            if self._is_positions:
                self._points.append([self._to_float(x) for x in self._cbuf.split()])
            if self._is_basis:
                self._lattice.append([self._to_float(x) for x in self._cbuf.split()])
            if self._is_born:
                self._born_atom.append([self._to_float(x) for x in self._cbuf.split()])
            if self._is_kpoints:
                if self._is_k_weights:
                    self._k_weights.append(self._to_float(self._cbuf))
                if self._is_kpointlist:
                    self._kpointlist.append(
                        [self._to_float(x) for x in self._cbuf.split()]
                    )
                if self._is_generation:
                    if self._is_divisions:
                        self._k_mesh = [self._to_int(x) for x in self._cbuf.split()]
            self._cbuf = None

    def _run_i(self):
        if self._is_i:
            if self._is_energy:
                self._energies.append(self._to_float(self._cbuf.strip()))
            if self._is_efermi:
                self._efermi = self._to_float(self._cbuf.strip())
                self._is_efermi = False
            if self._is_NELECT:
                self._NELECT = self._to_float(self._cbuf.strip())
                self._is_NELECT = False
            if self._is_volume:
                self._all_volumes.append(self._to_float(self._cbuf.strip()))
                self._is_volume = False
            if self._is_grids:
                for i, b in enumerate(self._is_NGXYZ):
                    if b:
                        self._fft_grid[i] = self._to_int(self._cbuf.strip())
                        self._is_NGXYZ[i] = False
                for i, b in enumerate(self._is_NGXYZF):
                    if b:
                        self._fft_fine_grid[i] = self._to_int(self._cbuf.strip())
                        self._is_NGXYZF[i] = False
            self._cbuf = None

    def _run_c(self):
        if self._is_c:
            if self._is_symbols:
                self._symbols.append(str(self._cbuf.strip()))
            if self._field_val == "pseudopotential" and self._is_set and self._is_rc:
                if len(self._ps_atom) == 0:
                    self._ps_atom.append(self._to_int(self._cbuf.strip()))
                elif len(self._ps_atom) == 1:
                    self._ps_atom.append(self._cbuf.strip())
                elif len(self._ps_atom) == 2:
                    self._ps_atom.append(self._to_float(self._cbuf.strip()))
                elif len(self._ps_atom) == 3:
                    self._ps_atom.append(self._to_float(self._cbuf.strip()))
                elif len(self._ps_atom) == 4:
                    self._ps_atom.append(self._cbuf.strip())
            self._cbuf = None

    def _run_r(self):
        if self._is_r:
            if self._is_projected and not self._is_proj_eig:
                s, k, b = self._proj_state
                vals = [self._to_float(x) for x in self._cbuf.split()]
                self._projectors[s][k][b].append(vals)
            elif self._is_eigenvalues:
                s, k = self._eig_state
                vals = [self._to_float(x) for x in self._cbuf.split()]
                self._eigenvalues[s][k].append(vals)
            self._cbuf = None

    def _run_field_string(self):
        if self._is_field_string:
            self._field_val = self._cbuf.strip()
            self._cbuf = None

    def _char_data(self, data):
        if self._cbuf is not None:
            self._cbuf += data

    def _to_float(self, x):
        try:
            val = float(x)
            return val
        except ValueError:
            raise

    def _to_int(self, x):
        try:
            val = int(x)
            return val
        except ValueError:
            raise


def parse_vasprunxml(filename):
    """Parse vasprun.xml using VasprunxmlExpat."""
    if not os.path.exists(filename):
        print("File %s not found." % filename)
        sys.exit(1)

    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rb") as f:
        vxml = VasprunxmlExpat(f)
        try:
            vxml.parse()
        except (xml.parsers.expat.ExpatError, ValueError, Exception):
            print("Warning: Probably xml structure of %s is broken." % filename)
            print("Opening %s failed." % filename)
            sys.exit(1)
        return vxml


#
# XDATCAR
#
def read_XDATCAR(filename: str = "XDATCAR", fileptr=None):
    """Read XDATCAR.

    filename : str, optional
        Input filename in `XDATCAR` format. Default is `XDATCAR`. This is used
        unless fileptr is specified.
    fileptr : readable file pointer, optional
        File pointer used to read `XDATCAR`.

    Returns
    -------
    tuple of (lattice, positions)

    lattice : ndarray
        Basis vectors in row vectors.
        shape=(3, 3), dtype='double', order='C'
    positions : ndarry
        Atomic points in crystallographic coordinates.
        shape=(MD_steps, atoms, 3), dtype='double', order='C'

    """
    lattice = None
    numbers_of_atoms = None

    if fileptr is None:
        myio = get_io_module_to_decompress(filename)
        with myio.open(filename) as f:
            lattice, numbers_of_atoms = _read_XDATCAR_fileptr(f)
    else:
        lattice, numbers_of_atoms = _read_XDATCAR_fileptr(fileptr)

    if lattice is not None:
        if fileptr is None:
            _file = filename
        else:
            _file = fileptr
            _file.seek(0)
        data = np.loadtxt(_file, skiprows=7, comments="D", dtype="double")
        pos = data.reshape((-1, numbers_of_atoms.sum(), 3), order="C")
        lat = np.array(lattice, dtype="double", order="C")
        return lat, pos
    else:
        return None


def _read_XDATCAR_fileptr(f):
    f.readline()
    scale = float(f.readline())
    a = [float(x) for x in f.readline().split()[:3]]
    b = [float(x) for x in f.readline().split()[:3]]
    c = [float(x) for x in f.readline().split()[:3]]
    lattice = np.transpose([a, b, c]) * scale
    symbols = f.readline().split()
    numbers_of_atoms = np.array(
        [int(x) for x in f.readline().split()[: len(symbols)]], dtype="intc"
    )
    return lattice, numbers_of_atoms


def get_XDATCAR_lines_from_vasprunxml(
    vasprunxml_filename: str = "vasprun.xml",
    vasprunxml_expat: Optional[VasprunxmlExpat] = None,
    shift: Union[Sequence, np.ndarray] = None,
):
    """Return XDATCAR lines from vasprun.xml or VasprunxmlExpat instance.

    vasprunxml_filename : str, optional
        File name of vasprun.xml. Default is "vasprun.xml". This is used unless
        `vasprunxml_expat` is specified.
    vasprunxml_expat : VasprunxmlExpat, optional
        Instalce of `VasprunxmlExpat`. It is assumed that this instance is
        already parsed. Default is None.
    filename : str, optional
        Output filename in `XDATCAR` format. Default is `XDATCAR`. This is used
        unless fileptr is specified.
    fileptr : writable file pointer, optional
        File pointer used to write `XDATCAR`.
    shift : array_like, optional
        All atoms are uniformly translated in reduced coordinates. Default is
        None. shape=(3,), dtype='double'.

    """
    if vasprunxml_expat is not None:
        vxml = vasprunxml_expat
    else:
        with open(vasprunxml_filename, "rb") as f:
            vxml = VasprunxmlExpat(f)
            vxml.parse()
    lines, _ = _get_vasp_structure_header_lines(vxml.cell)
    points = vxml.points
    if shift is not None:
        points += shift
    for i, scaled_positions in enumerate(points):
        lines.append(f"Direct configuration=    {i}")
        lines += _get_scaled_positions_lines(scaled_positions)
    return lines


def write_XDATCAR(
    vasprunxml_filename: str = "vasprun.xml",
    vasprunxml_expat: Optional[VasprunxmlExpat] = None,
    filename: str = "XDATCAR",
    fileptr=Union[str, bytes, os.PathLike, io.IOBase],
    shift: Union[Sequence, np.ndarray] = None,
):
    """Write XDATCAR from vasprun.xml or VasprunxmlExpat instance.

    See get_XDATCAR_lines_from_vasprunxml.

    """
    lines = get_XDATCAR_lines_from_vasprunxml(
        vasprunxml_filename=vasprunxml_filename,
        vasprunxml_expat=vasprunxml_expat,
        shift=shift,
    )
    if fileptr is None:
        with open(filename, "w") as w:
            w.write("\n".join(lines))
    else:
        fileptr.write(("\n".join(lines)).encode("utf-8"))


#
# OUTCAR handling (obsolete)
#
def read_force_constants_OUTCAR(filename):
    """Read force constants from OUTCAR."""
    return get_force_constants_OUTCAR(filename)


def get_force_constants_OUTCAR(filename):
    """Read force constants from OUTCAR."""
    file = open(filename)
    while 1:
        line = file.readline()
        if line == "":
            print("Force constants could not be found.")
            return 0

        if line[:19] == " SECOND DERIVATIVES":
            break

    file.readline()
    num_atom = int(((file.readline().split())[-1].strip())[:-1])

    fc_tmp = []
    for _ in range(num_atom * 3):
        fc_tmp.append([float(x) for x in (file.readline().split())[1:]])

    fc_tmp = np.array(fc_tmp)

    force_constants = np.zeros((num_atom, num_atom, 3, 3), dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            force_constants[i, j] = -fc_tmp[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]

    return force_constants
