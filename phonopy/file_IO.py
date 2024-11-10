"""File I/O related routines."""

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
import pathlib
import sys
from typing import Optional, Union

import numpy as np
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from phonopy.cui.settings import fracval
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.dataset import get_displacements_and_forces
from phonopy.structure.symmetry import Symmetry, elaborate_borns_and_epsilon
from phonopy.utils import similarity_transformation


#
# FORCE_SETS
#
def write_FORCE_SETS(dataset, filename="FORCE_SETS"):
    """Write FORCE_SETS from dataset.

    See more detail in ``get_FORCE_SETS_lines``.

    """
    lines = get_FORCE_SETS_lines(dataset)
    with open(filename, "w") as w:
        w.write("\n".join(lines))
        w.write("\n")


def get_FORCE_SETS_lines(dataset, forces=None):
    """Generate FORCE_SETS string.

    See the format of dataset in the docstring of
    Phonopy.dataset. Optionally, sets of forces of supercells
    can be given. In this case, these forces are unnecessary to be stored
    in the dataset.

    """
    if "first_atoms" in dataset:
        return _get_FORCE_SETS_lines_type1(dataset, forces=forces)
    elif "displacements" in dataset:
        if forces is not None:
            dataset["forces"] = forces
        return _get_FORCE_SETS_lines_type2(dataset)


def _get_FORCE_SETS_lines_type1(dataset, forces=None):
    num_atom = dataset["natom"]
    displacements = dataset["first_atoms"]
    if forces is None:
        _forces = [x["forces"] for x in dataset["first_atoms"]]
    else:
        _forces = forces

    lines = []
    lines.append("%-5d" % num_atom)
    lines.append("%-5d" % len(displacements))
    for count, disp in enumerate(displacements):
        lines.append("")
        lines.append("%-5d" % (disp["number"] + 1))
        lines.append("%20.16f %20.16f %20.16f" % tuple(disp["displacement"]))
        for f in _forces[count]:
            lines.append("%15.10f %15.10f %15.10f" % tuple(f))

    return lines


def _get_FORCE_SETS_lines_type2(dataset):
    lines = []
    for displacements, forces in zip(dataset["displacements"], dataset["forces"]):
        for d, f in zip(displacements, forces):
            lines.append(("%15.8f" * 6) % (tuple(d) + tuple(f)))

    return lines


def parse_FORCE_SETS(natom=None, filename="FORCE_SETS", to_type2=False):
    """Parse FORCE_SETS from file.

    to_type2 : bool
        dataset of type2 is returned when True.

    Returns
    -------
    dataset : dict
        Displacement dataset. See Phonopy.dataset.

    """
    with open(filename, "r") as f:
        return _get_dataset(
            f,
            natom=natom,
            to_type2=to_type2,
        )


def parse_FORCE_SETS_from_strings(strings, natom=None, to_type2=False):
    """Parse FORCE_SETS from strings."""
    return _get_dataset(io.StringIO(strings), natom=natom, to_type2=to_type2)


def _get_dataset(f, natom=None, to_type2=False):
    first_line_ary = _get_line_ignore_blank(f).split()
    f.seek(0)
    if len(first_line_ary) == 1:
        if natom is None or int(first_line_ary[0]) == natom:
            dataset = _get_dataset_type1(f)
        else:
            msg = "Number of forces is not consistent with supercell setting."
            raise RuntimeError(msg)

        if to_type2:
            disps, forces = get_displacements_and_forces(dataset)
            return {"displacements": disps, "forces": forces}
        else:
            return dataset

    elif len(first_line_ary) == 6:
        return get_dataset_type2(f, natom)


def _get_dataset_type1(f):
    set_of_forces = []
    num_atom = int(_get_line_ignore_blank(f))
    num_displacements = int(_get_line_ignore_blank(f))

    for _ in range(num_displacements):
        line = _get_line_ignore_blank(f)
        atom_number = int(line)
        line = _get_line_ignore_blank(f).split()
        displacement = np.array([float(x) for x in line])
        forces_tmp = []
        for _ in range(num_atom):
            line = _get_line_ignore_blank(f).split()
            forces_tmp.append(np.array([float(x) for x in line]))
        forces_tmp = np.array(forces_tmp, dtype="double")
        forces = {
            "number": atom_number - 1,
            "displacement": displacement,
            "forces": forces_tmp,
        }
        set_of_forces.append(forces)

    dataset = {"natom": num_atom, "first_atoms": set_of_forces}

    return dataset


def get_dataset_type2(f, natom):
    """Parse type2 FORCE_SETS text and return dataset."""
    data = np.loadtxt(f, dtype="double")
    if data.shape[1] != 6 or (natom and data.shape[0] % natom != 0):
        msg = "Data shape of forces and displacements is incorrect."
        raise RuntimeError(msg)
    if natom:
        data = data.reshape(-1, natom, 6)
        displacements = data[:, :, :3]
        forces = data[:, :, 3:]
    else:
        displacements = data[:, :3]
        forces = data[:, 3:]
    dataset = {
        "displacements": np.array(displacements, dtype="double", order="C"),
        "forces": np.array(forces, dtype="double", order="C"),
    }
    return dataset


def _get_line_ignore_blank(f):
    line = f.readline().strip()
    if line == "":
        line = _get_line_ignore_blank(f)
    return line


def collect_forces(f, num_atom, hook, force_pos, word=None):
    """General function to collect forces from lines of a text file.

    Parameters
    ----------
    f :
        Text file pointer such as that returned by ``open(filename)``.
    num_atom : int
        Number of atoms in cell. Quit parsing when number of forces reaches this
        number.
    hook : str
        When this word is found at a line, parsing will start from the next line.
    force_pos : list
        Positions of force values in `line.split()`.
    word : str, optional
        Lines containing this word is only parsed. Default is None.

    Example
    -------
    The following is the abinit output.

    ...
    cartesian forces (hartree/bohr) at end:
    1     -0.00093686935947    -0.00000000000000    -0.00000000000000
    2      0.00015427277409    -0.00000000000000    -0.00000000000000
    3     -0.00000200377550    -0.00000000000000    -0.00000000000000
    4      0.00000619017547    -0.00000000000000    -0.00000000000000
    ...

    hook = "cartesian forces (eV/Angstrom)"
    force_pos = [1, 2, 3]

    """
    for line in f:
        if hook in line:
            break

    forces = []
    for line in f:
        if line.strip() == "":
            continue
        if word is not None:
            if word not in line:
                continue

        elems = line.split()
        if len(elems) > force_pos[2]:
            try:
                forces.append([float(elems[i]) for i in force_pos])
            except ValueError:
                forces = []
                break
        else:
            return False

        if len(forces) == num_atom:
            break

    return forces


def iter_collect_forces(filename, num_atom, hook, force_pos, word=None, max_iter=1000):
    """Repeat ``collect_forces`` to get the last set of forces in the file.

    Details of parameters are explained in ``collect_forces``.

    """
    with open(filename) as f:
        forces = []
        prev_forces = []

        for i in range(max_iter):  # noqa B007
            forces = collect_forces(f, num_atom, hook, force_pos, word=word)
            if not forces:
                forces = prev_forces[:]
                break
            else:
                prev_forces = forces[:]

        if i == max_iter - 1:
            sys.stderr.write("Reached to max number of iterations (%d).\n" % max_iter)

        return forces


#
# FORCE_CONSTANTS, force_constants.hdf5
#
def write_FORCE_CONSTANTS(force_constants, filename="FORCE_CONSTANTS", p2s_map=None):
    """Write force constants in text file format.

    Parameters
    ----------
    force_constants: ndarray
        Force constants
        shape=(n_satom,n_satom,3,3) or (n_patom,n_satom,3,3)
        dtype=double
    filename: str
        Filename to be saved
    p2s_map: ndarray
        Primitive atom indices in supercell index system
        dtype=intc

    """
    lines = get_FORCE_CONSTANTS_lines(force_constants, p2s_map=p2s_map)
    with open(filename, "w") as w:
        w.write("\n".join(lines))


def get_FORCE_CONSTANTS_lines(force_constants, p2s_map=None):
    """Return text in FORCE_CONSTANTS format.

    See also ``write_FORCE_CONSTANTS``.

    """
    if p2s_map is not None and len(p2s_map) == force_constants.shape[0]:
        indices = p2s_map
    else:
        indices = np.arange(force_constants.shape[0], dtype="intc")

    lines = []
    fc_shape = force_constants.shape
    lines.append("%4d %4d" % fc_shape[:2])
    for i, s_i in enumerate(indices):
        for j in range(fc_shape[1]):
            lines.append("%d %d" % (s_i + 1, j + 1))
            for vec in force_constants[i][j]:
                lines.append(("%22.15f" * 3) % tuple(vec))

    return lines


def write_force_constants_to_hdf5(
    force_constants: np.ndarray,
    filename: str = "force_constants.hdf5",
    p2s_map: Optional[np.ndarray] = None,
    physical_unit: Optional[str] = None,
    compression: Optional[Union[str, int]] = None,
):
    """Write force constants in hdf5 format.

    Parameters
    ----------
    force_constants: ndarray
        Force constants
        shape=(n_satom,n_satom,3,3) or (n_patom,n_satom,3,3)
        dtype=double
    filename: str
        Filename to be saved
    p2s_map: ndarray
        Primitive atom indices in supercell index system
        shape=(n_patom,)
        dtype=intc
    physical_unit : str, optional
        Physical unit used for force contants. Default is None.
    compression : str or int, optional
        h5py's lossless compression filters (e.g., "gzip", "lzf").
        See the detail at docstring of h5py.Group.create_dataset. Default is
        None.

    """
    try:
        import h5py
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    with h5py.File(filename, "w") as w:
        w.create_dataset(
            "force_constants", data=force_constants, compression=compression
        )
        if p2s_map is not None:
            w.create_dataset("p2s_map", data=p2s_map)
        if physical_unit is not None:
            w.create_dataset("physical_unit", data=[physical_unit])


def parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS", p2s_map=None):
    """Parse FORCE_CONSTANTS.

    Parameters
    ----------
    filename : str, optional
        Filename.
    p2s_map : ndarray, optional
        Primitive.p2s_map. Supplied, this is used to check file format consistency.

    """
    with open(filename) as fcfile:
        idx1 = []

        line = fcfile.readline()
        idx = [int(x) for x in line.split()]
        if len(idx) == 1:
            idx = [idx[0], idx[0]]
        force_constants = np.zeros((idx[0], idx[1], 3, 3), dtype="double")
        for i in range(idx[0]):
            for j in range(idx[1]):
                s_i = int(fcfile.readline().split()[0]) - 1
                if s_i not in idx1:
                    idx1.append(s_i)
                tensor = []
                for _ in range(3):
                    tensor.append([float(x) for x in fcfile.readline().split()])
                force_constants[i, j] = tensor

        check_force_constants_indices(idx, idx1, p2s_map, filename)

        return force_constants


def read_force_constants_hdf5(
    filename="force_constants.hdf5", p2s_map=None, return_physical_unit=False
):
    """Parse force_constants.hdf5.

    Parameters
    ----------
    filename : str, optional
        Filename.
    p2s_map : ndarray, optional
        Primitive.p2s_map. Supplied, this is used to check file format consistency.
    return_physical_unit : bool, optional
        When True and physical_unit is in file, physical unit is returned.
        Default is False.

    """
    try:
        import h5py
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    with h5py.File(filename, "r") as f:
        if "fc2" in f:
            key = "fc2"
        elif "force_constants" in f:
            key = "force_constants"
        else:
            raise RuntimeError("%s doesn't contain necessary information" % filename)

        fc = f[key][:]
        if "p2s_map" in f:
            p2s_map_in_file = f["p2s_map"][:]
            check_force_constants_indices(
                fc.shape[:2], p2s_map_in_file, p2s_map, filename
            )

        if return_physical_unit:
            if "physical_unit" in f:
                physical_unit = f["physical_unit"][0].decode("utf-8")
            else:
                physical_unit = None
            return fc, physical_unit
        else:
            return fc


def check_force_constants_indices(shape, indices, p2s_map, filename):
    """Check consistency of force constants data type."""
    if shape[0] != shape[1] and p2s_map is not None:
        if len(p2s_map) != len(indices) or (p2s_map != indices).any():
            lines = [
                f"{filename} file is inconsistent with the calculation setting. "
                "PRIMITIVE_AXIS may not be set correctly.",
                "p2s_map in primitive != p2s_map in file",
                f"{p2s_map} != {indices}",
            ]
            raise RuntimeError("\n".join(lines))


def parse_disp_yaml(filename="disp.yaml", return_cell=False):
    """Read disp.yaml or phonopy_disp.yaml.

    This method was originally made for parsing disp.yaml. Later this
    started to work for phonopy_disp.yaml, too. But now this method is not
    allowed to read phonopy_disp.yaml because of existance of PhonopyYaml
    class.

    """
    with open(filename) as f:
        new_dataset = {}
        dataset = yaml.load(f, Loader=Loader)
        if "phonopy" in dataset and "calculator" in dataset["phonopy"]:
            new_dataset["calculator"] = dataset["phonopy"]["calculator"]
        if "natom" in dataset:
            natom = dataset["natom"]
        elif "supercell" and "points" in dataset["supercell"]:
            natom = len(dataset["supercell"]["points"])
        else:
            raise RuntimeError("%s doesn't contain necessary information.")
        new_dataset["natom"] = natom
        new_first_atoms = []

        try:
            displacements = dataset["displacements"]
        except KeyError:
            raise

        if isinstance(displacements[0], dict):
            for first_atoms in displacements:
                first_atoms["atom"] -= 1
                atom1 = first_atoms["atom"]
                disp1 = first_atoms["displacement"]
                new_first_atoms.append({"number": atom1, "displacement": disp1})
            new_dataset["first_atoms"] = new_first_atoms

        if return_cell:
            cell = get_cell_from_disp_yaml(dataset)
            return new_dataset, cell
        else:
            return new_dataset


def write_disp_yaml_from_dataset(dataset, supercell, filename="disp.yaml"):
    """Write disp.yaml from dataset.

    This function is obsolete, because disp.yaml is obsolete.

    """
    displacements = [
        (d["number"],) + tuple(d["displacement"]) for d in dataset["first_atoms"]
    ]
    write_disp_yaml(displacements, supercell, filename=filename)


def write_disp_yaml(displacements: dict, supercell: PhonopyAtoms, filename="disp.yaml"):
    """Write disp.yaml from displacements.

    This function is obsolete, because disp.yaml is obsolete.

    """
    lines = []
    lines.append("natom: %4d" % len(supercell))
    lines += _get_disp_yaml_lines(displacements, supercell)
    lines.append(str(supercell))

    with open(filename, "w") as w:
        w.write("\n".join(lines))


def _get_disp_yaml_lines(displacements, supercell):
    lines = []
    lines.append("displacements:")
    for _, disp in enumerate(displacements):
        lines.append("- atom: %4d" % (disp[0] + 1))
        lines.append("  displacement:")
        lines.append("    [ %20.16f,%20.16f,%20.16f ]" % tuple(disp[1:4]))
    return lines


#
# DISP (old phonopy displacement format)
#
def parse_DISP(filename="DISP"):
    """Parse DISP file.

    This function is obsolete, because DISP is obsolete.

    """
    with open(filename) as disp:
        displacements = []
        for line in disp:
            if line.strip() != "":
                a = line.split()
                displacements.append(
                    [int(a[0]) - 1, float(a[1]), float(a[2]), float(a[3])]
                )
        return displacements


#
# Parse supercell in disp.yaml
#
def get_cell_from_disp_yaml(dataset):
    """Read cell from disp.yaml like file."""
    if "lattice" in dataset:
        lattice = dataset["lattice"]
        if "points" in dataset:
            data_key = "points"
            pos_key = "coordinates"
        elif "atoms" in dataset:
            data_key = "atoms"
            pos_key = "position"
        else:
            data_key = None
            pos_key = None

        try:
            positions = [x[pos_key] for x in dataset[data_key]]
        except KeyError as exc:
            msg = (
                '"disp.yaml" format is too old. '
                'Please re-create it as "phonopy_disp.yaml" to contain '
                "supercell crystal structure information."
            )
            raise RuntimeError(msg) from exc
        symbols = [x["symbol"] for x in dataset[data_key]]
        cell = PhonopyAtoms(cell=lattice, scaled_positions=positions, symbols=symbols)
        return cell
    else:
        return get_cell_from_disp_yaml(dataset["supercell"])


#
# QPOINTS
#
def parse_QPOINTS(filename="QPOINTS"):
    """Read QPOINTS file."""
    with open(filename, "r") as f:
        num_qpoints = int(f.readline().strip())
        qpoints = []
        for _ in range(num_qpoints):
            qpoints.append([fracval(x) for x in f.readline().strip().split()])
        return np.array(qpoints)


#
# BORN
#
def write_BORN(primitive, borns, epsilon, filename="BORN"):
    """Write BORN from NAC paramters."""
    lines = get_BORN_lines(primitive, borns, epsilon)
    with open(filename, "w") as w:
        w.write("\n".join(lines))


def get_BORN_lines(
    unitcell,
    borns,
    epsilon,
    factor=None,
    primitive_matrix=None,
    supercell_matrix=None,
    symprec=1e-5,
):
    """Generate text of BORN file."""
    borns, epsilon, atom_indices = elaborate_borns_and_epsilon(
        unitcell,
        borns,
        epsilon,
        symmetrize_tensors=True,
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
    )

    text = "# epsilon and Z* of atoms "
    text += " ".join(["%d" % n for n in atom_indices + 1])
    lines = [
        text,
    ]
    lines.append(("%13.8f " * 9) % tuple(epsilon.flatten()))
    for z in borns:
        lines.append(("%13.8f " * 9) % tuple(z.flatten()))
    return lines


def parse_BORN(primitive, symprec=1e-5, is_symmetry=True, filename="BORN"):
    """Parse BORN file.

    Parameters
    ----------
    primitive : Primitive
        Primitive cell.
    symprec : float, optional
        Symmetry tolerance. Default is 1e-5.
    is_symmetry : bool, optional
        When True, parse values are symmetrized. Default is True.
    filename : str, optional
        Filename.

    """
    with open(filename, "r") as f:
        return _parse_BORN_from_file_object(f, primitive, symprec, is_symmetry)


def parse_BORN_from_strings(
    strings, primitive, symprec=1e-5, is_symmetry=True
) -> Optional[dict]:
    """Parse BORN file text.

    See `parse_BORN` for parameters.

    """
    f = io.StringIO(strings)
    return _parse_BORN_from_file_object(f, primitive, symprec, is_symmetry)


def _parse_BORN_from_file_object(f, primitive, symprec, is_symmetry) -> Optional[dict]:
    symmetry = Symmetry(primitive, symprec=symprec, is_symmetry=is_symmetry)
    return get_born_parameters(f, primitive, symmetry)


def get_born_parameters(
    f: io.IOBase, primitive: PhonopyAtoms, prim_symmetry: Symmetry
) -> Optional[dict]:
    """Parse BORN file text.

    Parameters
    ----------
    f :
        File pointer of BORN file.
    primitive : Primitive
        Primitive cell.
    prim_symmetry : Symmetry
        Symmetry of primitive cell.

    """
    line_arr = f.readline().split()
    if len(line_arr) < 1:
        print("BORN file format of line 1 is incorrect")
        return None

    factor = None
    G_cutoff = None
    Lambda = None

    if len(line_arr) > 0:
        try:
            factor = float(line_arr[0])
        except (ValueError, TypeError):
            factor = None
    if len(line_arr) > 1:
        try:
            G_cutoff = float(line_arr[1])
        except (ValueError, TypeError):
            G_cutoff = None
    if len(line_arr) > 2:
        try:
            Lambda = float(line_arr[2])
        except (ValueError, TypeError):
            Lambda = None

    # Read dielectric constant
    line = f.readline().split()
    if not len(line) == 9:
        print("BORN file format of line 2 is incorrect")
        return None
    dielectric = np.reshape([float(x) for x in line], (3, 3))

    # Read Born effective charge
    independent_atoms = prim_symmetry.get_independent_atoms()
    borns = np.zeros((len(primitive), 3, 3), dtype="double", order="C")

    for i in independent_atoms:
        line = f.readline().split()
        if len(line) == 0:
            print("Number of lines for Born effect charge is not enough.")
            return None
        if not len(line) == 9:
            print("BORN file format of line %d is incorrect" % (i + 3))
            return None
        borns[i] = np.reshape([float(x) for x in line], (3, 3))

    # Check that the number of atoms in the BORN file was correct
    line = f.readline().split()
    if len(line) > 0:
        print(
            "Too many atoms in the BORN file (it should only contain "
            "symmetry-independent atoms)"
        )
        return None

    _expand_borns(borns, primitive, prim_symmetry)
    non_anal = {"born": borns, "dielectric": dielectric}
    if factor is not None:
        non_anal["factor"] = factor
    if G_cutoff is not None:
        non_anal["G_cutoff"] = G_cutoff
    if Lambda is not None:
        non_anal["Lambda"] = Lambda

    return non_anal


def _expand_borns(borns, primitive: PhonopyAtoms, prim_symmetry: Symmetry):
    # Expand Born effective charges to all atoms in the primitive cell
    rotations = prim_symmetry.symmetry_operations["rotations"]
    map_operations = prim_symmetry.get_map_operations()
    map_atoms = prim_symmetry.get_map_atoms()

    for i in range(len(primitive)):
        # R_cart = L R L^-1
        rot_cartesian = similarity_transformation(
            primitive.cell.T, rotations[map_operations[i]]
        )
        # R_cart^T B R_cart^-T (inverse rotation is required to transform)
        borns[i] = similarity_transformation(rot_cartesian.T, borns[map_atoms[i]])


#
# phonopy.yaml
#
def is_file_phonopy_yaml(filename, keyword="phonopy"):
    """Check whether the file is phonopy.yaml like file or not.

    Parameters
    ----------
    filename : str
        Filename.
    keyword : str
        When this keyword is found in dict keys returned by yaml loader,
        this function return True.

    Example
    -------
    The initial part of phonopy_disp.yaml is like below.

        phonopy:
          version: 2.7.0
          frequency_unit_conversion_factor: 15.633302
          symmetry_tolerance: 1.00000e-05
          configuration:
              cell_filename: "POSCAR-unitcell"
              create_displacements: ".true."
              primitive_axes: "auto"
              dim: "2 2 2"
        ...

    """
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "r") as f:
        try:
            data = yaml.load(f, Loader=Loader)
            if data is None:
                return False
            if keyword in data:
                return True
            else:
                return False
        except yaml.YAMLError:
            return False


#
# e-v.dat, thermal_properties.yaml
#
def read_thermal_properties_yaml(filenames):
    """Read thermal_properties.yaml."""
    thermal_properties = []
    num_modes = []
    num_integrated_modes = []
    for filename in filenames:
        with open(filename) as f:
            tp_yaml = yaml.load(f, Loader=Loader)
            thermal_properties.append(tp_yaml["thermal_properties"])
            if "num_modes" in tp_yaml and "num_integrated_modes" in tp_yaml:
                num_modes.append(tp_yaml["num_modes"])
                num_integrated_modes.append(tp_yaml["num_integrated_modes"])

    temperatures = [v["temperature"] for v in thermal_properties[0]]
    temp = []
    cv = []
    entropy = []
    fe_phonon = []
    for _, tp in enumerate(thermal_properties):
        temp.append([v["temperature"] for v in tp])
        if not np.allclose(temperatures, temp):
            msg = [
                "",
            ]
            msg.append("Check your input files")
            msg.append("Disagreement of temperature range or step")
            for t, fname in zip(temp, filenames):
                msg.append(
                    "%s: Range [ %d, %d ], Step %f"
                    % (fname, int(t[0]), int(t[-1]), t[1] - t[0])
                )
            msg.append("")
            msg.append("Stop phonopy-qha")
            raise RuntimeError(msg)
        cv.append([v["heat_capacity"] for v in tp])
        entropy.append([v["entropy"] for v in tp])
        fe_phonon.append([v["free_energy"] for v in tp])

    # shape=(temperatures, volumes)
    cv = np.array(cv).T
    entropy = np.array(entropy).T
    fe_phonon = np.array(fe_phonon).T

    return (temperatures, cv, entropy, fe_phonon, num_modes, num_integrated_modes)


def read_v_e(filename):
    """Read v-e.dat file."""
    data = _parse_QHA_data(filename)
    if data.shape[1] != 2:
        msg = "File format of %s is incorrect for reading e-v data." % filename
        raise RuntimeError(msg)
    volumes, electronic_energies = data.T
    return volumes, electronic_energies


def read_efe(filename):
    """Read fe-v.dat (efe) file."""
    data = _parse_QHA_data(filename)
    temperatures = data[:, 0]
    free_energies = data[:, 1:]
    return temperatures, free_energies


def _parse_QHA_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            if line.strip() == "" or line.strip()[0] == "#":
                continue
            if "#" in line:
                data.append([float(x) for x in line.split("#")[0].split()])
            else:
                data.append([float(x) for x in line.split()])
        return np.array(data)


def get_io_module_to_decompress(filename):
    """Return io-module to decompress file.

    Filename extensions of lzma, xz, gzip, bz2 are supported.

    It is supported to use it like `returned_module.open(filename)`.

    """
    ext = pathlib.Path(filename).suffix
    if ext == ".xz" or ext == ".lzma":
        import lzma

        return lzma
    elif ext == ".gz":
        import gzip

        return gzip
    elif ext == ".bz2":
        import bz2

        return bz2
    else:
        import io

        return io


def get_supported_file_extensions_for_compression():
    """Return file extensions for compression.

    This function must be coupled with `get_io_module_to_decompress`.

    """
    return "", ".xz", ".lzma", ".gz", ".bz2"
