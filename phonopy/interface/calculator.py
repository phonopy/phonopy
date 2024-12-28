"""Routines to handle various calculator interfaces."""

# Copyright (C) 2014 Atsushi Togo
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

import os
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import yaml

from phonopy.file_IO import get_supported_file_extensions_for_compression
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.vasp import sort_positions_by_symbols
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import determinant
from phonopy.structure.dataset import get_displacements_and_forces
from phonopy.units import (
    AbinitToTHz,
    Bohr,
    CastepToTHz,
    CP2KToTHz,
    CrystalToTHz,
    DftbpToTHz,
    ElkToTHz,
    FleurToTHz,
    Hartree,
    PwscfToTHz,
    Rydberg,
    SiestaToTHz,
    TurbomoleToTHz,
    VaspToTHz,
    Wien2kToTHz,
)

calculator_info = {
    "abacus": {"option": {"name": "--abacus", "help": "Invoke ABACUS mode"}},
    "abinit": {"option": {"name": "--abinit", "help": "Invoke Abinit mode"}},
    "aims": {"option": {"name": "--aims", "help": "Invoke FHI-aims mode"}},
    "castep": {"option": {"name": "--castep", "help": "Invoke CASTEP mode"}},
    "cp2k": {"option": {"name": "--cp2k", "help": "Invoke CP2K mode"}},
    "crystal": {"option": {"name": "--crystal", "help": "Invoke CRYSTAL mode"}},
    "dftbp": {"option": {"name": "--dftb+", "help": "Invoke dftb+ mode"}},
    "elk": {"option": {"name": "--elk", "help": "Invoke elk mode"}},
    "fleur": {"option": {"name": "--fleur", "help": "Invoke Fleur mode"}},
    "lammps": {"option": {"name": "--lammps", "help": "Invoke Lammps mode"}},
    "qe": {"option": {"name": "--qe", "help": "Invoke Quantum espresso (QE) mode"}},
    "siesta": {"option": {"name": "--siesta", "help": "Invoke Siesta mode"}},
    "turbomole": {"option": {"name": "--turbomole", "help": "Invoke TURBOMOLE mode"}},
    "vasp": {"option": {"name": "--vasp", "help": "Invoke Vasp mode"}},
    "wien2k": {"option": {"name": "--wien2k", "help": "Invoke Wien2k mode"}},
    "pwmat": {"option": {"name": "--pwmat", "help": "Invoke PWmat mode"}},
}


def add_arguments_of_calculators(parser: ArgumentParser, calculator_info):
    """Add options of calculators to ArgumentParser class instance."""
    for calculator in calculator_info:
        option = calculator_info[calculator]["option"]
        parser.add_argument(
            option["name"],
            dest="%s_mode" % calculator,
            action="store_true",
            default=False,
            help=option["help"],
        )


def get_interface_mode(args_dict):
    """Return calculator name.

    The calculator name is obtained from command option arguments where
    argparse is used. The argument attribute name has to be
    "{calculator}_mode". Then this method returns "{calculator}".

    """
    for calculator in calculator_info:
        mode = "%s_mode" % calculator
        if mode in args_dict and args_dict[mode]:
            return calculator
    return None


def convert_crystal_structure(filename_in, interface_in, filename_out, interface_out):
    """Convert crystal structures between different calculator interfaces."""
    cell, _ = read_crystal_structure(filename=filename_in, interface_mode=interface_in)
    units_in = get_default_physical_units(interface_in)
    units_out = get_default_physical_units(interface_out)
    factor = units_in["distance_to_A"] / units_out["distance_to_A"]
    cell.cell = cell.cell * factor
    write_crystal_structure(filename_out, cell, interface_mode=interface_out)


def write_crystal_structure(
    filename: Optional[str],
    cell: PhonopyAtoms,
    interface_mode: Optional[str] = None,
    optional_structure_info: Optional[tuple] = None,
):
    """Write crystal structure to file in each calculator format.

    filename : str, optional
        File name to be used to write out the crystal structure.
    cell : PhonopyAtoms
        Crystal structure
    interface_mode : str, optional
        Calculator interface such as 'vasp', 'qe', ... Default is None,
        that is equivalent to 'vasp'.
    optional_structure_info : tuple, optional
        Information returned by the method ``read_crystal_structure``.
        See the docstring. Default is None.

    """
    if interface_mode is None or interface_mode == "vasp":
        import phonopy.interface.vasp as vasp

        vasp.write_vasp(filename, cell)
    elif interface_mode == "abinit":
        import phonopy.interface.abinit as abinit

        abinit.write_abinit(filename, cell)
    elif interface_mode == "qe":
        import phonopy.interface.qe as qe

        pp_filenames = optional_structure_info[1]
        qe.write_pwscf(filename, cell, pp_filenames)

    elif interface_mode == "wien2k":
        import phonopy.interface.wien2k as wien2k

        _, npts, r0s, rmts = optional_structure_info
        wien2k.write_wein2k(filename, cell, npts, r0s, rmts)
    elif interface_mode == "elk":
        import phonopy.interface.elk as elk

        sp_filenames = optional_structure_info[1]
        elk.write_elk(filename, cell, sp_filenames)
    elif interface_mode == "siesta":
        import phonopy.interface.siesta as siesta

        atypes = optional_structure_info[1]
        siesta.write_siesta(filename, cell, atypes)
    elif interface_mode == "cp2k":
        import phonopy.interface.cp2k as cp2k

        _, tree = optional_structure_info
        cp2k.write_cp2k_by_filename(filename, cell, tree)
    elif interface_mode == "crystal":
        import phonopy.interface.crystal as crystal

        conv_numbers = optional_structure_info[1]
        crystal.write_crystal(filename, cell, conv_numbers)
    elif interface_mode == "dftbp":
        import phonopy.interface.dftbp as dftbp

        dftbp.write_dftbp(filename, cell)
    elif interface_mode == "turbomole":
        import phonopy.interface.turbomole as turbomole

        turbomole.write_turbomole(filename, cell)
    elif interface_mode == "aims":
        import phonopy.interface.aims as aims

        aims.write_aims(filename, cell)
    elif interface_mode == "castep":
        import phonopy.interface.castep as castep

        castep.write_castep(filename, cell)
    elif interface_mode == "fleur":
        import phonopy.interface.fleur as fleur

        speci, restlines = optional_structure_info
        fleur.write_fleur(filename, cell, speci, 1, restlines)
    elif interface_mode == "abacus":
        import phonopy.interface.abacus as abacus

        pps = optional_structure_info[1]
        orbitals = optional_structure_info[2]
        abfs = optional_structure_info[3]
        abacus.write_abacus(filename, cell, pps, orbitals, abfs)
    elif interface_mode == "lammps":
        import phonopy.interface.lammps as lammps

        lammps.write_lammps(filename, cell)
    elif interface_mode == "pwmat":
        import phonopy.interface.pwmat as pwmat

        pwmat.write_atom_config(filename, cell)
    else:
        raise RuntimeError("No calculator interface was found.")


def write_supercells_with_displacements(
    interface_mode: str,
    supercell: PhonopyAtoms,
    cells_with_disps: Sequence[PhonopyAtoms],
    optional_structure_info: Optional[tuple] = None,
    displacement_ids: Optional[Union[Sequence, np.ndarray]] = None,
    zfill_width: int = 3,
    additional_info: Optional[dict] = None,
):
    """Write supercell with displacements to files in each calculator format.

    interface_mode : str
        Calculator interface such as 'vasp', 'qe', ...
    supercell : PhonopyAtoms
        Supercell.
    cells_with_disps : list of PhonopyAtoms
        Supercells with displacements.
    optional_structure_info : tuple
        Information returned by the method ``read_crystal_structure``.
        See the docstring.
    displacements_ids : array_like or None, optional
        Integer 1d array with the length of cells_with_disps, containing
        numbers to be assigned to the supercells with displacements.
        Default is None, which gives [1, 2, 3, ...].
    zfill_width : int, optional
        Supercell numbers are filled by zeros from the left with the digits
        as given, which results in 001, 002, ..., when zfill_width=3.
    additional_info : dict or None, optional
        Any information expected to be given to writers of calculators.
        Default is None.

    """
    if displacement_ids is None:
        ids = np.arange(len(cells_with_disps), dtype=int) + 1
    else:
        ids = displacement_ids

    args = (supercell, cells_with_disps, ids)
    kwargs = {"width": zfill_width}
    if additional_info is not None and "pre_filename" in additional_info:
        kwargs["pre_filename"] = additional_info["pre_filename"]

    if interface_mode is None or interface_mode == "vasp":
        import phonopy.interface.vasp as vasp

        vasp.write_supercells_with_displacements(*args, **kwargs)
        write_magnetic_moments(supercell, sort_by_elements=True)
    elif interface_mode == "abinit":
        import phonopy.interface.abinit as abinit

        abinit.write_supercells_with_displacements(*args, **kwargs)
    elif interface_mode == "qe":
        import phonopy.interface.qe as qe

        pp_filenames = optional_structure_info[1]
        qe_args = args + (pp_filenames,)
        qe.write_supercells_with_displacements(*qe_args, **kwargs)
        write_magnetic_moments(supercell, sort_by_elements=False)
    elif interface_mode == "pwmat":
        import phonopy.interface.pwmat as pwmat

        pwmat.write_supercells_with_displacements(*args, **kwargs)

    elif interface_mode == "wien2k":
        import phonopy.interface.wien2k as wien2k

        unitcell_filename, npts, r0s, rmts = optional_structure_info
        N = abs(determinant(additional_info["supercell_matrix"]))
        w2k_args = args + (npts, r0s, rmts, N)
        if "pre_filename" not in kwargs:
            kwargs["pre_filename"] = unitcell_filename
        wien2k.write_supercells_with_displacements(*w2k_args, **kwargs)
    elif interface_mode == "elk":
        import phonopy.interface.elk as elk

        sp_filenames = optional_structure_info[1]
        elk_args = args + (sp_filenames,)
        elk.write_supercells_with_displacements(*elk_args, **kwargs)
    elif interface_mode == "siesta":
        import phonopy.interface.siesta as siesta

        atypes = optional_structure_info[1]
        sst_args = args + (atypes,)
        siesta.write_supercells_with_displacements(*sst_args, **kwargs)
    elif interface_mode == "cp2k":
        import phonopy.interface.cp2k as cp2k

        cp2k_args = args + (optional_structure_info,)
        cp2k.write_supercells_with_displacements(*cp2k_args, **kwargs)
    elif interface_mode == "crystal":
        import phonopy.interface.crystal as crystal

        if additional_info is None:
            kwargs["template_file"] = "TEMPLATE"
        else:
            kwargs["template_file"] = additional_info.get("template_file", "TEMPLATE")
        conv_numbers = optional_structure_info[1]
        N = abs(determinant(additional_info["supercell_matrix"]))
        cst_args = args + (conv_numbers, N)
        crystal.write_supercells_with_displacements(*cst_args, **kwargs)
    elif interface_mode == "dftbp":
        import phonopy.interface.dftbp as dftbp

        dftbp.write_supercells_with_displacements(*args, **kwargs)
    elif interface_mode == "turbomole":
        import phonopy.interface.turbomole as turbomole

        turbomole.write_supercells_with_displacements(*args, **kwargs)
    elif interface_mode == "aims":
        import phonopy.interface.aims as aims

        aims.write_supercells_with_displacements(*args, **kwargs)
    elif interface_mode == "castep":
        import phonopy.interface.castep as castep

        castep.write_supercells_with_displacements(*args, **kwargs)
    elif interface_mode == "fleur":
        import phonopy.interface.fleur as fleur

        speci = optional_structure_info[1]
        restlines = optional_structure_info[2]
        N = abs(determinant(additional_info["supercell_matrix"]))
        fleur_args = args + (speci, N, restlines)
        fleur.write_supercells_with_displacements(*fleur_args, **kwargs)
    elif interface_mode == "abacus":
        import phonopy.interface.abacus as abacus

        pps = optional_structure_info[1]
        orbitals = optional_structure_info[2]
        abfs = optional_structure_info[3]
        abacus_args = args + (pps, orbitals, abfs)
        abacus.write_supercells_with_displacements(*abacus_args, **kwargs)
    elif interface_mode == "lammps":
        import phonopy.interface.lammps as lammps

        lammps.write_supercells_with_displacements(*args, **kwargs)
    else:
        raise RuntimeError("No calculator interface was found.")


def write_magnetic_moments(cell: PhonopyAtoms, sort_by_elements=False):
    """Write MAGMOM."""
    magmoms = cell.magnetic_moments
    if magmoms is not None:
        if sort_by_elements:
            (_, _, _, sort_list) = sort_positions_by_symbols(
                cell.symbols, cell.scaled_positions
            )
        else:
            sort_list = np.arange(len(cell))

        text = " MAGMOM = "
        text += " ".join([f"{v}" for v in magmoms[sort_list].ravel()])
        with open("MAGMOM", "w") as w:
            w.write(text)
            w.write("\n")


def read_crystal_structure(
    filename=None,
    interface_mode=None,
    chemical_symbols=None,
    phonopy_yaml_cls: type[PhonopyYaml] = PhonopyYaml,
):
    """Return crystal structure from file in each calculator format.

    Parameters
    ----------
    filename : str, optional
        Filename that contains cell structure information. Default is None.
        The predetermined filename for each interface_mode is used.
    interface_mode : str, optional
        This is used to recognize the file format. Default is None, which
        is equivalent to 'vasp' mode.
    chemical_symbols : list of str, optional
        This is only used for 'vasp' mode. VASP POSCAR file format can be
        written without chemical symbol information. With this option,
        chemical symbols can be given.
    phonopy_yaml_cls : PhonopyYaml, optional
        This brings PhonopyYaml-like class dependent parameters. Here,
        currently only the default filenames are provided by this.

    Returns
    -------
    tuple
        (Unit cell in PhonopyAtoms, optional_structure_info in tuple)

        The optional_structure_info is given by a tuple. The first element of
        it is the unit cell file name for which the unit cell data are read,
        and the rest is dependent on calculator interface.

    """
    if interface_mode == "phonopy_yaml":
        return _read_phonopy_yaml(filename, phonopy_yaml_cls)

    if filename is None:
        cell_filename = get_default_cell_filename(interface_mode)
        if not os.path.isfile(cell_filename):
            return None, (cell_filename, "(default file name)")
    else:
        cell_filename = filename
        if not os.path.isfile(cell_filename):
            return None, (cell_filename,)

    if interface_mode is None or interface_mode == "vasp":
        from phonopy.interface.vasp import read_vasp

        if chemical_symbols is None:
            unitcell = read_vasp(cell_filename)
        else:
            unitcell = read_vasp(cell_filename, symbols=chemical_symbols)
        return unitcell, (cell_filename,)
    elif interface_mode == "abinit":
        from phonopy.interface.abinit import read_abinit

        unitcell = read_abinit(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == "qe":
        from phonopy.interface.qe import read_pwscf

        unitcell, pp_filenames = read_pwscf(cell_filename)
        return unitcell, (cell_filename, pp_filenames)
    elif interface_mode == "pwmat":
        from phonopy.interface.pwmat import read_atom_config

        unitcell = read_atom_config(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == "wien2k":
        from phonopy.interface.wien2k import parse_wien2k_struct

        unitcell, npts, r0s, rmts = parse_wien2k_struct(cell_filename)
        return unitcell, (cell_filename, npts, r0s, rmts)
    elif interface_mode == "elk":
        from phonopy.interface.elk import read_elk

        unitcell, sp_filenames = read_elk(cell_filename)
        return unitcell, (cell_filename, sp_filenames)
    elif interface_mode == "siesta":
        from phonopy.interface.siesta import read_siesta

        unitcell, atypes = read_siesta(cell_filename)
        return unitcell, (cell_filename, atypes)
    elif interface_mode == "cp2k":
        from phonopy.interface.cp2k import read_cp2k

        unitcell, config_tree = read_cp2k(cell_filename)
        return unitcell, (cell_filename, config_tree)
    elif interface_mode == "crystal":
        from phonopy.interface.crystal import read_crystal

        unitcell, conv_numbers = read_crystal(cell_filename)
        return unitcell, (cell_filename, conv_numbers)
    elif interface_mode == "dftbp":
        from phonopy.interface.dftbp import read_dftbp

        unitcell = read_dftbp(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == "turbomole":
        from phonopy.interface.turbomole import read_turbomole

        unitcell = read_turbomole(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == "aims":
        from phonopy.interface.aims import read_aims

        unitcell = read_aims(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == "castep":
        from phonopy.interface.castep import read_castep

        unitcell = read_castep(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == "fleur":
        from phonopy.interface.fleur import read_fleur

        unitcell, speci, restlines = read_fleur(cell_filename)
        return unitcell, (cell_filename, speci, restlines)
    elif interface_mode == "abacus":
        from phonopy.interface.abacus import read_abacus

        unitcell, pps, orbitals, abfs = read_abacus(cell_filename)
        return unitcell, (cell_filename, pps, orbitals, abfs)
    elif interface_mode == "lammps":
        from phonopy.interface.lammps import read_lammps

        unitcell = read_lammps(cell_filename)
        return unitcell, (cell_filename,)
    else:
        raise RuntimeError("No calculator interface was found.")


def get_default_cell_filename(interface_mode):
    """Return default filename of unit cell structure of each calculator."""
    if interface_mode is None or interface_mode == "vasp":
        return "POSCAR"
    elif interface_mode in ("abinit", "qe"):
        return "unitcell.in"
    elif interface_mode == "wien2k":
        return "case.struct"
    elif interface_mode == "elk":
        return "elk.in"
    elif interface_mode == "siesta":
        return "input.fdf"
    elif interface_mode == "cp2k":
        return "unitcell.inp"
    elif interface_mode == "crystal":
        return "crystal.o"
    elif interface_mode == "dftbp":
        return "geo.gen"
    elif interface_mode == "turbomole":
        return "control"
    elif interface_mode == "aims":
        return "geometry.in"
    elif interface_mode == "castep":
        return "unitcell.cell"
    elif interface_mode == "fleur":
        return "fleur.in"
    elif interface_mode == "abacus":
        return "STRU"
    elif interface_mode == "lammps":
        return "unitcell"
    elif interface_mode == "pwmat":
        return "atom.config"
    else:
        return None


def get_default_supercell_filename(interface_mode):
    """Return default filename of supercell structure of each calculator."""
    if interface_mode == "phonopy_yaml":
        return "phonopy_disp.yaml"
    elif interface_mode is None or interface_mode == "vasp":
        return "SPOSCAR"
    elif interface_mode in ("abinit", "elk", "qe", "fleur"):
        return "supercell.in"
    elif interface_mode == "wien2k":
        return "case.structS"
    elif interface_mode == "siesta":
        return "supercell.fdf"
    elif interface_mode == "cp2k":
        # CP2K interface generates filenames based on original project name
        return None
    elif interface_mode == "crystal":
        return None  # supercell.ext can not be parsed by crystal interface.
    elif interface_mode == "dftbp":
        return "geo.genS"
    elif interface_mode == "turbomole":
        return None  # TURBOMOLE interface generates directories with inputs
    elif interface_mode == "aims":
        return "geometry.in.supercell"
    elif interface_mode == "castep":
        return "supercell.cell"
    elif interface_mode == "abacus":
        return "sSTRU"
    elif interface_mode == "lammps":
        return "supercell"
    elif interface_mode == "pwmat":
        return "supercell.config"
    else:
        return None


def get_default_displacement_distance(interface_mode):
    """Return default displacement distance of each calculator."""
    if interface_mode in (
        "wien2k",
        "abinit",
        "elk",
        "qe",
        "siesta",
        "turbomole",
        "fleur",
        "abacus",
    ):
        displacement_distance = 0.02
    else:  # default or vasp, crystal, cp2k, pwmat
        displacement_distance = 0.01
    return displacement_distance


def get_default_physical_units(interface_mode=None):
    """Return physical units of eachi calculator.

    Physical units: energy,  distance,  atomic mass, force,        force constants
    vasp          : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    wien2k        : Ry,      au(=borh), AMU,         mRy/au,       mRy/au^2
    abinit        : hartree, au,        AMU,         eV/angstrom,  eV/angstrom.au
    elk           : hartree, au,        AMU,         hartree/au,   hartree/au^2
    qe            : Ry,      au,        AMU,         Ry/au,        Ry/au^2
    siesta        : eV,      au,        AMU,         eV/angstrom,  eV/angstrom.au
    CRYSTAL       : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    DFTB+         : hartree, au,        AMU          hartree/au,   hartree/au^2
    TURBOMOLE     : hartree, au,        AMU,         hartree/au,   hartree/au^2
    CP2K          : hartree, angstrom,  AMU,         hartree/au,   hartree/angstrom.au
    FHI-aims      : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    castep        : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    fleur         : hartree, au,        AMU,         hartree/au,   hartree/au^2
    abacus        : eV,      au,        AMU,         eV/angstrom,  eV/angstrom.au
    lammps        : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    pwmat         : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2

    units['force_constants_unit'] is used in
    the 'get_force_constant_conversion_factor' method.

    """
    units = {
        "factor": None,
        "nac_factor": None,
        "distance_to_A": None,
        "force_to_eVperA": None,
        "force_constants_unit": None,
        "length_unit": None,
        "force_unit": None,
    }

    if interface_mode is None or interface_mode in ("vasp", "aims", "lammps", "pwmat"):
        units["factor"] = VaspToTHz
        units["nac_factor"] = Hartree * Bohr
        units["distance_to_A"] = 1.0
        units["force_constants_unit"] = "eV/angstrom^2"
        units["length_unit"] = "angstrom"
        units["force_unit"] = "eV/angstrom"
    elif interface_mode == "abinit":
        units["factor"] = AbinitToTHz
        units["nac_factor"] = Hartree / Bohr
        units["distance_to_A"] = Bohr
        units["force_constants_unit"] = "eV/angstrom.au"
        units["length_unit"] = "au"
        units["force_unit"] = "eV/angstrom"
    elif interface_mode == "qe":
        units["factor"] = PwscfToTHz
        units["nac_factor"] = 2.0
        units["distance_to_A"] = Bohr
        units["force_to_eVperA"] = Rydberg / Bohr
        units["force_constants_unit"] = "Ry/au^2"
        units["length_unit"] = "au"
        units["force_unit"] = "Ry/au"
    elif interface_mode == "wien2k":
        units["factor"] = Wien2kToTHz
        units["nac_factor"] = 2000.0
        units["distance_to_A"] = Bohr
        units["force_to_eVperA"] = 0.001 * Rydberg / Bohr
        units["force_constants_unit"] = "mRy/au^2"
        units["length_unit"] = "au"
        units["force_unit"] = "mRy/au"
    elif interface_mode == "elk":
        units["factor"] = ElkToTHz
        units["nac_factor"] = 1.0
        units["distance_to_A"] = Bohr
        units["force_constants_unit"] = "hartree/au^2"
        units["length_unit"] = "au"
        units["force_unit"] = "hartree/au"
    elif interface_mode in ["siesta", "abacus"]:
        units["factor"] = SiestaToTHz
        units["nac_factor"] = Hartree / Bohr
        units["distance_to_A"] = Bohr
        units["force_constants_unit"] = "eV/angstrom.au"
        units["length_unit"] = "au"
        units["force_unit"] = "eV/angstrom"
    elif interface_mode == "cp2k":
        units["factor"] = CP2KToTHz
        units["nac_factor"] = None  # not implemented
        units["distance_to_A"] = 1.0
        units["force_constants_unit"] = "hartree/angstrom.au"
        units["length_unit"] = "angstrom"
        units["force_unit"] = "hartree/au"
    elif interface_mode == "crystal":
        units["factor"] = CrystalToTHz
        units["nac_factor"] = Hartree * Bohr
        units["distance_to_A"] = 1.0
        units["force_constants_unit"] = "eV/angstrom^2"
        units["length_unit"] = "angstrom"
        units["force_unit"] = "eV/angstrom"
    elif interface_mode == "dftbp":
        units["factor"] = DftbpToTHz
        units["nac_factor"] = Hartree * Bohr
        units["distance_to_A"] = Bohr
        units["force_constants_unit"] = "hartree/au^2"
        units["length_unit"] = "au"
        units["force_unit"] = "hartree/au"
    elif interface_mode == "turbomole":
        units["factor"] = TurbomoleToTHz
        units["nac_factor"] = 1.0
        units["distance_to_A"] = Bohr
        units["force_to_eVperA"] = Hartree / Bohr
        units["force_constants_unit"] = "hartree/au^2"
        units["length_unit"] = "au"
        units["force_unit"] = "hartree/au"
    elif interface_mode == "castep":
        units["factor"] = CastepToTHz
        units["nac_factor"] = Hartree * Bohr
        units["distance_to_A"] = 1.0
        units["force_constants_unit"] = "eV/angstrom^2"
        units["length_unit"] = "angstrom"
        units["force_unit"] = "eV/angstrom"
    elif interface_mode == "fleur":
        units["factor"] = FleurToTHz
        units["nac_factor"] = 1.0
        units["distance_to_A"] = Bohr
        units["force_constants_unit"] = "hartree/au^2"
        units["length_unit"] = "au"
        units["force_unit"] = "hartree/au"

    return units


def get_calc_dataset(
    interface_mode: str,
    num_atoms: int,
    force_filenames: str,
    verbose: bool = True,
) -> dict:
    """Read calculator output files and parse force sets.

    Note
    ----
    Wien2k output is treated by ``get_calc_datasets_wien2k``.

    Returns
    -------
    dict:
        "forces": Set of forces in supercells.
        "supercell_energies": Set of supercell energies.

    """
    if interface_mode is None or interface_mode == "vasp":
        from phonopy.interface.vasp import parse_set_of_forces
    elif interface_mode == "abinit":
        from phonopy.interface.abinit import parse_set_of_forces
    elif interface_mode == "qe":
        from phonopy.interface.qe import parse_set_of_forces
    elif interface_mode == "elk":
        from phonopy.interface.elk import parse_set_of_forces
    elif interface_mode == "siesta":
        from phonopy.interface.siesta import parse_set_of_forces
    elif interface_mode == "cp2k":
        from phonopy.interface.cp2k import parse_set_of_forces
    elif interface_mode == "crystal":
        from phonopy.interface.crystal import parse_set_of_forces
    elif interface_mode == "dftbp":
        from phonopy.interface.dftbp import parse_set_of_forces
    elif interface_mode == "turbomole":
        from phonopy.interface.turbomole import parse_set_of_forces
    elif interface_mode == "aims":
        from phonopy.interface.aims import parse_set_of_forces
    elif interface_mode == "castep":
        from phonopy.interface.castep import parse_set_of_forces
    elif interface_mode == "fleur":
        from phonopy.interface.fleur import parse_set_of_forces
    elif interface_mode == "abacus":
        from phonopy.interface.abacus import parse_set_of_forces
    elif interface_mode == "lammps":
        from phonopy.interface.lammps import parse_set_of_forces
    elif interface_mode == "pwmat":
        from phonopy.interface.pwmat import parse_set_of_forces
    else:
        return []

    data_sets = parse_set_of_forces(num_atoms, force_filenames, verbose=verbose)
    if isinstance(data_sets, dict):
        return data_sets
    else:
        return {"forces": data_sets}


def get_calc_dataset_wien2k(
    force_filenames,
    supercell,
    disp_dataset,
    wien2k_P1_mode=False,
    symmetry_tolerance=None,
    verbose=False,
):
    """Read Wien2k output files and parse force sets."""
    from phonopy.interface.wien2k import parse_set_of_forces

    disps, _ = get_displacements_and_forces(disp_dataset)
    force_sets = parse_set_of_forces(
        disps,
        force_filenames,
        supercell,
        wien2k_P1_mode=wien2k_P1_mode,
        symmetry_tolerance=symmetry_tolerance,
        verbose=verbose,
    )
    return {"forces": force_sets}


def get_force_constant_conversion_factor(unit, interface_mode):
    """Return unit conversion factor of force constants."""
    _unit = unit.replace("Angstrom", "angstrom")  # for backward compatibility
    interface_default_units = get_default_physical_units(interface_mode)
    default_unit = interface_default_units["force_constants_unit"]
    factor_to_eVperA2 = {
        "eV/angstrom^2": 1,
        "eV/angstrom.au": 1 / Bohr,
        "Ry/au^2": Rydberg / Bohr**2,
        "mRy/au^2": Rydberg / Bohr**2 / 1000,
        "hartree/au^2": Hartree / Bohr**2,
        "hartree/angstrom.au": Hartree / Bohr,
    }
    if default_unit not in factor_to_eVperA2:
        msg = "Force constant conversion for %s unit is not implemented."
        raise NotImplementedError(msg)
    if default_unit != _unit:
        factor = factor_to_eVperA2[_unit] / factor_to_eVperA2[default_unit]
        return factor
    else:
        return 1.0


def _read_phonopy_yaml(filename, phonopy_yaml_cls: type[PhonopyYaml]):
    cell_filename = _get_cell_filename(filename, phonopy_yaml_cls)
    if cell_filename is None:
        return None, (None, None)

    phpy = phonopy_yaml_cls()
    try:
        phpy.read(cell_filename)
    except TypeError:  # yaml.load returns str: File format seems not YAML.
        return None, (cell_filename, None)
    except yaml.parser.ParserError:
        return None, (cell_filename, None)

    cell = phpy.unitcell
    return cell, (cell_filename, phpy)


def _get_cell_filename(filename, phonopy_yaml_cls: type[PhonopyYaml]):
    cell_filename = None

    default_filenames = []
    for fname in phonopy_yaml_cls.default_filenames:
        for ext in get_supported_file_extensions_for_compression():
            default_filenames.append(f"{fname}{ext}")

    for fname in [filename] + default_filenames:
        if fname and os.path.isfile(fname):
            cell_filename = fname
            break
    return cell_filename
