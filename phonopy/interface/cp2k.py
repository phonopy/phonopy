"""CP2K calculator interface."""

# vim: set fileencoding=utf-8 :
# Copyright (C) 2017-2019 Tiziano Müller
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

from __future__ import print_function

import sys
from fractions import Fraction
from os import path

import numpy as np

from phonopy.file_IO import iter_collect_forces
from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms, symbol_map


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        forces = iter_collect_forces(
            filename, num_atoms, "# Atom   Kind   Element", [3, 4, 5]
        )

        if not check_forces(forces, num_atoms, filename, verbose=verbose):
            return []  # if one file is invalid, the whole thing is broken

        drift_force = get_drift_forces(forces, filename=filename, verbose=verbose)
        force_sets.append(np.array(forces) - drift_force)

    return force_sets


def read_cp2k(filename):
    """Read crystal structure."""
    from cp2k_input_tools.parser import CP2KInputParser

    with open(filename) as fhandle:
        parser = CP2KInputParser()
        tree = parser.parse(fhandle)

    try:
        subsys = tree["+force_eval"][0]["+subsys"]
        cp2k_cell = subsys["+cell"]
    except IndexError:
        raise RuntimeError(
            "could not find a FORCE_EVAL/SUBSYS/CELL section in the given "
            "CP2K input file"
        )

    if len(tree["+force_eval"]) > 1:
        raise NotImplementedError(
            "the given CP2K input file contains multiple FORCE_EVAL sections, "
            "which is not (yet) supported"
        )

    # CP2K can get its cell information in two ways:
    # - A, B, C: cell vectors
    # - ABC: scaling of cell vectors, ALPHA_BETA_GAMMA: angles between the cell vectors
    # We'll parse either of them, but only write A, B, C
    if "a" in cp2k_cell:
        # unit vectors given
        unit_cell = np.array(
            [
                cp2k_cell["a"],
                cp2k_cell["b"],
                cp2k_cell["c"],
            ]
        )
    elif "abc" in cp2k_cell:
        # length of unit vectors given
        if "alpha_beta_gamma" in cp2k_cell:
            # if we also have the angles, construct the cell

            alpha, beta, gamma = cp2k_cell.pop("alpha_beta_gamma")

            cos_alpha = np.cos(alpha * np.pi / 180.0)
            cos_beta = np.cos(beta * np.pi / 180.0)
            cos_gamma = np.cos(gamma * np.pi / 180.0)
            sin_gamma = np.sin(gamma * np.pi / 180.0)

            unit_cell = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [cos_gamma, sin_gamma, 0.0],
                    [
                        cos_beta,
                        (cos_alpha - cos_gamma * cos_beta) / sin_gamma,
                        np.sqrt(
                            1.0
                            - cos_beta**2
                            - ((cos_alpha - cos_gamma * cos_beta) / sin_gamma) ** 2
                        ),
                    ],
                ]
            )
        else:
            unit_cell = np.eye(3)

        a, b, c = cp2k_cell.pop(
            "abc"
        )  # remove them from the tree since we pass it along

        unit_cell[0, :] *= a
        unit_cell[1, :] *= b
        unit_cell[2, :] *= c

    if "+cell_ref" in cp2k_cell:
        print("WARNING: the &CELL_REF section must be manually adjusted")

    cp2k_coord = subsys["+coord"]

    numbers = []
    positions = []

    for coordline in cp2k_coord["*"]:
        # coordinates are a series of strings according to the CP2K schema
        fields = coordline.split()
        numbers += [symbol_map[fields[0]]]
        # positions can also be fractions
        positions += [[float(Fraction(f)) for f in fields[1:4]]]

    if cp2k_coord.get("scaled", False):  # the keyword can be unavailable, true or false
        return (
            PhonopyAtoms(numbers=numbers, cell=unit_cell, scaled_positions=positions),
            tree,
        )
    else:
        return (
            PhonopyAtoms(numbers=numbers, cell=unit_cell, positions=positions),
            tree,
        )


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    optional_structure_info,
    pre_filename="supercell",
    width=3,
):
    """Write supercells with displacements to files."""
    orig_fname, tree = optional_structure_info

    fbase, fext = path.splitext(orig_fname)
    pbase = tree["+global"]["project_name"]

    supercell_ref_name = "{}-supercell{}".format(fbase, fext)
    with open(supercell_ref_name, "w") as fhandle:
        fhandle.write(
            """\
# Generated by Phonopy, based on {fname}
# Original configuration with the generated supercell for comparison
""".format(
                fname=orig_fname
            )
        )
        write_cp2k(fhandle, "{}-{}".format(pbase, pre_filename), supercell, tree)

    for i, cell in zip(ids, cells_with_displacements):
        suffix = "{pre_filename}-{0:0{width}}".format(
            i, pre_filename=pre_filename, width=width
        )

        with open("{}-{}{}".format(fbase, suffix, fext), "w") as fhandle:
            fhandle.write(
                """\
# Generated by Phonopy, based on {fname}
# Merged configuration with displacements
""".format(
                    fname=orig_fname
                )
            )
            write_cp2k(fhandle, "{}-{}".format(pbase, suffix), cell, tree)


def write_cp2k_by_filename(filename, cell, tree, header=None):
    """Wrap write_cp2k to write an arbitrary unit cell.

    Note
    ----
    This method is written by Atsushi Togo to use at
        phonopy.interface.calculator.write_crystal_structure.
    I am not a user of cp2k and this method can be wrongly written.
    Please rewrite if somebody who knows well about cp2k input.

    """
    fbase, fext = path.splitext(filename)
    pbase = tree["+global"]["project_name"]
    project_name = "{}-{}".format(pbase, fbase)

    if header is None:
        _header = "# Generated by Phonopy\n"
    else:
        _header = header

    with open(filename, "w") as w:
        w.write(_header)
        write_cp2k(w, project_name, cell, tree)


def write_cp2k(fhandle, project_name, atoms, tree):
    """Merge the new the atoms structure with the configuration tree to a new CP2K input file.

    :param fhandle: open file handle to which the routine will write to
    :param project_name: the project name to use (CP2K uses that as prefix for generated files)i
    :param atoms: the Atoms objects to use
    :param tree: the configuration tree as returned from CP2KInputParser
    """
    from cp2k_input_tools.generator import CP2KInputGenerator

    generator = CP2KInputGenerator()

    tree["+global"]["run_type"] = "ENERGY_FORCE"
    tree["+global"]["project_name"] = project_name

    force_eval = tree["+force_eval"][0]
    subsys = force_eval["+subsys"]

    # if the original input contained scaled positions, continue with scaled positions
    if subsys["+coord"].get("scaled", False):
        cp2k_coord = {
            "scaled": True,
            "*": [
                "{sym} {x} {y} {z}".format(sym=sym, x=coord[0], y=coord[1], z=coord[2])
                for sym, coord in zip(
                    atoms.get_chemical_symbols(), atoms.get_scaled_positions()
                )
            ],
        }
    # ... otherwise use absolute positions
    else:
        cp2k_coord = {
            "*": [
                "{sym} {x} {y} {z}".format(sym=sym, x=coord[0], y=coord[1], z=coord[2])
                for sym, coord in zip(
                    atoms.get_chemical_symbols(), atoms.get_positions()
                )
            ],
        }

    subsys["+cell"]["a"] = list(atoms.get_cell()[0])
    subsys["+cell"]["b"] = list(atoms.get_cell()[1])
    subsys["+cell"]["c"] = list(atoms.get_cell()[2])
    subsys["+cell"]["periodic"] = "XYZ"  # anything else does not make much sense

    subsys["+coord"] = cp2k_coord  # overwriting the coordinates

    if "+print" not in force_eval:
        force_eval["+print"] = {}
    if "+forces" not in force_eval["+print"]:
        force_eval["+print"]["+forces"] = {}
    force_eval["+print"]["+forces"][
        "filename"
    ] = "forces"  # uses the project name as base with 'forces' as suffix

    for line in generator.line_iter(tree):
        fhandle.write("{line}\n".format(line=line))
