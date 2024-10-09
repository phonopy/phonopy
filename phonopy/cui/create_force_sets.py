"""Utilities to create force sets for main CUI script."""

# Copyright (C) 2020 Atsushi Togo
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

from typing import Optional

import numpy as np

from phonopy.cui.load_helper import get_nac_params
from phonopy.file_IO import parse_disp_yaml, write_FORCE_SETS
from phonopy.interface.calculator import get_calc_dataset, get_calc_dataset_wien2k
from phonopy.interface.lammps import rotate_lammps_forces
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.dataset import get_displacements_and_forces


def create_FORCE_SETS(
    interface_mode: str,
    force_filenames: list[str],
    phpy_yaml: Optional[PhonopyYaml] = None,
    symmetry_tolerance: Optional[float] = None,
    wien2k_P1_mode: bool = False,
    force_sets_zero_mode: bool = False,
    disp_filename: str = "disp.yaml",
    force_sets_filename: str = "FORCE_SETS",
    save_params: bool = False,
    log_level=0,
):
    """Create FORCE_SETS from phonopy_disp.yaml and calculator output files.

    Reading disp.yaml instead of phonopy_disp.yaml is deprecated.

    """
    if log_level > 0:
        if interface_mode:
            print(f"Calculator interface: {interface_mode}")
        print(f'Displacements were read from "{disp_filename}".')
        if disp_filename == "disp.yaml":
            print("")
            print("NOTE:")
            print(
                "  From phonopy v2.0, displacements are written into "
                '"phonopy_disp.yaml".'
            )
            print(
                '  "disp.yaml" is still supported for reading except for '
                "Wien2k interface, "
            )
            print("  and this supported will be removed at version 3.")
            print("")
        if force_sets_zero_mode:
            print(
                f'Forces in "{force_filenames[0]}" are subtracted from forces in all '
                "other files."
            )

    if disp_filename == "disp.yaml":
        if interface_mode == "wien2k":
            disp_dataset, supercell = parse_disp_yaml(
                filename=disp_filename, return_cell=True
            )
        else:
            disp_dataset = parse_disp_yaml(filename=disp_filename)
    elif phpy_yaml is not None:
        supercell = phpy_yaml.supercell
        disp_dataset = phpy_yaml.dataset
    else:
        raise RuntimeError("Could not read displacement dataset.")

    if "natom" in disp_dataset:  # type-1 dataset
        num_atoms = disp_dataset["natom"]
        num_displacements = len(disp_dataset["first_atoms"])
        dataset_type = 1
    elif "displacements" in disp_dataset:  # type-2 dataset
        num_atoms = disp_dataset["displacements"].shape[1]
        num_displacements = disp_dataset["displacements"].shape[0]
        dataset_type = 2
    else:
        raise RuntimeError(
            "Number of atoms could not be retrieved from %s" % disp_filename
        )
    if force_sets_zero_mode:
        num_displacements += 1

    if not check_number_of_force_files(
        num_displacements, force_filenames, disp_filename
    ):
        force_sets = []
    elif interface_mode == "wien2k":
        calc_dataset = get_calc_dataset_wien2k(
            force_filenames,
            supercell,
            disp_dataset,
            wien2k_P1_mode=wien2k_P1_mode,
            symmetry_tolerance=symmetry_tolerance,
            verbose=(log_level > 0),
        )
        force_sets = calc_dataset["forces"]
    else:
        calc_dataset = get_calc_dataset(
            interface_mode,
            num_atoms,
            force_filenames,
            verbose=(log_level > 0),
        )
        force_sets = calc_dataset["forces"]
        if "points" in calc_dataset:
            if force_sets_zero_mode:
                range_start = 1
            else:
                range_start = 0
            if filename := check_agreements_of_displacements(
                supercell,
                disp_dataset,
                calc_dataset["points"][range_start:],
                force_filenames[range_start:],
            ):
                raise RuntimeError(
                    f'Displacements don\'t match with atomic positions in "{filename}".'
                )
            if force_sets_zero_mode:
                if check_agreement_of_supercell_positions(
                    supercell, calc_dataset["points"][0]
                ):
                    raise RuntimeError(
                        "Supercell doesn't match with atomic positions in "
                        f'"{force_filenames[0]}".'
                    )

    if interface_mode == "lammps":
        rotate_lammps_forces(force_sets, supercell.cell, verbose=(log_level > 0))

    if force_sets:
        if force_sets_zero_mode:
            force_sets = _subtract_residual_forces(force_sets)
        if dataset_type == 1:
            for forces, disp in zip(force_sets, disp_dataset["first_atoms"]):
                disp["forces"] = forces
        elif dataset_type == 2:
            disp_dataset["forces"] = np.array(force_sets, dtype="double", order="C")
        else:
            raise RuntimeError("Force sets could not be found.")

        if "supercell_energies" in calc_dataset:
            energies = np.array(calc_dataset["supercell_energies"], dtype="double")
            if dataset_type == 1:
                for energy, disp in zip(energies, disp_dataset["first_atoms"]):
                    disp["supercell_energy"] = energy
            elif dataset_type == 2:
                disp_dataset["supercell_energies"] = energies

        if save_params:
            phpy_yaml.dataset = disp_dataset
            nac_params = get_nac_params(primitive=phpy_yaml.primitive)
            if nac_params:
                phpy_yaml.nac_params = nac_params
            yaml_filename = "phonopy_params.yaml"
            with open(yaml_filename, "w") as w:
                w.write(str(phpy_yaml))
            if log_level > 0:
                print(f'"{yaml_filename}" has been created.')
        else:
            write_FORCE_SETS(disp_dataset, filename=force_sets_filename)
            if log_level > 0:
                print(f'"{force_sets_filename}" has been created.')

    else:
        if log_level > 0:
            print("%s could not be created." % force_sets_filename)


def check_number_of_force_files(num_displacements, force_filenames, disp_filename):
    """Verify number of supercell force files.

    This function is public because being used from phono3py.

    """
    if num_displacements != len(force_filenames):
        print(f"Number of files to be read ({len(force_filenames)}) don't match to")
        print(
            f'the number of displacements ({num_displacements}) in "{disp_filename}".'
        )
        return False
    else:
        return True


def check_agreements_of_displacements(
    supercell: PhonopyAtoms,
    dataset: dict,
    all_points: list[np.ndarray],
    force_filenames: list[str],
) -> Optional[str]:
    """Check agreements of displacements."""
    displacements = get_displacements_and_forces(dataset)[0] @ np.linalg.inv(
        supercell.cell
    )
    for disp, points, filename in zip(displacements, all_points, force_filenames):
        diff = supercell.scaled_positions + disp - points
        diff -= np.rint(diff)
        if (np.linalg.norm(diff @ supercell.cell, axis=1) > 1e-5).any():
            return filename


def check_agreement_of_supercell_positions(
    supercell: PhonopyAtoms, points: np.ndarray
) -> bool:
    """Check agreement of supercell positions."""
    diff = supercell.scaled_positions - points
    diff -= np.rint(diff)
    return (np.linalg.norm(diff @ supercell.cell, axis=1) > 1e-5).any()


def _subtract_residual_forces(force_sets):
    for i in range(1, len(force_sets)):
        force_sets[i] -= force_sets[0]
    return force_sets[1:]
