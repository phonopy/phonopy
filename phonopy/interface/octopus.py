"""Octopus calculator interface."""

# Copyright (C) 2025 Martin Lueders
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
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.file_IO import collect_forces
from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms


def write_octopus(filename: str | os.PathLike, cell: PhonopyAtoms) -> None:
    """Write geometry include file for octopus."""
    lines = get_octopus_structure_lines(cell)
    with open(filename, "w") as w:
        w.write("\n".join(lines))


def get_octopus_structure_lines(cell: PhonopyAtoms) -> list[str]:
    """Generate Octopus structure lines from a cell."""
    scaled_positions = cell.scaled_positions
    symbols = cell.symbols
    lattice_vectors = cell.cell
    lattice_params = np.linalg.norm(lattice_vectors, axis=1)

    lines = []
    lines.append("%LatticeParameters")
    lines.append(
        f" {lattice_params[0]:.9f} | {lattice_params[1]:.9f} | {lattice_params[2]:.9f}"
    )
    lines.append("%\n")

    lines.append("%LatticeVectors")
    for i in range(3):
        lines.append(
            "  "
            f"{lattice_vectors[i, 0] / lattice_params[0]:.9f} | "
            f"{lattice_vectors[i, 1] / lattice_params[1]:.9f} | "
            f"{lattice_vectors[i, 2] / lattice_params[2]:.9f}"
        )
    lines.append("%\n")

    lines.append("%ReducedCoordinates")
    for symbol, pos in zip(symbols, scaled_positions, strict=True):
        lines.append(f'  "{symbol}" | {pos[0]:.9f} | {pos[1]:.9f} | {pos[2]:.9f}')
    lines.append("%")

    return lines


def write_supercells_with_displacements(
    supercell: PhonopyAtoms,
    cells_with_displacements: Sequence[PhonopyAtoms],
    displacement_ids: Sequence[int] | NDArray[np.int_] | None = None,
    pre_filename: str = "geometry",
    width: int = 3,
) -> None:
    """Write supercells with displacements to files."""
    if displacement_ids is None:
        displacement_ids = np.arange(len(cells_with_displacements), dtype=int) + 1

    write_octopus(f"{pre_filename}-000", supercell)
    for i, cell in zip(displacement_ids, cells_with_displacements, strict=True):
        filename = f"{pre_filename}-{i:0{width}}"
        write_octopus(filename, cell)


def parse_set_of_forces(
    num_atoms: int,
    forces_filenames: Sequence[str | os.PathLike],
    verbose: bool = True,
) -> list[NDArray[np.double]] | None:
    """Parse forces from Octopus static/info files."""
    units = get_physical_units()
    force_sets = []
    is_parsed = True
    hook = "Ion "

    for i, filename in enumerate(forces_filenames):
        if verbose:
            print(f"{i + 1}. {filename}")

        conversion = 1.0
        with open(filename) as f:
            for line in f:
                if "Forces on the ions [H/b]" in line:
                    conversion = 1.0
                elif "Forces on the ions [eV/A]" in line:
                    conversion = units.Hartree / units.EV * units.Bohr / units.Angstrom

            f.seek(0)
            octopus_forces = collect_forces(f, num_atoms, hook, [2, 3, 4])

        for force in octopus_forces:
            force[0] *= conversion
            force[1] *= conversion
            force[2] *= conversion

        if check_forces(octopus_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                octopus_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(octopus_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    return None
