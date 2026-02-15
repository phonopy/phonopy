"""Fleur calculator interface."""

# Copyright (C) 2021 Alexander Neukirchen
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
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.interface.vasp import (
    get_drift_forces,
    sort_positions_by_symbols,
)
from phonopy.structure.atoms import PhonopyAtoms


def parse_set_of_forces(
    num_atoms: int,
    forces_filenames: Sequence[str | os.PathLike],
    verbose: bool = True,
) -> list[NDArray]:
    """Parse forces from force output files."""
    force_sets: list[NDArray] = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            print(f"{i + 1}. ", end="")

        forces: list[list[float]] = []
        previous_line = ""
        with open(filename) as f:
            for line in f:
                if "force" in line:
                    if previous_line == "":
                        break
                    vec = [float(x) for x in previous_line.split()[:3]]
                    forces.append(vec)
                previous_line = line

        if len(forces) != num_atoms:
            if verbose:
                print(
                    f"Force count mismatch in {filename}: {len(forces)} vs {num_atoms}"
                )
            return []

        forces_array = np.array(forces)
        drift_force = get_drift_forces(forces_array, filename=filename, verbose=verbose)
        force_sets.append(forces_array - drift_force)

    return force_sets


def read_fleur(
    filename: str | os.PathLike,
) -> tuple[PhonopyAtoms, list[str], list[str]]:
    """Read crystal structure from a Fleur input file.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the Fleur input file.

    Returns
    -------
    tuple
        (PhonopyAtoms, speci, restlines) where speci is a list of species
        identifiers and restlines contains extra lines from the input file.

    """
    with open(filename) as f:
        fleur_in = FleurIn(f.readlines())
    tags = fleur_in.get_variables()
    avec = tags["avec"]
    speci = tags["atoms"]["speci"]
    numbers = [int(x.split(".")[0]) for x in speci]

    for i, n in enumerate(numbers):
        if n == 0:
            for j in range(1, 119):
                if j not in numbers:
                    numbers[i] = j
                    break

    positions = tags["atoms"]["positions"]

    return (
        PhonopyAtoms(numbers=numbers, cell=avec, scaled_positions=positions),
        speci,
        fleur_in.restlines,
    )


def write_fleur(
    filename: str | os.PathLike,
    cell: PhonopyAtoms,
    speci: list[str],
    restlines: list[str] | None,
) -> None:
    """Write crystal structure to a Fleur input file."""
    with open(filename, "w") as f:
        f.write(get_fleur_structure(cell, speci, restlines))


def write_supercells_with_displacements(
    supercell: PhonopyAtoms,
    cells_with_displacements: Sequence[PhonopyAtoms],
    ids: Sequence[int] | NDArray,
    speci: list[str],
    n_repeat: int,
    restlines: list[str] | None,
    pre_filename: str = "supercell",
    width: int = 3,
) -> None:
    """Write supercells with displacements to files."""
    supercell_repci = []
    for _speci in speci:
        supercell_repci += [_speci] * n_repeat
    write_fleur(f"{pre_filename}.in", supercell, supercell_repci, restlines)
    for i, cell in zip(ids, cells_with_displacements, strict=True):
        filename = f"{pre_filename}-{i:0{width}}.in"
        write_fleur(filename, cell, supercell_repci, restlines)


def get_fleur_structure(
    cell: PhonopyAtoms,
    speci: Sequence[str],
    restlines: list[str] | None,
) -> str:
    """Return Fleur structure in text.

    Parameters
    ----------
    cell : PhonopyAtoms
        Crystal structure.
    speci : sequence or None
        Species identifiers for all atoms.
    restlines : list of str or None
        Additional lines to append (title, job info, etc.).

    Returns
    -------
    str
        Fleur input file content as a string.

    """
    lattice = cell.cell

    num_atoms, reduced_speci, scaled_positions, sort_list = sort_positions_by_symbols(
        speci, cell.scaled_positions
    )
    assert scaled_positions is not None
    if restlines is None:
        restlines = [
            "\n".join(
                ["Title line (Generated by phonopy)", "Additional job info here", ""]
            )
        ]
        warnings.warn(
            "No additional job info given. Writing minimal file.",
            UserWarning,
            stacklevel=2,
        )

    total_atoms = sum(num_atoms)
    all_speci = []
    for n, v in zip(num_atoms, reduced_speci, strict=True):
        all_speci += [v] * n

    lines = [restlines[0]]
    for vec in lattice:
        lines.append(" %21.16f %21.16f %21.16f" % tuple(vec))
    lines.append("1.0")
    lines.append("1.0 1.0 1.0")
    lines.append("")
    lines.append(str(total_atoms))
    for i in range(total_atoms):
        line = str(all_speci[i]).ljust(6)
        for j in range(3):
            line += f" {scaled_positions[i][j]:.10f}"
        lines.append(line)
    lines.append("")
    lines += restlines[1:]
    return "\n".join(lines)


class FleurIn:
    """Parser for Fleur crystal structure input files.

    Attributes
    ----------
    restlines : list of str
        Extra lines from the input (title, job directives, etc.).

    """

    def __init__(self, lines: list[str]) -> None:
        """Init method.

        Parameters
        ----------
        lines : list of str
            Lines read from a Fleur input file.

        """
        self._tags: dict[str, list[list[float]] | dict | None] = {
            "atoms": None,
            "avec": None,
        }
        self._lines = lines[:]
        self.restlines: list[str] = []
        self._parse()

    def get_variables(self) -> dict:
        """Return parsed tags dictionary."""
        return self._tags

    def _parse(self) -> None:
        """Parse Fleur input file lines."""
        # Store title line as first rest line
        if self._lines and self._lines[0].strip():
            self.restlines.append(self._lines[0].strip())

        i = 0
        while i < len(self._lines):
            line = self._lines[i].strip()

            if not line or line.startswith("!"):
                i += 1
                continue
            if i == 0:
                i += 1
                continue
            if line.startswith("&input") or line.startswith("&end"):
                i += 1
                continue

            # Detect lattice vectors: 3 consecutive lines each containing ≥3 floats
            if (
                i + 2 < len(self._lines)
                and self._is_vector_line(line)
                and self._is_vector_line(self._lines[i + 1].strip())
                and self._is_vector_line(self._lines[i + 2].strip())
            ):
                i = self._parse_lattice(i)
                continue

            # Detect atom count line followed by atom data
            if self._try_parse_atoms(i):
                break

            i += 1

    def _parse_lattice(self, start: int) -> int:
        """Parse lattice vectors, lattice constant, and scale factors.

        Parameters
        ----------
        start : int
            Index of the first lattice vector line.

        Returns
        -------
        int
            Index of the next unparsed line.

        """
        avec: list[list[float]] = []
        for j in range(3):
            avec.append([float(x) for x in self._lines[start + j].split()[:3]])
        i = start + 3

        # Lattice constant
        lattcon = 1.0
        if i < len(self._lines):
            try:
                lattcon = float(self._lines[i].split()[0])
                i += 1
            except (ValueError, IndexError):
                pass

        # Scale factors
        scale = [1.0, 1.0, 1.0]
        if i < len(self._lines):
            try:
                parts = self._lines[i].split()
                if len(parts) >= 3:
                    scale = [float(parts[j]) for j in range(3)]
                    i += 1
            except (ValueError, IndexError):
                pass

        # Apply lattice constant and scale factors
        for j in range(3):
            for k in range(3):
                if scale[k] < 0:
                    scale[k] = np.sqrt(np.abs(scale[k]))
                if scale[k] == 0.0:
                    scale[k] = 1.0
                avec[j][k] = lattcon * avec[j][k] * scale[k]

        self._tags["avec"] = avec
        return i

    def _try_parse_atoms(self, start: int) -> bool:
        """Try to parse atom count and atomic positions starting at *start*.

        Parameters
        ----------
        start : int
            Index of the candidate atom-count line.

        Returns
        -------
        bool
            True if atoms were successfully parsed.

        """
        line = self._lines[start].strip()
        try:
            natoms = int(line.split()[0])
        except (ValueError, IndexError):
            return False

        if natoms <= 0 or start + natoms >= len(self._lines):
            return False

        # Validate that following lines look like atom data
        for j in range(1, natoms + 1):
            if not self._is_atom_line(self._lines[start + j]):
                return False

        # Parse atom data
        speci: list[str] = []
        positions: list[list[float]] = []
        for j in range(1, natoms + 1):
            tokens = self._lines[start + j].split()
            speci.append(tokens[0])
            positions.append([float(tokens[k]) for k in range(1, 4)])

        # Collect remaining lines after atom block
        atom_end = start + natoms + 1
        for j in range(atom_end, len(self._lines)):
            rest = self._lines[j].strip()
            if rest and not rest.startswith("&end"):
                self.restlines.append(rest)

        self._tags["atoms"] = {"speci": speci, "positions": positions}
        return True

    @staticmethod
    def _is_vector_line(line: str) -> bool:
        """Return True if *line* contains at least 3 float-parseable tokens."""
        parts = line.split()
        if len(parts) < 3:
            return False
        try:
            for k in range(3):
                float(parts[k])
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_atom_line(line: str) -> bool:
        """Return True if *line* looks like an atom data line (id + 3 coords)."""
        parts = line.split()
        if len(parts) < 4:
            return False
        try:
            float(parts[0])
            for k in range(1, 4):
                float(parts[k])
            return True
        except (ValueError, IndexError):
            return False
