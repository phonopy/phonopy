"""LAMMPS calculator interface."""

# Copyright (C) 2023 Atsushi Togo
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
import re
import typing
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atomic_data import get_atomic_data
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_cell_matrix_from_lattice

_T = TypeVar("_T")


def write_supercells_with_displacements(
    supercell: PhonopyAtoms,
    cells_with_displacements: Sequence[PhonopyAtoms],
    ids: NDArray[np.int64] | Sequence[int],
    pre_filename: str | os.PathLike = "supercell",
    width: int = 3,
) -> None:
    """Write supercells with displacements to files."""
    write_lammps(pre_filename, supercell)
    for i, cell in zip(ids, cells_with_displacements, strict=True):
        filename = f"{pre_filename}-{i:0{width}d}"
        write_lammps(filename, cell)


def write_lammps(filename: str | os.PathLike, cell: PhonopyAtoms) -> None:
    """Write LAMMPS structure to file."""
    with open(filename, "w") as w:
        w.write("\n".join(LammpsStructureDumper(cell).get_lines()))


def read_lammps(cell_filename: str | os.PathLike) -> PhonopyAtoms:
    """Read LAMMPS structure file and return cell."""
    return LammpsStructureLoader().load(cell_filename).cell


def parse_set_of_forces(
    num_atoms: int,
    forces_filenames: Sequence[str | os.PathLike],
    verbose: bool = True,
) -> list[NDArray[np.double]]:
    """Parse forces from output files."""
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            print(f"{i + 1}. ", end="")
        forces = LammpsForcesLoader().load(filename).forces
        if check_forces(forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(forces, filename=filename, verbose=verbose)
            force_sets.append(np.array(forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def rotate_lammps_forces(
    force_sets: list[NDArray[np.double]],
    lattice: NDArray[np.double],
    verbose: bool = True,
) -> None:
    """Rotate forces of LAMMPS output.

    Parameters
    ----------
    force_sets : list
        Sets of supercell forces obtained by ``parse_set_of_forces``.
    lattice : ndarray
        Basis vectors in row vectors.
    verbose : bool
        Verbosity.

    """
    lat = lattice
    rot_lat = get_cell_matrix_from_lattice(lat)
    r = np.dot(np.linalg.inv(lat), rot_lat).T
    for forces in force_sets:
        # (Rinv.f)^T = f^T.Rinv^T = f^T.R
        forces[:] = np.dot(forces, r)
    if verbose:
        print("Forces parsed from LAMMPS output were rotated by F=R.F(lammps) with R:")
        for v in r.T:
            print(f"  {v[0]:7.5f} {v[1]:7.5f} {v[2]:7.5f}")


class LammpsStructureDumper:
    """Class to create LAMMPS input structure file.

    LAMMPS requires the basis vectors to be defined in the following way

    a = (a_x 0 0)
    b = (b_x b_y 0)
    c = (c_x c_y c_z)

    So the ``cell.cell`` has to be rotated to match this style. The coordinates
    of atoms have to be given in Cartesian coordinates with respect to this
    set of basis vectors.

    """

    def __init__(self, cell: PhonopyAtoms) -> None:
        """Init method."""
        self._lines: list[str] = []
        lattice = get_cell_matrix_from_lattice(cell.cell)
        lmps_cell = PhonopyAtoms(
            cell=lattice, scaled_positions=cell.scaled_positions, symbols=cell.symbols
        )
        self._run(lmps_cell)

    def get_lines(self) -> list[str]:
        """Return LAMMPS structure str lines."""
        return self._lines

    def _run(self, cell: PhonopyAtoms) -> None:
        unums, uids = np.unique(cell.numbers, return_index=True)
        usyms = [cell.symbols[i] for i in uids]
        num_map = {n: i for i, n in enumerate(unums)}

        lines = ["#", ""]
        lines.append(f"{len(cell.numbers)} atoms")
        lines.append(f"{len(np.unique(cell.numbers))} atom types")
        lines.append("")
        lines.append(f"0.0 {cell.cell[0, 0]} xlo xhi")
        lines.append(f"0.0 {cell.cell[1, 1]} ylo yhi")
        lines.append(f"0.0 {cell.cell[2, 2]} zlo zhi")
        lines.append("")
        lines.append(f"{cell.cell[1, 0]} {cell.cell[2, 0]} {cell.cell[2, 1]} xy xz yz")
        lines.append("")
        lines.append("")
        lines.append("Atom Type Labels")
        lines.append("")
        for i, symbol in enumerate(usyms):
            lines.append(f"{i + 1} {symbol}")
        lines.append("")
        lines.append("")
        lines.append("Atoms")
        lines.append("")
        for i, (position, number) in enumerate(
            zip(cell.positions, cell.numbers, strict=True)
        ):
            pos_str = f"{position[0]} {position[1]} {position[2]}"
            lines.append(f"{i + 1} {usyms[num_map[number]]} {pos_str}")
        self._lines = lines


class LammpsForcesLoader:
    """Class to load LAMMPS output file to get forces.

    Configuration by following three lines is assumed in the LAMMPS input script.

    dump phonopy all custom 1 forces.* id type x y z fx fy fz
    dump_modify phonopy format line "%d %d %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f"
    run 0

    Then ``forces.0`` file is created as the result of LAMMPS calculation. Forces
    are parsed from lines after:

    ITEM: ATOMS id type x y z fx fy fz

    The values of 6-8th columns are the forces in Cartesian coordinates.

    """

    _hooks = {
        "forces": r"ITEM:\s+ATOMS\s+id\s+type",
        "num_atoms": r"ITEM:\s+NUMBER\s+OF\s+ATOMS",
    }

    def __init__(self) -> None:
        """Init method."""
        self._forces: NDArray[np.double] | None = None

    def load(self, fp: str | os.PathLike | typing.IO) -> LammpsForcesLoader:
        """Load and parse LAMMPS structure file.

        Parameters
        ----------
        fp : filename or stream
            filename or stream.

        """
        return _load(self, fp, return_lines=False)

    @property
    def forces(self) -> NDArray[np.double]:
        """Return forces."""
        return self._forces  # type: ignore[return-value]

    def _parse(self, fp: typing.IO, column_start: int = 5, column_end: int = 8) -> None:
        """Parse lines of LAMMPS output file."""
        num_atoms = -1
        for line in fp:
            regex = re.compile(self._hooks["num_atoms"])
            if regex.search(line):
                num_atoms = int(fp.readline().strip())
                break

        for line in fp:
            regex = re.compile(self._hooks["forces"])
            if regex.search(line):
                break

        forces = np.zeros((num_atoms, 3), dtype="double")
        indices_found = [False] * num_atoms
        for i, line in enumerate(fp):
            if i == num_atoms:
                break
            ary = line.split()
            atom_id = int(ary[0])
            indices_found[atom_id - 1] = True
            forces[atom_id - 1] = np.array(ary[column_start:column_end], dtype="double")

        assert all(indices_found)
        self._forces = forces


class LammpsStructureLoader:
    """Class to load LAMMPS input structure file.

    lmps = LammpsStructureLoader()
    lmps.parse(lines)
    cell: PhonopyAtoms = lmps.cell

    """

    _header_hooks = {
        "xlo_xhi": (re.compile(r"xlo\s+xhi"), "_set_xlo_xhi"),
        "ylo_yhi": (re.compile(r"ylo\s+yhi"), "_set_xlo_xhi"),
        "zlo_zhi": (re.compile(r"zlo\s+zhi"), "_set_xlo_xhi"),
        "xy_xz_yz": (re.compile(r"xy\s+xz\s+yz"), "_set_xlo_xhi"),
        "atoms": (re.compile("atoms"), "_set_number_of_atoms"),
        "atom_types": (re.compile(r"atom\s+types"), "_set_number_of_atoms"),
    }

    def __init__(self) -> None:
        """Init method."""
        self._header_tags: dict[str, Any] = {}
        self._atom_type_labels: dict[str, int] = {}
        self._atom_ids: NDArray[np.int64] | None = None
        self._atom_labels: list[str] | NDArray[np.int64] | None = None
        self._atom_positions: NDArray[np.double] | None = None
        self._cell: PhonopyAtoms | None = None

    @property
    def cell(self) -> PhonopyAtoms:
        """Return parsed cell."""
        return self._cell  # type: ignore[return-value]

    def load(self, fp: str | os.PathLike | typing.IO) -> LammpsStructureLoader:
        """Load and parse LAMMPS structure file.

        Parameters
        ----------
        fp : filename or stream
            filename or stream.

        """
        return _load(self, fp)

    def _parse(self, lines: list[str]) -> None:
        """Parse LAMMPS structure file."""
        re_ATL = re.compile(r"Atom\s+Type\s+Labels")
        for line in lines:
            for key, (regex, method_name) in self._header_hooks.items():
                if regex.search(line):
                    getattr(self, method_name)(key, regex.sub("", line))
                    break

        i = 0
        while True:
            if i > len(lines):
                break
            line = lines[i]
            # Hook "Atom Type Labels"
            if re_ATL.search(line):
                i += self._parse_AtomTypeLabels(lines[(i + 1) :])
                continue
            # Hook "Atoms", this must be after "Atom Type Labels".
            if "Atoms" == line.split("#")[0].strip():
                self._parse_Atoms(lines[(i + 1) :])
                break
            i += 1

        lattice = np.zeros((3, 3), dtype="double", order="C")
        tag = self._header_tags
        lattice[0, 0] = tag["xlo_xhi"][1] - tag["xlo_xhi"][0]
        lattice[1, 1] = tag["ylo_yhi"][1] - tag["ylo_yhi"][0]
        lattice[2, 2] = tag["zlo_zhi"][1] - tag["zlo_zhi"][0]
        lattice[1, 0] = tag["xy_xz_yz"][0]  # xy
        lattice[2, 0] = tag["xy_xz_yz"][1]  # xz
        lattice[2, 1] = tag["xy_xz_yz"][2]  # yz

        if self._atom_type_labels:
            self._cell = PhonopyAtoms(
                cell=lattice,
                positions=self._atom_positions,
                symbols=self._atom_labels,  # type: ignore[arg-type]
            )
        else:
            atom_data = get_atomic_data().atom_data
            symbols = [atom_data[n][1] for n in self._atom_labels]  # type: ignore[union-attr,index]
            self._cell = PhonopyAtoms(
                cell=lattice, positions=self._atom_positions, symbols=symbols
            )

    def _parse_AtomTypeLabels(self, lines: list[str]) -> int:
        num_types = 0
        for i, line in enumerate(lines):  # noqa: B007
            _line = line.split("#")[0].strip()
            if _line == "":
                continue
            ary = _line.split()
            self._atom_type_labels[ary[1]] = int(ary[0])
            num_types += 1
            if num_types == self._header_tags["atom_types"]:
                break
        assert num_types == self._header_tags["atom_types"]
        return i + 1

    def _parse_Atoms(self, lines: list[str]) -> None:
        positions = np.zeros((self._header_tags["atoms"], 3), dtype="double", order="C")
        if self._atom_type_labels:
            lables: Any = []
        else:
            lables = np.zeros(self._header_tags["atoms"], dtype="int64")
        ids = np.zeros(self._header_tags["atoms"], dtype="int64")
        num_atoms = 0
        for line in lines:
            _line = line.split("#")[0].strip()
            if _line == "":
                continue
            ary = _line.split()
            ids[num_atoms] = int(ary[0])

            # ary[1] can be chemical symbol (key) or id (val).
            # Supporting id is useful for "write_data" LAMMPS command.
            if self._atom_type_labels:
                for key, val in self._atom_type_labels.items():
                    if ary[1] in (key, f"{val}"):
                        lables.append(key)
                        break
            else:
                lables[num_atoms] = int(ary[1])
            positions[num_atoms] = [float(v) for v in ary[2:5]]
            num_atoms += 1
            if num_atoms == self._header_tags["atoms"]:
                break
        assert num_atoms == self._header_tags["atoms"], (
            f"{num_atoms} != {self._header_tags['atoms']}"
        )
        self._atom_ids = ids
        self._atom_labels = lables
        self._atom_positions = positions

    def _set_xlo_xhi(self, key: str, line: str) -> None:
        self._header_tags[key] = np.array(
            [float(v) for v in line.split("#")[0].split()]
        )

    def _set_number_of_atoms(self, key: str, line: str) -> None:
        self._header_tags[key] = int(line.split("#")[0])


def _load(
    self: Any, fp: str | os.PathLike | typing.IO, return_lines: bool = True
) -> Any:
    """Load and parse LAMMPS structure file.

    Parameters
    ----------
    fp : filename or stream
        filename or stream.

    """
    if isinstance(fp, io.IOBase):
        if return_lines:
            self._parse(fp.readlines())
        else:
            self._parse(fp)
    else:
        with open(fp) as f:  # type: ignore[arg-type]
            if return_lines:
                self._parse(f.readlines())
            else:
                self._parse(f)
    return self
