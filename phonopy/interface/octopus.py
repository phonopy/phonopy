# SPDX-License-Identifier: BSD-3-Clause
"""Octopus calculator interface."""

# Copyright (C) 2026 Martin Lueders


from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.interface.vasp import check_forces, get_drift_forces, read_vasp
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import elaborate_borns_and_epsilon


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
    # Octopus reconstructs lattice vector i as LatticeParameters[i] * row i,
    # so each row must be normalized by its own length.
    for i in range(3):
        lines.append(
            "  "
            f"{lattice_vectors[i, 0] / lattice_params[i]:.9f} | "
            f"{lattice_vectors[i, 1] / lattice_params[i]:.9f} | "
            f"{lattice_vectors[i, 2] / lattice_params[i]:.9f}"
        )
    lines.append("%\n")

    lines.append("%ReducedCoordinates")
    for symbol, pos in zip(symbols, scaled_positions, strict=True):
        lines.append(f'  "{symbol}" | {pos[0]:.9f} | {pos[1]:.9f} | {pos[2]:.9f}')
    lines.append("%")

    return lines


def read_octopus(filename: str | os.PathLike) -> PhonopyAtoms:
    """Read a cell from an Octopus geometry include file.

    This is the inverse of :func:`write_octopus`. It only parses the canonical
    geometry-include format written by :func:`get_octopus_structure_lines`,
    i.e. numeric ``%LatticeParameters``, ``%LatticeVectors`` and
    ``%ReducedCoordinates`` blocks. It is not a general Octopus input parser:
    variables (e.g. ``a = 5.64*angstrom``), unit expressions and ``include``
    directives are not resolved. The returned cell uses the same length unit as
    the file (atomic units, i.e. bohr, for files written by this interface).
    """
    with open(filename) as f:
        lines = f.read().splitlines()
    return get_cell_from_octopus_lines(lines)


def read_octopus_or_poscar(filename: str | os.PathLike) -> PhonopyAtoms:
    """Read a unit cell from an Octopus geometry file or a VASP-style POSCAR.

    The format is auto-detected (see :func:`is_octopus_geometry`). Octopus uses
    atomic units internally, so a POSCAR (lattice in Angstrom) is converted to
    Bohr, while an Octopus geometry include file is already in Bohr.
    """
    if is_octopus_geometry(filename):
        return read_octopus(filename)
    cell = read_vasp(filename)
    cell.cell = np.array(cell.cell) / get_physical_units().Bohr
    return cell


def is_octopus_geometry(filename: str | os.PathLike) -> bool:
    """Return whether a cell file is an Octopus geometry include file.

    Detection is based on the presence of an Octopus structure block
    (``%LatticeParameters``, ``%LatticeVectors`` or ``%ReducedCoordinates``),
    which never appears in a VASP-style ``POSCAR``. Used to let the interface
    accept either format as the unit-cell input.
    """
    markers = ("%latticeparameters", "%latticevectors", "%reducedcoordinates")
    with open(filename) as f:
        for line in f:
            if line.strip().lower().startswith(markers):
                return True
    return False


def get_cell_from_octopus_lines(lines: Sequence[str]) -> PhonopyAtoms:
    """Build a cell from the lines of an Octopus geometry include file."""
    blocks = _parse_octopus_blocks(lines)
    for name in ("latticeparameters", "latticevectors", "reducedcoordinates"):
        if name not in blocks:
            raise ValueError(f"'%{name}' block not found in Octopus geometry file.")

    params = [float(v) for line in blocks["latticeparameters"] for v in _cols(line)]
    if len(params) != 3:
        raise ValueError(
            "Expected 3 %LatticeParameters; the angle form is not supported."
        )

    vec_rows = [[float(v) for v in _cols(line)] for line in blocks["latticevectors"]]
    if len(vec_rows) != 3 or any(len(row) != 3 for row in vec_rows):
        raise ValueError("%LatticeVectors must contain a 3x3 matrix.")

    # In Octopus lattice vector i is scaled by lattice parameter i.
    lattice = np.array([params[i] * np.array(vec_rows[i]) for i in range(3)])

    symbols = []
    scaled_positions = []
    for line in blocks["reducedcoordinates"]:
        cols = _cols(line)
        if len(cols) != 4:
            raise ValueError(f"Malformed %ReducedCoordinates line: {line!r}")
        symbols.append(cols[0].strip("\"'"))
        scaled_positions.append([float(v) for v in cols[1:]])

    return PhonopyAtoms(
        symbols=symbols, scaled_positions=scaled_positions, cell=lattice
    )


def _cols(line: str) -> list[str]:
    """Split an Octopus block line at ``|`` and strip each column."""
    return [col.strip() for col in line.split("|")]


def _parse_octopus_blocks(lines: Sequence[str]) -> dict[str, list[str]]:
    """Collect ``%Block ... %`` sections, keyed by lower-case block name.

    Comments (``#``) and blank lines are dropped; lines outside any block
    (e.g. ordinary ``Variable = value`` statements) are ignored.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    data: list[str] = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("%"):
            name = line[1:].strip()
            if name:  # start of a block
                current = name.lower()
                data = []
            else:  # end of the current block
                if current is not None:
                    blocks[current] = data
                current = None
        elif current is not None:
            data.append(line)
    return blocks


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
    """Parse forces from Octopus static/info files.

    Forces are read from the "Forces on the ions" block of Octopus'
    ``static/info`` file, which looks like::

        Forces on the ions [H/b]
         Ion                        x              y              z
           1        Si  -7.96388897E-03  -4.49992101E-03  -3.91040537E-03
           2        Si   7.96388897E-03   4.49992101E-03   3.91040537E-03
         ----------------------------------------------------------

    Forces are printed in Hartree/bohr ([H/b]) by default. If the run uses
    eV/Angstrom units ([eV/A]), they are converted back to atomic units.
    """
    units = get_physical_units()
    force_sets = []
    is_parsed = True

    for i, filename in enumerate(forces_filenames):
        if verbose:
            print(f"{i + 1}. {filename}")

        octopus_forces: list[list[float]] = []
        with open(filename) as f:
            for line in f:
                if "Forces on the ions" in line:
                    if "[eV/A]" in line:
                        # eV/Angstrom -> Hartree/bohr
                        conversion = units.Bohr / units.Hartree
                    else:  # [H/b], atomic units
                        conversion = 1.0
                    f.readline()  # skip the " Ion   x   y   z" header line
                    for _ in range(num_atoms):
                        vals = f.readline().split()
                        if len(vals) < 5:
                            break
                        octopus_forces.append(
                            [float(vals[j]) * conversion for j in (2, 3, 4)]
                        )
                    break

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


def parse_octopus_epsilon(filename: str | os.PathLike) -> NDArray[np.double]:
    """Parse the dielectric tensor from an Octopus ``em_resp`` ``epsilon`` file.

    Reads the 3x3 real part of the macroscopic dielectric tensor (the block
    following the "Real part of dielectric constant" header).
    """
    with open(filename) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Real part of dielectric constant" in line:
            try:
                rows = [[float(x) for x in lines[i + 1 + j].split()] for j in range(3)]
            except (IndexError, ValueError) as e:
                raise ValueError(f"Malformed dielectric tensor in '{filename}'.") from e
            epsilon = np.array(rows, dtype="double")
            if epsilon.shape != (3, 3):
                raise ValueError(f"Dielectric tensor in '{filename}' is not 3x3.")
            return epsilon
    raise ValueError(f"Dielectric tensor not found in '{filename}'.")


def parse_octopus_born_charges(
    filename: str | os.PathLike,
) -> tuple[NDArray[np.double], list[str | None]]:
    """Parse Born effective-charge tensors from an Octopus ``born_charges`` file.

    Returns the tensors (``shape=(n_atoms, 3, 3)``) and the per-atom species
    labels, in the order Octopus printed them (its internal atom order). The
    trailing acoustic-sum-rule discrepancy block (which has no ``Index:`` header)
    is ignored.
    """
    with open(filename) as f:
        lines = f.readlines()
    borns: list[list[list[float]]] = []
    labels: list[str | None] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Index:"):
            label: str | None = None
            if "Label:" in lines[i]:
                tail = lines[i].split("Label:", 1)[1].split()
                if tail:
                    label = tail[0]
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("Real:"):
                # Complex Born charges (finite-frequency em_resp or complex
                # wavefunctions) are printed as "Real:"/"Imaginary:" blocks.
                raise ValueError(
                    f"Born charges in '{filename}' are complex "
                    "(frequency-dependent em_resp output or complex "
                    "wavefunctions); only static (omega=0, real) Born charges "
                    "are supported."
                )
            try:
                rows = [[float(x) for x in lines[i + 1 + j].split()] for j in range(3)]
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Malformed Born charge tensor in '{filename}' near line {i + 1}."
                ) from e
            borns.append(rows)
            labels.append(label)
            i += 4
        else:
            i += 1
    if not borns:
        raise ValueError(f"No Born effective charges found in '{filename}'.")
    return np.array(borns, dtype="double"), labels


def get_born_octopus(
    cell_filename: str | os.PathLike,
    epsilon_filename: str | os.PathLike,
    born_filename: str | os.PathLike,
    primitive_matrix: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
    is_symmetry: bool = True,
    symmetrize_tensors: bool = True,
    symprec: float = 1e-5,
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.int64]]:
    """Assemble NAC parameters from an Octopus ``em_resp`` calculation.

    The unit cell is read from either an Octopus geometry include file (see
    :func:`read_octopus`) or a VASP-style ``POSCAR`` (auto-detected), and the
    dielectric tensor and Born effective charges from the ``em_resp`` output.
    Octopus already uses the units expected by phonopy (Born charges in units of
    the elementary charge, dielectric tensor dimensionless), so the values are
    used as-is. Passing the Octopus geometry file that was included in the
    ``em_resp`` run guarantees the atom order matches ``born_charges``.

    Returns
    -------
    See :func:`phonopy.structure.symmetry.elaborate_borns_and_epsilon`.

    """
    if primitive_matrix is None:
        primitive_matrix = np.eye(3)
    if supercell_matrix is None:
        supercell_matrix = np.eye(3, dtype="int64")

    ucell = read_octopus_or_poscar(cell_filename)
    epsilon = parse_octopus_epsilon(epsilon_filename)
    borns, labels = parse_octopus_born_charges(born_filename)

    symbols = list(ucell.symbols)
    if len(borns) != len(symbols):
        raise ValueError(
            f"Number of Born charge tensors ({len(borns)}) does not match the "
            f"number of atoms in '{cell_filename}' ({len(symbols)})."
        )
    if labels != symbols:
        print(
            f"Warning: species labels in '{born_filename}' ({labels}) do not match "
            f"the cell '{cell_filename}' ({symbols}); ensure the geometry file "
            "corresponds to the em_resp calculation.",
            file=sys.stderr,
        )

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
