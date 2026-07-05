"""Symmetry-aware random sampling of lattice parameters.

Supports the anisotropic QHA / machine-learning-potential workflow by
generating cells whose free lattice-vector lengths are randomly sampled
within user-given ranges. The independent free lattice degrees of freedom
(1 for cubic, 2 for tetragonal / hexagonal / trigonal in the hexagonal
setting, 3 for orthorhombic) are determined from the crystal symmetry.
Cell angles are held fixed, so monoclinic and triclinic crystals (whose
angles are additional degrees of freedom) are out of scope. All lengths
are in the native length unit of the input cell (calculator dependent);
no unit conversion is performed.

"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import spglib
from numpy.typing import NDArray

from phonopy.structure.atoms import PhonopyAtoms


@dataclasses.dataclass(frozen=True)
class LatticeDOF:
    """Free lattice-length degrees of freedom of a crystal.

    Attributes
    ----------
    crystal_system : str
        Crystal system name (e.g. "hexagonal").
    spacegroup_number : int
        International space-group number.
    labels : tuple of str
        Free-DOF labels among "a", "b", "c".
    rows : dict
        Maps each label to the lattice-vector row indices it scales
        (symmetry-tied vectors share a label).
    current_lengths : dict
        Maps each label to the current lattice-vector length.
    tie_description : str
        Human-readable description of the tied lengths (e.g. "b = a"),
        empty when no lengths are tied.

    """

    crystal_system: str
    spacegroup_number: int
    labels: tuple[str, ...]
    rows: dict[str, tuple[int, ...]]
    current_lengths: dict[str, float]
    tie_description: str


def _crystal_system(number: int) -> str:
    """Return the crystal system name for an international space-group number."""
    if number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    return "cubic"


def _unique_axis_row(lengths: NDArray[np.double], tol: float) -> int | None:
    """Return the row index of the odd-length axis, or None if all equal.

    For a two-length crystal (a = b != c) the two nearly equal rows are the
    tied a-axes and the remaining row is c. Returns None when all three
    lengths are equal within tol (isotropic length, e.g. a rhombohedral
    cell in the rhombohedral setting).

    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    diffs = [abs(lengths[i] - lengths[j]) for i, j in pairs]
    if max(diffs) <= tol:
        return None
    tied = pairs[int(np.argmin(diffs))]
    return next(r for r in range(3) if r not in tied)


def get_free_lattice_dof(cell: PhonopyAtoms, symprec: float = 1e-5) -> LatticeDOF:
    """Determine the free lattice-length DOF of a cell from its symmetry.

    Parameters
    ----------
    cell : PhonopyAtoms
        Crystal structure in its native length unit.
    symprec : float, optional
        Symmetry search tolerance passed to spglib.

    Returns
    -------
    LatticeDOF

    Raises
    ------
    ValueError
        For monoclinic and triclinic crystals, whose cell angles are
        additional degrees of freedom not supported here.

    """
    dataset = spglib.get_symmetry_dataset(
        cell.totuple(),  # type: ignore[arg-type]
        symprec,
    )
    if dataset is None:
        raise RuntimeError("Space group could not be determined.")
    number = int(dataset.number)
    system = _crystal_system(number)

    if system in ("triclinic", "monoclinic"):
        raise ValueError(
            f"{system} crystals are not supported: their cell angles are "
            "additional degrees of freedom."
        )

    lengths = np.linalg.norm(cell.cell, axis=1)

    labels: tuple[str, ...]
    rows: dict[str, tuple[int, ...]]
    tie: str
    if system == "cubic":
        labels = ("a",)
        rows = {"a": (0, 1, 2)}
        tie = "b = c = a"
    elif system == "orthorhombic":
        labels = ("a", "b", "c")
        rows = {"a": (0,), "b": (1,), "c": (2,)}
        tie = ""
    else:  # tetragonal, hexagonal, trigonal
        c_row = _unique_axis_row(lengths, tol=float(lengths.mean()) * 1e-3)
        if c_row is None:
            labels = ("a",)
            rows = {"a": (0, 1, 2)}
            tie = "b = c = a"
        else:
            a_rows = tuple(r for r in range(3) if r != c_row)
            labels = ("a", "c")
            rows = {"a": a_rows, "c": (c_row,)}
            tie = "b = a"

    current_lengths = {label: float(lengths[rows[label][0]]) for label in labels}
    return LatticeDOF(
        crystal_system=system,
        spacegroup_number=number,
        labels=labels,
        rows=rows,
        current_lengths=current_lengths,
        tie_description=tie,
    )


def sample_strained_cells(
    cell: PhonopyAtoms,
    dof: LatticeDOF,
    ranges: dict[str, tuple[float, float]],
    num: int,
    seed: int | None = None,
) -> list[PhonopyAtoms]:
    """Return cells with free lattice lengths uniformly sampled in ranges.

    Each free DOF is drawn independently from a uniform distribution over
    its (min, max) range and applied by scaling the lattice vectors it
    controls, so cell angles and fractional atomic positions are preserved.

    Parameters
    ----------
    cell : PhonopyAtoms
        Base crystal structure.
    dof : LatticeDOF
        Free lattice DOF from get_free_lattice_dof.
    ranges : dict
        Maps each free-DOF label to its (min, max) length range, in the
        native length unit of the cell. Must cover exactly dof.labels.
    num : int
        Number of cells to generate.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    list of PhonopyAtoms

    """
    if set(ranges) != set(dof.labels):
        raise ValueError(
            f"ranges must be given for exactly the free DOF {dof.labels}, "
            f"but got {tuple(ranges)}."
        )
    for label, (lo, hi) in ranges.items():
        if not lo < hi:
            raise ValueError(f"range for {label} must have min < max, got {(lo, hi)}.")
    if num < 1:
        raise ValueError("num must be a positive integer.")

    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(num):
        lattice = cell.cell.copy()
        for label in dof.labels:
            lo, hi = ranges[label]
            target = rng.uniform(lo, hi)
            scale = target / dof.current_lengths[label]
            for row in dof.rows[label]:
                lattice[row] *= scale
        cells.append(
            PhonopyAtoms(
                symbols=cell.symbols,
                cell=lattice,
                scaled_positions=cell.scaled_positions,
                masses=cell.masses,
            )
        )
    return cells


def build_random_displacement_supercells(
    unitcells: Sequence[PhonopyAtoms],
    supercell_matrix: NDArray[np.int64] | Sequence[Sequence[int]],
    distance: float,
    seed: int | None = None,
) -> list[PhonopyAtoms]:
    """Return one random-displacement supercell per input unit cell.

    Each unit cell is expanded by supercell_matrix and all its atoms are
    displaced in random directions by a fixed distance, producing a
    structure ready for machine-learning-potential training without any
    prior internal-coordinate relaxation.

    Parameters
    ----------
    unitcells : sequence of PhonopyAtoms
        Unit cells, e.g. from sample_strained_cells.
    supercell_matrix : array_like
        Supercell matrix, e.g. from the phonopy_disp.yaml.
    distance : float
        Displacement distance in the native length unit of the cells.
    seed : int, optional
        Base seed; cell i uses seed + i so the set is reproducible while
        each supercell gets distinct displacements.

    Returns
    -------
    list of PhonopyAtoms

    """
    from phonopy import Phonopy

    supercells = []
    for i, unitcell in enumerate(unitcells):
        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix, log_level=0)
        phonon.generate_displacements(
            distance=distance,
            number_of_snapshots=1,
            random_seed=None if seed is None else seed + i,
        )
        displaced = phonon.supercells_with_displacements
        if displaced is None or displaced[0] is None:
            raise RuntimeError("Failed to generate a displacement supercell.")
        supercells.append(displaced[0])
    return supercells


def build_strain_cells_manifest(
    *,
    phonopy_version: str,
    calculator: str,
    length_unit: str,
    source: str,
    dof: LatticeDOF,
    command_line: str,
    ranges: dict[str, tuple[float, float]],
    num: int,
    rd_distance: float | None,
    symprec: float,
    seed: int,
    prefix: str,
    kind: str,
    unitcells: Sequence[PhonopyAtoms],
    filenames: Sequence[str],
) -> dict[str, Any]:
    """Build a provenance manifest for a phonopy-strain-cells run.

    The manifest records everything needed to reproduce the run -- most
    importantly the resolved random seed -- together with the sampled free
    lattice lengths of every generated cell, so the (a[, b], c) grid can be
    matched against the files later consumed by run_anisotropic_qha.

    The recorded per-cell lengths are the unit-cell free-DOF lengths (the
    physically meaningful strained grid), even when the written files are
    random-displacement supercells.

    Parameters
    ----------
    phonopy_version : str
        Version string of phonopy that produced the cells.
    calculator : str
        Calculator (interface) name the cells were written for.
    length_unit : str
        Native length unit of the cells.
    source : str
        Input phonopy(_disp).yaml filename.
    dof : LatticeDOF
        Free lattice DOF from get_free_lattice_dof.
    command_line : str
        Human-readable reconstruction of the invoked command.
    ranges : dict
        Sampled (min, max) length range per free-DOF label.
    num : int
        Number of cells requested.
    rd_distance : float or None
        Random-displacement distance, or None for plain unit cells.
    symprec : float
        Symmetry tolerance used for DOF detection.
    seed : int
        Resolved random seed (never None).
    prefix : str
        Filename prefix of the written cells ("unitcell" or "supercell").
    kind : str
        Human-readable description of the written cell kind.
    unitcells : sequence of PhonopyAtoms
        Sampled unit cells (before any supercell expansion).
    filenames : sequence of str
        Written filenames, aligned with unitcells.

    Returns
    -------
    dict

    """
    cells: list[dict[str, Any]] = []
    for filename, unitcell in zip(filenames, unitcells, strict=True):
        lengths = np.linalg.norm(unitcell.cell, axis=1)
        entry: dict[str, Any] = {"file": filename}
        for label in dof.labels:
            entry[label] = round(float(lengths[dof.rows[label][0]]), 6)
        cells.append(entry)

    return {
        "phonopy_version": phonopy_version,
        "calculator": calculator,
        "length_unit": length_unit,
        "source": source,
        "crystal_system": dof.crystal_system,
        "spacegroup_number": int(dof.spacegroup_number),
        "free_dof": list(dof.labels),
        "tie": dof.tie_description,
        "command_line": command_line,
        "parameters": {
            "ranges": {
                label: [float(lo), float(hi)] for label, (lo, hi) in ranges.items()
            },
            "num": int(num),
            "rd_distance": None if rd_distance is None else float(rd_distance),
            "symprec": float(symprec),
            "seed": int(seed),
        },
        "output": {
            "prefix": prefix,
            "kind": kind,
            "num_cells": len(cells),
            "cells": cells,
        },
    }


def write_strain_cells_manifest(
    filename: str | os.PathLike, manifest: dict[str, Any]
) -> None:
    """Write a strain-cells provenance manifest to a YAML file.

    Parameters
    ----------
    filename : str or os.PathLike
        Output path.
    manifest : dict
        Manifest from build_strain_cells_manifest.

    """
    import yaml  # type: ignore[import-untyped]

    with open(filename, "w") as w:
        yaml.dump(manifest, w, sort_keys=False, default_flow_style=False)
