"""Symmetry-aware sampling of lattice parameters.

Supports the anisotropic QHA / machine-learning-potential workflow by
generating cells whose free lattice-vector lengths are sampled within
user-given ranges, either randomly (:func:`sample_strained_cells`) or on a
regular tensor grid (:func:`grid_strained_cells`). The independent free
lattice degrees of freedom (1 for cubic, 2 for tetragonal / hexagonal /
trigonal in the hexagonal setting, 3 for orthorhombic) are determined from
the crystal symmetry. Cell angles are held fixed, so monoclinic and
triclinic crystals (whose angles are additional degrees of freedom) are out
of scope. All lengths are in the native length unit of the input cell
(calculator dependent); no unit conversion is performed.

"""

from __future__ import annotations

import dataclasses
import itertools
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


def _check_conventional_cell(
    lengths: NDArray[np.double],
    std_lattice: NDArray[np.double],
    symprec: float,
) -> None:
    """Raise unless each lattice-vector row is the standard axis of that row.

    The free lattice DOF are assigned by row from the crystal system, which
    is only meaningful for the standardized conventional cell: there row 0 is
    a, row 1 is b and row 2 is c. Comparing the row lengths with those of
    spglib's standardized lattice tests exactly that, while still accepting a
    conventional cell that is rigidly rotated in Cartesian space (a rotation
    leaves every length unchanged).

    The primitive cell of a centred lattice is rejected here. Its rows are
    centring vectors, not crystal axes, so scaling them cannot reach the
    lattice DOF: for a body-centred tetragonal cell, for instance, all three
    rows have the same length and scaling them together changes the volume
    only, never c/a.

    """
    std_lengths = np.linalg.norm(std_lattice, axis=1)
    if np.allclose(lengths, std_lengths, rtol=1e-5, atol=symprec):
        return
    raise ValueError(
        "The cell is not the standardized conventional cell: its "
        f"lattice-vector lengths {np.round(lengths, 6).tolist()} do not match "
        f"the standardized ones {np.round(std_lengths, 6).tolist()}. The "
        "lattice DOF are taken per lattice-vector row (row 0 = a, row 1 = b, "
        "row 2 = c), so the conventional cell is required; a primitive cell "
        "of a centred lattice, or a non-standard axis order or setting, "
        "cannot be used. The conventional cell is written as BPOSCAR by "
        '"phonopy --symmetry".'
    )


def get_free_lattice_dof(cell: PhonopyAtoms, symprec: float = 1e-5) -> LatticeDOF:
    """Determine the free lattice-length DOF of a cell from its symmetry.

    The cell must be the standardized conventional cell, whose rows are the
    crystal axes a, b and c in that order. The DOF are then fixed by the
    crystal system alone, without inspecting the lattice-vector lengths.

    Parameters
    ----------
    cell : PhonopyAtoms
        Standardized conventional cell in its native length unit, e.g. the
        BPOSCAR written by "phonopy --symmetry".
    symprec : float, optional
        Symmetry search tolerance passed to spglib.

    Returns
    -------
    LatticeDOF

    Raises
    ------
    ValueError
        For monoclinic and triclinic crystals, whose cell angles are
        additional degrees of freedom not supported here, and for a cell that
        is not the standardized conventional one (e.g. the primitive cell of
        a centred lattice, or a rhombohedral cell in the rhombohedral
        setting).

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
    _check_conventional_cell(lengths, np.array(dataset.std_lattice), symprec)

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
    else:
        # Tetragonal, hexagonal and trigonal all have a = b != c in the
        # conventional setting (trigonal in the hexagonal setting, the only
        # one _check_conventional_cell accepts), so c is row 2 by convention.
        # The lengths are deliberately not consulted: an accidental a = c
        # would look isotropic while the crystal is still tetragonal.
        labels = ("a", "c")
        rows = {"a": (0, 1), "c": (2,)}
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


def grid_strained_cells(
    cell: PhonopyAtoms,
    dof: LatticeDOF,
    ranges: dict[str, tuple[float, float]],
    num: int | dict[str, int],
) -> list[PhonopyAtoms]:
    """Return cells on a regular tensor grid of the free lattice lengths.

    Each free DOF is sampled at evenly spaced points over its (min, max)
    range, and the cells are the full tensor (Cartesian) product. With a
    single ``num`` every axis gets that many points (a square grid of
    ``num ** len(dof.labels)`` cells); a per-axis mapping allows different
    counts per DOF. The lattice vectors a free DOF controls are scaled to the
    target length, so cell angles and fractional atomic positions are
    preserved.

    On this grid the cells whose rank is equal along every free axis form the
    main diagonal, a monotone 1D volume path. When every free DOF spans the
    same fractional strain with the same count -- e.g. symmetric +/- p percent
    ranges and equal counts -- that diagonal has constant cell shape and is the
    isotropic volume path a Vinet EOS cross-check fits.

    Parameters
    ----------
    cell : PhonopyAtoms
        Base crystal structure.
    dof : LatticeDOF
        Free lattice DOF from get_free_lattice_dof.
    ranges : dict
        Maps each free-DOF label to its (min, max) length range, in the
        native length unit of the cell. Must cover exactly dof.labels.
    num : int or dict
        Grid points per free DOF (each >= 2), as a single count shared by all
        axes or a mapping from each free-DOF label to its own count. The total
        number of cells is the product of the per-axis counts.

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
    counts = {label: num for label in dof.labels} if isinstance(num, int) else dict(num)
    if set(counts) != set(dof.labels):
        raise ValueError(
            f"grid counts must be given for exactly the free DOF {dof.labels}, "
            f"but got {tuple(counts)}."
        )
    for label, n in counts.items():
        if n < 2:
            raise ValueError(f"grid count for {label} must be at least 2, got {n}.")

    axes = {label: np.linspace(*ranges[label], counts[label]) for label in dof.labels}
    cells = []
    for targets in itertools.product(*(axes[label] for label in dof.labels)):
        lattice = cell.cell.copy()
        for label, target in zip(dof.labels, targets, strict=True):
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
    distance: float | None = None,
    max_distance: float | None = None,
    distance_per_atom: bool = False,
    count: int = 1,
    seed: int | None = None,
) -> tuple[list[PhonopyAtoms], NDArray[np.double]]:
    """Return random-displacement supercells and their displacements.

    Each unit cell is expanded by supercell_matrix and all its atoms are
    displaced in random directions, producing structures ready for
    machine-learning-potential training without any prior internal-coordinate
    relaxation. ``count`` supercells are generated per unit cell; the returned
    list is flat, cell 0's supercells first.

    Without max_distance every atom is displaced by exactly ``distance``, which
    suits training for harmonic force constants. Giving max_distance instead
    draws a random distance, spanning the large-amplitude region a
    temperature-dependent (SSCHA) calculation visits; see `max_distance`.

    With max_distance, distance_per_atom decides how the amplitude axis is
    sampled against the strain axis. Drawn per supercell (the default), a
    strained cell gets only ``count`` amplitudes, so with a small count the
    amplitude and the strain can end up correlated by chance -- large
    amplitudes landing mostly on large-volume cells, which an MLP cannot then
    separate. Drawn per atom, every supercell covers the whole amplitude range
    internally, so the coverage is complete at every strain even for
    ``count = 1``, at the cost of a supercell no longer having one amplitude
    that labels it. It also samples [distance, max_distance) directly instead
    of flooring at ``distance``: the reserved weight at the floor buys a share
    of wholly near-equilibrium supercells only when the distance is drawn per
    supercell.

    Parameters
    ----------
    unitcells : sequence of PhonopyAtoms
        Unit cells, e.g. from sample_strained_cells.
    supercell_matrix : array_like
        Supercell matrix, e.g. from the phonopy_disp.yaml.
    distance : float, optional
        Displacement distance in the native length unit of the cells. With
        max_distance it is the floor of the random distance rather than the
        distance itself; see `max_distance`. None uses phonopy's default
        distance.
    max_distance : float, optional
        Upper bound of the random displacement distance. One distance is drawn
        per supercell from the uniform distribution over [0, max_distance) and
        then raised to ``distance`` when smaller. When None, the distance is
        fixed to ``distance``.
    distance_per_atom : bool, optional
        Requires ``max_distance``. Draw the random distance per atom rather
        than per supercell, uniformly over [distance, max_distance). Default
        is False; see the note below on which to use when the lattice is
        strained.
    count : int, optional
        Number of random-displacement supercells per unit cell (default 1).
    seed : int, optional
        Base seed; cell i uses seed + i so the set is reproducible while
        each supercell gets distinct displacements.

    Returns
    -------
    list of PhonopyAtoms
        Displaced supercells, flat, cell 0's supercells first.
    NDArray[np.double]
        Cartesian displacements aligned with the supercells.
        shape=(len(unitcells) * count, natom_of_supercell, 3)

    """
    from phonopy import Phonopy

    supercells = []
    displacements = []
    for i, unitcell in enumerate(unitcells):
        phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix, log_level=0)
        phonon.generate_displacements(
            distance=distance,
            max_distance=max_distance,
            distance_per_atom=distance_per_atom,
            number_of_snapshots=count,
            random_seed=None if seed is None else seed + i,
        )
        displaced = phonon.supercells_with_displacements
        if displaced is None or any(d is None for d in displaced):
            raise RuntimeError("Failed to generate a displacement supercell.")
        supercells.extend(displaced)
        displacements.append(np.array(phonon.displacements, dtype="double"))
    return supercells, np.concatenate(displacements)


def build_strain_cells_manifest(
    *,
    phonopy_version: str,
    calculator: str,
    length_unit: str,
    source: str,
    dof: LatticeDOF,
    command_line: str,
    ranges: dict[str, tuple[float, float]],
    num: int | None,
    grid_shape: list[int] | None,
    displacement_distance: float | None,
    displacement_distance_max: float | None,
    displacement_distance_per_atom: bool,
    random_displacements: int | None,
    symprec: float,
    seed: int | None,
    sampling: str,
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
    num : int or None
        Number of random cells requested; None for grid sampling.
    grid_shape : list of int or None
        Grid points per free DOF for grid sampling; None for random sampling.
    displacement_distance : float or None
        Random-displacement distance (the minimum distance when
        displacement_distance_max is given), or None for plain unit cells (or
        the phonopy default distance).
    displacement_distance_max : float or None
        Maximum random-displacement distance, or None when the distance is
        fixed rather than sampled from a range.
    displacement_distance_per_atom : bool
        Whether the random distance was drawn per atom rather than per
        supercell.
    random_displacements : int or None
        Number of random-displacement supercells per cell, or None for plain
        unit cells.
    symprec : float
        Symmetry tolerance used for DOF detection.
    seed : int or None
        Resolved random seed, or None when the run is fully deterministic
        (grid sampling without random displacements).
    sampling : str
        Sampling mode of the free lattice lengths ("random" or "grid").
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
            "sampling": sampling,
            "ranges": {
                label: [float(lo), float(hi)] for label, (lo, hi) in ranges.items()
            },
            "num": None if num is None else int(num),
            "grid_shape": None if grid_shape is None else [int(n) for n in grid_shape],
            "displacement_distance": (
                None if displacement_distance is None else float(displacement_distance)
            ),
            "displacement_distance_max": (
                None
                if displacement_distance_max is None
                else float(displacement_distance_max)
            ),
            "displacement_distance_per_atom": bool(displacement_distance_per_atom),
            "random_displacements": (
                None if random_displacements is None else int(random_displacements)
            ),
            "symprec": float(symprec),
            "seed": None if seed is None else int(seed),
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


@dataclasses.dataclass(frozen=True)
class StrainCells:
    """Structures and displacements of random-displacement supercells.

    The ideal (undisplaced) fractional coordinates are shared by every cell,
    because straining only scales the lattice vectors. The displaced structure
    of cell i is therefore ``ideal_scaled_positions @ lattices[i] +
    displacements[i]`` in Cartesian coordinates.

    """

    ideal_scaled_positions: NDArray[np.double]
    """Ideal fractional coordinates shared by all cells. shape=(natom, 3)"""

    lattices: NDArray[np.double]
    """Lattice of each cell, row vectors. shape=(ncell, 3, 3)"""

    displacements: NDArray[np.double]
    """Cartesian displacements of each cell. shape=(ncell, natom, 3)"""

    numbers: NDArray[np.int64]
    """Atomic numbers. shape=(natom,)"""

    masses: NDArray[np.double]
    """Atomic masses. shape=(natom,)"""

    supercell_matrix: NDArray[np.int64]
    """Supercell matrix the cells were expanded by. shape=(3, 3)"""

    calculator: str
    """Calculator (interface) name the cells were written for."""

    length_unit: str
    """Native length unit of the cells."""

    phonopy_version: str | None = None
    """Version of phonopy that produced the cells."""

    displacement_distance: float | None = None
    """Displacement distance, or its floor when displacement_distance_max is set."""

    displacement_distance_max: float | None = None
    """Upper bound of the random displacement distance."""

    displacement_distance_per_atom: bool = False
    """Whether the random distance was drawn per atom rather than per supercell."""


def write_strain_cells(
    filename: str | os.PathLike,
    *,
    unitcells: Sequence[PhonopyAtoms],
    supercells: Sequence[PhonopyAtoms],
    displacements: NDArray[np.double],
    supercell_matrix: NDArray[np.int64] | Sequence[Sequence[int]],
    phonopy_version: str | None = None,
    calculator: str = "vasp",
    length_unit: str = "angstrom",
    displacement_distance: float | None = None,
    displacement_distance_max: float | None = None,
    displacement_distance_per_atom: bool = False,
) -> None:
    """Write structures and displacements of a phonopy-strain-cells run.

    Stores the displacements themselves rather than only the displaced
    structures, so a consumer never has to rebuild the ideal reference to
    recover them. The written file reproduces the displaced supercells
    exactly.

    One ideal-position array is stored for all cells, which requires every
    unit cell to share its fractional coordinates. That holds while straining
    only scales lattice vectors; it is checked here so that a future
    internal-coordinate relaxation cannot break it silently.

    Parameters
    ----------
    filename : str or os.PathLike
        Output path.
    unitcells : sequence of PhonopyAtoms
        Strained unit cells the supercells were built from.
    supercells : sequence of PhonopyAtoms
        Displaced supercells from build_random_displacement_supercells.
    displacements : NDArray[np.double]
        Cartesian displacements aligned with supercells.
        shape=(ncell, natom, 3)
    supercell_matrix : array_like
        Supercell matrix the unit cells were expanded by. shape=(3, 3)
    phonopy_version : str or None, optional
        Version string of phonopy that produced the cells. Default is None.
    calculator : str, optional
        Calculator (interface) name. Default is "vasp".
    length_unit : str, optional
        Native length unit of the cells. Default is "angstrom".
    displacement_distance : float or None, optional
        Displacement distance, or its floor when displacement_distance_max is
        given. Default is None.
    displacement_distance_max : float or None, optional
        Upper bound of the random displacement distance. Default is None.
    displacement_distance_per_atom : bool, optional
        Whether the random distance was drawn per atom rather than per
        supercell. Default is False.

    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    from phonopy.structure.cells import get_supercell

    if not unitcells or not supercells:
        raise ValueError("No cells to write.")

    reference = np.array(unitcells[0].scaled_positions)
    for i, unitcell in enumerate(unitcells[1:], start=1):
        if not np.allclose(unitcell.scaled_positions, reference):
            raise ValueError(
                f"Unit cell {i} does not share the fractional coordinates of "
                "unit cell 0, so one ideal-position array cannot describe "
                "every cell."
            )

    ideal = get_supercell(unitcells[0], np.array(supercell_matrix, dtype="int64"))
    lattices = np.array([supercell.cell for supercell in supercells], dtype="double")
    disps = np.array(displacements, dtype="double")
    if disps.shape != (len(supercells), len(ideal), 3):
        raise ValueError(
            f"displacements has shape {disps.shape}, expected "
            f"{(len(supercells), len(ideal), 3)}."
        )

    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        if phonopy_version is not None:
            w.attrs["phonopy_version"] = phonopy_version
        w.attrs["calculator"] = calculator
        w.attrs["length_unit"] = length_unit
        w.attrs["n_cells"] = len(supercells)
        if displacement_distance is not None:
            w.attrs["displacement_distance"] = float(displacement_distance)
        if displacement_distance_max is not None:
            w.attrs["displacement_distance_max"] = float(displacement_distance_max)
            w.attrs["displacement_distance_per_atom"] = bool(
                displacement_distance_per_atom
            )
        w.create_dataset(
            "supercell_matrix", data=np.array(supercell_matrix, dtype="int64")
        )
        w.create_dataset("numbers", data=np.array(ideal.numbers, dtype="int64"))
        w.create_dataset("masses", data=np.array(ideal.masses, dtype="double"))
        w.create_dataset(
            "ideal_scaled_positions",
            data=np.array(ideal.scaled_positions, dtype="double"),
        )
        w.create_dataset("lattices", data=lattices, compression="gzip")
        w.create_dataset("displacements", data=disps, compression="gzip")


def read_strain_cells(filename: str | os.PathLike) -> StrainCells:
    """Read structures and displacements written by write_strain_cells.

    Parameters
    ----------
    filename : str or os.PathLike
        Input path.

    Returns
    -------
    StrainCells

    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    with h5py.File(filename, "r") as r:
        phonopy_version = r.attrs.get("phonopy_version")
        distance = r.attrs.get("displacement_distance")
        distance_max = r.attrs.get("displacement_distance_max")
        # Absent in files written before the per-atom mode existed.
        per_atom = r.attrs.get("displacement_distance_per_atom", False)
        return StrainCells(
            ideal_scaled_positions=r["ideal_scaled_positions"][:],
            lattices=r["lattices"][:],
            displacements=r["displacements"][:],
            numbers=r["numbers"][:],
            masses=r["masses"][:],
            supercell_matrix=r["supercell_matrix"][:],
            calculator=str(r.attrs["calculator"]),
            length_unit=str(r.attrs["length_unit"]),
            phonopy_version=None if phonopy_version is None else str(phonopy_version),
            displacement_distance=None if distance is None else float(distance),
            displacement_distance_max=(
                None if distance_max is None else float(distance_max)
            ),
            displacement_distance_per_atom=bool(per_atom),
        )
