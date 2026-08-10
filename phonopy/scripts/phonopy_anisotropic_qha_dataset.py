# SPDX-License-Identifier: BSD-3-Clause
"""Build the anisotropic QHA intermediate dataset from calculator outputs.

Gathers the scattered per-grid-point outputs into one self-contained
aniso_qha_dataset.hdf5 that the anisotropic QHA analysis reads. The force sets
come from the pre-computed displaced supercells of the phonon grid, the static
U(a, c) from the static single points of the static grid.

Omitting --phonon builds from the static grid alone: the grid points then
carry the cells, U and the electronic states but no displacements or forces,
which is the dataset a method with temperature-dependent force constants needs,
its free energies going to run_anisotropic_qha through phonon_free_energies.

The grid points are given as explicit path lists with --static and --phonon,
typically expanded by the shell, and the two lists are paired by position, so
no naming convention applies at all. Rather than trusting that pairing, the
builder checks it: the lattice of each static single point must match the cell
of the phonon grid point it is paired with, which catches a mis-pairing however
the directories happen to be named.

A phonon grid point is either a directory holding phonopy_disp.yaml and the
per-displacement disp-* subdirectories, or a single phonopy_params.yaml-like
file that already carries the forces (as written by ``phonopy-init -f``). The
latter keeps the force collection in phonopy's own tools, so this command
need not know how the displaced supercells were laid out.

In either case the grid-point cell is the relaxed cell of the static single
point, so internal-coordinate relaxation is honored. The displacements and
forces are stored raw; the analysis recomputes the force constants.

The static single point must come from VASP: the internal energy U(a, c) and
the electronic states are read from VASP outputs (vaspout.h5 / vasprun.xml),
because phonopy has no interface yet to read the static single-point energy of
other calculators.

"""

from __future__ import annotations

import glob
import os
from argparse import ArgumentParser, Namespace

import numpy as np
from numpy.typing import NDArray

import phonopy
from phonopy import Phonopy
from phonopy.harmonic.displacement import DisplacementDataset
from phonopy.interface.vasp import (
    electronic_states_from_vaspout,
    read_vasprun_calculation,
)
from phonopy.physical_units import get_calculator_physical_units
from phonopy.qha.anisotropic_dataset import (
    AnisoQHADataset,
    AnisoQHAGridPoint,
    write_aniso_qha_dataset,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.qha.lattice_sampling import get_free_lattice_dof
from phonopy.structure.atoms import PhonopyAtoms


def calc_file(dirpath: str) -> str:
    """Return dirpath/vaspout.h5 if present, else dirpath/vasprun.xml.

    vaspout.h5 keeps full numerical precision, so prefer it for forces and
    energies when available.

    """
    h5 = os.path.join(dirpath, "vaspout.h5")
    return h5 if os.path.exists(h5) else os.path.join(dirpath, "vasprun.xml")


def as_calc_file(path: str) -> str:
    """Return the calculator output file for a path that may be a directory.

    A directory is resolved to the VASP output it contains; a file is taken as
    given. This lets --static accept either "grid-000" or
    "grid-000/vaspout.h5".

    """
    return calc_file(path) if os.path.isdir(path) else path


def read_electronic_states(static_path: str) -> ElectronicStates | None:
    """Read electronic states from the vaspout.h5 beside the static output.

    ``static_path`` is the static single-point output file or its directory.
    Returns the ElectronicStates when a vaspout.h5 is found there and carries
    the electron eigenvalues, otherwise None. A missing vaspout.h5 (only
    vasprun.xml was written) or one without eigenvalues is not an error: the
    grid point is built without F_el and a notice is printed. This lets the
    electronic term default to on while degrading gracefully where it is
    unavailable.

    """
    sdir = static_path if os.path.isdir(static_path) else os.path.dirname(static_path)
    path = os.path.join(sdir, "vaspout.h5")
    if not os.path.exists(path):
        print(f"  {static_path}: no vaspout.h5; electronic states (F_el) not stored")
        return None
    try:
        return electronic_states_from_vaspout(path)
    except (KeyError, OSError):
        print(
            f"  {static_path}: vaspout.h5 has no electron eigenvalues; F_el not stored"
        )
        return None


def dataset_has_forces(dataset: DisplacementDataset | None) -> bool:
    """Return whether a displacement dataset carries forces.

    Both dataset types are recognized: type-1 keeps the forces per displaced
    atom under "first_atoms", type-2 in a single "forces" array.

    """
    if dataset is None:
        return False
    if "first_atoms" in dataset:
        first_atoms = dataset["first_atoms"]  # type: ignore[typeddict-item]
        return bool(first_atoms) and all("forces" in d for d in first_atoms)
    return "forces" in dataset


def load_phonon_from_disp_dirs(gdir: str) -> Phonopy:
    """Return a Phonopy with forces read from the disp-* subdirectories.

    The cell, supercell / primitive matrices and displacements come from
    gdir/phonopy_disp.yaml (the relaxed cell that was displaced); the forces
    from its disp-* subdirectories, taken in sorted order. phonopy_disp.yaml
    may hold either a type-1 (site-symmetry-reduced, one displaced atom per
    supercell -- the ``phonopy -d`` default) or a type-2 (dense/random)
    dataset; the forces are set on the phonopy object, which embeds them into
    the dataset in its native form.

    Sorting by name is only a guess at the displacement order: it is wrong for
    unpadded names such as disp-1, disp-10, disp-2. Rather than trusting it,
    each calculator output is checked against the displaced supercell it is
    supposed to belong to, which the same file already provides. A wrong order
    that happens to keep the count would otherwise pass silently and produce
    force constants built from mismatched forces.

    """
    ph = phonopy.load(
        os.path.join(gdir, "phonopy_disp.yaml"),
        produce_fc=False,
        is_nac=False,
        log_level=0,
    )

    disp_dirs = sorted(glob.glob(os.path.join(gdir, "disp-*")))
    n_disp = len(ph.displacements)
    if len(disp_dirs) != n_disp:
        raise ValueError(
            f"{gdir}: {len(disp_dirs)} disp-* directory(ies) do not match "
            f"{n_disp} displacement(s)."
        )

    expected = ph.supercells_with_displacements
    force_sets = []
    for i, disp_dir in enumerate(disp_dirs):
        cell, _, force, _ = read_vasprun_calculation(calc_file(disp_dir))
        _check_displaced_supercell(disp_dir, cell, expected[i], i)
        force_sets.append(force)
    ph.forces = np.array(force_sets, dtype="double")
    return ph


def _check_displaced_supercell(
    disp_dir: str,
    found: PhonopyAtoms,
    expected: PhonopyAtoms,
    position: int,
    tolerance: float = 1e-3,
) -> None:
    """Raise unless a disp-* holds the displaced supercell of its position.

    The calculator output carries the structure it was run on, so the sorted
    position of a disp-* directory can be verified instead of assumed. The
    comparison is of the atomic positions under the minimum image convention,
    since an atom displaced across a cell boundary comes back wrapped.

    The tolerance is absolute and in the length unit of the cell. It has to
    separate two different displacements, which differ by at least the
    displacement distance (0.01 Angstrom in the smallest practical case), while
    tolerating the digits a calculator writes, which cost far less than that.

    """
    diff = np.array(found.scaled_positions) - np.array(expected.scaled_positions)
    diff -= np.rint(diff)
    distances = np.linalg.norm(diff @ np.array(expected.cell), axis=1)
    worst = float(distances.max())
    if worst <= tolerance:
        return
    raise ValueError(
        f"{disp_dir}: the structure is not the displaced supercell of "
        f"displacement {position + 1}; atoms differ by up to {worst:.4f} "
        f"(largest at atom {int(np.argmax(distances)) + 1}).\n"
        f"The disp-* directories are taken in sorted order, so unpadded names "
        f"such as disp-1, disp-10, disp-2 put the force sets in the wrong "
        f"order. Zero-pad them, or pass a phonopy_params.yaml whose forces "
        f"phonopy-init has already collected."
    )


def load_phonon(path: str) -> Phonopy:
    """Return a Phonopy with displacements and forces for one grid point.

    ``path`` is either a directory holding phonopy_disp.yaml and the
    per-displacement disp-* subdirectories, or a phonopy.yaml-like file whose
    forces are already collected -- a phonopy_params.yaml carrying them, or a
    phonopy_disp.yaml with a FORCE_SETS beside it. The file form leaves the
    force collection to phonopy's own tools (``phonopy-init -f``), so no layout
    convention is imposed on the displaced supercells.

    """
    if os.path.isdir(path):
        return load_phonon_from_disp_dirs(path)
    # phonopy.load() searches FORCE_SETS in the current directory, not next to
    # the yaml, so point it at the neighboring one when there is any.
    force_sets = os.path.join(os.path.dirname(path), "FORCE_SETS")
    ph = phonopy.load(
        path,
        force_sets_filename=force_sets if os.path.exists(force_sets) else None,
        produce_fc=False,
        is_nac=False,
        log_level=0,
    )
    if not dataset_has_forces(ph.dataset):
        raise ValueError(
            f"{path} carries no forces, and no FORCE_SETS was found beside it. "
            f"Give a phonopy_params.yaml written by 'phonopy-init --sp -f', a "
            f"phonopy_disp.yaml with its FORCE_SETS, or a directory holding "
            f"phonopy_disp.yaml and the disp-* subdirectories."
        )
    return ph


def _check_paired_cells(
    index: int,
    static_path: str,
    static_cell: PhonopyAtoms,
    phonon_path: str,
    phonon_cell: PhonopyAtoms,
    rtol: float = 1e-5,
) -> None:
    """Raise unless the paired static and phonon entries are the same cell.

    --static and --phonon are paired by position, so a point missing from each
    list would pair the U of one lattice with the forces of another and go
    unnoticed: the lengths still match and nothing in the names says otherwise.
    Comparing the lattice-vector lengths catches that regardless of how the
    directories are named. Only the lattice is compared, because a relaxation
    moves the internal coordinates while the lattice is what the grid samples.

    """
    static_lengths = np.linalg.norm(np.array(static_cell.cell), axis=1)
    phonon_lengths = np.linalg.norm(np.array(phonon_cell.cell), axis=1)
    if np.allclose(static_lengths, phonon_lengths, rtol=rtol, atol=0.0):
        return
    raise SystemExit(
        f"Grid point {index}: the static and phonon entries are different "
        f"cells, so they are mis-paired.\n"
        f"  {static_path}: a, b, c = {np.round(static_lengths, 6).tolist()}\n"
        f"  {phonon_path}: a, b, c = {np.round(phonon_lengths, 6).tolist()}\n"
        f"--static and --phonon are paired by position; check that the two "
        f"lists enumerate the grid points in the same order."
    )


def build_calculator_grid_point(
    index: int, static_path: str, phonon_path: str, with_electronic: bool
) -> AnisoQHAGridPoint:
    """Assemble one grid point from pre-computed calculator outputs.

    The cell, supercell / primitive matrices, displacements and forces come
    from the phonon grid point; U and the optional electronic states from the
    static single point.

    """
    ph = load_phonon(phonon_path)
    dataset = ph.dataset
    assert dataset is not None

    static_cell, energy, _, _ = read_vasprun_calculation(static_path)
    _check_paired_cells(index, static_path, static_cell, phonon_path, ph.unitcell)
    electronic = read_electronic_states(static_path) if with_electronic else None

    return AnisoQHAGridPoint(
        index=index,
        cell=ph.unitcell,
        supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
        dataset=dataset,
        internal_energy=energy,
        electronic_states=electronic,
    )


def build_static_grid_point(
    index: int, static_path: str, reference: Phonopy, with_electronic: bool
) -> AnisoQHAGridPoint:
    """Assemble one grid point from the static single point alone.

    No phonon calculation is read, so the grid point carries no displacement
    dataset: the cell, U and the optional electronic states come from the
    static single point, and the supercell / primitive matrices from the
    reference. This is the dataset for a workflow whose vibrational free
    energy is computed outside and handed to run_anisotropic_qha through
    phonon_free_energies, as methods with temperature-dependent force
    constants must.

    """
    cell, energy, _, _ = read_vasprun_calculation(static_path)
    electronic = read_electronic_states(static_path) if with_electronic else None

    return AnisoQHAGridPoint(
        index=index,
        cell=cell,
        supercell_matrix=np.array(reference.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(reference.primitive_matrix, dtype="double"),
        dataset=None,
        internal_energy=energy,
        electronic_states=electronic,
    )


def resolve_static_paths(args: Namespace) -> list[tuple[int, str]]:
    """Return the (index, path) of the static single points.

    The paths are taken as given and indexed by position; a directory entry is
    resolved to the VASP output it holds. The index is a label only: the
    analysis reads the lattice parameters from each stored cell.

    """
    if not args.static:
        raise SystemExit("Error: --static is required.")
    return list(enumerate(as_calc_file(p) for p in args.static))


def resolve_phonon_paths(args: Namespace, indices: list[int]) -> list[str]:
    """Return the phonon grid-point paths, aligned with the static points.

    The paths are paired with the static points by position, so the two lists
    must have equal length. An empty list is the static-only mode and is not
    reached from here.

    """
    if len(args.phonon) != len(indices):
        raise SystemExit(
            f"{len(args.phonon)} --phonon path(s) do not match "
            f"{len(indices)} static grid point(s)."
        )
    return list(args.phonon)


def get_options() -> Namespace:
    """Parse command-line options."""
    parser = ArgumentParser(
        description=(
            "Build the anisotropic QHA intermediate dataset "
            "(aniso_qha_dataset.hdf5) from calculator outputs."
        )
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="phonopy_disp.yaml",
        help="reference phonopy_disp.yaml (calculator, free lattice DOF)",
    )
    parser.add_argument(
        "--static",
        nargs="+",
        metavar="PATH",
        help="static single-point outputs, one per grid point (shell-expanded, "
        "e.g. 'runs/*/vaspout.h5'); a directory is resolved to the VASP output "
        "it holds",
    )
    parser.add_argument(
        "--phonon",
        nargs="+",
        metavar="PATH",
        help="phonon grid points, one per static point and paired by position: "
        "either a phonopy_params.yaml carrying forces, or a directory holding "
        "phonopy_disp.yaml and disp-* subdirectories. Omit it to build from "
        "the static grid alone, for use with the phonon_free_energies "
        "argument of run_anisotropic_qha",
    )
    parser.add_argument(
        "--no-electronic",
        dest="electronic",
        action="store_false",
        help="do not store electronic states (for F_el) even when the static "
        "vaspout.h5 provides them; otherwise they are stored automatically when "
        "available",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aniso_qha_dataset.hdf5",
        help="output HDF5 file (default: aniso_qha_dataset.hdf5)",
    )
    return parser.parse_args()


def _detect_grid_shape(
    free_lengths: NDArray[np.double],
) -> tuple[int, ...] | None:
    """Return the shape of the tensor grid the cells form, or None.

    The analysis takes the main-diagonal volume path from this shape, so it is
    recorded only when the cells really are a tensor grid laid out in
    row-major order with ascending values along every axis. Anything else --
    randomly sampled cells, or grid cells gathered in another order -- gives
    None, and the analysis then declines to guess a diagonal.

    Parameters
    ----------
    free_lengths : ndarray
        Lattice-vector lengths of the free DOF of every grid point, in the
        order the points are stored. shape=(n_points, n_free_dof).

    """
    n_points, ndof = free_lengths.shape
    rounded = np.round(free_lengths, 6)

    # A tensor grid samples n_j distinct values along axis j and visits every
    # combination of them exactly once.
    counts = [len(np.unique(rounded[:, j])) for j in range(ndof)]
    if int(np.prod(counts)) != n_points:
        return None

    grid = rounded.reshape(*counts, ndof)
    for j in range(ndof):
        # Row-major order means the j-th length depends on the j-th index
        # alone, so every slice taken at a fixed j-th index is one value.
        slices = np.moveaxis(grid[..., j], j, 0).reshape(counts[j], -1)
        if not np.allclose(slices, slices[:, :1]):
            return None
        # Ascending, so that the diagonal runs from the smallest cell to the
        # largest and the volume path it forms is monotonic.
        if not (np.diff(slices[:, 0]) > 0).all():
            return None

    return tuple(counts)


def run() -> None:
    """Run the phonopy-anisotropic-qha-dataset command."""
    args = get_options()

    reference = phonopy.load(args.filename, produce_fc=False, is_nac=False, log_level=0)
    calculator = reference.calculator or "vasp"
    if calculator != "vasp":
        raise SystemExit(
            f"phonopy-anisotropic-qha-dataset reads the static single point from "
            f"VASP outputs only (vaspout.h5 / vasprun.xml), but the reference "
            f"'{args.filename}' specifies '{calculator}'. The static internal "
            f"energy U(a, c) and the electronic states have no reader for other "
            f"calculators yet."
        )
    length_unit = get_calculator_physical_units(calculator).length_unit
    dof = get_free_lattice_dof(reference.unitcell)

    static_paths = resolve_static_paths(args)
    indices = [idx for idx, _ in static_paths]

    points = []
    if not args.phonon:
        print(
            f"No --phonon given: building from the static grid alone, "
            f"{len(static_paths)} grid point(s), with no displacements or "
            f"forces. Such a dataset is for use with the phonon_free_energies "
            f"argument of run_anisotropic_qha."
        )
        for idx, static_path in static_paths:
            points.append(
                build_static_grid_point(idx, static_path, reference, args.electronic)
            )
    else:
        phonon_paths = resolve_phonon_paths(args, indices)
        print(f"Reading pre-computed forces for {len(phonon_paths)} grid point(s)")
        for (idx, static_path), phonon_path in zip(
            static_paths, phonon_paths, strict=True
        ):
            points.append(
                build_calculator_grid_point(
                    idx, static_path, phonon_path, args.electronic
                )
            )

    for point in points:
        print(
            f"  grid {point.index:03d} U={point.internal_energy:.6f} eV "
            f"n_disp={point.n_displacements}"
        )

    free_rows = [dof.rows[label][0] for label in dof.labels]
    free_lengths = np.array(
        [np.linalg.norm(point.cell.cell, axis=1)[free_rows] for point in points]
    )
    grid_shape = _detect_grid_shape(free_lengths)
    dataset = AnisoQHADataset(
        grid_points=tuple(points),
        calculator=calculator,
        length_unit=length_unit,
        free_dof=tuple(dof.labels),
        crystal_system=dof.crystal_system,
        tie_description=dof.tie_description,
        grid_shape=grid_shape,
        phonopy_version=phonopy.__version__,
    )
    if grid_shape is None:
        print("The cells do not form an ordered tensor grid; no grid shape is stored.")
    else:
        print(f"Grid shape {list(grid_shape)} recorded for the main-diagonal path.")
    write_aniso_qha_dataset(dataset, args.output)
    print(f"Wrote {len(points)} grid point(s) to {args.output}")


if __name__ == "__main__":
    run()
