# SPDX-License-Identifier: BSD-3-Clause
"""Build the anisotropic QHA intermediate dataset from calculator outputs.

Gathers the scattered per-grid-point outputs into one self-contained
aniso_qha_dataset.hdf5 that the anisotropic QHA analysis reads. Two modes
produce the same dataset:

- default: force sets come from the pre-computed displaced supercells of the
  phonon grid, the static U(a, c) from the static single points of the static
  grid.
- --from-mlp POLYMLP: force sets are evaluated on the fly from a
  machine-learning potential for each grid point; U(a, c) still comes from
  the static single points of the static grid.

The grid points are located in either of two ways. Without path options the
conventional layout is assumed: grid-NNN subdirectories under static-grid and
phonon-grid, paired by the index NNN. With --static / --phonon the paths are
given explicitly, typically expanded by the shell, and the two lists are
paired by position; then no naming convention applies at all.

A phonon grid point is either a directory holding phonopy_disp.yaml and the
per-displacement disp-* subdirectories, or a single phonopy_params.yaml-like
file that already carries the forces (as written by ``phonopy-init -f``). The
latter keeps the force collection in phonopy's own tools, so this command
need not know how the displaced supercells were laid out.

In both modes the grid-point cell is the relaxed cell of the static single
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

import phonopy
from phonopy import Phonopy
from phonopy.harmonic.displacement import DisplacementDataset
from phonopy.interface.vasp import read_vasprun_calculation
from phonopy.physical_units import get_calculator_physical_units
from phonopy.qha.anisotropic_dataset import (
    AnisoQHADataset,
    AnisoQHAGridPoint,
    write_aniso_qha_dataset,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.qha.lattice_sampling import get_free_lattice_dof

DEFAULT_STATIC_GRID = "static-grid"
DEFAULT_PHONON_GRID = "phonon-grid"


def discover_grid_dirs(directory: str) -> list[tuple[int, str]]:
    """Return the (index, path) of the grid-NNN subdirectories, index-sorted.

    The grid points are the "grid-NNN" directories of the given grid; the
    integer NNN is the index. The path is returned as found rather than
    rebuilt from the index, so any zero padding is accepted. The lattice
    lengths are read from each cell later, so no separate index file is
    needed.

    """
    found = []
    for path in glob.glob(os.path.join(directory, "grid-*")):
        suffix = os.path.basename(path)[len("grid-") :]
        if os.path.isdir(path) and suffix.isdigit():
            found.append((int(suffix), path))
    if not found:
        raise FileNotFoundError(f"No grid-NNN directories found in {directory}.")
    return sorted(found)


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


def electronic_states_from_vaspout(path: str) -> ElectronicStates:
    """Build ElectronicStates from a VASP vaspout.h5 (numerically exact).

    Reads the eigenvalues, symmetry k-point weights, and electron count of the
    static single point. vaspout.h5 avoids the digit truncation of vasprun.xml.

    """
    import h5py  # type: ignore[import-untyped]

    with h5py.File(path, "r") as f:
        g = f["results/electron_eigenvalues"]
        return ElectronicStates(
            eigenvalues=g["eigenvalues"][:],  # (spin, kpoints, bands)
            weights=g["kpoints_symmetry_weight"][:],
            n_electrons=float(g["nelectrons"][()]),
        )


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
    from its disp-* subdirectories in sorted order, which must be the
    displacement order. phonopy_disp.yaml may hold either a type-1
    (site-symmetry-reduced, one displaced atom per supercell -- the
    ``phonopy -d`` default) or a type-2 (dense/random) dataset; the forces are
    set on the phonopy object, which embeds them into the dataset in its
    native form.

    """
    ph = phonopy.load(
        os.path.join(gdir, "phonopy_disp.yaml"),
        produce_fc=False,
        is_nac=False,
        log_level=0,
    )

    force_sets = []
    for disp_dir in sorted(glob.glob(os.path.join(gdir, "disp-*"))):
        _, _, force, _ = read_vasprun_calculation(calc_file(disp_dir))
        force_sets.append(force)
    forces = np.array(force_sets, dtype="double")
    n_disp = len(ph.displacements)
    if len(forces) != n_disp:
        raise ValueError(
            f"{gdir}: {len(forces)} force set(s) do not match {n_disp} displacement(s)."
        )
    ph.forces = forces
    return ph


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

    _, energy, _, _ = read_vasprun_calculation(static_path)
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


def build_mlp_grid_point(
    index: int,
    static_path: str,
    reference: Phonopy,
    mlp_file: str,
    distance: float,
    snapshots: int,
    seed: int | None,
    with_electronic: bool,
) -> AnisoQHAGridPoint:
    """Assemble one grid point with MLP forces and a static U.

    The relaxed cell and U come from the static single point; the supercell /
    primitive matrices from the reference cell. Random displacements are
    generated and the forces evaluated by the MLP, then stored raw so the
    analysis is blind to their MLP origin.

    """
    cell, energy, _, _ = read_vasprun_calculation(static_path)

    ph = Phonopy(
        cell,
        supercell_matrix=reference.supercell_matrix,
        primitive_matrix=reference.primitive_matrix,
        log_level=0,
    )
    ph.load_mlp(mlp_file)
    ph.generate_displacements(
        distance=distance, number_of_snapshots=snapshots, random_seed=seed
    )
    ph.evaluate_mlp()
    dataset = ph.dataset
    assert dataset is not None

    electronic = read_electronic_states(static_path) if with_electronic else None

    return AnisoQHAGridPoint(
        index=index,
        cell=cell,
        supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
        dataset=dataset,
        internal_energy=energy,
        electronic_states=electronic,
    )


def resolve_static_paths(args: Namespace) -> list[tuple[int, str]]:
    """Return the (index, path) of the static single points.

    With --static the paths are taken as given and indexed by position; a
    directory entry is resolved to the VASP output it holds. Otherwise the
    conventional grid-NNN layout of --static-grid is discovered and the index
    is the one in the directory name.

    """
    if args.static:
        if args.static_grid is not None:
            raise SystemExit("Give either --static or --static-grid, not both.")
        return list(enumerate(as_calc_file(p) for p in args.static))
    directory = args.static_grid or DEFAULT_STATIC_GRID
    return [(idx, calc_file(d)) for idx, d in discover_grid_dirs(directory)]


def resolve_phonon_paths(args: Namespace, indices: list[int]) -> list[str]:
    """Return the phonon grid-point paths, aligned with the static points.

    With --phonon the paths are paired with the static points by position, so
    the two lists must have equal length. Otherwise the conventional grid-NNN
    layout of --phonon-grid is discovered and paired by index, and every
    static index must have a match.

    """
    if args.phonon:
        if args.phonon_grid is not None:
            raise SystemExit("Give either --phonon or --phonon-grid, not both.")
        if len(args.phonon) != len(indices):
            raise SystemExit(
                f"{len(args.phonon)} --phonon path(s) do not match "
                f"{len(indices)} static grid point(s)."
            )
        return list(args.phonon)

    directory = args.phonon_grid or DEFAULT_PHONON_GRID
    dirs = dict(discover_grid_dirs(directory))
    missing = [idx for idx in indices if idx not in dirs]
    if missing:
        listed = ", ".join(f"grid-{idx:03d}" for idx in missing)
        raise SystemExit(f"No phonon grid point in {directory} for: {listed}.")
    return [dirs[idx] for idx in indices]


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
        help="reference phonopy_disp.yaml (supercell / primitive matrices, "
        "free lattice DOF)",
    )
    parser.add_argument(
        "--from-mlp",
        metavar="POLYMLP",
        help="evaluate forces on the fly from this machine-learning potential "
        "file, instead of reading pre-computed forces from the phonon grid",
    )
    parser.add_argument(
        "--static",
        nargs="+",
        metavar="PATH",
        help="static single-point outputs, one per grid point (shell-expanded, "
        "e.g. 'runs/*/vaspout.h5'); a directory is resolved to the VASP output "
        "it holds. Replaces --static-grid and imposes no naming convention",
    )
    parser.add_argument(
        "--phonon",
        nargs="+",
        metavar="PATH",
        help="phonon grid points, one per static point and paired by position: "
        "either a phonopy_params.yaml carrying forces, or a directory holding "
        "phonopy_disp.yaml and disp-* subdirectories. Replaces --phonon-grid",
    )
    parser.add_argument(
        "--static-grid",
        default=None,
        help="directory of static single points in the conventional grid-NNN "
        f"layout (default: {DEFAULT_STATIC_GRID})",
    )
    parser.add_argument(
        "--phonon-grid",
        default=None,
        help="directory of pre-computed displaced supercells in the "
        "conventional grid-NNN layout, used when --from-mlp is not given "
        f"(default: {DEFAULT_PHONON_GRID})",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.03,
        help="MLP displacement distance (default: 0.03)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=20,
        help="number of MLP random-displacement snapshots (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for MLP displacements"
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

    if args.from_mlp:
        phonon_paths: list[str | None] = [None] * len(static_paths)
        print(f"Evaluating forces from MLP {args.from_mlp}")
    else:
        phonon_paths = list(resolve_phonon_paths(args, indices))
        print(f"Reading pre-computed forces for {len(phonon_paths)} grid point(s)")

    points = []
    for (idx, static_path), phonon_path in zip(static_paths, phonon_paths, strict=True):
        if args.from_mlp:
            point = build_mlp_grid_point(
                idx,
                static_path,
                reference,
                args.from_mlp,
                args.distance,
                args.snapshots,
                args.seed,
                args.electronic,
            )
        else:
            assert phonon_path is not None
            point = build_calculator_grid_point(
                idx, static_path, phonon_path, args.electronic
            )
        points.append(point)
        print(
            f"  grid {idx:03d} U={point.internal_energy:.6f} eV "
            f"n_disp={point.n_displacements}"
        )

    dataset = AnisoQHADataset(
        grid_points=tuple(points),
        calculator=calculator,
        length_unit=length_unit,
        free_dof=tuple(dof.labels),
        crystal_system=dof.crystal_system,
        tie_description=dof.tie_description,
        phonopy_version=phonopy.__version__,
    )
    write_aniso_qha_dataset(dataset, args.output)
    print(f"Wrote {len(points)} grid point(s) to {args.output}")


if __name__ == "__main__":
    run()
