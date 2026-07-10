"""Build the anisotropic QHA intermediate dataset from calculator outputs.

Gathers the scattered per-grid-point outputs into one self-contained
aniso_qha_dataset.hdf5 that the anisotropic QHA analysis reads. Two
front-ends produce the same dataset:

- --from-dft: force sets come from the displaced supercells in
  phonon-grid/grid-NNN/disp-*, the static U(a, c) from the static single
  points in static-grid/grid-NNN.
- --from-mlp POLYMLP: force sets are evaluated on the fly from a
  machine-learning potential for each grid point; U(a, c) still comes from
  the DFT static single points in static-grid/grid-NNN.

In both cases the grid-point cell is the relaxed cell of the static single
point, so internal-coordinate relaxation is honored. The displacements and
forces are stored raw; the analysis recomputes the force constants.

"""

from __future__ import annotations

import glob
import os
from argparse import ArgumentParser, Namespace

import numpy as np

import phonopy
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasprun_calculation
from phonopy.physical_units import get_calculator_physical_units
from phonopy.qha.anisotropic_dataset import (
    AnisoQHADataset,
    AnisoQHAGridPoint,
    write_aniso_qha_dataset,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.qha.lattice_sampling import get_free_lattice_dof


def discover_grid_indices(directory: str) -> list[int]:
    """Return the sorted grid indices from the grid-NNN subdirectories.

    The grid points are the "grid-NNN" directories of the given grid; the
    integer NNN is the index. The lattice lengths are read from each cell
    later, so no separate index file is needed.

    """
    indices = []
    for path in glob.glob(os.path.join(directory, "grid-*")):
        name = os.path.basename(path)
        suffix = name[len("grid-") :]
        if os.path.isdir(path) and suffix.isdigit():
            indices.append(int(suffix))
    if not indices:
        raise FileNotFoundError(f"No grid-NNN directories found in {directory}.")
    return sorted(indices)


def calc_file(dirpath: str) -> str:
    """Return dirpath/vaspout.h5 if present, else dirpath/vasprun.xml.

    vaspout.h5 keeps full numerical precision, so prefer it for forces and
    energies when available.

    """
    h5 = os.path.join(dirpath, "vaspout.h5")
    return h5 if os.path.exists(h5) else os.path.join(dirpath, "vasprun.xml")


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


def build_dft_grid_point(
    idx: int, static_grid: str, phonon_grid: str, with_electronic: bool
) -> AnisoQHAGridPoint:
    """Assemble one grid point from DFT static and phonon outputs.

    The cell, supercell / primitive matrices and displacements come from
    phonon-grid/grid-NNN/phonopy_disp.yaml (the relaxed cell that was
    displaced); the forces from its disp-* subdirectories in displacement
    order; U and the optional electronic states from static-grid/grid-NNN.

    """
    gdir = os.path.join(phonon_grid, f"grid-{idx:03d}")
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
    displacements = np.array(ph.displacements, dtype="double")
    if forces.shape != displacements.shape:
        raise ValueError(
            f"grid {idx:03d}: {len(forces)} force set(s) do not match "
            f"{len(displacements)} displacement(s)."
        )

    sdir = os.path.join(static_grid, f"grid-{idx:03d}")
    _, energy, _, _ = read_vasprun_calculation(calc_file(sdir))
    electronic = (
        electronic_states_from_vaspout(os.path.join(sdir, "vaspout.h5"))
        if with_electronic
        else None
    )

    return AnisoQHAGridPoint(
        index=idx,
        cell=ph.unitcell,
        supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
        displacements=displacements,
        forces=forces,
        internal_energy=energy,
        electronic_states=electronic,
    )


def build_mlp_grid_point(
    idx: int,
    static_grid: str,
    reference: Phonopy,
    mlp_file: str,
    distance: float,
    snapshots: int,
    seed: int | None,
    with_electronic: bool,
) -> AnisoQHAGridPoint:
    """Assemble one grid point with MLP forces and DFT static U.

    The relaxed cell and U come from static-grid/grid-NNN; the supercell /
    primitive matrices from the reference cell. Random displacements are
    generated and the forces evaluated by the MLP, then stored raw so the
    analysis is blind to their MLP origin.

    """
    sdir = os.path.join(static_grid, f"grid-{idx:03d}")
    cell, energy, _, _ = read_vasprun_calculation(calc_file(sdir))

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

    electronic = (
        electronic_states_from_vaspout(os.path.join(sdir, "vaspout.h5"))
        if with_electronic
        else None
    )

    return AnisoQHAGridPoint(
        index=idx,
        cell=cell,
        supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
        displacements=np.array(ph.displacements, dtype="double"),
        forces=np.array(ph.forces, dtype="double"),
        internal_energy=energy,
        electronic_states=electronic,
    )


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-dft",
        action="store_true",
        help="gather DFT forces from the phonon grid",
    )
    mode.add_argument(
        "--from-mlp",
        metavar="POLYMLP",
        help="evaluate forces from this machine-learning potential file",
    )
    parser.add_argument(
        "--static-grid",
        default="static-grid",
        help="directory of static single points (default: static-grid)",
    )
    parser.add_argument(
        "--phonon-grid",
        default="phonon-grid",
        help="directory of displaced supercells for --from-dft (default: phonon-grid)",
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
        "--electronic",
        action="store_true",
        help="include electronic states from static-grid vaspout.h5 (for F_el)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aniso_qha_dataset.hdf5",
        help="output HDF5 file (default: aniso_qha_dataset.hdf5)",
    )
    return parser.parse_args()


def run() -> None:
    """Run the phonopy-aniso-qha-dataset command."""
    args = get_options()

    reference = phonopy.load(args.filename, produce_fc=False, is_nac=False, log_level=0)
    calculator = reference.calculator or "vasp"
    length_unit = get_calculator_physical_units(calculator).length_unit
    dof = get_free_lattice_dof(reference.unitcell)

    indices = discover_grid_indices(args.static_grid)

    points = []
    for idx in indices:
        if args.from_mlp:
            point = build_mlp_grid_point(
                idx,
                args.static_grid,
                reference,
                args.from_mlp,
                args.distance,
                args.snapshots,
                args.seed,
                args.electronic,
            )
        else:
            point = build_dft_grid_point(
                idx, args.static_grid, args.phonon_grid, args.electronic
            )
        points.append(point)
        print(
            f"  grid {idx:03d} U={point.internal_energy:.6f} eV "
            f"n_disp={len(point.displacements)}"
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
