"""Intermediate dataset for the anisotropic QHA workflow.

Gathers the per-grid-point results -- the relaxed unit cell, the
random-displacement supercell displacements and forces, the static internal
energy U, and optional electronic states -- that the anisotropic QHA analysis
consumes, into a single self-contained HDF5 file.

Displacements and forces are stored raw (not as force constants), so the force
constants are recomputed at analysis time. This keeps the file independent of
the force-constant method (symfc options, cutoff, sum rules can be revisited)
and lets it serve as a self-contained archive once the calculator scratch is
discarded. Whether the forces came from DFT or from a machine-learning
potential is not recorded: the analysis reads (displacements, forces)
uniformly, blind to their origin.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from phonopy.qha.electron import ElectronicStates
from phonopy.structure.atoms import PhonopyAtoms

if TYPE_CHECKING:
    from phonopy import Phonopy


@dataclass(frozen=True)
class AnisoQHAGridPoint:
    """One grid point of the anisotropic QHA dataset.

    Attributes
    ----------
    index : int
        Grid-point index (need not be contiguous across a dataset).
    cell : PhonopyAtoms
        Relaxed unit cell at this grid point.
    supercell_matrix : ndarray
        Supercell matrix, shape (3, 3), dtype int64.
    primitive_matrix : ndarray
        Primitive matrix, shape (3, 3), dtype double.
    displacements : ndarray
        Supercell displacements, shape (n_disp, n_satom, 3), dtype double.
    forces : ndarray
        Supercell forces, same shape as displacements, dtype double.
    internal_energy : float
        Static internal energy U of the unit cell (eV).
    electronic_states : ElectronicStates, optional
        Electronic states of the static single point, for F_el. None when the
        electronic free energy is not used.

    """

    index: int
    cell: PhonopyAtoms
    supercell_matrix: NDArray[np.int64]
    primitive_matrix: NDArray[np.double]
    displacements: NDArray[np.double]
    forces: NDArray[np.double]
    internal_energy: float
    electronic_states: ElectronicStates | None = None

    def to_phonopy(self, fc_calculator: str = "symfc") -> Phonopy:
        """Return a Phonopy with force constants from the stored disp/forces.

        The raw displacements and forces are set as a type-2 dataset and the
        force constants are produced with the given calculator (symfc by
        default, needed for random displacements). The origin of the forces
        (DFT or MLP) does not matter here.

        """
        from phonopy import Phonopy

        phonon = Phonopy(
            self.cell,
            supercell_matrix=self.supercell_matrix,
            primitive_matrix=self.primitive_matrix,
            log_level=0,
        )
        phonon.dataset = {
            "displacements": self.displacements,
            "forces": self.forces,
        }
        # fc_calculator is a user string; phonopy validates it at runtime.
        phonon.produce_force_constants(fc_calculator=fc_calculator)  # type: ignore[arg-type]
        return phonon


@dataclass(frozen=True)
class AnisoQHADataset:
    """Self-contained dataset feeding the anisotropic QHA analysis.

    Attributes
    ----------
    grid_points : tuple of AnisoQHAGridPoint
        The grid points, in ascending index order after a read.
    calculator : str
        Calculator name (e.g. "vasp").
    length_unit : str
        Native length unit of the cells (e.g. "angstrom").
    free_dof : tuple of str
        Free lattice DOF labels among "a", "b", "c".
    crystal_system : str
        Crystal system of the reference cell.
    tie_description : str
        Human-readable tie relation of the free DOF (e.g. "b = a"), or "".
    phonopy_version : str, optional
        Phonopy version that wrote the dataset.

    """

    grid_points: tuple[AnisoQHAGridPoint, ...]
    calculator: str = "vasp"
    length_unit: str = "angstrom"
    free_dof: tuple[str, ...] = ()
    crystal_system: str = ""
    tie_description: str = ""
    phonopy_version: str | None = None


def write_aniso_qha_dataset(
    dataset: AnisoQHADataset,
    filename: str | os.PathLike = "aniso_qha_dataset.hdf5",
) -> None:
    """Write an anisotropic QHA dataset to an HDF5 file.

    One group "grid/NNN" per grid point holds the cell, supercell / primitive
    matrices, raw displacements and forces, and (optionally) the electronic
    states. Global metadata is stored in root attributes. Displacements,
    forces and eigenvalues are gzip-compressed.

    Parameters
    ----------
    dataset : AnisoQHADataset
        Dataset to write.
    filename : str or os.PathLike, optional
        Output HDF5 file name.

    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        if dataset.phonopy_version is not None:
            w.attrs["phonopy_version"] = dataset.phonopy_version
        w.attrs["calculator"] = dataset.calculator
        w.attrs["length_unit"] = dataset.length_unit
        w.attrs["free_dof"] = " ".join(dataset.free_dof)
        w.attrs["crystal_system"] = dataset.crystal_system
        w.attrs["tie_description"] = dataset.tie_description
        w.attrs["n_grid_points"] = len(dataset.grid_points)
        grid = w.create_group("grid")
        for point in dataset.grid_points:
            _write_grid_point(grid, point)


def _write_grid_point(grid, point: AnisoQHAGridPoint) -> None:
    """Write one grid point into a subgroup "NNN" of the grid group."""
    if point.displacements.shape != point.forces.shape:
        raise ValueError(
            "displacements and forces must have the same shape, got "
            f"{point.displacements.shape} and {point.forces.shape}."
        )
    g = grid.create_group(f"{point.index:03d}")
    g.attrs["index"] = point.index
    g.attrs["internal_energy"] = float(point.internal_energy)
    cell = point.cell
    g.create_dataset("lattice", data=np.array(cell.cell, dtype="double"))
    g.create_dataset(
        "scaled_positions", data=np.array(cell.scaled_positions, dtype="double")
    )
    g.create_dataset("numbers", data=np.array(cell.numbers, dtype="int64"))
    g.create_dataset("masses", data=np.array(cell.masses, dtype="double"))
    g.create_dataset(
        "lattice_lengths", data=np.linalg.norm(np.array(cell.cell), axis=1)
    )
    g.create_dataset(
        "supercell_matrix", data=np.array(point.supercell_matrix, dtype="int64")
    )
    g.create_dataset(
        "primitive_matrix", data=np.array(point.primitive_matrix, dtype="double")
    )
    g.create_dataset(
        "displacements",
        data=np.array(point.displacements, dtype="double"),
        compression="gzip",
    )
    g.create_dataset(
        "forces", data=np.array(point.forces, dtype="double"), compression="gzip"
    )
    if point.electronic_states is not None:
        _write_electronic_states(g, point.electronic_states)


def _write_electronic_states(g, electronic_states: ElectronicStates) -> None:
    """Write electronic states into an "electronic_states" subgroup of g."""
    eg = g.create_group("electronic_states")
    eg.create_dataset(
        "eigenvalues",
        data=np.array(electronic_states.eigenvalues, dtype="double"),
        compression="gzip",
    )
    eg.create_dataset("weights", data=electronic_states.weights)
    eg.create_dataset("n_electrons", data=float(electronic_states.n_electrons))


def read_aniso_qha_dataset(
    filename: str | os.PathLike = "aniso_qha_dataset.hdf5",
) -> AnisoQHADataset:
    """Read an anisotropic QHA dataset written by write_aniso_qha_dataset.

    Grid points are returned in ascending index order.

    Parameters
    ----------
    filename : str or os.PathLike, optional
        Input HDF5 file name.

    Returns
    -------
    AnisoQHADataset

    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    with h5py.File(filename, "r") as f:
        calculator = str(f.attrs.get("calculator", "vasp"))
        length_unit = str(f.attrs.get("length_unit", "angstrom"))
        free_dof_attr = str(f.attrs.get("free_dof", ""))
        free_dof = tuple(free_dof_attr.split())
        crystal_system = str(f.attrs.get("crystal_system", ""))
        tie_description = str(f.attrs.get("tie_description", ""))
        version = f.attrs.get("phonopy_version")
        phonopy_version = None if version is None else str(version)
        grid = f["grid"]
        points = tuple(_read_grid_point(grid[key]) for key in sorted(grid.keys()))

    return AnisoQHADataset(
        grid_points=points,
        calculator=calculator,
        length_unit=length_unit,
        free_dof=free_dof,
        crystal_system=crystal_system,
        tie_description=tie_description,
        phonopy_version=phonopy_version,
    )


def _read_grid_point(g) -> AnisoQHAGridPoint:
    """Read one grid point from a subgroup of the grid group."""
    cell = PhonopyAtoms(
        numbers=g["numbers"][:],
        cell=g["lattice"][:],
        scaled_positions=g["scaled_positions"][:],
        masses=g["masses"][:],
    )
    electronic_states = (
        _read_electronic_states(g["electronic_states"])
        if "electronic_states" in g
        else None
    )
    return AnisoQHAGridPoint(
        index=int(g.attrs["index"]),
        cell=cell,
        supercell_matrix=np.array(g["supercell_matrix"][:], dtype="int64"),
        primitive_matrix=np.array(g["primitive_matrix"][:], dtype="double"),
        displacements=np.array(g["displacements"][:], dtype="double"),
        forces=np.array(g["forces"][:], dtype="double"),
        internal_energy=float(g.attrs["internal_energy"]),
        electronic_states=electronic_states,
    )


def _read_electronic_states(eg) -> ElectronicStates:
    """Read electronic states from an "electronic_states" subgroup."""
    return ElectronicStates(
        eigenvalues=eg["eigenvalues"][:],
        weights=eg["weights"][:],
        n_electrons=float(eg["n_electrons"][()]),
    )
