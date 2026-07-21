# SPDX-License-Identifier: BSD-3-Clause
"""Intermediate dataset for the anisotropic QHA workflow.

A single self-contained HDF5 file gathers the per-grid-point inputs the
analysis needs: the relaxed cell, the phonopy displacement-force dataset (raw
displacements and forces), the static internal energy U, and optional
electronic states. Storing raw forces rather than force constants keeps the
file a method-independent archive.

The displacement dataset is stored in its native phonopy form -- type-1 (one
displaced atom per supercell, the ``phonopy -d`` default) or type-2
(dense/random) -- so the FC solver is chosen from the dataset type, never
guessed from the data.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.displacement import (
    DisplacementDataset,
    FirstAtomDisplacementWithForces,
    Type1DisplacementDataset,
    Type2DisplacementDataset,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.structure.atoms import PhonopyAtoms

if TYPE_CHECKING:
    import h5py  # type: ignore[import-untyped]

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
    dataset : DisplacementDataset
        Phonopy displacement-force dataset with forces embedded, in either the
        type-1 or type-2 format (see :attr:`phonopy.Phonopy.dataset`).
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
    dataset: DisplacementDataset
    internal_energy: float
    electronic_states: ElectronicStates | None = None

    @property
    def n_displacements(self) -> int:
        """Return the number of displaced supercells."""
        if "first_atoms" in self.dataset:
            return len(cast(Type1DisplacementDataset, self.dataset)["first_atoms"])
        return len(cast(Type2DisplacementDataset, self.dataset)["displacements"])

    def to_phonopy(self, fc_calculator: str = "symfc") -> Phonopy:
        """Return a Phonopy with force constants from the stored dataset.

        For a type-1 dataset (one displaced atom per supercell) phonopy's
        site-symmetry solver produces the force constants and ``fc_calculator``
        is ignored, since that minimal data requires it. For a type-2
        (dense/random) dataset the given ``fc_calculator`` (symfc by default)
        is used. The origin of the forces (DFT or MLP) does not matter here.

        """
        from phonopy import Phonopy

        phonon = Phonopy(
            self.cell,
            supercell_matrix=self.supercell_matrix,
            primitive_matrix=self.primitive_matrix,
            log_level=0,
        )
        phonon.dataset = self.dataset
        if "first_atoms" in self.dataset:
            phonon.produce_force_constants()
        else:
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
    matrices, the displacement-force dataset (tagged type-1 or type-2), and
    (optionally) the electronic states. Global metadata is stored in root
    attributes. Displacements, forces and eigenvalues are gzip-compressed.

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


def _write_grid_point(grid: h5py.Group, point: AnisoQHAGridPoint) -> None:
    """Write one grid point into a subgroup "NNN" of the grid group."""
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
    _write_dataset(g, point.dataset)
    if point.electronic_states is not None:
        _write_electronic_states(g, point.electronic_states)


def _write_dataset(g: h5py.Group, dataset: DisplacementDataset) -> None:
    """Write the displacement-force dataset, tagged by its type.

    Type-1 stores the displaced-atom index, its displacement and the supercell
    forces per entry; type-2 stores the full displacement and force arrays.
    The "displacement_type" attribute selects the layout on read.

    """
    if "first_atoms" in dataset:
        first_atoms = cast(Type1DisplacementDataset, dataset)["first_atoms"]
        if any("forces" not in entry for entry in first_atoms):
            raise ValueError("Every type-1 displacement must carry forces.")
        g.attrs["displacement_type"] = "type1"
        g.create_dataset(
            "displaced_atoms",
            data=np.array([entry["number"] for entry in first_atoms], dtype="int64"),
        )
        g.create_dataset(
            "displacements",
            data=np.array(
                [entry["displacement"] for entry in first_atoms], dtype="double"
            ),
            compression="gzip",
        )
        g.create_dataset(
            "forces",
            data=np.array([entry["forces"] for entry in first_atoms], dtype="double"),
            compression="gzip",
        )
    else:
        type2 = cast(Type2DisplacementDataset, dataset)
        if "forces" not in type2:
            raise ValueError("The type-2 dataset must carry forces.")
        displacements = np.array(type2["displacements"], dtype="double")
        forces = np.array(type2["forces"], dtype="double")
        if displacements.shape != forces.shape:
            raise ValueError(
                "displacements and forces must have the same shape, got "
                f"{displacements.shape} and {forces.shape}."
            )
        g.attrs["displacement_type"] = "type2"
        g.create_dataset("displacements", data=displacements, compression="gzip")
        g.create_dataset("forces", data=forces, compression="gzip")


def _write_electronic_states(
    g: h5py.Group, electronic_states: ElectronicStates
) -> None:
    """Write electronic states into an "electronic_states" subgroup of g."""
    eg = g.create_group("electronic_states")
    eg.create_dataset(
        "eigenvalues",
        data=np.array(electronic_states.eigenvalues, dtype="double"),
        compression="gzip",
    )
    eg.create_dataset("weights", data=electronic_states.weights)
    eg.create_dataset("n_electrons", data=float(electronic_states.n_electrons))
    if electronic_states.spin_degeneracy is not None:
        eg.create_dataset(
            "spin_degeneracy", data=int(electronic_states.spin_degeneracy)
        )
    if electronic_states.fermi_energy is not None:
        eg.create_dataset("fermi_energy", data=float(electronic_states.fermi_energy))


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
        points = tuple(
            _read_grid_point(grid[key]) for key in sorted(grid.keys(), key=int)
        )

    return AnisoQHADataset(
        grid_points=points,
        calculator=calculator,
        length_unit=length_unit,
        free_dof=free_dof,
        crystal_system=crystal_system,
        tie_description=tie_description,
        phonopy_version=phonopy_version,
    )


def _read_grid_point(g: h5py.Group) -> AnisoQHAGridPoint:
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
        dataset=_read_dataset(g),
        internal_energy=float(g.attrs["internal_energy"]),
        electronic_states=electronic_states,
    )


def _read_dataset(g: h5py.Group) -> DisplacementDataset:
    """Read the displacement-force dataset written by _write_dataset."""
    if str(g.attrs["displacement_type"]) == "type1":
        forces = g["forces"][:]
        first_atoms: list[FirstAtomDisplacementWithForces] = [
            {
                "number": int(number),
                "displacement": np.array(displacement, dtype="double"),
                "forces": np.array(force, dtype="double"),
            }
            for number, displacement, force in zip(
                g["displaced_atoms"][:], g["displacements"][:], forces, strict=True
            )
        ]
        return {"natom": int(forces.shape[1]), "first_atoms": first_atoms}
    return {
        "displacements": np.array(g["displacements"][:], dtype="double"),
        "forces": np.array(g["forces"][:], dtype="double"),
    }


def _read_electronic_states(eg: h5py.Group) -> ElectronicStates:
    """Read electronic states from an "electronic_states" subgroup."""
    return ElectronicStates(
        eigenvalues=eg["eigenvalues"][:],
        weights=eg["weights"][:],
        n_electrons=float(eg["n_electrons"][()]),
        spin_degeneracy=(
            int(eg["spin_degeneracy"][()]) if "spin_degeneracy" in eg else None
        ),
        fermi_energy=(float(eg["fermi_energy"][()]) if "fermi_energy" in eg else None),
    )
