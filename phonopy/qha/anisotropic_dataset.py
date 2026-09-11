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

A grid point may also carry no displacement dataset at all. Such a file is
built from the static grid alone and holds the cells, U and the electronic
states; the vibrational free energy is then computed outside and passed to
run_anisotropic_qha through phonon_free_energies, which is how methods with
temperature-dependent force constants (SSCHA, TDEP) enter the workflow.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
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
from phonopy.qha.electron_states import ElectronicStates
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
    dataset : DisplacementDataset, optional
        Phonopy displacement-force dataset with forces embedded, in either the
        type-1 or type-2 format (see :attr:`phonopy.Phonopy.dataset`). None
        when the grid point carries no phonon calculation, which is the case
        when the free energies are computed outside and handed to
        run_anisotropic_qha through phonon_free_energies.
    internal_energy : float
        Static internal energy U of the unit cell (eV).
    electronic_states : ElectronicStates, optional
        Electronic states of the static single point, for F_el. None when the
        electronic free energy is not used. Its ``kpoints`` and ``mesh`` are
        stored when both are present, which is what lets a reader integrate
        F_el by the linear tetrahedron rather than by the k-point sum; its
        ``cell`` is not, being this grid point's own, and is restored on read.

    """

    index: int
    cell: PhonopyAtoms
    supercell_matrix: NDArray[np.int64]
    primitive_matrix: NDArray[np.double]
    dataset: DisplacementDataset | None
    internal_energy: float
    electronic_states: ElectronicStates | None = None

    @property
    def n_displacements(self) -> int:
        """Return the number of displaced supercells, 0 without a dataset."""
        if self.dataset is None:
            return 0
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

        Raises
        ------
        ValueError
            When the grid point carries no displacement dataset.

        """
        from phonopy import Phonopy

        if self.dataset is None:
            raise ValueError(
                f"Grid point {self.index} carries no displacement dataset, so "
                "force constants cannot be produced. Such a dataset is built "
                "from the static grid alone, for use with the "
                "phonon_free_energies argument of run_anisotropic_qha."
            )

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


def check_cells_are_one_crystal(
    primitive_matrices: Sequence[NDArray[np.double]],
    supercell_matrices: Sequence[NDArray[np.int64]],
    symbols: Sequence[Sequence[str]],
) -> NDArray[np.double]:
    """Check that the cells differ in their lattice lengths and nothing else.

    The free energies of the grid points are compared with one another, so
    they have to be the same crystal computed the same way: the same
    primitive cell (the volume every energy is normalized per), the same
    supercell (the reach of the force constants, hence how converged each
    free energy is) and the same atoms.

    Parameters
    ----------
    primitive_matrices : sequence of ndarray
        Primitive matrix of each cell. shape=(3, 3) each
    supercell_matrices : sequence of ndarray
        Supercell matrix of each cell. shape=(3, 3) each
    symbols : sequence of sequence of str
        Chemical symbols of each cell, in their stored order.

    Returns
    -------
    ndarray
        The primitive matrix they share. shape=(3, 3)

    Raises
    ------
    ValueError
        When any of the three differs from cell to cell.

    """
    first = np.asarray(primitive_matrices[0], dtype="double")
    for i, matrix in enumerate(primitive_matrices[1:], start=1):
        if not np.allclose(matrix, first):
            raise ValueError(
                f"Cell {i} has a primitive matrix of its own; every cell must "
                "share one, since it sets the volume the free energies are "
                "normalized per."
            )
    for i, supercell_matrix in enumerate(supercell_matrices[1:], start=1):
        if not np.array_equal(supercell_matrix, supercell_matrices[0]):
            raise ValueError(
                f"Cell {i} has a supercell matrix of its own; every cell must "
                "share one, or their free energies differ in how converged "
                "they are rather than in the lattice."
            )
    for i, cell_symbols in enumerate(symbols[1:], start=1):
        if list(cell_symbols) != list(symbols[0]):
            raise ValueError(
                f"Cell {i} holds different atoms from the first one; every "
                "cell must be the same crystal."
            )
    return first


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
    grid_shape : tuple of int, optional
        Number of sampled values along each free DOF, when the grid points
        form a tensor grid stored in row-major order. None when the sampling
        was not a tensor grid, or when the shape was not recorded. Only
        analyses that need the grid structure, such as the main-diagonal
        volume path, read it.
    phonopy_version : str, optional
        Phonopy version that wrote the dataset.

    """

    grid_points: tuple[AnisoQHAGridPoint, ...]
    calculator: str = "vasp"
    length_unit: str = "angstrom"
    free_dof: tuple[str, ...] = ()
    crystal_system: str = ""
    tie_description: str = ""
    grid_shape: tuple[int, ...] | None = None
    phonopy_version: str | None = None

    def __post_init__(self) -> None:
        """Check that the grid points can be compared with one another.

        A dataset is read from a file that a run may have written point by
        point, so what holds the points together is checked here rather
        than assumed by every reader.

        """
        if not self.grid_points:
            raise ValueError("A dataset needs at least one grid point.")
        check_cells_are_one_crystal(
            [point.primitive_matrix for point in self.grid_points],
            [point.supercell_matrix for point in self.grid_points],
            [point.cell.symbols for point in self.grid_points],
        )
        indices = [point.index for point in self.grid_points]
        if len(set(indices)) != len(indices):
            raise ValueError(
                "Two grid points share an index; the indices are what a "
                "sweep addresses its runs by."
            )
        if self.grid_shape is not None:
            n_expected = int(np.prod(self.grid_shape))
            if n_expected != len(self.grid_points):
                raise ValueError(
                    f"grid_shape {self.grid_shape} describes {n_expected} "
                    f"cells, but the dataset holds {len(self.grid_points)}."
                )


def detect_grid_shape(
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
        Lattice-vector lengths of the conventional unit cell over the free
        DOF of every grid point, in the order the points are stored.
        shape=(n_points, n_free_dof)

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


def aniso_qha_dataset_from_points(
    grid_points: Sequence[AnisoQHAGridPoint],
    calculator: str = "vasp",
    length_unit: str = "angstrom",
) -> AnisoQHADataset:
    """Return a dataset of these grid points, with its metadata derived.

    The free lattice DOF come from the symmetry of the first point's cell,
    and the grid shape from the lengths of every point; both describe the
    points themselves, so they are read off rather than passed in.

    Parameters
    ----------
    grid_points : sequence of AnisoQHAGridPoint
        The grid points, in the order they are to be stored.
    calculator : str, optional
        Calculator name recorded with the dataset. Default is "vasp".
    length_unit : str, optional
        Native length unit of the cells. Default is "angstrom".

    """
    import phonopy
    from phonopy.qha.lattice_sampling import get_free_lattice_dof

    dof = get_free_lattice_dof(grid_points[0].cell)
    free_rows = [dof.rows[label][0] for label in dof.labels]
    free_lengths = np.array(
        [np.linalg.norm(point.cell.cell, axis=1)[free_rows] for point in grid_points]
    )
    return AnisoQHADataset(
        grid_points=tuple(grid_points),
        calculator=calculator,
        length_unit=length_unit,
        free_dof=tuple(dof.labels),
        crystal_system=dof.crystal_system,
        tie_description=dof.tie_description,
        grid_shape=detect_grid_shape(free_lengths),
        phonopy_version=phonopy.__version__,
    )


def build_aniso_qha_dataset(
    phonopys: Sequence[Phonopy],
    internal_energies: Sequence[float],
    electronic_structures: Sequence[ElectronicStates] | None = None,
    indices: Sequence[int] | None = None,
    calculator: str = "vasp",
    length_unit: str = "angstrom",
) -> AnisoQHADataset:
    """Return the dataset of a lattice grid whose points are already computed.

    The counterpart of run_anisotropic_qha for building the input it reads:
    what a Phonopy carries (the cell, the matrices, the displacements and
    forces) plus the static energy of each point, gathered into one file's
    worth of dataset.

    Parameters
    ----------
    phonopys : sequence of Phonopy
        One per grid point, each holding that point's cell and, unless the
        free energies are to be computed elsewhere, its displacement-force
        dataset.
    internal_energies : sequence of float
        Static internal energy U of each grid point in eV, per primitive
        cell, which is the normalization run_anisotropic_qha and the phonon
        free energies use.
    electronic_structures : sequence of ElectronicStates, optional
        Electronic states of each point's static single point, for F_el.
    indices : sequence of int, optional
        Grid-point index of each point, which a sweep addresses its runs by.
        Defaults to 0, 1, 2, ... in the order given.
    calculator : str, optional
        Calculator name recorded with the dataset. Default is "vasp".
    length_unit : str, optional
        Native length unit of the cells. Default is "angstrom".

    """
    if len(internal_energies) != len(phonopys):
        raise ValueError(
            f"internal_energies has {len(internal_energies)} entries for "
            f"{len(phonopys)} grid points."
        )
    if indices is None:
        indices = range(len(phonopys))
    elif len(indices) != len(phonopys):
        raise ValueError(
            f"indices has {len(indices)} entries for {len(phonopys)} grid points."
        )
    if electronic_structures is not None:
        if len(electronic_structures) != len(phonopys):
            raise ValueError(
                f"electronic_structures has {len(electronic_structures)} "
                f"entries for {len(phonopys)} grid points."
            )
        _check_electronic_states_cells(phonopys, electronic_structures)

    points = [
        AnisoQHAGridPoint(
            index=int(index),
            cell=ph.unitcell,
            # Copied, so that a grid point does not share the arrays the
            # Phonopy it came from keeps using.
            supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
            primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
            dataset=ph.dataset,
            internal_energy=float(energy),
            electronic_states=(
                None if electronic_structures is None else electronic_structures[i]
            ),
        )
        for i, (index, ph, energy) in enumerate(
            zip(indices, phonopys, internal_energies, strict=True)
        )
    ]
    return aniso_qha_dataset_from_points(
        points, calculator=calculator, length_unit=length_unit
    )


def _check_electronic_states_cells(
    phonopys: Sequence[Phonopy],
    electronic_structures: Sequence[ElectronicStates],
    rtol: float = 1e-5,
) -> None:
    """Raise unless each states entry belongs to its grid point's cell.

    The two sequences are paired by position, so a point missing from one of
    them would put the states of one lattice with the forces of another and
    go unnoticed. The single point may have been run on the unit cell or on
    the primitive cell, so both volumes are accepted.

    """
    for i, (ph, states) in enumerate(zip(phonopys, electronic_structures, strict=True)):
        if states.volume is None:
            continue
        accepted = (ph.unitcell.volume, ph.primitive.volume)
        if not any(np.isclose(states.volume, v, rtol=rtol) for v in accepted):
            raise ValueError(
                f"Grid point {i}: the electronic states were computed at a "
                f"volume of {states.volume:.6f} A^3, which is neither the "
                f"unit cell's {accepted[0]:.6f} nor the primitive cell's "
                f"{accepted[1]:.6f}. The sequences are paired by position; "
                f"check that both enumerate the grid points in one order."
            )


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
        if dataset.grid_shape is not None:
            w.attrs["grid_shape"] = np.array(dataset.grid_shape, dtype="int64")
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
    if cell.magnetic_moments is not None:
        # The symmetry behind the tetrahedron grid is searched on this cell,
        # so a collinear magnetic calculation has to keep its moments here.
        g.create_dataset(
            "magnetic_moments", data=np.array(cell.magnetic_moments, dtype="double")
        )
    g.create_dataset(
        "lattice_lengths", data=np.linalg.norm(np.array(cell.cell), axis=1)
    )
    g.create_dataset(
        "supercell_matrix", data=np.array(point.supercell_matrix, dtype="int64")
    )
    g.create_dataset(
        "primitive_matrix", data=np.array(point.primitive_matrix, dtype="double")
    )
    if point.dataset is not None:
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
    # The k points and the mesh they sit on are what the tetrahedron method
    # needs; without them a reader of this file can only sum over k points.
    # ElectronicStates.cell is not written: it is the cell of the grid point
    # this subgroup belongs to, and _read_grid_point supplies it on read.
    if electronic_states.kpoints is not None and electronic_states.mesh is not None:
        eg.create_dataset(
            "kpoints",
            data=np.array(electronic_states.kpoints, dtype="double"),
            compression="gzip",
        )
        eg.create_dataset("mesh", data=np.array(electronic_states.mesh, dtype="int64"))


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
        shape_attr = f.attrs.get("grid_shape")
        grid_shape = None if shape_attr is None else tuple(int(n) for n in shape_attr)
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
        grid_shape=grid_shape,
        phonopy_version=phonopy_version,
    )


def _read_grid_point(g: h5py.Group) -> AnisoQHAGridPoint:
    """Read one grid point from a subgroup of the grid group."""
    cell = PhonopyAtoms(
        numbers=g["numbers"][:],
        cell=g["lattice"][:],
        scaled_positions=g["scaled_positions"][:],
        masses=g["masses"][:],
        magnetic_moments=(
            g["magnetic_moments"][:] if "magnetic_moments" in g else None
        ),
    )
    electronic_states = (
        _read_electronic_states(g["electronic_states"], cell)
        if "electronic_states" in g
        else None
    )
    return AnisoQHAGridPoint(
        index=int(g.attrs["index"]),
        cell=cell,
        supercell_matrix=np.array(g["supercell_matrix"][:], dtype="int64"),
        primitive_matrix=np.array(g["primitive_matrix"][:], dtype="double"),
        dataset=_read_dataset(g) if "displacement_type" in g.attrs else None,
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


def _read_electronic_states(
    eg: h5py.Group, cell: PhonopyAtoms | None = None
) -> ElectronicStates:
    """Read electronic states from an "electronic_states" subgroup.

    `kpoints`, `mesh` and `cell` are what the tetrahedron method needs and
    ElectronicStates takes them together or not at all, so the cell of the
    grid point is passed in and attached only when the file carries the other
    two. A file written before they were stored reads back as a k-point sum,
    which is what it was.

    `volume` is set from the same cell whether or not the grid is there. It is
    what says which cell these states describe, and the analysis needs that to
    put F_el on the primitive-cell normalization of the phonon free energy.

    """
    has_grid = "kpoints" in eg and "mesh" in eg
    return ElectronicStates(
        eigenvalues=eg["eigenvalues"][:],
        weights=eg["weights"][:],
        n_electrons=float(eg["n_electrons"][()]),
        volume=None if cell is None else float(cell.volume),
        spin_degeneracy=(
            int(eg["spin_degeneracy"][()]) if "spin_degeneracy" in eg else None
        ),
        fermi_energy=(float(eg["fermi_energy"][()]) if "fermi_energy" in eg else None),
        kpoints=eg["kpoints"][:] if has_grid else None,
        mesh=eg["mesh"][:] if has_grid else None,
        cell=cell if has_grid else None,
    )
