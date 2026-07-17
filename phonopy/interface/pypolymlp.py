"""Polynomial machine learning potential interface."""

# Copyright (C) 2024 Atsushi Togo
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

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from phonopy.exception import PypolymlpDevelopmentError, PypolymlpRelaxationError
from phonopy.file_IO import get_io_module_to_decompress
from phonopy.harmonic.displacement import Type2DisplacementDataset
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

try:
    from pypolymlp.mlp_dev.pypolymlp import Pypolymlp  # type: ignore[import-untyped]
except ImportError:
    Pypolymlp = Any

_DatasetT = TypeVar("_DatasetT", "PypolymlpData", "PypolymlpStructureData")


@dataclass
class PypolymlpParams:
    """Parameters for pypolymlp.

    cutoff : float, optional
        Cutoff radius. Default is 8.0.
    model_type : int, optional
        Polynomial function type. Default is 3. model_type = 1: Linear
        polynomial of polynomial invariants model_type = 2: Polynomial of
        polynomial invariants model_type = 3: Polynomial of pair invariants
                        + linear polynomial of polynomial invariants
        model_type = 4: Polynomial of pair and second-order invariants
                        + linear polynomial of polynomial invariants
    max_p : int, optional
        Order of polynomial function. Default is 2.
    gtinv_order : int, optional
        Maximum order of polynomial invariants. Default is 3.
    gtinv_maxl: Sequence[int], optional
        Maximum angular numbers of polynomial invariants. [maxl for order=2,
        maxl for order=3, ...] Default is (8, 8).
    gaussian_params1, gaussian_params2 : Sequence[float, float, int], optional
        Parameters for exp[- param1 * (r - param2)**2]. Parameters are given as
        np.linspace(p[0], p[1], p[2]), where p[0], p[1], and p[2] are given by
        gaussian_params1 and gaussian_params2. Normally it is recommended to
        modify only gaussian_params2. Default is (1.0, 1.0, 1) and (0.0, 7.0,
        10), respectively.
    atom_energies: dict[str, float], optional
        Atomic energies specified by dictionary, e.g., {'Si': -0.35864636, 'O':
        -0.95743902}, where the order is irrelevant. Default is None, which
        gives zero energies for all atoms.

    """

    cutoff: float = 8.0
    model_type: Literal[1, 2, 3, 4] = 3
    max_p: Literal[1, 2, 3] = 2
    gtinv_order: int = 3
    gtinv_maxl: tuple[int, ...] = (8, 8)
    gaussian_params1: tuple[float, float, int] = (1.0, 1.0, 1)
    gaussian_params2: tuple[float, float, int] = (0.0, 7.0, 10)
    atom_energies: dict[str, float] | None = None
    ntrain: int | None = None
    ntest: int | None = None


@dataclass
class PypolymlpData:
    """Displacement dataset for pypolymlp input.

    All the snapshots are displacements of one reference supercell, so the
    lattice is shared and stress is not carried. See
    PypolymlpStructureData for the dataset of independent structures.

    displacements : np.ndarray
        Displacements of atoms. shape=(n, natoms, 3)
    forces : np.ndarray
        Displacements of atoms. shape=(n, natoms, 3)
    supercell_energies : np.ndarray, optional
        Energies of supercells. shape=(n,)
    supercell : PhonopyAtoms
        Reference supercell the displacements are relative to.

    """

    displacements: NDArray[np.double]
    forces: NDArray[np.double]
    supercell_energies: NDArray[np.double]
    supercell: PhonopyAtoms

    @classmethod
    def from_displacement_dataset(
        cls, dataset: Type2DisplacementDataset, supercell: PhonopyAtoms
    ) -> PypolymlpData:
        """Return the dataset built from a phonopy type-2 dataset.

        Parameters
        ----------
        dataset : Type2DisplacementDataset
            Displacements with their forces and supercell energies.
        supercell : PhonopyAtoms
            Supercell the displacements are relative to.

        """
        return cls(
            displacements=dataset["displacements"],
            forces=dataset["forces"],  # type: ignore[typeddict-item]
            supercell_energies=dataset["supercell_energies"],  # type: ignore[typeddict-item]
            supercell=supercell,
        )

    def __len__(self) -> int:
        """Return number of snapshots."""
        return len(self.displacements)

    def __getitem__(self, index: slice) -> PypolymlpData:
        """Return the sliced snapshots, sharing the reference supercell."""
        if not isinstance(index, slice):
            raise TypeError("Only slices are supported.")
        return PypolymlpData(
            displacements=self.displacements[index],
            forces=self.forces[index],
            supercell_energies=self.supercell_energies[index],
            supercell=self.supercell,
        )


def develop_pypolymlp(
    train_data: PypolymlpData | PypolymlpStructureData,
    test_data: PypolymlpData | PypolymlpStructureData | None = None,
    params: PypolymlpParams | None = None,
    test_size: float = 0.1,
    verbose: bool = False,
) -> Pypolymlp:  # type: ignore
    """Develop polynomial MLPs of pypolymlp.

    The training mode follows the dataset type. PypolymlpData holds
    displacements of one reference supercell and trains on energies and
    forces. PypolymlpStructureData holds independent structures, so the
    lattices may differ (e.g. strained cells) and stress is used as well.
    Both datasets must be of the same type.

    Parameters
    ----------
    train_data : PypolymlpData or PypolymlpStructureData
        Training dataset. With `test_data` None, this is the whole dataset
        and it is split by `test_size`.
    test_data : PypolymlpData or PypolymlpStructureData, optional
        Test dataset. Default is None, i.e. split `train_data` instead.
        Pass it explicitly to keep the test dataset fixed while the
        training dataset varies, e.g. when measuring how many structures
        the MLP needs.
    params : PypolymlpParams, optional
        Parameters for pypolymlp. Default is None. When `test_data` is None
        and both `params.ntrain` and `params.ntest` are given, they select
        that many entries from the head and the tail, respectively, and
        `test_size` is unused.
    test_size : float, optional
        Fraction of `train_data` used as the test dataset when `test_data`
        is None; see split_pypolymlp_dataset. Default is 0.1.
    verbose : bool, optional
        Verbosity. Default is False.

    Returns
    -------
    polymlp : Pypolymlp
        Pypolymlp object.

    """
    try:
        from pypolymlp.mlp_dev.pypolymlp import (
            Pypolymlp,  # type: ignore[import-untyped]
        )
        from pypolymlp.utils.phonopy_utils import (
            phonopy_cell_to_structure,  # type: ignore[import-untyped]
        )
    except ImportError as exc:
        raise ModuleNotFoundError("Pypolymlp python module was not found.") from exc

    if test_data is None:
        if (
            params is not None
            and params.ntrain is not None
            and params.ntest is not None
        ):
            train_data, test_data = (
                train_data[: params.ntrain],
                train_data[-params.ntest :],
            )
        else:
            n = _split_index(len(train_data), test_size)
            train_data, test_data = train_data[:n], train_data[n:]
    if type(train_data) is not type(test_data):
        raise TypeError(
            "train_data and test_data must be of the same type, but they are "
            f"{type(train_data).__name__} and {type(test_data).__name__}."
        )

    _params = PypolymlpParams() if params is None else params

    if isinstance(train_data, PypolymlpData):
        symbols = train_data.supercell.symbols
    else:
        symbols = train_data.structures[0].symbols
    if _params.atom_energies is None:
        elements_energies = {s: 0.0 for s in symbols}
    else:
        elements_energies = {s: _params.atom_energies[s] for s in symbols}
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=tuple(elements_energies.keys()),
        cutoff=_params.cutoff,
        model_type=_params.model_type,
        max_p=_params.max_p,
        gtinv_order=_params.gtinv_order,
        gtinv_maxl=_params.gtinv_maxl,
        gaussian_params2=_params.gaussian_params2,
        atomic_energy=tuple(elements_energies.values()),
    )
    if isinstance(train_data, PypolymlpData):
        assert isinstance(test_data, PypolymlpData)
        polymlp.set_datasets_displacements(
            train_data.displacements.transpose(0, 2, 1),
            train_data.forces.transpose(0, 2, 1),
            train_data.supercell_energies,
            test_data.displacements.transpose(0, 2, 1),
            test_data.forces.transpose(0, 2, 1),
            test_data.supercell_energies,
            phonopy_cell_to_structure(train_data.supercell),
        )
    else:
        assert isinstance(test_data, PypolymlpStructureData)
        polymlp.set_datasets_structures(
            train_structures=[
                phonopy_cell_to_structure(cell) for cell in train_data.structures
            ],
            test_structures=[
                phonopy_cell_to_structure(cell) for cell in test_data.structures
            ],
            train_energies=train_data.energies,
            test_energies=test_data.energies,
            train_forces=[force.T for force in train_data.forces],
            test_forces=[force.T for force in test_data.forces],
            train_stresses=_structures_virial(train_data),
            test_stresses=_structures_virial(test_data),
        )
    try:
        polymlp.run(verbose=verbose)
    except RuntimeError as e:
        if "singular" in str(e):
            raise PypolymlpDevelopmentError(
                "Pypolymlp development failed due to singularity of "
                "(X.T @ X + alpha * I)"
            ) from e
        else:
            raise RuntimeError(str(e)) from e
    return polymlp


@dataclass
class PypolymlpStructureData:
    """Structure dataset for pypolymlp training with energies, forces, stresses.

    Unlike PypolymlpData, which stores displacements relative to a single
    reference supercell, this stores full structures and can therefore mix
    cells with different lattices (e.g. strained cells) and carry stress.

    structures : list of PhonopyAtoms
        Structures, possibly with different lattices. Length n.
    energies : ndarray
        Total energies in eV. shape=(n,)
    forces : list of ndarray
        Atomic forces in eV/angstrom, one (natoms, 3) array per structure.
    stresses : ndarray or None
        Stress tensors in GPa. shape=(n, 3, 3), or None when stresses are
        not available for all structures.

    """

    structures: list[PhonopyAtoms]
    energies: NDArray[np.double]
    forces: list[NDArray[np.double]]
    stresses: NDArray[np.double] | None

    def __len__(self) -> int:
        """Return number of structures."""
        return len(self.structures)

    def __getitem__(self, index: slice) -> PypolymlpStructureData:
        """Return the sliced structures and their properties."""
        if not isinstance(index, slice):
            raise TypeError("Only slices are supported.")
        return PypolymlpStructureData(
            structures=self.structures[index],
            energies=self.energies[index],
            forces=self.forces[index],
            stresses=None if self.stresses is None else self.stresses[index],
        )


def _split_index(n_total: int, test_size: float) -> int:
    """Return the index that splits n_total entries by test_size."""
    n = int(n_total * (1 - test_size))
    if n < 1 or n >= n_total:
        raise ValueError(
            f"test_size={test_size} leaves {n} training entries out of "
            f"{n_total}; both datasets must be non-empty."
        )
    return n


def split_pypolymlp_dataset(
    data: _DatasetT, test_size: float = 0.1
) -> tuple[_DatasetT, _DatasetT]:
    """Split a dataset into training and test datasets.

    The dataset is not shuffled: the first `1 - test_size` fraction becomes
    the training dataset and the rest becomes the test dataset. Datasets
    also slice directly, e.g. `data[:20]`, which is what a series over
    training-set sizes needs.

    Parameters
    ----------
    data : PypolymlpData or PypolymlpStructureData
        Dataset to split.
    test_size : float, optional
        Fraction of the dataset used as the test dataset. Default is 0.1.

    Returns
    -------
    train_data : PypolymlpData or PypolymlpStructureData
    test_data : PypolymlpData or PypolymlpStructureData

    """
    n = _split_index(len(data), test_size)
    return data[:n], data[n:]


def read_vasprun_dataset(
    filenames: Sequence[str | os.PathLike],
) -> PypolymlpStructureData:
    """Assemble a pypolymlp structure dataset from vasprun.xml files.

    Each file contributes its final ionic step. Stresses are included only
    when every file provides them.

    Parameters
    ----------
    filenames : sequence of str or os.PathLike
        vasprun.xml file names (optionally compressed).

    Returns
    -------
    PypolymlpStructureData

    """
    from phonopy.interface.vasp import read_vasprun_calculation

    structures: list[PhonopyAtoms] = []
    energies: list[float] = []
    forces: list[NDArray[np.double]] = []
    stresses: list[NDArray[np.double]] = []
    have_stress = True
    for filename in filenames:
        cell, energy, force, stress = read_vasprun_calculation(filename)
        structures.append(cell)
        energies.append(energy)
        forces.append(force)
        if stress is None:
            have_stress = False
        else:
            stresses.append(stress)
    return PypolymlpStructureData(
        structures=structures,
        energies=np.array(energies, dtype="double"),
        forces=forces,
        stresses=np.array(stresses, dtype="double")
        if have_stress and stresses
        else None,
    )


def write_pypolymlp_structure_dataset(
    data: PypolymlpStructureData,
    filename: str | os.PathLike = "polymlp_dataset.hdf5",
) -> None:
    """Write a structure dataset to an HDF5 file.

    Per-atom quantities (numbers, scaled positions, forces) are stored
    concatenated over structures with an ``n_atoms`` index, so structures
    with different numbers of atoms are supported.

    All datasets are gzip compressed and stored in their full precision;
    floating-point quantities stay double so that the training data keeps
    the precision of the calculation it was read from.

    Parameters
    ----------
    data : PypolymlpStructureData
        Dataset to write.
    filename : str or os.PathLike, optional
        Output HDF5 file name.

    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    n_atoms = np.array([len(cell) for cell in data.structures], dtype="int64")
    lattices = np.array([cell.cell for cell in data.structures], dtype="double")
    numbers = np.concatenate([cell.numbers for cell in data.structures])
    positions = np.concatenate([cell.scaled_positions for cell in data.structures])
    forces = np.concatenate(data.forces)
    with h5py.File(filename, "w") as w:
        w.create_dataset("n_atoms", data=n_atoms, compression="gzip")
        w.create_dataset("lattices", data=lattices, compression="gzip")
        w.create_dataset("numbers", data=numbers, compression="gzip")
        w.create_dataset("scaled_positions", data=positions, compression="gzip")
        w.create_dataset("energies", data=data.energies, compression="gzip")
        w.create_dataset("forces", data=forces, compression="gzip")
        if data.stresses is not None:
            w.create_dataset("stresses", data=data.stresses, compression="gzip")


def read_pypolymlp_structure_dataset(
    filename: str | os.PathLike = "polymlp_dataset.hdf5",
) -> PypolymlpStructureData:
    """Read a structure dataset written by write_pypolymlp_structure_dataset.

    Parameters
    ----------
    filename : str or os.PathLike, optional
        Input HDF5 file name.

    Returns
    -------
    PypolymlpStructureData

    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install python-h5py.") from exc

    with h5py.File(filename, "r") as f:
        n_atoms = f["n_atoms"][:]
        lattices = f["lattices"][:]
        numbers = f["numbers"][:]
        positions = f["scaled_positions"][:]
        energies = np.array(f["energies"][:], dtype="double")
        forces_flat = f["forces"][:]
        stresses = (
            np.array(f["stresses"][:], dtype="double") if "stresses" in f else None
        )

    offsets = np.concatenate([[0], np.cumsum(n_atoms)])
    structures: list[PhonopyAtoms] = []
    forces: list[NDArray[np.double]] = []
    for i in range(len(n_atoms)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        structures.append(
            PhonopyAtoms(
                numbers=numbers[start:end],
                cell=lattices[i],
                scaled_positions=positions[start:end],
            )
        )
        forces.append(np.array(forces_flat[start:end], dtype="double"))
    return PypolymlpStructureData(
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=stresses,
    )


def _structures_virial(data: PypolymlpStructureData) -> NDArray[np.double] | None:
    """Convert stored stresses (GPa) to pypolymlp virials (eV, 3x3 each).

    The pypolymlp virial is stress times cell volume; here the stress in
    GPa is converted with the eV/angstrom^3-to-GPa factor.

    """
    if data.stresses is None:
        return None
    ev_angstrom_to_gpa = get_physical_units().EVAngstromToGPa
    return np.array(
        [
            stress * structure.volume / ev_angstrom_to_gpa
            for stress, structure in zip(data.stresses, data.structures, strict=True)
        ],
        dtype="double",
    )


def evalulate_pypolymlp(
    polymlp: Pypolymlp,  # type: ignore
    supercells_with_displacements: list[PhonopyAtoms],
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """Run force calculation using pypolymlp.

    Parameters
    ----------
    polymlp : Pypolymlp
        Pypolymlp object.
    supercells_with_displacements : Sequence[PhonopyAtoms]
        Sequence of supercells with displacements.

    Returns
    -------
    energies : np.ndarray
        Energies of supercells. shape=(n,)
    forces : np.ndarray
        Forces of supercells. shape=(n, natoms, 3)
    stresses : np.ndarray
        Stresses of supercells (xx, yy, zz, xy, yz, zx). shape=(n, 6)

    """
    try:
        from pypolymlp.calculator.properties import (
            Properties,  # type: ignore[import-untyped]
        )
        from pypolymlp.utils.phonopy_utils import (
            phonopy_cell_to_structure,  # type: ignore[import-untyped]
        )
    except ImportError as exc:
        raise ModuleNotFoundError("Pypolymlp python module was not found.") from exc

    prop = Properties(params=polymlp.parameters, coeffs=polymlp.coeffs)
    energies, forces, stresses = prop.eval_multiple(
        [phonopy_cell_to_structure(scell) for scell in supercells_with_displacements]
    )
    energies = np.array(energies, dtype="double")
    forces = np.array(np.transpose(forces, (0, 2, 1)), dtype="double", order="C")
    stresses = np.array(stresses, dtype="double", order="C")
    return energies, forces, stresses


def parse_mlp_params(params: str | dict | PypolymlpParams) -> PypolymlpParams:
    """Parse MLP parameters string and return PypolymlpParams.

    Supported MLP parameters
    ------------------------
    cutoff: float = 8.0
    model_type: int = 3
    max_p: int = 2
    gtinv_order: int = 3
    gtinv_maxl: Sequence[int] = (8, 8)
    gaussian_params1: Sequence[float, float, int] = (1.0, 1.0, 1)
    gaussian_params2: Sequence[float, float, int] = (0.0, 7.0, 10)
    atom_energies: Optional[dict[str, float]] = None
    ntrain: Optional[int] = None
    ntest: Optional[int] = None

    Parameters
    ----------
    params : str, dict, PyPolymlpParams
        Parameters for pypolymlp.

    Note
    ----
    When str, it should be written as follows:

        "cutoff = 10.0, gtinv_maxl = 8 8"
        "atom_energies = Si -0.35864636 O -0.95743902"


    """
    if isinstance(params, dict):
        return PypolymlpParams(**params)
    elif isinstance(params, PypolymlpParams):
        return params
    elif isinstance(params, str):
        params_dict: dict[str, Any] = {}
        for param in params.split(","):
            key_val = [v.strip().lower() for v in param.split("=")]
            if len(key_val) != 2:
                break
            key, val = key_val
            if key == "gtinv_maxl":
                params_dict[key] = tuple(map(int, val.split()))
            elif key == "gaussian_params1" or key == "gaussian_params2":
                vals = val.split()
                params_dict[key] = (float(vals[0]), float(vals[1]), int(vals[2]))
            elif key == "atom_energies":
                vals = val.split()
                if len(vals) % 2 != 0:
                    raise ValueError(
                        "The input list must have an even number of elements."
                    )
                params_dict[key] = {
                    vals[i]: float(vals[i + 1]) for i in range(0, len(vals), 2)
                }
            elif key == "cutoff":
                params_dict[key] = float(val)
            else:
                if key in ("model_type", "max_p", "gtinv_order", "ntrain", "ntest"):
                    params_dict[key] = int(val)
        return PypolymlpParams(**params_dict)
    else:
        raise RuntimeError("params has to be dict, str, or PypolymlpParams.")


def save_pypolymlp(mlp: Pypolymlp, filename: str) -> None:  # type: ignore
    """Save MLP data to file."""
    mlp.save_mlp(filename=filename)


def load_pypolymlp(filename: str | os.PathLike | None) -> Pypolymlp:  # type: ignore
    """Load MLP data from file."""
    mlp = Pypolymlp()  # type: ignore
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rt") as fp:
        mlp.load_mlp(fp)
    return mlp


def relax_atomic_positions(
    unitcell: PhonopyAtoms,
    polymlp: Pypolymlp,  # type: ignore
    method: Literal["BFGS", "CG", "L-BFGS-B", "SLSQP"] = "BFGS",
    gtol: float = 1e-4,
    maxiter: int = 1000,
    c1: float | None = None,
    c2: float | None = None,
    verbose: bool = False,
) -> PhonopyAtoms | None:
    """Relax structure using pypolymlp.

    Parameters
    ----------
    unitcell : PhonopyAtoms
        Unit cell structure to be relaxed.
    polymlp : Pypolymlp
        Pypolymlp object with parameters and coefficients.
    verbose : bool, optional
        Verbosity. Default is False.

    Returns
    -------
    relaxed_cell : PhonopyAtoms or None
        Relaxed atomic positions. Return None if no relaxation is performed due
        to symmetry constraints.

    """
    try:
        from pypolymlp.api.pypolymlp_calc import (
            PypolymlpCalc,  # type: ignore[import-untyped]
        )
        from pypolymlp.utils.phonopy_utils import (
            structure_to_phonopy_cell,  # type: ignore[import-untyped]
        )
    except ImportError as exc:
        raise ModuleNotFoundError("Pypolymlp python module was not found.") from exc

    polymlp = PypolymlpCalc(
        params=polymlp.parameters, coeffs=polymlp.coeffs, verbose=verbose
    )
    polymlp.load_phonopy_structures([unitcell])
    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=False,
        relax_positions=True,
    )
    _, _, success = polymlp.run_geometry_optimization(
        method=method, gtol=gtol, maxiter=maxiter, c1=c1, c2=c2
    )
    if success is None:
        relaxed_cell = None
    elif success is False:
        raise PypolymlpRelaxationError(
            "Relaxation of atomic positions by pypolymlp failed."
        )
    else:
        relaxed_cell = structure_to_phonopy_cell(polymlp.first_structure)

    return relaxed_cell


def get_change_in_positions(
    relaxed_cell: PhonopyAtoms, original_cell: PhonopyAtoms, verbose: bool = False
) -> NDArray[np.double]:
    """Show change by relaxation."""
    diffs = relaxed_cell.scaled_positions - original_cell.scaled_positions
    diffs -= np.rint(diffs)
    disps = np.linalg.norm(diffs @ original_cell.cell, axis=1)
    if verbose:
        print("Change in fractional position and in distance:")
        for i, (symbol, d, disp) in enumerate(
            zip(original_cell.symbols, diffs, disps, strict=True)
        ):
            print(
                f"{i + 1:3d} {symbol:<2}: {d[0]:11.8f} {d[1]:11.8f} {d[2]:11.8f} "
                f"(|d|={disp:.8f})"
            )
    return diffs
