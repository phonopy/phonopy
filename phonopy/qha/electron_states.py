# SPDX-License-Identifier: BSD-3-Clause
"""Electronic states at one volume point, their file format and primitives.

What the two integration routes of the electronic free energy share: the
container the eigenvalues arrive in, the hdf5 format that carries it between
machines, and the Fermi-Dirac occupation and entropy integrand both are built
on.

"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.structure.atoms import PhonopyAtoms


@dataclasses.dataclass(frozen=True)
class ElectronicStates:
    """Electronic states at a volume point.

    Input container for computing electronic free energies with
    ElectronFreeEnergy.

    Attributes
    ----------
    eigenvalues : ndarray
        Eigenvalues in eV. shape=(spin, kpoints, bands). The spin axis has
        length 1 for non-spin-polarized and 2 for spin-polarized systems.
    weights : ndarray
        Relative k-point weights (e.g., number of arms of the k-star).
        shape=(kpoints,)
    n_electrons : float
        Number of electrons in the unit cell.
    volume : float, optional
        Unit cell volume in angstrom^3. Used only for consistency checks
        against the unit cells the states belong to.
    internal_energy : float, optional
        Static internal energy of the unit cell in eV, e.g., the
        energy (sigma->0) of the calculation the eigenvalues come from.
    spin_degeneracy : int, optional
        Number of electrons one eigenvalue can hold. With None (default) it
        is inferred from the length of the spin axis of eigenvalues. The
        inference is wrong for non-collinear calculations, where the spin
        axis has length 1 but each spinor state holds one electron, so 1
        must be given explicitly. See ElectronFreeEnergy.
    fermi_energy : float, optional
        Fermi energy in eV as reported by the calculation the eigenvalues
        come from. Used to anchor the electron count when the free energy is
        computed from a density of states; see free_energy_from_dos.
    kpoints : ndarray, optional
        The irreducible k-points the eigenvalues sit on, in fractional
        coordinates of the reciprocal basis vectors.
        shape=(kpoints, 3), dtype='double'
    mesh : ndarray, optional
        Numbers of mesh divisions, or the grid generating matrix of a
        generalized regular grid. shape=(3,) or (3, 3), dtype='int64'
    cell : PhonopyAtoms, optional
        The cell the eigenvalues were computed for, needed for the symmetry
        search behind the grid.

    kpoints, mesh and cell are what the tetrahedron method needs and are
    given together or not at all. Without them only the k-point sum is
    available, which is what every caller before them did.

    """

    eigenvalues: NDArray[np.double]
    weights: NDArray[np.int64] | NDArray[np.double]
    n_electrons: float
    volume: float | None = None
    internal_energy: float | None = None
    spin_degeneracy: int | None = None
    fermi_energy: float | None = None
    kpoints: NDArray[np.double] | None = None
    mesh: NDArray[np.int64] | None = None
    cell: PhonopyAtoms | None = None

    def __post_init__(self) -> None:
        """Convert the array fields to ndarray and validate their shapes."""
        object.__setattr__(
            self, "eigenvalues", np.asarray(self.eigenvalues, dtype="double")
        )
        object.__setattr__(self, "weights", np.asarray(self.weights))
        if self.kpoints is not None:
            object.__setattr__(
                self, "kpoints", np.asarray(self.kpoints, dtype="double")
            )
        if self.mesh is not None:
            object.__setattr__(self, "mesh", np.asarray(self.mesh))
        if self.eigenvalues.ndim != 3:
            raise ValueError(
                "eigenvalues must have shape (spin, kpoints, bands), not "
                f"{self.eigenvalues.shape}."
            )
        if self.eigenvalues.shape[0] not in (1, 2):
            raise ValueError(
                "The spin axis of eigenvalues must have length 1 or 2, not "
                f"{self.eigenvalues.shape[0]}."
            )
        if self.weights.ndim != 1 or len(self.weights) != self.eigenvalues.shape[1]:
            raise ValueError("weights must have one value per k-point of eigenvalues.")
        if self.spin_degeneracy not in (None, 1, 2):
            raise ValueError(
                f"spin_degeneracy must be 1 or 2, not {self.spin_degeneracy}."
            )
        grid = (self.kpoints, self.mesh, self.cell)
        if any(item is not None for item in grid) and any(
            item is None for item in grid
        ):
            raise ValueError(
                "kpoints, mesh and cell describe the sampling grid together; "
                "give all three or none."
            )
        if self.kpoints is not None and self.mesh is not None:
            if self.kpoints.shape != (self.eigenvalues.shape[1], 3):
                raise ValueError(
                    "kpoints must have shape (kpoints, 3) matching "
                    f"eigenvalues, not {self.kpoints.shape}."
                )
            if self.mesh.shape not in ((3,), (3, 3)):
                raise ValueError(
                    f"mesh must have shape (3,) or (3, 3), not {self.mesh.shape}."
                )


def resolve_spin_degeneracy(electronic_states: ElectronicStates) -> int:
    """Return the number of electrons one eigenvalue holds.

    Inferred from the spin axis as ElectronFreeEnergy does, unless given
    explicitly. The inference is wrong for a non-collinear calculation, whose
    spin axis has length 1 while each spinor holds one electron.

    """
    if electronic_states.spin_degeneracy is not None:
        return electronic_states.spin_degeneracy
    return 2 if electronic_states.eigenvalues.shape[0] == 1 else 1


def fermi_dirac_occupation(
    energies: NDArray[np.double], mu: float, kt: float
) -> NDArray[np.double]:
    """Return the Fermi-Dirac occupation, guarded against overflow.

    The exponent is clipped at +-100, where the occupation is within 4e-44 of
    0 or 1 and nothing downstream can tell the difference. Without the clip it
    overflows: 0 K is carried as a k_B T small enough that (E - mu) / kt
    leaves the exponent range of a double.

    """
    return 1.0 / (1.0 + np.exp(np.clip((energies - mu) / kt, -100.0, 100.0)))


def entropy_terms(occupations: NDArray[np.double]) -> NDArray[np.double]:
    """Return f ln f + (1 - f) ln(1 - f), of the same shape as occupations.

    The terms vanish wherever the occupation saturates, and they are masked
    out there rather than left to cancel: at f = 1e-12 the term is already
    -3e-11, and below it lies log(0). The 0.5 the mask puts in their place is
    any value in range that keeps the logarithms finite; it is discarded.

    """
    mask = (occupations > 1e-12) & (occupations < 1.0 - 1e-12)
    safe = np.where(mask, occupations, 0.5)
    return np.where(mask, safe * np.log(safe) + (1.0 - safe) * np.log1p(-safe), 0.0)


def write_electronic_states_hdf5(
    electronic_structures: Sequence[ElectronicStates],
    filename: str | os.PathLike = "electronic_states.hdf5",
) -> None:
    """Write electronic states in hdf5.

    All ElectronicStates must carry volume and internal_energy. The file
    contains one group "volume-XXX" per volume point with the datasets
    eigenvalues ((spin, kpoints, bands), eV), weights ((kpoints,)),
    n_electrons, volume (angstrom^3), and energy (eV, static internal
    energy). spin_degeneracy and fermi_energy are written only when they are
    set. The number of volume points is stored in the root attribute
    "n_volumes".

    States that carry the grid they were computed on also get kpoints
    ((kpoints, 3)), mesh ((3,) or (3, 3)) and a "cell" subgroup of lattice,
    scaled_positions, numbers and magnetic_moments, which is what the
    tetrahedron method needs. A file without them reads back as a k-point sum,
    which is what it was.

    """
    import h5py

    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        w.attrs["n_volumes"] = len(electronic_structures)
        for i, electronic_states in enumerate(electronic_structures):
            if (
                electronic_states.volume is None
                or electronic_states.internal_energy is None
            ):
                raise ValueError(
                    f"electronic_structures[{i}] must carry volume and internal_energy."
                )
            group = w.create_group(f"volume-{i:03d}")
            group.create_dataset(
                "eigenvalues",
                data=electronic_states.eigenvalues,
                compression="gzip",
            )
            group.create_dataset("weights", data=electronic_states.weights)
            group.create_dataset(
                "n_electrons", data=float(electronic_states.n_electrons)
            )
            group.create_dataset("volume", data=float(electronic_states.volume))
            group.create_dataset(
                "energy", data=float(electronic_states.internal_energy)
            )
            if electronic_states.spin_degeneracy is not None:
                group.create_dataset(
                    "spin_degeneracy", data=int(electronic_states.spin_degeneracy)
                )
            if electronic_states.fermi_energy is not None:
                group.create_dataset(
                    "fermi_energy", data=float(electronic_states.fermi_energy)
                )
            if electronic_states.cell is not None:
                assert electronic_states.kpoints is not None
                assert electronic_states.mesh is not None
                group.create_dataset("kpoints", data=electronic_states.kpoints)
                group.create_dataset("mesh", data=electronic_states.mesh)
                _write_cell(group, electronic_states.cell)


def _write_cell(group, cell: PhonopyAtoms) -> None:
    """Write the cell of one volume point into a "cell" subgroup."""
    g = group.create_group("cell")
    g.create_dataset("lattice", data=np.array(cell.cell, dtype="double"))
    g.create_dataset(
        "scaled_positions", data=np.array(cell.scaled_positions, dtype="double")
    )
    g.create_dataset("numbers", data=np.array(cell.numbers, dtype="int64"))
    if cell.magnetic_moments is not None:
        g.create_dataset(
            "magnetic_moments", data=np.array(cell.magnetic_moments, dtype="double")
        )


def _read_cell(group) -> PhonopyAtoms | None:
    """Return the cell of one volume point, or None without a "cell" subgroup."""
    if "cell" not in group:
        return None
    g = group["cell"]
    return PhonopyAtoms(
        numbers=g["numbers"][:],
        cell=g["lattice"][:],
        scaled_positions=g["scaled_positions"][:],
        magnetic_moments=(
            g["magnetic_moments"][:] if "magnetic_moments" in g else None
        ),
    )


def read_electronic_states_hdf5(
    filename: str | os.PathLike = "electronic_states.hdf5",
) -> list[ElectronicStates]:
    """Read electronic states from hdf5.

    Returns a list of ElectronicStates in the file order, each carrying
    volume and internal_energy. The list is the electronic_structures
    parameter of run_qha; internal_energies can then be given as None.

    """
    import h5py

    electronic_structures = []
    with h5py.File(filename, "r") as f:
        n_volumes = int(f.attrs["n_volumes"])
        for i in range(n_volumes):
            group = f[f"volume-{i:03d}"]
            electronic_structures.append(
                ElectronicStates(
                    eigenvalues=group["eigenvalues"][:],
                    weights=group["weights"][:],
                    n_electrons=float(group["n_electrons"][()]),
                    volume=float(group["volume"][()]),
                    internal_energy=float(group["energy"][()]),
                    spin_degeneracy=(
                        int(group["spin_degeneracy"][()])
                        if "spin_degeneracy" in group
                        else None
                    ),
                    fermi_energy=(
                        float(group["fermi_energy"][()])
                        if "fermi_energy" in group
                        else None
                    ),
                    kpoints=group["kpoints"][:] if "kpoints" in group else None,
                    mesh=group["mesh"][:] if "mesh" in group else None,
                    cell=_read_cell(group),
                )
            )
    return electronic_structures
