# SPDX-License-Identifier: BSD-3-Clause
"""Read and write free energies computed outside the QHA drivers.

The phonon free energy of a temperature-dependent method, and the electronic
free energy on a dense k-point mesh, are both expensive enough to be computed
on another machine than the analysis. Both are one array of shape
(temperatures, grid points) in eV per primitive cell, so one file format
carries either, tagged with which it is.

The file also carries the temperatures it was computed on, and optionally the
lattice lengths of the grid points it was computed for. A reader can then
refuse a file that does not belong to the dataset at hand instead of pairing
its rows and columns with whatever is there.

"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import Literal, get_args

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from phonopy import __version__

FreeEnergyKind = Literal["phonon", "electronic"]
KINDS = get_args(FreeEnergyKind)


@dataclasses.dataclass(frozen=True)
class FreeEnergies:
    """Free energies over a temperature grid and a lattice grid.

    Attributes
    ----------
    kind : Literal["phonon", "electronic"]
        Which term the file holds.
    temperatures : ndarray
        Temperatures in K. shape=(temperatures,)
    free_energies : ndarray
        Free energies in eV per primitive cell.
        shape=(temperatures, grid_points)
    errors : ndarray, optional
        Uncertainties of free_energies in eV per primitive cell, as a sampled
        method reports them. Same shape, or None.
    lattice_lengths : ndarray, optional
        Lattice-vector lengths (a, b, c) of the grid points in angstrom,
        shape=(grid_points, 3), or None. Carried so that a reader can check
        the file against the grid it is about to be used with.

    """

    kind: FreeEnergyKind
    temperatures: NDArray[np.double]
    free_energies: NDArray[np.double]
    errors: NDArray[np.double] | None = None
    lattice_lengths: NDArray[np.double] | None = None

    @property
    def n_grid_points(self) -> int:
        """Return the number of grid points."""
        return int(self.free_energies.shape[1])


def write_free_energies_hdf5(
    temperatures: Sequence[float] | NDArray[np.double],
    free_energies: Sequence[Sequence[float]] | NDArray[np.double],
    filename: str | os.PathLike = "free_energies.hdf5",
    kind: FreeEnergyKind = "phonon",
    errors: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    lattice_lengths: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
) -> None:
    """Write free energies over a temperature grid and a lattice grid.

    Parameters
    ----------
    temperatures : array_like
        Temperatures in K. shape=(temperatures,)
    free_energies : array_like
        Free energies in eV per primitive cell.
        shape=(temperatures, grid_points)
    filename : str or os.PathLike, optional
        Output file name.
    kind : Literal["phonon", "electronic"], optional
        Which term the file holds. Default is "phonon".
    errors : array_like, optional
        Uncertainties of free_energies, same shape and unit.
    lattice_lengths : array_like, optional
        Lattice-vector lengths (a, b, c) of the grid points in angstrom.
        shape=(grid_points, 3). Recommended: it is what lets a reader check
        the file against the dataset it is used with.

    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}.")

    temps = np.array(temperatures, dtype="double")
    values = np.array(free_energies, dtype="double")
    if values.ndim != 2 or len(values) != len(temps):
        raise ValueError(
            f"free_energies must have shape (temperatures, grid_points) with "
            f"{len(temps)} rows, but has {values.shape}."
        )

    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        w.attrs["phonopy_version"] = __version__
        w.attrs["kind"] = kind
        w.attrs["unit"] = "eV/primitive_cell"
        w.create_dataset("temperatures", data=temps)
        w.create_dataset("free_energies", data=values)
        if errors is not None:
            err = np.array(errors, dtype="double")
            if err.shape != values.shape:
                raise ValueError(
                    f"errors must have the shape of free_energies, "
                    f"{values.shape}, but has {err.shape}."
                )
            w.create_dataset("errors", data=err)
        if lattice_lengths is not None:
            lengths = np.array(lattice_lengths, dtype="double")
            if lengths.shape != (values.shape[1], 3):
                raise ValueError(
                    f"lattice_lengths must have shape "
                    f"{(values.shape[1], 3)}, but has {lengths.shape}."
                )
            w.create_dataset("lattice_lengths", data=lengths)


def read_free_energies_hdf5(
    filename: str | os.PathLike = "free_energies.hdf5",
) -> FreeEnergies:
    """Read free energies written by write_free_energies_hdf5."""
    with h5py.File(filename, "r") as f:
        kind = str(f.attrs["kind"])
        if kind not in KINDS:
            raise ValueError(
                f"{filename} holds free energies of kind {kind!r}, which is "
                f"not one of {KINDS}."
            )
        return FreeEnergies(
            kind=kind,  # type: ignore[arg-type]
            temperatures=np.array(f["temperatures"][:], dtype="double"),
            free_energies=np.array(f["free_energies"][:], dtype="double"),
            errors=(
                np.array(f["errors"][:], dtype="double") if "errors" in f else None
            ),
            lattice_lengths=(
                np.array(f["lattice_lengths"][:], dtype="double")
                if "lattice_lengths" in f
                else None
            ),
        )


def check_free_energies(
    free_energies: FreeEnergies,
    kind: FreeEnergyKind,
    temperatures: NDArray[np.double],
    lattice_lengths: NDArray[np.double],
    filename: str | os.PathLike | None = None,
) -> NDArray[np.double]:
    """Check a file against the run it is about to be used in.

    The kind, the temperature grid and the grid points all have to match, and
    a mismatch is reported rather than paired row by row: the file is
    typically computed on another machine, where nothing ties it to the
    dataset the analysis reads.

    Parameters
    ----------
    free_energies : FreeEnergies
        What was read.
    kind : Literal["phonon", "electronic"]
        The kind the caller needs.
    temperatures : ndarray
        The temperature grid of the run in K.
    lattice_lengths : ndarray
        Lattice-vector lengths of the run's grid points in angstrom.
        shape=(grid_points, 3)
    filename : str or os.PathLike, optional
        Where the free energies were read from, named in the error messages.

    Returns
    -------
    ndarray
        The free energies. shape=(temperatures, grid_points)

    """
    source = "The free energies" if filename is None else f"{filename}"
    if free_energies.kind != kind:
        raise ValueError(
            f"{source} hold {free_energies.kind} free energies, but "
            f"{kind} ones are expected here."
        )
    temps = free_energies.temperatures
    if len(temps) != len(temperatures) or not np.allclose(temps, temperatures):
        raise ValueError(
            f"{source} were computed on a different temperature grid: "
            f"{len(temps)} points from {temps[0]} to {temps[-1]} K against "
            f"{len(temperatures)} from {temperatures[0]} to "
            f"{temperatures[-1]} K here."
        )
    n_points = len(lattice_lengths)
    if free_energies.n_grid_points != n_points:
        raise ValueError(
            f"{source} hold {free_energies.n_grid_points} grid points "
            f"against {n_points} in the dataset."
        )
    if free_energies.lattice_lengths is not None and not np.allclose(
        free_energies.lattice_lengths, lattice_lengths, atol=1e-5
    ):
        raise ValueError(
            f"{source} were computed for different grid points: their lattice "
            f"lengths do not match those of the dataset."
        )
    return free_energies.free_energies
