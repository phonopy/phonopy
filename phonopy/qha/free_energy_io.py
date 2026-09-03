# SPDX-License-Identifier: BSD-3-Clause
"""Read and write free energies computed outside the QHA drivers.

The phonon free energy of a temperature-dependent method and the electronic
free energy on a dense k-point mesh are expensive enough to be computed on
another machine than the analysis, so they travel as files.

Which term a file holds is its type -- ElectronicFreeEnergies,
PhononFreeEnergies or SSCHAFreeEnergies -- and each says in its own docstring
what it carries. The file records the type that wrote it, and
check_free_energies refuses one that does not belong to the run at hand.

"""

from __future__ import annotations

import dataclasses
import os
from typing import ClassVar

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from phonopy import __version__


@dataclasses.dataclass(frozen=True)
class FreeEnergies:
    """Free energies over a temperature grid and a lattice grid.

    What every term has in common. The type of the instance says which term it
    is, so this one is not written or read on its own.

    Attributes
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(temperatures,)
    free_energies : ndarray
        Free energies in eV per primitive cell.
        shape=(temperatures, grid_points)
    errors : ndarray, optional
        Uncertainties of free_energies in eV per primitive cell, as a sampled
        method reports them, or None.
        shape=(temperatures, grid_points)
    lattice_lengths : ndarray, optional
        Lattice-vector lengths (a, b, c) of the grid points in angstrom, or
        None. Carried so that a reader can check the file against the grid it
        is about to be used with.
        shape=(grid_points, 3)

    """

    # The arrays with a temperature axis, and those with one value per grid
    # point. The subtypes extend them; __post_init__ and check_free_energies
    # are what read them.
    OVER_TEMPERATURE: ClassVar[tuple[str, ...]] = ("free_energies", "errors")
    OVER_GRID: ClassVar[tuple[str, ...]] = ()

    temperatures: NDArray[np.double]
    free_energies: NDArray[np.double]
    errors: NDArray[np.double] | None = None
    lattice_lengths: NDArray[np.double] | None = None

    def __post_init__(self) -> None:
        """Check the arrays against each other."""
        shape = self.free_energies.shape
        if len(shape) != 2 or shape[0] != len(self.temperatures):
            raise ValueError(
                f"free_energies must have shape (temperatures, grid_points) "
                f"with {len(self.temperatures)} rows, but has {shape}."
            )
        for name in self.OVER_TEMPERATURE:
            values = getattr(self, name, None)
            if values is not None and values.shape != shape:
                raise ValueError(
                    f"{name} must have the shape of free_energies, {shape}, "
                    f"but has {values.shape}."
                )
        for name in self.OVER_GRID:
            values = getattr(self, name, None)
            if values is not None and values.shape != (shape[1],):
                raise ValueError(
                    f"{name} is one value per grid point, so it must have "
                    f"shape {(shape[1],)}, but has {values.shape}."
                )
        lengths = self.lattice_lengths
        if lengths is not None and lengths.shape != (shape[1], 3):
            raise ValueError(
                f"lattice_lengths must have shape {(shape[1], 3)}, "
                f"but has {lengths.shape}."
            )

    @property
    def n_grid_points(self) -> int:
        """Return the number of grid points."""
        return int(self.free_energies.shape[1])


@dataclasses.dataclass(frozen=True)
class ElectronicFreeEnergies(FreeEnergies):
    """Electronic free energies F_el(T) - F_el(0) over the lattice grid."""


@dataclasses.dataclass(frozen=True)
class PhononFreeEnergies(FreeEnergies):
    """Phonon free energies of a method that reports the free energy alone."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SSCHAFreeEnergies(PhononFreeEnergies):
    """Phonon free energies of the SSCHA, with the terms they were built from.

    Attributes
    ----------
    reference_energies : ndarray
        The energy the free energies are measured from, in eV per primitive
        cell: ``MLPSSCHA.supercell_energy`` over the primitive cells in the
        supercell. Added to free_energies it puts the surface on the
        potential's own energy scale, which is the potential's U.
        potential_energies is measured from it too, so
        ``reference_energies + potential_energies`` is the ensemble average of
        the supercell energy.
        shape=(grid_points,)
    potential_energies : ndarray, optional
        Ensemble average of the potential energy measured from
        reference_energies, in eV per primitive cell, or None.
        shape=(temperatures, grid_points)
    harmonic_potential_energies : ndarray, optional
        Ensemble average of the harmonic potential energy of the force
        constants the free energy belongs to, in eV per primitive cell, or
        None.
        shape=(temperatures, grid_points)
    anharmonic_corrections : ndarray
        potential_energies - harmonic_potential_energies, always present once
        the instance is made. Deprecated as an input; see the notes.
        shape=(temperatures, grid_points)

    Notes
    -----
    The decomposition is

        free_energies = harmonic + anharmonic_corrections,
        anharmonic_corrections = potential_energies
                                 - harmonic_potential_energies,

    so the harmonic free energy is the term not stored. potential_energies and
    harmonic_potential_energies are kept rather than only their difference,
    because they cancel: each varies across the lattice grid far more than
    anharmonic_corrections does.

    Giving anharmonic_corrections instead of potential_energies and
    harmonic_potential_energies is deprecated.

    """

    OVER_TEMPERATURE: ClassVar[tuple[str, ...]] = FreeEnergies.OVER_TEMPERATURE + (
        "anharmonic_corrections",
        "potential_energies",
        "harmonic_potential_energies",
    )
    OVER_GRID: ClassVar[tuple[str, ...]] = ("reference_energies",)

    reference_energies: NDArray[np.double]
    potential_energies: NDArray[np.double] | None = None
    harmonic_potential_energies: NDArray[np.double] | None = None
    # Never None once __post_init__ has run, which is why it is not typed
    # optional: every reader of the attribute gets an array.
    anharmonic_corrections: NDArray[np.double] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Check the two terms, and derive their difference when it is absent."""
        super().__post_init__()
        potential = self.potential_energies
        harmonic = self.harmonic_potential_energies
        if (potential is None) != (harmonic is None):
            raise ValueError(
                "potential_energies and harmonic_potential_energies are the "
                "two terms anharmonic_corrections is the difference of, and "
                "are given together."
            )

        if potential is not None and harmonic is not None:
            difference = potential - harmonic
            if self.anharmonic_corrections is None:
                # frozen=True makes plain assignment raise; the generated
                # __init__ writes its own fields this way, and construction is
                # not over until __post_init__ returns.
                object.__setattr__(self, "anharmonic_corrections", difference)
            elif not np.allclose(difference, self.anharmonic_corrections, atol=1e-8):
                raise ValueError(
                    "anharmonic_corrections must equal potential_energies "
                    "- harmonic_potential_energies."
                )
        elif self.anharmonic_corrections is None:
            raise ValueError(
                "Give potential_energies and harmonic_potential_energies. "
                "Giving anharmonic_corrections instead of them is deprecated."
            )


TYPES: dict[str, type[FreeEnergies]] = {
    cls.__name__: cls
    for cls in (ElectronicFreeEnergies, PhononFreeEnergies, SSCHAFreeEnergies)
}


def write_free_energies_hdf5(
    free_energies: FreeEnergies,
    filename: str | os.PathLike = "free_energies.hdf5",
) -> None:
    """Write free energies over a temperature grid and a lattice grid.

    Parameters
    ----------
    free_energies : FreeEnergies
        One of the types of this module. The file records which, so that
        reading it back as another term is refused rather than silent.
    filename : str or os.PathLike, optional
        Output file name.

    """
    name = type(free_energies).__name__
    if name not in TYPES:
        raise ValueError(
            f"{name} is not one of the free energy types, {', '.join(TYPES)}."
        )

    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        w.attrs["phonopy_version"] = __version__
        w.attrs["type"] = name
        w.attrs["unit"] = "eV/primitive_cell"
        for field in dataclasses.fields(free_energies):
            values = getattr(free_energies, field.name)
            if values is not None:
                w.create_dataset(field.name, data=values)


def read_free_energies_hdf5(
    filename: str | os.PathLike = "free_energies.hdf5",
) -> FreeEnergies:
    """Read free energies written by write_free_energies_hdf5.

    They come back as the type that wrote them.

    """
    with h5py.File(filename, "r") as f:
        name = str(f.attrs["type"]) if "type" in f.attrs else ""
        if name not in TYPES:
            raise ValueError(
                f"{filename} records the free energy type {name!r}, which is "
                f"not one of {', '.join(TYPES)}."
            )
        cls = TYPES[name]
        # Only what the type declares, so a file written when it had a field
        # more is still read, ignoring that one.
        declared = {field.name for field in dataclasses.fields(cls)}
        stored = {
            key: np.array(f[key][:], dtype="double") for key in f if key in declared
        }
        try:
            return cls(**stored)
        except TypeError as error:
            raise ValueError(f"{filename} is a {name}, but {error}") from error


def _temperature_index(
    stored: NDArray[np.double], wanted: NDArray[np.double], atol: float = 1e-6
) -> NDArray[np.int64] | None:
    """Return where each wanted temperature sits in stored.

    None when stored is missing any of them, which is what separates a file
    computed over a wider range from one computed on another grid.

    """
    close = np.abs(stored[None, :] - wanted[:, None]) <= atol
    if not np.all(close.any(axis=1)):
        return None
    return np.argmax(close, axis=1).astype("int64")


def check_free_energies(
    free_energies: FreeEnergies,
    expected: type[FreeEnergies],
    temperatures: NDArray[np.double],
    lattice_lengths: NDArray[np.double],
    filename: str | os.PathLike | None = None,
) -> FreeEnergies:
    """Check a file against the run it is about to be used in.

    The type and the grid points have to match, and a mismatch is reported
    rather than paired row by row: the file is typically computed on another
    machine, where nothing ties it to the dataset the analysis reads.

    The temperatures are the exception. A file computed over a wider range is
    used for the run's own temperatures, so one expensive sweep to 1000 K can
    be replotted to 400 K. Every temperature of the run has to be in it; one
    that is not stops the run, since interpolating would hide a grid the file
    was never computed on.

    Parameters
    ----------
    free_energies : FreeEnergies
        What was read.
    expected : type
        The type the caller needs. A subtype passes, so asking for
        PhononFreeEnergies takes the SSCHA ones as well.
    temperatures : ndarray
        The temperature grid of the run in K.
    lattice_lengths : ndarray
        Lattice-vector lengths of the run's grid points in angstrom.
        shape=(grid_points, 3)
    filename : str or os.PathLike, optional
        Where the free energies were read from, named in the error messages.

    Returns
    -------
    FreeEnergies
        What was read, narrowed to the run's temperature grid, and of the type
        it was read as, which is `expected` or a subtype of it; narrow it
        with isinstance where the subtype matters. Every term
        with a temperature axis is narrowed with it, so they stay aligned with
        each other; the ones without are returned as they are.

    """
    source = "The free energies" if filename is None else f"{filename}"
    if not isinstance(free_energies, expected):
        raise ValueError(
            f"{source} hold {type(free_energies).__name__}, but "
            f"{expected.__name__} are expected here."
        )
    temps = free_energies.temperatures
    index = _temperature_index(temps, temperatures)
    if index is None:
        raise ValueError(
            f"{source} were computed on a different temperature grid: "
            f"{len(temps)} points from {temps[0]} to {temps[-1]} K against "
            f"{len(temperatures)} from {temperatures[0]} to "
            f"{temperatures[-1]} K here, and do not hold every temperature "
            "of this run. A file covering more temperatures is used as it is."
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
    narrowed = {
        name: getattr(free_energies, name)[index]
        for name in free_energies.OVER_TEMPERATURE
        if getattr(free_energies, name) is not None
    }
    return dataclasses.replace(free_energies, temperatures=temps[index], **narrowed)
