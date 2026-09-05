# SPDX-License-Identifier: BSD-3-Clause
"""Read and write what one SSCHA run sampled.

A run reports one free energy per iteration, and which of them to average is
chosen afterwards. The file therefore holds every iteration and no average,
so that another choice costs no sampling. The averages, over a grid of runs,
are phonopy.qha.free_energy_io's SSCHAFreeEnergies.

"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, NamedTuple

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from phonopy import __version__

TYPE = "SSCHARun"


class SSCHAAverage(NamedTuple):
    """One run's estimate, over the iterations after its transient."""

    free_energy: float
    error: float
    potential_energy: float
    harmonic_potential_energy: float


@dataclasses.dataclass(frozen=True)
class SSCHARun:
    """What one SSCHA run sampled, before any averaging.

    Attributes
    ----------
    temperature : float
        Temperature of the run in K.
    free_energies : ndarray
        SSCHA free energy of each iteration, in eV per primitive cell.
        shape=(iterations,)
    errors : ndarray
        Statistical error of each iteration's free energy, in eV per
        primitive cell. shape=(iterations,)
    potential_energies : ndarray
        Ensemble average of the potential energy of each iteration, measured
        from reference_energy, in eV per primitive cell.
        shape=(iterations,)
    harmonic_potential_energies : ndarray
        Ensemble average of the harmonic potential energy of each iteration's
        own force constants, in eV per primitive cell. shape=(iterations,)
    reference_energy : float
        The energy the free energies are measured from, in eV per primitive
        cell: the supercell without displacements over the primitive cells in
        it.
    lattice_lengths : ndarray, optional
        Lattice-vector lengths (a, b, c) of the run's cell in angstrom, or
        None. Carried so that a sweep can place the run on its grid.
        shape=(3,)

    Notes
    -----
    The iteration axis holds the iterations in the order they were made,
    starting with the first one the run recorded. MLPSSCHA numbers and logs
    those from 1, so index k on this axis is iteration k + 1.

    The first iterations start from force constants that are not yet
    self-consistent, and how many of them to leave out of an average is a
    property of the run rather than of this file, so nothing here records it.
    SSCHAFreeEnergies records the choice that was made from these.

    """

    PER_ITERATION = (
        "free_energies",
        "errors",
        "potential_energies",
        "harmonic_potential_energies",
    )

    temperature: float
    free_energies: NDArray[np.double]
    errors: NDArray[np.double]
    potential_energies: NDArray[np.double]
    harmonic_potential_energies: NDArray[np.double]
    reference_energy: float
    lattice_lengths: NDArray[np.double] | None = None

    def __post_init__(self) -> None:
        """Check the arrays against each other."""
        shape = self.free_energies.shape
        if len(shape) != 1:
            raise ValueError(
                f"free_energies is one value per iteration, so it must have "
                f"shape (iterations,), but has {shape}."
            )
        for name in self.PER_ITERATION:
            values = getattr(self, name)
            if values.shape != shape:
                raise ValueError(
                    f"{name} must have the shape of free_energies, {shape}, "
                    f"but has {values.shape}."
                )
        lengths = self.lattice_lengths
        if lengths is not None and lengths.shape != (3,):
            raise ValueError(
                f"lattice_lengths must have shape (3,), but has {lengths.shape}."
            )

    @property
    def n_iterations(self) -> int:
        """Return the number of iterations the run made."""
        return int(self.free_energies.shape[0])

    def _kept(self, transient: int) -> slice:
        """Return the iterations left once the transient is taken off."""
        if not 0 <= transient < self.n_iterations:
            raise ValueError(
                f"transient is {transient}, and the run made "
                f"{self.n_iterations} iterations."
            )
        return slice(transient, None)

    def averaged(self, transient: int = 1) -> SSCHAAverage:
        """Return the estimate over the iterations after the transient.

        The free energy and the two ensemble averages are means of the kept
        iterations. Those are independent draws, so the error of their mean
        adds theirs in quadrature, sqrt(sum(e**2))/n, rather than averaging
        them: n iterations are sqrt(n) more precise than one.

        """
        kept = self._kept(transient)
        errors = self.errors[kept]
        return SSCHAAverage(
            float(self.free_energies[kept].mean()),
            float(np.sqrt(np.square(errors).sum()) / errors.size),
            float(self.potential_energies[kept].mean()),
            float(self.harmonic_potential_energies[kept].mean()),
        )

    def departures(self, transient: int = 1) -> NDArray[np.double]:
        """Return how far each iteration sits from the mean of the kept ones.

        In units of that iteration's own error, so the reading is the same
        whatever the system. An iteration past the transient scatters about
        the fixed point by about its own error and gives about 1; a larger
        value means the iteration was still approaching that point.

        Every iteration is returned, the transient included, since seeing
        where it ends is what the number is for. shape=(iterations,)

        """
        mean = self.free_energies[self._kept(transient)].mean()
        return (self.free_energies - mean) / self.errors

    def report(self, transient: int = 1) -> None:
        """List the iterations, their errors and their departures, in meV."""
        z = self.departures(transient)
        kept = self._kept(transient)
        print("  iter       F [meV]   error [meV]   (F - mean)/error", flush=True)
        for i, (f, e, d) in enumerate(
            zip(self.free_energies, self.errors, z, strict=True)
        ):
            mark = "*" if i < transient else " "
            print(
                f"  {i + 1:4d}{mark} {f * 1e3:12.4f} {e * 1e3:13.4f} {d:+18.1f}",
                flush=True,
            )
        worst = int(np.argmax(np.abs(z[kept]))) + transient
        print(
            f"  * left out as the transient. Of the kept iterations the "
            f"furthest from the mean is {worst + 1}, at {abs(z[worst]):.1f} sigma.",
            flush=True,
        )
        print(
            "  A kept iteration far outside the scatter of the rest is still "
            "in the transient: raise the transient and look again.",
            flush=True,
        )


def write_sscha_run_hdf5(
    run: SSCHARun, filename: str | os.PathLike = "sscha.hdf5"
) -> None:
    """Write what one SSCHA run sampled.

    The file records its type, so that reading it as a free energy is refused
    rather than silent.

    """
    if not isinstance(run, SSCHARun):
        raise ValueError(f"{type(run).__name__} is not an {TYPE}.")
    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        w.attrs["phonopy_version"] = __version__
        w.attrs["type"] = TYPE
        w.attrs["unit"] = "eV/primitive_cell"
        for field in dataclasses.fields(run):
            values = getattr(run, field.name)
            if values is not None:
                w.create_dataset(field.name, data=values)


def read_sscha_run_hdf5(filename: str | os.PathLike = "sscha.hdf5") -> SSCHARun:
    """Read what write_sscha_run_hdf5 wrote."""
    with h5py.File(filename, "r") as f:
        name = str(f.attrs["type"]) if "type" in f.attrs else ""
        if name != TYPE:
            raise ValueError(
                f"{filename} records the type {name!r}, not {TYPE}. The "
                "averages of a grid of runs are read with "
                "read_free_energies_hdf5."
            )
        declared = {field.name for field in dataclasses.fields(SSCHARun)}
        stored: dict[str, Any] = {}
        for key in f:
            if key not in declared:
                continue
            # temperature and reference_energy are stored as scalar datasets.
            if f[key].shape == ():
                stored[key] = float(f[key][()])
            else:
                stored[key] = np.array(f[key][:], dtype="double")
        try:
            return SSCHARun(**stored)
        except TypeError as error:
            raise ValueError(f"{filename} is an {TYPE}, but {error}") from error
