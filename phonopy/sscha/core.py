# SPDX-License-Identifier: BSD-3-Clause
"""SSCHA calculation."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy import Phonopy
from phonopy.harmonic.force_constants import compact_fc_to_full_fc
from phonopy.interface.mlp import PhonopyMLP
from phonopy.physical_units import get_physical_units


@dataclass(frozen=True)
class SSCHAIterationResult:
    """Free energy obtained in one SSCHA iteration.

    The values belong to the force constants that the iteration sampled, i.e.
    those current when it started, not to the ones it produced from that
    sample. The harmonic part and the ensemble averaged for the anharmonic
    part then come from the same force constants, which makes the free energy
    the SSCHA free energy of those force constants.

    Energies are in eV per primitive cell.

    """

    iteration: int
    free_energy: float
    free_energy_error: float
    harmonic: float
    anharmonic: float


class MLPSSCHA:
    """Iterative approach SSCHA using MLP."""

    def __init__(
        self,
        ph: Phonopy,
        mlp: PhonopyMLP,
        temperature: float | None = None,
        number_of_snapshots: int | Literal["auto"] | None = None,
        max_iterations: int | None = None,
        distance: float | None = None,
        fc_calculator: str | None = None,
        mesh: float | Sequence[int] | NDArray[np.int64] | None = None,
        random_seed: int | None = None,
        log_level: int = 0,
    ) -> None:
        """Init method.

        ph : Phonopy
            Phonopy instance.
        mlp : PhonopyMLP
            PhonopyMLP instance.
        temperature : float, optional
            Temperature in K, by default 300.0.
        number_of_snapshots : int, optional
            Number of snapshots, by default 2000.
        max_iterations : int, optional
            Maximum number of iterations, by default 10.
        distance : float, optional
            Distance of displacements, by default is None, which gives 0.01.
        fc_calculator : str, optional
            Force constants calculator. The default is None, which means "symfc".
        mesh : float, array_like, or None, optional
            Sampling mesh used to compute the harmonic part of the free energy,
            by default 100.0.
        random_seed : int or None, optional
            Seed for random number generator passed to generate_displacements.
            The default is None.
        log_level : int, optional
            Log level, by default 0.

        """
        if mlp is None:
            raise ValueError("MLP is not provided.")

        if temperature is None:
            self._temperature = 300.0
        else:
            self._temperature = temperature
        self._number_of_snapshots: int | Literal["auto"]
        if number_of_snapshots is None:
            self._number_of_snapshots = 1000
        else:
            self._number_of_snapshots = number_of_snapshots
        if max_iterations is None:
            self._max_iterations = 10
        else:
            self._max_iterations = max_iterations
        if distance is None:
            self._distance = 0.01
        else:
            self._distance = distance
        if fc_calculator is None:
            self._fc_calculator = "symfc"
        else:
            self._fc_calculator = fc_calculator
        self._mesh: float | Sequence[int] | NDArray[np.int64]
        if mesh is None:
            self._mesh = 100.0
        else:
            self._mesh = mesh
        self._random_seed = random_seed
        self._log_level = log_level

        self._free_energy: float | None = None
        self._free_energy_error: float | None = None
        self._harmonic_free_energy: float | None = None
        self._anharmonic_free_energy: float | None = None
        self._history: list[SSCHAIterationResult] = []
        self._initial_force_constants_provided = ph.force_constants is not None

        self._ph = ph.replicate()
        self._ph.mlp = PhonopyMLP(mlp=mlp.mlp)
        self._ph.nac_params = copy.deepcopy(ph.nac_params)

        # Calculate supercell energy without displacements
        self._ph.generate_displacements(distance=0, number_of_snapshots=1)
        self._ph.evaluate_mlp()
        self._supercell_energy = float(self._ph.supercell_energies[0])
        self._ph.dataset = None

        if ph.force_constants is None:
            self._iter_counter = 0
        else:
            if log_level:
                print("Use provided force constants.")
                print("")
            fc = ph.force_constants
            if fc.shape[0] != fc.shape[1]:  # compact form
                fc = compact_fc_to_full_fc(self._ph.primitive, fc)
            self._ph.force_constants = fc
            self._iter_counter = 1

    @property
    def phonopy(self) -> Phonopy:
        """Return Phonopy instance."""
        return self._ph

    @property
    def free_energy(self) -> float:
        """Return free energy in eV."""
        return self._require_free_energy(self._free_energy)

    @property
    def harmonic_free_energy(self) -> float:
        """Return harmonic part of the free energy in eV.

        This is the free energy of the harmonic phonons of the force
        constants, computed by mesh sampling. It carries no sampling noise.

        """
        return self._require_free_energy(self._harmonic_free_energy)

    @property
    def anharmonic_free_energy(self) -> float:
        """Return anharmonic part of the free energy in eV.

        This is the ensemble average of the anharmonic correction over the
        supercells with random displacements. The statistical error of the
        free energy comes entirely from it.

        """
        return self._require_free_energy(self._anharmonic_free_energy)

    @property
    def history(self) -> tuple[SSCHAIterationResult, ...]:
        """Return free energies of the iterations run so far.

        The initialization step (iteration 0) is absent: its displacements are
        drawn at a fixed distance rather than from a canonical ensemble, so no
        free energy is defined for it.

        """
        return tuple(self._history)

    @property
    def temperature(self) -> float:
        """Return temperature in K."""
        return self._temperature

    @property
    def number_of_snapshots(self) -> int | Literal["auto"]:
        """Return number of snapshots sampled in each iteration."""
        return self._number_of_snapshots

    @property
    def max_iterations(self) -> int:
        """Return maximum number of iterations."""
        return self._max_iterations

    @property
    def distance(self) -> float:
        """Return displacement distance used in the initialization step."""
        return self._distance

    @property
    def fc_calculator(self) -> str:
        """Return force constants calculator."""
        return self._fc_calculator

    @property
    def mesh(self) -> float | Sequence[int] | NDArray[np.int64]:
        """Return sampling mesh used for the harmonic free energy."""
        return self._mesh

    @property
    def random_seed(self) -> int | None:
        """Return seed of the random number generator."""
        return self._random_seed

    @property
    def initial_force_constants_provided(self) -> bool:
        """Return whether force constants were given at instantiation.

        When they were, the initialization step is skipped and the iterations
        start from them.

        """
        return self._initial_force_constants_provided

    @property
    def force_constants(self) -> NDArray[np.double]:
        """Return force constants."""
        fc = self._ph.force_constants
        assert fc is not None
        return fc

    @property
    def free_energy_error(self) -> float:
        """Return statistical error of free energy in eV.

        This is the standard error of the mean of the anharmonic correction
        over the supercells with random displacements. The harmonic free
        energy is determined by the force constants and does not contribute
        to it.

        """
        return self._require_free_energy(self._free_energy_error)

    @staticmethod
    def _require_free_energy(value: float | None) -> float:
        if value is None:
            raise RuntimeError(
                "Free energy is not calculated yet. Run an iteration, or call "
                "sample_supercells() and calculate_free_energy()."
            )
        return value

    @property
    def harmonic_potential_energy(self) -> float:
        """Return supercell energies."""
        return float(np.average(self._harmonic_potential_energies))

    @property
    def potential_energy(self) -> float:
        """Return potential energy."""
        return float(np.average(self._potential_energies))

    @property
    def _harmonic_potential_energies(self) -> NDArray[np.double]:
        """Return harmonic potential energies of individual supercells."""
        d = self._ph.displacements
        assert isinstance(d, np.ndarray)
        return np.einsum("ijkl,mik,mjl->m", self.force_constants, d, d) / 2

    @property
    def _potential_energies(self) -> NDArray[np.double]:
        """Return potential energies of individual supercells."""
        return self._ph.supercell_energies - self._supercell_energy

    def sample_supercells(self) -> None:
        """Sample supercells with random displacements and evaluate the MLPs.

        The displacements are sampled from the canonical ensemble of the
        harmonic phonons of the current force constants at the temperature,
        and the forces and energies of the supercells are evaluated by the
        MLPs.

        This is the sampling step of one SSCHA iteration. It is also usable on
        its own, followed by ``calculate_free_energy``, to obtain the free
        energy of the force constants currently set.

        """
        # Mutating the dataset clears the force constants, which are needed to
        # evaluate the free energy of this sampling. They are restored below.
        fc = self._ph.force_constants
        self._ph.generate_displacements(
            number_of_snapshots=self._number_of_snapshots,
            temperature=self._temperature,
            random_seed=self._random_seed,
        )

        if self._log_level:
            displacements = self._ph.displacements
            assert isinstance(displacements, np.ndarray)
            hist, bin_edges = np.histogram(
                np.linalg.norm(displacements, axis=2), bins=10
            )
            size = np.prod(displacements.shape[0:2])
            for i, h in enumerate(hist):
                length = round(h / size * 100)
                print(
                    f"  [{bin_edges[i]:4.3f}, {bin_edges[i + 1]:4.3f}] " + "*" * length
                )
            print("Evaluate MLP to obtain forces using pypolymlp", flush=True)

        self._ph.evaluate_mlp()
        self._ph.force_constants = fc

    def calculate_free_energy(
        self, mesh: float | Sequence[int] | NDArray[np.int64] | None = None
    ) -> None:
        """Calculate SSCHA free energy and its statistical error.

        Given the force constants Phi, the free energy per primitive cell is

            F = F_harm + (1/N) sum_i a_i,

        where F_harm is the harmonic free energy of the current force
        constants and

            a_i = (E_i - E_0 - (1/2) sum_ab Phi_ab u_ia u_ib) / n_cell

        is the anharmonic correction obtained from the i-th of the N
        supercells with random displacements. E_i is the supercell energy,
        E_0 the energy of the supercell without displacements, Phi the force
        constants, u_i the displacements, and n_cell the number of primitive
        cells in the supercell. The indices a and b run over the atoms in
        the supercell and the Cartesian directions.

        F_harm is determined by the force constants and carries no sampling
        noise. Therefore the statistical error of F is the standard error of
        the mean of a_i,

            error = std(a, ddof=1) / sqrt(N),

        which is returned by the ``free_energy_error`` property. This error is
        conditional on Phi. The uncertainty of Phi itself, which is determined
        from a stochastic sampling, is not included.

        Parameters
        ----------
        mesh : float, array_like, or None, optional
            Sampling mesh for F_harm. The default is None, which means the
            mesh given at instantiation.

        """
        self._ph.run_mesh(mesh=self._mesh if mesh is None else mesh)
        self._ph.run_thermal_properties(temperatures=[self._temperature])
        hfe = (
            self._ph.thermal_properties.free_energy[0] / get_physical_units().EvTokJmol
        )
        n_cell = len(self._ph.supercell) / len(self._ph.primitive)
        anharmonic = (
            self._potential_energies - self._harmonic_potential_energies
        ) / n_cell
        self._harmonic_free_energy = float(hfe)
        self._anharmonic_free_energy = float(np.average(anharmonic))
        self._free_energy = self._harmonic_free_energy + self._anharmonic_free_energy
        if len(anharmonic) > 1:
            self._free_energy_error = float(
                np.std(anharmonic, ddof=1) / np.sqrt(len(anharmonic))
            )
        else:
            self._free_energy_error = float("nan")

    def run(self) -> MLPSSCHA:
        """Run through all iterations."""
        for _ in self:
            if self._log_level:
                print("")
        return self

    def __iter__(self) -> MLPSSCHA:
        """Iterate over force constants calculations."""
        return self

    def __next__(self) -> int:
        """Calculate next force constants."""
        if self._iter_counter == self._max_iterations + 1:
            self._iter_counter = 0
            raise StopIteration
        self._run()
        self._iter_counter += 1
        return self._iter_counter - 1

    def _run(self) -> None:
        if self._log_level and self._iter_counter == 0:
            print(
                f"[ SSCHA initialization (rd={self._distance}, "
                f"n_supercells={self._number_of_snapshots}) ]",
                flush=True,
            )
        if self._log_level and self._iter_counter > 0:
            print(f"[ SSCHA iteration {self._iter_counter} / {self._max_iterations} ]")
            print(
                f"Generate {self._number_of_snapshots} supercells with displacements "
                f"at {self._temperature} K",
                flush=True,
            )

        if self._iter_counter == 0:
            self._ph.generate_displacements(
                distance=self._distance,
                number_of_snapshots=self._number_of_snapshots,
                random_seed=self._random_seed,
            )
            if self._log_level:
                print("Evaluate MLP to obtain forces using pypolymlp", flush=True)
            self._ph.evaluate_mlp()
        else:
            self.sample_supercells()
            # The free energy is evaluated here, before the force constants are
            # refitted below, so that its harmonic part and the ensemble
            # averaged for its anharmonic part belong to the same force
            # constants. Evaluated after the refit, the two would come from
            # different force constants and the value would be no SSCHA free
            # energy of either. The initialization step is left out: its
            # displacements are drawn at a fixed distance rather than from a
            # canonical ensemble, so it has no free energy to record.
            self.calculate_free_energy()
            self._history.append(
                SSCHAIterationResult(
                    iteration=self._iter_counter,
                    free_energy=self.free_energy,
                    free_energy_error=self.free_energy_error,
                    harmonic=self.harmonic_free_energy,
                    anharmonic=self.anharmonic_free_energy,
                )
            )

        if self._log_level:
            print("Calculate force constants using symfc", flush=True)
        self._ph.produce_force_constants(
            fc_calculator="symfc",
            fc_calculator_log_level=self._log_level if self._log_level > 1 else 0,
            calculate_full_force_constants=True,
            show_drift=False,
        )
