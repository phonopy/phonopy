"""SSCHA calculation."""

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

import copy
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy import Phonopy
from phonopy.interface.mlp import PhonopyMLP
from phonopy.physical_units import get_physical_units


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
        self._random_seed = random_seed
        self._log_level = log_level

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
            self._ph.force_constants = ph.force_constants
            self._iter_counter = 1

    @property
    def phonopy(self) -> Phonopy:
        """Return Phonopy instance."""
        return self._ph

    @property
    def free_energy(self) -> float:
        """Return free energy in eV."""
        return self._free_energy

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
        return self._free_energy_error

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

    def calculate_free_energy(self, mesh: float = 100.0) -> None:
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

        """
        self._ph.run_mesh(mesh=mesh)
        self._ph.run_thermal_properties(temperatures=[self._temperature])
        hfe = (
            self._ph.thermal_properties.free_energy[0] / get_physical_units().EvTokJmol
        )
        n_cell = len(self._ph.supercell) / len(self._ph.primitive)
        anharmonic = (
            self._potential_energies - self._harmonic_potential_energies
        ) / n_cell
        self._free_energy = hfe + float(np.average(anharmonic))
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
        else:
            self._ph.generate_displacements(
                number_of_snapshots=self._number_of_snapshots,
                temperature=self._temperature,
                random_seed=self._random_seed,
            )
            displacements = self._ph.displacements
            assert isinstance(displacements, np.ndarray)
            hist, bin_edges = np.histogram(
                np.linalg.norm(displacements, axis=2), bins=10
            )

            if self._log_level:
                size = np.prod(displacements.shape[0:2])
                for i, h in enumerate(hist):
                    length = round(h / size * 100)
                    print(
                        f"  [{bin_edges[i]:4.3f}, {bin_edges[i + 1]:4.3f}] "
                        + "*" * length
                    )

        if self._log_level:
            print("Evaluate MLP to obtain forces using pypolymlp", flush=True)

        self._ph.evaluate_mlp()

        if self._log_level:
            print("Calculate force constants using symfc", flush=True)
        self._ph.produce_force_constants(
            fc_calculator="symfc",
            fc_calculator_log_level=self._log_level if self._log_level > 1 else 0,
            calculate_full_force_constants=True,
            show_drift=False,
        )
