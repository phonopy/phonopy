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
from typing import Optional

import numpy as np

from phonopy import Phonopy
from phonopy.interface.mlp import PhonopyMLP


class MLPSSCHA:
    """Iterative approach SSCHA using MLP."""

    def __init__(
        self,
        ph: Phonopy,
        mlp: PhonopyMLP,
        temperature: Optional[float] = None,
        number_of_snapshots: Optional[int] = None,
        max_iterations: Optional[int] = None,
        fc_calculator: Optional[str] = None,
        log_level: int = 0,
    ):
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
        fc_calculator : str, optional
            Force constants calculator. The default is None, which means "symfc".
        log_level : int, optional
            Log level, by default 0.

        """
        if mlp is None:
            raise ValueError("MLP is not provided.")

        if temperature is None:
            self._temperature = 300.0
        else:
            self._temperature = temperature
        if number_of_snapshots is None:
            self._number_of_snapshots = 1000
        else:
            self._number_of_snapshots = number_of_snapshots
        if max_iterations is None:
            self._max_iterations = 10
        else:
            self._max_iterations = max_iterations
        self._max_iterations = max_iterations
        if fc_calculator is None:
            self._fc_calculator = "symfc"
        else:
            self._fc_calculator = fc_calculator
        self._iter_counter = 0
        self._ph = ph
        self._mlp = mlp
        self._log_level = log_level

        self._last_fc: Optional[np.diagonal] = None

    @property
    def force_constants(self) -> np.ndarray:
        """Return force constants."""
        return self._last_fc

    def run(self):
        """Run through all iterations."""
        self._run()
        for _ in self:
            pass

    def __iter__(self):
        """Iterate over force constants calculations."""
        return self

    def __next__(self) -> Phonopy:
        """Calculate next force constants."""
        if self._iter_counter == self._max_iterations:
            self._iter_counter = 0
            raise StopIteration
        self._iter_counter += 1
        return self._run()

    def _run(self) -> Phonopy:
        if self._log_level and self._iter_counter == 0:
            print("[ SSCHA initialization (rd=0.03, n_supercells=20) ]")
        if self._log_level and self._iter_counter > 0:
            print(f"[ SSCHA iteration {self._iter_counter} / {self._max_iterations} ]")
            print(
                f"Generate {self._number_of_snapshots} supercells with displacements "
                f"at {self._temperature} K",
                flush=True,
            )

        ph = self._ph.copy()
        ph.mlp = PhonopyMLP(mlp=self._mlp.mlp)
        ph.nac_params = copy.deepcopy(self._ph.nac_params)

        if self._iter_counter == 0:
            ph.generate_displacements(distance=0.03, number_of_snapshots=20)
        else:
            ph.force_constants = self._last_fc
            ph.generate_displacements(
                number_of_snapshots=self._number_of_snapshots,
                temperature=self._temperature,
            )
            hist, bin_edges = np.histogram(
                np.linalg.norm(ph.displacements, axis=2), bins=10
            )

            if self._log_level:
                size = np.prod(ph.displacements.shape[0:2])
                for i, h in enumerate(hist):
                    length = round(h / size * 100)
                    print(
                        f"  [{bin_edges[i]:4.3f}, {bin_edges[i+1]:4.3f}] "
                        + "*" * length
                    )

        if self._log_level:
            print("Evaluate MLP to obtain forces using pypolymlp", flush=True)

        ph.evaluate_mlp()

        if self._log_level:
            print("Calculate force constants using symfc", flush=True)
        ph.produce_force_constants(
            fc_calculator="symfc",
            fc_calculator_log_level=self._log_level if self._log_level > 1 else 0,
            show_drift=False,
        )
        self._last_fc = ph.force_constants

        if self._log_level:
            print("")

        return ph
