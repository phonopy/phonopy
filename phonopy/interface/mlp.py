"""MLP interfaces."""

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
from typing import Any, Optional, Union

import numpy as np

from phonopy.interface.pypolymlp import (
    PypolymlpParams,
    develop_mlp_by_pypolymlp,
    evalulate_pypolymlp,
    load_pypolymlp,
    save_pypolymlp,
)
from phonopy.structure.atoms import PhonopyAtoms


class PhonopyMLP:
    """PhonopyMLP class."""

    def __init__(self, mlp: Optional[Any] = None, log_level: int = 0):
        self._mlp = mlp
        self._log_level = log_level

    @property
    def mlp(self) -> Any:
        """Return MLP instance."""
        return self._mlp

    def save(self, filename: Optional[str] = None):
        """Save MLP."""
        if filename is None:
            _filename = "phonopy.pmlp"
        else:
            _filename = filename
        save_pypolymlp(self._mlp, _filename)

    def load(
        self, filename: Optional[Union[str, bytes, os.PathLike]] = None
    ) -> PhonopyMLP:
        """Load MLP."""
        if filename is None:
            _filename = "phonopy.pmlp"
        else:
            _filename = filename
        self._mlp = load_pypolymlp(_filename)
        return self

    def evaluate(
        self, supercells_with_displacements: list[PhonopyAtoms]
    ) -> list[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate MLP."""
        return evalulate_pypolymlp(self._mlp, supercells_with_displacements)

    def develop(
        self,
        mlp_dataset: dict,
        supercell: PhonopyAtoms,
        params: Optional[Union[PypolymlpParams, dict, str]] = None,
        test_size: float = 0.1,
    ):
        """Develop MLP."""
        self._mlp = develop_mlp_by_pypolymlp(
            mlp_dataset,
            supercell,
            params=params,
            test_size=test_size,
            log_level=self._log_level,
        )
