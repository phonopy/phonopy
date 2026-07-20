# SPDX-License-Identifier: BSD-3-Clause
"""MLP interfaces."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.displacement import Type2DisplacementDataset
from phonopy.interface.pypolymlp import (
    PypolymlpData,
    PypolymlpParams,
    PypolymlpStructureData,
    develop_pypolymlp,
    evalulate_pypolymlp,
    load_pypolymlp,
    parse_mlp_params,
    save_pypolymlp,
)
from phonopy.structure.atoms import PhonopyAtoms


class PhonopyMLP:
    """PhonopyMLP class."""

    def __init__(self, mlp: Any | None = None, log_level: int = 0) -> None:
        self._mlp = mlp
        self._log_level = log_level

    @property
    def mlp(self) -> Any:
        """Return MLP instance."""
        return self._mlp

    def save(self, filename: str | os.PathLike | None = None) -> None:
        """Save MLP."""
        _filename: str | os.PathLike
        if filename is None:
            _filename = "polymlp.yaml"
        else:
            _filename = filename
        save_pypolymlp(self._mlp, _filename)  # type: ignore[arg-type]

    def load(self, filename: str | os.PathLike | None = None) -> PhonopyMLP:
        """Load MLP."""
        _filename: str | os.PathLike
        if filename is None:
            _filename = "polymlp.yaml"
        else:
            _filename = filename
        self._mlp = load_pypolymlp(_filename)
        return self

    def evaluate(
        self, supercells_with_displacements: Sequence[PhonopyAtoms | None]
    ) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
        """Evaluate MLP."""
        return evalulate_pypolymlp(self._mlp, supercells_with_displacements)  # type: ignore[arg-type]

    def develop(
        self,
        mlp_dataset: Type2DisplacementDataset,
        supercell: PhonopyAtoms,
        params: PypolymlpParams | dict | str | None = None,
        test_size: float = 0.1,
    ) -> None:
        """Develop MLP from displacements of one supercell."""
        self._mlp = develop_pypolymlp(
            PypolymlpData.from_displacement_dataset(mlp_dataset, supercell),
            params=None if params is None else parse_mlp_params(params),
            test_size=test_size,
            verbose=self._log_level - 1 > 0,
        )

    def develop_from_structures(
        self,
        train_data: PypolymlpStructureData,
        test_data: PypolymlpStructureData | None = None,
        params: PypolymlpParams | None = None,
        test_size: float = 0.1,
    ) -> None:
        """Develop MLP from structures with individual lattices.

        Unlike `develop`, the structures need not share one supercell, and
        stress is used in the training. This suits datasets of strained
        cells; see develop_pypolymlp.

        """
        self._mlp = develop_pypolymlp(
            train_data,
            test_data=test_data,
            params=params,
            test_size=test_size,
            verbose=self._log_level > 0,
        )
