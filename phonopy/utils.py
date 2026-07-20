# SPDX-License-Identifier: BSD-3-Clause
"""Utility functions."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import spglib
from numpy.typing import NDArray

try:
    spglib.error.OLD_ERROR_HANDLING = False
except AttributeError:
    pass


def similarity_transformation(
    rot: Sequence[Sequence[float]]
    | NDArray[np.double]
    | Sequence[Sequence[int]]
    | NDArray[np.int64],
    mat: Sequence[Sequence[float]]
    | NDArray[np.double]
    | Sequence[Sequence[int]]
    | NDArray[np.int64],
) -> NDArray[np.double]:
    """Similarity transformation by R x M x R^-1."""
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))  # type: ignore


def get_dot_access_dataset(dataset) -> SimpleNamespace | spglib.SpglibDataset:
    """Return dataset with dot access.

    From spglib 2.5, dataset is returned as dataclass.
    To emulate it for older versions, this function is used.

    """
    import spglib

    spg_version = tuple(int(v) for v in spglib.__version__.split(".")[:3])  # type: ignore

    if spg_version < (2, 5, 0):
        return SimpleNamespace(**dataset)
    else:
        return dataset
