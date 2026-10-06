# SPDX-License-Identifier: BSD-3-Clause
"""Deprecated location of phonopy.electron.kpoint_sum."""

import warnings

from phonopy.electron.kpoint_sum import (  # noqa: F401
    ElectronFreeEnergy,
    compute_free_energy_by_kpoint_sum,
    get_free_energy_at_T,
)

warnings.warn(
    "phonopy.qha.electron_kpoint_sum is deprecated. Use "
    "phonopy.electron.kpoint_sum instead.",
    DeprecationWarning,
    stacklevel=2,
)

# The name the k-point sum carried while it was the only route in electron.py.
compute_free_energy_and_entropy = compute_free_energy_by_kpoint_sum
