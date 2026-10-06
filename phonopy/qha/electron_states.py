# SPDX-License-Identifier: BSD-3-Clause
"""Deprecated location of phonopy.electron.states."""

import warnings

from phonopy.electron.states import (  # noqa: F401
    ElectronicStates,
    entropy_terms,
    fermi_dirac_occupation,
    read_electronic_states_hdf5,
    resolve_spin_degeneracy,
    write_electronic_states_hdf5,
)

warnings.warn(
    "phonopy.qha.electron_states is deprecated. Use phonopy.electron.states instead.",
    DeprecationWarning,
    stacklevel=2,
)
