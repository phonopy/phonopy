# SPDX-License-Identifier: BSD-3-Clause
"""Deprecated location of the electronic free energy; use phonopy.electron."""

import warnings

from phonopy.electron.kpoint_sum import (  # noqa: F401
    ElectronFreeEnergy,
    compute_free_energy_by_kpoint_sum,
    get_free_energy_at_T,
)
from phonopy.electron.states import (  # noqa: F401
    ElectronicStates,
    entropy_terms,
    fermi_dirac_occupation,
    read_electronic_states_hdf5,
    resolve_spin_degeneracy,
    write_electronic_states_hdf5,
)
from phonopy.electron.tetrahedron import (  # noqa: F401
    compute_free_energy_by_tetrahedron,
    free_energy_from_dos,
    resolve_energy_window,
)

warnings.warn(
    "phonopy.qha.electron is deprecated. Use phonopy.electron.states, "
    "phonopy.electron.tetrahedron and phonopy.electron.kpoint_sum instead.",
    DeprecationWarning,
    stacklevel=2,
)

# The name the k-point sum carried while it was the only route here.
compute_free_energy_and_entropy = compute_free_energy_by_kpoint_sum
