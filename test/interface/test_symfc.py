"""Tests for symfc interface."""

from typing import Optional

import pytest

from phonopy import Phonopy
from phonopy.interface.symfc import SymfcFCSolver


@pytest.mark.parametrize("cutoff", [None, {3: 5.0}])
def test_symfc_cutoff(ph_nacl: Phonopy, cutoff: Optional[dict]):
    """Test symfc interface with cutoff distance."""
    ph = ph_nacl
    symfc_solver = SymfcFCSolver(
        ph.supercell,
        ph.symmetry,
        options={"cutoff": cutoff},
        log_level=1,
    )
    symfc_solver.compute_basis_set(
        orders=[2, 3],
    )
    if cutoff is None:
        assert symfc_solver.basis_set[3].basis_set.shape == (786, 758)
    else:
        assert symfc_solver.basis_set[3].basis_set.shape == (80, 67)
