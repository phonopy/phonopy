"""Tests of custom exceptions."""

import pytest

from phonopy import Phonopy
from phonopy.exception import ForcesetsNotFoundError


def test_ForcesetsNotFoundError(nacl_unitcell_order1: Phonopy):
    """Test of ForcesetsNotFoundError."""
    ph = Phonopy(nacl_unitcell_order1)
    ph.generate_displacements()
    with pytest.raises(ForcesetsNotFoundError):
        ph.produce_force_constants()
