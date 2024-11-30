"""Tests for MLP SSCHA."""

import pathlib

import pytest

from phonopy import Phonopy
from phonopy.interface.mlp import PhonopyMLP
from phonopy.sscha.core import MLPSSCHA

cwd = pathlib.Path(__file__).parent


def test_MLPSSCHA(ph_kcl: Phonopy):
    """Test MLPSSCHA class."""
    pytest.importorskip("pypolymlp")
    pytest.importorskip("symfc")
    mlp = PhonopyMLP().load(cwd / ".." / "mlpsscha_KCl-120.pmlp")
    sscha = MLPSSCHA(
        ph_kcl,
        mlp,
        number_of_snapshots=2,
        max_iterations=2,
        temperature=300,
        log_level=2,
    )
    sscha.run()

    sscha = MLPSSCHA(
        ph_kcl,
        mlp,
        number_of_snapshots=2,
        max_iterations=2,
        temperature=300,
        log_level=1,
    )
    count = 0
    for i, _ in enumerate(sscha):
        count = i
    assert count == 2
