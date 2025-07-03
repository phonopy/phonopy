"""Tests of Phonopy API."""

from pathlib import Path

import pytest

import phonopy

cwd = Path(__file__).parent


def test_unit_conversion_factor():
    """Test phonopy_load with phonopy_params.yaml."""
    ph = phonopy.load(
        cwd / ".." / "phonopy_params_NaCl-fd.yaml.xz", produce_fc=False, log_level=2
    )
    assert ph.calculator is None
    assert ph.unit_conversion_factor == pytest.approx(15.6333023)

    with pytest.warns(DeprecationWarning):
        ph = phonopy.load(
            cwd / ".." / "phonopy_params_NaCl-fd.yaml.xz",
            factor=100,
            produce_fc=False,
            log_level=2,
        )
        assert ph.unit_conversion_factor == pytest.approx(100)


def test_unit_conversion_factor_QE():
    """Test phonopy_load with QE phonopy_params.yaml."""
    ph = phonopy.load(
        cwd / ".." / "phonopy_params_NaCl-QE.yaml.xz", produce_fc=False, log_level=2
    )
    assert ph.calculator == "qe"
    assert ph.unit_conversion_factor == pytest.approx(108.9707718)
