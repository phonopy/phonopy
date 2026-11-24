# SPDX-License-Identifier: BSD-3-Clause
"""Tests of Phonopy API."""

from pathlib import Path

import numpy as np
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

    ph = phonopy.load(
        cwd / ".." / "phonopy_params_NaCl-fd.yaml.xz",
        produce_fc=False,
        log_level=2,
    )
    ph.unit_conversion_factor = 100
    assert ph.unit_conversion_factor == pytest.approx(100)


def test_unit_conversion_factor_QE():
    """Test phonopy_load with QE phonopy_params.yaml."""
    ph = phonopy.load(
        cwd / ".." / "phonopy_params_NaCl-QE.yaml.xz", produce_fc=False, log_level=2
    )
    assert ph.calculator == "qe"
    assert ph.unit_conversion_factor == pytest.approx(108.9707718)


def test_load_yaml_with_numpy_primitive_matrix():
    """Test primitive_matrix given as ndarray with a phonopy.yaml-like file.

    The given matrix takes priority over the one stored in the file.

    """
    filename = cwd / ".." / "phonopy_disp_NaCl.yaml"
    ph = phonopy.load(filename, produce_fc=False)
    assert len(ph.primitive) == 2

    ph = phonopy.load(filename, primitive_matrix=np.eye(3), produce_fc=False)
    np.testing.assert_allclose(ph.primitive_matrix, np.eye(3))
    assert len(ph.primitive) == 8
