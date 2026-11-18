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


@pytest.mark.parametrize("as_array", [False, True])
def test_load_yaml_with_primitive_matrix(as_array: bool):
    """An explicit matrix overrides the primitive transformation saved in YAML."""
    filename = cwd / ".." / "phonopy_disp_NaCl.yaml"
    matrix = np.eye(3)
    explicit = matrix if as_array else matrix.tolist()
    ph = phonopy.load(filename, primitive_matrix=explicit, produce_fc=False)
    np.testing.assert_allclose(ph.primitive_matrix, matrix)
    np.testing.assert_allclose(ph.primitive.cell, ph.unitcell.cell)
    assert len(ph.primitive) == len(ph.unitcell) == 8

    default = phonopy.load(filename, produce_fc=False)
    assert len(default.primitive) == 2


def test_load_yaml_numpy_primitive_matrix_frequencies(ph_si):
    """A matrix taken from a calculation can be supplied when reloading it."""
    ph = phonopy.load(
        cwd / ".." / "phonopy_params_Si.yaml", primitive_matrix=ph_si.primitive_matrix
    )
    qpoints = [[0.13, 0.17, 0.19], [0.1, 0.2, 0.3]]
    actual = ph.run_qpoints(qpoints).frequencies
    expected = ph_si.run_qpoints(qpoints).frequencies
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
