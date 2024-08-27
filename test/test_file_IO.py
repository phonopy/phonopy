"""Tests of file_IO functions."""

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy.file_IO import parse_BORN

cwd = pathlib.Path(__file__).parent


def test_parse_BORN():
    """Test of parse_BORN."""
    ph = phonopy.load(cwd / "phonopy_disp_NaCl.yaml")
    nac_params = parse_BORN(ph.primitive, filename=cwd / "BORN_NaCl")
    z = np.eye(3) * 1.086875
    epsilon = np.eye(3) * 2.43533967
    np.testing.assert_allclose(nac_params["born"], [z, -z], atol=1e-5)
    np.testing.assert_allclose(nac_params["dielectric"], epsilon, atol=1e-5)
    assert pytest.approx(14.400) == nac_params["factor"]
