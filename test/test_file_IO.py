"""Tests of file_IO functions."""

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy.file_IO import (
    parse_BORN,
    read_force_constants_hdf5,
    write_force_constants_to_hdf5,
)

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


def test_parse_BORN():
    """Test of parse_BORN."""
    ph = phonopy.load(cwd / "phonopy_disp_NaCl.yaml")
    nac_params = parse_BORN(ph.primitive, filename=cwd / "BORN_NaCl")
    z = np.eye(3) * 1.086875
    epsilon = np.eye(3) * 2.43533967
    np.testing.assert_allclose(nac_params["born"], [z, -z], atol=1e-5)
    np.testing.assert_allclose(nac_params["dielectric"], epsilon, atol=1e-5)
    assert pytest.approx(14.400) == nac_params["factor"]


def test_write_force_constants_to_hdf5():
    """Test write_force_constants_to_hdf5."""
    pytest.importorskip("h5py")

    write_force_constants_to_hdf5(np.zeros(1), physical_unit="eV/angstrom^2")
    for created_filename in ["force_constants.hdf5"]:
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        fc, physical_unit = read_force_constants_hdf5(
            file_path, return_physical_unit=True
        )
        assert fc[0] == pytest.approx(0)
        assert physical_unit == "eV/angstrom^2"
        file_path.unlink()
