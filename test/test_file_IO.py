"""Tests of file_IO functions."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Literal

import h5py
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


@pytest.mark.parametrize("compression", ["gzip", "lzf", 1, 2, None])
def test_write_force_constants_to_hdf5(
    compression: Literal["gzip", "lzf"] | int | None,
):
    """Test write_force_constants_to_hdf5."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        write_force_constants_to_hdf5(
            np.zeros(1), physical_unit="eV/angstrom^2", compression=compression
        )
        with h5py.File("force_constants.hdf5", "r") as f:
            fc = f["force_constants"]
            if compression in ("gzip", "lzf"):
                assert fc.compression == compression  # type: ignore
            elif isinstance(compression, int):
                assert fc.compression == "gzip"  # type: ignore
                assert fc.compression_opts == compression  # type: ignore
            else:
                assert fc.compression is None  # type: ignore

        for created_filename in ["force_constants.hdf5"]:
            file_path = pathlib.Path(created_filename)
            assert file_path.exists()
            fc, physical_unit = read_force_constants_hdf5(
                file_path, return_physical_unit=True
            )
            assert fc[0] == pytest.approx(0)
            assert physical_unit == "eV/angstrom^2"
            file_path.unlink()

        _check_no_files()

        os.chdir(original_cwd)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())
