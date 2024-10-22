"""Tests for band structure calculation."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints

cwd_called = pathlib.Path.cwd()


def test_band_structure(ph_nacl: Phonopy):
    """Test band structure calculation by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=False
    )
    freqs = ph_nacl.get_band_structure_dict()["frequencies"]
    assert len(freqs) == 3
    assert freqs[0].shape == (11, 6)
    np.testing.assert_allclose(
        freqs[0][0], [0, 0, 0, 4.61643516, 4.61643516, 7.39632718], atol=1e-3
    )


def test_band_structure_gv(ph_nacl: Phonopy):
    """Test band structure calculation with group velocity by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=True, is_band_connection=False
    )
    assert "group_velocities" in ph_nacl.get_band_structure_dict()


def test_band_structure_bc(ph_nacl: Phonopy):
    """Test band structure calculation with band connection by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=False
    )
    freqs = ph_nacl.get_band_structure_dict()["frequencies"]
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=True
    )
    freqs_bc = ph_nacl.get_band_structure_dict()["frequencies"]

    # Order of bands is changed by is_band_connection=True.
    np.testing.assert_allclose(
        freqs_bc[1][-1], freqs[1][-1][[0, 1, 5, 3, 4, 2]], atol=1e-3
    )


def test_band_structure_write_hdf5(ph_nacl: Phonopy):
    """Test band structure calculation by NaCl.

    G -> L  False
    X -> G  True
    G -> W  False (last one has to be False)

    """
    pytest.importorskip("h5py")

    _test_band_structure_write_hdf5(ph_nacl, labels=["G", "L", "X", "G", "W"])


def _test_band_structure_write_hdf5(ph_nacl: Phonopy, labels: list[str]):
    import h5py

    ph_nacl.run_band_structure(
        _get_band_qpoints(),
        path_connections=[False, True, False],
        with_group_velocities=False,
        is_band_connection=False,
        is_legacy_plot=False,
        labels=labels,
    )
    ph_nacl.band_structure.write_hdf5()
    for created_filename in ["band.hdf5"]:
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        pairs_ref = [labels[i] for i in (0, 1, 2, 3, 3, 4)]

        with h5py.File(file_path) as f:
            pairs = []
            for pair in f["label"][:]:
                pairs += [pair[0].decode(), pair[1].decode()]
            assert pairs == pairs_ref
            print(pairs)
        file_path.unlink()


def _get_band_qpoints():
    band_paths = [
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0], [0, 0, 0], [0.5, 0.25, 0.75]],
    ]
    qpoints = get_band_qpoints(band_paths, npoints=11)
    return qpoints
