"""Tests for band structure calculation."""

import numpy as np

from phonopy.phonon.band_structure import get_band_qpoints


def test_band_structure(ph_nacl):
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


def test_band_structure_gv(ph_nacl):
    """Test band structure calculation with group velocity by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=True, is_band_connection=False
    )
    assert "group_velocities" in ph_nacl.get_band_structure_dict()


def test_band_structure_bc(ph_nacl):
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


def _get_band_qpoints():
    band_paths = [
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0], [0, 0, 0], [0.5, 0.25, 0.75]],
    ]
    qpoints = get_band_qpoints(band_paths, npoints=11)
    return qpoints
