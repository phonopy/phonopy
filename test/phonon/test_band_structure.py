"""Tests for band structure calculation."""
from phonopy.phonon.band_structure import get_band_qpoints


def test_band_structure(ph_nacl):
    """Test band structure calculation by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=False
    )
    ph_nacl.get_band_structure_dict()


def test_band_structure_gv(ph_nacl):
    """Test band structure calculation with group velocity by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=True, is_band_connection=False
    )
    ph_nacl.get_band_structure_dict()


def test_band_structure_bc(ph_nacl):
    """Test band structure calculation with band connection by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=True
    )
    ph_nacl.get_band_structure_dict()


def _get_band_qpoints():
    band_paths = [
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0], [0, 0, 0], [0.5, 0.25, 0.75]],
    ]
    qpoints = get_band_qpoints(band_paths, npoints=11)
    return qpoints
