"""Tests of phonopy-bandplot command."""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.cui.phonopy_bandplot_script import (
    PhonopyBandplotMockArgs,
    _arrange_band_data,
    _cut_dos,
    _find_wrong_path_connections,
    _get_dos,
    _get_label_for_latex,
    _get_max_frequency,
    _read_band_hdf5,
    _read_band_yaml,
    _read_dos_dat,
    _write_gnuplot_data,
)
from phonopy.phonon.band_structure import get_band_qpoints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_band_structure(ph: Phonopy, tmp_dir: str | os.PathLike):
    """Run band structure on NaCl and write band.yaml / band.hdf5."""
    band_paths = [
        [[0, 0, 0], [0.5, 0, 0.5]],  # G -> X
        [[0.5, 0.25, 0.75], [0, 0, 0]],  # W -> G (disconnected)
    ]
    qpoints = get_band_qpoints(band_paths, npoints=11)
    ph.run_band_structure(
        qpoints, path_connections=[False, False], labels=["G", "X", "W", "G"]
    )
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_dir)
    ph.band_structure.write_yaml()
    ph.band_structure.write_hdf5()
    return original_cwd


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def test_get_label_for_latex():
    """Test underscore escaping for LaTeX labels."""
    assert _get_label_for_latex("Gamma_1") == r"Gamma\_1"
    assert _get_label_for_latex("G") == "G"
    assert _get_label_for_latex("K_2_3") == r"K\_2\_3"


def test_get_max_frequency():
    """Test maximum frequency extraction from a list of arrays."""
    freqs = [np.array([1.0, 2.0, 3.0]), np.array([0.5, 4.0, 2.5])]
    assert _get_max_frequency(freqs) == pytest.approx(4.0)


def test_find_wrong_path_connections_consistent():
    """Return 0 when all path_connections lists are the same."""
    pcs = [[True, False], [True, False], [True, False]]
    assert _find_wrong_path_connections(pcs) == 0


def test_find_wrong_path_connections_inconsistent():
    """Return the index of the first inconsistent list."""
    pcs = [[True, False], [True, False], [False, False]]
    assert _find_wrong_path_connections(pcs) == 2


def test_cut_dos_both_below_dmax():
    """Both DOS values below dmax: points returned unchanged."""
    result = _cut_dos((0.0, 1.0), (1.0, 2.0), dmax=3.0)
    assert result == [(0.0, 1.0), (1.0, 2.0)]


def test_cut_dos_p1_below_p2_above():
    """p1 below, p2 above dmax: interpolated crossing point inserted."""
    result = _cut_dos((0.0, 1.0), (2.0, 3.0), dmax=2.0)
    assert len(result) == 3
    # crossing at d=2: f = (2-1)/(3-1) * (2-0) + 0 = 1.0
    np.testing.assert_allclose(result[1][0], 1.0)
    assert result[1][1] == 2.0
    assert result[2][1] == 2.0


def test_cut_dos_p1_above_p2_below():
    """p1 above, p2 below dmax: interpolated crossing point inserted."""
    result = _cut_dos((0.0, 3.0), (2.0, 1.0), dmax=2.0)
    assert len(result) == 3
    assert result[0][1] == 2.0
    assert result[2] == (2.0, 1.0)


def test_cut_dos_both_above_dmax():
    """Both DOS values above dmax: both clipped to dmax."""
    result = _cut_dos((0.0, 4.0), (1.0, 5.0), dmax=3.0)
    assert result == [[0.0, 3.0], [0.0, 3.0]]


def test_get_dos_clips_at_dmax():
    """_get_dos clips DOS values that exceed dmax."""
    f = np.array([0.0, 1.0, 2.0])
    d = np.array([0.5, 2.5, 1.5])  # middle point exceeds dmax=2.0
    result = _get_dos(d, f, dmax=2.0)
    assert result.shape[0] == 2  # (frequencies, dos) rows
    assert np.all(result[1] <= 2.0)


def test_arrange_band_data_connected_no_labels():
    """Segments sharing a boundary q-point are marked connected."""
    qpoints = np.array(
        [
            [0, 0, 0],
            [0.25, 0, 0],
            [0.5, 0, 0],
            [0.5, 0, 0],
            [0.5, 0.25, 0],
            [0.5, 0.5, 0],
        ]
    )
    distances = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    frequencies = np.zeros((6, 2))
    segment_nqpoints = np.array([3, 3])

    labels, path_connections, freq_list, dist_list = _arrange_band_data(
        distances, frequencies, qpoints, segment_nqpoints, []
    )

    assert labels is None
    assert path_connections == [True, False]
    assert len(freq_list) == 2
    assert len(dist_list) == 2


def test_arrange_band_data_disconnected_no_labels():
    """Segments with different boundary q-points are marked disconnected."""
    qpoints = np.array(
        [
            [0, 0, 0],
            [0.25, 0, 0],
            [0.5, 0, 0],
            [0.0, 0.5, 0],
            [0.25, 0.5, 0],
            [0.5, 0.5, 0],
        ]
    )
    distances = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])
    frequencies = np.zeros((6, 2))
    segment_nqpoints = np.array([3, 3])

    labels, path_connections, _, _ = _arrange_band_data(
        distances, frequencies, qpoints, segment_nqpoints, []
    )

    assert labels is None
    assert path_connections == [False, False]


def test_arrange_band_data_with_labels_disconnected():
    """Disconnected label_pairs: all endpoint labels are kept."""
    qpoints = np.zeros((6, 3))
    distances = np.zeros(6)
    frequencies = np.zeros((6, 2))
    segment_nqpoints = np.array([3, 3])
    label_pairs = [["G", "X"], ["M", "G"]]

    labels, path_connections, _, _ = _arrange_band_data(
        distances, frequencies, qpoints, segment_nqpoints, label_pairs
    )

    assert labels == ["G", "X", "M", "G"]
    assert path_connections == [False, False]


def test_arrange_band_data_with_labels_connected():
    """Connected label_pairs: shared endpoint label is merged."""
    qpoints = np.zeros((6, 3))
    distances = np.zeros(6)
    frequencies = np.zeros((6, 2))
    segment_nqpoints = np.array([3, 3])
    label_pairs = [["G", "X"], ["X", "M"]]

    labels, path_connections, _, _ = _arrange_band_data(
        distances, frequencies, qpoints, segment_nqpoints, label_pairs
    )

    assert labels == ["G", "X", "M"]
    assert path_connections == [True, False]


# ---------------------------------------------------------------------------
# File reading functions (generated on-the-fly from NaCl fixture)
# ---------------------------------------------------------------------------


def test_read_band_yaml(ph_nacl: Phonopy):
    """Test _read_band_yaml parses distances, frequencies, segments, labels."""
    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = _make_band_structure(ph_nacl, tmp)
        try:
            distances, frequencies, qpoints, segment_nqpoints, labels = _read_band_yaml(
                "band.yaml"
            )
        finally:
            os.chdir(original_cwd)

    assert len(distances) == 22  # 2 segments x 11 points
    assert frequencies.shape == (22, 6)  # 6 branches for NaCl
    assert qpoints.shape == (22, 3)
    np.testing.assert_array_equal(segment_nqpoints, [11, 11])
    assert labels == [["G", "X"], ["W", "G"]]
    np.testing.assert_allclose(
        frequencies[0], [0, 0, 0, 4.61643516, 4.61643516, 7.39632718], atol=1e-3
    )


def test_read_band_hdf5(ph_nacl: Phonopy):
    """Test _read_band_hdf5 parses same data as yaml."""
    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = _make_band_structure(ph_nacl, tmp)
        try:
            distances_y, frequencies_y, _, segment_nqpoints_y, labels_y = (
                _read_band_yaml("band.yaml")
            )
            distances_h, frequencies_h, _, segment_nqpoints_h, labels_h = (
                _read_band_hdf5("band.hdf5")
            )
        finally:
            os.chdir(original_cwd)

    np.testing.assert_allclose(distances_y, distances_h, atol=1e-6)
    np.testing.assert_allclose(frequencies_y, frequencies_h, atol=1e-3)
    np.testing.assert_array_equal(segment_nqpoints_y, segment_nqpoints_h)
    assert labels_y == labels_h


def test_read_dos_dat(ph_nacl: Phonopy):
    """Test _read_dos_dat reads frequencies and DOS correctly."""
    ph_nacl.run_mesh([10, 10, 10])
    ph_nacl.run_total_dos()

    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = pathlib.Path.cwd()
        os.chdir(tmp)
        try:
            ph_nacl.write_total_dos()
            frequencies, dos = _read_dos_dat("total_dos.dat")
        finally:
            os.chdir(original_cwd)

    assert frequencies.shape == (201,)
    assert dos.shape == (201, 2)  # total_dos.dat has 1 column → 1 + total = 2
    assert np.all(np.diff(frequencies) > 0)  # sorted


def test_read_dos_dat_with_factor(ph_nacl: Phonopy):
    """Test _read_dos_dat applies frequency factor correctly."""
    ph_nacl.run_mesh([10, 10, 10])
    ph_nacl.run_total_dos()

    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = pathlib.Path.cwd()
        os.chdir(tmp)
        try:
            ph_nacl.write_total_dos()
            freqs, _ = _read_dos_dat("total_dos.dat")
            freqs_scaled, _ = _read_dos_dat("total_dos.dat", factor=2.0)
        finally:
            os.chdir(original_cwd)

    np.testing.assert_allclose(freqs_scaled, freqs * 2.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# gnuplot output (main with is_gnuplot=True)
# ---------------------------------------------------------------------------


def test_main_gnuplot(capsys, ph_nacl: Phonopy):
    """Test gnuplot output header and first data point."""
    with tempfile.TemporaryDirectory() as tmp:
        original_cwd = _make_band_structure(ph_nacl, tmp)
        try:
            params = PhonopyBandplotMockArgs(filenames=["band.yaml"], is_gnuplot=True)
            _write_gnuplot_data(params)
        finally:
            os.chdir(original_cwd)

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert lines[0] == "# End points of segments: "
    # Three segment endpoints: 0, end-of-G->X, end-of-W->G
    assert lines[1].startswith("#")
    endpoints = [float(v) for v in lines[1].split() if v != "#"]
    assert len(endpoints) == 3
    np.testing.assert_allclose(endpoints[0], 0.0, atol=1e-6)

    # First data line: distance=0, frequency≈0 (Gamma point acoustic)
    first_data = [float(v) for v in lines[2].split()]
    np.testing.assert_allclose(first_data[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(first_data[1], 0.0, atol=1e-3)
