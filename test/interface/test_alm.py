"""Tests for ALM force constants calculator interface."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.interface.alm import (
    ALMFCSolver,
    _slice_displacements_and_forces,
    _update_options,
    run_alm,
)
from phonopy.structure.dataset import get_displacements_and_forces

# ---------------------------------------------------------------------------
# _slice_displacements_and_forces (no ALM dependency)
# ---------------------------------------------------------------------------


def test_slice_no_slicing():
    """No slicing when ndata, nstart, nend are all None."""
    d = np.zeros((10, 5, 3))
    f = np.zeros((10, 5, 3))
    _d, _f, msg = _slice_displacements_and_forces(d, f, None, None, None)
    assert _d is d
    assert _f is f
    assert msg is None


def test_slice_ndata():
    """Slice first ndata snapshots when ndata is given."""
    d = np.arange(10 * 4 * 3, dtype=float).reshape(10, 4, 3)
    f = d * 2
    _d, _f, msg = _slice_displacements_and_forces(d, f, 3, None, None)
    assert _d.shape == (3, 4, 3)
    assert _f.shape == (3, 4, 3)
    np.testing.assert_array_equal(_d, d[:3])
    np.testing.assert_array_equal(_f, f[:3])
    assert msg is not None
    assert "3" in msg


def test_slice_nstart_nend():
    """Slice snapshots by 1-based range [nstart, nend]."""
    d = np.arange(10 * 4 * 3, dtype=float).reshape(10, 4, 3)
    f = d * 2
    _d, _f, msg = _slice_displacements_and_forces(d, f, None, 3, 6)
    # 1-based [3, 6] → 0-based [2:6] → 4 snapshots
    assert _d.shape == (4, 4, 3)
    assert _f.shape == (4, 4, 3)
    np.testing.assert_array_equal(_d, d[2:6])
    assert msg is not None
    assert "3" in msg and "6" in msg


def test_slice_ndata_returns_contiguous_double():
    """Sliced arrays are contiguous C-order float64."""
    d = np.ones((10, 4, 3), dtype="float32")
    f = np.ones((10, 4, 3), dtype="float32")
    _d, _f, _ = _slice_displacements_and_forces(d, f, 5, None, None)
    assert _d.dtype == np.float64
    assert _f.dtype == np.float64
    assert _d.flags["C_CONTIGUOUS"]
    assert _f.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# _update_options (requires ALM)
# ---------------------------------------------------------------------------


def test_update_options_defaults():
    """Default options dict is returned when options string is None."""
    pytest.importorskip("alm")
    opts = _update_options(None)
    assert opts["solver"] == "dense"
    assert opts["ndata"] is None
    assert opts["nstart"] is None
    assert opts["nend"] is None
    assert opts["cutoff"] is None
    assert opts["iconst"] == 11


def test_update_options_solver():
    """Solver option is parsed as string."""
    pytest.importorskip("alm")
    opts = _update_options("solver = sparse")
    assert opts["solver"] == "sparse"


def test_update_options_ndata():
    """Ndata option is parsed as int."""
    pytest.importorskip("alm")
    opts = _update_options("ndata = 5")
    assert opts["ndata"] == 5
    assert isinstance(opts["ndata"], int)


def test_update_options_cutoff_scalar():
    """A single cutoff value is stored as 1-element double array."""
    pytest.importorskip("alm")
    opts = _update_options("cutoff = 6.0")
    assert opts["cutoff"] is not None
    assert opts["cutoff"].dtype == np.float64
    assert opts["cutoff"][0] == pytest.approx(6.0)


def test_update_options_multiple():
    """Multiple options separated by comma are all parsed."""
    pytest.importorskip("alm")
    opts = _update_options("solver = dense, ndata = 4, iconst = 0")
    assert opts["solver"] == "dense"
    assert opts["ndata"] == 4
    assert opts["iconst"] == 0


# ---------------------------------------------------------------------------
# Integration tests: ALMFCSolver / run_alm with NaCl (requires ALM)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nacl_alm_data():
    """Load NaCl phonopy data without computing FC (for ALM tests)."""
    pytest.importorskip("alm")
    import pathlib

    cwd = pathlib.Path(__file__).parent.parent
    import phonopy

    ph = phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        is_compact_fc=False,
        produce_fc=False,
        log_level=0,
    )
    disps, forces = get_displacements_and_forces(ph.dataset)
    return ph, disps, forces


def test_alm_fc_shape_full(nacl_alm_data):
    """Full FC has shape (natom, natom, 3, 3) for harmonic order."""
    ph, disps, forces = nacl_alm_data
    fc_dict = ALMFCSolver(
        ph.supercell,
        ph.primitive,
        disps,
        forces,
        maxorder=1,
        is_compact_fc=False,
    ).force_constants
    natom = len(ph.supercell)
    assert 2 in fc_dict
    assert fc_dict[2].shape == (natom, natom, 3, 3)


def test_alm_fc_shape_compact(nacl_alm_data):
    """Compact FC has shape (nprim, natom, 3, 3) for harmonic order."""
    ph, disps, forces = nacl_alm_data
    fc_dict = ALMFCSolver(
        ph.supercell,
        ph.primitive,
        disps,
        forces,
        maxorder=1,
        is_compact_fc=True,
    ).force_constants
    natom = len(ph.supercell)
    nprim = len(ph.primitive)
    assert 2 in fc_dict
    assert fc_dict[2].shape == (nprim, natom, 3, 3)


def test_alm_acoustic_sum_rule(nacl_alm_data):
    """Row sums of FC over second atom index are close to zero (ASR)."""
    ph, disps, forces = nacl_alm_data
    fc_dict = ALMFCSolver(
        ph.supercell,
        ph.primitive,
        disps,
        forces,
        maxorder=1,
        is_compact_fc=False,
    ).force_constants
    fc = fc_dict[2]
    # Sum over second atom index for each (i, alpha, beta)
    row_sums = fc.sum(axis=1)  # shape (natom, 3, 3)
    np.testing.assert_allclose(row_sums, 0, atol=1e-8)


def test_run_alm_returns_dict(nacl_alm_data):
    """run_alm returns a dict keyed by FC order."""
    ph, disps, forces = nacl_alm_data
    result = run_alm(
        ph.supercell,
        ph.primitive,
        disps,
        forces,
        maxorder=1,
    )
    assert isinstance(result, dict)
    assert 2 in result


def test_alm_ndata_option(nacl_alm_data):
    """Ndata option limits the number of displacement snapshots used."""
    ph, disps, forces = nacl_alm_data
    fc_dict = ALMFCSolver(
        ph.supercell,
        ph.primitive,
        disps,
        forces,
        maxorder=1,
        is_compact_fc=False,
        options="ndata = 1",
    ).force_constants
    natom = len(ph.supercell)
    assert fc_dict[2].shape == (natom, natom, 3, 3)


def test_alm_log_level(nacl_alm_data, capsys):
    """log_level=1 produces output mentioning ALM."""
    ph, disps, forces = nacl_alm_data
    run_alm(
        ph.supercell,
        ph.primitive,
        disps,
        forces,
        maxorder=1,
        log_level=1,
    )
    captured = capsys.readouterr()
    assert "ALM" in captured.out
