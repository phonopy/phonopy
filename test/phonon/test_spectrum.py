# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.phonon.spectrum."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.grid import BZGrid, get_ir_grid_points
from phonopy.phonon.spectrum import (
    SmearingDOSAccumulator,
    SmearingDOSResult,
    TetrahedronDOSAccumulator,
    TetrahedronDOSResult,
)


def _ir_setup(
    ph: Phonopy, mesh: list[int]
) -> tuple[BZGrid, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build BZGrid and frequencies on its irreducible grid points."""
    bzgrid = BZGrid(
        mesh,
        lattice=ph.primitive.cell,
        symmetry_dataset=ph.primitive_symmetry.dataset,
    )
    ir_grid_points, ir_weights, ir_grid_map = get_ir_grid_points(bzgrid)
    qpoints_int = bzgrid.addresses[bzgrid.grg2bzg[ir_grid_points]]
    qpoints_frac = qpoints_int / np.array(bzgrid.D_diag, dtype=float)
    ph.run_qpoints(qpoints_frac)
    assert ph.qpoints is not None
    freqs_ir = np.asarray(ph.qpoints.frequencies, dtype="double")
    return bzgrid, freqs_ir, ir_grid_points, ir_weights, ir_grid_map


def test_tetrahedron_dos_default_mode_property_matches_explicit_ones(
    ph_nacl_nonac: Phonopy,
):
    """``mode_property=None`` must match ``mode_property=ones[None,...,None]``."""
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    ones = np.ones(freqs.shape + (1,), dtype="double")[None]

    auto = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=20,
    ).result
    explicit = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        mode_property=ones,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=20,
    ).result

    assert isinstance(auto, TetrahedronDOSResult)
    np.testing.assert_array_equal(auto.sampling_points, explicit.sampling_points)
    np.testing.assert_array_equal(auto.cumulative, explicit.cumulative)
    np.testing.assert_array_equal(auto.density, explicit.density)
    assert auto.density.shape == (1, 20, 1)
    assert auto.cumulative.shape == (1, 20, 1)


def test_tetrahedron_dos_3d_bin_values_matches_2d_for_shared_axis(
    ph_nacl_nonac: Phonopy,
):
    """3D bin_values with identical batches must match 2D bin_values.

    The 3D path produces per-batch sampling axes (shape (n_batch, n_sampling)),
    so each batch's sampling axis matches the shared 2D axis exactly when all
    batches share the same bin range.

    """
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    n_batch = 3
    mode_prop = np.broadcast_to(
        np.ones(freqs.shape + (2,), dtype="double")[None],
        (n_batch,) + freqs.shape + (2,),
    ).copy()
    bins_3d = np.broadcast_to(freqs[None], (n_batch,) + freqs.shape).copy()

    res_2d = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        mode_property=mode_prop,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=20,
    ).result
    res_3d = TetrahedronDOSAccumulator(
        bins_3d,
        bzgrid,
        mode_property=mode_prop,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=20,
    ).result

    assert res_2d.sampling_points.shape == (20,)
    assert res_3d.sampling_points.shape == (n_batch, 20)
    for b in range(n_batch):
        np.testing.assert_array_equal(res_2d.sampling_points, res_3d.sampling_points[b])
    np.testing.assert_allclose(res_2d.cumulative, res_3d.cumulative, atol=1e-15)
    np.testing.assert_allclose(res_2d.density, res_3d.density, atol=1e-15)
    assert res_3d.density.shape == (n_batch, 20, 2)


def test_tetrahedron_dos_3d_bin_values_per_batch_sampling(
    ph_nacl_nonac: Phonopy,
):
    """3D bin_values with batch-varying ranges produces per-batch sampling."""
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    bins_3d = np.stack([freqs, freqs * 2.0, freqs * 0.5], axis=0)
    n_batch = 3
    mode_prop = np.ones((n_batch,) + freqs.shape + (1,), dtype="double")

    res = TetrahedronDOSAccumulator(
        bins_3d,
        bzgrid,
        mode_property=mode_prop,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=15,
    ).result

    assert res.sampling_points.shape == (n_batch, 15)
    for b in range(n_batch):
        assert res.sampling_points[b, 0] == pytest.approx(bins_3d[b].min())
        assert res.sampling_points[b, -1] >= bins_3d[b].max()
    # Each batch's sampling axis differs because the bin ranges differ.
    assert not np.allclose(res.sampling_points[0], res.sampling_points[1])
    assert not np.allclose(res.sampling_points[1], res.sampling_points[2])


def test_tetrahedron_dos_explicit_sampling_points(ph_nacl_nonac: Phonopy):
    """``sampling_points`` argument is preserved verbatim."""
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    points = np.linspace(-1.0, 10.0, 11, dtype="double")
    res = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        sampling_points=points,
    ).result
    np.testing.assert_array_equal(res.sampling_points, points)
    assert res.density.shape == (1, 11, 1)


def test_tetrahedron_dos_density_integral_normalisation(ph_nacl_nonac: Phonopy):
    """Cumulative DOS at the upper end equals total band count per primitive cell.

    For pure DOS over all bands the cumulative must approach n_band as the
    sampling point exceeds the maximum frequency.

    """
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    n_band = freqs.shape[1]
    above_max = float(freqs.max()) + 1.0
    res = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        sampling_points=np.array([above_max], dtype="double"),
    ).result
    np.testing.assert_allclose(res.cumulative[0, 0, 0], n_band, atol=1e-10)


def test_tetrahedron_dos_lang_rust_matches_c(ph_nacl_nonac: Phonopy):
    """Rust backend matches C bit-for-bit when available."""
    pytest.importorskip("phonors")
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    res_c = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=20,
        lang="C",
    ).result
    res_rust = TetrahedronDOSAccumulator(
        freqs,
        bzgrid,
        ir_grid_points=ir_gp,
        ir_grid_weights=ir_w,
        ir_grid_map=ir_map,
        num_sampling_points=20,
        lang="Rust",
    ).result
    np.testing.assert_allclose(res_c.cumulative, res_rust.cumulative, atol=1e-14)
    np.testing.assert_allclose(res_c.density, res_rust.density, atol=1e-14)


def test_tetrahedron_dos_rejects_invalid_bin_values_ndim(ph_nacl_nonac: Phonopy):
    """1D or 4D bin_values must raise."""
    bzgrid, freqs, ir_gp, ir_w, ir_map = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    with pytest.raises(ValueError, match="bin_values must be 2D or 3D"):
        TetrahedronDOSAccumulator(freqs.ravel(), bzgrid)


def test_tetrahedron_dos_rejects_mode_property_batch_mismatch(
    ph_nacl_nonac: Phonopy,
):
    """3D bin_values and 4D mode_property must agree on the batch axis."""
    bzgrid, freqs, _, _, _ = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    bins_3d = np.broadcast_to(freqs[None], (3,) + freqs.shape).copy()
    bad_prop = np.ones((2,) + freqs.shape + (1,), dtype="double")
    with pytest.raises(ValueError, match="batch"):
        TetrahedronDOSAccumulator(bins_3d, bzgrid, mode_property=bad_prop)


def test_smearing_dos_default_mode_property_matches_explicit_ones(
    ph_nacl_nonac: Phonopy,
):
    """``mode_property=None`` must match ``mode_property=ones``."""
    _, freqs, _, ir_w, _ = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    auto = SmearingDOSAccumulator(freqs, ir_w, sigma=0.3, num_sampling_points=20).result
    explicit = SmearingDOSAccumulator(
        freqs,
        ir_w,
        mode_property=np.ones((1,) + freqs.shape, dtype="double"),
        sigma=0.3,
        num_sampling_points=20,
    ).result
    assert isinstance(auto, SmearingDOSResult)
    np.testing.assert_array_equal(auto.sampling_points, explicit.sampling_points)
    np.testing.assert_array_equal(auto.density, explicit.density)
    assert auto.density.shape == (1, 20)


def test_smearing_dos_sigma_is_used(ph_nacl_nonac: Phonopy):
    """Different sigma values must produce different DOS (regression test).

    Earlier the sigma kwarg was silently ignored; this guards the fix.

    """
    _, freqs, _, ir_w, _ = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    res_narrow = SmearingDOSAccumulator(
        freqs, ir_w, sigma=0.1, num_sampling_points=20
    ).result
    res_wide = SmearingDOSAccumulator(
        freqs, ir_w, sigma=1.0, num_sampling_points=20
    ).result
    assert not np.allclose(res_narrow.density, res_wide.density)


def test_smearing_dos_explicit_sampling_points(ph_nacl_nonac: Phonopy):
    """``sampling_points`` argument is preserved verbatim."""
    _, freqs, _, ir_w, _ = _ir_setup(ph_nacl_nonac, [5, 5, 5])
    points = np.linspace(0.0, 8.0, 9, dtype="double")
    res = SmearingDOSAccumulator(freqs, ir_w, sampling_points=points, sigma=0.3).result
    np.testing.assert_array_equal(res.sampling_points, points)
    assert res.density.shape == (1, 9)
