# SPDX-License-Identifier: BSD-3-Clause
"""Tests for exclude_gamma_acoustic of the thermal properties."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.thermal_properties import (
    GammaAcousticWarning,
    gamma_acoustic_bands,
)

TEMPERATURES = [0.0, 100.0, 300.0]


def _si_with_gamma(
    ph_si: Phonopy, gamma: list[float] | None = None, is_gamma_center: bool = True
) -> Phonopy:
    """Return a Si Phonopy on a 4x4x4 mesh, its Gamma frequencies overwritten.

    The session fixture is copied so that its mesh is left alone. The mesh is
    Gamma centred unless is_gamma_center is False, which gives phonopy's
    default grid, shifted by half a division and without Gamma.

    """
    ph = Phonopy(
        ph_si.unitcell,
        supercell_matrix=ph_si.supercell_matrix,
        primitive_matrix=ph_si.primitive_matrix,
        log_level=0,
    )
    ph.force_constants = ph_si.force_constants
    ph.run_mesh([4, 4, 4], is_gamma_center=is_gamma_center)
    if gamma is not None:
        index, _ = gamma_acoustic_bands(ph.mesh)
        assert index is not None
        ph.mesh.frequencies[index] = gamma
    return ph


def _properties(ph: Phonopy, **kwargs) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GammaAcousticWarning)
        tp = ph.run_thermal_properties(temperatures=TEMPERATURES, **kwargs)
    return (
        tp.free_energy,
        tp.entropy,
        tp.heat_capacity,
        tp.number_of_integrated_modes,
        tp.zero_point_energy,
    )


def _assert_same(a: tuple, b: tuple, zpe_atol: float = 1e-12) -> None:
    for x, y in zip(a[:-1], b[:-1], strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a[-1], b[-1], rtol=0, atol=zpe_atol)


# cutoff_frequency does not act on the zero-point energy, which sums over the
# positive frequencies, so there it keeps h nu / 2 of the acoustic modes, about
# 1e-9 kJ/mol here. exclude_gamma_acoustic excludes them from that sum too.
ZPE_ATOL_AGAINST_CUTOFF = 1e-8


def test_signs_at_gamma_do_not_matter(ph_si: Phonopy):
    """With exclude_gamma_acoustic, the signs of the acoustic modes do not matter."""
    optical = [15.5, 15.5, 15.5]
    plus = _si_with_gamma(ph_si, [1e-7, -1e-7, 2e-7] + optical)
    minus = _si_with_gamma(ph_si, [-1e-7, 1e-7, -2e-7] + optical)
    _assert_same(
        _properties(plus, exclude_gamma_acoustic=True),
        _properties(minus, exclude_gamma_acoustic=True),
    )
    # Without it the two differ by the modes that came out positive.
    assert not np.allclose(_properties(plus)[0], _properties(minus)[0])


def test_same_as_a_small_cutoff(ph_si: Phonopy):
    """Excluding the acoustic modes equals a cutoff that only they fall below."""
    ph = _si_with_gamma(ph_si, [1e-7, 1.5e-7, 2e-7, 15.5, 15.5, 15.5])
    _assert_same(
        _properties(ph, exclude_gamma_acoustic=True),
        _properties(ph, cutoff_frequency=1e-3),
        zpe_atol=ZPE_ATOL_AGAINST_CUTOFF,
    )


def test_mesh_keeps_its_frequencies(ph_si: Phonopy):
    """Only the copy used for the sums is changed."""
    gamma = [1e-7, 1.5e-7, 2e-7, 15.5, 15.5, 15.5]
    ph = _si_with_gamma(ph_si, gamma)
    _properties(ph, exclude_gamma_acoustic=True)
    np.testing.assert_array_equal(ph.mesh.frequencies[0], gamma)


def test_large_imaginary_mode_is_not_taken(ph_si: Phonopy):
    """A large imaginary mode at Gamma is not taken for an acoustic mode."""
    ph = _si_with_gamma(ph_si, [-5.0, -1e-7, 1e-7, 2e-7, 15.5, 15.5])
    _, bands = gamma_acoustic_bands(ph.mesh)
    np.testing.assert_array_equal(bands, [1, 2, 3])
    _assert_same(
        _properties(ph, exclude_gamma_acoustic=True),
        _properties(ph, cutoff_frequency=1e-3),
        zpe_atol=ZPE_ATOL_AGAINST_CUTOFF,
    )


@pytest.mark.filterwarnings("ignore::phonopy.phonon.mesh.MeshSymmetryFallbackWarning")
def test_mesh_without_gamma(ph_si: Phonopy):
    """Nothing changes on a mesh that does not contain Gamma."""
    ph = _si_with_gamma(ph_si, is_gamma_center=False)
    assert gamma_acoustic_bands(ph.mesh) == (None, None)
    _assert_same(
        _properties(ph, exclude_gamma_acoustic=True),
        _properties(ph),
    )


def test_band_indices(ph_si: Phonopy):
    """The acoustic modes are found before band_indices selects bands."""
    ph = _si_with_gamma(ph_si, [1e-7, 1.5e-7, 2e-7, 15.5, 15.5, 15.5])
    _assert_same(
        _properties(ph, band_indices=[[0, 1, 2]], exclude_gamma_acoustic=True),
        _properties(ph, band_indices=[[0, 1, 2]], cutoff_frequency=1e-3),
        zpe_atol=ZPE_ATOL_AGAINST_CUTOFF,
    )
    # Optical bands only: the option has nothing to act on.
    _assert_same(
        _properties(ph, band_indices=[[3, 4, 5]], exclude_gamma_acoustic=True),
        _properties(ph, band_indices=[[3, 4, 5]]),
    )


def test_warning_when_acoustic_modes_enter(ph_si: Phonopy):
    """Without the option, a positive acoustic mode at Gamma is reported."""
    ph = _si_with_gamma(ph_si, [-1e-7, 1e-7, 2e-7, 15.5, 15.5, 15.5])
    with pytest.warns(GammaAcousticWarning, match="2 acoustic mode"):
        ph.run_thermal_properties(temperatures=TEMPERATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GammaAcousticWarning)
        ph.run_thermal_properties(
            temperatures=TEMPERATURES, exclude_gamma_acoustic=True
        )


def test_warning_when_acoustic_modes_are_far_from_zero(ph_si: Phonopy):
    """Acoustic frequencies far from zero are reported, with or without the option."""
    ph = _si_with_gamma(ph_si, [-0.5, -0.4, -0.3, 15.5, 15.5, 15.5])
    for exclude in (False, True):
        with pytest.warns(GammaAcousticWarning, match="far from zero"):
            ph.run_thermal_properties(
                temperatures=TEMPERATURES, exclude_gamma_acoustic=exclude
            )
