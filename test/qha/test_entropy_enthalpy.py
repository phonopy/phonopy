# SPDX-License-Identifier: BSD-3-Clause
"""Tests for S(T, p) and H(T, p) from S(V) fits at V_eq."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.qha.calc import compute_entropy_enthalpy_temperature


def test_polyfit_recovers_exact_quadratic_entropy() -> None:
    """A degree-2 S(V) is recovered exactly by the degree-4 fit."""
    volumes = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype="double")
    temperatures = np.array([0.0, 100.0, 300.0], dtype="double")
    v_eq = np.array([12.3, 12.4, 12.6], dtype="double")

    def s_of_v(v: np.ndarray) -> np.ndarray:
        # S is in eV/K, so the scale is that of a few k_B per cell. At
        # J/K/mol magnitudes T S would swamp G and the absolute tolerance
        # below would fall under the rounding of H.
        return 1e-4 * (3.0 + 0.5 * v + 0.01 * v**2)

    entropy = np.vstack([s_of_v(volumes), s_of_v(volumes), s_of_v(volumes)])
    gibbs = np.array([-1.0, -1.2, -1.8], dtype="double")

    result = compute_entropy_enthalpy_temperature(
        temperatures, volumes, v_eq, entropy, gibbs
    )
    # S is in eV/K and G in eV, so H = G + TS needs no conversion.
    expected_s = s_of_v(v_eq)
    expected_h = gibbs + temperatures * expected_s

    np.testing.assert_allclose(result.entropy, expected_s, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.enthalpy, expected_h, rtol=0, atol=1e-12)


def test_fewer_than_five_volumes_is_refused() -> None:
    """Degree-4 S(V) fit needs at least five volume points."""
    volumes = np.array([10.0, 11.0, 12.0, 13.0], dtype="double")
    temperatures = np.array([100.0, 200.0], dtype="double")
    entropy = np.ones((2, 4), dtype="double")
    gibbs = np.zeros(2, dtype="double")
    with pytest.raises(RuntimeError, match="At least 5 volume points"):
        compute_entropy_enthalpy_temperature(
            temperatures, volumes, volumes[:2], entropy, gibbs
        )
