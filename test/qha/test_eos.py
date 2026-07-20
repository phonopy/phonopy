# SPDX-License-Identifier: BSD-3-Clause
"""Tests for equation-of-state functions in phonopy.qha.eos."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.qha.eos import fit_to_eos, get_eos

# Si E-V data (volume, energy), copied from test_QHA.ev_vs_v_Si.
ev_vs_v_Si = np.array(
    [
        [140.030000, -42.132246],
        [144.500000, -42.600974],
        [149.060000, -42.949142],
        [153.720000, -43.188162],
        [158.470000, -43.326751],
        [163.320000, -43.375124],
        [168.270000, -43.339884],
        [173.320000, -43.230619],
        [178.470000, -43.054343],
        [183.720000, -42.817825],
        [189.070000, -42.527932],
    ]
)

# Fitted [E_0, B_0, B'_0, V_0] for each EOS form, pinned for regression.
fitted_parameters = {
    "vinet": [-43.37462334, 0.55591331, 4.33034647, 163.63380396],
    "birch_murnaghan": [-43.37420061, 0.55384865, 4.31229001, 163.64029252],
    "murnaghan": [-43.37331571, 0.54951131, 4.26887110, 163.65572202],
}


def test_vinet_finite_for_negative_ratio() -> None:
    """Vinet EOS must stay finite when v / V_0 is negative.

    The least-squares optimizer can probe a negative V_0 (or negative volume
    ratio) while fitting, especially for high-pressure data. The real cube
    root keeps the result finite there, whereas a fractional power produces
    NaN. This is a regression guard for that fix.

    """
    vinet = get_eos("vinet")

    # p = [E_0, B_0, B'_0, V_0]; a negative V_0 makes v / p[3] negative.
    p = np.array([-43.0, 0.5, 4.0, -160.0])
    v = np.array([140.0, 160.0, 190.0])

    assert np.isfinite(vinet(v, p)).all()


def test_vinet_finite_for_negative_ratio_reported_data() -> None:
    """Vinet EOS stays finite for the data from the original bug report.

    The volumes and the negative V_0 below are the values from the report
    that motivated switching to the real cube root: while fitting, the
    optimizer probed a negative V_0 and a fractional power produced NaN.
    This is a regression guard tied to that specific report.

    """
    vinet = get_eos("vinet")

    volumes = np.array(
        [
            46.191144,
            53.889668,
            61.58819199,
            69.28671599,
            73.13597799,
            73.90583039,
            74.67568279,
            75.44553519,
            76.21538759,
            76.98523999,
            77.75509239,
            78.52494479,
            79.29479719,
            80.06464959,
            80.83450199,
            84.68376399,
            92.38228799,
        ]
    )
    # p = [E_0, B_0, B'_0, V_0]; the negative V_0 makes v / p[3] negative.
    p = np.array([-43.0, 0.5, 4.0, -0.18013697985445276])

    assert np.isfinite(vinet(volumes, p)).all()


def test_vinet_returns_E0_at_V0() -> None:
    """Vinet EOS reduces exactly to E_0 at v == V_0.

    At v == V_0 the scaled length x is 1, so the bracketed term vanishes and
    only p[0] (E_0) remains. This pins the analytic value and confirms the
    cube-root form leaves valid (positive-ratio) results unchanged.

    """
    vinet = get_eos("vinet")

    E_0 = -43.0
    p = np.array([E_0, 0.5, 4.0, 160.0])
    np.testing.assert_allclose(vinet(np.array([160.0]), p), [E_0])


@pytest.mark.parametrize("eos", ["vinet", "birch_murnaghan", "murnaghan"])
def test_fit_to_eos(eos: str) -> None:
    """Fitted EOS parameters of Si E-V data match pinned references."""
    volumes = ev_vs_v_Si[:, 0]
    energies = ev_vs_v_Si[:, 1]
    parameters = fit_to_eos(volumes, energies, get_eos(eos))
    np.testing.assert_allclose(parameters, fitted_parameters[eos], atol=1e-5)
