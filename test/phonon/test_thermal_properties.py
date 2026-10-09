# SPDX-License-Identifier: BSD-3-Clause
"""Tests for thermal property calculation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy._lang import have_c_ext
from phonopy.phonon.thermal_properties import ThermalProperties

temps = [
    0.000000,
    100.000000,
    200.000000,
    300.000000,
    400.000000,
    500.000000,
    600.000000,
    700.000000,
    800.000000,
    900.000000,
]
fes = [
    4.856522,
    3.916257,
    -0.275662,
    -6.808753,
    -14.961276,
    -24.341220,
    -34.707527,
    -45.897738,
    -57.795132,
    -70.311860,
]
entropies = [
    0.000000,
    26.327525,
    55.256537,
    74.268068,
    88.155267,
    99.052543,
    108.007856,
    115.604464,
    122.198503,
    128.022838,
]
cvs = [
    0.000000,
    36.207244,
    45.673361,
    47.838756,
    48.634184,
    49.009290,
    49.214930,
    49.339570,
    49.420727,
    49.476488,
]


def test_thermal_properties(ph_nacl):
    """Test thermal property calculation with t_step and t_max parameters."""
    ph_nacl.run_mesh([5, 5, 5])
    ph_nacl.run_thermal_properties(t_step=100, t_max=900, cutoff_frequency=1e-5)
    _test_thermal_properties(ph_nacl)


def test_thermal_properties_at_temperatues(ph_nacl):
    """Test thermal property calculation with temperatures parameter."""
    ph_nacl.run_mesh([5, 5, 5])
    temperatures = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    ph_nacl.run_thermal_properties(temperatures=temperatures, cutoff_frequency=1e-5)
    _test_thermal_properties(ph_nacl)


def test_thermal_properties_individual_properties(ph_nacl):
    """Test free_energy, entropy, and heat_capacity properties."""
    ph_nacl.run_mesh([5, 5, 5])
    ph_nacl.run_thermal_properties(t_step=100, t_max=900, cutoff_frequency=1e-5)
    tp = ph_nacl.thermal_properties
    np.testing.assert_allclose(tp.temperatures, temps, atol=1e-5)
    np.testing.assert_allclose(tp.free_energy, fes, atol=1e-5)
    np.testing.assert_allclose(tp.entropy, entropies, atol=1e-5)
    np.testing.assert_allclose(tp.heat_capacity, cvs, atol=1e-5)


LOW_TEMPERATURES = [0.01, 0.1, 1.0, 10.0, 300.0]


def _run_at(ph: Phonopy, lang: str) -> tuple[np.ndarray, ...]:
    tp = ThermalProperties(ph.mesh, cutoff_frequency=1e-5, lang=lang)
    tp.temperatures = LOW_TEMPERATURES
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tp.run()
    return tp.thermal_properties[1:]


def test_thermal_properties_at_low_temperature(ph_nacl):
    """Test that hnu / k_B T in the thousands gives finite, vanishing values.

    At 0.01 and 0.1 K, exp(hnu / k_B T) overflows for every mode of NaCl.
    The heat capacity and the entropy were NaN there; they are now finite,
    non-negative and increase with temperature.

    """
    ph_nacl.run_mesh([5, 5, 5])
    fe, entropy, cv = _run_at(ph_nacl, "Rust")
    for vals in (fe, entropy, cv):
        assert np.all(np.isfinite(vals))
    assert np.all(entropy >= 0)
    assert np.all(cv >= 0)
    assert np.all(np.diff(entropy) > 0)
    assert np.all(np.diff(cv) > 0)
    np.testing.assert_allclose(cv[-1], 47.838756, atol=1e-5)


@pytest.mark.skipif(not have_c_ext(), reason="the C extension is not built")
def test_thermal_properties_at_low_temperature_c_matches_python(ph_nacl):
    """Test that the C path agrees with the Python path down to 0.01 K."""
    ph_nacl.run_mesh([5, 5, 5])
    for vals_c, vals_py in zip(
        _run_at(ph_nacl, "C"), _run_at(ph_nacl, "Rust"), strict=True
    ):
        np.testing.assert_allclose(vals_c, vals_py, rtol=1e-10, atol=1e-300)


def _test_thermal_properties(ph: Phonopy):
    tp = ph.thermal_properties

    # for vals in tp.thermal_properties:
    #     print(", ".join(["%.6f" % v for v in vals]))

    for i in range(2):
        if i == 1:
            tp.run(lang="Python")
        for vals_ref, vals in zip(
            (temps, fes, entropies, cvs), tp.thermal_properties, strict=True
        ):
            np.testing.assert_allclose(vals_ref, vals, atol=1e-5)
