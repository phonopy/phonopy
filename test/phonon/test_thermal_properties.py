"""Tests for thermal property calculation."""

import numpy as np

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


def _test_thermal_properties(ph):
    tp = ph.thermal_properties

    # for vals in tp.thermal_properties:
    #     print(", ".join(["%.6f" % v for v in vals]))

    for i in range(2):
        if i == 1:
            tp.run(lang="py")
        for vals_ref, vals in zip((temps, fes, entropies, cvs), tp.thermal_properties):
            np.testing.assert_allclose(vals_ref, vals, atol=1e-5)
