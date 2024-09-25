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
    0.000000,
    3.094415,
    -0.699291,
    -7.092875,
    -15.174823,
    -24.512227,
    -34.850111,
    -46.019993,
    -57.902128,
    -70.406982,
]
entropies = [
    0.000000,
    18.743138,
    53.183741,
    73.330203,
    87.624345,
    98.711742,
    107.770806,
    115.430135,
    122.064948,
    127.917267,
]
cvs = [
    0.000000,
    49.687287,
    49.687287,
    49.687287,
    49.687287,
    49.687287,
    49.687287,
    49.687287,
    49.687287,
    49.687287,
]


def test_thermal_properties(ph_nacl):
    """Test thermal property calculation with t_step and t_max parameters."""
    ph_nacl.run_mesh([5, 5, 5])
    ph_nacl.run_thermal_properties(
        t_step=100, t_max=900, cutoff_frequency=1e-5, classical=True
    )
    _test_thermal_properties(ph_nacl)


def test_thermal_properties_at_temperatues(ph_nacl):
    """Test thermal property calculation with temperatures parameter."""
    ph_nacl.run_mesh([5, 5, 5])
    temperatures = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    ph_nacl.run_thermal_properties(
        temperatures=temperatures, cutoff_frequency=1e-5, classical=True
    )
    _test_thermal_properties(ph_nacl)


def _test_thermal_properties(ph):
    tp = ph.thermal_properties

    for vals in tp.thermal_properties:
        print(",\n".join(["%.6f" % v for v in vals]))

    for i in range(2):
        if i == 1:
            tp.run(lang="py")
        for vals_ref, vals in zip((temps, fes, entropies, cvs), tp.thermal_properties):
            np.testing.assert_allclose(vals_ref, vals, atol=1e-5)
