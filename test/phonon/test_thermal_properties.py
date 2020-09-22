import numpy as np

temps = [0.000000, 100.000000, 200.000000, 300.000000, 400.000000, 500.000000,
         600.000000, 700.000000, 800.000000, 900.000000]
fes = [4.856522, 3.795742, -0.525913, -7.192220, -15.480220, -24.997321,
       -35.502125, -46.831946, -58.869904, -71.528029]
entropies = [0.000000, 27.599190, 56.574307, 75.612808, 89.519142,
             100.431261, 109.398701, 117.005563, 123.608484, 129.440653]
cvs = [0.000000, 36.273760, 45.739877, 47.905271, 48.700700,
       49.075806, 49.281446, 49.406086, 49.487243, 49.543004]


def test_thermal_properties(ph_nacl):
    ph_nacl.run_mesh([5, 5, 5])
    ph_nacl.run_thermal_properties(t_step=100, t_max=900)
    tp = ph_nacl.thermal_properties

    for vals in tp.thermal_properties:
        print(", ".join(["%.6f" % v for v in vals]))

    for vals_ref, vals in zip(
            (temps, fes, entropies, cvs), tp.thermal_properties):
        np.testing.assert_allclose(vals_ref, vals, atol=1e-5)


def test_thermal_properties_at_temperatues(ph_nacl):
    ph_nacl.run_mesh([5, 5, 5])
    temperatures = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    ph_nacl.run_thermal_properties(temperatures=temperatures)
    tp = ph_nacl.thermal_properties

    for vals in tp.thermal_properties:
        print(", ".join(["%.6f" % v for v in vals]))

    for vals_ref, vals in zip(
            (temps, fes, entropies, cvs), tp.thermal_properties):
        np.testing.assert_allclose(vals_ref, vals, atol=1e-5)
