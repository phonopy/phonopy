import numpy as np

temps = [0.000000, 100.000000, 200.000000, 300.000000, 400.000000, 500.000000,
         600.000000, 700.000000, 800.000000, 900.000000]
fes = [4.856522, 3.797312, -0.522774, -7.187512, -15.473942,
       -24.989474, -35.492709, -46.820961, -58.857349, -71.513905]
entropies = [0.000000, 27.583497, 56.558614, 75.597115, 89.503449,
             100.415568, 109.383008, 116.989870, 123.592790, 129.424960]
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
