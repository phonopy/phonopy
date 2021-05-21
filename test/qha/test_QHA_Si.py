import numpy as np
from phonopy import PhonopyQHA

ev_vs_v = np.array([[140.030000, -42.132246],
                    [144.500000, -42.600974],
                    [149.060000, -42.949142],
                    [153.720000, -43.188162],
                    [158.470000, -43.326751],
                    [163.320000, -43.375124],
                    [168.270000, -43.339884],
                    [173.320000, -43.230619],
                    [178.470000, -43.054343],
                    [183.720000, -42.817825],
                    [189.070000, -42.527932]])
temperatures = np.arange(0, 2101, 10)
cv, entropy, fe_phonon = np.loadtxt("tprop.dat").reshape(3, 211, 11)

thermal_expansion = np.array([
    0.0, -0.6332219, 5.6139850, 9.6750859, 11.8141234, 13.0844083,
    13.9458837, 14.5977009, 15.1336183, 15.6020829, 16.0296249])
helmholtz_volume = np.array([
    -41.5839894,
    -41.6004724, -41.6770546, -41.8127769, -42.0001647, -42.2311973,
    -42.4992712, -42.7992502, -43.1271352, -43.4797635, -43.8545876])


def test_QHA_Si():
    indices = list(range(11))
    phonopy_qha = PhonopyQHA(volumes=ev_vs_v[indices, 0],
                             electronic_energies=ev_vs_v[indices, 1],
                             eos="vinet",
                             temperatures=temperatures,
                             free_energy=fe_phonon[:, indices],
                             cv=cv[:, indices],
                             entropy=entropy[:, indices],
                             t_max=1000,
                             verbose=True)
    t_indices = list(range(0, 101, 10))

    # Bulk modulus without phonon
    np.testing.assert_almost_equal(phonopy_qha.bulk_modulus,
                                   0.5559133052877888)
    # Thermal expansion
    np.testing.assert_allclose(
        [phonopy_qha.thermal_expansion[i] for i in t_indices],
        thermal_expansion * 1e-6,
        atol=1e-5)

    # Helmholtz free energies vs volumes
    np.testing.assert_allclose(
        phonopy_qha.helmholtz_volume[t_indices, 0],
        helmholtz_volume,
        atol=1e-5)

    # print_values()


def print_values(values):
    print("%.7f," % values[0])
    for line in np.reshape(values[1:], (-1, 5)):
        print("".join(["%.7f, " % v for v in line]))
