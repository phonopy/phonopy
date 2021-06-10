import os
import numpy as np
from phonopy import PhonopyQHA

current_dir = os.path.dirname(os.path.abspath(__file__))

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
tprop_file = os.path.join(current_dir, "tprop.dat")
cv, entropy, fe_phonon = np.loadtxt(tprop_file).reshape(3, 211, 11)

thermal_expansion = np.array([
    0.0, -0.6332219, 5.6139850, 9.6750859, 11.8141234, 13.0844083,
    13.9458837, 14.5977009, 15.1336183, 15.6020829, 16.0296249])
helmholtz_volume = np.array([
    -41.5839894,
    -41.6004724, -41.6770546, -41.8127769, -42.0001647, -42.2311973,
    -42.4992712, -42.7992502, -43.1271352, -43.4797635, -43.8545876])
volume_temperature = np.array([
    164.4548783,
    164.4442152, 164.4847063, 164.6142652, 164.7929816, 164.9990617,
    165.2226063, 165.4587889, 165.7050586, 165.9599810, 166.2227137])
gibbs_temperature = np.array([
    -42.8932829,
    -42.9039937, -42.9721912, -43.1059496, -43.2954558, -43.5308843,
    -43.8047360, -44.1114190, -44.4466860, -44.8072307, -45.1904183])
bulkmodulus_temperature = np.array([
    87.4121501,
    87.2126795, 86.5084539, 85.5863262, 84.5997708, 83.5933127,
    82.5823028, 81.5733203, 80.5697051, 79.5733964, 78.5856509])
cp_temperature = np.array([
    0.0000000,
    61.6615825, 128.3828570, 161.0031288, 176.4325115, 184.6087521,
    189.4345190, 192.5460752, 194.6985008, 196.2812400, 197.5052927])
cp_temperature_polyfit = np.array([
    0.0000000,
    61.7161021, 128.3966796, 160.9982814, 176.4240892, 184.6003622,
    189.4249754, 192.5323933, 194.6826330, 196.2629828, 197.4862251])
gruneisen_temperature = np.array([
    0.0000000,
    -0.0886154, 0.3748304, 0.5106203, 0.5637079, 0.5910095,
    0.6080069, 0.6201481, 0.6297201, 0.6378298, 0.6450508])


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

    # Volume vs temperature
    np.testing.assert_allclose(
        phonopy_qha.volume_temperature[t_indices],
        volume_temperature,
        atol=1e-5)

    # Volume vs temperature
    np.testing.assert_allclose(
        phonopy_qha.gibbs_temperature[t_indices],
        gibbs_temperature,
        atol=1e-5)

    # Bulk modulus vs temperature
    np.testing.assert_allclose(
        phonopy_qha.bulk_modulus_temperature[t_indices],
        bulkmodulus_temperature,
        atol=1e-5)

    # Cp vs temperature by numerical second derivative
    np.testing.assert_allclose(
        np.array(phonopy_qha.heat_capacity_P_numerical)[t_indices],
        cp_temperature,
        atol=0.01)

    # Cp vs temperature by polynomial fittings of Cv and S
    np.testing.assert_allclose(
        np.array(phonopy_qha.heat_capacity_P_polyfit)[t_indices],
        cp_temperature_polyfit,
        atol=1e-5)

    # Gruneisen parameters vs temperature
    np.testing.assert_allclose(
        np.array(phonopy_qha.gruneisen_temperature)[t_indices],
        gruneisen_temperature,
        atol=1e-5)


def print_values(values):
    print("%.7f," % values[0])
    for line in np.reshape(values[1:], (-1, 5)):
        print("".join(["%.7f, " % v for v in line]))
