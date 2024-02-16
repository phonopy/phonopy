"""Tests for QHA calculations."""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from phonopy import PhonopyQHA

current_dir = Path(__file__).resolve().parent

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
temperatures_Si = np.arange(0, 2101, 10)
tprop_file_Si = current_dir / "tprop-Si.dat"
cv_Si, entropy_Si, fe_phonon_Si = np.loadtxt(tprop_file_Si).reshape(3, 211, 11)

thermal_expansion_Si = np.array(
    [
        [
            0.0,
            -0.6332219,
            5.6139850,
            9.6750859,
            11.8141234,
            13.0844083,
            13.9458837,
            14.5977009,
            15.1336183,
            15.6020829,
            16.0296249,
        ],
        [
            0.0,
            -3.609581,
            0.603469,
            3.925014,
            5.727342,
            6.786175,
            7.483775,
            7.993026,
            8.39725,
            8.73977,
            9.044691,
        ],
    ]
)
helmholtz_volume_Si = np.array(
    [
        [
            -41.5839894,
            -41.6004724,
            -41.6770546,
            -41.8127769,
            -42.0001647,
            -42.2311973,
            -42.4992712,
            -42.7992502,
            -43.1271352,
            -43.4797635,
            -43.8545876,
        ],
        [
            -37.213999,
            -37.230482,
            -37.307064,
            -37.442786,
            -37.630174,
            -37.861207,
            -38.129281,
            -38.42926,
            -38.757145,
            -39.109773,
            -39.484597,
        ],
    ]
)
volume_temperature_Si = np.array(
    [
        [
            164.4548783,
            164.4442152,
            164.4847063,
            164.6142652,
            164.7929816,
            164.9990617,
            165.2226063,
            165.4587889,
            165.7050586,
            165.9599810,
            166.2227137,
        ],
        [
            156.244117,
            156.213533,
            156.187108,
            156.225134,
            156.301943,
            156.400428,
            156.512393,
            156.633737,
            156.762257,
            156.896699,
            157.036316,
        ],
    ]
)
gibbs_temperature_Si = np.array(
    [
        [
            -42.8932829,
            -42.9039937,
            -42.9721912,
            -43.1059496,
            -43.2954558,
            -43.5308843,
            -43.8047360,
            -44.1114190,
            -44.4466860,
            -44.8072307,
            -45.1904183,
        ],
        [
            -37.894953,
            -37.906312,
            -37.974345,
            -38.105573,
            -38.291188,
            -38.521965,
            -38.790687,
            -39.091899,
            -39.421427,
            -39.776008,
            -40.153031,
        ],
    ]
)
bulkmodulus_temperature_Si = np.array(
    [
        [
            87.4121501,
            87.2126795,
            86.5084539,
            85.5863262,
            84.5997708,
            83.5933127,
            82.5823028,
            81.5733203,
            80.5697051,
            79.5733964,
            78.5856509,
        ],
        [
            108.270658,
            107.969426,
            107.10972,
            106.024551,
            104.868627,
            103.688565,
            102.500301,
            101.310521,
            100.122537,
            98.938209,
            97.758706,
        ],
    ]
)
cp_temperature_Si = np.array(
    [
        [
            0.0000000,
            61.6615825,
            128.3828570,
            161.0031288,
            176.4325115,
            184.6087521,
            189.4345190,
            192.5460752,
            194.6985008,
            196.2812400,
            197.5052927,
        ],
        [
            0.0,
            60.243263,
            123.636837,
            157.210244,
            173.595065,
            182.34616,
            187.486514,
            190.752806,
            192.956457,
            194.539917,
            195.71405,
        ],
    ]
)
cp_temperature_polyfit_Si = np.array(
    [
        [
            0.0000000,
            61.7161021,
            128.3966796,
            160.9982814,
            176.4240892,
            184.6003622,
            189.4249754,
            192.5323933,
            194.6826330,
            196.2629828,
            197.4862251,
        ],
        [
            0.0,
            60.273522,
            123.650818,
            157.211247,
            173.596217,
            182.352784,
            187.498509,
            190.768773,
            192.98318,
            194.563187,
            195.741693,
        ],
    ]
)
gruneisen_temperature_Si = np.array(
    [
        [
            0.0000000,
            -0.0886154,
            0.3748304,
            0.5106203,
            0.5637079,
            0.5910095,
            0.6080069,
            0.6201481,
            0.6297201,
            0.6378298,
            0.6450508,
        ],
        [
            0.0,
            -0.608397,
            0.049168,
            0.249118,
            0.325926,
            0.363913,
            0.386311,
            0.401342,
            0.412475,
            0.421374,
            0.428913,
        ],
    ]
)


@pytest.mark.parametrize("pressure,index", [(None, 0), (5, 1)])
def test_QHA_Si(pressure: Optional[float], index: int):
    """Test of QHA calculation by Si."""
    indices = list(range(11))
    volumes = ev_vs_v_Si[indices, 0]
    electronic_energies = ev_vs_v_Si[indices, 1]

    phonopy_qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=electronic_energies,
        eos="vinet",
        temperatures=temperatures_Si,
        free_energy=fe_phonon_Si[:, indices],
        cv=cv_Si[:, indices],
        entropy=entropy_Si[:, indices],
        pressure=pressure,
        t_max=1000,
        verbose=True,
    )
    t_indices = list(range(0, 101, 10))

    # Bulk modulus without phonon
    bulk_moduli = [0.5559133052877888, 0.6865238809710272]
    np.testing.assert_almost_equal(phonopy_qha.bulk_modulus, bulk_moduli[index])

    # Thermal expansion
    np.testing.assert_allclose(
        [phonopy_qha.thermal_expansion[i] * 1e6 for i in t_indices],
        thermal_expansion_Si[index],
        atol=0.01,
    )

    # Helmholtz free energies vs volumes
    np.testing.assert_allclose(
        phonopy_qha.helmholtz_volume[t_indices, 0],
        helmholtz_volume_Si[index],
        atol=1e-5,
    )

    # Volume vs temperature
    np.testing.assert_allclose(
        phonopy_qha.volume_temperature[t_indices],
        volume_temperature_Si[index],
        atol=1e-5,
    )

    # Volume vs temperature
    np.testing.assert_allclose(
        phonopy_qha.gibbs_temperature[t_indices], gibbs_temperature_Si[index], atol=1e-5
    )

    # Bulk modulus vs temperature
    np.testing.assert_allclose(
        phonopy_qha.bulk_modulus_temperature[t_indices],
        bulkmodulus_temperature_Si[index],
        atol=1e-5,
    )

    # Cp vs temperature by numerical second derivative
    np.testing.assert_allclose(
        np.array(phonopy_qha.heat_capacity_P_numerical)[t_indices],
        cp_temperature_Si[index],
        atol=0.02,
    )

    # Cp vs temperature by polynomial fittings of Cv and S
    np.testing.assert_allclose(
        np.array(phonopy_qha.heat_capacity_P_polyfit)[t_indices],
        cp_temperature_polyfit_Si[index],
        atol=1e-5,
    )

    # Gruneisen parameters vs temperature
    np.testing.assert_allclose(
        np.array(phonopy_qha.gruneisen_temperature)[t_indices],
        gruneisen_temperature_Si[index],
        atol=1e-5,
    )


def _print_values(values):
    print("%.7f," % values[0])
    for line in np.reshape(values[1:], (-1, 5)):
        print("".join(["%.7f, " % v for v in line]))
