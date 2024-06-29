"""Tests for QHA calculations."""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from phonopy import PhonopyQHA

current_dir = Path(__file__).resolve().parent

ev_vs_v_Cu = np.array(
    [
        [43.0804791127649, -17.2788599300000],
        [43.9779889420447, -17.3222749000000],
        [44.8754989139109, -17.3433656900000],
        [45.7730090104272, -17.3447976000000],
        [46.6705189089890, -17.3284360400000],
        [47.5680287743940, -17.2967389600000],
        [48.4655386546724, -17.2508195400000],
        [49.3630487614241, -17.1926333700000],
        [50.2605586587267, -17.1235681600000],
        [51.1580685636184, -17.0446799700000],
        [52.0555787437377, -16.9575215500000],
    ]
)

temperatures_Cu = np.arange(0, 2501, 10)
tprop_file_Cu = current_dir / "tprop-Cu.dat"
cv_Cu, entropy_Cu, fe_phonon_Cu = np.loadtxt(tprop_file_Cu).reshape(3, 151, 11)
fe_file_Cu = current_dir / "fe-v-Cu.dat"
electronic_energies_Cu = np.loadtxt(fe_file_Cu)[:, 1:]

thermal_expansion_Cu = np.array(
    [
        [
            0.0,
            27.179674,
            40.768512,
            45.480682,
            48.33115,
            50.691166,
            52.940689,
            55.197348,
            57.520059,
            59.951488,
            62.524902,
            65.278122,
            68.245896,
            71.464227,
        ],
        [
            0.0,
            22.396147,
            34.193735,
            38.180049,
            40.432133,
            42.177188,
            43.775513,
            45.349463,
            46.948238,
            48.595796,
            50.313124,
            52.114894,
            54.022677,
            56.055905,
        ],
    ]
)
helmholtz_volume_Cu = np.array(
    [
        [
            -17.134247,
            -17.144045,
            -17.202472,
            -17.305176,
            -17.440419,
            -17.600875,
            -17.781832,
            -17.980047,
            -18.193167,
            -18.419404,
            -18.65736,
            -18.905908,
            -19.164119,
            -19.431221,
        ],
        [
            -15.789812,
            -15.79961,
            -15.858036,
            -15.960741,
            -16.095984,
            -16.25644,
            -16.437396,
            -16.635612,
            -16.848731,
            -17.074969,
            -17.312925,
            -17.561472,
            -17.819684,
            -18.086785,
        ],
    ]
)
volume_temperature_Cu = np.array(
    [
        [
            45.650459,
            45.699559,
            45.862073,
            46.061591,
            46.27849,
            46.50828,
            46.749907,
            47.003341,
            47.268955,
            47.54736,
            47.839362,
            48.145964,
            48.468368,
            48.808003,
        ],
        [
            44.366298,
            44.405143,
            44.536874,
            44.699481,
            44.875847,
            45.061687,
            45.255786,
            45.457904,
            45.668157,
            45.886822,
            46.114284,
            46.351024,
            46.597619,
            46.854742,
        ],
    ]
)
gibbs_temperature_Cu = np.array(
    [
        [
            -17.216711,
            -17.229086,
            -17.296568,
            -17.410934,
            -17.559394,
            -17.734382,
            -17.931134,
            -18.14642,
            -18.377923,
            -18.623907,
            -18.883029,
            -19.154228,
            -19.436652,
            -19.729607,
        ],
        [
            -15.812649,
            -15.823667,
            -15.886593,
            -15.995349,
            -16.137717,
            -16.306269,
            -16.496275,
            -16.704512,
            -16.928653,
            -17.16695,
            -17.418044,
            -17.68085,
            -17.954489,
            -18.238237,
        ],
    ]
)
bulkmodulus_temperature_Cu = np.array(
    [
        [
            163.552734,
            162.181213,
            158.511502,
            154.424793,
            150.197642,
            145.87202,
            141.465929,
            136.996641,
            132.478348,
            127.920638,
            123.328887,
            118.705581,
            114.051223,
            109.365776,
        ],
        [
            187.141941,
            186.082783,
            182.791708,
            178.986461,
            175.018232,
            170.947634,
            166.791905,
            162.566674,
            158.286532,
            153.963093,
            149.603604,
            145.211603,
            140.78879,
            136.335411,
        ],
    ]
)
cp_temperature_Cu = np.array(
    [
        [
            0.0,
            61.553521,
            89.257594,
            97.467186,
            101.547821,
            104.411954,
            106.877269,
            109.209632,
            111.527181,
            113.899965,
            116.384629,
            118.960247,
            121.879828,
            124.800692,
        ],
        [
            0.0,
            58.305685,
            87.488273,
            96.146922,
            100.228999,
            102.918331,
            105.111866,
            107.101014,
            109.088738,
            110.894673,
            112.806166,
            114.813926,
            116.967151,
            119.208311,
        ],
    ]
)
gruneisen_temperature_Cu = np.array(
    [
        [
            0.0,
            1.985396,
            2.042369,
            2.071046,
            2.094979,
            2.11868,
            2.142885,
            2.166831,
            2.190327,
            2.21355,
            2.236645,
            2.259986,
            2.283762,
            2.308092,
        ],
        [
            0.0,
            1.928324,
            1.951816,
            1.971588,
            1.989733,
            2.007841,
            2.026807,
            2.04642,
            2.06633,
            2.086262,
            2.106278,
            2.126338,
            2.146745,
            2.167684,
        ],
    ]
)


@pytest.mark.parametrize("pressure,index", [(None, 0), (5, 1)])
def test_QHA_Cu(pressure: Optional[float], index: int):
    """Test of QHA calculation by Cu."""
    indices = list(range(11))
    volumes = ev_vs_v_Cu[indices, 0]

    phonopy_qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=electronic_energies_Cu,
        eos="vinet",
        temperatures=temperatures_Cu,
        free_energy=fe_phonon_Cu[:, indices],
        cv=cv_Cu[:, indices],
        entropy=entropy_Cu[:, indices],
        pressure=pressure,
        t_max=1300,
        verbose=True,
    )
    t_indices = list(range(0, 131, 10))

    # Bulk modulus without phonon at 300K of electronic free energy
    bulk_moduli = [1.044907871562742, 1.1946818642018968]
    np.testing.assert_almost_equal(phonopy_qha.bulk_modulus[30], bulk_moduli[index])

    # Thermal expansion
    np.testing.assert_allclose(
        [phonopy_qha.thermal_expansion[i] * 1e6 for i in t_indices],
        thermal_expansion_Cu[index],
        atol=0.01,
    )

    # Helmholtz free energies vs volumes
    np.testing.assert_allclose(
        phonopy_qha.helmholtz_volume[t_indices, 0],
        helmholtz_volume_Cu[index],
        atol=1e-5,
    )

    # Volume vs temperature
    np.testing.assert_allclose(
        phonopy_qha.volume_temperature[t_indices],
        volume_temperature_Cu[index],
        atol=1e-5,
    )

    # Volume vs temperature
    np.testing.assert_allclose(
        phonopy_qha.gibbs_temperature[t_indices],
        gibbs_temperature_Cu[index],
        atol=1e-5,
    )

    # Bulk modulus vs temperature
    np.testing.assert_allclose(
        phonopy_qha.bulk_modulus_temperature[t_indices],
        bulkmodulus_temperature_Cu[index],
        atol=5e-4,
    )

    # Cp vs temperature by numerical second derivative
    np.testing.assert_allclose(
        np.array(phonopy_qha.heat_capacity_P_numerical)[t_indices],
        cp_temperature_Cu[index],
        atol=0.3,
    )

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        phonopy_qha.heat_capacity_P_polyfit  # noqa B018

    # Gruneisen parameters vs temperature
    np.testing.assert_allclose(
        np.array(phonopy_qha.gruneisen_temperature)[t_indices],
        gruneisen_temperature_Cu[index],
        atol=5e-4,
    )


def _print_values(values):
    print("%.7f," % values[0])
    for line in np.reshape(values[1:], (-1, 5)):
        print("".join(["%.7f, " % v for v in line]))
