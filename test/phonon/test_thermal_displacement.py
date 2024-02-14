"""Tests for thermal displacement calculations."""

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
td_ref = np.array(
    [
        [0.00571624, 0.00571624, 0.00571624, 0.00403776, 0.00403776, 0.00403776],
        [0.00877353, 0.00877353, 0.00877353, 0.00654962, 0.00654962, 0.00654962],
        [0.01513305, 0.01513305, 0.01513305, 0.01151749, 0.01151749, 0.01151749],
        [0.02198817, 0.02198817, 0.02198817, 0.01681392, 0.01681392, 0.01681392],
        [0.02898107, 0.02898107, 0.02898107, 0.02220032, 0.02220032, 0.02220032],
        [0.03603064, 0.03603064, 0.03603064, 0.02762357, 0.02762357, 0.02762357],
        [0.04310888, 0.04310888, 0.04310888, 0.03306543, 0.03306543, 0.03306543],
        [0.05020359, 0.05020359, 0.05020359, 0.03851798, 0.03851798, 0.03851798],
        [0.05730864, 0.05730864, 0.05730864, 0.04397723, 0.04397723, 0.04397723],
        [0.06442059, 0.06442059, 0.06442059, 0.04944096, 0.04944096, 0.04944096],
    ]
)

tdm_ref = np.array(
    [
        [0.00334373, 0.00137025, -0.00000000],
        [0.00137025, 0.00334373, -0.00000000],
        [-0.00000000, -0.00000000, 0.00237467],
        [0.00334373, 0.00137025, -0.00000000],
        [0.00137025, 0.00334373, -0.00000000],
        [-0.00000000, -0.00000000, 0.00237467],
        [0.00334373, -0.00137025, -0.00000000],
        [-0.00137025, 0.00334373, -0.00000000],
        [-0.00000000, -0.00000000, 0.00237467],
        [0.00334373, -0.00137025, -0.00000000],
        [-0.00137025, 0.00334373, -0.00000000],
        [-0.00000000, -0.00000000, 0.00237467],
        [0.00093643, 0.00007622, -0.00000000],
        [0.00007622, 0.00093643, -0.00000000],
        [-0.00000000, -0.00000000, 0.00075837],
        [0.00093643, -0.00007622, -0.00000000],
        [-0.00007622, 0.00093643, -0.00000000],
        [-0.00000000, -0.00000000, 0.00075837],
        [0.01070050, 0.00604871, -0.00000000],
        [0.00604871, 0.01070050, -0.00000000],
        [-0.00000000, -0.00000000, 0.00584587],
        [0.01070050, 0.00604871, -0.00000000],
        [0.00604871, 0.01070050, -0.00000000],
        [-0.00000000, -0.00000000, 0.00584587],
        [0.01070050, -0.00604871, -0.00000000],
        [-0.00604871, 0.01070050, -0.00000000],
        [-0.00000000, -0.00000000, 0.00584587],
        [0.01070050, -0.00604871, -0.00000000],
        [-0.00604871, 0.01070050, -0.00000000],
        [-0.00000000, -0.00000000, 0.00584587],
        [0.00530410, 0.00071531, -0.00000000],
        [0.00071531, 0.00530410, -0.00000000],
        [-0.00000000, -0.00000000, 0.00378992],
        [0.00530411, -0.00071531, -0.00000000],
        [-0.00071531, 0.00530411, -0.00000000],
        [-0.00000000, -0.00000000, 0.00378992],
    ]
)


def test_ThermalDisplacements(ph_nacl):
    """Test for ThermalDisplacements."""
    ph_nacl.init_mesh(
        [5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False, use_iter_mesh=True
    )

    # t_min, t_max, t_step
    ph_nacl.run_thermal_displacements(t_min=0, t_max=901, t_step=100, freq_min=1e-2)
    # for td_t in td.thermal_displacements:
    #     print(", ".join(["%10.8f" % x for x in td_t]))
    td = ph_nacl.thermal_displacements
    np.testing.assert_allclose(td_ref, td.thermal_displacements, atol=1e-5)

    # temperatures
    temperatures = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    ph_nacl.run_thermal_displacements(temperatures=temperatures, freq_min=1e-2)
    td = ph_nacl.thermal_displacements
    np.testing.assert_allclose(td_ref, td.thermal_displacements, atol=1e-5)

    # direction
    ph_nacl.run_thermal_displacements(
        temperatures=temperatures, direction=[1, 0, 0], freq_min=1e-2
    )
    td = ph_nacl.thermal_displacements
    np.testing.assert_allclose(td_ref[:, [0, 3]], td.thermal_displacements, atol=1e-5)


def test_ThermalDisplacementMatrices(ph_sno2):
    """Test for ThermalDisplacementMatrices."""
    ph_sno2.init_mesh(
        [5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False, use_iter_mesh=True
    )
    # t_min, t_max, t_step
    temperatures = [0, 500]
    ph_sno2.run_thermal_displacement_matrices(
        t_min=0, t_max=501, t_step=500, freq_min=1e-2
    )
    tdm = ph_sno2.thermal_displacement_matrices
    mat = tdm.thermal_displacement_matrices
    # for v in mat.reshape(-1, 3):
    #     print("[%11.8f, %11.8f, %11.8f]," % tuple(v))
    np.testing.assert_allclose(tdm_ref, mat.reshape(-1, 3), atol=1e-5)

    # temperatures
    temperatures = [0, 500]
    ph_sno2.run_thermal_displacement_matrices(temperatures=temperatures, freq_min=1e-2)
    tdm = ph_sno2.thermal_displacement_matrices
    mat = tdm.thermal_displacement_matrices
    np.testing.assert_allclose(tdm_ref, mat.reshape(-1, 3), atol=1e-5)
