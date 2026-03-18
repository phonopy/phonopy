"""Tests for thermal displacement calculations."""

from pathlib import Path

import numpy as np
import pytest

from phonopy import Phonopy

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


def test_compare_TD_and_TDM(ph_nacl: Phonopy, ph_sno2: Phonopy):
    """Test for comparing ThermalDisplacements and ThermalDisplacementMatrices."""
    for ph in (ph_nacl, ph_sno2):
        ph.init_mesh(
            [5, 5, 5],
            with_eigenvectors=True,
            is_mesh_symmetry=False,
            use_iter_mesh=True,
        )
        temperatures = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        ph.run_thermal_displacements(temperatures=temperatures, freq_min=1e-2)
        td = ph.thermal_displacements
        ph.run_thermal_displacement_matrices(temperatures=temperatures, freq_min=1e-2)
        tdm = ph.thermal_displacement_matrices

        assert td is not None
        assert tdm is not None
        assert td.thermal_displacements is not None
        assert tdm.thermal_displacement_matrices is not None
        td_from_tdm = np.zeros(
            td.thermal_displacements.shape, dtype=td.thermal_displacements.dtype
        )
        for i, tdm_t in enumerate(tdm.thermal_displacement_matrices):
            for j, tdm_t_n in enumerate(tdm_t):
                td_from_tdm[i, j * 3 : (j + 1) * 3] = np.diag(tdm_t_n)

        np.testing.assert_allclose(td.thermal_displacements, td_from_tdm, atol=1e-8)


# Reference displacements (ux, uy, uz) per atom for NaCl (Na, Cl).
# Shape: (temperatures, atoms, 3).  Temperatures: [0, 300].
_td_yaml_ref = np.array(
    [
        [  # T=0
            [0.0057162, 0.0057162, 0.0057162],  # Na
            [0.0040378, 0.0040378, 0.0040378],  # Cl
        ],
        [  # T=300
            [0.0219882, 0.0219882, 0.0219882],  # Na
            [0.0168139, 0.0168139, 0.0168139],  # Cl
        ],
    ]
)


def test_ThermalDisplacements_write_yaml(ph_nacl: Phonopy, tmp_path: Path) -> None:
    """Regression test for ThermalDisplacements.write_yaml."""
    import yaml

    ph_nacl.init_mesh(
        [5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False, use_iter_mesh=True
    )
    ph_nacl.run_thermal_displacements(temperatures=[0, 300], freq_min=1e-2)
    td = ph_nacl.thermal_displacements
    assert td is not None

    yaml_file = tmp_path / "thermal_displacements.yaml"
    td.write_yaml(filename=yaml_file)
    assert yaml_file.exists()

    data = yaml.safe_load(yaml_file.read_text())
    assert data["natom"] == 2
    assert data["freq_min"] == pytest.approx(0.01)

    td_data = data["thermal_displacements"]
    assert len(td_data) == 2
    assert td_data[0]["temperature"] == pytest.approx(0.0)
    assert td_data[1]["temperature"] == pytest.approx(300.0)

    for i_t, block in enumerate(td_data):
        disps = np.array(block["displacements"])
        np.testing.assert_allclose(disps, _td_yaml_ref[i_t], atol=1e-5)


# Reference values (U11, U22, U33, U23, U13, U12) per atom for SnO2.
# Shape: (temperatures, atoms, 6).  Temperatures: [0, 300].
_yaml_ref = np.array(
    [
        [  # T=0
            [0.00334373, 0.00334373, 0.00237467, 0.00000000, 0.00000000, 0.00137025],
            [0.00334373, 0.00334373, 0.00237467, 0.00000000, 0.00000000, 0.00137025],
            [0.00334373, 0.00334373, 0.00237467, 0.00000000, 0.00000000, -0.00137025],
            [0.00334373, 0.00334373, 0.00237467, 0.00000000, 0.00000000, -0.00137025],
            [0.00093643, 0.00093643, 0.00075837, 0.00000000, 0.00000000, 0.00007622],
            [0.00093643, 0.00093643, 0.00075837, 0.00000000, 0.00000000, -0.00007622],
        ],
        [  # T=300
            [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, 0.00366419],
            [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, 0.00366419],
            [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, -0.00366419],
            [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, -0.00366419],
            [0.00325322, 0.00325322, 0.00234390, 0.00000000, 0.00000000, 0.00042936],
            [0.00325322, 0.00325322, 0.00234390, 0.00000000, 0.00000000, -0.00042936],
        ],
    ]
)


def test_ThermalDisplacementMatrices_write_yaml(
    ph_sno2: Phonopy, tmp_path: Path
) -> None:
    """Regression test for ThermalDisplacementMatrices.write_yaml."""
    import yaml

    ph_sno2.init_mesh(
        [5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False, use_iter_mesh=True
    )
    ph_sno2.run_thermal_displacement_matrices(temperatures=[0, 300], freq_min=1e-2)
    tdm = ph_sno2.thermal_displacement_matrices
    assert tdm is not None

    yaml_file = tmp_path / "thermal_displacement_matrices.yaml"
    tdm.write_yaml(filename=yaml_file)
    assert yaml_file.exists()

    data = yaml.safe_load(yaml_file.read_text())
    assert data["natom"] == 6
    assert data["freq_min"] == pytest.approx(0.01)

    tdm_data = data["thermal_displacement_matrices"]
    assert len(tdm_data) == 2
    assert tdm_data[0]["temperature"] == pytest.approx(0.0)
    assert tdm_data[1]["temperature"] == pytest.approx(300.0)

    for i_t, block in enumerate(tdm_data):
        mats = np.array(block["displacement_matrices"])
        np.testing.assert_allclose(np.abs(mats), np.abs(_yaml_ref[i_t]), atol=1e-5)
        mats_cif = np.array(block["displacement_matrices_cif"])
        np.testing.assert_allclose(np.abs(mats_cif), np.abs(_yaml_ref[i_t]), atol=1e-5)


# Reference U_cif values (U11, U22, U33, U23, U13, U12) at T=300 for SnO2.
# Atom order: O1, O2, O3, O4, Sn1, Sn2 (primitive cell).
_cif_ref = np.array(
    [
        [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, 0.00366419],
        [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, 0.00366419],
        [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, -0.00366419],
        [0.00690715, 0.00690715, 0.00398393, 0.00000000, 0.00000000, -0.00366419],
        [0.00325322, 0.00325322, 0.00234390, 0.00000000, 0.00000000, 0.00042936],
        [0.00325322, 0.00325322, 0.00234390, 0.00000000, 0.00000000, -0.00042936],
    ]
)


def test_ThermalDisplacementMatrices_write_cif(
    ph_sno2: Phonopy, tmp_path: Path
) -> None:
    """Regression test for ThermalDisplacementMatrices.write_cif."""
    ph_sno2.init_mesh(
        [5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False, use_iter_mesh=True
    )
    ph_sno2.run_thermal_displacement_matrices(temperatures=[300], freq_min=1e-2)
    tdm = ph_sno2.thermal_displacement_matrices
    assert tdm is not None

    # Check that thermal_displacement_matrices_cif matches reference.
    cif_mats = tdm.thermal_displacement_matrices_cif
    assert cif_mats is not None
    for i, m in enumerate(cif_mats[0]):
        vals = np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])
        np.testing.assert_allclose(np.abs(vals), np.abs(_cif_ref[i]), atol=1e-5)

    # Check that the CIF file is written and contains the expected aniso U values.
    cif_file = tmp_path / "tdispmat.cif"
    tdm.write_cif(ph_sno2.primitive, 0, filename=cif_file)
    assert cif_file.exists()

    lines = cif_file.read_text().splitlines()
    aniso_lines = [
        ln.split()
        for ln in lines
        if ln.strip()
        and ln.strip()[0].isalpha()
        and "_atom_site_aniso" not in ln
        and "loop_" not in ln
        and ln.strip().split()[0][0] in ("O", "S")
        and len(ln.split()) == 7
    ]
    assert len(aniso_lines) == 6
    for i, parts in enumerate(aniso_lines):
        vals = np.array([float(x) for x in parts[1:]])
        np.testing.assert_allclose(np.abs(vals), np.abs(_cif_ref[i]), atol=1e-4)
