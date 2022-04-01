"""Tests of PhonopyYaml."""
import os
from io import StringIO

import numpy as np
import yaml

from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.vasp import read_vasp
from phonopy.structure.dataset import get_displacements_and_forces

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_poscar_yaml():
    """Test to parse PhonopyAtoms.__str__ output."""
    filename = os.path.join(data_dir, "NaCl-vasp.yaml")
    cell = _get_unitcell(filename)
    _compare(cell)


def test_read_phonopy_yaml():
    """Test to parse phonopy.yaml like file."""
    filename = os.path.join(data_dir, "phonopy.yaml")
    cell = _get_unitcell(filename)
    _compare(cell)


def test_write_phonopy_yaml(ph_nacl_nofcsym: Phonopy, helper_methods):
    """Test PhonopyYaml.set_phonon_info, __str__, yaml_data, parse."""
    phonon = ph_nacl_nofcsym
    phpy_yaml = PhonopyYaml(calculator="vasp")
    phpy_yaml.set_phonon_info(phonon)
    phpy_yaml_test = PhonopyYaml()
    phpy_yaml_test.yaml_data = yaml.safe_load(StringIO(str(phpy_yaml)))
    phpy_yaml_test.parse()
    helper_methods.compare_cells_with_order(
        phpy_yaml.primitive, phpy_yaml_test.primitive
    )
    helper_methods.compare_cells_with_order(phpy_yaml.unitcell, phpy_yaml_test.unitcell)
    helper_methods.compare_cells_with_order(
        phpy_yaml.supercell, phpy_yaml_test.supercell
    )
    assert phpy_yaml.version == phpy_yaml_test.version
    np.testing.assert_allclose(
        phpy_yaml.supercell_matrix, phpy_yaml_test.supercell_matrix, atol=1e-8
    )
    np.testing.assert_allclose(
        phpy_yaml.primitive_matrix, phpy_yaml_test.primitive_matrix, atol=1e-8
    )


def test_write_phonopy_yaml_extra(ph_nacl_nofcsym: Phonopy):
    """Test PhonopyYaml.set_phonon_info, __str__, yaml_data, parse.

    settings parameter controls amount of yaml output. In this test,
    more data than the default are dumped and those are tested.

    """
    phonon = ph_nacl_nofcsym
    settings = {
        "force_sets": True,
        "displacements": True,
        "force_constants": True,
        "born_effective_charge": True,
        "dielectric_constant": True,
    }
    phpy_yaml = PhonopyYaml(calculator="vasp", settings=settings)
    phpy_yaml.set_phonon_info(phonon)
    phpy_yaml_test = PhonopyYaml()
    phpy_yaml_test.yaml_data = yaml.safe_load(StringIO(str(phpy_yaml)))
    phpy_yaml_test.parse()
    np.testing.assert_allclose(
        phpy_yaml.force_constants, phpy_yaml_test.force_constants, atol=1e-8
    )
    np.testing.assert_allclose(
        phpy_yaml.nac_params["born"], phpy_yaml_test.nac_params["born"], atol=1e-8
    )
    np.testing.assert_allclose(
        phpy_yaml.nac_params["dielectric"],
        phpy_yaml_test.nac_params["dielectric"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        phpy_yaml.nac_params["factor"],
        phpy_yaml_test.nac_params["factor"],
        atol=1e-8,
    )
    disps, forces = get_displacements_and_forces(phpy_yaml.dataset)
    disps_test, forces_test = get_displacements_and_forces(phpy_yaml_test.dataset)
    np.testing.assert_allclose(forces, forces_test, atol=1e-8)
    np.testing.assert_allclose(disps, disps_test, atol=1e-8)


def _compare(cell):
    cell_ref = read_vasp(os.path.join(data_dir, "..", "POSCAR_NaCl"))
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r


def _get_unitcell(filename):
    phpy_yaml = PhonopyYaml()
    phpy_yaml.read(filename)
    return phpy_yaml.unitcell
