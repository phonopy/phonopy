"""Tests of PhonopyYaml."""

import io
from pathlib import Path

import numpy as np
import yaml

import phonopy
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import (
    PhonopyYaml,
    PhonopyYamlLoader,
    load_phonopy_yaml,
    read_cell_yaml,
    read_phonopy_yaml,
)
from phonopy.interface.vasp import read_vasp
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive
from phonopy.structure.dataset import get_displacements_and_forces

cwd = Path(__file__).parent


def test_read_poscar_yaml(helper_methods):
    """Test to parse PhonopyAtoms.__str__ output."""
    filename = cwd / "NaCl-vasp.yaml"
    cell = _get_unitcell(filename)

    _compare_NaCl_convcell(cell, helper_methods.compare_cells_with_order)


def test_read_phonopy_yaml(helper_methods):
    """Test to parse phonopy.yaml like file."""
    filename = cwd / "phonopy.yaml"
    cell = read_phonopy_yaml(filename).unitcell
    _compare_NaCl_convcell(cell, helper_methods.compare_cells_with_order)


def test_read_phonopy_yaml_with_stream(helper_methods):
    """Test to parse phonopy.yaml like file stream."""
    filename = cwd / "phonopy.yaml"
    with open(filename) as fp:
        cell = read_phonopy_yaml(fp).unitcell
        _compare_NaCl_convcell(cell, helper_methods.compare_cells_with_order)


def test_PhonopyYaml_read(helper_methods):
    """Test to parse phonopy.yaml like file using PhonopyYaml.read()."""
    filename = cwd / "phonopy.yaml"
    cell = _get_unitcell(filename)
    _compare_NaCl_convcell(cell, helper_methods.compare_cells_with_order)


def test_PhonopyYaml_read_with_stream(helper_methods):
    """Test to parse phonopy.yaml like file stream using PhonopyYaml.read()."""
    filename = cwd / "phonopy.yaml"
    with open(filename) as fp:
        cell = _get_unitcell(fp)
        _compare_NaCl_convcell(cell, helper_methods.compare_cells_with_order)


def test_read_cell_yaml(helper_methods):
    """Test to parse phonopy_symcells.yaml like file."""
    filename = cwd / "phonopy_symcells_NaCl.yaml"
    cell = read_cell_yaml(filename)
    _compare_NaCl_convcell(cell, helper_methods.compare_cells)

    pcell = read_cell_yaml(filename, cell_type="primitive")
    helper_methods.compare_cells(pcell, get_primitive(cell, "F"))


def test_read_cell_yaml_with_stream(helper_methods):
    """Test to parse phonopy_symcells.yaml like file."""
    filename = cwd / "phonopy_symcells_NaCl.yaml"
    with open(filename) as fp:
        cell = _get_unitcell(fp)
        _compare_NaCl_convcell(cell, helper_methods.compare_cells)
        fp.seek(0)
        pcell = read_cell_yaml(fp, cell_type="primitive")
        helper_methods.compare_cells(pcell, get_primitive(cell, "F"))


def test_write_phonopy_yaml(ph_nacl_nofcsym: Phonopy, helper_methods):
    """Test PhonopyYaml.set_phonon_info, __str__, yaml_data, parse."""
    phonon = ph_nacl_nofcsym
    phpy_yaml = PhonopyYaml(calculator="vasp")
    phpy_yaml.set_phonon_info(phonon)
    phpy_yaml_test = PhonopyYaml()
    phpy_yaml_test._data = load_phonopy_yaml(
        yaml.safe_load(io.StringIO(str(phpy_yaml))), calculator=phpy_yaml.calculator
    )
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
    phpy_yaml_test._data = load_phonopy_yaml(
        yaml.safe_load(io.StringIO(str(phpy_yaml))), calculator=phpy_yaml.calculator
    )
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


def test_load_nac_yaml():
    """Test to read NAC params using PhonopyYamlLoader."""
    pyl = PhonopyYamlLoader(yaml.safe_load(open(cwd / "nac.yaml"))).parse()
    assert pyl.data.nac_params
    for key in (
        "dielectric",
        "born",
        "factor",
        "method",
    ):
        assert key in pyl.data.nac_params
    assert pyl.data.nac_params["dielectric"].shape == (3, 3)
    assert pyl.data.nac_params["born"].shape == (2, 3, 3)
    assert isinstance(pyl.data.nac_params["factor"], float)
    assert isinstance(pyl.data.nac_params["method"], str)


def test_phonopy_yaml_extended_symbol(nacl_unitcell_order1: PhonopyAtoms):
    """Test of PhonopyYaml with extended symbol."""
    unitcell = nacl_unitcell_order1
    symbols = unitcell.symbols
    symbols[-1] = "Cl1"
    cell = PhonopyAtoms(
        cell=unitcell.cell,
        symbols=symbols,
        scaled_positions=unitcell.scaled_positions,
        masses=unitcell.masses,
    )
    ph = Phonopy(cell, supercell_matrix=[2, 2, 2])
    assert ph.primitive.symbols[:4] == ["Na"] * 4
    assert ph.primitive.symbols[4:7] == ["Cl"] * 3
    assert ph.primitive.symbols[-1] == "Cl1"
    assert ph.unitcell.symbols[:4] == ["Na"] * 4
    assert ph.unitcell.symbols[4:7] == ["Cl"] * 3
    assert ph.unitcell.symbols[-1] == "Cl1"
    assert ph.supercell.symbols[:32] == ["Na"] * 32
    assert ph.supercell.symbols[-32:-8] == ["Cl"] * 24
    assert ph.supercell.symbols[-8:] == ["Cl1"] * 8

    ph_load = phonopy.load(io.StringIO(str(PhonopyYaml().set_phonon_info(ph))))
    assert ph_load.primitive.symbols[:4] == ["Na"] * 4
    assert ph_load.primitive.symbols[4:7] == ["Cl"] * 3
    assert ph_load.primitive.symbols[-1] == "Cl1"
    assert ph_load.unitcell.symbols[:4] == ["Na"] * 4
    assert ph_load.unitcell.symbols[4:7] == ["Cl"] * 3
    assert ph_load.unitcell.symbols[-1] == "Cl1"
    assert ph_load.supercell.symbols[:32] == ["Na"] * 32
    assert ph_load.supercell.symbols[-32:-8] == ["Cl"] * 24
    assert ph_load.supercell.symbols[-8:] == ["Cl1"] * 8


def _compare_NaCl_convcell(cell, compare_cells):
    cell_ref = read_vasp(cwd / ".." / "POSCAR_NaCl")
    compare_cells(cell, cell_ref)


def _get_unitcell(filename):
    phpy_yaml = PhonopyYaml().read(filename)
    return phpy_yaml.unitcell
