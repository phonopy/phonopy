"""Tests for QE calculator interface."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from phonopy.file_IO import get_io_module_to_decompress
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.qe import (
    PH_Q2R,
    get_pwscf_structure,
    read_pwscf,
    write_pwscf,
    write_supercells_with_displacements,
)
from phonopy.structure.cells import isclose
from phonopy.structure.symmetry import Symmetry

cwd = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# read_pwscf tests
# ---------------------------------------------------------------------------


def test_read_pwscf() -> None:
    """Test of read_pwscf with default scaled positions.

    Keywords appear in the following order:

    ATOMIC_SPECIES
    ATOMIC_POSITIONS
    CELL_PARAMETERS
    K_POINTS

    """
    _test_read_pwscf("NaCl-pwscf.in")


def test_read_pwscf_2() -> None:
    """Test of read_pwscf with default scaled positions.

    Keywords appear in different order from test_read_pwscf.

    ATOMIC_SPECIES
    ATOMIC_POSITIONS
    K_POINTS
    CELL_PARAMETERS

    """
    _test_read_pwscf("NaCl-pwscf-2.in")


def test_read_pwscf_angstrom() -> None:
    """Test of read_pwscf with angstrom coordinates."""
    _test_read_pwscf("NaCl-pwscf-angstrom.in")


def test_read_pwscf_bohr() -> None:
    """Test of read_pwscf with bohr coordinates."""
    _test_read_pwscf("NaCl-pwscf-bohr.in")


def test_read_pwscf_NaCl_Xn() -> None:
    """Test of read_pwscf."""
    cell, pp_filenames = read_pwscf(cwd / "NaCl-pwscf-Xn.in")
    print(cell)
    symnums = pp_filenames.keys()
    assert set(symnums) == {"Na", "Cl", "Cl1"}
    np.testing.assert_allclose(
        cell.masses,
        [
            22.98976928,
            22.98976928,
            22.98976928,
            22.98976928,
            35.453,
            35.453,
            70.0,
            70.0,
        ],
    )
    assert ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl1", "Cl1"] == cell.symbols

    cell_ref, pp_filenames = read_pwscf(cwd / "NaCl-pwscf.in")
    symops = Symmetry(cell).symmetry_operations
    symops_ref = Symmetry(cell_ref).symmetry_operations
    np.testing.assert_allclose(symops["translations"], symops_ref["translations"])
    np.testing.assert_array_equal(symops["rotations"], symops_ref["rotations"])


def test_read_pwscf_atom_count() -> None:
    """NaCl conventional cell has 8 atoms."""
    cell, _ = read_pwscf(cwd / "NaCl-pwscf.in")
    assert len(cell) == 8


def test_read_pwscf_species() -> None:
    """NaCl cell has 4 Na and 4 Cl atoms."""
    cell, _ = read_pwscf(cwd / "NaCl-pwscf.in")
    assert cell.symbols.count("Na") == 4
    assert cell.symbols.count("Cl") == 4


def test_read_pwscf_pp_filenames() -> None:
    """pp_filenames dict maps species to pseudopotential filenames."""
    _, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    assert set(pp.keys()) == {"Na", "Cl"}
    assert pp["Na"] == "Na.pbe-spn-kjpaw_psl.0.2.UPF"
    assert pp["Cl"] == "Cl.pbe-n-kjpaw_psl.0.1.UPF"


# ---------------------------------------------------------------------------
# get_pwscf_structure content tests
# ---------------------------------------------------------------------------


def test_get_pwscf_structure_keywords() -> None:
    """Required block keywords must appear in the output."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    text = get_pwscf_structure(cell, pp)

    assert "CELL_PARAMETERS" in text
    assert "ATOMIC_SPECIES" in text
    assert "ATOMIC_POSITIONS" in text
    assert "crystal" in text


def test_get_pwscf_structure_ibrav_header() -> None:
    """Header line must contain correct nat and ntyp."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    text = get_pwscf_structure(cell, pp)
    header = text.splitlines()[0]
    assert "nat = 8" in header
    assert "ntyp = 2" in header


def test_get_pwscf_structure_cell_parameters_bohr() -> None:
    """CELL_PARAMETERS block must be in bohr and match cell.cell."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    text = get_pwscf_structure(cell, pp)
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "CELL_PARAMETERS" in ln)
    assert "bohr" in lines[idx].lower()
    parsed = np.array(
        [[float(x) for x in lines[idx + i + 1].split()] for i in range(3)]
    )
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_pwscf_structure_atomic_positions_count() -> None:
    """Number of ATOMIC_POSITIONS lines must equal number of atoms."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    text = get_pwscf_structure(cell, pp)
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if "ATOMIC_POSITIONS" in ln)
    pos_lines = [ln for ln in lines[idx + 1 :] if ln.strip()]
    assert len(pos_lines) == len(cell)


def test_get_pwscf_structure_pp_filenames_present() -> None:
    """Pseudopotential filenames must appear in ATOMIC_SPECIES block."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    text = get_pwscf_structure(cell, pp)
    assert "Na.pbe-spn-kjpaw_psl.0.2.UPF" in text
    assert "Cl.pbe-n-kjpaw_psl.0.1.UPF" in text


def test_get_pwscf_structure_no_pp_warns() -> None:
    """Missing pp_filenames triggers a UserWarning."""
    cell, _ = read_pwscf(cwd / "NaCl-pwscf.in")
    with pytest.warns(UserWarning):
        text = get_pwscf_structure(cell, pp_filenames=None)
    assert "Na_PP_filename" in text
    assert "Cl_PP_filename" in text


def test_get_pwscf_structure_atomic_species_masses() -> None:
    """ATOMIC_SPECIES block must contain correct masses."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    text = get_pwscf_structure(cell, pp)
    lines = text.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "ATOMIC_SPECIES")
    # 2 species lines follow
    na_line = lines[idx + 1].split()
    cl_line = lines[idx + 2].split()
    assert na_line[0] == "Na"
    assert float(na_line[1]) == pytest.approx(22.98977, abs=1e-4)
    assert cl_line[0] == "Cl"
    assert float(cl_line[1]) == pytest.approx(35.45300, abs=1e-4)


# ---------------------------------------------------------------------------
# write_pwscf tests
# ---------------------------------------------------------------------------


def test_write_pwscf_creates_file() -> None:
    """write_pwscf must create a file at the given path."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "out.in"
        write_pwscf(fpath, cell, pp)  # type: ignore[arg-type]
        assert fpath.exists()


def test_write_pwscf_content_matches_get_pwscf_structure() -> None:
    """write_pwscf writes the same content as get_pwscf_structure."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    expected = get_pwscf_structure(cell, pp)
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "out.in"
        write_pwscf(fpath, cell, pp)  # type: ignore[arg-type]
        actual = fpath.read_text()
    assert actual == expected


# ---------------------------------------------------------------------------
# write_supercells_with_displacements tests
# ---------------------------------------------------------------------------


def test_write_supercells_filenames() -> None:
    """Supercell file and displacement files are created with correct names."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    ids = np.array([1, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "supercell")
        write_supercells_with_displacements(
            cell, [cell, cell], ids, pp, pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "supercell.in").exists()
        assert pathlib.Path(tmpdir, "supercell-001.in").exists()
        assert pathlib.Path(tmpdir, "supercell-002.in").exists()


def test_write_supercells_custom_prefix() -> None:
    """Custom pre_filename is respected."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "QE")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pp, pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "QE.in").exists()
        assert pathlib.Path(tmpdir, "QE-001.in").exists()


def test_write_supercells_content_has_keywords() -> None:
    """Written supercell file contains required QE keywords."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "supercell")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pp, pre_filename=pre
        )
        text = (pathlib.Path(tmpdir) / "supercell.in").read_text()
    assert "CELL_PARAMETERS" in text
    assert "ATOMIC_SPECIES" in text
    assert "ATOMIC_POSITIONS" in text


def test_write_supercells_custom_width() -> None:
    """Custom width parameter controls zero-padding of displacement file indices."""
    cell, pp = read_pwscf(cwd / "NaCl-pwscf.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "sc")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pp, pre_filename=pre, width=4
        )
        assert pathlib.Path(tmpdir, "sc-0001.in").exists()


# ---------------------------------------------------------------------------
# PH_Q2R tests
# ---------------------------------------------------------------------------


def test_make_fc_q2r() -> None:
    """Test make_fc_q2r."""
    fc_0_10 = [
        [6.06001648e-05, -3.48358667e-05, -8.14194922e-05],
        [-3.48358667e-05, 1.21530469e-05, -1.65827117e-04],
        [-8.14194922e-05, -1.65827117e-04, -1.14989696e-04],
    ]
    fc_1_10 = [
        [-4.05313258e-04, 1.92325415e-10, -9.48168187e-11],
        [5.67066085e-11, -3.30626094e-04, 1.06319726e-03],
        [-7.34782548e-11, 3.68433780e-04, 8.99705485e-04],
    ]

    fc_filename = cwd / "NaCl-q2r.fc.xz"
    myio = get_io_module_to_decompress(fc_filename)
    with myio.open(fc_filename, "rt") as f:
        primcell_filename = cwd / "NaCl-q2r.in"
        cell, _ = read_pwscf(primcell_filename)
        q2r = PH_Q2R(f)
        q2r.run(cell)

    assert q2r.fc is not None
    np.testing.assert_allclose(fc_0_10, q2r.fc[0, 10], atol=1e-8)
    np.testing.assert_allclose(fc_1_10, q2r.fc[1, 10], atol=1e-8)


def test_q2r_dimension() -> None:
    """PH_Q2R.dimension must be set after run()."""
    fc_filename = cwd / "NaCl-q2r.fc.xz"
    myio = get_io_module_to_decompress(fc_filename)
    with myio.open(fc_filename, "rt") as f:
        cell, _ = read_pwscf(cwd / "NaCl-q2r.in")
        q2r = PH_Q2R(f)
        q2r.run(cell)

    assert q2r.dimension is not None
    assert q2r.dimension.shape == (3,)


def test_q2r_fc_shape() -> None:
    """Force constants shape must be (natom_prim, natom_super, 3, 3)."""
    fc_filename = cwd / "NaCl-q2r.fc.xz"
    myio = get_io_module_to_decompress(fc_filename)
    with myio.open(fc_filename, "rt") as f:
        cell, _ = read_pwscf(cwd / "NaCl-q2r.in")
        q2r = PH_Q2R(f)
        q2r.run(cell)

    assert q2r.fc is not None
    assert q2r.primitive is not None
    assert q2r.supercell is not None
    natom_prim = len(q2r.primitive)
    natom_super = len(q2r.supercell)
    assert q2r.fc.shape == (natom_prim, natom_super, 3, 3)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _test_read_pwscf(filename: str) -> None:
    """Test of read_pwscf."""
    cell, pp_filenames = read_pwscf(cwd / filename)
    cell_ref = read_cell_yaml(cwd / "NaCl-abinit-pwscf.yaml")
    isclose(cell, cell_ref, atol=1e-5)
