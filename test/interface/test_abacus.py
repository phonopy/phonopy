"""Tests for ABACUS calculater interface."""

import os
import pathlib
import tempfile
import warnings

import numpy as np
import pytest

from phonopy.interface.abacus import (
    get_abacus_structure,
    read_abacus,
    read_abacus_output,
    write_abacus,
    write_supercells_with_displacements,
)
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure import cells

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_abacus_nomag():
    """Test of read_ABACUS."""
    cell, pps, orbitals, abfs = read_abacus(os.path.join(data_dir, "NaCl-abacus.stru"))
    filename = os.path.join(data_dir, "NaCl-abinit-pwscf.yaml")
    cell_ref = read_cell_yaml(filename)
    # assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    assert cells.isclose(cell, cell_ref)
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r
    assert pps["Na"] == "Na_ONCV_PBE-1.0.upf"
    assert pps["Cl"] == "Cl_ONCV_PBE-1.0.upf"
    assert orbitals["Na"] == "Na_gga_9au_100Ry_4s2p1d.orb"
    assert orbitals["Cl"] == "Cl_gga_8au_100Ry_2s2p1d.orb"


def test_read_abacus_mag():
    """Test of read_ABACUS with magnetic moments."""
    cell, pps, orbitals, abfs = read_abacus(
        os.path.join(data_dir, "NaCl-abacus-mag.stru")
    )
    filename = os.path.join(data_dir, "NaCl-abacus-mag.yaml")
    cell_ref = read_cell_yaml(filename)
    # assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    assert cells.isclose(cell, cell_ref)
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r
    assert pps["Na"] == "Na_ONCV_PBE-1.0.upf"
    assert pps["Cl"] == "Cl_ONCV_PBE-1.0.upf"
    assert orbitals["Na"] == "Na_gga_9au_100Ry_4s2p1d.orb"
    assert orbitals["Cl"] == "Cl_gga_8au_100Ry_2s2p1d.orb"

    diff_mag = cell_ref.magnetic_moments - np.array([1] * 4 + [2] * 4)
    assert (np.abs(diff_mag) < 1e-5).all()


def test_read_abacus_mag_noncolin():
    """Test of read_ABACUS with magnetic moments."""
    cell, pps, orbitals, abfs = read_abacus(
        os.path.join(data_dir, "NaCl-abacus-mag-noncolin.stru")
    )
    filename = os.path.join(data_dir, "NaCl-abacus-mag-noncolin.yaml")
    cell_ref = read_cell_yaml(filename)
    # assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    assert cells.isclose(cell, cell_ref)
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r
    assert pps["Na"] == "Na_ONCV_PBE-1.0.upf"
    assert pps["Cl"] == "Cl_ONCV_PBE-1.0.upf"
    assert orbitals["Na"] == "Na_gga_9au_100Ry_4s2p1d.orb"
    assert orbitals["Cl"] == "Cl_gga_8au_100Ry_2s2p1d.orb"
    diff_mag = cell_ref.magnetic_moments - cell.magnetic_moments
    assert (np.abs(diff_mag) < 1e-5).all()


def test_read_abacus_output():
    """Test of read abacus output."""
    force = read_abacus_output(os.path.join(data_dir, "NaCl-abacus.out"))
    assert force.mean() < 1e-10
    assert force[0][0] + 1.85537138e-02 < 1e-5


# ---------------------------------------------------------------------------
# Regression tests for write_abacus / get_abacus_structure
# (written before refactoring to capture current behaviour)
# ---------------------------------------------------------------------------


def _stru(name: str) -> str:
    return os.path.join(data_dir, name)


# ── round-trip tests (read → write → read back) ──────────────────────────


def test_write_abacus_roundtrip_nomag():
    """Round-trip without magnetic moments preserves cell, positions, pps, orbitals."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus.stru"))
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "STRU")
        write_abacus(fpath, cell, pps, orbitals, abfs)
        cell2, pps2, orbitals2, abfs2 = read_abacus(fpath)

    assert cells.isclose(cell, cell2)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert list(cell.symbols) == list(cell2.symbols)
    assert cell2.magnetic_moments is None
    assert pps2 == pps
    assert orbitals2 == orbitals
    assert abfs2 is None


def test_write_abacus_roundtrip_colinear_mag():
    """Round-trip with colinear magnetic moments preserves all data."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus-mag.stru"))
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "STRU")
        write_abacus(fpath, cell, pps, orbitals, abfs)
        cell2, pps2, orbitals2, abfs2 = read_abacus(fpath)

    assert cells.isclose(cell, cell2)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert cell.magnetic_moments is not None
    assert cell2.magnetic_moments is not None
    np.testing.assert_allclose(
        np.ravel(cell.magnetic_moments), np.ravel(cell2.magnetic_moments), atol=1e-10
    )
    assert pps2 == pps
    assert orbitals2 == orbitals


def test_write_abacus_roundtrip_noncolin_mag():
    """Round-trip with non-colinear magnetic moments preserves all data."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus-mag-noncolin.stru"))
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "STRU")
        write_abacus(fpath, cell, pps, orbitals, abfs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cell2, pps2, orbitals2, abfs2 = read_abacus(fpath)

    assert cells.isclose(cell, cell2)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert cell.magnetic_moments is not None
    assert cell2.magnetic_moments is not None
    np.testing.assert_allclose(
        cell.magnetic_moments, cell2.magnetic_moments, atol=1e-10
    )
    assert pps2 == pps


# ── get_abacus_structure content tests ───────────────────────────────────


def test_get_abacus_structure_sections_nomag():
    """Required STRU sections are present; no mag keyword without magnetic moments."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus.stru"))
    text = get_abacus_structure(cell, pps, orbitals, abfs)

    assert "ATOMIC_SPECIES" in text
    assert "NUMERICAL_ORBITAL" in text
    assert "LATTICE_CONSTANT" in text
    assert "LATTICE_VECTORS" in text
    assert "ATOMIC_POSITIONS" in text
    assert "Direct" in text
    assert "Na_ONCV_PBE-1.0.upf" in text
    assert "Cl_ONCV_PBE-1.0.upf" in text
    assert "Na_gga_9au_100Ry_4s2p1d.orb" in text
    assert "Cl_gga_8au_100Ry_2s2p1d.orb" in text
    assert "mag" not in text


def test_get_abacus_structure_no_orbitals():
    """Without orbitals/abfs the corresponding sections must be absent."""
    cell, pps, _, _ = read_abacus(_stru("NaCl-abacus.stru"))
    text = get_abacus_structure(cell, pps)

    assert "NUMERICAL_ORBITAL" not in text
    assert "ABFS_ORBITAL" not in text


def test_get_abacus_structure_abfs_section():
    """ABFS_ORBITAL section appears when abfs dict is given."""
    cell, pps, orbitals, _ = read_abacus(_stru("NaCl-abacus.stru"))
    abfs = {"Na": "Na_abfs.orb", "Cl": "Cl_abfs.orb"}
    text = get_abacus_structure(cell, pps, orbitals, abfs)

    assert "ABFS_ORBITAL" in text
    assert "Na_abfs.orb" in text
    assert "Cl_abfs.orb" in text


def test_get_abacus_structure_lattice_constant_is_one():
    """LATTICE_CONSTANT is always written as 1.0 (lattice vectors in Angstrom)."""
    cell, pps, _, _ = read_abacus(_stru("NaCl-abacus.stru"))
    text = get_abacus_structure(cell, pps)
    lines = text.splitlines()
    idx = lines.index("LATTICE_CONSTANT")
    assert float(lines[idx + 1]) == pytest.approx(1.0)


def test_get_abacus_structure_lattice_vectors():
    """Lattice vectors in the output match the cell (in Angstrom)."""
    cell, pps, _, _ = read_abacus(_stru("NaCl-abacus.stru"))
    text = get_abacus_structure(cell, pps)
    lines = text.splitlines()
    idx = lines.index("LATTICE_VECTORS")
    parsed = np.array(
        [[float(x) for x in lines[idx + i + 1].split()] for i in range(3)]
    )
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_abacus_structure_no_pps_warns():
    """Missing pseudopotentials trigger a UserWarning and placeholder names."""
    cell, _, _, _ = read_abacus(_stru("NaCl-abacus.stru"))
    with pytest.warns(UserWarning, match="pseudopotential"):
        text = get_abacus_structure(cell, pps=None)
    assert "Na_pp_filename_here" in text
    assert "Cl_pp_filename_here" in text


def test_get_abacus_structure_colinear_mag():
    """Colinear magnetic moments appear with mag keyword and move flags."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus-mag.stru"))
    text = get_abacus_structure(cell, pps, orbitals, abfs)

    assert "mag" in text
    assert "1 1 1" in text


def test_get_abacus_structure_noncolin_mag():
    """Non-colinear magnetic moments (3-vector) appear in the output."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus-mag-noncolin.stru"))
    text = get_abacus_structure(cell, pps, orbitals, abfs)

    assert "mag" in text
    assert "1 1 1" in text
    # Each mag line should contain 3 components
    mag_lines = [ln for ln in text.splitlines() if "mag" in ln]
    assert all(len(ln.split()) >= 7 for ln in mag_lines)  # x y z 1 1 1 mag mx my mz


# ── write_supercells_with_displacements tests ─────────────────────────────


def test_write_supercells_filenames():
    """Supercell file (pre.in) and displacement files (pre-001, …) are created."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus.stru"))
    ids = np.array([1, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "STRU")
        write_supercells_with_displacements(
            cell, [cell, cell], ids, pps, orbitals, abfs, pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "STRU.in").exists()
        assert pathlib.Path(tmpdir, "STRU-001").exists()
        assert pathlib.Path(tmpdir, "STRU-002").exists()


def test_write_supercells_custom_prefix():
    """Custom pre_filename is respected."""
    cell, pps, _, _ = read_abacus(_stru("NaCl-abacus.stru"))
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "ABACUS")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pps, pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "ABACUS.in").exists()
        assert pathlib.Path(tmpdir, "ABACUS-001").exists()


def test_write_supercells_content_readable():
    """Written supercell file must be parseable by read_abacus."""
    cell, pps, orbitals, abfs = read_abacus(_stru("NaCl-abacus.stru"))
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "STRU")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pps, orbitals, abfs, pre_filename=pre
        )
        cell2, _, _, _ = read_abacus(str(pathlib.Path(tmpdir, "STRU.in")))
    assert cells.isclose(cell, cell2)
