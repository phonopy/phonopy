"""Tests for wien2k interface."""

import os
import pathlib

import numpy as np

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.wien2k import (
    parse_set_of_forces,
    parse_wien2k_struct,
    write_supercells_with_displacements,
    write_wein2k,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Minimal inline struct content (NaCl-like cubic, 2 atoms)
# ---------------------------------------------------------------------------
# Lattice parameters: a=b=c=5.64 Bohr, alpha=beta=gamma=90°
# parse_wien2k_struct reads fixed-column fields, so spacing is critical.
_NACL_STRUCT = """\
NaCl test
P   LATTICE,NONEQUIV.ATOMS:  2
MODE OF CALC=RELA unit=bohr
  5.640000  5.640000  5.640000 90.000000 90.000000 90.000000
ATOM  -1: X=0.00000000 Y=0.00000000 Z=0.00000000
          MULT= 1          ISPLIT= 8
Na         NPT=  781  R0=0.00001000 RMT=    2.0000   Z: 11.0
LOCAL ROT MATRIX:    1.0000000 0.0000000 0.0000000
                     0.0000000 1.0000000 0.0000000
                     0.0000000 0.0000000 1.0000000
ATOM  -2: X=0.50000000 Y=0.50000000 Z=0.50000000
          MULT= 1          ISPLIT= 8
Cl         NPT=  781  R0=0.00001000 RMT=    2.0000   Z: 17.0
LOCAL ROT MATRIX:    1.0000000 0.0000000 0.0000000
                     0.0000000 1.0000000 0.0000000
                     0.0000000 0.0000000 1.0000000
   0      NUMBER OF SYMMETRY OPERATIONS
"""


def _scf_force_line(atom_num, fx, fy, fz):
    """Build a :FGL line in Wien2k .scf format.

    Column layout: [0:4]=:FGL, [4:7]=atom_num, "total forces" in [7:29],
    [29:45]=fx, [45:61]=fy, [61:77]=fz (all 16-char fields).
    """
    header = ":FGL%3d  total forces" % atom_num  # 21 chars
    header = header + " " * (29 - len(header))  # pad to 29 chars
    return header + "%16.8f%16.8f%16.8f\n" % (fx, fy, fz)


def _make_scf(forces):
    """Return a minimal .scf file string with the given per-atom forces.

    forces : list of (fx, fy, fz) tuples
    """
    lines = ["# dummy scf file\n"]
    for i, (fx, fy, fz) in enumerate(forces, start=1):
        lines.append(_scf_force_line(i, fx, fy, fz))
    return "".join(lines)


# ---------------------------------------------------------------------------
# parse_wien2k_struct (existing data file)
# ---------------------------------------------------------------------------


def test_parse_wien2k_struct():
    """Test structure parsing."""
    filename_BaGa2 = cwd / "BaGa2.struct"
    cell, _, _, _ = parse_wien2k_struct(filename_BaGa2)
    filename = cwd / "BaGa2-wien2k.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_parse_wien2k_struct_atom_count():
    """parse_wien2k_struct returns the correct total number of atoms."""
    # BaGa2: 1 Ba (MULT=1) + 2 Ga (MULT=2) = 3 atoms total
    cell, npts, r0s, rmts = parse_wien2k_struct(cwd / "BaGa2.struct")
    assert len(cell) == 3
    assert len(npts) == 3
    assert len(r0s) == 3
    assert len(rmts) == 3


def test_parse_wien2k_struct_npts():
    """parse_wien2k_struct returns correct NPT values."""
    _, npts, _, _ = parse_wien2k_struct(cwd / "BaGa2.struct")
    assert npts == [781, 781, 781]


def test_parse_wien2k_struct_rmt():
    """parse_wien2k_struct returns correct RMT values."""
    _, _, _, rmts = parse_wien2k_struct(cwd / "BaGa2.struct")
    np.testing.assert_allclose(rmts, [2.5, 2.28, 2.28], atol=1e-5)


def test_parse_wien2k_struct_inline(tmp_path):
    """Inline NaCl struct is parsed with correct symbols and positions."""
    fpath = tmp_path / "NaCl.struct"
    fpath.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath)

    assert cell.symbols == ["Na", "Cl"]
    assert len(cell) == 2
    np.testing.assert_allclose(cell.scaled_positions[0], [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(cell.scaled_positions[1], [0.5, 0.5, 0.5], atol=1e-7)
    assert npts == [781, 781]
    np.testing.assert_allclose(r0s, [1e-5, 1e-5], atol=1e-10)
    np.testing.assert_allclose(rmts, [2.0, 2.0], atol=1e-5)


def test_parse_wien2k_struct_cubic_lattice(tmp_path):
    """Cubic cell (a=b=c=5.64, all 90°) gives diagonal lattice matrix."""
    fpath = tmp_path / "NaCl.struct"
    fpath.write_text(_NACL_STRUCT)
    cell, _, _, _ = parse_wien2k_struct(fpath)
    a = 5.64
    np.testing.assert_allclose(cell.cell, np.diag([a, a, a]), atol=1e-5)


# ---------------------------------------------------------------------------
# write_wein2k
# ---------------------------------------------------------------------------


def test_write_wein2k_creates_file(tmp_path):
    """write_wein2k creates a struct file."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    fpath_out = str(tmp_path / "NaCl_out.struct")
    write_wein2k(fpath_out, cell, npts, r0s, rmts)
    assert pathlib.Path(fpath_out).exists()


def test_write_wein2k_content_keywords(tmp_path):
    """Written struct file contains expected Wien2k keywords."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    fpath_out = str(tmp_path / "NaCl_out.struct")
    write_wein2k(fpath_out, cell, npts, r0s, rmts)
    text = pathlib.Path(fpath_out).read_text()

    assert "LATTICE,NONEQUIV.ATOMS:" in text
    assert "MODE OF CALC=" in text
    assert "LOCAL ROT MATRIX:" in text
    assert "NUMBER OF SYMMETRY OPERATIONS" in text


def test_write_wein2k_roundtrip(tmp_path):
    """write_wein2k → parse_wien2k_struct preserves cell, positions, and symbols."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    fpath_out = str(tmp_path / "NaCl_out.struct")
    write_wein2k(fpath_out, cell, npts, r0s, rmts)
    cell2, npts2, r0s2, rmts2 = parse_wien2k_struct(fpath_out)

    assert isclose(cell, cell2)
    assert npts == npts2
    np.testing.assert_allclose(r0s, r0s2, atol=1e-8)
    np.testing.assert_allclose(rmts, rmts2, atol=1e-5)


def test_write_wein2k_roundtrip_baga2(tmp_path):
    """Round-trip for BaGa2 (multi-atom site, MULT>1 written as MULT=1)."""
    cell, npts, r0s, rmts = parse_wien2k_struct(cwd / "BaGa2.struct")

    fpath_out = str(tmp_path / "BaGa2_out.struct")
    write_wein2k(fpath_out, cell, npts, r0s, rmts)
    cell2, npts2, r0s2, rmts2 = parse_wien2k_struct(fpath_out)

    assert isclose(cell, cell2)
    assert cell.symbols == cell2.symbols


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_creates_files(tmp_path):
    """wien2kS and wien2kS-001.in are created with default pre_filename."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(
            cell, [cell, cell], [1, 2], npts, r0s, rmts, 1
        )
        assert (tmp_path / "wien2kS").exists()
        assert (tmp_path / "wien2kS-001.in").exists()
        assert (tmp_path / "wien2kS-002.in").exists()
    finally:
        os.chdir(old_cwd)


def test_write_supercells_content_readable(tmp_path):
    """Displacement files written by write_supercells are parseable."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(cell, [cell], [1], npts, r0s, rmts, 1)
        cell2, _, _, _ = parse_wien2k_struct(str(tmp_path / "wien2kS-001.in"))
    finally:
        os.chdir(old_cwd)

    assert isclose(cell, cell2)


def test_write_supercells_custom_prefix(tmp_path):
    """Custom pre_filename produces correctly named files."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(
            cell, [cell], [1], npts, r0s, rmts, 1, pre_filename="mycase"
        )
        assert (tmp_path / "mycaseS").exists()
        assert (tmp_path / "mycaseS-001.in").exists()
    finally:
        os.chdir(old_cwd)


def test_write_supercells_num_unitcells_expands_params(tmp_path):
    """num_unitcells_in_supercell repeats npts/r0s/rmts for each unit cell."""
    fpath_in = tmp_path / "NaCl.struct"
    fpath_in.write_text(_NACL_STRUCT)
    cell, npts, r0s, rmts = parse_wien2k_struct(fpath_in)

    # Build a 2×1×1 supercell (4 atoms) manually
    sup_cell = PhonopyAtoms(
        cell=cell.cell * np.array([[2, 1, 1]]).T,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.0, 0.0],
            [0.75, 0.25, 0.25],
        ],
        symbols=["Na", "Cl", "Na", "Cl"],
    )

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(
            sup_cell, [sup_cell], [1], npts, r0s, rmts, 2
        )
        # The supercell struct file must have 4 atoms
        cell_sc, npts_sc, _, _ = parse_wien2k_struct(str(tmp_path / "wien2kS"))
    finally:
        os.chdir(old_cwd)

    assert len(cell_sc) == 4
    assert len(npts_sc) == 4


# ---------------------------------------------------------------------------
# parse_set_of_forces (P1 mode)
# ---------------------------------------------------------------------------


def test_parse_set_of_forces_p1_values(tmp_path):
    """parse_set_of_forces in P1 mode returns negated forces with zero drift."""
    # Cubic cell → red_lattice = identity → forces unchanged by basis transform
    fpath_struct = tmp_path / "NaCl.struct"
    fpath_struct.write_text(_NACL_STRUCT)
    cell, _, _, _ = parse_wien2k_struct(fpath_struct)

    # Create mock .scf file: force on atom 1 is (+0.01, -0.01, 0),
    # force on atom 2 is (-0.01, +0.01, 0) → drift already zero
    scf_content = _make_scf([(0.01, -0.01, 0.0), (-0.01, 0.01, 0.0)])
    scf_path = tmp_path / "disp.scf"
    scf_path.write_text(scf_content)

    disps = [[0.01, 0.0, 0.0]]
    frs = parse_set_of_forces(
        disps,
        [str(scf_path)],
        cell,
        wien2k_P1_mode=True,
        verbose=False,
    )

    assert len(frs) == 1
    assert frs[0].shape == (2, 3)
    # Forces = -(gradients) → sign is inverted? Actually Wien2k outputs forces directly.
    # _get_forces_wien2k returns forces as-is (no sign change).
    np.testing.assert_allclose(frs[0][0], [0.01, -0.01, 0.0], atol=1e-8)
    np.testing.assert_allclose(frs[0][1], [-0.01, 0.01, 0.0], atol=1e-8)


def test_parse_set_of_forces_p1_drift_correction(tmp_path):
    """Forces after drift correction sum to zero."""
    fpath_struct = tmp_path / "NaCl.struct"
    fpath_struct.write_text(_NACL_STRUCT)
    cell, _, _, _ = parse_wien2k_struct(fpath_struct)

    # Non-zero drift: forces don't sum to zero
    scf_content = _make_scf([(0.02, 0.0, 0.0), (0.00, 0.0, 0.0)])
    scf_path = tmp_path / "disp.scf"
    scf_path.write_text(scf_content)

    frs = parse_set_of_forces(
        [[0.01, 0.0, 0.0]],
        [str(scf_path)],
        cell,
        wien2k_P1_mode=True,
        verbose=False,
    )

    assert len(frs) == 1
    np.testing.assert_allclose(frs[0].sum(axis=0), 0.0, atol=1e-10)


def test_parse_set_of_forces_p1_multiple_files(tmp_path):
    """parse_set_of_forces accumulates forces from multiple .scf files."""
    fpath_struct = tmp_path / "NaCl.struct"
    fpath_struct.write_text(_NACL_STRUCT)
    cell, _, _, _ = parse_wien2k_struct(fpath_struct)

    scf1 = _make_scf([(0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)])
    scf2 = _make_scf([(0.0, 0.02, 0.0), (0.0, -0.02, 0.0)])
    (tmp_path / "d1.scf").write_text(scf1)
    (tmp_path / "d2.scf").write_text(scf2)

    disps = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
    frs = parse_set_of_forces(
        disps,
        [str(tmp_path / "d1.scf"), str(tmp_path / "d2.scf")],
        cell,
        wien2k_P1_mode=True,
        verbose=False,
    )

    assert len(frs) == 2
    for f in frs:
        assert f.shape == (2, 3)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
