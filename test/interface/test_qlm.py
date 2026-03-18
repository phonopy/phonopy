"""Tests for the QLM calculator interface."""

import os
import pathlib

import numpy as np

from phonopy.interface.qlm import (
    QlmFl,
    parse_set_of_forces,
    read_qlm,
    to_site_str,
    write_qlm,
    write_supercells_with_displacements,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# NaCl conventional supercell (xpos=True, plat=identity) used by several tests.
_NACL_SITEX = (
    "% site-data vn=3.0 xpos fast io=15 nbas=8"
    " alat=10.7531113565 plat= 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
    "#                            pos\n"
    " Na        0.0000000000   0.0000000000   0.0000000000\n"
    " Na        0.0000000000   0.5000000000   0.5000000000\n"
    " Na2       0.5000000000   0.0000000000   0.5000000000\n"
    " Na3       0.5000000000   0.5000000000   0.0000000000\n"
    " Cl        0.5000000000   0.5000000000   0.5000000000\n"
    " Clu       0.5000000000   0.0000000000   0.0000000000\n"
    " Cl        0.0000000000   0.5000000000   0.0000000000\n"
    " Cld       0.0000000000   0.0000000000   0.5000000000\n"
)

# Simple 2-atom NaCl primitive cell without xpos (Cartesian in alat units).
# plat has an off-diagonal element to make the conversion non-trivial.
# plat = [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], alat = 5.64
# Na frac=[0,0,0] → cart=[0,0,0]; Cl frac=[0.5,0.5,0] → cart=[0.5,0.75,0]
_NACL_CART_SITEX = (
    "% site-data vn=3.0 fast io=15 nbas=2"
    " alat=5.64 plat= 1.0 0.5 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
    "#                            pos\n"
    " Na        0.0000000000   0.0000000000   0.0000000000\n"
    " Cl        0.5000000000   0.7500000000   0.0000000000\n"
)


def _write_tmp(content: str, tmp_path: pathlib.Path, name: str = "site.lm") -> str:
    """Write string content to a temp file and return its path."""
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# parse_set_of_forces
# ---------------------------------------------------------------------------


def test_parse_set_of_forces(tmp_path):
    """parse_set_of_forces returns correct forces for a single file."""
    force_ref = (
        "% rows 2 cols 3 real\n"
        "    0.00000000   -0.00406659   -0.00406659\n"
        "   -0.00000000    0.00406659    0.00406659"
    )
    fpath = _write_tmp(force_ref, tmp_path, "forces.lm")

    frs = parse_set_of_forces(2, (fpath,), verbose=False)

    np.testing.assert_allclose(
        frs,
        [
            np.array(
                [
                    [0.00000000, -0.00406659, -0.00406659],
                    [-0.00000000, 0.00406659, 0.00406659],
                ]
            )
        ],
        atol=1e-7,
    )


def test_parse_set_of_forces_multiple_files(tmp_path):
    """parse_set_of_forces accumulates forces from multiple files."""
    row1 = "% rows 2 cols 3 real\n    0.01   -0.01    0.00\n   -0.01    0.01    0.00"
    row2 = "% rows 2 cols 3 real\n    0.00    0.02   -0.02\n    0.00   -0.02    0.02"
    f1 = _write_tmp(row1, tmp_path, "f1.lm")
    f2 = _write_tmp(row2, tmp_path, "f2.lm")

    frs = parse_set_of_forces(2, (f1, f2), verbose=False)

    assert len(frs) == 2
    # Each set must have zero drift after subtraction
    for f in frs:
        assert f.shape == (2, 3)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# QlmFl.split_symbol_element
# ---------------------------------------------------------------------------


def test_split_symbol_element_plain():
    """Plain element symbol gives empty suffix."""
    ctx = QlmFl()
    assert ctx.split_symbol_element("Na") == ("Na", "")
    assert ctx.split_symbol_element("Cl") == ("Cl", "")


def test_split_symbol_element_numeric_suffix():
    """Numeric suffix is separated from element."""
    ctx = QlmFl()
    el, sfx = ctx.split_symbol_element("Na2")
    assert el == "Na"
    assert sfx == "2"


def test_split_symbol_element_alpha_suffix():
    """Alphabetic suffix is separated from element."""
    ctx = QlmFl()
    el, sfx = ctx.split_symbol_element("Clu")
    assert el == "Cl"
    assert sfx == "u"


def test_split_symbol_element_unknown():
    """Unknown token gives empty element."""
    ctx = QlmFl()
    el, sfx = ctx.split_symbol_element("XYZ")
    assert el == ""


# ---------------------------------------------------------------------------
# read_qlm – xpos=True (fractional positions)
# ---------------------------------------------------------------------------


def test_read_qlm_roundtrip_xpos(tmp_path):
    """read_qlm → to_site_str → read_qlm preserves cell/positions/symbols."""
    f1 = _write_tmp(_NACL_SITEX, tmp_path, "site1.lm")
    cell1, (ctx1,) = read_qlm(f1)

    f2 = _write_tmp(ctx1.to_site_str(cell1), tmp_path, "site2.lm")
    cell2, _ = read_qlm(f2)

    isclose(cell1, cell2, atol=1e-7)


# ---------------------------------------------------------------------------
# read_qlm – xpos=False (Cartesian in alat units)
# ---------------------------------------------------------------------------


def test_read_qlm_non_xpos_positions(tmp_path):
    """read_qlm with xpos=False converts Cartesian→fractional correctly."""
    fpath = _write_tmp(_NACL_CART_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath)

    assert not ctx.xpos
    # Na stays at origin
    np.testing.assert_allclose(cell.scaled_positions[0], [0.0, 0.0, 0.0], atol=1e-10)
    # Cl: cart=[0.5,0.75,0] → frac = cart @ inv(plat)
    # plat = [[1,0.5,0],[0,1,0],[0,0,1]], inv(plat) = [[1,-0.5,0],[0,1,0],[0,0,1]]
    # frac = [0.5*1+0.75*0, 0.5*(-0.5)+0.75*1, 0] = [0.5, 0.5, 0]
    np.testing.assert_allclose(cell.scaled_positions[1], [0.5, 0.5, 0.0], atol=1e-10)


def test_read_qlm_non_xpos_roundtrip(tmp_path):
    """read_qlm → to_site_str → read_qlm preserves structure for xpos=False."""
    f1 = _write_tmp(_NACL_CART_SITEX, tmp_path, "site1.lm")
    cell1, (ctx1,) = read_qlm(f1)

    f2 = _write_tmp(ctx1.to_site_str(cell1), tmp_path, "site2.lm")
    cell2, _ = read_qlm(f2)
    isclose(cell1, cell2, atol=1e-7)


# ---------------------------------------------------------------------------
# to_site_str (standalone function)
# ---------------------------------------------------------------------------


def test_to_site_str_header_content(tmp_path):
    """to_site_str output contains required header keywords."""
    fpath = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath)
    text = to_site_str(cell, ctx)

    assert "site-data" in text
    assert "nbas=8" in text
    assert "alat=" in text
    assert "plat=" in text


def test_to_site_str_atom_count(tmp_path):
    """to_site_str writes exactly natoms position lines."""
    fpath = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath)
    text = to_site_str(cell, ctx)

    # Skip the header (line 0) and comment (line 1)
    data_lines = [ln for ln in text.splitlines()[2:] if ln.strip()]
    assert len(data_lines) == len(cell)


def test_to_site_str_without_context():
    """to_site_str called without extra_args uses a fresh QlmFl."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 5.64,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Na", "Cl"],
    )
    text = to_site_str(cell)
    assert "nbas=2" in text
    assert "Na" in text
    assert "Cl" in text


# ---------------------------------------------------------------------------
# write_qlm
# ---------------------------------------------------------------------------


def test_write_qlm_creates_file(tmp_path):
    """write_qlm produces a file that read_qlm can parse."""
    fpath_in = _write_tmp(_NACL_SITEX, tmp_path, "site_in.lm")
    cell_in, (ctx,) = read_qlm(fpath_in)

    fpath_out = str(tmp_path / "site_out.lm")
    write_qlm(fpath_out, cell_in, ctx)

    assert pathlib.Path(fpath_out).exists()
    cell_out, _ = read_qlm(fpath_out)
    isclose(cell_in, cell_out, atol=1e-7)


def test_write_qlm_without_context(tmp_path):
    """write_qlm without a QlmFl context creates a valid site file."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 5.64,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Na", "Cl"],
    )
    fpath = str(tmp_path / "site.lm")
    write_qlm(fpath, cell)
    assert pathlib.Path(fpath).exists()
    cell2, _ = read_qlm(fpath)
    assert cell2.symbols == ["Na", "Cl"]


# ---------------------------------------------------------------------------
# QlmFl.write_site with prefix and index
# ---------------------------------------------------------------------------


def test_write_site_with_prefix(tmp_path):
    """write_site with prefix creates supercell.lm."""
    fpath_in = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath_in)

    old_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx.write_site(cell, prefix="supercell")
        assert pathlib.Path(tmp_path / "supercell.lm").exists()
    finally:
        os.chdir(old_dir)


def test_write_site_with_prefix_and_index(tmp_path):
    """write_site with prefix + idx creates supercell-001.lm."""
    fpath_in = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath_in)

    old_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        ctx.write_site(cell, prefix="supercell", idx=1, width=3)
        assert pathlib.Path(tmp_path / "supercell-001.lm").exists()
    finally:
        os.chdir(old_dir)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path):
    """supercell.lm and supercell-001.lm … are created."""
    fpath_in = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath_in)

    old_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(cell, [cell, cell], [1, 2], ctx)
        assert pathlib.Path(tmp_path / "supercell.lm").exists()
        assert pathlib.Path(tmp_path / "supercell-001.lm").exists()
        assert pathlib.Path(tmp_path / "supercell-002.lm").exists()
    finally:
        os.chdir(old_dir)


def test_write_supercells_content_readable(tmp_path):
    """Written supercell files are parseable by read_qlm."""
    fpath_in = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath_in)

    old_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(cell, [cell], [1], ctx)
        cell2, _ = read_qlm(str(tmp_path / "supercell-001.lm"))
    finally:
        os.chdir(old_dir)

    np.testing.assert_allclose(cell.cell, cell2.cell, atol=1e-7)
    np.testing.assert_allclose(cell.scaled_positions, cell2.scaled_positions, atol=1e-7)


def test_write_supercells_custom_width(tmp_path):
    """Width parameter controls zero-padding in displacement filenames."""
    fpath_in = _write_tmp(_NACL_SITEX, tmp_path, "site.lm")
    cell, (ctx,) = read_qlm(fpath_in)

    old_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(cell, [cell], [5], ctx, width=4)
        assert pathlib.Path(tmp_path / "supercell-0005.lm").exists()
    finally:
        os.chdir(old_dir)
