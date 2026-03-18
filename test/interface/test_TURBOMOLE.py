"""Tests for TURBOMOLE calculator interface."""

import os
import pathlib

import numpy as np

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.turbomole import (
    parse_set_of_forces,
    read_turbomole,
    write_supercells_with_displacements,
    write_turbomole,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# 2-atom Si primitive cell (in Bohr, as TURBOMOLE uses atomic units).
_SI2_CONTROL = """\
$periodic 3
$lattice
   5.17898186576   0.00000000000   0.00000000000
   0.00000000000   5.17898186576   0.00000000000
   0.00000000000   0.00000000000   5.17898186576
$coord
   0.00000000000   0.00000000000   0.00000000000    si
   2.58949093288   2.58949093288   2.58949093288    si
$end
"""


def _make_gradient(coords, gradients):
    """Return a TURBOMOLE gradient file string for the given atoms."""
    lines = ["$grad          cartesian gradients"]
    lines.append(
        " cycle =      1    SCF energy =   -578.5931883878   |dE/dxyz| =  0.000007"
    )
    for c in coords:
        lines.append("  %16.10f  %16.10f  %16.10f" % tuple(c))
    for g in gradients:
        lines.append("  %16.10f  %16.10f  %16.10f" % tuple(g))
    lines.append("$end")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# read_turbomole (existing data file)
# ---------------------------------------------------------------------------


def test_read_turbomole():
    """read_turbomole returns correct cell, positions, and symbols."""
    cell = read_turbomole(cwd / "Si-TURBOMOLE-control")
    cell_ref = read_cell_yaml(cwd / "Si-TURBOMOLE.yaml")
    isclose(cell, cell_ref, atol=1e-5)


def test_read_turbomole_symbols(tmp_path):
    """read_turbomole converts lowercase element symbols (si → Si)."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)
    assert cell.symbols == ["Si", "Si"]


def test_read_turbomole_cell(tmp_path):
    """read_turbomole parses lattice vectors correctly."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)
    a = 5.17898186576
    np.testing.assert_allclose(cell.cell, np.diag([a, a, a]), atol=1e-10)


def test_read_turbomole_atom_count(tmp_path):
    """read_turbomole returns the correct number of atoms."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)
    assert len(cell) == 2


def test_read_turbomole_skips_coordinateupdate():
    """$coordinateupdate line must not be parsed as coordinates."""
    cell = read_turbomole(cwd / "Si-TURBOMOLE-control")
    assert len(cell) == 8  # only the 8 $coord atoms, not spurious extras


# ---------------------------------------------------------------------------
# write_turbomole
# ---------------------------------------------------------------------------


def test_write_turbomole_creates_directory(tmp_path):
    """write_turbomole creates the named directory."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    outdir = str(tmp_path / "supercell")
    write_turbomole(outdir, cell)
    assert pathlib.Path(outdir).is_dir()


def test_write_turbomole_creates_control_and_coord(tmp_path):
    """write_turbomole creates both control and coord files."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    outdir = str(tmp_path / "supercell")
    write_turbomole(outdir, cell)
    assert (pathlib.Path(outdir) / "control").exists()
    assert (pathlib.Path(outdir) / "coord").exists()


def test_write_turbomole_control_contains_lattice(tmp_path):
    """Control file written by write_turbomole contains $lattice block."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    outdir = str(tmp_path / "supercell")
    write_turbomole(outdir, cell)
    text = (pathlib.Path(outdir) / "control").read_text()
    assert "$lattice" in text
    assert "$end" in text


def test_write_turbomole_coord_contains_atoms(tmp_path):
    """Coord file written by write_turbomole contains $coord block with atoms."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    outdir = str(tmp_path / "supercell")
    write_turbomole(outdir, cell)
    text = (pathlib.Path(outdir) / "coord").read_text()
    assert "$coord" in text
    assert "si" in text  # write_turbomole uses lower-case symbols


def test_write_turbomole_on_existing_directory(tmp_path):
    """write_turbomole does not crash when the directory already exists."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    outdir = str(tmp_path / "supercell")
    write_turbomole(outdir, cell)
    # Second call must not raise
    write_turbomole(outdir, cell)
    assert pathlib.Path(outdir).is_dir()


def test_write_turbomole_roundtrip(tmp_path):
    """write_turbomole → read_turbomole (embedded $coord) preserves structure."""
    cell_orig = PhonopyAtoms(
        cell=np.diag([5.17898186576, 5.17898186576, 5.17898186576]),
        positions=[[0.0, 0.0, 0.0], [2.58949093288, 2.58949093288, 2.58949093288]],
        symbols=["Si", "Si"],
    )
    outdir = str(tmp_path / "sc")
    write_turbomole(outdir, cell_orig)

    # The control file uses "file=coord"; read_turbomole opens "coord" relative
    # to CWD, so we must chdir into the output directory.
    old_cwd = os.getcwd()
    try:
        os.chdir(outdir)
        cell_read = read_turbomole("control")
    finally:
        os.chdir(old_cwd)

    isclose(cell_orig, cell_read, atol=1e-7)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_default_filenames(tmp_path):
    """supercell/ and supercell-001/ directories are created."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(cell, [cell, cell], [1, 2])
        assert (tmp_path / "supercell").is_dir()
        assert (tmp_path / "supercell-001").is_dir()
        assert (tmp_path / "supercell-002").is_dir()
    finally:
        os.chdir(old_cwd)


def test_write_supercells_custom_prefix(tmp_path):
    """Custom pre_filename is used for all output directories."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(
            cell, [cell], [1], pre_filename="disp", width=3
        )
        assert (tmp_path / "disp").is_dir()
        assert (tmp_path / "disp-001").is_dir()
    finally:
        os.chdir(old_cwd)


def test_write_supercells_custom_width(tmp_path):
    """Width parameter controls zero-padding in directory names."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(
            cell, [cell], [5], pre_filename="supercell", width=4
        )
        assert (tmp_path / "supercell-0005").is_dir()
    finally:
        os.chdir(old_cwd)


def test_write_supercells_each_has_control_and_coord(tmp_path):
    """Each displacement directory contains both control and coord."""
    ctrl = tmp_path / "control"
    ctrl.write_text(_SI2_CONTROL)
    cell = read_turbomole(ctrl)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        write_supercells_with_displacements(cell, [cell], [1])
        d = tmp_path / "supercell-001"
        assert (d / "control").exists()
        assert (d / "coord").exists()
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# parse_set_of_forces
# ---------------------------------------------------------------------------


def test_parse_set_of_forces_single(tmp_path):
    """parse_set_of_forces returns correct (negated) forces for one file."""
    coords = [[0.0, 0.0, 0.0], [2.589, 2.589, 2.589]]
    grads = [[0.01, -0.01, 0.00], [-0.01, 0.01, 0.00]]
    dispdir = tmp_path / "supercell-001"
    dispdir.mkdir()
    (dispdir / "gradient").write_text(_make_gradient(coords, grads))

    frs = parse_set_of_forces(2, [str(dispdir)], verbose=False)

    assert len(frs) == 1
    # Forces = -gradients; drift correction applied (sum already zero here)
    np.testing.assert_allclose(frs[0][0], [-0.01, 0.01, 0.00], atol=1e-10)
    np.testing.assert_allclose(frs[0][1], [0.01, -0.01, 0.00], atol=1e-10)


def test_parse_set_of_forces_zero_drift(tmp_path):
    """Forces after drift correction sum to zero."""
    coords = [[0.0, 0.0, 0.0], [2.589, 2.589, 2.589]]
    grads = [[0.01, -0.01, 0.00], [-0.01, 0.01, 0.00]]
    dispdir = tmp_path / "d"
    dispdir.mkdir()
    (dispdir / "gradient").write_text(_make_gradient(coords, grads))

    frs = parse_set_of_forces(2, [str(dispdir)], verbose=False)
    np.testing.assert_allclose(frs[0].sum(axis=0), 0.0, atol=1e-10)


def test_parse_set_of_forces_fortran_d_notation(tmp_path):
    """Fortran D-exponent notation (1.0D-02) is parsed correctly."""
    dispdir = tmp_path / "d"
    dispdir.mkdir()
    gradient_content = """\
$grad          cartesian gradients
 cycle =      1    SCF energy =   -578.5931883878   |dE/dxyz| =  0.000007
  0.00000000000D+00   0.00000000000D+00   0.00000000000D+00
  2.58949093288D+00   2.58949093288D+00   2.58949093288D+00
  1.00000000000D-02  -1.00000000000D-02   0.00000000000D+00
 -1.00000000000D-02   1.00000000000D-02   0.00000000000D+00
$end
"""
    (dispdir / "gradient").write_text(gradient_content)

    frs = parse_set_of_forces(2, [str(dispdir)], verbose=False)
    assert len(frs) == 1
    np.testing.assert_allclose(frs[0][0], [-0.01, 0.01, 0.0], atol=1e-10)
    np.testing.assert_allclose(frs[0][1], [0.01, -0.01, 0.0], atol=1e-10)


def test_parse_set_of_forces_multiple(tmp_path):
    """parse_set_of_forces accumulates forces from multiple directories."""
    coords = [[0.0, 0.0, 0.0], [2.589, 2.589, 2.589]]
    grads1 = [[0.02, 0.00, 0.00], [-0.02, 0.00, 0.00]]
    grads2 = [[0.00, 0.03, 0.00], [0.00, -0.03, 0.00]]
    for name, g in [("d1", grads1), ("d2", grads2)]:
        d = tmp_path / name
        d.mkdir()
        (d / "gradient").write_text(_make_gradient(coords, g))

    frs = parse_set_of_forces(
        2, [str(tmp_path / "d1"), str(tmp_path / "d2")], verbose=False
    )

    assert len(frs) == 2
    for f in frs:
        assert f.shape == (2, 3)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
