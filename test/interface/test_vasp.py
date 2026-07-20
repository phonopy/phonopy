# SPDX-License-Identifier: BSD-3-Clause
"""Tests of VASP calculator interface."""

import lzma
import shutil
import tarfile
import tempfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pytest

import phonopy
from phonopy.file_IO import parse_FORCE_SETS, write_FORCE_SETS
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.vasp import (
    Vasprun,
    VasprunxmlExpat,
    check_forces,
    get_born_vaspout,
    get_born_vasprunxml,
    get_drift_forces,
    get_scaled_positions_lines,
    get_vasp_structure_lines,
    get_vasp_vca_hint_lines,
    parse_force_constants,
    parse_set_of_forces,
    parse_vasprunxml,
    read_vasp,
    read_vasp_from_strings,
    read_vaspout_calculation,
    read_vasprun_calculation,
    read_XDATCAR,
    write_supercells_with_displacements,
    write_vasp,
    write_XDATCAR,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    argsort_by_key,
    build_mixture_cell,
    group_by_key,
    isclose,
)

cwd = Path(__file__).parent

# ---------------------------------------------------------------------------
# Minimal inline POSCAR strings
# ---------------------------------------------------------------------------

# VASP5 format: species line between lattice and counts
_POSCAR_VASP5 = """\
NaCl test
   1.00000000000000
     5.6903014761756712    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.6903014761756712    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.6903014761756712
   Na   Cl
    4    4
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.5000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000  0.5000000000000000
"""

# VASP4 format: species in line 0, no species line before counts
_POSCAR_VASP4 = """\
Na Cl
   1.00000000000000
     5.6903014761756712    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.6903014761756712    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.6903014761756712
    4    4
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.5000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000  0.5000000000000000
"""

# Selective dynamics (extra "Selective dynamics" line)
_POSCAR_SELECTIVE = """\
NaCl selective dynamics
   1.00000000000000
     5.6903014761756712    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.6903014761756712    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.6903014761756712
   Na   Cl
    4    4
Selective dynamics
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.5000000000000000  0.5000000000000000  0.5000000000000000
  0.5000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.0000000000000000
  0.0000000000000000  0.0000000000000000  0.5000000000000000
"""

# Cartesian coordinates
_POSCAR_CARTESIAN = """\
NaCl Cartesian
   1.00000000000000
     5.6903014761756712    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.6903014761756712    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.6903014761756712
   Na   Cl
    1    1
Cartesian
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  2.8451507380878356  2.8451507380878356  2.8451507380878356
"""


# ---------------------------------------------------------------------------
# read_vasp
# ---------------------------------------------------------------------------


def test_read_vasp():
    """Test read_vasp."""
    cell = read_vasp(cwd / ".." / "POSCAR_NaCl")
    filename = cwd / "NaCl-vasp.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_read_vasp_vasp5_format():
    """VASP5 format (species line before counts) is parsed correctly."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    assert cell.symbols[:4] == ["Na", "Na", "Na", "Na"]
    assert cell.symbols[4:] == ["Cl", "Cl", "Cl", "Cl"]
    assert len(cell) == 8


def test_read_vasp_vasp4_format():
    """VASP4 format (species in comment line, no species line) is parsed."""
    cell = read_vasp_from_strings(_POSCAR_VASP4)
    assert cell.symbols[:4] == ["Na", "Na", "Na", "Na"]
    assert len(cell) == 8


def test_read_vasp_selective_dynamics():
    """Selective dynamics line is ignored; positions are read correctly."""
    cell = read_vasp_from_strings(_POSCAR_SELECTIVE)
    assert len(cell) == 8
    np.testing.assert_allclose(cell.scaled_positions[0], [0.0, 0.0, 0.0], atol=1e-10)


def test_read_vasp_cartesian():
    """Cartesian coordinate format is parsed and converted to fractional."""
    cell = read_vasp_from_strings(_POSCAR_CARTESIAN)
    assert len(cell) == 2
    np.testing.assert_allclose(cell.scaled_positions[0], [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(cell.scaled_positions[1], [0.5, 0.5, 0.5], atol=1e-10)


def test_read_vasp_scale_factor():
    """Scale factor (line 2) multiplies the lattice correctly."""
    poscar = _POSCAR_VASP5.replace("   1.00000000000000", "   2.00000000000000")
    cell = read_vasp_from_strings(poscar)
    a = 5.6903014761756712 * 2.0
    np.testing.assert_allclose(cell.cell[0, 0], a, atol=1e-7)


# ---------------------------------------------------------------------------
# check_forces / get_drift_forces
# ---------------------------------------------------------------------------


def test_check_forces_passes_correct_count():
    """check_forces returns True when force array length matches num_atom."""
    forces = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    assert check_forces(forces, 2, "test.xml", verbose=False) is True


def test_check_forces_fails_wrong_count():
    """check_forces returns False when force array length != num_atom."""
    forces = [[0.1, 0.0, 0.0]]
    assert check_forces(forces, 2, "test.xml", verbose=False) is False


def test_get_drift_forces_zero_for_balanced():
    """Drift force is zero when forces already sum to zero."""
    forces = np.array([[0.01, -0.01, 0.0], [-0.01, 0.01, 0.0]])
    drift = get_drift_forces(forces, verbose=False)
    np.testing.assert_allclose(drift, [0.0, 0.0, 0.0], atol=1e-10)


def test_get_drift_forces_nonzero():
    """Drift force equals mean of forces."""
    forces = np.array([[0.03, 0.0, 0.0], [0.01, 0.0, 0.0]])
    drift = get_drift_forces(forces, verbose=False)
    np.testing.assert_allclose(drift, [0.02, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# get_scaled_positions_lines
# ---------------------------------------------------------------------------


def test_get_scaled_positions_lines_format():
    """get_scaled_positions_lines returns one line per atom."""
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    text = get_scaled_positions_lines(pos)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2


def test_get_scaled_positions_lines_wraps_negative():
    """Negative fractional coordinates are mapped to [0, 1)."""
    pos = np.array([[-0.1, -0.5, -0.9]])
    text = get_scaled_positions_lines(pos)
    vals = [float(x) for x in text.split()]
    assert all(0.0 <= v < 1.0 for v in vals)


def test_get_scaled_positions_lines_near_zero_negative():
    """Tiny negative values (e.g. -1e-30) are mapped to 0, not 1."""
    pos = np.array([[-1e-30, 0.0, 0.0]])
    text = get_scaled_positions_lines(pos)
    val = float(text.split()[0])
    assert val < 1.0


# ---------------------------------------------------------------------------
# group_by_key
# ---------------------------------------------------------------------------


def test_argsort_by_key_already_sorted():
    """Symbols already grouped keep their order (identity permutation)."""
    assert argsort_by_key(["Na", "Na", "Cl", "Cl"]) == [0, 1, 2, 3]


def test_argsort_by_key_interleaved():
    """Interleaved symbols are stably grouped."""
    assert argsort_by_key(["Na", "Cl", "Na", "Cl"]) == [0, 2, 1, 3]


def test_argsort_by_key_first_occurrence_order():
    """Groups follow first-occurrence symbol order, stable within a group."""
    assert argsort_by_key(["Na", "Cl", "Na"]) == [0, 2, 1]


def test_group_by_key_already_sorted():
    """Symbols already grouped are returned in the same order."""
    counts, reduced, _ = group_by_key(["Na", "Na", "Cl", "Cl"])
    assert reduced == ["Na", "Cl"]
    assert counts == [2, 2]


def test_group_by_key_interleaved():
    """Interleaved symbols are stably grouped."""
    counts, reduced, _ = group_by_key(["Na", "Cl", "Na", "Cl"])
    assert reduced == ["Na", "Cl"]
    assert counts == [2, 2]


def test_group_by_key_sorts_positions():
    """Positions are permuted in first-occurrence symbol order."""
    # ["Na", "Cl", "Na"] → reduced = ["Na", "Cl"], perm = [0, 2, 1]
    symbols = ["Na", "Cl", "Na"]
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    counts, reduced, sorted_pos = group_by_key(symbols, positions)
    assert reduced == ["Na", "Cl"]
    assert counts == [2, 1]
    assert sorted_pos is not None
    np.testing.assert_allclose(sorted_pos[0], [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(sorted_pos[1], [0.1, 0.1, 0.1], atol=1e-10)
    np.testing.assert_allclose(sorted_pos[2], [0.5, 0.5, 0.5], atol=1e-10)


def test_group_by_key_none_positions():
    """When positions=None, sorted_positions is also None."""
    _, _, sorted_pos = group_by_key(["Na", "Cl"])
    assert sorted_pos is None


# ---------------------------------------------------------------------------
# get_vasp_structure_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_line_str, is_vasp4",
    [(None, True), (None, False), ("my_comment", True), ("my_comment", False)],
)
def test_get_vasp_structure_lines(helper_methods, is_vasp4, first_line_str):
    """Test get_vasp_structure_lines (almost write_vasp)."""
    filename = cwd / "NaCl-vasp.yaml"
    cell_ref = read_cell_yaml(filename)
    lines = get_vasp_structure_lines(
        cell_ref, direct=True, is_vasp4=is_vasp4, first_line_str=first_line_str
    )
    cell = read_vasp_from_strings("\n".join(lines))
    helper_methods.compare_cells_with_order(cell, cell_ref)
    if is_vasp4:
        # is_vasp4 is True, first_line_str is ignored.
        assert lines[0] == "Na Cl"
    elif first_line_str:
        assert lines[0] == first_line_str


def test_get_vasp_structure_lines_shuffled_positions(helper_methods):
    """Test get_vasp_structure_lines with a cell having shuffled positions.

    Order of atoms is sorted by chemical symbols. Therefore,
    helper_methods.compare_cells_with_order fails.

    """
    poscar_yaml = """lattice:
- [     5.690301476175672,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     5.690301476175672,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     5.690301476175672 ] # c
points:
- symbol: Na # 1
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 22.989769
- symbol: Cl # 2
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 35.453000
- symbol: Na # 3
  coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 22.989769
- symbol: Cl # 4
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 35.453000
- symbol: Na # 5
  coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 22.989769
- symbol: Cl # 6
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 35.453000
- symbol: Na # 7
  coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 22.989769
- symbol: Cl # 8
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 35.453000
  """
    cell_ref = read_cell_yaml(StringIO(poscar_yaml))
    lines = get_vasp_structure_lines(cell_ref, direct=True)
    cell = read_vasp_from_strings("\n".join(lines))
    with pytest.raises(AssertionError):
        helper_methods.compare_cells_with_order(cell, cell_ref)
    helper_methods.compare_cells(cell, cell_ref)
    perm = []
    for p_i in cell.scaled_positions:
        diff = cell_ref.scaled_positions - p_i
        diff -= np.rint(diff)
        dists = np.linalg.norm(diff @ cell.cell, axis=1)
        perm.append(np.where(dists < 1e-8)[0][0])
    np.testing.assert_array_equal(perm, [0, 2, 4, 6, 1, 3, 5, 7])
    np.testing.assert_array_equal(cell.numbers, [11, 11, 11, 11, 17, 17, 17, 17])


def test_get_vasp_structure_lines_is_vasp5_false_warns():
    """is_vasp5=False emits a DeprecationWarning."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    with pytest.warns(DeprecationWarning, match="is_vasp5"):
        get_vasp_structure_lines(cell, is_vasp5=False)


def test_get_vasp_structure_lines_direct_false_warns():
    """direct=False emits a DeprecationWarning."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    with pytest.warns(DeprecationWarning):
        get_vasp_structure_lines(cell, direct=False)


def test_get_vasp_structure_lines_default_comment():
    """Default first line is 'generated by phonopy' in VASP5 mode."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    lines = get_vasp_structure_lines(cell)
    assert lines[0] == "generated by phonopy"


def test_get_vasp_structure_lines_species_line_vasp5():
    """VASP5 output contains a species-name line before the counts line."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    lines = get_vasp_structure_lines(cell, is_vasp4=False)
    # Line 0: comment, lines 1-4: lattice, line 5: species, line 6: counts
    assert "Na" in lines[5] and "Cl" in lines[5]
    counts = [int(x) for x in lines[6].split()]
    assert counts == [4, 4]


# ---------------------------------------------------------------------------
# write_vasp
# ---------------------------------------------------------------------------


def test_write_vasp_creates_file(tmp_path):
    """write_vasp creates a POSCAR-style file parseable by read_vasp."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    fpath = tmp_path / "POSCAR"
    write_vasp(fpath, cell)
    assert fpath.exists()
    cell2 = read_vasp(fpath)
    assert isclose(cell, cell2)


def test_write_vasp_roundtrip(tmp_path):
    """write_vasp → read_vasp preserves structure."""
    cell_orig = read_vasp(cwd / ".." / "POSCAR_NaCl")
    fpath = tmp_path / "POSCAR"
    write_vasp(fpath, cell_orig)
    cell_rt = read_vasp(fpath)
    assert isclose(cell_orig, cell_rt)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_default_filenames(tmp_path):
    """SPOSCAR and POSCAR-001, POSCAR-002 are created by default."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    pre = tmp_path / "POSCAR"
    write_supercells_with_displacements(cell, [cell, cell], [1, 2], pre_filename=pre)
    assert (tmp_path / "SPOSCAR").exists()
    assert (tmp_path / "POSCAR-001").exists()
    assert (tmp_path / "POSCAR-002").exists()


def test_write_supercells_custom_prefix(tmp_path):
    """Custom pre_filename is used for all output files."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    write_supercells_with_displacements(cell, [cell], [1], pre_filename=tmp_path / "SC")
    assert (tmp_path / "SSC").exists()
    assert (tmp_path / "SC-001").exists()


def test_write_supercells_custom_width(tmp_path):
    """Width parameter controls zero-padding in filenames."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    write_supercells_with_displacements(
        cell, [cell], [3], pre_filename=tmp_path / "POSCAR", width=4
    )
    assert (tmp_path / "POSCAR-0003").exists()


def test_write_supercells_content_readable(tmp_path):
    """Written displacement files are parseable and match the original cell."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    write_supercells_with_displacements(
        cell, [cell], [1], pre_filename=tmp_path / "POSCAR"
    )
    cell2 = read_vasp(tmp_path / "POSCAR-001")
    assert isclose(cell, cell2)


# ---------------------------------------------------------------------------
# parse_vasprun_xml / Vasprun / VasprunxmlExpat
# ---------------------------------------------------------------------------


def test_parse_vasprun_xml():
    """Test parsing vasprun.xml with expat."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    filename = cwd / ".." / "FORCE_SETS_NaCl"
    dataset = parse_FORCE_SETS(filename=filename)
    energy_ref = [-216.82820693, -216.82817843]
    for i, member in enumerate(_tar.getmembers()):
        vr = Vasprun(_tar.extractfile(member), use_expat=True)
        ref = dataset["first_atoms"][i]["forces"]
        np.testing.assert_allclose(ref, vr.read_forces(), atol=1e-8)
        np.testing.assert_allclose(energy_ref[i], vr.read_energy(), atol=1e-8)


def test_VasprunxmlExpat():
    """Test VasprunxmlExpat."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    for member in _tar.getmembers():
        vasprun = VasprunxmlExpat(_tar.extractfile(member))
        vasprun.parse()
        np.testing.assert_equal(vasprun.fft_grid, [64, 64, 64])
        np.testing.assert_equal(vasprun.fft_fine_grid, [128, 128, 128])
        assert vasprun.efermi is None
        assert vasprun.symbols == ["Na"] * 32 + ["Cl"] * 32
        np.testing.assert_almost_equal(vasprun.NELECT, 448)
        np.testing.assert_almost_equal(vasprun.volume, 1473.99433936)
        break


def test_VasprunxmlExpat_characterization():
    """Lock current parser output as a regression baseline.

    These golden values guard against behavior changes while the
    VasprunxmlExpat internals are refactored. The fixture is a
    Gamma-point NaCl supercell calculation (no kpoints_opt).

    """
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    members = _tar.getmembers()
    assert [m.name for m in members] == ["vasprun.xml-001", "vasprun.xml-002"]

    vasprun = VasprunxmlExpat(_tar.extractfile(members[0]))
    vasprun.parse()

    # Scalars and grids.
    assert vasprun.NELECT == pytest.approx(448.0)
    assert vasprun.efermi is None
    np.testing.assert_equal(vasprun.fft_grid, [64, 64, 64])
    np.testing.assert_equal(vasprun.fft_fine_grid, [128, 128, 128])
    assert vasprun.symbols == ["Na"] * 32 + ["Cl"] * 32

    # Structure quantities.
    np.testing.assert_almost_equal(vasprun.volume, [1473.99433936])
    assert vasprun.forces.shape == (1, 64, 3)
    np.testing.assert_allclose(vasprun.forces[0, 0], [-0.01806194, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(
        vasprun.energies, [[-216.82820693, -216.82820693, 0.0]], atol=1e-8
    )

    # k-points and eigenvalues.
    np.testing.assert_equal(vasprun.k_mesh, [2, 2, 2])
    assert vasprun.kpointlist.shape == (1, 3)
    assert vasprun.k_weights.shape == (1,)
    np.testing.assert_allclose(np.sum(vasprun.k_weights), 1.0, atol=1e-10)
    assert vasprun.eigenvalues.shape == (1, 1, 268, 2)
    np.testing.assert_allclose(vasprun.eigenvalues[0, 0, 0], [-21.6488, 1.0], atol=1e-4)
    np.testing.assert_allclose(vasprun.eigenvalues[0, 0, 1, 0], -21.6487, atol=1e-4)

    # Absent in this fixture.
    assert vasprun.born.shape == (0,)
    assert vasprun.epsilon.shape == ()
    assert vasprun.pseudopotentials == []


def test_VasprunxmlExpat_kpoints_opt():
    """Test parsing of a KPOINTS_OPT calculation.

    The fixture is a TiSe2 calculation with a coarse SCF mesh (14
    irreducible k-points) and a denser kpoints_opt mesh (24
    irreducible k-points). The plain ``eigenvalues`` / ``k_weights``
    / ``efermi`` must return the SCF-mesh quantities, while the
    ``*_kpoints_opt`` properties return the dense-mesh quantities.

    The kpoints_opt eigenvalues carry occupations sourced from
    ``<projected_kpoints_opt>`` so the shape matches the SCF
    eigenvalues (..., 2).

    """
    filename = cwd / "vasprun_kpoints_opt.xml.xz"
    vxml = parse_vasprunxml(filename)

    # Stable structure quantities.
    assert vxml.NELECT == pytest.approx(24.0)
    np.testing.assert_almost_equal(vxml.volume, [33.23606544])
    assert vxml.symbols == ["Ti", "Ti"]

    # SCF mesh (plain properties).
    assert vxml.eigenvalues.shape == (1, 14, 18, 2)
    np.testing.assert_allclose(vxml.eigenvalues[0, 0, 0], [-47.8415, 1.0], atol=1e-4)
    np.testing.assert_allclose(vxml.eigenvalues[0, -1, -1], [12.0109, 0.0], atol=1e-4)
    assert vxml.k_weights.shape == (14,)
    np.testing.assert_allclose(np.sum(vxml.k_weights), 1.0, atol=1e-6)
    np.testing.assert_allclose(vxml.k_weights[0], 0.00925926, atol=1e-7)
    assert vxml.kpointlist.shape == (14, 3)
    np.testing.assert_equal(vxml.k_mesh, [6, 6, 3])
    assert vxml.efermi == pytest.approx(8.73771591)

    # kpoints_opt mesh (dedicated properties).
    assert vxml.has_kpoints_opt
    assert vxml.eigenvalues_kpoints_opt.shape == (1, 24, 18, 2)
    np.testing.assert_allclose(
        vxml.eigenvalues_kpoints_opt[0, 0, 0], [-47.8416, 1.0], atol=1e-4
    )
    np.testing.assert_allclose(
        vxml.eigenvalues_kpoints_opt[0, -1, -1], [12.13, 0.0], atol=1e-4
    )
    assert vxml.k_weights_kpoints_opt.shape == (24,)
    np.testing.assert_allclose(np.sum(vxml.k_weights_kpoints_opt), 1.0, atol=1e-6)
    np.testing.assert_allclose(vxml.k_weights_kpoints_opt[0], 0.00510204, atol=1e-7)
    assert vxml.kpointlist_kpoints_opt.shape == (24, 3)
    np.testing.assert_equal(vxml.k_mesh_kpoints_opt, [7, 7, 4])
    assert vxml.efermi_kpoints_opt == pytest.approx(8.73880881)


def test_VasprunxmlExpat_kpoints_opt_spin():
    """Test KPOINTS_OPT parsing for a spin-polarized (ISPIN=2) run.

    This exercises the spin dimension of the eigenvalue sets, both
    for the SCF mesh and for the kpoints_opt mesh whose occupations
    are merged per spin from <projected_kpoints_opt>.

    """
    filename = cwd / "vasprun_kpoints_opt_spin.xml.xz"
    vxml = parse_vasprunxml(filename)

    assert vxml.NELECT == pytest.approx(24.0)
    np.testing.assert_almost_equal(vxml.volume, [33.23606544])

    # SCF mesh has two spin channels.
    assert vxml.eigenvalues.shape == (2, 14, 18, 2)
    assert vxml.efermi == pytest.approx(8.73771247)

    # kpoints_opt mesh, both spins, occupations merged in.
    assert vxml.eigenvalues_kpoints_opt.shape == (2, 24, 18, 2)
    np.testing.assert_allclose(
        vxml.eigenvalues_kpoints_opt[0, 21, 17, 0], 12.4904, atol=1e-4
    )
    np.testing.assert_allclose(
        vxml.eigenvalues_kpoints_opt[1, 21, 17, 0], 12.2976, atol=1e-4
    )
    assert vxml.k_weights_kpoints_opt.shape == (24,)
    assert vxml.efermi_kpoints_opt == pytest.approx(8.73881616)


def test_VasprunxmlExpat_kpoints_opt_requires_projected():
    """kpoints_opt occupations must come from <projected_kpoints_opt>.

    When a kpoints_opt block is present but the projected block
    (carrying occupations) is absent, accessing the kpoints_opt
    eigenvalues must raise rather than silently fall back.

    """
    xml_str = b"""<modeling>
 <calculation>
  <eigenvalues_kpoints_opt comment="kpoints_opt">
   <kpoints>
    <varray name="weights">
     <v> 0.5 </v>
     <v> 0.5 </v>
    </varray>
   </kpoints>
   <eigenvalues>
    <array>
     <field>eigene</field>
     <set>
      <set comment="spin 1">
       <set comment="kpoint 1"><r> -1.0 </r></set>
       <set comment="kpoint 2"><r> -2.0 </r></set>
      </set>
     </set>
    </array>
   </eigenvalues>
  </eigenvalues_kpoints_opt>
  <eigenvalues>
   <array>
    <field>eigene</field><field>occ</field>
    <set>
     <set comment="spin 1">
      <set comment="kpoint 1"><r> -1.0 1.0 </r></set>
     </set>
    </set>
   </array>
  </eigenvalues>
 </calculation>
</modeling>
"""
    vxml = VasprunxmlExpat(BytesIO(xml_str))
    vxml.parse()
    with pytest.raises(RuntimeError):
        _ = vxml.eigenvalues_kpoints_opt


def test_parse_set_of_forces():
    """Test parse_set_of_forces."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    fps = [_tar.extractfile(member) for member in _tar.getmembers()]
    calc_dataset = parse_set_of_forces(64, fps)
    filename = cwd / ".." / "FORCE_SETS_NaCl"
    dataset = parse_FORCE_SETS(filename=filename)
    force_sets = [dataset["first_atoms"][i]["forces"] for i in (0, 1)]
    energy_ref = [-216.82820693, -216.82817843]
    np.testing.assert_allclose(
        calc_dataset["points"][0][0], [0.00087869, 0.0, 0.0], atol=1e-5
    )
    np.testing.assert_allclose(
        calc_dataset["points"][1][32], [0.25087869, 0.25, 0.25], atol=1e-5
    )
    np.testing.assert_allclose(force_sets, calc_dataset["forces"], atol=1e-8)
    np.testing.assert_allclose(
        energy_ref, calc_dataset["supercell_energies"], atol=1e-8
    )


# ---------------------------------------------------------------------------
# read_XDATCAR / write_XDATCAR
# ---------------------------------------------------------------------------


def test_read_XDATCAR():
    """Test read_XDATCAR."""
    filename_xdatcar = cwd / "XDATCAR-NaCl"
    lattice, positions = read_XDATCAR(filename_xdatcar)

    np.testing.assert_allclose(lattice, np.eye(3) * 22.562240, atol=1e-8)
    np.testing.assert_allclose(
        positions[0, 0], [0.99854664, 0.00265936, 0.00701660], atol=1e-8
    )
    np.testing.assert_allclose(
        positions[-1, -1], [0.75034458, 0.74972945, 0.87262656], atol=1e-8
    )


def test_write_XDATCAR():
    """Test write_XDATCAR."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    for _, member in enumerate(_tar.getmembers()):
        vasprun = VasprunxmlExpat(_tar.extractfile(member))
        vasprun.parse()
        break

    with tempfile.TemporaryFile() as fp:
        write_XDATCAR(vasprunxml_expat=vasprun, fileptr=fp)
        fp.seek(0)
        lattice, positions = read_XDATCAR(fileptr=fp)

    np.testing.assert_allclose(lattice, np.eye(3) * 11.38060295, atol=1e-8)
    np.testing.assert_allclose(positions[0, 0], [0.00087869, 0, 0], atol=1e-8)
    np.testing.assert_allclose(positions[-1, -1], [0.5, 0.5, 0.75], atol=1e-8)


# ---------------------------------------------------------------------------
# parse_force_constants
# ---------------------------------------------------------------------------


def test_read_force_constants_from_vasprun_xml():
    """Test reading force constants from vasprun.xml."""
    fc, symbols = parse_force_constants(cwd / "vasprun_fc.xml.xz")
    assert fc.shape == (64, 64, 3, 3)
    assert symbols == ["Na", "Cl"]
    assert fc[0, 1, 2, 2] == pytest.approx(0.0046225992999999995)


# ---------------------------------------------------------------------------
# expand_mixtures (VASP VCA-compatible writing)
# ---------------------------------------------------------------------------


def _gesn_unitcell() -> PhonopyAtoms:
    a = 2.82173
    return PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )


def _parse_species_and_counts(lines: list[str]) -> tuple[list[str], list[int]]:
    """Pull the species row and the count row from a VASP5 POSCAR."""
    return lines[5].split(), [int(x) for x in lines[6].split()]


def test_expand_mixtures_caseA_GeSn_50_50():
    """Single GeSn 50/50 mixture expands into Ge Sn / N N rows."""
    cell = build_mixture_cell(_gesn_unitcell(), [0.5, 0.5, 0.5, 0.5])
    lines = get_vasp_structure_lines(cell, expand_mixtures=True)
    species, counts = _parse_species_and_counts(lines)
    assert species == ["Ge", "Sn"]
    assert counts == [2, 2]
    # Same fractional positions repeat for each constituent.
    pos_lines = lines[8 : 8 + 4]
    assert pos_lines[0] == pos_lines[2]
    assert pos_lines[1] == pos_lines[3]


def test_expand_mixtures_caseB_pure_plus_mixture():
    """Si + GeSn 50/50: Si Ge Sn / 1 1 1."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        symbols=["Si", "Ge", "Sn"],
    )
    cell = build_mixture_cell(cell, [1.0, 0.5, 0.5])
    lines = get_vasp_structure_lines(cell, expand_mixtures=True)
    species, counts = _parse_species_and_counts(lines)
    assert species == ["Si", "Ge", "Sn"]
    assert counts == [1, 1, 1]


def test_expand_mixtures_caseC_two_distinct_ratios():
    """Two GeSn mixtures with different ratios get four POSCAR rows."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        symbols=["Ge", "Sn", "Ge", "Sn"],
    )
    cell = build_mixture_cell(cell, [0.5, 0.5, 0.25, 0.75])
    lines = get_vasp_structure_lines(cell, expand_mixtures=True)
    species, counts = _parse_species_and_counts(lines)
    assert species == ["Ge", "Sn", "Ge", "Sn"]
    assert counts == [1, 1, 1, 1]


def test_expand_mixtures_caseD_distinct_constituents():
    """GeSn + SiGe mixtures become two pairs of constituent rows."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        symbols=["Ge", "Sn", "Si", "Ge"],
    )
    cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])
    lines = get_vasp_structure_lines(cell, expand_mixtures=True)
    species, counts = _parse_species_and_counts(lines)
    # build_mixture_cell canonicalizes constituents to the order in which the
    # symbols first appear in the cell, so the Si-Ge site emits Ge before Si.
    assert species == ["Ge", "Sn", "Ge", "Si"]
    assert counts == [1, 1, 1, 1]


def test_expand_mixtures_no_op_for_pure_cell():
    """A cell without mixtures emits the same output regardless of the flag."""
    cell = read_vasp_from_strings(_POSCAR_VASP5)
    lines_off = get_vasp_structure_lines(cell, expand_mixtures=False)
    lines_on = get_vasp_structure_lines(cell, expand_mixtures=True)
    assert lines_off == lines_on


def test_expand_mixtures_round_trip_via_build_mixture_cell(tmp_path):
    """Round-trip an expanded POSCAR back through build_mixture_cell.

    Reading the per-element-expanded POSCAR yields the un-merged input;
    re-applying build_mixture_cell with the canonical weights recovers the
    original mixed cell.
    """
    cell = build_mixture_cell(_gesn_unitcell(), [0.5, 0.5, 0.5, 0.5])
    fpath = tmp_path / "POSCAR_expanded"
    write_vasp(fpath, cell, expand_mixtures=True)
    parsed = read_vasp(fpath)
    # parsed has 4 atoms (Ge, Ge, Sn, Sn) — the inverse of build_mixture_cell.
    assert len(parsed) == 4
    # Re-applying with the canonical 0.5 weights collapses back to the mixture.
    rt = build_mixture_cell(parsed, [0.5, 0.5, 0.5, 0.5])
    assert rt.has_mixtures
    assert rt.symbols == ["GeSn", "GeSn"]
    np.testing.assert_allclose(rt.scaled_positions, cell.scaled_positions, atol=1e-10)


def test_get_vasp_vca_hint_lines_caseA():
    """Hint reports Ge Sn rows, counts, and INCAR VCA = 0.5 0.5."""
    cell = build_mixture_cell(_gesn_unitcell(), [0.5, 0.5, 0.5, 0.5])
    text = "\n".join(get_vasp_vca_hint_lines(cell))
    assert "POSCAR species rows: Ge Sn" in text
    assert "POSCAR counts:       2 2" in text
    assert "VCA = 0.5 0.5" in text


# ---------------------------------------------------------------------------
# Real GeSn 99/1 VCA fixtures (vasprun.xml + phonopy_disp.yaml + FORCE_SETS)
# ---------------------------------------------------------------------------


def test_parse_set_of_forces_GeSn_vca_vasprun():
    """parse_set_of_forces reads 32 expanded forces from a real GeSn vasprun.xml."""
    with lzma.open(cwd / "GeSn-vca-vasprun-001.xml.xz", "rb") as fp:
        calc_dataset = parse_set_of_forces(32, [fp], verbose=False)

    assert "forces" in calc_dataset
    forces = calc_dataset["forces"]
    assert len(forces) == 1
    assert forces[0].shape == (32, 3)
    np.testing.assert_allclose(
        forces[0][0], [-0.00108926, -0.07349826, -0.07349826], atol=1e-8
    )


def test_GeSn_vca_FORCE_SETS_fixture_format():
    """Reference FORCE_SETS uses (gamma) layout: line1=32, atom_index in 1..16."""
    text = (cwd / "GeSn-vca-FORCE_SETS").read_text().splitlines()
    assert int(text[0].strip()) == 32  # n_expanded
    assert int(text[1].strip()) == 1  # n_disp
    # Find the atom-number line (first non-blank after header)
    body = [ln.strip() for ln in text[2:] if ln.strip()]
    atom_number = int(body[0])
    assert 1 <= atom_number <= 16  # site index, 1-based, in n_sites range


def test_parse_FORCE_SETS_GeSn_vca_fixture_expanded_mode():
    """parse_FORCE_SETS with natom=16 detects expanded mode and stores 32-row forces."""
    dataset = parse_FORCE_SETS(natom=16, filename=cwd / "GeSn-vca-FORCE_SETS")
    assert dataset["natom"] == 16  # restamped to site count
    assert len(dataset["first_atoms"]) == 1
    fa = dataset["first_atoms"][0]
    assert fa["forces"].shape == (32, 3)
    np.testing.assert_allclose(
        fa["forces"][0], [-0.00108926, -0.07349826, -0.07349826], atol=1e-8
    )


def test_GeSn_vca_vasprun_to_FORCE_SETS_matches_fixture(tmp_path):
    """Writing FORCE_SETS from parsed vasprun forces reproduces the fixture file.

    Loads phonopy_disp.yaml to get the mixture supercell + displacement
    metadata, parses vasprun.xml with the mixture-aware row count, and
    writes a FORCE_SETS via the same path the CLI uses. The output must
    match the bundled GeSn-vca-FORCE_SETS byte-for-byte (modulo
    whitespace) to confirm the end-to-end pipeline is wire-compatible.

    """
    ph = phonopy.load(str(cwd / "GeSn-vca-phonopy_disp.yaml"), produce_fc=False)
    assert ph.supercell.has_mixtures
    assert len(ph.supercell) == 16
    assert ph.dataset is not None
    assert len(ph.dataset["first_atoms"]) == 1

    with lzma.open(cwd / "GeSn-vca-vasprun-001.xml.xz", "rb") as fp:
        calc_dataset = parse_set_of_forces(32, [fp], verbose=False)
    forces = calc_dataset["forces"]
    assert forces[0].shape == (32, 3)

    # Mirror what create_FORCE_SETS does: stamp forces into dataset and write.
    for entry, f in zip(ph.dataset["first_atoms"], forces, strict=True):
        entry["forces"] = f
    out = tmp_path / "FORCE_SETS"
    write_FORCE_SETS(ph.dataset, filename=out)

    written = parse_FORCE_SETS(natom=16, filename=out)
    reference = parse_FORCE_SETS(natom=16, filename=cwd / "GeSn-vca-FORCE_SETS")
    assert written["natom"] == reference["natom"] == 16
    assert written["first_atoms"][0]["number"] == reference["first_atoms"][0]["number"]
    np.testing.assert_allclose(
        written["first_atoms"][0]["displacement"],
        reference["first_atoms"][0]["displacement"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        written["first_atoms"][0]["forces"],
        reference["first_atoms"][0]["forces"],
        atol=1e-9,
    )


def test_GeSn_vca_phonopy_load_builds_FC(tmp_path):
    """End-to-end: load phonopy_disp.yaml + FORCE_SETS, build FC with traditional FD.

    This is the CLI path ``phonopy-load phonopy_disp.yaml`` exercised
    in-process. A single-displacement dataset is enough for the
    finite-difference solver as long as the displaced sites cover one
    representative per primitive-cell orbit; that is the case for a
    GeSn 50/50 zincblende cell with 1 disp on site 0.

    """
    shutil.copy(cwd / "GeSn-vca-phonopy_disp.yaml", tmp_path / "phonopy_disp.yaml")
    shutil.copy(cwd / "GeSn-vca-FORCE_SETS", tmp_path / "FORCE_SETS")
    ph = phonopy.load(
        str(tmp_path / "phonopy_disp.yaml"),
        force_sets_filename=str(tmp_path / "FORCE_SETS"),
        fc_calculator="traditional",
    )
    assert ph.force_constants is not None
    n_sites = len(ph.supercell)
    # phonopy.load defaults to compact FC, shape (n_patom, n_satom, 3, 3).
    fc = ph.force_constants
    assert fc.ndim == 4
    assert fc.shape[1] == n_sites
    assert fc.shape[2:] == (3, 3)
    # Raw expanded forces are still in the dataset.
    assert ph.dataset["first_atoms"][0]["forces"].shape == (32, 3)


def test_GeSn_vca_phonopy_load_builds_FC_symfc(tmp_path):
    """Same as above, but through the default symfc fc_calculator path."""
    pytest.importorskip("symfc")
    shutil.copy(cwd / "GeSn-vca-phonopy_disp.yaml", tmp_path / "phonopy_disp.yaml")
    shutil.copy(cwd / "GeSn-vca-FORCE_SETS", tmp_path / "FORCE_SETS")
    ph = phonopy.load(
        str(tmp_path / "phonopy_disp.yaml"),
        force_sets_filename=str(tmp_path / "FORCE_SETS"),
        fc_calculator="symfc",
    )
    assert ph.force_constants is not None
    n_sites = len(ph.supercell)
    # phonopy.load defaults to compact FC, shape (n_patom, n_satom, 3, 3).
    fc = ph.force_constants
    assert fc.ndim == 4
    assert fc.shape[1] == n_sites
    assert fc.shape[2:] == (3, 3)


#
# vaspout.h5
#
def _write_minimal_vaspout(path, with_stress=True, with_nac=False):
    """Build a minimal vaspout.h5 with known values for reader tests."""
    h5py = pytest.importorskip("h5py")
    lattice = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    stress_kbar = np.array([[-15.0, 0.0, 0.0], [0.0, -15.0, 0.0], [0.0, 0.0, -15.0]])
    energies = np.array([[-10.0, -11.0, -12.0]])  # [free, wo_entropy, sigma->0]
    tags = np.array(
        [b"free energy    TOTEN", b"energy without entropy", b"energy(sigma->0)"]
    )
    with h5py.File(path, "w") as f:
        f["input/poscar/ion_types"] = np.array([b"Na", b"Cl"])
        f["input/poscar/number_ion_types"] = np.array([1, 1], dtype="int32")
        f["input/poscar/direct_coordinates"] = np.int32(1)
        f["input/poscar/scale"] = 1.0
        f["input/poscar/lattice_vectors"] = lattice
        f["input/poscar/position_ions"] = positions
        g = "intermediate/ion_dynamics"
        f[f"{g}/scale"] = 1.0
        f[f"{g}/lattice_vectors"] = lattice[None, :, :]
        f[f"{g}/position_ions"] = positions[None, :, :]
        f[f"{g}/forces"] = forces[None, :, :]
        if with_stress:
            f[f"{g}/stress"] = stress_kbar[None, :, :]
        f[f"{g}/energies"] = energies
        f[f"{g}/energies_tags"] = tags
        if with_nac:
            lr = "results/linear_response"
            born = np.array([np.eye(3) * 1.09, -np.eye(3) * 1.09], dtype="double")
            f[f"{lr}/born_charges"] = born
            f[f"{lr}/electron_dielectric_tensor"] = np.eye(3) * 2.5


def test_read_vaspout_calculation(tmp_path):
    """read_vaspout_calculation returns sigma->0 energy, forces, GPa stress."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path)
    cell, energy, forces, stress = read_vaspout_calculation(path)

    assert cell.symbols == ["Na", "Cl"]
    np.testing.assert_allclose(cell.cell, np.eye(3) * 4.0)
    assert energy == pytest.approx(-12.0)  # energy(sigma->0), not -11.0
    np.testing.assert_allclose(forces, [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    # kBar -> GPa
    np.testing.assert_allclose(stress, np.eye(3) * -1.5)


def test_read_vaspout_calculation_no_stress(tmp_path):
    """Stress is None when the vaspout.h5 has no stress dataset."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, with_stress=False)
    _, _, _, stress = read_vaspout_calculation(path)
    assert stress is None


def test_parse_set_of_forces_vaspout(tmp_path):
    """parse_set_of_forces dispatches to the h5 reader for .h5 files."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path)
    dataset = parse_set_of_forces(2, [path], verbose=False)
    np.testing.assert_allclose(
        dataset["forces"][0], [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    )
    assert dataset["supercell_energies"][0] == pytest.approx(-12.0)


def test_read_vasprun_calculation_dispatches_h5(tmp_path):
    """read_vasprun_calculation routes .h5 files to the h5 reader."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path)
    _, energy, _, _ = read_vasprun_calculation(path)
    assert energy == pytest.approx(-12.0)


def test_get_born_vaspout(tmp_path):
    """get_born_vaspout reads Born charges and dielectric tensor."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, with_nac=True)
    borns, epsilon, atom_indices = get_born_vaspout(path, is_symmetry=False)
    np.testing.assert_allclose(epsilon, np.eye(3) * 2.5)
    np.testing.assert_allclose(borns[0], np.eye(3) * 1.09)
    np.testing.assert_allclose(borns[1], -np.eye(3) * 1.09)
    np.testing.assert_array_equal(atom_indices, [0, 1])


def test_get_born_vasprunxml_dispatches_h5(tmp_path):
    """get_born_vasprunxml routes .h5 files to get_born_vaspout."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, with_nac=True)
    _, epsilon, _ = get_born_vasprunxml(path, is_symmetry=False)
    np.testing.assert_allclose(epsilon, np.eye(3) * 2.5)


def test_get_born_vaspout_without_lepsilon(tmp_path):
    """get_born_vaspout raises when linear_response data are absent."""
    pytest.importorskip("h5py")
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, with_nac=False)
    with pytest.raises(RuntimeError):
        get_born_vaspout(path)


def test_energy_sigma0_version_index():
    """energy_sigma0 selects the version-dependent column and needs a version."""
    vasprun = VasprunxmlExpat(BytesIO(b""))
    vasprun._all_energies = [[-10.0, -11.0, -12.0]]

    vasprun._version = "6.4.2"  # VASP 6: e_0_energy (index 2)
    assert vasprun.energy_sigma0 == pytest.approx(-12.0)

    vasprun._version = "5.4.4"  # VASP 5: e_wo_entrp slot (index 1)
    assert vasprun.energy_sigma0 == pytest.approx(-11.0)

    vasprun._version = None  # unknown version cannot pick a column
    with pytest.raises(RuntimeError):
        _ = vasprun.energy_sigma0
