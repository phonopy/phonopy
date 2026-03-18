"""Tests for phonopy/phonon/animation.py."""

import pathlib

import pytest

from phonopy import Phonopy
from phonopy.phonon.animation import write_animation


def _get_dynamical_matrix(ph: Phonopy):
    """Return DynamicalMatrix after building it (pre-run at a non-Gamma q-point)."""
    ph.get_dynamical_matrix_at_q([0.5, 0.5, 0.5])
    dm = ph.dynamical_matrix
    assert dm is not None
    return dm


def test_write_animation_v_sim(ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path):
    """Test write_animation with v_sim format for NaCl."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "anime.ascii")
    result = write_animation(dm, q_point=[0, 0, 0], anime_type="v_sim", filename=fname)
    assert result == fname
    lines = pathlib.Path(fname).read_text().splitlines()
    assert lines[0] == "# Phonopy generated file for v_sim 3.6"
    # Lines 1-2: oriented lattice parameters (3 floats each)
    assert len(lines[1].split()) == 3
    assert len(lines[2].split()) == 3
    # Lines 3-4: atom positions with element symbol (2 atoms: Na and Cl)
    assert lines[3].split()[-1] == "Na"
    assert lines[4].split()[-1] == "Cl"
    # NaCl primitive cell has 2 atoms -> 6 modes
    assert sum(1 for line in lines if line.startswith("#metaData:")) == 6
    # Each metaData block ends with "# ]"
    assert sum(1 for line in lines if line.strip() == "# ]") == 6
    # q-point [0,0,0] is embedded in each metaData line
    assert all(
        "0.000000;0.000000;0.000000" in line
        for line in lines
        if line.startswith("#metaData:")
    )


def test_write_animation_v_sim_default_filename(
    ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path, monkeypatch
):
    """Test write_animation with v_sim format uses 'anime.ascii' as default filename."""
    monkeypatch.chdir(tmp_path)
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    result = write_animation(dm, q_point=[0, 0, 0], anime_type="v_sim")
    assert result == "anime.ascii"
    out = tmp_path / "anime.ascii"
    assert out.is_file()
    lines = out.read_text().splitlines()
    assert lines[0] == "# Phonopy generated file for v_sim 3.6"
    assert sum(1 for line in lines if line.startswith("#metaData:")) == 6


def test_write_animation_v_sim_nonzero_q(
    ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path
):
    """Test write_animation v_sim at a non-Gamma q-point."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "anime.ascii")
    q = [0.5, 0.5, 0.0]
    result = write_animation(dm, q_point=q, anime_type="v_sim", filename=fname)
    assert result == fname
    lines = pathlib.Path(fname).read_text().splitlines()
    assert lines[0] == "# Phonopy generated file for v_sim 3.6"
    assert sum(1 for line in lines if line.startswith("#metaData:")) == 6
    # q-point [0.5, 0.5, 0.0] is embedded in metaData lines
    assert all(
        "0.500000;0.500000;0.000000" in line
        for line in lines
        if line.startswith("#metaData:")
    )


def test_write_animation_v_sim_with_shift(
    ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path
):
    """Test write_animation v_sim with atomic position shift changes atom coords."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname_no_shift = str(tmp_path / "anime_no_shift.ascii")
    fname_shift = str(tmp_path / "anime_shifted.ascii")
    write_animation(dm, q_point=[0, 0, 0], anime_type="v_sim", filename=fname_no_shift)
    result = write_animation(
        dm,
        q_point=[0, 0, 0],
        anime_type="v_sim",
        shift=[0.5, 0.5, 0.5],
        filename=fname_shift,
    )
    assert result == fname_shift
    lines_no_shift = pathlib.Path(fname_no_shift).read_text().splitlines()
    lines_shift = pathlib.Path(fname_shift).read_text().splitlines()
    # Both files have the same structure
    assert len(lines_no_shift) == len(lines_shift)
    assert sum(1 for line in lines_shift if line.startswith("#metaData:")) == 6
    # Atom position lines (indices 3 and 4) differ due to shift
    assert lines_no_shift[3] != lines_shift[3]
    assert lines_no_shift[4] != lines_shift[4]


def test_write_animation_arc(ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path):
    """Test write_animation with arc format for NaCl."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "anime.arc")
    num_div = 10
    result = write_animation(
        dm,
        anime_type="arc",
        band_index=4,
        amplitude=1.0,
        num_div=num_div,
        filename=fname,
    )
    assert result == fname
    lines = pathlib.Path(fname).read_text().splitlines()
    assert lines[0] == "!BIOSYM archive 3"
    assert lines[1] == "PBC=ON"
    # Each frame ends with two "end" lines
    assert lines.count("end") == num_div * 2
    # Each frame has a PBC cell-parameter line (format: "PBC " + 6 floats)
    # "PBC=ON" at line 1 is excluded by requiring a space after "PBC"
    assert sum(1 for line in lines if line.startswith("PBC ")) == num_div
    # Each frame has 2 atom lines with CORE keyword (Na and Cl)
    assert sum(1 for line in lines if "CORE" in line) == num_div * 2
    # Both element symbols appear
    assert any(line.startswith("Na") for line in lines)
    assert any(line.startswith("Cl") for line in lines)


def test_write_animation_arc_none_type(ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path):
    """Test write_animation with anime_type=None falls back to arc format."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "anime.arc")
    num_div = 5
    result = write_animation(
        dm,
        anime_type=None,
        band_index=4,
        amplitude=1.0,
        num_div=num_div,
        filename=fname,
    )
    assert result == fname
    lines = pathlib.Path(fname).read_text().splitlines()
    assert lines[0] == "!BIOSYM archive 3"
    assert lines[1] == "PBC=ON"
    assert lines.count("end") == num_div * 2


def test_write_animation_xyz(ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path):
    """Test write_animation with xyz format for NaCl."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "anime.xyz")
    num_div = 10
    band_index = 4
    result = write_animation(
        dm,
        anime_type="xyz",
        band_index=band_index,
        amplitude=1.0,
        num_div=num_div,
        filename=fname,
    )
    assert result == fname
    lines = pathlib.Path(fname).read_text().splitlines()
    # Each frame: 1 atom-count line + 1 comment line + 2 atom lines = 4 lines
    assert len(lines) == num_div * 4
    # Each frame's first line is the atom count (2 atoms)
    for i in range(num_div):
        assert lines[i * 4] == "2"
    # Each frame's comment line contains band index and "generated by Phonopy"
    for i in range(num_div):
        comment = lines[i * 4 + 1]
        assert f"b {band_index}" in comment
        assert f"div {i} / {num_div}" in comment
        assert "generated by Phonopy" in comment
    # Both element symbols appear in atom coordinate lines
    atom_lines = [ln for ln in lines if ln.startswith("Na") or ln.startswith("Cl")]
    assert len(atom_lines) == num_div * 2


def test_write_animation_jmol(ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path):
    """Test write_animation with jmol format for NaCl."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "anime.xyz_jmol")
    num_modes = 6  # NaCl primitive cell: 2 atoms -> 6 modes
    result = write_animation(
        dm,
        anime_type="jmol",
        band_index=4,
        amplitude=10.0,
        num_div=20,
        filename=fname,
    )
    assert result == fname
    lines = pathlib.Path(fname).read_text().splitlines()
    # Each mode block: 1 atom-count line + 1 comment line + 2 atom lines = 4 lines
    assert len(lines) == num_modes * 4
    # Each block's first line is the atom count (2 atoms in NaCl primitive cell)
    for i in range(num_modes):
        assert lines[i * 4] == "2"
    # Each block's comment line lists band index and q-point
    for i in range(num_modes):
        comment = lines[i * 4 + 1]
        assert f"b {i + 1}" in comment
        assert "generated by Phonopy" in comment
    # Atom lines contain element symbols and 6 float columns (x y z ux uy uz)
    atom_lines = [ln for ln in lines if ln.startswith(("Na", "Cl"))]
    assert len(atom_lines) == num_modes * 2
    for al in atom_lines:
        parts = al.split()
        assert len(parts) == 7  # symbol + 3 position + 3 displacement


def test_write_animation_poscar(ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path):
    """Test write_animation with poscar format for NaCl."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    fname = str(tmp_path / "APOSCAR")
    num_div = 5
    result = write_animation(
        dm,
        anime_type="poscar",
        band_index=4,
        amplitude=1.0,
        num_div=num_div,
        filename=fname,
    )
    assert result == fname
    # POSCAR format writes one file per frame: APOSCAR-000, APOSCAR-001, ...
    for i in range(num_div):
        poscar = tmp_path / f"APOSCAR-{i:03d}"
        assert poscar.is_file()
        lines = poscar.read_text().splitlines()
        # POSCAR structure: title, scale, 3 lattice, species, counts, Direct, coords
        assert lines[0] == "generated by phonopy"
        assert lines[1].strip() == "1.0"
        # Species line contains Na and Cl
        assert "Na" in lines[5] and "Cl" in lines[5]
        # Counts line: 1 Na + 1 Cl
        assert lines[6].split() == ["1", "1"]
        assert lines[7] == "Direct"
        # 2 coordinate lines follow
        assert len(lines[8].split()) == 3
        assert len(lines[9].split()) == 3


def test_write_animation_invalid_type(ph_nacl_nonac: Phonopy):
    """Test write_animation raises RuntimeError for unknown anime_type."""
    dm = _get_dynamical_matrix(ph_nacl_nonac)
    with pytest.raises(RuntimeError, match="Animation format"):
        write_animation(dm, anime_type="unknown_format")


def test_write_animation_via_phonopy_api(
    ph_nacl_nonac: Phonopy, tmp_path: pathlib.Path
):
    """Test Phonopy.write_animation() API produces same output as write_animation."""
    ph = ph_nacl_nonac
    ph.get_dynamical_matrix_at_q([0.5, 0.5, 0.5])
    fname_api = str(tmp_path / "api_anime.ascii")
    fname_direct = str(tmp_path / "direct_anime.ascii")
    q = [0, 0, 0]
    ph.write_animation(q_point=q, anime_type="v_sim", filename=fname_api)
    write_animation(
        ph.dynamical_matrix, q_point=q, anime_type="v_sim", filename=fname_direct
    )  # type: ignore[arg-type]
    api_lines = pathlib.Path(fname_api).read_text().splitlines()
    direct_lines = pathlib.Path(fname_direct).read_text().splitlines()
    assert api_lines[0] == "# Phonopy generated file for v_sim 3.6"
    # Both calls must produce identical output
    assert api_lines == direct_lines
    assert sum(1 for line in api_lines if line.startswith("#metaData:")) == 6
