"""Tests of file_IO functions."""

from __future__ import annotations

import io
import os
import pathlib
import tempfile
from typing import Literal

import h5py
import numpy as np
import pytest

import phonopy
from phonopy.file_IO import (
    check_force_constants_indices,
    collect_forces,
    get_BORN_lines,
    get_dataset_type2,
    get_FORCE_CONSTANTS_lines,
    get_FORCE_SETS_lines,
    get_io_module_to_decompress,
    is_file_phonopy_yaml,
    iter_collect_forces,
    parse_BORN,
    parse_BORN_from_strings,
    parse_FORCE_CONSTANTS,
    parse_FORCE_SETS,
    parse_FORCE_SETS_from_strings,
    read_force_constants_hdf5,
    read_v_e,
    write_FORCE_CONSTANTS,
    write_force_constants_to_hdf5,
    write_FORCE_SETS,
)

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


# ---------------------------------------------------------------------------
# parse_BORN / parse_BORN_from_strings
# ---------------------------------------------------------------------------


def test_parse_BORN():
    """Test of parse_BORN."""
    ph = phonopy.load(cwd / "phonopy_disp_NaCl.yaml")
    nac_params = parse_BORN(ph.primitive, filename=cwd / "BORN_NaCl")
    z = np.eye(3) * 1.086875
    epsilon = np.eye(3) * 2.43533967
    np.testing.assert_allclose(nac_params["born"], [z, -z], atol=1e-5)
    np.testing.assert_allclose(nac_params["dielectric"], epsilon, atol=1e-5)
    assert pytest.approx(14.400) == nac_params["factor"]


def test_parse_BORN_from_strings_nacl():
    """parse_BORN_from_strings gives same result as parse_BORN."""
    ph = phonopy.load(cwd / "phonopy_disp_NaCl.yaml")
    born_text = (cwd / "BORN_NaCl").read_text()
    nac_params = parse_BORN_from_strings(born_text, ph.primitive)
    z = np.eye(3) * 1.086875
    epsilon = np.eye(3) * 2.43533967
    np.testing.assert_allclose(nac_params["born"], [z, -z], atol=1e-5)
    np.testing.assert_allclose(nac_params["dielectric"], epsilon, atol=1e-5)


def test_get_BORN_lines_roundtrip():
    """BORN lines written and re-parsed reproduce original values."""
    ph = phonopy.load(cwd / "phonopy_disp_NaCl.yaml")
    nac_params = parse_BORN(ph.primitive, filename=cwd / "BORN_NaCl")
    lines = get_BORN_lines(ph.primitive, nac_params["born"], nac_params["dielectric"])
    text = "\n".join(lines)
    nac2 = parse_BORN_from_strings(text, ph.primitive)
    np.testing.assert_allclose(nac2["born"], nac_params["born"], atol=1e-5)
    np.testing.assert_allclose(nac2["dielectric"], nac_params["dielectric"], atol=1e-5)


# ---------------------------------------------------------------------------
# FORCE_SETS type1
# ---------------------------------------------------------------------------


def test_parse_FORCE_SETS_type1():
    """parse_FORCE_SETS reads type1 FORCE_SETS correctly."""
    dataset = parse_FORCE_SETS(filename=cwd / "FORCE_SETS_NaCl")
    assert dataset["natom"] == 64
    assert len(dataset["first_atoms"]) == 2
    fa = dataset["first_atoms"][0]
    assert fa["number"] == 0
    assert fa["forces"].shape == (64, 3)


def test_parse_FORCE_SETS_from_strings_type1():
    """parse_FORCE_SETS_from_strings gives same result as parse_FORCE_SETS."""
    text = (cwd / "FORCE_SETS_NaCl").read_text()
    dataset = parse_FORCE_SETS_from_strings(text)
    ref = parse_FORCE_SETS(filename=cwd / "FORCE_SETS_NaCl")
    assert dataset["natom"] == ref["natom"]
    np.testing.assert_allclose(
        dataset["first_atoms"][0]["forces"],
        ref["first_atoms"][0]["forces"],
        atol=1e-10,
    )


def test_get_FORCE_SETS_lines_type1_roundtrip():
    """FORCE_SETS lines written from dataset reproduce original dataset."""
    dataset = parse_FORCE_SETS(filename=cwd / "FORCE_SETS_NaCl")
    lines = get_FORCE_SETS_lines(dataset)
    text = "\n".join(lines)
    dataset2 = parse_FORCE_SETS_from_strings(text)
    assert dataset2["natom"] == dataset["natom"]
    assert len(dataset2["first_atoms"]) == len(dataset["first_atoms"])
    for fa1, fa2 in zip(dataset["first_atoms"], dataset2["first_atoms"], strict=True):
        assert fa1["number"] == fa2["number"]
        np.testing.assert_allclose(fa1["forces"], fa2["forces"], atol=1e-9)


def test_write_read_FORCE_SETS_type1(tmp_path):
    """write_FORCE_SETS and parse_FORCE_SETS roundtrip."""
    dataset = parse_FORCE_SETS(filename=cwd / "FORCE_SETS_NaCl")
    out = tmp_path / "FORCE_SETS"
    write_FORCE_SETS(dataset, filename=out)
    dataset2 = parse_FORCE_SETS(filename=out)
    assert dataset2["natom"] == dataset["natom"]
    np.testing.assert_allclose(
        dataset2["first_atoms"][0]["forces"],
        dataset["first_atoms"][0]["forces"],
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# FORCE_SETS type2
# ---------------------------------------------------------------------------

_TYPE2_TEXT = """\
  0.01000000  0.00000000  0.00000000 -0.01806194  0.00000000  0.00000000
  0.00000000  0.00000000  0.00000000  0.00302404  0.00000000  0.00000000
  0.01000000  0.00000000  0.00000000 -0.01500000  0.00100000  0.00000000
  0.00000000  0.00000000  0.00000000  0.00250000  0.00000000  0.00000000
"""


def test_get_dataset_type2():
    """get_dataset_type2 with natom=2 produces correct shape."""
    f = io.StringIO(_TYPE2_TEXT)
    dataset = get_dataset_type2(f, natom=2)
    assert dataset["displacements"].shape == (2, 2, 3)
    assert dataset["forces"].shape == (2, 2, 3)
    assert dataset["displacements"].dtype == np.float64
    np.testing.assert_allclose(dataset["displacements"][0, 0], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(dataset["forces"][0, 0], [-0.01806194, 0.0, 0.0])


def test_parse_FORCE_SETS_from_strings_type2():
    """parse_FORCE_SETS_from_strings handles type2 format."""
    dataset = parse_FORCE_SETS_from_strings(_TYPE2_TEXT, natom=2)
    assert "displacements" in dataset
    assert dataset["displacements"].shape == (2, 2, 3)


def test_get_FORCE_SETS_lines_type2_roundtrip():
    """type2 FORCE_SETS roundtrip via get_FORCE_SETS_lines."""
    f = io.StringIO(_TYPE2_TEXT)
    dataset = get_dataset_type2(f, natom=2)
    lines = get_FORCE_SETS_lines(dataset)
    text2 = "\n".join(lines)
    f2 = io.StringIO(text2)
    dataset2 = get_dataset_type2(f2, natom=2)
    np.testing.assert_allclose(
        dataset2["displacements"], dataset["displacements"], atol=1e-7
    )
    np.testing.assert_allclose(dataset2["forces"], dataset["forces"], atol=1e-7)


# ---------------------------------------------------------------------------
# FORCE_CONSTANTS text format
# ---------------------------------------------------------------------------


def test_get_FORCE_CONSTANTS_lines_full():
    """FORCE_CONSTANTS lines for full (N,N,3,3) fc."""
    fc = np.zeros((4, 4, 3, 3))
    fc[0, 1, 0, 0] = 1.5
    lines = get_FORCE_CONSTANTS_lines(fc)
    assert lines[0].strip() == "4    4"
    text = "\n".join(lines)
    assert "1.500000000000000" in text


def test_get_FORCE_CONSTANTS_lines_compact():
    """FORCE_CONSTANTS lines for compact (n_prim,N,3,3) fc use p2s_map."""
    fc = np.zeros((2, 4, 3, 3))
    p2s_map = np.array([0, 2], dtype="int64")
    lines = get_FORCE_CONSTANTS_lines(fc, p2s_map=p2s_map)
    assert lines[0].strip() == "2    4"
    # Collect all index lines (those with exactly 2 integer tokens)
    index_lines = [ln for ln in lines[1:] if len(ln.split()) == 2]
    first_indices = [int(ln.split()[0]) for ln in index_lines]
    # First block: all rows have first index 1 (0-based 0 → 1-based 1)
    assert first_indices[:4] == [1, 1, 1, 1]
    # Second block: all rows have first index 3 (0-based 2 → 1-based 3)
    assert first_indices[4:] == [3, 3, 3, 3]


def test_write_parse_FORCE_CONSTANTS_roundtrip(tmp_path):
    """write_FORCE_CONSTANTS / parse_FORCE_CONSTANTS roundtrip."""
    rng = np.random.default_rng(0)
    fc = rng.random((4, 4, 3, 3))
    out = tmp_path / "FORCE_CONSTANTS"
    write_FORCE_CONSTANTS(fc, filename=out)
    fc2 = parse_FORCE_CONSTANTS(filename=out)
    np.testing.assert_allclose(fc2, fc, atol=1e-14)


def test_write_parse_FORCE_CONSTANTS_compact_roundtrip(tmp_path):
    """Compact fc write/parse roundtrip with p2s_map."""
    rng = np.random.default_rng(1)
    fc = rng.random((2, 4, 3, 3))
    p2s_map = np.array([0, 2], dtype="int64")
    out = tmp_path / "FORCE_CONSTANTS"
    write_FORCE_CONSTANTS(fc, filename=out, p2s_map=p2s_map)
    fc2 = parse_FORCE_CONSTANTS(filename=out, p2s_map=p2s_map)
    np.testing.assert_allclose(fc2, fc, atol=1e-14)


# ---------------------------------------------------------------------------
# force_constants.hdf5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["gzip", "lzf", 1, 2, None])
def test_write_force_constants_to_hdf5(
    compression: Literal["gzip", "lzf"] | int | None,
):
    """Test write_force_constants_to_hdf5."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        write_force_constants_to_hdf5(
            np.zeros(1), physical_unit="eV/angstrom^2", compression=compression
        )
        with h5py.File("force_constants.hdf5", "r") as f:
            fc = f["force_constants"]
            if compression in ("gzip", "lzf"):
                assert fc.compression == compression  # type: ignore
            elif isinstance(compression, int):
                assert fc.compression == "gzip"  # type: ignore
                assert fc.compression_opts == compression  # type: ignore
            else:
                assert fc.compression is None  # type: ignore

        for created_filename in ["force_constants.hdf5"]:
            file_path = pathlib.Path(created_filename)
            assert file_path.exists()
            fc, physical_unit = read_force_constants_hdf5(
                file_path, return_physical_unit=True
            )
            assert fc[0] == pytest.approx(0)
            assert physical_unit == "eV/angstrom^2"
            file_path.unlink()

        _check_no_files()

        os.chdir(original_cwd)


def test_write_read_force_constants_hdf5_with_p2s_map(tmp_path):
    """p2s_map is stored and verified on read."""
    rng = np.random.default_rng(2)
    fc = rng.random((2, 4, 3, 3))
    p2s_map = np.array([0, 2], dtype="int64")
    out = tmp_path / "fc.hdf5"
    write_force_constants_to_hdf5(fc, filename=str(out), p2s_map=p2s_map)
    fc2 = read_force_constants_hdf5(out, p2s_map=p2s_map)
    np.testing.assert_allclose(fc2, fc, atol=1e-15)


# ---------------------------------------------------------------------------
# collect_forces / iter_collect_forces
# ---------------------------------------------------------------------------

_COLLECT_FORCES_TEXT = """\
Some header text
cartesian forces (eV/Angstrom) at end:
  1  -0.10000000   0.20000000  -0.30000000
  2   0.10000000  -0.20000000   0.30000000
More text after
"""


def test_collect_forces_basic():
    """collect_forces extracts forces from text."""
    f = io.StringIO(_COLLECT_FORCES_TEXT)
    forces = collect_forces(
        f,
        num_atom=2,
        hook="cartesian forces (eV/Angstrom)",
        force_pos=[1, 2, 3],
    )
    assert len(forces) == 2
    np.testing.assert_allclose(forces[0], [-0.1, 0.2, -0.3])
    np.testing.assert_allclose(forces[1], [0.1, -0.2, 0.3])


def test_collect_forces_missing_hook():
    """collect_forces returns empty list when hook is not found."""
    f = io.StringIO("no hook here\n1 0.1 0.2 0.3\n")
    forces = collect_forces(f, num_atom=1, hook="MISSING", force_pos=[1, 2, 3])
    assert forces == []


def test_collect_forces_with_word_filter():
    """collect_forces with word filter skips non-matching lines."""
    text = "hook\n  1  0.1  0.2  0.3\n  SKIP  0.9  0.9  0.9\n  2  0.4  0.5  0.6\n"
    f = io.StringIO(text)
    forces = collect_forces(f, num_atom=2, hook="hook", force_pos=[1, 2, 3], word="  ")
    assert len(forces) == 2


def test_iter_collect_forces(tmp_path):
    """iter_collect_forces returns the last set of forces in file."""
    text = (
        "cartesian forces (eV/Angstrom) at end:\n"
        "  1  -0.10  0.20  -0.30\n"
        "  2   0.10  -0.20  0.30\n"
        "cartesian forces (eV/Angstrom) at end:\n"
        "  1  -0.50  0.60  -0.70\n"
        "  2   0.50  -0.60  0.70\n"
    )
    f = tmp_path / "output.txt"
    f.write_text(text)
    forces = iter_collect_forces(
        f, num_atom=2, hook="cartesian forces (eV/Angstrom)", force_pos=[1, 2, 3]
    )
    # Returns last set
    np.testing.assert_allclose(forces[0], [-0.5, 0.6, -0.7])
    np.testing.assert_allclose(forces[1], [0.5, -0.6, 0.7])


# ---------------------------------------------------------------------------
# check_force_constants_indices
# ---------------------------------------------------------------------------


def test_check_force_constants_indices_consistent():
    """No error when p2s_map matches indices."""
    p2s_map = np.array([0, 2], dtype="int64")
    indices = np.array([0, 2], dtype="int64")
    check_force_constants_indices((2, 4), indices, p2s_map, "test.hdf5")


def test_check_force_constants_indices_full_fc():
    """No error for full (square) fc even with mismatching p2s_map."""
    p2s_map = np.array([0, 1], dtype="int64")
    indices = np.array([0, 2], dtype="int64")
    check_force_constants_indices((4, 4), indices, p2s_map, "test.hdf5")


def test_check_force_constants_indices_inconsistent():
    """RuntimeError when p2s_map mismatches indices in compact fc."""
    p2s_map = np.array([0, 1], dtype="int64")
    indices = np.array([0, 2], dtype="int64")
    with pytest.raises(RuntimeError, match="p2s_map in primitive"):
        check_force_constants_indices((2, 4), indices, p2s_map, "test.hdf5")


# ---------------------------------------------------------------------------
# is_file_phonopy_yaml
# ---------------------------------------------------------------------------


def test_is_file_phonopy_yaml_true():
    """phonopy_params yaml is identified as phonopy yaml."""
    assert is_file_phonopy_yaml(cwd / "phonopy_params_Si.yaml") is True


def test_is_file_phonopy_yaml_false():
    """FORCE_SETS is not a phonopy yaml."""
    assert is_file_phonopy_yaml(cwd / "FORCE_SETS_NaCl") is False


def test_is_file_phonopy_yaml_xz():
    """Compressed xz file is handled correctly."""
    assert is_file_phonopy_yaml(cwd / "phonopy_params_NaCl-1.00.yaml.xz") is True


# ---------------------------------------------------------------------------
# get_io_module_to_decompress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected_module",
    [
        ("file.xz", "lzma"),
        ("file.lzma", "lzma"),
        ("file.gz", "gzip"),
        ("file.bz2", "bz2"),
        ("file.yaml", "io"),
        ("file.txt", "io"),
    ],
)
def test_get_io_module_to_decompress(filename, expected_module):
    """Correct module is returned for each extension."""
    mod = get_io_module_to_decompress(filename)
    assert mod.__name__ == expected_module


# ---------------------------------------------------------------------------
# read_v_e
# ---------------------------------------------------------------------------


def test_read_v_e():
    """read_v_e reads volumes and energies correctly."""
    filename = cwd / "cui/phonopy_qha/Cu-QHA/e-v.dat"
    volumes, energies = read_v_e(filename)
    assert len(volumes) == len(energies)
    assert len(volumes) > 0
    assert volumes[0] == pytest.approx(43.0804791127649, rel=1e-6)
    assert energies[0] == pytest.approx(-17.27885993, rel=1e-6)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())
