"""Regression tests for GruneisenMesh and GruneisenBandStructure write_yaml."""

from __future__ import annotations

import gzip
import io
import lzma

from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.phonon.band_structure import get_band_qpoints

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_mesh(ph_nacl_gruneisen) -> PhonopyGruneisen:
    ph0, ph_minus, ph_plus = ph_nacl_gruneisen
    phg = PhonopyGruneisen(ph0, ph_minus, ph_plus)
    phg.set_mesh([4, 4, 4])
    return phg


def _make_band(ph_nacl_gruneisen) -> PhonopyGruneisen:
    paths = get_band_qpoints([[[0.05, 0.05, 0.05], [0.5, 0.5, 0.5]]], npoints=10)
    ph0, ph_minus, ph_plus = ph_nacl_gruneisen
    phg = PhonopyGruneisen(ph0, ph_minus, ph_plus)
    phg.set_band_structure(paths)
    return phg


def _check_mesh_content(text: str) -> None:
    assert "mesh: [     4,     4,     4 ]" in text
    assert "nqpoint: 10" in text
    assert "reciprocal_lattice:" in text
    assert "natom:" in text
    assert "phonon:" in text
    # first q-point
    assert "q-position: [  0.1250000,  0.1250000,  0.1250000 ]" in text
    assert "multiplicity: 2" in text
    assert "gruneisen:" in text
    assert "frequency:" in text


def _check_band_content(text: str) -> None:
    assert "nqpoint:" in text
    assert "npath:" in text
    assert "reciprocal_lattice:" in text
    assert "natom:" in text
    assert "path:" in text
    assert "q-position:" in text
    assert "gruneisen:" in text
    assert "frequency:" in text


# ---------------------------------------------------------------------------
# GruneisenMesh._write_yaml  (via StringIO)
# ---------------------------------------------------------------------------


def test_mesh_write_yaml_stringio(ph_nacl_gruneisen):
    """_write_yaml writes correct text content via StringIO."""
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    buf = io.StringIO()
    mesh._write_yaml(buf, comment=None)
    _check_mesh_content(buf.getvalue())


# ---------------------------------------------------------------------------
# GruneisenMesh.write_yaml  (plain / gzip / lzma)
# ---------------------------------------------------------------------------


def test_mesh_write_yaml_plain(ph_nacl_gruneisen, tmp_path):
    """write_yaml writes readable plain-text yaml."""
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    out = tmp_path / "gruneisen.yaml"
    mesh.write_yaml(filename=out)
    _check_mesh_content(out.read_text())


def test_mesh_write_yaml_gzip(ph_nacl_gruneisen, tmp_path):
    """write_yaml with compression='gzip' writes readable gzip yaml."""
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    out = tmp_path / "gruneisen.yaml.gz"
    mesh.write_yaml(filename=out, compression="gzip")
    with gzip.open(out, "rt") as f:
        _check_mesh_content(f.read())  # type: ignore[arg-type]


def test_mesh_write_yaml_lzma(ph_nacl_gruneisen, tmp_path):
    """write_yaml with compression='lzma' writes readable lzma yaml."""
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    out = tmp_path / "gruneisen.yaml.xz"
    mesh.write_yaml(filename=out, compression="lzma")
    with lzma.open(out, "rt") as f:
        _check_mesh_content(f.read())  # type: ignore[arg-type]


def test_mesh_write_yaml_default_filename(ph_nacl_gruneisen, tmp_path, monkeypatch):
    """write_yaml uses default filename 'gruneisen.yaml' when none given."""
    monkeypatch.chdir(tmp_path)
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    mesh.write_yaml()
    _check_mesh_content((tmp_path / "gruneisen.yaml").read_text())


def test_mesh_write_yaml_default_filename_gzip(
    ph_nacl_gruneisen, tmp_path, monkeypatch
):
    """write_yaml gzip uses default filename 'gruneisen.yaml.gz'."""
    monkeypatch.chdir(tmp_path)
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    mesh.write_yaml(compression="gzip")
    with gzip.open(tmp_path / "gruneisen.yaml.gz", "rt") as f:
        _check_mesh_content(f.read())  # type: ignore[arg-type]


def test_mesh_write_yaml_default_filename_lzma(
    ph_nacl_gruneisen, tmp_path, monkeypatch
):
    """write_yaml lzma uses default filename 'gruneisen.yaml.xz'."""
    monkeypatch.chdir(tmp_path)
    phg = _make_mesh(ph_nacl_gruneisen)
    mesh = phg._mesh
    assert mesh is not None
    mesh.write_yaml(compression="lzma")
    with lzma.open(tmp_path / "gruneisen.yaml.xz", "rt") as f:
        _check_mesh_content(f.read())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GruneisenBandStructure._write_yaml  (via StringIO)
# ---------------------------------------------------------------------------


def test_band_write_yaml_stringio(ph_nacl_gruneisen):
    """_write_yaml writes correct text content via StringIO."""
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    buf = io.StringIO()
    bs._write_yaml(buf, comment=None)
    _check_band_content(buf.getvalue())


# ---------------------------------------------------------------------------
# GruneisenBandStructure.write_yaml  (plain / gzip / lzma)
# ---------------------------------------------------------------------------


def test_band_write_yaml_plain(ph_nacl_gruneisen, tmp_path):
    """write_yaml writes readable plain-text yaml."""
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    out = tmp_path / "gruneisen.yaml"
    bs.write_yaml(filename=out)
    _check_band_content(out.read_text())


def test_band_write_yaml_gzip(ph_nacl_gruneisen, tmp_path):
    """write_yaml with compression='gzip' writes readable gzip yaml."""
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    out = tmp_path / "gruneisen.yaml.gz"
    bs.write_yaml(filename=out, compression="gzip")
    with gzip.open(out, "rt") as f:
        _check_band_content(f.read())  # type: ignore[arg-type]


def test_band_write_yaml_lzma(ph_nacl_gruneisen, tmp_path):
    """write_yaml with compression='lzma' writes readable lzma yaml."""
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    out = tmp_path / "gruneisen.yaml.xz"
    bs.write_yaml(filename=out, compression="lzma")
    with lzma.open(out, "rt") as f:
        _check_band_content(f.read())  # type: ignore[arg-type]


def test_band_write_yaml_default_filename(ph_nacl_gruneisen, tmp_path, monkeypatch):
    """write_yaml uses default filename 'gruneisen.yaml' when none given."""
    monkeypatch.chdir(tmp_path)
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    bs.write_yaml()
    _check_band_content((tmp_path / "gruneisen.yaml").read_text())


def test_band_write_yaml_default_filename_gzip(
    ph_nacl_gruneisen, tmp_path, monkeypatch
):
    """write_yaml gzip uses default filename 'gruneisen.yaml.gz'."""
    monkeypatch.chdir(tmp_path)
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    bs.write_yaml(compression="gzip")
    with gzip.open(tmp_path / "gruneisen.yaml.gz", "rt") as f:
        _check_band_content(f.read())  # type: ignore[arg-type]


def test_band_write_yaml_default_filename_lzma(
    ph_nacl_gruneisen, tmp_path, monkeypatch
):
    """write_yaml lzma uses default filename 'gruneisen.yaml.xz'."""
    monkeypatch.chdir(tmp_path)
    phg = _make_band(ph_nacl_gruneisen)
    bs = phg._band_structure
    assert bs is not None
    bs.write_yaml(compression="lzma")
    with lzma.open(tmp_path / "gruneisen.yaml.xz", "rt") as f:
        _check_band_content(f.read())  # type: ignore[arg-type]
