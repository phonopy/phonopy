"""Tests for phonopy.qha.lattice_sampling."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.qha.lattice_sampling import (
    build_random_displacement_supercells,
    build_strain_cells_manifest,
    get_free_lattice_dof,
    sample_strained_cells,
    write_strain_cells_manifest,
)
from phonopy.structure.atoms import PhonopyAtoms


def _cubic() -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=["Cu"], cell=np.diag([4.0, 4.0, 4.0]), scaled_positions=[[0, 0, 0]]
    )


def _tetragonal() -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=["Cu"], cell=np.diag([4.0, 4.0, 6.0]), scaled_positions=[[0, 0, 0]]
    )


def _hexagonal() -> PhonopyAtoms:
    a, c = 4.0, 6.0
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    return PhonopyAtoms(symbols=["Cu"], cell=cell, scaled_positions=[[0, 0, 0]])


def _orthorhombic() -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=["Cu"], cell=np.diag([4.0, 5.0, 6.0]), scaled_positions=[[0, 0, 0]]
    )


def _monoclinic() -> PhonopyAtoms:
    cell = np.array([[4.0, 0, 0], [0, 5.0, 0], [1.0, 0, 6.0]])
    return PhonopyAtoms(symbols=["Cu"], cell=cell, scaled_positions=[[0, 0, 0]])


def test_dof_cubic() -> None:
    """A cubic cell has a single length DOF tying all three axes."""
    dof = get_free_lattice_dof(_cubic())
    assert dof.crystal_system == "cubic"
    assert dof.labels == ("a",)
    assert dof.rows == {"a": (0, 1, 2)}
    assert dof.current_lengths["a"] == pytest.approx(4.0)
    assert dof.tie_description == "b = c = a"


@pytest.mark.parametrize("cell_func", [_tetragonal, _hexagonal])
def test_dof_two_length(cell_func) -> None:
    """Tetragonal and hexagonal cells have two DOF with c the unique axis."""
    dof = get_free_lattice_dof(cell_func())
    assert dof.labels == ("a", "c")
    assert dof.rows["a"] == (0, 1)
    assert dof.rows["c"] == (2,)
    assert dof.current_lengths["a"] == pytest.approx(4.0)
    assert dof.current_lengths["c"] == pytest.approx(6.0)
    assert dof.tie_description == "b = a"


def test_dof_orthorhombic() -> None:
    """An orthorhombic cell has three independent length DOF."""
    dof = get_free_lattice_dof(_orthorhombic())
    assert dof.crystal_system == "orthorhombic"
    assert dof.labels == ("a", "b", "c")
    assert dof.rows == {"a": (0,), "b": (1,), "c": (2,)}
    assert dof.tie_description == ""


def test_dof_monoclinic_rejected() -> None:
    """Monoclinic and triclinic crystals raise ValueError."""
    with pytest.raises(ValueError):
        get_free_lattice_dof(_monoclinic())


def test_sample_hexagonal() -> None:
    """Sampling preserves b = a and fractional positions and honors ranges."""
    cell = _hexagonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    cells = sample_strained_cells(cell, dof, ranges, num=8, seed=0)

    assert len(cells) == 8
    for c in cells:
        lengths = np.linalg.norm(c.cell, axis=1)
        # b = a preserved.
        np.testing.assert_allclose(lengths[0], lengths[1], rtol=1e-12)
        assert 3.9 <= lengths[0] <= 4.1
        assert 5.8 <= lengths[2] <= 6.2
        # Fractional positions unchanged.
        np.testing.assert_allclose(c.scaled_positions, cell.scaled_positions)


def test_sample_reproducible() -> None:
    """The same seed reproduces the same cells."""
    cell = _tetragonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    a = sample_strained_cells(cell, dof, ranges, num=5, seed=42)
    b = sample_strained_cells(cell, dof, ranges, num=5, seed=42)
    for ca, cb in zip(a, b, strict=True):
        np.testing.assert_allclose(ca.cell, cb.cell, rtol=1e-15)


def test_sample_invalid_ranges() -> None:
    """Wrong range keys or inverted ranges raise ValueError."""
    cell = _tetragonal()
    dof = get_free_lattice_dof(cell)
    with pytest.raises(ValueError):
        sample_strained_cells(cell, dof, {"a": (3.9, 4.1)}, num=3)  # missing c
    with pytest.raises(ValueError):
        sample_strained_cells(cell, dof, {"a": (3.9, 4.1), "b": (1, 2)}, num=3)
    with pytest.raises(ValueError):
        sample_strained_cells(cell, dof, {"a": (4.1, 3.9), "c": (5.8, 6.2)}, num=3)


def test_random_displacement_supercells() -> None:
    """RD supercells have the expected size and distinct displacements."""
    cell = _tetragonal()
    dof = get_free_lattice_dof(cell)
    unitcells = sample_strained_cells(
        cell, dof, {"a": (3.9, 4.1), "c": (5.8, 6.2)}, 3, seed=0
    )
    supercell_matrix = np.diag([2, 2, 2])
    supercells = build_random_displacement_supercells(
        unitcells, supercell_matrix, distance=0.1, seed=0
    )

    assert len(supercells) == 3
    for sc, uc in zip(supercells, unitcells, strict=True):
        assert len(sc) == 8 * len(uc)
    # Different cells give different displaced structures.
    assert not np.allclose(supercells[0].positions, supercells[1].positions)


def test_build_strain_cells_manifest() -> None:
    """The manifest records the seed, ranges and per-cell free lengths."""
    cell = _hexagonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    unitcells = sample_strained_cells(cell, dof, ranges, num=3, seed=7)
    filenames = [f"unitcell-{i + 1:05d}" for i in range(len(unitcells))]

    manifest = build_strain_cells_manifest(
        phonopy_version="0.0.0",
        calculator="vasp",
        length_unit="angstrom",
        source="phonopy_disp.yaml",
        dof=dof,
        command_line="phonopy-strain-cells phonopy_disp.yaml",
        ranges=ranges,
        num=3,
        rd_distance=None,
        symprec=1e-5,
        seed=7,
        prefix="unitcell",
        kind="strained unit cell",
        unitcells=unitcells,
        filenames=filenames,
    )

    assert manifest["free_dof"] == ["a", "c"]
    assert manifest["parameters"]["seed"] == 7
    assert manifest["parameters"]["ranges"] == {"a": [3.9, 4.1], "c": [5.8, 6.2]}
    assert manifest["parameters"]["rd_distance"] is None
    cells = manifest["output"]["cells"]
    assert manifest["output"]["num_cells"] == 3
    assert len(cells) == 3
    for entry, uc in zip(cells, unitcells, strict=True):
        lengths = np.linalg.norm(uc.cell, axis=1)
        assert entry["a"] == pytest.approx(lengths[0], abs=1e-6)
        assert entry["c"] == pytest.approx(lengths[2], abs=1e-6)


def test_write_strain_cells_manifest_roundtrip(tmp_path) -> None:
    """The manifest is written as YAML that loads back with plain types."""
    yaml = pytest.importorskip("yaml")
    cell = _tetragonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    unitcells = sample_strained_cells(cell, dof, ranges, num=2, seed=1)
    filenames = [f"unitcell-{i + 1:05d}" for i in range(len(unitcells))]
    manifest = build_strain_cells_manifest(
        phonopy_version="0.0.0",
        calculator="vasp",
        length_unit="angstrom",
        source="phonopy_disp.yaml",
        dof=dof,
        command_line="phonopy-strain-cells phonopy_disp.yaml",
        ranges=ranges,
        num=2,
        rd_distance=0.03,
        symprec=1e-5,
        seed=1,
        prefix="supercell",
        kind="random-displacement supercell",
        unitcells=unitcells,
        filenames=filenames,
    )

    path = tmp_path / "strain_cells.yaml"
    write_strain_cells_manifest(path, manifest)
    loaded = yaml.safe_load(path.read_text())
    assert loaded == manifest
