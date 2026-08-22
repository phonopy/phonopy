# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.qha.lattice_sampling."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

from phonopy.qha.lattice_sampling import (
    build_strain_cells_manifest,
    get_free_lattice_dof,
    grid_strained_cells,
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


def test_dof_tetragonal_with_c_close_to_a() -> None:
    """A tetragonal cell keeps two DOF when c is accidentally close to a.

    The DOF follow the crystal system, so a near-degenerate c/a ~ 1 must not
    look isotropic and collapse the cell onto a single volume-like DOF.

    """
    cell = PhonopyAtoms(
        symbols=["Cu", "Cu"],
        cell=np.diag([3.0, 3.0, 3.002]),
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    dof = get_free_lattice_dof(cell)
    assert dof.crystal_system == "tetragonal"
    assert dof.labels == ("a", "c")
    assert dof.current_lengths["c"] == pytest.approx(3.002)


def test_dof_rotated_conventional_cell_accepted() -> None:
    """A rigidly rotated conventional cell is still accepted.

    A rotation leaves every lattice-vector length unchanged, so each row is
    still its crystal axis and the DOF are unaffected.

    """
    cell = _hexagonal()
    angle = np.deg2rad(53.13)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = PhonopyAtoms(
        symbols=cell.symbols,
        cell=cell.cell @ rotation.T,
        scaled_positions=cell.scaled_positions,
    )
    dof = get_free_lattice_dof(rotated)
    assert dof.labels == ("a", "c")
    assert dof.rows["c"] == (2,)


def test_dof_centred_primitive_cell_rejected() -> None:
    """The primitive cell of a centred lattice is rejected.

    Its rows are centring vectors rather than crystal axes. For this
    body-centred tetragonal cell all three rows have the same length, so
    scaling them could only change the volume and never c/a -- the
    anisotropic calculation would silently degenerate to the volume path.

    """
    a, c = 3.0, 5.0
    cell = PhonopyAtoms(
        symbols=["In"],
        cell=np.array(
            [[-a / 2, a / 2, c / 2], [a / 2, -a / 2, c / 2], [a / 2, a / 2, -c / 2]]
        ),
        scaled_positions=[[0, 0, 0]],
    )
    with pytest.raises(ValueError, match="standardized conventional cell"):
        get_free_lattice_dof(cell)


def test_dof_rhombohedral_setting_rejected() -> None:
    """A rhombohedral cell in the rhombohedral setting is rejected.

    Its three rows are equally long whatever the angle, so the hexagonal
    setting is required to expose the a and c DOF.

    """
    a, alpha = 4.0, np.deg2rad(70.0)
    cos_a = np.cos(alpha)
    row2 = np.array([a * cos_a, a * (cos_a - cos_a**2) / np.sin(alpha), 0.0])
    row2[2] = np.sqrt(a**2 - row2[0] ** 2 - row2[1] ** 2)
    cell = PhonopyAtoms(
        symbols=["As"],
        cell=np.array([[a, 0, 0], [a * cos_a, a * np.sin(alpha), 0], row2]),
        scaled_positions=[[0, 0, 0]],
    )
    with pytest.raises(ValueError, match="standardized conventional cell"):
        get_free_lattice_dof(cell)


def test_grid_hexagonal_tensor_product() -> None:
    """A hexagonal grid is the tensor product of the per-axis linspaces."""
    cell = _hexagonal()  # a = b = 4, c = 6
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.92, 4.08), "c": (5.88, 6.12)}
    cells = grid_strained_cells(cell, dof, ranges, num=5)

    assert len(cells) == 25  # 5 x 5
    a_len = np.array([np.linalg.norm(c.cell, axis=1)[0] for c in cells])
    c_len = np.array([np.linalg.norm(c.cell, axis=1)[2] for c in cells])
    # Each free axis takes exactly the 5 evenly spaced grid values.
    np.testing.assert_allclose(
        np.unique(np.round(a_len, 6)), np.linspace(3.92, 4.08, 5)
    )
    np.testing.assert_allclose(
        np.unique(np.round(c_len, 6)), np.linspace(5.88, 6.12, 5)
    )
    for c in cells:
        lengths = np.linalg.norm(c.cell, axis=1)
        np.testing.assert_allclose(lengths[0], lengths[1], rtol=1e-12)  # b = a
        np.testing.assert_allclose(c.scaled_positions, cell.scaled_positions)


def test_grid_symmetric_ranges_isotropic_diagonal() -> None:
    """Symmetric equal-fraction ranges give a constant-c/a main diagonal."""
    cell = _hexagonal()  # c/a = 1.5
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.92, 4.08), "c": (5.88, 6.12)}  # both +/- 2%
    cells = grid_strained_cells(cell, dof, ranges, num=5)

    ratios = np.array(
        [
            np.linalg.norm(c.cell, axis=1)[2] / np.linalg.norm(c.cell, axis=1)[0]
            for c in cells
        ]
    )
    # Exactly the 5 diagonal cells keep the original shape c/a = 1.5.
    assert np.count_nonzero(np.abs(ratios - 1.5) < 1e-9) == 5


def test_grid_cubic_is_isotropic() -> None:
    """A cubic grid has one DOF; every cell is an isotropic volume point."""
    cell = _cubic()
    dof = get_free_lattice_dof(cell)
    cells = grid_strained_cells(cell, dof, {"a": (3.9, 4.1)}, num=6)

    assert len(cells) == 6
    for c in cells:
        lengths = np.linalg.norm(c.cell, axis=1)
        np.testing.assert_allclose(lengths, lengths[0], rtol=1e-12)


def test_grid_rectangular_per_axis_counts() -> None:
    """A dict of per-axis counts gives a rectangular tensor grid."""
    cell = _hexagonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    cells = grid_strained_cells(cell, dof, ranges, num={"a": 5, "c": 6})

    assert len(cells) == 30  # 5 x 6
    a_len = np.array([np.linalg.norm(c.cell, axis=1)[0] for c in cells])
    c_len = np.array([np.linalg.norm(c.cell, axis=1)[2] for c in cells])
    assert np.unique(np.round(a_len, 6)).size == 5
    assert np.unique(np.round(c_len, 6)).size == 6


def test_grid_invalid() -> None:
    """A grid needs count >= 2 and a range/count for every free DOF."""
    cell = _hexagonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    with pytest.raises(ValueError):
        grid_strained_cells(cell, dof, ranges, num=1)
    with pytest.raises(ValueError):
        grid_strained_cells(cell, dof, ranges, num={"a": 5, "c": 1})  # count < 2
    with pytest.raises(ValueError):
        grid_strained_cells(cell, dof, ranges, num={"a": 5})  # missing c count
    with pytest.raises(ValueError):
        grid_strained_cells(cell, dof, {"a": (3.9, 4.1)}, num=3)  # missing c range
    with pytest.raises(ValueError):
        grid_strained_cells(cell, dof, {"a": (4.1, 3.9), "c": (5.8, 6.2)}, num=3)


def test_build_strain_cells_manifest() -> None:
    """The manifest records the ranges, grid shape and per-cell lengths."""
    cell = _hexagonal()
    dof = get_free_lattice_dof(cell)
    ranges = {"a": (3.9, 4.1), "c": (5.8, 6.2)}
    unitcells = grid_strained_cells(cell, dof, ranges, num=3)
    filenames = [f"unitcell-{i + 1:03d}" for i in range(len(unitcells))]

    manifest = build_strain_cells_manifest(
        phonopy_version="0.0.0",
        calculator="vasp",
        length_unit="angstrom",
        source="phonopy_disp.yaml",
        dof=dof,
        command_line="phonopy-strain-cells phonopy_disp.yaml",
        ranges=ranges,
        grid_shape=[3, 3],
        symprec=1e-5,
        prefix="unitcell",
        kind="strained unit cell",
        unitcells=unitcells,
        filenames=filenames,
    )

    assert manifest["free_dof"] == ["a", "c"]
    assert manifest["parameters"]["ranges"] == {"a": [3.9, 4.1], "c": [5.8, 6.2]}
    assert manifest["parameters"]["grid_shape"] == [3, 3]
    cells = manifest["output"]["cells"]
    assert manifest["output"]["num_cells"] == 9
    assert len(cells) == 9
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
    unitcells = grid_strained_cells(cell, dof, ranges, num=2)
    filenames = [f"unitcell-{i + 1:03d}" for i in range(len(unitcells))]
    manifest = build_strain_cells_manifest(
        phonopy_version="0.0.0",
        calculator="vasp",
        length_unit="angstrom",
        source="phonopy_disp.yaml",
        dof=dof,
        command_line="phonopy-strain-cells phonopy_disp.yaml",
        ranges=ranges,
        grid_shape=[2, 2],
        symprec=1e-5,
        prefix="unitcell",
        kind="strained unit cell",
        unitcells=unitcells,
        filenames=filenames,
    )

    path = tmp_path / "strain_cells.yaml"
    write_strain_cells_manifest(path, manifest)
    loaded = yaml.safe_load(path.read_text())
    assert loaded == manifest


@pytest.mark.parametrize(
    "option", ["--rd", "--amplitude", "--amin", "--amax", "--amax-per-atom"]
)
def test_strain_cells_cli_rejects_displacement_options(option: str) -> None:
    """The displacement options are gone, not silently ignored.

    Generating random-displacement supercells over a strained box existed to
    train one machine-learning potential across the whole box. That strategy
    was dropped in favour of one potential per grid point at a fixed lattice,
    whose training structures are generated by the ordinary phonopy workflow,
    so a command line still carrying these options must fail rather than
    quietly produce plain unit cells.

    """
    from phonopy.scripts.phonopy_strain_cells import get_options

    argv = ["phonopy-strain-cells", "phonopy_disp.yaml", option]
    with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
        get_options()
