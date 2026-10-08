# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.interface.calculator."""

import pathlib

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.interface.calculator import (
    get_default_displacement_distance,
    write_supercells_with_displacements,
)
from phonopy.interface.octopus import parse_set_of_forces
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture


def test_get_default_displacement_distance_octopus():
    """Octopus works in atomic units, so the default distance is 0.02 Bohr."""
    assert get_default_displacement_distance("octopus") == pytest.approx(0.02)


def test_write_supercells_without_structure_info(
    ph_nacl: Phonopy, tmp_path: pathlib.Path
):
    """optional_structure_info may be omitted (e.g. for the VASP default).

    API users building PhonopyAtoms programmatically have no
    StructureInfo from read_crystal_structure; interfaces that do not
    need it must work without it.

    """
    cells = ph_nacl.supercells_with_displacements
    assert cells is not None
    pre = str(tmp_path / "POSCAR")
    write_supercells_with_displacements(
        "vasp",
        ph_nacl.supercell,
        cells,
        additional_info={"pre_filename": pre},
    )
    assert (tmp_path / "SPOSCAR").exists()
    assert (tmp_path / "POSCAR-001").exists()


def _write_vasp_supercells_of_split_cell(
    weights: list[float], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write VASP 2x2x2 supercells of diamond Ge and Sn co-located at sites."""
    monkeypatch.chdir(tmp_path)
    a = 2.8944775347946359
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0, 0, 0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    ph = Phonopy(
        apply_site_mixture(cell, weights),
        supercell_matrix=[2, 2, 2],
        site_mixture_scheme="split",
    )
    ph.generate_displacements()
    cells = ph.supercells_with_displacements
    assert cells is not None
    write_supercells_with_displacements("vasp", ph.supercell, cells)


def test_write_supercells_vasp_vca_hint_of_split_cell(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """VASP VCA hint is printed for a cell with weighted species."""
    _write_vasp_supercells_of_split_cell([0.9, 0.9, 0.1, 0.1], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "POSCAR species rows: Ge Sn" in out
    assert "VCA = 0.9 0.1" in out
    assert (tmp_path / "SPOSCAR").exists()


def test_write_supercells_vasp_vca_hint_of_split_cell_in_four_rows(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Atoms of one element with two weights are in two species rows."""
    _write_vasp_supercells_of_split_cell([0.5, 0.25, 0.5, 0.75], tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "POSCAR species rows: Ge Ge Sn Sn" in out
    assert "POSCAR counts:       8 8 8 8" in out
    assert "VCA = 0.5 0.25 0.5 0.75" in out


def test_write_supercells_vasp_magmom_keeps_order_of_atoms(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """MAGMOM follows the order of atoms, as POSCAR does."""
    monkeypatch.chdir(tmp_path)
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0.5]],
        symbols=["Fe", "O", "Fe", "O"],
        magnetic_moments=[1.0, 0.0, -1.0, 0.0],
    )
    ph = Phonopy(cell, supercell_matrix=[1, 1, 1])
    ph.generate_displacements()
    cells = ph.supercells_with_displacements
    assert cells is not None
    write_supercells_with_displacements("vasp", ph.supercell, cells)
    magmom = (tmp_path / "MAGMOM").read_text().split("=")[1].split()
    assert [float(v) for v in magmom] == [1.0, 0.0, -1.0, 0.0]
    assert (tmp_path / "SPOSCAR").read_text().split("\n")[5].split() == [
        "Fe",
        "O",
        "Fe",
        "O",
    ]


def test_write_supercells_wien2k_requires_structure_info(ph_nacl: Phonopy):
    """Interfaces that need structure information raise a clear error."""
    cells = ph_nacl.supercells_with_displacements
    assert cells is not None
    with pytest.raises(ValueError, match="wien2k requires structure information"):
        write_supercells_with_displacements("wien2k", ph_nacl.supercell, cells)


def test_write_supercells_octopus_creates_geometry_files(
    ph_nacl: Phonopy, tmp_path: pathlib.Path
):
    """Octopus writer should create geometry files."""
    cells = ph_nacl.supercells_with_displacements
    assert cells is not None
    pre = str(tmp_path / "geometry")
    write_supercells_with_displacements(
        "octopus",
        ph_nacl.supercell,
        cells,
        additional_info={"pre_filename": pre},
    )
    assert (tmp_path / "geometry-000").exists()
    assert (tmp_path / "geometry-001").exists()


def test_parse_set_of_forces_octopus_records_are_converted_and_drift_removed(
    tmp_path: pathlib.Path,
):
    """Octopus force parsing should convert units and remove drift."""
    info = tmp_path / "static_info.txt"
    info.write_text(
        "Forces on the ions [eV/A]\n"
        " Ion                        x              y              z\n"
        "   1        Na   0.1   0.2   0.3\n"
        "   2        Cl   0.4   0.5   0.6\n"
        " ----------------------------------------------------------\n"
    )

    result = parse_set_of_forces(2, [str(info)], verbose=False)
    assert result is not None
    assert len(result) == 1
    forces = result[0]
    assert forces.shape == (2, 3)
    np.testing.assert_allclose(forces.sum(axis=0), [0.0, 0.0, 0.0], atol=1e-10)
