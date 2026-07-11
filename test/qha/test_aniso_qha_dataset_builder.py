"""Functional tests for the anisotropic QHA dataset builder (DFT front-end)."""

from __future__ import annotations

import lzma
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from phonopy.interface.vasp import read_vasprun_calculation
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.scripts.phonopy_aniso_qha_dataset import (
    build_dft_grid_point,
    discover_grid_indices,
    run,
)

FIXTURE = Path(__file__).parent.parent / "cui" / "phonopy_command" / "vaspruns_NaCl_rd"
VASPRUNS = [f"vasprun-0000{i}.xml.xz" for i in range(4)]


def _decompress(src: Path, dst: Path) -> None:
    """Decompress an .xz fixture to a plain file."""
    with lzma.open(src) as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _make_grid_point_dirs(base: Path, idx: int) -> None:
    """Create static-grid/grid-NNN and phonon-grid/grid-NNN for one point.

    The same NaCl random-displacement fixtures are reused for every grid
    point; only the pipeline plumbing is under test here.

    """
    tag = f"grid-{idx:03d}"

    # vasprun-00000 is the perfect (undisplaced) supercell; reuse it as the
    # static single point. vasprun-00001..00003 are the 3 displaced supercells.
    sdir = base / "static-grid" / tag
    sdir.mkdir(parents=True)
    _decompress(FIXTURE / VASPRUNS[0], sdir / "vasprun.xml")

    pdir = base / "phonon-grid" / tag
    pdir.mkdir(parents=True)
    _decompress(FIXTURE / "phonopy_disp.yaml.xz", pdir / "phonopy_disp.yaml")
    for j, name in enumerate(VASPRUNS[1:]):
        ddir = pdir / f"disp-{j + 1:03d}"
        ddir.mkdir()
        _decompress(FIXTURE / name, ddir / "vasprun.xml")


def test_discover_grid_indices(tmp_path):
    """Grid indices are the sorted integers of the grid-NNN directories."""
    root = tmp_path / "static-grid"
    for name in ("grid-002", "grid-000", "grid-001"):
        (root / name).mkdir(parents=True)
    (root / "not-a-grid").mkdir()
    (root / "grid-xyz").mkdir()
    (root / "grid-003.txt").write_text("x")
    assert discover_grid_indices(str(root)) == [0, 1, 2]


def test_discover_grid_indices_empty(tmp_path):
    """An empty grid directory raises rather than returning nothing."""
    (tmp_path / "static-grid").mkdir()
    with pytest.raises(FileNotFoundError):
        discover_grid_indices(str(tmp_path / "static-grid"))


def test_build_dft_grid_point(tmp_path):
    """build_dft_grid_point gathers forces, displacements, and U correctly."""
    _make_grid_point_dirs(tmp_path, 0)
    pgrid = tmp_path / "phonon-grid"
    sgrid = tmp_path / "static-grid"

    point = build_dft_grid_point(0, str(sgrid), str(pgrid), with_electronic=False)

    # Forces match the disp vaspruns, in disp-* order.
    expected_forces = np.array(
        [
            read_vasprun_calculation(
                str(pgrid / "grid-000" / f"disp-{j + 1:03d}" / "vasprun.xml")
            )[2]
            for j in range(3)
        ]
    )
    np.testing.assert_allclose(point.forces, expected_forces)
    assert point.forces.shape == point.displacements.shape
    assert point.forces.shape[0] == 3

    # Internal energy matches the static single point.
    _, energy, _, _ = read_vasprun_calculation(str(sgrid / "grid-000" / "vasprun.xml"))
    assert point.internal_energy == energy
    assert point.electronic_states is None


def test_builder_run_and_analysis(tmp_path, monkeypatch):
    """run() writes a dataset that rebuilds into working phonons."""
    pytest.importorskip("symfc")
    for idx in (0, 1):
        _make_grid_point_dirs(tmp_path, idx)

    reference = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    out = tmp_path / "aniso_qha_dataset.hdf5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-aniso-qha-dataset",
            str(reference),
            "--from-dft",
            "--static-grid",
            str(tmp_path / "static-grid"),
            "--phonon-grid",
            str(tmp_path / "phonon-grid"),
            "-o",
            str(out),
        ],
    )
    run()

    dataset = read_aniso_qha_dataset(out)
    assert len(dataset.grid_points) == 2
    assert dataset.free_dof == ("a",)  # NaCl is cubic

    # dataset -> Phonopy -> phonons works (symfc on real forces).
    ph = dataset.grid_points[0].to_phonopy()
    ph.run_mesh([5, 5, 5])
    tp = ph.run_thermal_properties(temperatures=[300.0])
    assert np.isfinite(tp.free_energy[0])
