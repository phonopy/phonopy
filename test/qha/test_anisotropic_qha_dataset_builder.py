# SPDX-License-Identifier: BSD-3-Clause
"""Functional tests for the anisotropic QHA dataset builder (calculator front-end)."""

from __future__ import annotations

import lzma
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from phonopy.file_IO import write_FORCE_SETS
from phonopy.interface.vasp import read_vasprun_calculation
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.scripts.phonopy_anisotropic_qha_dataset import (
    build_calculator_grid_point,
    discover_grid_dirs,
    load_phonon,
    load_phonon_from_disp_dirs,
    read_electronic_states,
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


def _write_params_yaml(base: Path, idx: int, dst: Path) -> Path:
    """Collect the disp-* forces of one grid point into a phonopy_params.yaml."""
    ph = load_phonon_from_disp_dirs(str(base / "phonon-grid" / f"grid-{idx:03d}"))
    ph.save(dst)
    return dst


def test_read_electronic_states_missing_vaspout(tmp_path, capsys):
    """A static output without vaspout.h5 beside it yields None instead of raising."""
    sdir = tmp_path / "static-grid" / "grid-000"
    sdir.mkdir(parents=True)
    (sdir / "vasprun.xml").write_text("<modeling/>")  # no vaspout.h5 alongside

    # Both the directory and the output file itself are accepted.
    assert read_electronic_states(str(sdir)) is None
    assert read_electronic_states(str(sdir / "vasprun.xml")) is None
    assert "no vaspout.h5" in capsys.readouterr().out


def test_discover_grid_dirs(tmp_path):
    """Grid dirs are the index-sorted grid-NNN directories, paths as found."""
    root = tmp_path / "static-grid"
    for name in ("grid-002", "grid-000", "grid-001"):
        (root / name).mkdir(parents=True)
    (root / "not-a-grid").mkdir()
    (root / "grid-xyz").mkdir()
    (root / "grid-003.txt").write_text("x")
    assert discover_grid_dirs(str(root)) == [
        (0, str(root / "grid-000")),
        (1, str(root / "grid-001")),
        (2, str(root / "grid-002")),
    ]


def test_discover_grid_dirs_unpadded(tmp_path):
    """Directory names need no zero padding; the path is not rebuilt from the index."""
    root = tmp_path / "static-grid"
    for name in ("grid-10", "grid-2"):
        (root / name).mkdir(parents=True)
    assert discover_grid_dirs(str(root)) == [
        (2, str(root / "grid-2")),
        (10, str(root / "grid-10")),
    ]


def test_discover_grid_dirs_empty(tmp_path):
    """An empty grid directory raises rather than returning nothing."""
    (tmp_path / "static-grid").mkdir()
    with pytest.raises(FileNotFoundError):
        discover_grid_dirs(str(tmp_path / "static-grid"))


def test_build_calculator_grid_point(tmp_path):
    """build_calculator_grid_point gathers forces, displacements, and U correctly."""
    _make_grid_point_dirs(tmp_path, 0)
    pgrid = tmp_path / "phonon-grid"
    sgrid = tmp_path / "static-grid"

    point = build_calculator_grid_point(
        0,
        str(sgrid / "grid-000" / "vasprun.xml"),
        str(pgrid / "grid-000"),
        with_electronic=False,
    )

    # Forces match the disp vaspruns, in disp-* order.
    expected_forces = np.array(
        [
            read_vasprun_calculation(
                str(pgrid / "grid-000" / f"disp-{j + 1:03d}" / "vasprun.xml")
            )[2]
            for j in range(3)
        ]
    )
    np.testing.assert_allclose(point.dataset["forces"], expected_forces)
    assert point.dataset["forces"].shape == point.dataset["displacements"].shape
    assert point.n_displacements == 3

    # Internal energy matches the static single point.
    _, energy, _, _ = read_vasprun_calculation(str(sgrid / "grid-000" / "vasprun.xml"))
    assert point.internal_energy == energy
    assert point.electronic_states is None


def test_load_phonon_from_params_yaml(tmp_path):
    """A phonopy_params.yaml carrying forces is an equivalent phonon input."""
    _make_grid_point_dirs(tmp_path, 0)
    params = _write_params_yaml(tmp_path, 0, tmp_path / "phonopy_params.yaml")

    from_dirs = load_phonon_from_disp_dirs(str(tmp_path / "phonon-grid" / "grid-000"))
    from_yaml = load_phonon(str(params))

    np.testing.assert_allclose(
        from_yaml.dataset["forces"], from_dirs.dataset["forces"], atol=1e-8
    )
    np.testing.assert_allclose(
        from_yaml.dataset["displacements"],
        from_dirs.dataset["displacements"],
        atol=1e-8,
    )


def test_load_phonon_from_force_sets(tmp_path):
    """A phonopy_disp.yaml with a FORCE_SETS beside it is an equivalent input."""
    _make_grid_point_dirs(tmp_path, 0)
    from_dirs = load_phonon_from_disp_dirs(str(tmp_path / "phonon-grid" / "grid-000"))

    # phonopy_disp.yaml and FORCE_SETS side by side, away from the cwd.
    point_dir = tmp_path / "elsewhere"
    point_dir.mkdir()
    shutil.copy(
        tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml",
        point_dir / "phonopy_disp.yaml",
    )
    write_FORCE_SETS(from_dirs.dataset, filename=point_dir / "FORCE_SETS")

    from_force_sets = load_phonon(str(point_dir / "phonopy_disp.yaml"))
    np.testing.assert_allclose(
        from_force_sets.dataset["forces"], from_dirs.dataset["forces"], atol=1e-8
    )


def test_load_phonon_without_forces_raises(tmp_path):
    """A displacement-only yaml is rejected with an actionable message."""
    _make_grid_point_dirs(tmp_path, 0)
    disp_yaml = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    with pytest.raises(ValueError, match="carries no forces"):
        load_phonon(str(disp_yaml))


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
            "phonopy-anisotropic-qha-dataset",
            str(reference),
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


def test_builder_run_with_explicit_paths(tmp_path, monkeypatch):
    """--static / --phonon accept any layout and pair the lists by position."""
    for idx in (0, 1):
        _make_grid_point_dirs(tmp_path, idx)

    # An arbitrary layout: names carry no index, and the phonon side is a
    # phonopy_params.yaml rather than a directory of disp-* subdirectories.
    statics = []
    phonons = []
    for idx, name in enumerate(("small", "large")):
        point_dir = tmp_path / "runs" / name
        point_dir.mkdir(parents=True)
        shutil.copy(
            tmp_path / "static-grid" / f"grid-{idx:03d}" / "vasprun.xml",
            point_dir / "vasprun.xml",
        )
        statics.append(str(point_dir))  # a directory, resolved to its VASP output
        phonons.append(str(_write_params_yaml(tmp_path, idx, point_dir / "ph.yaml")))

    reference = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    out = tmp_path / "aniso_qha_dataset.hdf5"
    monkeypatch.setattr(
        sys,
        "argv",
        ["phonopy-anisotropic-qha-dataset", str(reference)]
        + ["--static"]
        + statics
        + ["--phonon"]
        + phonons
        + ["-o", str(out)],
    )
    run()

    dataset = read_aniso_qha_dataset(out)
    assert len(dataset.grid_points) == 2
    assert [p.index for p in dataset.grid_points] == [0, 1]
    assert all(p.n_displacements == 3 for p in dataset.grid_points)


def test_builder_run_rejects_mixed_static_options(tmp_path, monkeypatch):
    """--static and --static-grid are mutually exclusive."""
    _make_grid_point_dirs(tmp_path, 0)
    reference = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-anisotropic-qha-dataset",
            str(reference),
            "--static",
            str(tmp_path / "static-grid" / "grid-000" / "vasprun.xml"),
            "--static-grid",
            str(tmp_path / "static-grid"),
        ],
    )
    with pytest.raises(SystemExit, match="not both"):
        run()


def test_builder_run_rejects_length_mismatch(tmp_path, monkeypatch):
    """A --phonon list shorter than the static list is an error, not a silent zip."""
    for idx in (0, 1):
        _make_grid_point_dirs(tmp_path, idx)
    reference = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-anisotropic-qha-dataset",
            str(reference),
            "--static",
            str(tmp_path / "static-grid" / "grid-000" / "vasprun.xml"),
            str(tmp_path / "static-grid" / "grid-001" / "vasprun.xml"),
            "--phonon",
            str(tmp_path / "phonon-grid" / "grid-000"),
        ],
    )
    with pytest.raises(SystemExit, match="do not match"):
        run()
