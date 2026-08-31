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
from phonopy.interface.vasp import (
    electronic_states_from_vaspout,
    read_vasprun_calculation,
)
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.scripts.phonopy_anisotropic_qha_dataset import (
    build_calculator_grid_point,
    load_phonon,
    load_phonon_from_disp_dirs,
    primitive_cell_fraction,
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

    # The static single point is run on the unit cell, and the builder checks
    # the paired cells agree. NaCl is face-centred, so its unit cell holds four
    # primitive cells and the builder stores U as a quarter of the energy read
    # here. vasprun-00001..00003 are the 3 displaced supercells.
    sdir = base / "static-grid" / tag
    sdir.mkdir(parents=True)
    _decompress(FIXTURE / "unitcell-static.xml.xz", sdir / "vasprun.xml")

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


def _tensor_grid(a_values, c_values):
    """Return the free lengths of an (a, c) tensor grid in row-major order."""
    return np.array([[a, c] for a in a_values for c in c_values])


def test_detect_grid_shape_of_a_tensor_grid():
    """A tensor grid is recognised, with one count per free DOF."""
    from phonopy.scripts.phonopy_anisotropic_qha_dataset import _detect_grid_shape

    assert _detect_grid_shape(_tensor_grid([3.0, 3.1, 3.2], [5.0, 5.1])) == (3, 2)
    assert _detect_grid_shape(np.array([[3.0], [3.1], [3.2]])) == (3,)


def test_detect_grid_shape_of_scattered_cells():
    """Randomly sampled cells are not a grid and get no shape.

    Every length is then distinct, so the counts multiply to far more than
    the number of cells.

    """
    from phonopy.scripts.phonopy_anisotropic_qha_dataset import _detect_grid_shape

    rng = np.random.default_rng(0)
    assert _detect_grid_shape(rng.uniform(3.0, 3.5, size=(12, 2))) is None


def test_detect_grid_shape_rejects_a_reordered_grid():
    """The cells have to be stored in row-major order.

    The counts alone cannot tell: the same cells in another order still
    multiply to the number of cells, while the main diagonal computed from
    the shape would pick the wrong ones.

    """
    from phonopy.scripts.phonopy_anisotropic_qha_dataset import _detect_grid_shape

    grid = _tensor_grid([3.0, 3.1, 3.2], [5.0, 5.1, 5.2])
    assert _detect_grid_shape(grid) == (3, 3)

    shuffled = grid[np.random.default_rng(1).permutation(len(grid))]
    assert _detect_grid_shape(shuffled) is None
    # Column-major, the plausible mistake, is rejected too.
    assert (
        _detect_grid_shape(grid.reshape(3, 3, 2).transpose(1, 0, 2).reshape(9, 2))
        is None
    )


def test_detect_grid_shape_requires_ascending_axes():
    """Axes have to ascend so that the diagonal is a monotonic volume path."""
    from phonopy.scripts.phonopy_anisotropic_qha_dataset import _detect_grid_shape

    assert _detect_grid_shape(_tensor_grid([3.2, 3.1, 3.0], [5.0, 5.1, 5.2])) is None
    assert _detect_grid_shape(_tensor_grid([3.0, 3.1, 3.2], [5.2, 5.1, 5.0])) is None


def test_build_calculator_grid_point(tmp_path):
    """build_calculator_grid_point gathers forces, displacements, and U correctly."""
    _make_grid_point_dirs(tmp_path, 0)
    pgrid = tmp_path / "phonon-grid"
    sgrid = tmp_path / "static-grid"

    point, fraction = build_calculator_grid_point(
        0,
        str(sgrid / "grid-000" / "vasprun.xml"),
        str(pgrid / "grid-000"),
        with_electronic=False,
    )
    assert fraction == 0.25

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

    # Internal energy matches the static single point, per primitive cell.
    _, energy, _, _ = read_vasprun_calculation(str(sgrid / "grid-000" / "vasprun.xml"))
    assert point.internal_energy == pytest.approx(energy / 4)
    assert point.electronic_states is None


def test_internal_energy_is_stored_per_primitive_cell(tmp_path, capsys):
    """U is scaled to the primitive cell, matching the phonon free energy.

    The calculator reports U for the unit cell it was run on. The analysis
    normalizes the phonon free energy and the volumes per primitive cell, so
    a centred lattice would otherwise mix two normalizations. NaCl is
    face-centred: 8 atoms in the unit cell, 2 in the primitive cell.

    """
    _make_grid_point_dirs(tmp_path, 0)
    sgrid = tmp_path / "static-grid"
    pgrid = tmp_path / "phonon-grid"

    ph = load_phonon_from_disp_dirs(str(pgrid / "grid-000"))
    assert len(ph.unitcell) == 8
    assert len(ph.primitive) == 2

    # Which cell the static single point used is read off the atom counts,
    # so either cell is accepted and neither is assumed.
    assert primitive_cell_fraction(ph.primitive, ph.unitcell) == 0.25
    assert primitive_cell_fraction(ph.primitive, ph.primitive) == 1.0

    point, fraction = build_calculator_grid_point(
        0,
        str(sgrid / "grid-000" / "vasprun.xml"),
        str(pgrid / "grid-000"),
        with_electronic=False,
    )
    _, energy, _, _ = read_vasprun_calculation(str(sgrid / "grid-000" / "vasprun.xml"))
    assert fraction == 0.25
    assert point.internal_energy == pytest.approx(energy * 0.25)


def test_paired_cells_accept_either_cell_of_the_grid_point(tmp_path):
    """The static entry may be the grid point's unit cell or its primitive cell.

    U may be computed on whichever of the two suits the calculation. A cell
    that is neither is still a mis-pairing and stops the builder.

    """
    from phonopy.scripts.phonopy_anisotropic_qha_dataset import _check_paired_cells

    _make_grid_point_dirs(tmp_path, 0)
    ph = load_phonon_from_disp_dirs(str(tmp_path / "phonon-grid" / "grid-000"))

    for cell in (ph.unitcell, ph.primitive):
        _check_paired_cells(0, "static", cell, "phonon", ph.unitcell, ph.primitive)

    other = ph.unitcell.copy()
    other.cell = np.array(other.cell) * 1.05
    with pytest.raises(SystemExit, match="mis-paired"):
        _check_paired_cells(0, "static", other, "phonon", ph.unitcell, ph.primitive)


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
            "--static",
            str(tmp_path / "static-grid" / "grid-000" / "vasprun.xml"),
            str(tmp_path / "static-grid" / "grid-001" / "vasprun.xml"),
            "--phonon",
            str(tmp_path / "phonon-grid" / "grid-000"),
            str(tmp_path / "phonon-grid" / "grid-001"),
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


def _write_minimal_vaspout(
    path,
    lnoncollinear: int | None = None,
    lsorbit: int | None = None,
    ncdij: int | None = None,
) -> None:
    """Write the smallest vaspout.h5 electronic_states_from_vaspout can read.

    A tag of None is omitted, which is what input/incar does when the INCAR did
    not set it. An ncdij of None omits the DOS group altogether, which is how
    the input/incar fallback is reached.

    """
    import h5py

    with h5py.File(path, "w") as w:
        g = w.create_group("results/electron_eigenvalues")
        g.create_dataset("eigenvalues", data=np.zeros((1, 2, 3), dtype="double"))
        g.create_dataset("kpoints_symmetry_weight", data=np.ones(2, dtype="double"))
        g.create_dataset("nelectrons", data=4.0)
        incar = w.create_group("input/incar")
        for tag, value in (("LNONCOLLINEAR", lnoncollinear), ("LSORBIT", lsorbit)):
            if value is not None:
                incar.create_dataset(tag, data=np.int32(value))
        if ncdij is not None:
            w.create_group("results/electron_dos").create_dataset(
                "ncdij", data=np.int32(ncdij)
            )


@pytest.mark.parametrize(
    "ncdij,expected",
    [(1, None), (2, None), (4, 1)],
)
def test_electronic_states_from_vaspout_spin_degeneracy_from_ncdij(
    tmp_path, ncdij, expected
):
    """NCDIJ decides the spin degeneracy without consulting input/incar.

    NCDIJ counts the spin components of the density -- 1 non-spin-polarized, 2
    collinear spin-polarized, 4 non-collinear -- and VASP resolves it from the
    input, so it holds even when no INCAR tag was echoed. The collinear cases
    report None, leaving the unambiguous spin axis to speak for itself.

    """
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, ncdij=ncdij)

    states = electronic_states_from_vaspout(str(path))

    assert states.spin_degeneracy == expected
    assert states.n_electrons == pytest.approx(4.0)


def test_electronic_states_from_vaspout_ncdij_outranks_incar(tmp_path):
    """NCDIJ is believed over an input/incar echo that never saw the tag.

    This is the real spin-orbit case: LSORBIT alone leaves LNONCOLLINEAR out of
    the echo, and NCDIJ reports the non-collinear run regardless.

    """
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, lsorbit=1, ncdij=4)

    assert electronic_states_from_vaspout(str(path)).spin_degeneracy == 1


@pytest.mark.parametrize(
    "lnoncollinear,lsorbit,expected",
    [
        (None, None, None),  # plain collinear
        (0, None, None),
        (None, 0, None),
        (1, None, 1),  # non-collinear asked for explicitly
        (None, 1, 1),  # spin-orbit only: LNONCOLLINEAR never reaches the echo
        (0, 1, 1),  # LSORBIT wins; VASP forces non-collinear regardless
        (1, 1, 1),
    ],
)
def test_electronic_states_from_vaspout_spin_degeneracy_from_incar(
    tmp_path, lnoncollinear, lsorbit, expected
):
    """Without a DOS group the INCAR echo decides, and needs both tags.

    input/incar echoes only the tags the INCAR set, not the values VASP
    resolved, so LSORBIT alone leaves LNONCOLLINEAR absent while the run is
    non-collinear. Missing that case makes the spinor states look like the
    doubly occupied states of a non-spin-polarized run.

    """
    path = tmp_path / "vaspout.h5"
    _write_minimal_vaspout(path, lnoncollinear, lsorbit)

    states = electronic_states_from_vaspout(str(path))

    assert states.spin_degeneracy == expected
    assert states.n_electrons == pytest.approx(4.0)


def test_builder_run_rejects_mispaired_cells(tmp_path, monkeypatch):
    """A static point paired with a different cell is caught, not averaged in.

    --static and --phonon are paired by position, so a point missing from each
    list would silently combine the U of one lattice with the forces of
    another: the list lengths still match and no naming convention says
    otherwise. Comparing the paired cells is what catches it.

    """
    for idx in (0, 1):
        _make_grid_point_dirs(tmp_path, idx)
    # Give grid-001 a visibly different lattice, then pair it with grid-000.
    other = tmp_path / "static-grid" / "grid-001" / "vasprun.xml"
    other.write_text(other.read_text().replace("5.60328748", "5.90328748"))

    reference = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-anisotropic-qha-dataset",
            str(reference),
            "--static",
            str(other),
            "--phonon",
            str(tmp_path / "phonon-grid" / "grid-000"),
        ],
    )
    with pytest.raises(SystemExit, match="mis-paired"):
        run()


def test_builder_run_static_only(tmp_path, monkeypatch):
    """Without --phonon the builder writes cells, U and F_el and no forces.

    That dataset is for a method whose force constants depend on temperature:
    its free energies go to run_anisotropic_qha through phonon_free_energies.
    The analysis command refuses it with an explanation rather than a
    traceback, which is what makes omitting --phonon safe to detect late.

    """
    for idx in (0, 1):
        _make_grid_point_dirs(tmp_path, idx)

    reference = tmp_path / "phonon-grid" / "grid-000" / "phonopy_disp.yaml"
    out = tmp_path / "static_only.hdf5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-anisotropic-qha-dataset",
            str(reference),
            "--static",
            str(tmp_path / "static-grid" / "grid-000" / "vasprun.xml"),
            str(tmp_path / "static-grid" / "grid-001" / "vasprun.xml"),
            "-o",
            str(out),
        ],
    )
    run()

    dataset = read_aniso_qha_dataset(out)
    assert len(dataset.grid_points) == 2
    for point in dataset.grid_points:
        assert point.dataset is None
        assert point.n_displacements == 0
        assert np.isfinite(point.internal_energy)
        with pytest.raises(ValueError, match="carries no displacement dataset"):
            point.to_phonopy()

    # The analysis command explains the situation instead of raising through.
    from phonopy.scripts.phonopy_anisotropic_qha import run as run_analysis

    monkeypatch.setattr(sys, "argv", ["phonopy-anisotropic-qha", str(out)])
    with pytest.raises(SystemExit, match="carries no displacements or forces"):
        run_analysis()


def test_load_phonon_detects_reordered_disp_dirs(tmp_path):
    """Force sets in the wrong order are caught by the structures themselves.

    The disp-* directories are taken in sorted order, which is only a guess at
    the displacement order and is wrong for unpadded names. Each calculator
    output carries the structure it was run on, so the guess is verified rather
    than trusted; without that, a swap keeping the count would silently build
    force constants from mismatched forces.

    """
    _make_grid_point_dirs(tmp_path, 0)
    gdir = tmp_path / "phonon-grid" / "grid-000"

    # Swap the outputs of the first two displacements, keeping the count.
    a = gdir / "disp-001" / "vasprun.xml"
    b = gdir / "disp-002" / "vasprun.xml"
    a_text, b_text = a.read_text(), b.read_text()
    a.write_text(b_text)
    b.write_text(a_text)

    with pytest.raises(ValueError, match="not the displaced supercell"):
        load_phonon_from_disp_dirs(str(gdir))


def test_load_phonon_reports_disp_dir_count(tmp_path):
    """A missing disp-* is reported against the displacement count."""
    _make_grid_point_dirs(tmp_path, 0)
    gdir = tmp_path / "phonon-grid" / "grid-000"
    shutil.rmtree(gdir / "disp-003")

    with pytest.raises(ValueError, match="do not match 3 displacement"):
        load_phonon_from_disp_dirs(str(gdir))
