# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the helpers of the phonopy-anisotropic-qha script."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.typing import NDArray
from qha_utils import MESH, TEMPERATURES, internal_energies, scaled_phonopy

from phonopy import Phonopy, run_anisotropic_qha
from phonopy.qha.anisotropic import AnisotropicQHAResult
from phonopy.scripts.phonopy_anisotropic_qha import (
    _read_free_energies,
    compare_thermal_expansion_eos,
    main_diagonal_positions,
    suggest_eos_cells,
)


def _result_with_lattice(
    lattice_lengths: NDArray[np.double], free_lattice_indices: list[int]
) -> AnisotropicQHAResult:
    """Return a result carrying only what the grid geometry helpers read.

    The remaining fields are left empty: the helpers under test reach the rest
    of the result only after the grid has been selected.

    """
    empty = np.zeros(0, dtype="double")
    return AnisotropicQHAResult(
        temperatures=empty,
        lattice_lengths=np.array(lattice_lengths, dtype="double"),
        free_lattice_indices=np.array(free_lattice_indices, dtype="int64"),
        surface_degree=2,
        helmholtz_lattice=empty,
        equilibrium_lattice_parameters=np.zeros((0, 3)),
        equilibrium_volumes=empty,
        gibbs_free_energies=empty,
        thermal_expansion=empty,
        axial_thermal_expansions=np.zeros((0, 3)),
        surface_fit_rms=empty,
        surface_fit_rank=6,
        surface_n_terms=6,
        minimum_extrapolated=np.zeros(0, dtype=bool),
    )


def _grid(a_values: list[float], c_values: list[float]) -> NDArray[np.double]:
    """Return the lattice lengths of a tetragonal (a, a, c) tensor grid."""
    return np.array([[a, a, c] for a in a_values for c in c_values], dtype="double")


def test_main_diagonal_square_grid() -> None:
    """The diagonal of an N x N grid is every (N + 1)-th cell."""
    np.testing.assert_array_equal(main_diagonal_positions([3, 3]), [0, 4, 8])


def test_main_diagonal_rectangular_grid() -> None:
    """A rectangular grid has as many diagonal cells as its shortest axis.

    With three a and four c, the fourth c is left out: there is no fourth a
    to pair it with. The step is 5, one row of four plus one column.

    """
    np.testing.assert_array_equal(main_diagonal_positions([3, 4]), [0, 5, 10])


def test_main_diagonal_three_dof() -> None:
    """Three free axes stride by the sum of the three row-major strides."""
    np.testing.assert_array_equal(main_diagonal_positions([2, 3, 4]), [0, 17])


def test_main_diagonal_one_dof() -> None:
    """With a single free axis every cell lies on the diagonal."""
    np.testing.assert_array_equal(main_diagonal_positions([4]), [0, 1, 2, 3])


def test_compare_eos_skips_a_short_path(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Fewer than five cells on the path is reported and nothing is written.

    run_qha cannot fit an equation of state to four points, so the comparison
    has to bow out rather than fail inside the fit.

    """
    monkeypatch.chdir(tmp_path)
    result = _result_with_lattice(_grid([3.0, 3.1, 3.2], [5.0, 5.1, 5.2]), [0, 2])

    compare_thermal_expansion_eos(
        result, [], np.zeros(0), [], None, MESH, positions=[0, 4, 8]
    )

    assert "Only 3 cells selected" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_suggest_eos_cells_names_a_constant_shape_path(
    capsys: pytest.CaptureFixture,
) -> None:
    """Cells of one shape are named as a ready-made --eos-index argument.

    The grid here samples c in proportion to a, as a grid over equal
    fractional ranges does, so its main diagonal is the set of one c/a. Every
    off-diagonal cell has a c/a of its own.

    """
    a_values = np.linspace(3.0, 3.4, 5)
    c_values = (5.0 / 3.0) * a_values
    lengths = np.array([[a, a, c] for a in a_values for c in c_values])
    result = _result_with_lattice(lengths, [0, 2])

    suggest_eos_cells(result, indices=list(range(len(lengths))))

    out = capsys.readouterr().out
    # The diagonal of a 5 x 5 grid, every sixth cell, numbered from 1.
    assert "--eos-index 1 7 13 19 25" in out

    # Listed by volume, smallest first.
    listed = [
        int(line.split()[0]) - 1
        for line in out.splitlines()
        if line.startswith("  ") and "index" not in line
    ]
    volumes = lengths.prod(axis=1)[listed]
    assert (np.diff(volumes) > 0).all()


def test_suggest_eos_cells_without_a_constant_shape_path(
    capsys: pytest.CaptureFixture,
) -> None:
    """When no five cells share a shape, that is said rather than guessed."""
    rng = np.random.default_rng(0)
    result = _result_with_lattice(rng.uniform(3.0, 3.5, size=(6, 3)), [0, 2])

    suggest_eos_cells(result, indices=list(range(6)))

    out = capsys.readouterr().out
    assert "No five cells share a c/a" in out
    assert "constant-shape volume path" not in out


def test_compare_eos_writes_the_comparison(
    ph_nacl: Phonopy, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparison writes both files, and its table holds what it plots.

    A cubic series is used, whose every cell lies on the diagonal. There the
    volume path is not an approximation, so the two methods have to agree; a
    disagreement would mean the comparison is not comparing what it claims.

    """
    monkeypatch.chdir(tmp_path)
    phonopys = [
        scaled_phonopy(ph_nacl, np.array([s, s, s])) for s in np.linspace(0.98, 1.03, 6)
    ]
    volumes = np.array([ph.primitive.volume for ph in phonopys])
    energies = internal_energies(volumes)
    result = run_anisotropic_qha(
        phonopys,
        TEMPERATURES,
        internal_energies=energies,
        mesh=MESH,
        surface_degree=2,
    )

    compare_thermal_expansion_eos(
        result, phonopys, TEMPERATURES, energies, None, MESH, positions=range(6)
    )

    assert (tmp_path / "thermal_expansion_compare.png").is_file()
    table = np.loadtxt(tmp_path / "thermal_expansion_compare.dat")
    assert table.shape == (len(result.temperatures), 7)

    # The table columns are the result's own arrays, not a re-computation.
    np.testing.assert_allclose(table[:, 0], result.temperatures, rtol=1e-12)
    np.testing.assert_allclose(table[:, 1], result.thermal_expansion, rtol=1e-12)
    np.testing.assert_allclose(
        table[:, 3], result.axial_thermal_expansions[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(
        table[:, 5], result.axial_thermal_expansions[:, 2], rtol=1e-12
    )

    # The settings that produced the numbers travel with them.
    header = (tmp_path / "thermal_expansion_compare.dat").read_text()
    assert "surface_degree=2" in header
    assert "alpha_c_vinet" in header

    # Cubic, so both methods must put the same number in the a and the c
    # column, and three times an axial expansion must be the volumetric one.
    # These are exact relations and catch a mixed-up column outright.
    np.testing.assert_allclose(table[:, 3], table[:, 5], rtol=1e-12)
    np.testing.assert_allclose(table[:, 4], table[:, 6], rtol=1e-12)
    warm = result.temperatures > 300.0
    np.testing.assert_allclose(3.0 * table[warm, 3], table[warm, 1], rtol=1e-2)
    np.testing.assert_allclose(3.0 * table[warm, 4], table[warm, 2], rtol=1e-2)

    # The two methods describe the same expansion of the same cubic cell, so
    # each pair of columns has to stay within a factor of two of the other.
    # The tolerance is deliberately loose: a polynomial surface in a and a
    # Vinet fit in V are different functional forms, and how far they drift
    # apart depends on the fit and on the platform. What this catches is a
    # mixed-up column or a wrong unit, not the quality of either fit.
    for aniso, vinet in ((1, 2), (3, 4), (5, 6)):
        ratio = table[warm, aniso] / table[warm, vinet]
        assert ((ratio > 0.5) & (ratio < 2.0)).all()


def test_free_energies_round_trip(tmp_path: pathlib.Path) -> None:
    """A written file is read back and accepted on a matching grid."""
    from phonopy.qha.free_energy_io import (
        PhononFreeEnergies,
        read_free_energies_hdf5,
        write_free_energies_hdf5,
    )

    temperatures = np.arange(0.0, 101.0, 10.0)
    values = -0.01 * (temperatures[:, None] / 100.0) ** 2 * np.arange(1, 4)[None, :]
    lengths = np.array([[3.0, 3.0, 5.0], [3.1, 3.1, 5.1], [3.2, 3.2, 5.2]])
    path = tmp_path / "fe.hdf5"
    write_free_energies_hdf5(
        PhononFreeEnergies(temperatures, values, lattice_lengths=lengths), path
    )

    read = _read_free_energies(str(path), PhononFreeEnergies, temperatures, lengths)
    np.testing.assert_allclose(read.free_energies, values)

    back = read_free_energies_hdf5(path)
    assert isinstance(back, PhononFreeEnergies)
    assert back.n_grid_points == 3
    np.testing.assert_allclose(back.temperatures, temperatures)
    np.testing.assert_allclose(back.lattice_lengths, lengths)


def test_free_energies_checked_against_the_run(tmp_path: pathlib.Path) -> None:
    """The kind, the temperatures and the grid points are all checked.

    The file is written on another machine, so nothing but these checks ties
    it to the dataset it is used with.

    """
    from phonopy.qha.free_energy_io import (
        ElectronicFreeEnergies,
        PhononFreeEnergies,
        write_free_energies_hdf5,
    )

    temperatures = np.arange(0.0, 101.0, 10.0)
    values = np.zeros((len(temperatures), 3))
    lengths = np.array([[3.0, 3.0, 5.0], [3.1, 3.1, 5.1], [3.2, 3.2, 5.2]])
    path = tmp_path / "fe.hdf5"
    write_free_energies_hdf5(
        PhononFreeEnergies(temperatures, values, lattice_lengths=lengths), path
    )

    with pytest.raises(ValueError, match="hold PhononFreeEnergies"):
        _read_free_energies(str(path), ElectronicFreeEnergies, temperatures, lengths)
    with pytest.raises(ValueError, match="different temperature grid"):
        # Reaches past the file.
        _read_free_energies(
            str(path), PhononFreeEnergies, np.arange(0.0, 151.0, 10.0), lengths
        )
    with pytest.raises(ValueError, match="different temperature grid"):
        # Inside the file's range, but on temperatures it does not hold.
        _read_free_energies(
            str(path), PhononFreeEnergies, np.arange(0.0, 51.0, 5.0), lengths
        )
    with pytest.raises(ValueError, match="grid points against"):
        _read_free_energies(str(path), PhononFreeEnergies, temperatures, lengths[:2])
    with pytest.raises(ValueError, match="different grid points"):
        _read_free_energies(str(path), PhononFreeEnergies, temperatures, lengths + 0.5)


def test_free_energies_over_a_wider_range(tmp_path: pathlib.Path) -> None:
    """A file covering more temperatures than the run is narrowed to the run.

    One expensive sweep to a high temperature is then replotted over any
    subset of its own grid.

    """
    from phonopy.qha.free_energy_io import (
        PhononFreeEnergies,
        write_free_energies_hdf5,
    )

    stored = np.arange(0.0, 1001.0, 10.0)
    values = -0.01 * (stored[:, None] / 100.0) ** 2 * np.arange(1, 4)[None, :]
    lengths = np.array([[3.0, 3.0, 5.0], [3.1, 3.1, 5.1], [3.2, 3.2, 5.2]])
    path = tmp_path / "fe.hdf5"
    write_free_energies_hdf5(
        PhononFreeEnergies(stored, values, lattice_lengths=lengths), path
    )

    wanted = np.arange(0.0, 401.0, 10.0)
    read = _read_free_energies(str(path), PhononFreeEnergies, wanted, lengths)
    assert read.free_energies.shape == (len(wanted), 3)
    np.testing.assert_allclose(read.free_energies, values[: len(wanted)])

    # Every second temperature, which is a subset but not a leading slice.
    coarse = np.arange(0.0, 401.0, 20.0)
    read = _read_free_energies(str(path), PhononFreeEnergies, coarse, lengths)
    assert read.free_energies.shape == (len(coarse), 3)
    np.testing.assert_allclose(read.free_energies, values[: len(wanted) : 2])


def test_phonon_free_energies_round_trip(
    ph_nacl: Phonopy, tmp_path: pathlib.Path
) -> None:
    """Phonon free energies survive a write and read, and pass the checks.

    This is the route of a temperature-dependent method: the free energies are
    computed elsewhere, written to a file, and read back for the analysis.

    """
    from phonopy.physical_units import get_physical_units
    from phonopy.qha.free_energy_io import (
        PhononFreeEnergies,
        read_free_energies_hdf5,
        write_free_energies_hdf5,
    )
    from phonopy.qha.thermal import compute_thermal_properties

    phonopys = [scaled_phonopy(ph_nacl, np.array([s, s, s])) for s in (0.99, 1.0, 1.01)]
    lengths = np.array([np.linalg.norm(ph.unitcell.cell, axis=1) for ph in phonopys])
    fe_phonon, _, _ = compute_thermal_properties(phonopys, TEMPERATURES, MESH)
    fe_phonon_ev = fe_phonon / get_physical_units().EvTokJmol

    path = tmp_path / "fph.hdf5"
    write_free_energies_hdf5(
        PhononFreeEnergies(TEMPERATURES, fe_phonon_ev, lattice_lengths=lengths), path
    )

    back = read_free_energies_hdf5(path)
    assert isinstance(back, PhononFreeEnergies)
    np.testing.assert_allclose(back.free_energies, fe_phonon_ev)
    np.testing.assert_allclose(
        _read_free_energies(
            str(path), PhononFreeEnergies, TEMPERATURES, lengths
        ).free_energies,
        fe_phonon_ev,
    )


def _decomposed(temperatures: NDArray[np.double]) -> dict[str, NDArray[np.double]]:
    """Return a free energy and the three terms it was assembled from."""
    grid = np.arange(1, 4)[None, :]
    potential = 0.02 * (temperatures[:, None] / 100.0) * grid
    reference = -40.0 - 0.1 * grid[0]
    return {
        "free_energies": -0.01 * (temperatures[:, None] / 100.0) ** 2 * grid,
        "reference_energies": reference,
        "potential_energies": potential,
        "harmonic_potential_energies": 0.5 * potential,
    }


def test_free_energy_terms_round_trip(tmp_path: pathlib.Path) -> None:
    """The SSCHA terms survive a write and read, and narrow with the values.

    reference_energies is what lets the reader choose the energy scale: added
    back it puts F on the potential's own, and left out F stays measured from
    the undisplaced cell.

    """
    from phonopy.qha.free_energy_io import (
        PhononFreeEnergies,
        SSCHAFreeEnergies,
        read_free_energies_hdf5,
        write_free_energies_hdf5,
    )

    stored = np.arange(0.0, 1001.0, 10.0)
    terms = _decomposed(stored)
    lengths = np.array([[3.0, 3.0, 5.0], [3.1, 3.1, 5.1], [3.2, 3.2, 5.2]])
    path = tmp_path / "fph.hdf5"
    write_free_energies_hdf5(
        SSCHAFreeEnergies(stored, lattice_lengths=lengths, **terms), path
    )

    back = read_free_energies_hdf5(path)
    assert isinstance(back, SSCHAFreeEnergies)
    for name, values in terms.items():
        np.testing.assert_allclose(getattr(back, name), values)

    # The terms that have a temperature axis are narrowed with the free
    # energies, so they stay aligned; the two that do not are left alone.
    # Narrowing keeps the type, and the SSCHA ones pass as phonon free
    # energies.
    wanted = np.arange(0.0, 401.0, 20.0)
    read = _read_free_energies(str(path), PhononFreeEnergies, wanted, lengths)
    assert isinstance(read, SSCHAFreeEnergies)
    for name in SSCHAFreeEnergies.OVER_TEMPERATURE:
        if name in terms:
            assert getattr(read, name).shape == (len(wanted), 3)
            np.testing.assert_allclose(
                getattr(read, name), terms[name][: len(wanted) * 2 : 2]
            )
    for name in SSCHAFreeEnergies.OVER_GRID:
        np.testing.assert_allclose(getattr(read, name), terms[name])


def test_free_energy_terms_are_checked() -> None:
    """The halves are the input; their difference is derived and checked.

    Passing the difference on its own is the way in for a sweep that recorded
    no more than that, and is transitional.

    """
    from phonopy.qha.free_energy_io import ElectronicFreeEnergies, SSCHAFreeEnergies

    temperatures = np.arange(0.0, 101.0, 10.0)
    terms = _decomposed(temperatures)
    values = terms.pop("free_energies")
    reference = terms["reference_energies"]
    halves = {
        "potential_energies": terms["potential_energies"],
        "harmonic_potential_energies": terms["harmonic_potential_energies"],
    }
    difference = halves["potential_energies"] - halves["harmonic_potential_energies"]

    with pytest.raises(TypeError, match="reference_energies"):
        SSCHAFreeEnergies(temperatures, values, **halves)

    # The halves are the ordinary input, and the difference comes from them.
    made = SSCHAFreeEnergies(
        temperatures,
        values,
        reference_energies=reference,
        **halves,
    )
    np.testing.assert_allclose(made.anharmonic_corrections, difference)

    # Without them, only the transitional input will do.
    with pytest.raises(ValueError, match="Give potential_energies"):
        SSCHAFreeEnergies(
            temperatures,
            values,
            reference_energies=reference,
        )
    SSCHAFreeEnergies(
        temperatures,
        values,
        reference_energies=reference,
        anharmonic_corrections=difference,
    )

    with pytest.raises(ValueError, match="given together"):
        SSCHAFreeEnergies(
            temperatures,
            values,
            reference_energies=reference,
            potential_energies=halves["potential_energies"],
        )
    with pytest.raises(ValueError, match="must equal potential_energies"):
        SSCHAFreeEnergies(
            temperatures,
            values,
            reference_energies=reference,
            anharmonic_corrections=halves["potential_energies"],
            **halves,
        )
    with pytest.raises(ValueError, match="reference_energies is one value per grid"):
        SSCHAFreeEnergies(
            temperatures,
            values,
            reference_energies=reference[:2],
            **halves,
        )

    # The electronic term has no such fields to be given at all.
    with pytest.raises(TypeError, match="reference_energies"):
        ElectronicFreeEnergies(
            temperatures,
            values,
            reference_energies=reference,
            **halves,
        )


def test_internal_energies_from_the_free_energies(tmp_path: pathlib.Path) -> None:
    """--use-mlp-internal-energies needs a file that carries them.

    Falling back to U = 0 would leave the static energy out of the surface
    altogether, so a file that does not record it stops the run.

    """
    from phonopy.qha.free_energy_io import (
        PhononFreeEnergies,
        SSCHAFreeEnergies,
        write_free_energies_hdf5,
    )
    from phonopy.scripts.phonopy_anisotropic_qha import (
        internal_energies_from_the_potential,
    )

    temperatures = np.arange(0.0, 101.0, 10.0)
    terms = _decomposed(temperatures)
    lengths = np.array([[3.0, 3.0, 5.0], [3.1, 3.1, 5.1], [3.2, 3.2, 5.2]])

    with pytest.raises(SystemExit, match="needs --phonon-free-energies"):
        internal_energies_from_the_potential(None)

    bare = tmp_path / "bare.hdf5"
    write_free_energies_hdf5(
        PhononFreeEnergies(
            temperatures, terms["free_energies"], lattice_lengths=lengths
        ),
        bare,
    )
    read = _read_free_energies(str(bare), PhononFreeEnergies, temperatures, lengths)
    assert isinstance(read, PhononFreeEnergies)
    with pytest.raises(SystemExit, match="do not record the energy"):
        internal_energies_from_the_potential(read, str(bare))

    full = tmp_path / "full.hdf5"
    write_free_energies_hdf5(
        SSCHAFreeEnergies(temperatures, lattice_lengths=lengths, **terms),
        full,
    )
    read = _read_free_energies(str(full), PhononFreeEnergies, temperatures, lengths)
    np.testing.assert_allclose(
        internal_energies_from_the_potential(read, str(full)),
        terms["reference_energies"],
    )
