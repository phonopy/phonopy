# SPDX-License-Identifier: BSD-3-Clause
"""Tests for what one SSCHA run sampled."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from phonopy.sscha.run import SSCHARun, read_sscha_run_hdf5, write_sscha_run_hdf5


def _run(iterations: int = 6) -> SSCHARun:
    """Return a run with distinguishable values in every field."""
    rng = np.random.default_rng(0)
    return SSCHARun(
        temperature=250.0,
        free_energies=0.1 + rng.normal(0.0, 1e-4, iterations),
        errors=rng.uniform(1e-6, 1e-5, iterations),
        potential_energies=0.2 + rng.normal(0.0, 1e-4, iterations),
        harmonic_potential_energies=0.15 + rng.normal(0.0, 1e-4, iterations),
        reference_energy=-23.4,
        lattice_lengths=np.array([4.56, 4.56, 2.818]),
        force_constants=rng.normal(0.0, 1.0, (2, 8, 3, 3)),
        force_constants_history=rng.normal(0.0, 1.0, (iterations, 2, 8, 3, 3)),
        p2s_map=np.array([0, 4]),
    )


def test_sscha_run_round_trip(tmp_path: pathlib.Path) -> None:
    """A run survives a write and a read, with its scalars still scalars."""
    pytest.importorskip("h5py")
    run = _run()
    path = tmp_path / "sscha.hdf5"
    write_sscha_run_hdf5(run, path)

    back = read_sscha_run_hdf5(path)
    assert isinstance(back, SSCHARun)
    assert back.temperature == 250.0
    assert isinstance(back.temperature, float)
    assert back.reference_energy == pytest.approx(-23.4)
    assert back.n_iterations == 6
    for name in SSCHARun.PER_ITERATION:
        np.testing.assert_allclose(getattr(back, name), getattr(run, name))
    np.testing.assert_allclose(back.lattice_lengths, run.lattice_lengths)
    np.testing.assert_allclose(back.force_constants, run.force_constants)
    np.testing.assert_allclose(
        back.force_constants_history, run.force_constants_history
    )
    np.testing.assert_array_equal(back.p2s_map, run.p2s_map)
    assert back.p2s_map.dtype == np.int64


def test_sscha_run_without_force_constants(tmp_path: pathlib.Path) -> None:
    """A file written before the fields existed still reads, as None."""
    pytest.importorskip("h5py")
    fields = {
        name: getattr(_run(), name)
        for name in ("temperature", "reference_energy", *SSCHARun.PER_ITERATION)
    }
    path = tmp_path / "sscha.hdf5"
    write_sscha_run_hdf5(SSCHARun(**fields), path)

    back = read_sscha_run_hdf5(path)
    assert back.force_constants is None
    assert back.force_constants_history is None
    assert back.p2s_map is None


def test_sscha_run_is_not_a_free_energy(tmp_path: pathlib.Path) -> None:
    """Neither file is read as the other.

    The analysis takes free energies. Reading a run as one would average it
    over iterations nobody chose, so each reader refuses the other's file.

    """
    pytest.importorskip("h5py")
    from phonopy.qha.free_energy_io import (
        PhononFreeEnergies,
        read_free_energies_hdf5,
        write_free_energies_hdf5,
    )

    sampled = tmp_path / "sscha.hdf5"
    write_sscha_run_hdf5(_run(), sampled)
    with pytest.raises(ValueError, match="free energy type 'SSCHARun'"):
        read_free_energies_hdf5(sampled)

    averaged = tmp_path / "fph.hdf5"
    write_free_energies_hdf5(
        PhononFreeEnergies(np.arange(0.0, 31.0, 10.0), np.zeros((4, 2))), averaged
    )
    with pytest.raises(ValueError, match="not SSCHARun"):
        read_sscha_run_hdf5(averaged)


def test_sscha_run_shapes_are_checked() -> None:
    """Every term needs the iteration axis, and the cell has three lengths."""
    run = _run()
    fields = {
        name: getattr(run, name)
        for name in ("temperature", "reference_energy", *SSCHARun.PER_ITERATION)
    }
    SSCHARun(**fields)

    with pytest.raises(ValueError, match="free_energies is one value per"):
        SSCHARun(**{**fields, "free_energies": run.free_energies[:, None]})
    with pytest.raises(ValueError, match="errors must have the shape"):
        SSCHARun(**{**fields, "errors": run.errors[:3]})
    with pytest.raises(ValueError, match="lattice_lengths must have shape"):
        SSCHARun(lattice_lengths=np.zeros(2), **fields)


def test_averaged_force_constants_takes_off_the_transient() -> None:
    """The history is averaged over the iterations the transient leaves."""
    rng = np.random.default_rng(1)
    history = rng.normal(0.0, 1.0, (6, 2, 8, 3, 3))
    run = dataclasses.replace(_run(), force_constants_history=history)

    np.testing.assert_allclose(
        run.averaged_force_constants(2), history[2:].mean(axis=0)
    )
    np.testing.assert_allclose(run.averaged_force_constants(0), history.mean(axis=0))
    with pytest.raises(ValueError, match="transient is 6"):
        run.averaged_force_constants(6)


def test_averaged_force_constants_needs_the_history() -> None:
    """The refit is a different quantity, so it does not stand in for it."""
    run = dataclasses.replace(_run(), force_constants_history=None)
    assert run.force_constants is not None
    with pytest.raises(ValueError, match="needs force_constants_history"):
        run.averaged_force_constants()
