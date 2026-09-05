# SPDX-License-Identifier: BSD-3-Clause
"""Tests for what one SSCHA run sampled."""

from __future__ import annotations

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
