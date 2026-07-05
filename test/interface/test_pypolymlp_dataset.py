"""Tests for VASP-to-pypolymlp structure dataset assembly."""

from __future__ import annotations

import lzma
import sys
from pathlib import Path

import numpy as np
import pytest

from phonopy.interface.pypolymlp import (
    PypolymlpStructureData,
    _structures_virial,
    develop_pypolymlp_from_structures,
    read_pypolymlp_structure_dataset,
    read_vasprun_dataset,
    write_pypolymlp_structure_dataset,
)
from phonopy.interface.vasp import read_vasprun_calculation

vaspruns_dir = (
    Path(__file__).parent.parent / "cui" / "phonopy_command" / "vaspruns_NaCl_rd"
)
vasprun_files = sorted(vaspruns_dir.glob("vasprun-*.xml.xz"))


def test_read_vasprun_calculation() -> None:
    """Extraction returns the final cell, energy, forces and stress (GPa)."""
    cell, energy, forces, stress = read_vasprun_calculation(vasprun_files[0])

    assert len(cell) == 64
    assert energy == pytest.approx(-223.94835772)
    assert forces.shape == (64, 3)
    assert stress is not None
    assert stress.shape == (3, 3)
    # Stress is reported in GPa (vasprun kBar / 10).
    assert stress[0, 0] == pytest.approx(-3.937097e-02 / 10.0)


def test_read_vasprun_dataset() -> None:
    """A dataset collects all files with aligned energies, forces, stresses."""
    data = read_vasprun_dataset(vasprun_files)

    n = len(vasprun_files)
    assert isinstance(data, PypolymlpStructureData)
    assert len(data.structures) == n
    assert data.energies.shape == (n,)
    assert len(data.forces) == n
    assert all(f.shape == (64, 3) for f in data.forces)
    assert data.stresses is not None
    assert data.stresses.shape == (n, 3, 3)


def test_virial_matches_pypolymlp(tmp_path) -> None:
    """Our GPa-to-virial conversion matches pypolymlp's own vasprun parse."""
    pytest.importorskip("pypolymlp")
    from pypolymlp.core import interface_vasp as iv
    from pypolymlp.core.units import EVtoKbar

    src = vasprun_files[0]
    dst = tmp_path / "vasprun.xml"
    dst.write_bytes(lzma.open(src).read())

    data = read_vasprun_dataset([src])
    virial = _structures_virial(data)
    assert virial is not None

    vasprun = iv.Vasprun(str(dst))
    reference = np.array(vasprun.stress) * vasprun.structure.volume / EVtoKbar

    np.testing.assert_allclose(virial[0], reference, rtol=1e-5, atol=1e-8)
    assert data.energies[0] == pytest.approx(vasprun.energy)


def test_hdf5_round_trip(tmp_path) -> None:
    """Writing and reading an HDF5 dataset reproduces all quantities."""
    pytest.importorskip("h5py")
    data = read_vasprun_dataset(vasprun_files)

    filename = tmp_path / "polymlp_dataset.hdf5"
    write_pypolymlp_structure_dataset(data, filename=filename)
    loaded = read_pypolymlp_structure_dataset(filename)

    assert len(loaded.structures) == len(data.structures)
    np.testing.assert_allclose(loaded.energies, data.energies)
    assert loaded.stresses is not None
    np.testing.assert_allclose(loaded.stresses, data.stresses)
    for orig, new in zip(data.structures, loaded.structures, strict=True):
        np.testing.assert_allclose(new.cell, orig.cell)
        np.testing.assert_allclose(new.scaled_positions, orig.scaled_positions)
        np.testing.assert_array_equal(new.numbers, orig.numbers)
    for orig_f, new_f in zip(data.forces, loaded.forces, strict=True):
        np.testing.assert_allclose(new_f, orig_f)


def _split(data: PypolymlpStructureData, k: int):
    """Split a dataset into first-k train and rest test."""

    def take(sl):
        return PypolymlpStructureData(
            structures=data.structures[sl],
            energies=data.energies[sl],
            forces=data.forces[sl],
            stresses=None if data.stresses is None else data.stresses[sl],
        )

    return take(slice(0, k)), take(slice(k, None))


def test_develop_marshals_energy_force_stress(monkeypatch) -> None:
    """develop_pypolymlp_from_structures passes E/F/stress in pypolymlp form."""
    pytest.importorskip("pypolymlp")
    import pypolymlp.mlp_dev.pypolymlp as ppm

    captured: dict = {}

    class FakePolymlp:
        def set_params(self, **kwargs):
            captured["params"] = kwargs

        def set_datasets_structures(self, **kwargs):
            captured["datasets"] = kwargs

        def run(self, verbose=False):
            captured["ran"] = True

    monkeypatch.setattr(ppm, "Pypolymlp", FakePolymlp)

    data = read_vasprun_dataset(vasprun_files)
    train, test = _split(data, 3)
    develop_pypolymlp_from_structures(train, test)

    ds = captured["datasets"]
    assert captured["ran"] is True
    # Forces are transposed to pypolymlp's (3, natoms) convention.
    assert ds["train_forces"][0].shape == (3, 64)
    np.testing.assert_allclose(ds["train_forces"][0], train.forces[0].T)
    # Energies passed through.
    np.testing.assert_allclose(ds["train_energies"], train.energies)
    # Stresses become (n, 3, 3) virials in eV.
    assert ds["train_stresses"].shape == (3, 3, 3)
    np.testing.assert_allclose(ds["train_stresses"], _structures_virial(train))


def test_cli_vasp_mlp_dataset(tmp_path, monkeypatch, capsys) -> None:
    """The phonopy-vasp-mlp-dataset command writes a readable HDF5 dataset."""
    pytest.importorskip("h5py")
    from phonopy.scripts.phonopy_vasp_mlp_dataset import run

    out = tmp_path / "dataset.hdf5"
    monkeypatch.setattr(
        sys,
        "argv",
        ["phonopy-vasp-mlp-dataset", *map(str, vasprun_files), "-o", str(out)],
    )

    run()

    assert out.exists()
    loaded = read_pypolymlp_structure_dataset(out)
    assert len(loaded.structures) == len(vasprun_files)
    assert "stress: yes" in capsys.readouterr().out
