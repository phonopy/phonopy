# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.cui.create_force_sets."""

from __future__ import annotations

import io
import pathlib
from typing import cast

import numpy as np
import pytest

import phonopy
import phonopy.cui.create_force_sets as create_force_sets
from phonopy import Phonopy
from phonopy.cui.create_force_sets import create_FORCE_SETS
from phonopy.harmonic.displacement import Type2DisplacementDataset
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture


def _get_merged_phonon() -> Phonopy:
    """Return Phonopy of GeSn 50/50 zincblende with the merge scheme."""
    a = 2.894478
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]] * 2,
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    return Phonopy(
        apply_site_mixture(cell, [0.5, 0.5, 0.5, 0.5]),
        supercell_matrix=[2, 2, 2],
        primitive_matrix="P",
    )


def test_create_FORCE_SETS_type2_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Type-2 FORCE_SETS of the merge scheme has displacements of the atoms.

    The displacements of the sites are given to their atoms in FORCE_SETS, and
    phonopy.load returns them as the displacements of the sites with the
    forces on the atoms.

    """
    phonon = _get_merged_phonon()
    phonon.generate_displacements(number_of_snapshots=3, random_seed=11)
    cells = phonon.supercells_with_displacements
    assert cells is not None
    assert phonon.unmerged_supercell is not None
    n_atoms = len(phonon.unmerged_supercell)
    rng = np.random.default_rng(7)
    forces = rng.standard_normal((len(cells), n_atoms, 3))

    def _get_calc_dataset(interface_mode, num_atoms, force_filenames, verbose):
        assert num_atoms == n_atoms
        return {
            "forces": list(forces),
            "points": [cell.scaled_positions for cell in cells],
        }

    monkeypatch.setattr(create_force_sets, "get_calc_dataset", _get_calc_dataset)
    yaml_text = str(phonon.to_phonopy_yaml())
    phpy_yaml = PhonopyYaml().read(io.StringIO(yaml_text))
    force_sets = tmp_path / "FORCE_SETS"
    create_FORCE_SETS(
        "vasp",
        [f"vasprun.xml-{i:03d}" for i in range(len(cells))],
        phpy_yaml=phpy_yaml,
        force_sets_filename=force_sets,
    )

    data = np.loadtxt(force_sets)
    assert data.shape == (len(cells) * n_atoms, 6)
    disps = data[:, :3].reshape(len(cells), n_atoms, 3)
    for disp, cell in zip(disps, cells, strict=True):
        np.testing.assert_allclose(
            disp, cell.positions - phonon.unmerged_supercell.positions, atol=1e-7
        )

    ph_load = phonopy.load(
        io.StringIO(yaml_text), force_sets_filename=force_sets, produce_fc=False
    )
    dataset = cast(Type2DisplacementDataset, ph_load.dataset)
    assert dataset is not None
    np.testing.assert_allclose(
        dataset["displacements"], phonon.displacements, atol=1e-7
    )
    np.testing.assert_allclose(dataset["forces"], forces, atol=1e-7)
