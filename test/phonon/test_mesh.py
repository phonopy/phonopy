# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonon calculation on sampling mesh."""

from __future__ import annotations

import itertools
import os
import pathlib
import tempfile
import warnings
from typing import Literal

import h5py
import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.mesh import (
    IterMesh,
    Mesh,
    MeshGRGridFallbackWarning,
    MeshSymmetryFallbackWarning,
)
from phonopy.structure.atoms import PhonopyAtoms

freqs_full_fcsym_ref = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.509780,
    2.509780,
    4.087812,
    4.143926,
    4.143926,
    6.697105,
    2.106719,
    2.106719,
    4.571979,
    4.798841,
    4.798841,
    5.116023,
    3.115897,
    3.487080,
    4.228067,
    4.459883,
    5.037473,
    5.066083,
]

freqs_nofcsym = [
    -0.037009,
    -0.037009,
    -0.037009,
    4.608453,
    4.608453,
    4.608453,
    2.509883,
    2.509883,
    4.088001,
    4.143404,
    4.143404,
    6.696706,
    2.107450,
    2.107450,
    4.572128,
    4.799497,
    4.799497,
    5.116806,
    3.115923,
    3.486963,
    4.227909,
    4.460058,
    5.037201,
    5.066048,
]

freqs_compact_fcsym_ref = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.509780,
    2.509780,
    4.087812,
    4.143926,
    4.143926,
    6.697105,
    2.106719,
    2.106719,
    4.571979,
    4.798841,
    4.798841,
    5.116023,
    3.115897,
    3.487080,
    4.228067,
    4.459883,
    5.037473,
    5.066083,
]

freqs_nonac_ref = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.458402,
    2.458402,
    4.146058,
    4.395223,
    4.395223,
    6.236554,
    2.106889,
    2.106889,
    4.572116,
    4.798842,
    4.798842,
    5.508652,
    2.737708,
    3.491448,
    4.231340,
    4.583972,
    5.066096,
    5.087983,
]

freqs_full_fcsym_ref_wang = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.458402,
    2.458402,
    4.142037,
    4.395223,
    4.395223,
    6.373675,
    2.106889,
    2.106889,
    4.571823,
    4.798842,
    4.798842,
    5.116018,
    2.737708,
    3.491448,
    4.231340,
    4.613570,
    5.066096,
    5.112448,
]

freqs_full_fcsym_ref_si = [
    0.000000,
    0.000000,
    0.000000,
    15.111196,
    15.111196,
    15.111196,
    2.823122,
    2.823122,
    8.281999,
    13.784543,
    14.544795,
    14.544795,
    3.662488,
    3.662488,
    8.879957,
    13.943485,
    13.943485,
    14.047251,
    3.868017,
    6.872823,
    10.247355,
    11.783252,
    13.879338,
    14.048571,
]


def test_Mesh_nofcsym(ph_nacl_nofcsym: Phonopy):
    """Test by NaCl without symmetrizing force constants."""
    _test_IterMesh(ph_nacl_nofcsym, freqs_nofcsym)


def test_Mesh_full_fcsym(ph_nacl: Phonopy):
    """Test by NaCl with symmetrizing force constants."""
    _test_IterMesh(ph_nacl, freqs_full_fcsym_ref)


def test_Mesh_compact_fcsym(ph_nacl_compact_fcsym: Phonopy):
    """Test by NaCl with symmetrizing force constants in compact format."""
    _test_IterMesh(ph_nacl_compact_fcsym, freqs_compact_fcsym_ref)


def test_Mesh_full_fcsym_nonac(ph_nacl_nonac: Phonopy):
    """Test by NaCl without NAC."""
    _test_IterMesh(ph_nacl_nonac, freqs_nonac_ref)


def test_Mesh_full_fcsym_wang(ph_nacl_wang: Phonopy):
    """Test by NaCl with symmetrizing force constants."""
    _test_IterMesh(ph_nacl_wang, freqs_full_fcsym_ref_wang)


def test_Mesh_full_fcsym_si(ph_si: Phonopy):
    """Test by NaCl with symmetrizing force constants."""
    _test_IterMesh(ph_si, freqs_full_fcsym_ref_si)


@pytest.mark.parametrize(
    "compression,with_eigenvectors,with_group_velocities",
    itertools.product(
        ["gzip", "lzf", 1, 2, None],
        [False, True],
        [False, True],
    ),
)
def test_mesh_write_hdf5(
    ph_nacl: Phonopy,
    compression: Literal["gzip", "lzf"] | int | None,
    with_eigenvectors: bool,
    with_group_velocities: bool,
):
    """Test hdf5 output of mesh calculation by NaCl."""
    ph_nacl.run_mesh(
        [3, 3, 3],
        with_group_velocities=with_group_velocities,
        with_eigenvectors=with_eigenvectors,
    )
    assert ph_nacl.mesh is not None
    assert isinstance(ph_nacl.mesh, Mesh)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        ph_nacl.mesh.write_hdf5(compression=compression)

        for created_filename in ["mesh.hdf5"]:
            file_path = pathlib.Path(created_filename)
            assert file_path.exists()

            if created_filename == "mesh.hdf5":
                hdf5_keys = [
                    "frequency",
                    "mesh",
                    "qpoint",
                    "weight",
                ]
                if with_eigenvectors:
                    hdf5_keys.append("eigenvector")
                if with_group_velocities:
                    hdf5_keys.append("group_velocity")

            with h5py.File(file_path) as f:
                assert set(f.keys()) == set(hdf5_keys)

                freqs = f["frequency"]
                if compression in ("gzip", "lzf"):
                    assert freqs.compression == compression  # type: ignore
                elif isinstance(compression, int):
                    assert freqs.compression == "gzip"  # type: ignore
                    assert freqs.compression_opts == compression  # type: ignore
                else:
                    assert freqs.compression is None  # type: ignore

            file_path.unlink()

        _check_no_files()

        os.chdir(original_cwd)


def _test_IterMesh(ph_nacl: Phonopy, freqs_ref):
    ph_nacl.init_mesh(mesh=[3, 3, 3], with_eigenvectors=True, use_iter_mesh=True)
    assert isinstance(ph_nacl.mesh, IterMesh)
    freqs = []
    eigvecs = []
    for f, e in ph_nacl.mesh:
        freqs.append(f)
        eigvecs.append(e)

    # for freqs_q in freqs:
    #     print("".join(["%f, " % f for f in freqs_q]))

    np.testing.assert_allclose(freqs_ref, np.reshape(freqs, -1), atol=1e-5)
    ph_nacl.run_mesh([3, 3, 3], with_eigenvectors=True)
    assert isinstance(ph_nacl.mesh, Mesh)
    mesh_obj = ph_nacl.mesh
    mesh_freqs = mesh_obj.frequencies
    np.testing.assert_allclose(mesh_freqs, freqs, atol=1e-5)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())


def _agno2_phonopy(agno2_cell: PhonopyAtoms) -> Phonopy:
    """Build a minimal Phonopy on the AgNO2 (Imm2) primitive cell.

    Force constants are zeroed because the fallback path is exercised at
    Mesh construction; no real phonons are needed.

    """
    ph = Phonopy(agno2_cell, supercell_matrix=[1, 1, 1], primitive_matrix="auto")
    n_atoms = len(ph.supercell)
    ph.force_constants = np.zeros((n_atoms, n_atoms, 3, 3))
    return ph


def test_mesh_grg_fallback_on_float_input(agno2_cell: PhonopyAtoms):
    """Float input on body-centered primitive triggers GR-grid fallback."""
    ph = _agno2_phonopy(agno2_cell)
    with pytest.warns(MeshGRGridFallbackWarning):
        ph.init_mesh(mesh=20.0)
    # GR-grid result for length=20 on AgNO2 Imm2 primitive.
    np.testing.assert_array_equal(ph.mesh.mesh_numbers, [1, 6, 24])
    assert len(ph.mesh.qpoints) == 36
    # Total weight equals prod(D_diag) (all GR points covered).
    assert int(ph.mesh.weights.sum()) == int(np.prod(ph.mesh.mesh_numbers))


def test_mesh_tr_only_fallback_on_tuple_input(agno2_cell: PhonopyAtoms):
    """3-tuple input keeps the legacy time-reversal-only fallback."""
    ph = _agno2_phonopy(agno2_cell)
    with pytest.warns(MeshSymmetryFallbackWarning):
        ph.init_mesh(mesh=[5, 7, 7])
    # mesh_numbers must match the user-requested 3-tuple (no GR-grid swap).
    np.testing.assert_array_equal(ph.mesh.mesh_numbers, [5, 7, 7])
    # TR-only on [5, 7, 7] = 245 grid points -> 123 ir-qpoints.
    assert len(ph.mesh.qpoints) == 123


def test_mesh_no_fallback_on_uniform_tuple(agno2_cell: PhonopyAtoms):
    """Uniform [N, N, N] is compatible with Imm2 primitive: no fallback."""
    ph = _agno2_phonopy(agno2_cell)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MeshGRGridFallbackWarning)
        warnings.simplefilter("error", MeshSymmetryFallbackWarning)
        ph.init_mesh(mesh=[4, 4, 4])
    np.testing.assert_array_equal(ph.mesh.mesh_numbers, [4, 4, 4])
