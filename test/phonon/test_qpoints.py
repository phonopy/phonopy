"""Tests for phonon calculation at specific q-points."""

from __future__ import annotations

import itertools
import os
import pathlib
import tempfile
from typing import Literal

import h5py
import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.physical_units import get_physical_units


def test_Qpoints(ph_nacl_nofcsym: Phonopy):
    """Test phonon calculation at specific q-points by NaCl."""
    phonon = ph_nacl_nofcsym
    qpoints = [[0, 0, 0], [0, 0, 0.5]]
    phonon.run_qpoints(qpoints, with_dynamical_matrices=True)
    assert phonon.qpoints is not None
    assert phonon.qpoints.dynamical_matrices is not None

    for i, _ in enumerate(qpoints):
        dm = phonon.qpoints.dynamical_matrices[i]
        dm_eigs = np.linalg.eigvalsh(dm).real
        eigs = phonon.qpoints.eigenvalues[i]
        freqs = phonon.qpoints.frequencies[i] / get_physical_units().DefaultToTHz
        np.testing.assert_allclose(dm_eigs, eigs)
        np.testing.assert_allclose(freqs**2 * np.sign(freqs), eigs)


def test_Qpoints_with_NAC_qdirection(ph_nacl: Phonopy):
    """Test phonon calculation at specific q-points by NaCl."""
    phonon = ph_nacl
    qpoints = [[0, 0, 0]]
    phonon.run_qpoints(qpoints)
    freqs = phonon.get_qpoints_dict()["frequencies"]
    np.testing.assert_allclose(
        freqs, [[0, 0, 0, 4.61643516, 4.61643516, 4.61643516]], atol=1e-5
    )
    phonon.run_qpoints(qpoints, nac_q_direction=[1, 0, 0])
    freqs = phonon.get_qpoints_dict()["frequencies"]
    np.testing.assert_allclose(
        freqs, [[0, 0, 0, 4.61643516, 4.61643516, 7.39632718]], atol=1e-5
    )


@pytest.mark.parametrize(
    "compression,with_eigenvectors,with_group_velocities,with_dynamical_matrices",
    itertools.product(
        ["gzip", "lzf", 1, 2, None],
        [False, True],
        [False, True],
        [False, True],
    ),
)
def test_qpoints_write_hdf5(
    ph_nacl: Phonopy,
    compression: Literal["gzip", "lzf"] | int | None,
    with_eigenvectors: bool,
    with_group_velocities: bool,
    with_dynamical_matrices: bool,
):
    """Test hdf5 output of mesh calculation by NaCl."""
    ph_nacl.run_qpoints(
        [[0, 0, 0], [0, 0, 0.5]],
        with_group_velocities=with_group_velocities,
        with_eigenvectors=with_eigenvectors,
        with_dynamical_matrices=with_dynamical_matrices,
    )
    assert ph_nacl.qpoints is not None

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        ph_nacl.qpoints.write_hdf5(compression=compression)

        for created_filename in ["qpoints.hdf5"]:
            file_path = pathlib.Path(created_filename)
            assert file_path.exists()

            if created_filename == "qpoints.hdf5":
                hdf5_keys = [
                    "frequency",
                    "masses",
                    "qpoint",
                    "reciprocal_lattice",
                ]
                if with_eigenvectors:
                    hdf5_keys.append("eigenvector")
                if with_group_velocities:
                    hdf5_keys.append("group_velocity")
                if with_dynamical_matrices:
                    hdf5_keys.append("dynamical_matrix")

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


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())
