"""Tests for band structure calculation."""

from __future__ import annotations

import itertools
import os
import pathlib
import tempfile
from typing import Literal, cast

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints

cwd_called = pathlib.Path.cwd()


def test_band_structure(ph_nacl: Phonopy):
    """Test band structure calculation by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=False
    )
    freqs = ph_nacl.get_band_structure_dict()["frequencies"]
    assert len(freqs) == 3
    assert freqs[0].shape == (11, 6)
    np.testing.assert_allclose(
        freqs[0][0], [0, 0, 0, 4.61643516, 4.61643516, 7.39632718], atol=1e-3
    )


def test_band_structure_gv(ph_nacl: Phonopy):
    """Test band structure calculation with group velocity by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=True, is_band_connection=False
    )
    assert "group_velocities" in ph_nacl.get_band_structure_dict()


def test_band_structure_bc(ph_nacl: Phonopy):
    """Test band structure calculation with band connection by NaCl."""
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=False
    )
    freqs = ph_nacl.get_band_structure_dict()["frequencies"]
    ph_nacl.run_band_structure(
        _get_band_qpoints(), with_group_velocities=False, is_band_connection=True
    )
    freqs_bc = ph_nacl.get_band_structure_dict()["frequencies"]

    # Order of bands is changed by is_band_connection=True.
    np.testing.assert_allclose(
        freqs_bc[1][-1], freqs[1][-1][[0, 1, 5, 3, 4, 2]], atol=1e-3
    )


@pytest.mark.parametrize(
    "compression,is_band_const_interval,with_eigenvectors,with_group_velocities",
    itertools.product(
        ["gzip", "lzf", 1, 2, None],
        [False, True],
        [False, True],
        [False, True],
    ),
)
def test_band_structure_write_hdf5(
    ph_nacl: Phonopy,
    compression: Literal["gzip", "lzf"] | int | None,
    is_band_const_interval: bool,
    with_group_velocities: bool,
    with_eigenvectors: bool,
):
    """Test band structure calculation by NaCl.

    G -> L  False
    X -> G  True
    G -> W  False (last one has to be False)

    """
    _test_band_structure_write_hdf5(
        ph_nacl,
        labels=["G", "L", "X", "G", "W"],
        compression=compression,
        is_band_const_interval=is_band_const_interval,
        with_eigenvectors=with_eigenvectors,
        with_group_velocities=with_group_velocities,
    )


def _test_band_structure_write_hdf5(
    ph_nacl: Phonopy,
    labels: list[str],
    compression: Literal["gzip", "lzf"] | int | None = None,
    is_band_const_interval: bool = False,
    with_eigenvectors: bool = False,
    with_group_velocities: bool = False,
):
    """Test writing hdf5 of band structure calculation by NaCl."""
    if is_band_const_interval:
        qpoints = _get_band_qpoints(np.linalg.inv(ph_nacl.primitive.cell))
    else:
        qpoints = _get_band_qpoints()
    ph_nacl.run_band_structure(
        qpoints,
        path_connections=[False, True, False],
        with_group_velocities=with_group_velocities,
        with_eigenvectors=with_eigenvectors,
        is_band_connection=False,
        is_legacy_plot=False,
        labels=labels,
    )
    assert ph_nacl.band_structure is not None

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        ph_nacl.band_structure.write_hdf5(compression=compression)

        for created_filename in ["band.hdf5"]:
            file_path = pathlib.Path(created_filename)
            assert file_path.exists()
            pairs_ref = [labels[i] for i in (0, 1, 2, 3, 3, 4)]

            if created_filename == "band.hdf5":
                hdf5_keys = [
                    "coordinates",
                    "distance",
                    "frequency",
                    "label",
                    "lattice",
                    "masses",
                    "natom",
                    "nqpoint",
                    "numbers",
                    "path",
                    "reciprocal_lattice",
                    "segment_nqpoint",
                    "symbols",
                ]
                if with_eigenvectors:
                    hdf5_keys.append("eigenvector")
                if with_group_velocities:
                    hdf5_keys.append("group_velocity")

            with h5py.File(file_path) as f:
                assert set(f.keys()) == set(hdf5_keys)

                pairs = []
                for pair in f["label"][:]:  # type: ignore
                    pairs += [pair[0].decode(), pair[1].decode()]
                assert pairs == pairs_ref

                freqs = f["frequency"]
                if compression in ("gzip", "lzf"):
                    assert freqs.compression == compression  # type: ignore
                elif isinstance(compression, int):
                    assert freqs.compression == "gzip"  # type: ignore
                    assert freqs.compression_opts == compression  # type: ignore
                else:
                    assert freqs.compression is None  # type: ignore

                _assert_band_structure_hdf5_arrays(
                    cast(h5py.Dataset, freqs), ph_nacl.band_structure.frequencies
                )
                _assert_band_structure_hdf5_arrays(
                    cast(h5py.Dataset, f["distance"]),
                    ph_nacl.band_structure.distances,
                )
                _assert_band_structure_hdf5_arrays(
                    cast(h5py.Dataset, f["path"]),
                    ph_nacl.band_structure.qpoints,
                )
                if with_eigenvectors:
                    _assert_band_structure_hdf5_arrays(
                        cast(h5py.Dataset, f["eigenvector"]),
                        ph_nacl.band_structure.eigenvectors,
                    )
                if with_group_velocities:
                    _assert_band_structure_hdf5_arrays(
                        cast(h5py.Dataset, f["group_velocity"]),
                        ph_nacl.band_structure.group_velocities,
                    )

            file_path.unlink()

        _check_no_files()

        os.chdir(original_cwd)


def _assert_band_structure_hdf5_arrays(
    dataset: h5py.Dataset, ref_array: list[NDArray] | None
):
    assert ref_array is not None
    val_array = dataset[:]
    for i, ref_f in enumerate(ref_array):
        np.testing.assert_allclose(
            val_array[i][: len(ref_f)],  # type: ignore
            ref_f,
            atol=1e-5,
        )  # type: ignore


def _get_band_qpoints(reclat: NDArray | None = None):
    band_paths = [
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [[0.5, 0.5, 0], [0, 0, 0], [0.5, 0.25, 0.75]],
    ]
    qpoints = get_band_qpoints(band_paths, npoints=11, rec_lattice=reclat)
    return qpoints


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())
