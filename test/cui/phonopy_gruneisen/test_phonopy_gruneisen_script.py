"""Tests of phonopy-gruneisen command."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Sequence

import pytest

from phonopy.cui.phonopy_gruneisen_script import PhonopyGruneisenMockArgs, main

cwd = pathlib.Path(__file__).parent


def test_phonopy_gruneisen_band():
    """Test phonopy-gruneisen --band command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_args(
                band_paths="0 0 0 1/2 1/2 1/2",
                cell_filename="POSCAR-unitcell",
                dirnames=[
                    cwd / "NaCl-gruneisen" / f"NaCl-{vol}"
                    for vol in ("1.00", "0.995", "1.005")
                ],
                primitive_axes="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0",
                supercell_dimension="2 2 2",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-gruneisen script.
            for created_filename in ("gruneisen_band.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phonopy_gruneisen_mesh():
    """Test phonopy-gruneisen --mesh command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_args(
                sampling_mesh="10 10 10",
                cell_filename="POSCAR-unitcell",
                dirnames=[
                    cwd / "NaCl-gruneisen" / f"NaCl-{vol}"
                    for vol in ("1.00", "0.995", "1.005")
                ],
                primitive_axes="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0",
                supercell_dimension="2 2 2",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-gruneisen script.
            for created_filename in ("gruneisen_mesh.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def _ls():
    current_dir = pathlib.Path(".")
    for file in current_dir.iterdir():
        print(file.name)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())


def _get_args(
    dirnames: Sequence[os.PathLike | str],
    band_paths: str | None = None,
    cell_filename: os.PathLike | str | None = None,
    sampling_mesh: str | None = None,
    primitive_axes: str | None = None,
    supercell_dimension: str | None = None,
) -> dict[str, PhonopyGruneisenMockArgs]:
    mockargs = PhonopyGruneisenMockArgs(
        dirnames=dirnames,
        band_paths=band_paths,
        cell_filename=cell_filename,
        sampling_mesh=sampling_mesh,
        primitive_axes=primitive_axes,
        supercell_dimension=supercell_dimension,
    )
    return {"args": mockargs}
