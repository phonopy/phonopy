"""Tests of phonopy-vasp-efe command."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Sequence

import pytest

from phonopy.cui.phonopy_vasp_efe_script import PhonopyVaspEfeMockArgs, main

cwd = pathlib.Path(__file__).parent


def test_phonopy_vasp_efe():
    """Test phonopy-vasp-efe command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_args(
                filenames=[
                    cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)
                ]
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-vasp-efe script.
            for created_filename in ("e-v.dat", "fe-v.dat"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)
    pass


def _ls():
    current_dir = pathlib.Path(".")
    for file in current_dir.iterdir():
        print(file.name)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())


def _get_args(
    tmax: float = 1000.0,
    tmin: float = 0.0,
    tstep: float = 10.0,
    filenames: Sequence[os.PathLike | str] | None = None,
) -> dict[str, PhonopyVaspEfeMockArgs]:
    mockargs = PhonopyVaspEfeMockArgs(
        tmax=tmax, tmin=tmin, tstep=tstep, filenames=filenames
    )
    return {"args": mockargs}
