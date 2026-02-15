"""Tests of phonopy-qe-born command."""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest

from phonopy.cui.phonopy_qe_born_script import PhonopyQeBornMockArgs, main

cwd = pathlib.Path(__file__).parent


@pytest.mark.parametrize("ph_out_file", ["NaCl.ph.out", "NaCl-noborn.ph.out"])
def test_phonopy_qe_born(capsys, ph_out_file: str):
    """Test phonopy-qe-born command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            data_path = cwd / ".." / ".." / "interface" / "qe" / "NaCl-ph"
            argparse_control = _get_args(data_path / "NaCl.in", data_path / ph_out_file)

            if ph_out_file == "NaCl-noborn.ph.out":
                with pytest.raises(RuntimeError) as excinfo:
                    main(**argparse_control)
                assert "Could not find Born effective charges in ph.x output." in str(
                    excinfo.value
                )
            else:
                with pytest.raises(SystemExit) as excinfo:
                    main(**argparse_control)
                assert excinfo.value.code == 0
                captured = capsys.readouterr()

                data = []
                for line in captured.out.splitlines()[1:]:
                    data.append([float(x) for x in line.split()])

                np.testing.assert_allclose(data[0], (np.eye(3) * 2.47441024).ravel())
                np.testing.assert_allclose(data[1], (np.eye(3) * 1.10075500).ravel())
                np.testing.assert_allclose(data[2], (np.eye(3) * -1.10075500).ravel())

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
    file_pw_in: os.PathLike, file_ph_out: os.PathLike
) -> dict[str, PhonopyQeBornMockArgs]:
    mockargs = PhonopyQeBornMockArgs(
        file_pw_in=file_pw_in,
        file_ph_out=file_ph_out,
    )
    return {"args": mockargs}
