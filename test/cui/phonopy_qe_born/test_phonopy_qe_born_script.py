"""Tests of phonopy-qe-born command."""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest

from phonopy.cui.phonopy_qe_born_script import (
    PhonopyQeBornMockArgs,
    get_born_qe_ph,
    main,
    parse_ph_out,
)

cwd = pathlib.Path(__file__).parent
data_path = cwd / ".." / ".." / "interface" / "qe" / "NaCl-ph"


@pytest.mark.parametrize("ph_out_file", ["NaCl.ph.out", "NaCl-noborn.ph.out"])
def test_phonopy_qe_born(capsys, ph_out_file: str):
    """Test phonopy-qe-born command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_args(data_path / "NaCl.in", data_path / ph_out_file)

            if ph_out_file == "NaCl-noborn.ph.out":
                with pytest.raises(RuntimeError) as excinfo:
                    main(**argparse_control)
                assert "Could not find dielectric tensor in ph.x output." in str(
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


def test_parse_ph_out():
    """Test parse_ph_out parses epsilon and BEC from NaCl.ph.out."""
    epsilon, bec = parse_ph_out(data_path / "NaCl.ph.out", natoms=2)

    np.testing.assert_allclose(epsilon, np.eye(3) * 2.47441024, rtol=1e-6)
    assert bec.shape == (2, 3, 3)
    np.testing.assert_allclose(bec[0], np.eye(3) * 1.09885, rtol=1e-4)
    np.testing.assert_allclose(bec[1], np.eye(3) * -1.10266, rtol=1e-4)


def test_parse_ph_out_natoms_mismatch():
    """Test parse_ph_out raises AssertionError when natoms doesn't match."""
    with pytest.raises(AssertionError):
        parse_ph_out(data_path / "NaCl.ph.out", natoms=3)


def test_get_born_qe_ph_no_symmetrize():
    """Test get_born_qe_ph with symmetrize_tensors=False returns raw parsed values."""
    borns, epsilon, indices = get_born_qe_ph(
        data_path / "NaCl.in",
        data_path / "NaCl.ph.out",
        symmetrize_tensors=False,
    )

    np.testing.assert_allclose(epsilon, np.eye(3) * 2.47441024, rtol=1e-6)
    assert borns.shape == (2, 3, 3)
    np.testing.assert_allclose(borns[0], np.eye(3) * 1.09885, rtol=1e-4)
    np.testing.assert_allclose(borns[1], np.eye(3) * -1.10266, rtol=1e-4)
    np.testing.assert_array_equal(indices, [0, 1])


def test_phonopy_qe_born_no_symmetrize(capsys):
    """Test phonopy-qe-born --no-symmetrize-tensors outputs raw BEC values."""
    argparse_control = {
        "args": PhonopyQeBornMockArgs(
            file_pw_in=data_path / "NaCl.in",
            file_ph_out=data_path / "NaCl.ph.out",
            symmetrize_tensors=False,
        )
    }
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    data = []
    for line in captured.out.splitlines()[1:]:
        data.append([float(x) for x in line.split()])

    # Without symmetrization, BEC values differ from the symmetrized ones
    np.testing.assert_allclose(data[0], (np.eye(3) * 2.47441024).ravel(), rtol=1e-6)
    np.testing.assert_allclose(data[1], (np.eye(3) * 1.09885).ravel(), rtol=1e-4)
    np.testing.assert_allclose(data[2], (np.eye(3) * -1.10266).ravel(), rtol=1e-4)


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
