"""Tests of phonopy-qha command."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Sequence

import numpy as np
import pytest

from phonopy.cui.phonopy_qha_script import PhonopyQHAMockArgs, main

cwd = pathlib.Path(__file__).parent


def test_phonopy_qha():
    """Test phonopy-qha command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            filenames = [cwd / "Cu-QHA" / "e-v.dat"] + [
                cwd / "Cu-QHA" / f"thermal_properties.yaml-{i:02d}.xz"
                for i in range(11)
            ]
            argparse_control = _get_args(filenames=filenames, tmax=1010)
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            output_filenames = [
                "helmholtz-volume.dat",
                "gibbs-temperature.dat",
                "dsdv-temperature.dat",
                "Cp-temperature_polyfit.dat",
                "helmholtz-volume_fitted.dat",
                "Cv-volume.dat",
                "thermal_expansion.dat",
                "Cp-temperature.dat",
                "volume-temperature.dat",
                "entropy-volume.dat",
                "gruneisen-temperature.dat",
                "bulk_modulus-temperature.dat",
            ]
            # Clean files created by phonopy-qha script.
            for created_filename in output_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

                if created_filename == "gibbs-temperature.dat":
                    with open(file_path) as f:
                        last_lines = []
                        for line in f.readlines()[-2:]:
                            last_lines.append([float(v) for v in line.split()])
                        np.testing.assert_allclose(
                            last_lines[0],
                            [1000, -18.86959466],
                            rtol=1e-5,
                        )
                        np.testing.assert_allclose(
                            last_lines[1],
                            [1010, -18.89588952],
                            rtol=1e-5,
                        )

                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phonopy_qha_efe():
    """Test phonopy-qha --efe command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            filenames = [cwd / "Cu-QHA" / "e-v.dat"] + [
                cwd / "Cu-QHA" / f"thermal_properties.yaml-{i:02d}.xz"
                for i in range(11)
            ]
            argparse_control = _get_args(
                filenames=filenames, efe_file=cwd / "Cu-QHA" / "fe-v.dat"
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            output_filenames = [
                "helmholtz-volume.dat",
                "gibbs-temperature.dat",
                "dsdv-temperature.dat",
                "Cp-temperature_polyfit.dat",
                "helmholtz-volume_fitted.dat",
                "Cv-volume.dat",
                "thermal_expansion.dat",
                "Cp-temperature.dat",
                "volume-temperature.dat",
                "entropy-volume.dat",
                "gruneisen-temperature.dat",
                "bulk_modulus-temperature.dat",
            ]
            # Clean files created by phonopy-qha script.
            for created_filename in output_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

                if created_filename == "gibbs-temperature.dat":
                    with open(file_path) as f:
                        last_line = [float(v) for v in f.readlines()[-1].split()]
                        np.testing.assert_allclose(
                            last_line,
                            [1000, -18.88302869],
                            rtol=1e-5,
                        )

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
    filenames: Sequence[os.PathLike | str],
    efe_file: os.PathLike | str | None = None,
    tmax: float = 1000.0,
) -> dict[str, PhonopyQHAMockArgs]:
    mockargs = PhonopyQHAMockArgs(filenames=filenames, efe_file=efe_file, tmax=tmax)
    return {"args": mockargs}
