"""Tests of phonopy-vasp-efe command."""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Sequence

import numpy as np
import pytest

from phonopy.cui.phonopy_vasp_efe_script import (
    PhonopyVaspEfeMockArgs,
    get_fe_ev_lines,
    get_free_energy_lines,
    main,
)

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


def test_phonopy_vasp_efe_ev_values():
    """Test e-v.dat numerical values."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    args = PhonopyVaspEfeMockArgs(filenames=filenames)
    _, lines_ev = get_fe_ev_lines(args)

    data_lines = [line for line in lines_ev if not line.startswith("#")]
    assert len(data_lines) == 3

    row0 = [float(v) for v in data_lines[0].split()]
    np.testing.assert_allclose(row0[0], 43.08047896, rtol=1e-6)
    np.testing.assert_allclose(row0[1], -17.27885993, rtol=1e-6)

    row2 = [float(v) for v in data_lines[2].split()]
    np.testing.assert_allclose(row2[0], 44.87549882, rtol=1e-6)
    np.testing.assert_allclose(row2[1], -17.34336569, rtol=1e-6)


def test_phonopy_vasp_efe_fe_values():
    """Test fe-v.dat numerical values."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    args = PhonopyVaspEfeMockArgs(filenames=filenames)
    lines_fe, _ = get_fe_ev_lines(args)

    data_lines = [line for line in lines_fe if not line.startswith("#")]
    assert len(data_lines) == 101  # T=0..1000 step 10

    # T=0 row: free energies equal to sigma->0 energies
    row0 = [float(v) for v in data_lines[0].split()]
    np.testing.assert_allclose(row0[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(row0[1], -17.27885993, rtol=1e-6)
    np.testing.assert_allclose(row0[2], -17.32227490, rtol=1e-6)
    np.testing.assert_allclose(row0[3], -17.34336569, rtol=1e-6)

    # T=1000 row
    row_last = [float(v) for v in data_lines[-1].split()]
    np.testing.assert_allclose(row_last[0], 1000.0, atol=1e-5)
    np.testing.assert_allclose(row_last[1], -17.29111981, rtol=1e-6)
    np.testing.assert_allclose(row_last[2], -17.33482259, rtol=1e-6)
    np.testing.assert_allclose(row_last[3], -17.35625526, rtol=1e-6)


def test_phonopy_vasp_efe_temperature_range():
    """Test phonopy-vasp-efe with custom temperature range."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    args = PhonopyVaspEfeMockArgs(filenames=filenames, tmax=200.0, tstep=100.0)
    lines_fe, _ = get_fe_ev_lines(args)

    data_lines = [line for line in lines_fe if not line.startswith("#")]
    assert len(data_lines) == 3  # T=0, 100, 200

    row1 = [float(v) for v in data_lines[1].split()]
    np.testing.assert_allclose(row1[0], 100.0, atol=1e-5)
    np.testing.assert_allclose(row1[1], -17.27897379, rtol=1e-6)

    row2 = [float(v) for v in data_lines[2].split()]
    np.testing.assert_allclose(row2[0], 200.0, atol=1e-5)
    np.testing.assert_allclose(row2[1], -17.27926217, rtol=1e-6)


def test_get_free_energy_lines():
    """Test get_free_energy_lines formatting."""
    temperatures = np.array([0.0, 100.0, 200.0])
    free_energies = np.array([[-1.0, -2.0], [-1.1, -2.1], [-1.2, -2.2]])
    lines = get_free_energy_lines(temperatures, free_energies)
    assert len(lines) == 3
    row = [float(v) for v in lines[1].split()]
    np.testing.assert_allclose(row[0], 100.0, atol=1e-5)
    np.testing.assert_allclose(row[1], -1.1, rtol=1e-6)
    np.testing.assert_allclose(row[2], -2.1, rtol=1e-6)


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
