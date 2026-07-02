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


def test_phonopy_vasp_efe_uses_kpoints_opt():
    """Free energy must be evaluated on the kpoints_opt mesh when present.

    The fixture's SCF mesh (14 k-points) and kpoints_opt mesh (24 k-points)
    have different sizes, so the output must match an explicit computation on
    the kpoints_opt mesh.

    """
    from phonopy.interface.vasp import parse_vasprunxml
    from phonopy.qha.electron import get_free_energy_at_T

    filename = cwd.parents[1] / "interface" / "vasprun_kpoints_opt.xml.xz"
    args = PhonopyVaspEfeMockArgs(filenames=[filename], tmax=100.0, tstep=50.0)
    lines_fe, _ = get_fe_ev_lines(args)
    rows = _data_rows(lines_fe)
    assert len(rows) == 3  # T = 0, 50, 100

    vxml = parse_vasprunxml(filename)
    assert vxml.has_kpoints_opt
    _, fe = get_free_energy_at_T(
        0.0,
        100.0,
        50.0,
        vxml.eigenvalues_kpoints_opt[:, :, :, 0],
        vxml.k_weights_kpoints_opt,
        vxml.NELECT,
    )
    ref = vxml.energies[-1, 1] - fe[0] + fe
    for row, r in zip(rows, ref, strict=True):
        np.testing.assert_allclose(row[1], r, rtol=1e-8)


def test_phonopy_vasp_efe_kpoints_opt_values():
    """Golden-value regression test for the KPOINTS_OPT path.

    Unlike test_phonopy_vasp_efe_uses_kpoints_opt (which recomputes the
    reference with the same library functions), this pins absolute output
    numbers so a future change of the underlying free-energy computation on
    the kpoints_opt mesh is caught.

    """
    filename = cwd.parents[1] / "interface" / "vasprun_kpoints_opt.xml.xz"
    args = PhonopyVaspEfeMockArgs(filenames=[filename], tmax=100.0, tstep=50.0)
    lines_fe, lines_ev = get_fe_ev_lines(args)

    ev_rows = _data_rows(lines_ev)
    assert len(ev_rows) == 1
    np.testing.assert_allclose(ev_rows[0][0], 33.23606544, rtol=1e-6)  # volume
    np.testing.assert_allclose(ev_rows[0][1], -21.10777233, rtol=1e-6)  # energy

    fe_rows = _data_rows(lines_fe)
    assert len(fe_rows) == 3  # T = 0, 50, 100
    np.testing.assert_allclose(fe_rows[0], [0.0, -21.10777233], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(fe_rows[1], [50.0, -21.10779563], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(fe_rows[2], [100.0, -21.10797222], rtol=1e-6, atol=1e-5)


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


def test_phonopy_vasp_efe_factor_default_is_one():
    """Test that the default factor (1.0) leaves values unscaled."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    args_default = PhonopyVaspEfeMockArgs(filenames=filenames)
    args_one = PhonopyVaspEfeMockArgs(filenames=filenames, scale_factor=1.0)
    assert args_default.scale_factor == 1.0

    lines_fe_default, lines_ev_default = get_fe_ev_lines(args_default)
    lines_fe_one, lines_ev_one = get_fe_ev_lines(args_one)
    assert lines_fe_default == lines_fe_one
    assert lines_ev_default == lines_ev_one


def test_phonopy_vasp_efe_factor_scales_ev_values():
    """Test that scale_factor scales e-v.dat volumes and energies."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    scale_factor = 2.5

    _, lines_ev_ref = get_fe_ev_lines(PhonopyVaspEfeMockArgs(filenames=filenames))
    _, lines_ev = get_fe_ev_lines(
        PhonopyVaspEfeMockArgs(filenames=filenames, scale_factor=scale_factor)
    )

    ref_rows = _data_rows(lines_ev_ref)
    rows = _data_rows(lines_ev)
    assert len(rows) == len(ref_rows) == 3
    for ref_row, row in zip(ref_rows, rows, strict=True):
        # Both volume and energy columns are scaled by scale_factor.
        np.testing.assert_allclose(row, np.array(ref_row) * scale_factor, rtol=1e-6)


def test_phonopy_vasp_efe_factor_scales_fe_values():
    """Test that scale_factor scales fe-v.dat free energies but not temperatures."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    scale_factor = 2.5

    lines_fe_ref, _ = get_fe_ev_lines(PhonopyVaspEfeMockArgs(filenames=filenames))
    lines_fe, _ = get_fe_ev_lines(
        PhonopyVaspEfeMockArgs(filenames=filenames, scale_factor=scale_factor)
    )

    ref_rows = _data_rows(lines_fe_ref)
    rows = _data_rows(lines_fe)
    assert len(rows) == len(ref_rows) == 101
    for ref_row, row in zip(ref_rows, rows, strict=True):
        # Temperature column (index 0) is untouched.
        np.testing.assert_allclose(row[0], ref_row[0], atol=1e-5)
        # Free energy columns are scaled by scale_factor.
        np.testing.assert_allclose(
            row[1:], np.array(ref_row[1:]) * scale_factor, rtol=1e-6
        )

    # The volume header line is also scaled.
    vol_ref = [float(v) for v in lines_fe_ref[0].split()[2:]]
    vol = [float(v) for v in lines_fe[0].split()[2:]]
    np.testing.assert_allclose(vol, np.array(vol_ref) * scale_factor, rtol=1e-6)


def test_phonopy_vasp_efe_factor_main(tmp_path):
    """Test scale_factor passed through main() writes scaled output files."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        argparse_control = _get_args(scale_factor=2.0, filenames=filenames)
        with pytest.raises(SystemExit) as excinfo:
            main(**argparse_control)
        assert excinfo.value.code == 0
        for created_filename in ("e-v.dat", "fe-v.dat"):
            assert pathlib.Path(created_filename).exists()
    finally:
        os.chdir(original_cwd)


def test_phonopy_vasp_efe_write_electronic_states(tmp_path):
    """--write-electronic-states writes electronic_states.hdf5 and e-v.dat.

    The stored states must reproduce the fe-v.dat values when the free
    energies are recomputed from them.

    """
    from phonopy.qha.electron import (
        get_free_energy_at_T,
        read_electronic_states_hdf5,
    )

    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        mockargs = PhonopyVaspEfeMockArgs(
            filenames=filenames, write_electronic_states=True, quiet=True
        )
        with pytest.raises(SystemExit) as excinfo:
            main(args=mockargs)
        assert excinfo.value.code == 0

        assert not pathlib.Path("fe-v.dat").exists()
        assert pathlib.Path("e-v.dat").exists()

        volumes, energies, states = read_electronic_states_hdf5(
            "electronic_states.hdf5"
        )
        assert len(states) == 3
        np.testing.assert_allclose(volumes[0], 43.08047896, rtol=1e-6)
        np.testing.assert_allclose(energies[0], -17.27885993, rtol=1e-6)
        assert states[0].volume is not None
        np.testing.assert_allclose(states[0].volume, 43.08047896, rtol=1e-6)

        # Recomputed F_el(T=1000 K) matches the pinned fe-v.dat value of
        # test_phonopy_vasp_efe_fe_values.
        _, fe = get_free_energy_at_T(
            0.0,
            1000.0,
            10.0,
            states[0].eigenvalues,
            states[0].weights,
            states[0].n_electrons,
        )
        np.testing.assert_allclose(
            energies[0] - fe[0] + fe[-1], -17.29111981, rtol=1e-6
        )
    finally:
        os.chdir(original_cwd)


def test_phonopy_vasp_efe_write_electronic_states_rejects_scale_factor(tmp_path):
    """--scale-factor combined with --write-electronic-states exits with 1."""
    filenames = [cwd / f"vasprun.xmls/vasprun.xml-{i:02d}.xz" for i in range(3)]
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    try:
        mockargs = PhonopyVaspEfeMockArgs(
            filenames=filenames,
            write_electronic_states=True,
            scale_factor=2.0,
            quiet=True,
        )
        with pytest.raises(SystemExit) as excinfo:
            main(args=mockargs)
        assert excinfo.value.code == 1
        _check_no_files()
    finally:
        os.chdir(original_cwd)


def _data_rows(lines: list[str]) -> list[list[float]]:
    return [
        [float(v) for v in line.split()] for line in lines if not line.startswith("#")
    ]


def _ls():
    current_dir = pathlib.Path(".")
    for file in current_dir.iterdir():
        print(file.name)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())


def _get_args(
    scale_factor: float = 1.0,
    tmax: float = 1000.0,
    tmin: float = 0.0,
    tstep: float = 10.0,
    filenames: Sequence[os.PathLike | str] | None = None,
) -> dict[str, PhonopyVaspEfeMockArgs]:
    mockargs = PhonopyVaspEfeMockArgs(
        scale_factor=scale_factor,
        tmax=tmax,
        tmin=tmin,
        tstep=tstep,
        filenames=filenames,
    )
    return {"args": mockargs}
