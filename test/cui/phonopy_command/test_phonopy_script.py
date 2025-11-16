"""Tests of Phonopy --symmetry."""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest
import yaml

import phonopy
from phonopy.cui.phonopy_argparse import PhonopyMockArgs
from phonopy.cui.phonopy_script import main
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive

cwd = pathlib.Path(__file__).parent


@pytest.mark.parametrize("is_ncl", [False, True])
def test_phonopy_disp_Cr(is_ncl: bool):
    """Test phonopy -d option."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            cell_filename = cwd / "POSCAR-unitcell_Cr"
            if is_ncl:
                magmoms = "0 0 1 0 0 -1"
            else:
                magmoms = "1 -1"

            argparse_control = _get_phonopy_args(
                cell_filename=cell_filename,
                supercell_dimension="2 2 2",
                is_displacement=True,
                magmoms=magmoms,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            magmom_file_path = pathlib.Path("MAGMOM")
            assert magmom_file_path.exists()
            with open(magmom_file_path) as f:
                vals = [float(v) for v in f.readline().split()[2:]]

            if is_ncl:
                np.testing.assert_allclose(
                    vals,
                    np.ravel([[0, 0, 1]] * 8 + [[0, 0, -1]] * 8),  # type: ignore
                )
            else:
                np.testing.assert_allclose(vals, [1.0] * 8 + [-1.0] * 8)

            for created_filename in [
                "MAGMOM",
                "POSCAR-001",
                "SPOSCAR",
                "phonopy_disp.yaml",
            ]:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_create_force_sets():
    """Test phonopy --create-force-sets command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "NaCl" / "phonopy_disp.yaml.xz",
                create_force_sets=[
                    cwd / "NaCl" / "vasprun.xml-001.xz",
                    cwd / "NaCl" / "vasprun.xml-002.xz",
                ],
                load_phonopy_yaml=False,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("FORCE_SETS",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "NaCl" / "phonopy_disp.yaml.xz",
                create_force_sets=[
                    cwd / "NaCl" / "vasprun.xml-002.xz",
                    cwd / "NaCl" / "vasprun.xml-001.xz",
                ],
            )
            with pytest.raises(RuntimeError) as excinfo:
                main(**argparse_control)

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_phonopy_load(load_phonopy_yaml: bool):
    """Test phonopy/phonopy-load command."""
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                load_phonopy_yaml=load_phonopy_yaml,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_unit_conversion_factor(load_phonopy_yaml: bool):
    """Test unit_conversion_factor using phonopy/phonopy-load command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-fd.yaml.xz",
                band_paths="0 0 0 0 0 1/2",
                frequency_conversion_factor=100,
                load_phonopy_yaml=load_phonopy_yaml,
            )
            with pytest.warns(DeprecationWarning):
                with pytest.raises(SystemExit) as excinfo:
                    main(**argparse_control)
            assert excinfo.value.code == 0

            if load_phonopy_yaml:
                ref_freq = 47.3113552091
            else:
                ref_freq = 29.5294946098
            with open("band.yaml") as f:
                band = yaml.safe_load(f)
                assert band["phonon"][0]["band"][5]["frequency"] == pytest.approx(
                    ref_freq
                )

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "band.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_unit_conversion_factor_QE(load_phonopy_yaml: bool):
    """Test unit_conversion_factor for QE using phonopy/phonopy-load command."""
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-QE.yaml.xz",
                band_paths="0 0 0 0 0 1/2",
                load_phonopy_yaml=load_phonopy_yaml,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if load_phonopy_yaml:
                ref_freq = 7.3823649712
            else:
                ref_freq = 4.5259552147
            with open("band.yaml") as f:
                band = yaml.safe_load(f)
                assert band["phonon"][0]["band"][5]["frequency"] == pytest.approx(
                    ref_freq
                )

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "band.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phonopy_is_check_symmetry():
    """Test phonopy --symmetry command with phonopy.yaml input structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                load_phonopy_yaml=False,
                is_check_symmetry=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy --symmetry command.
            file_path = pathlib.Path("phonopy_symcells.yaml")
            assert file_path.exists()
            ph = phonopy.load(file_path)
            assert type(ph.unitcell) is PhonopyAtoms
            assert type(ph.primitive) is Primitive
            file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_conf_file():
    """Test phonopy CONFILE."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / "dim.conf",
                cell_filename=cwd / "POSCAR-unitcell_Cr",
                load_phonopy_yaml=False,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("POSCAR-001", "SPOSCAR", "phonopy_disp.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_config_option():
    """Test phonopy-yaml --config."""
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "mesh.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "mesh.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_anime():
    """Test phonopy/phonopy-load command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                anime="0 0 0",
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            _ls()

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "anime.ascii"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_tdm_cif():
    """Test phonopy command with thermal displacement matrices cif output."""
    pytest.importorskip("symfc")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                thermal_displacement_matrices_cif=1000,
                mesh_numbers="5 5 5",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in (
                "phonopy.yaml",
                "tdispmat.cif",
                "thermal_displacement_matrices.yaml",
            ):
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


def _get_phonopy_args(
    anime: str | None = None,
    band_paths: str | None = None,
    cell_filename: str | os.PathLike | None = None,
    conf_filename: str | os.PathLike | None = None,
    create_force_sets: list[str | os.PathLike] | None = None,
    filename: str | os.PathLike | None = None,
    frequency_conversion_factor: float | None = None,
    is_displacement: bool | None = None,
    is_check_symmetry: bool = False,
    load_phonopy_yaml: bool = False,
    magmoms: str | None = None,
    mesh_numbers: str | None = None,
    supercell_dimension: str | None = None,
    thermal_displacement_matrices_cif: float | None = None,
    use_pypolymlp: bool = False,
):
    if filename is None:
        _filename = []
    else:
        _filename = [filename]
    mockargs = PhonopyMockArgs(
        anime=anime,
        band_paths=band_paths,
        cell_filename=cell_filename,
        conf_filename=conf_filename,
        create_force_sets=create_force_sets,
        filename=_filename,
        frequency_conversion_factor=frequency_conversion_factor,
        is_displacement=is_displacement,
        is_check_symmetry=is_check_symmetry,
        log_level=1,
        magmoms=magmoms,
        mesh_numbers=mesh_numbers,
        thermal_displacement_matrices_cif=thermal_displacement_matrices_cif,
        supercell_dimension=supercell_dimension,
        use_pypolymlp=use_pypolymlp,
    )

    if load_phonopy_yaml:
        argparse_control = {
            "load_phonopy_yaml": True,
            "args": mockargs,
        }
    else:
        argparse_control = {
            "load_phonopy_yaml": False,
            "args": mockargs,
        }
    return argparse_control
