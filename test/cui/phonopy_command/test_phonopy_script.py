"""Tests of Phonopy --symmetry."""

from __future__ import annotations

import itertools
import os
import pathlib
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import yaml

import phonopy
from phonopy.cui.phonopy_argparse import (
    PhonopyMockArgs,
    get_init_parser,
    get_run_parser,
)
from phonopy.cui.phonopy_script import _detect_init_operation, main
from phonopy.cui.settings import PhonopySettings
from phonopy.exception import PypolymlpDevelopmentError, PypolymlpFileNotFoundError
from phonopy.structure.atomic_data import set_atomic_data
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


def test_create_force_sets_GeSn_vca():
    """Phonopy -f on a real GeSn 99/1 VCA vasprun.xml writes the expected FORCE_SETS.

    Exercises the CLI ``phonopy -f vasprun.xml -c phonopy_disp.yaml`` path
    on a mixture supercell. The supercell has 16 sites but the SPOSCAR
    was written with ``expand_mixtures=True`` so VASP returned 32 force
    rows; the writer must therefore emit Type-1 with line 1 = 32, and
    the disp atom number must be a 1..16 site index (gamma layout).

    """
    fixtures = cwd / ".." / ".." / "interface"
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                cell_filename=fixtures / "GeSn-vca-phonopy_disp.yaml",
                create_force_sets=[fixtures / "GeSn-vca-vasprun-001.xml.xz"],
                load_phonopy_yaml=False,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            written = pathlib.Path("FORCE_SETS")
            assert written.exists()
            lines = written.read_text().splitlines()
            assert int(lines[0].strip()) == 32  # n_expanded on line 1
            assert int(lines[1].strip()) == 1  # one displacement
            body = [ln.strip() for ln in lines[2:] if ln.strip()]
            atom_number = int(body[0])
            assert 1 <= atom_number <= 16  # 1-based site index

            # Compare against the bundled reference FORCE_SETS.
            from phonopy.file_IO import parse_FORCE_SETS

            ref = parse_FORCE_SETS(natom=16, filename=fixtures / "GeSn-vca-FORCE_SETS")
            got = parse_FORCE_SETS(natom=16, filename=written)
            assert got["natom"] == ref["natom"] == 16
            assert got["first_atoms"][0]["number"] == ref["first_atoms"][0]["number"]
            np.testing.assert_allclose(
                got["first_atoms"][0]["displacement"],
                ref["first_atoms"][0]["displacement"],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                got["first_atoms"][0]["forces"],
                ref["first_atoms"][0]["forces"],
                atol=1e-9,
            )
            written.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phonopy_load_GeSn_vca():
    """Phonopy-load on a GeSn 99/1 site-mixture bundle writes valid force constants.

    Drops phonopy_disp.yaml and FORCE_SETS into the cwd (mirroring how a
    user would have them after running ``phonopy -f``) and invokes the
    phonopy-load entry point with ``--writefc`` so that the resulting
    force constants are persisted as ``FORCE_CONSTANTS``. Verifying the
    written file's shape inside pytest exercises the entire mixture
    pipeline (yaml round-trip, expanded FORCE_SETS parse, FC-time
    reduction, default symfc fc-calculator, FC writer).

    """
    import shutil

    from phonopy.file_IO import parse_FORCE_CONSTANTS

    fixtures = cwd / ".." / ".." / "interface"
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            shutil.copy(
                fixtures / "GeSn-vca-phonopy_disp.yaml",
                pathlib.Path("phonopy_disp.yaml"),
            )
            shutil.copy(fixtures / "GeSn-vca-FORCE_SETS", pathlib.Path("FORCE_SETS"))

            argparse_control = _get_phonopy_args(
                filename=pathlib.Path("phonopy_disp.yaml"),
                load_phonopy_yaml=True,
                write_force_constants=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            assert pathlib.Path("phonopy.yaml").exists()
            assert pathlib.Path("FORCE_CONSTANTS").exists()

            fc = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
            n_sites = 16  # GeSn 2x2x2 mixture supercell
            assert fc.ndim == 4
            assert fc.shape[1] == n_sites
            assert fc.shape[2:] == (3, 3)

            pathlib.Path("phonopy.yaml").unlink()
            pathlib.Path("FORCE_CONSTANTS").unlink()
            pathlib.Path("phonopy_disp.yaml").unlink()
            pathlib.Path("FORCE_SETS").unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_create_force_sets_zero():
    """Test phonopy --force-sets-zero command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "vaspruns_SnO2" / "phonopy_disp.yaml.xz",
                create_force_sets_zero=[
                    cwd / "vaspruns_SnO2" / "vasprun.xml-000.xz",
                    cwd / "vaspruns_SnO2" / "vasprun.xml-001.xz",
                    cwd / "vaspruns_SnO2" / "vasprun.xml-002.xz",
                    cwd / "vaspruns_SnO2" / "vasprun.xml-003.xz",
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

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("save_params", [False, True])
def test_create_force_sets_rd(save_params: bool):
    """Test phonopy with random displacements."""

    def check_supercell_energies(num_force_files: int):
        ph = phonopy.load("phonopy_params.yaml", produce_fc=False)
        np.testing.assert_allclose(
            ph.supercell_energies,
            [-223.886324, -223.875031, -223.880732][:num_force_files],
        )

    if save_params:
        created_filenames = ("phonopy_params.yaml",)
    else:
        created_filenames = ("FORCE_SETS",)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "vaspruns_NaCl_rd" / "phonopy_disp.yaml.xz",
                create_force_sets_zero=[
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00000.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00001.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00002.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00003.xml.xz",
                ],
                load_phonopy_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(3)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            # Allows less number of force files (three disps in phonopy_disp.yaml).
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "vaspruns_NaCl_rd" / "phonopy_disp.yaml.xz",
                create_force_sets_zero=[
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00000.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00001.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00002.xml.xz",
                ],
                load_phonopy_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(2)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "vaspruns_NaCl_rd" / "phonopy_disp.yaml.xz",
                create_force_sets=[
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00001.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00002.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00003.xml.xz",
                ],
                load_phonopy_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(3)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

            # Allows less number of force files (three disps in phonopy_disp.yaml).
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "vaspruns_NaCl_rd" / "phonopy_disp.yaml.xz",
                create_force_sets=[
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00001.xml.xz",
                    cwd / "vaspruns_NaCl_rd" / "vasprun-00002.xml.xz",
                ],
                load_phonopy_yaml=False,
                save_params=save_params,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if save_params:
                check_supercell_energies(2)

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_phonopy_load(load_phonopy_yaml: bool):
    """Test phonopy/phonopy-load command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
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
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-fd.yaml.xz",
                band_paths="0 0 0 0 0 1/2",
                frequency_conversion_factor=100,
                load_phonopy_yaml=load_phonopy_yaml,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if load_phonopy_yaml:
                ref_freq = 47.3113552091
            else:
                ref_freq = 29.4784373936
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
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
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
                ref_freq = 4.5382475036
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
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
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


@pytest.mark.parametrize(
    "hdf5_compression,with_eigenvectors",
    itertools.product(["gzip", "None"], [True, False]),
)
def test_band_h5py(hdf5_compression: str, with_eigenvectors: bool):
    """Test phonopy band structure output in HDF5 format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                band_paths="0 0 0 1/2 1/2 1/2",
                band_points=11,
                load_phonopy_yaml=True,
                is_graph_plot=True,
                is_graph_save=True,
                is_hdf5=True,
                is_eigenvectors=with_eigenvectors,
                hdf5_compression=hdf5_compression,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "band.hdf5", "band.pdf"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

                if created_filename == "band.hdf5":
                    with h5py.File(file_path, "r") as f:
                        if hdf5_compression == "None":
                            assert f["frequency"].compression is None  # type: ignore
                        else:
                            assert f["frequency"].compression == hdf5_compression  # type: ignore
                        if with_eigenvectors:
                            assert "eigenvector" in f
                        else:
                            assert "eigenvector" not in f

                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize(
    "hdf5_compression,with_eigenvectors",
    itertools.product(["gzip", "None"], [True, False]),
)
def test_mesh_h5py(hdf5_compression: str, with_eigenvectors: bool):
    """Test phonopy mesh output in HDF5 format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                mesh_numbers="3 3 3",
                load_phonopy_yaml=True,
                is_hdf5=True,
                is_eigenvectors=with_eigenvectors,
                hdf5_compression=hdf5_compression,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "mesh.hdf5"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

                if created_filename == "mesh.hdf5":
                    with h5py.File(file_path, "r") as f:
                        if hdf5_compression == "None":
                            assert f["frequency"].compression is None  # type: ignore
                        else:
                            assert f["frequency"].compression == hdf5_compression  # type: ignore
                        if with_eigenvectors:
                            assert "eigenvector" in f
                        else:
                            assert "eigenvector" not in f

                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize(
    "hdf5_compression,with_eigenvectors,write_dynamical_matrices",
    itertools.product(["gzip", "None"], [True, False], [True, False]),
)
def test_qpoints_h5py(
    hdf5_compression: str, with_eigenvectors: bool, write_dynamical_matrices: bool
):
    """Test phonopy qpoints output in HDF5 format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                qpoints="0 0 0 0 0 0.5",
                load_phonopy_yaml=True,
                is_hdf5=True,
                is_eigenvectors=with_eigenvectors,
                write_dynamical_matrices=write_dynamical_matrices,
                hdf5_compression=hdf5_compression,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in ("phonopy.yaml", "qpoints.hdf5"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()

                if created_filename == "qpoints.hdf5":
                    with h5py.File(file_path, "r") as f:
                        if hdf5_compression == "None":
                            assert f["frequency"].compression is None  # type: ignore
                        else:
                            assert f["frequency"].compression == hdf5_compression  # type: ignore
                        if with_eigenvectors:
                            assert "eigenvector" in f
                        else:
                            assert "eigenvector" not in f
                        if write_dynamical_matrices:
                            assert "dynamical_matrix" in f
                        else:
                            assert "dynamical_matrix" not in f

                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_import_masses_from_ASE():
    """Test import masses from ASE."""
    pytest.importorskip("ase")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / ".." / ".." / "POSCAR_NaCl",
                supercell_dimension="1 1 1",
                is_displacement=True,
                primitive_axes="auto",
                load_phonopy_yaml=False,
                import_ase_masses_iupac2016=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            ph = phonopy.load("phonopy_disp.yaml", produce_fc=False)
            masses = ph.unitcell.masses
            assert masses[-1] == pytest.approx(35.45, abs=1e-3)

            for created_filename in (
                "POSCAR-001",
                "POSCAR-002",
                "SPOSCAR",
                "phonopy_disp.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            # Reset default atomic masses.
            set_atomic_data()
            os.chdir(original_cwd)


def test_thermal_properties():
    """Test phonopy --tprop command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "tprop.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in (
                "phonopy.yaml",
                "mesh.yaml",
                "thermal_properties.yaml",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_total_dos():
    """Test phonopy --dos command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "dos.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("phonopy.yaml", "mesh.yaml", "total_dos.dat"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_pdos():
    """Test phonopy --pdos command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "pdos.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("phonopy.yaml", "mesh.yaml", "projected_dos.dat"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_irreps():
    """Test phonopy --irreps command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "irreps.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("phonopy.yaml", "irreps.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_thermal_displacements():
    """Test phonopy --thermal-displacements command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "td.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in ("phonopy.yaml", "thermal_displacements.yaml"):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists(), f"{created_filename} was not created"
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_band_mesh():
    """Test phonopy band+mesh (band_mesh) mode."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                conf_filename=cwd / "band_mesh.conf",
                load_phonopy_yaml=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            for created_filename in (
                "phonopy.yaml",
                "band.yaml",
                "mesh.yaml",
                "total_dos.dat",
            ):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_use_pypolymlp_develop(load_phonopy_yaml: bool):
    """Test phonopy --pypolymlp when develop_or_load_pypolymlp succeeds.

    When use_pypolymlp=True and the dataset contains forces, the dataset should
    be moved to mlp_dataset and develop_or_load_pypolymlp should be called.
    Without --rd or -d, the script finalizes and exits with code 0.

    Since pypolymlp is not exectued using MagicMock, polymlp.yaml is not created.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-rd.yaml.xz",
                use_pypolymlp=True,
                load_phonopy_yaml=load_phonopy_yaml,
            )
            with patch(
                "phonopy.cui.phonopy_script.develop_or_load_pypolymlp"
            ) as mock_develop:
                with pytest.raises(SystemExit) as excinfo:
                    main(**argparse_control)
                assert excinfo.value.code == 0
                mock_develop.assert_called_once()

            for created_filename in ("phonopy.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_use_pypolymlp_load_with_displacement_only_dataset(load_phonopy_yaml: bool):
    """Test phonopy --pypolymlp with a displacement-only dataset.

    When use_pypolymlp=True and the dataset contains displacements but no forces
    (e.g. from phonopy_disp.yaml), the dataset must not be moved to mlp_dataset.
    It is discarded so that an existing MLP (polymlp.yaml) is loaded instead of
    triggering training. This used to raise AssertionError before the MLP was
    even loaded.

    Since develop_or_load_pypolymlp is replaced by MagicMock, polymlp.yaml is
    neither read nor created.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_disp_NaCl.yaml",
                use_pypolymlp=True,
                load_phonopy_yaml=load_phonopy_yaml,
            )

            captured = {}

            def _capture(phonon, *args, **kwargs):
                captured["mlp_dataset"] = phonon.mlp_dataset
                captured["dataset"] = phonon.dataset

            with patch(
                "phonopy.cui.phonopy_script.develop_or_load_pypolymlp",
                side_effect=_capture,
            ) as mock_develop:
                with pytest.raises(SystemExit) as excinfo:
                    main(**argparse_control)
                assert excinfo.value.code == 0
                mock_develop.assert_called_once()

            # Displacement-only dataset is discarded, not used as training data.
            assert captured["mlp_dataset"] is None
            assert captured["dataset"] is None

            for created_filename in ("phonopy.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize(
    "error_class",
    [PypolymlpDevelopmentError, PypolymlpFileNotFoundError],
)
def test_use_pypolymlp_develop_error(error_class: type):
    """Test phonopy --pypolymlp when develop_or_load_pypolymlp raises an error.

    PypolymlpDevelopmentError and PypolymlpFileNotFoundError should both cause
    the script to print an error message and exit with code 1.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-rd.yaml.xz",
                use_pypolymlp=True,
                load_phonopy_yaml=True,
            )
            with patch(
                "phonopy.cui.phonopy_script.develop_or_load_pypolymlp",
                side_effect=error_class("test error"),
            ):
                with pytest.raises(SystemExit) as excinfo:
                    main(**argparse_control)
                assert excinfo.value.code == 1

            _check_no_files()

        finally:
            os.chdir(original_cwd)


def test_phonopy_run_mode_rejects_init_flag(capsys: pytest.CaptureFixture[str]):
    """Phonopy in mode='run' errors when an init-only flag like -d is given."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        try:
            argparse_control = _get_phonopy_args(
                cell_filename=cwd / "POSCAR-unitcell_Cr",
                supercell_dimension="2 2 2",
                is_displacement=True,
                primitive_axes="P",
            )
            argparse_control["mode"] = "run"
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 1
            captured = capsys.readouterr()
            assert "phonopy-init" in captured.out
        finally:
            os.chdir(original_cwd)


def test_phonopy_init_mode_requires_init_flag(capsys: pytest.CaptureFixture[str]):
    """Phonopy-init (mode='init') errors when no init flag is given."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
            )
            argparse_control["mode"] = "init"
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 1
            captured = capsys.readouterr()
            assert "No setup operation" in captured.out
        finally:
            os.chdir(original_cwd)


def test_run_parser_accepts_displacement_flag():
    """The phonopy (run) parser accepts -d instead of rejecting it at parse time.

    -d is a shared flag now: the run parser must parse it (so --pypolymlp -d can
    proceed), deferring the setup-operation decision to the settings-level gate.

    """
    parser, _ = get_run_parser()
    args = parser.parse_args(["phonopy_disp.yaml", "-d", "--pypolymlp"])
    assert args.is_displacement is True
    assert args.use_pypolymlp is True

    args = parser.parse_args(["phonopy_disp.yaml", "-d"])
    assert args.is_displacement is True


def test_init_parser_still_accepts_displacement_flag():
    """The phonopy-init parser continues to accept -d after it became shared."""
    parser, _ = get_init_parser()
    args = parser.parse_args(["-c", "POSCAR", "--dim", "2", "2", "2", "-d"])
    assert args.is_displacement is True


def test_detect_init_operation_displacements_with_pypolymlp():
    """-d combined with --pypolymlp is a run operation, not a setup operation.

    Plain -d generates displacement supercells and exits (setup/init), but with
    --pypolymlp it drives displacement generation followed by MLP force
    evaluation and a phonon calculation, so it must not be flagged as init-only.

    """
    settings = PhonopySettings()
    settings.create_displacements = True

    assert _detect_init_operation(False, settings) == "-d"

    settings.use_pypolymlp = True
    assert _detect_init_operation(False, settings) is None


def test_phonopy_run_mode_accepts_pypolymlp_displacements(
    capsys: pytest.CaptureFixture[str],
):
    """Phonopy in mode='run' accepts -d when combined with --pypolymlp.

    Plain -d is rejected in run mode (see test_phonopy_run_mode_rejects_init_flag),
    but --pypolymlp -d is a phonon-calculation workflow and must pass the
    setup-operation guard, entering the pypolymlp displacement-creation path.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_disp_NaCl.yaml",
                is_displacement=True,
                use_pypolymlp=True,
            )
            argparse_control["mode"] = "run"
            with (
                patch("phonopy.cui.phonopy_script.prepare_dataset_by_pypolymlp"),
                patch("phonopy.cui.phonopy_script.develop_or_load_pypolymlp"),
            ):
                with pytest.raises(SystemExit):
                    main(**argparse_control)
            # The run-mode gate must not reject the flag combination. Any exit
            # comes from later stages, so only assert the guard was passed.
            captured = capsys.readouterr()
            assert "setup operation" not in captured.out
            assert "Pypolymlp displacements creation mode" in captured.out

            for created_filename in ("phonopy.yaml",):
                file_path = pathlib.Path(created_filename)
                if file_path.exists():
                    file_path.unlink()
        finally:
            os.chdir(original_cwd)


def test_phonopy_load_deprecation_warning(capsys: pytest.CaptureFixture[str]):
    """Phonopy-load prints a deprecation message pointing to phonopy."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)
        try:
            argparse_control = _get_phonopy_args(
                filename=cwd / ".." / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
                load_phonopy_yaml=True,
            )
            argparse_control["mode"] = "run"
            argparse_control["deprecated_command"] = "phonopy-load"
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0
            captured = capsys.readouterr()
            assert "phonopy-load' is deprecated" in captured.out

            for created_filename in ("phonopy.yaml",):
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


def _get_phonopy_args(
    anime: str | None = None,
    band_paths: str | None = None,
    band_points: int | None = None,
    cell_filename: str | os.PathLike | None = None,
    conf_filename: str | os.PathLike | None = None,
    create_force_sets: list[str | os.PathLike] | None = None,
    create_force_sets_zero: list[str | os.PathLike] | None = None,
    fc_spg_symmetry: bool | None = None,
    filename: str | os.PathLike | None = None,
    frequency_conversion_factor: float | None = None,
    hdf5_compression: str | None = None,
    import_ase_masses_iupac2016: bool | None = None,
    is_check_symmetry: bool | None = None,
    is_displacement: bool | None = None,
    is_eigenvectors: bool | None = None,
    is_graph_plot: bool | None = None,
    is_graph_save: bool | None = None,
    is_hdf5: bool | None = None,
    load_phonopy_yaml: bool = False,
    magmoms: str | None = None,
    mesh_numbers: str | None = None,
    primitive_axes: str | None = None,
    qpoints: str | None = None,
    save_params: bool | None = None,
    supercell_dimension: str | None = None,
    thermal_displacement_matrices_cif: float | None = None,
    use_pypolymlp: bool | None = None,
    write_dynamical_matrices: bool | None = None,
    write_force_constants: bool | None = None,
    writefc_format: str | None = None,
):
    if filename is None:
        _filename = []
    else:
        _filename = [filename]

    mockargs = PhonopyMockArgs(
        anime=anime,
        band_paths=band_paths,
        band_points=band_points,
        cell_filename=cell_filename,
        conf_filename=conf_filename,
        create_force_sets=create_force_sets,
        create_force_sets_zero=create_force_sets_zero,
        fc_spg_symmetry=fc_spg_symmetry,
        filename=_filename,
        frequency_conversion_factor=frequency_conversion_factor,
        hdf5_compression=hdf5_compression,
        import_ase_masses_iupac2016=import_ase_masses_iupac2016,
        is_check_symmetry=is_check_symmetry,
        is_displacement=is_displacement,
        is_eigenvectors=is_eigenvectors,
        is_graph_plot=is_graph_plot,
        is_graph_save=is_graph_save,
        is_hdf5=is_hdf5,
        log_level=1,
        magmoms=magmoms,
        mesh_numbers=mesh_numbers,
        primitive_axes=primitive_axes,
        qpoints=qpoints,
        thermal_displacement_matrices_cif=thermal_displacement_matrices_cif,
        save_params=save_params,
        supercell_dimension=supercell_dimension,
        use_pypolymlp=use_pypolymlp,
        write_dynamical_matrices=write_dynamical_matrices,
        write_force_constants=write_force_constants,
        writefc_format=writefc_format,
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
