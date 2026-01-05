"""Tests of Phonopy --symmetry."""

from __future__ import annotations

import itertools
import os
import pathlib
import tempfile

import h5py
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
            # Check sys.exit(0)
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
            # Check sys.exit(0)
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
            # Check sys.exit(0)
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
    is_check_symmetry: bool | None = None,
    is_displacement: bool | None = None,
    is_eigenvectors: bool | None = None,
    is_graph_plot: bool | None = None,
    is_graph_save: bool | None = None,
    is_hdf5: bool | None = None,
    load_phonopy_yaml: bool = False,
    magmoms: str | None = None,
    mesh_numbers: str | None = None,
    qpoints: str | None = None,
    save_params: bool | None = None,
    supercell_dimension: str | None = None,
    thermal_displacement_matrices_cif: float | None = None,
    use_pypolymlp: bool | None = None,
    write_dynamical_matrices: bool | None = None,
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
        is_check_symmetry=is_check_symmetry,
        is_displacement=is_displacement,
        is_eigenvectors=is_eigenvectors,
        is_graph_plot=is_graph_plot,
        is_graph_save=is_graph_save,
        is_hdf5=is_hdf5,
        log_level=1,
        magmoms=magmoms,
        mesh_numbers=mesh_numbers,
        qpoints=qpoints,
        thermal_displacement_matrices_cif=thermal_displacement_matrices_cif,
        save_params=save_params,
        supercell_dimension=supercell_dimension,
        use_pypolymlp=use_pypolymlp,
        write_dynamical_matrices=write_dynamical_matrices,
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
