"""Tests of Phonopy --symmetry."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import pytest

from phonopy.cui.phonopy_script import main

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    filename: Optional[Sequence[Union[os.PathLike, str]]] = None
    conf_filename: Optional[os.PathLike] = None
    log_level: Optional[int] = None
    fc_symmetry: bool = True
    cell_filename: Optional[str] = None
    conf_filename: Optional[str] = None
    create_force_sets: Optional[list[str]] = None
    is_check_symmetry: Optional[bool] = None
    is_graph_plot: Optional[bool] = None
    is_graph_save: Optional[bool] = None
    is_legend: Optional[bool] = None
    is_displacement: Optional[bool] = None
    supercell_dimension: Optional[str] = None
    magmoms: Optional[str] = None
    use_pypolymlp: bool = False

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))

    def __contains__(self, item):
        """Implement in operator."""
        return item in (field.name for field in fields(self))


@pytest.mark.parametrize("is_ncl", [False, True])
def test_phonopy_disp_Cr(is_ncl: bool):
    """Test phonopy -d option."""
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

    magmom_file_path = pathlib.Path(cwd_called / "MAGMOM")
    assert magmom_file_path.exists()
    with open(magmom_file_path) as f:
        vals = [float(v) for v in f.readline().split()[2:]]

    if is_ncl:
        np.testing.assert_allclose(vals, np.ravel([[0, 0, 1]] * 8 + [[0, 0, -1]] * 8))
    else:
        np.testing.assert_allclose(vals, [1.0] * 8 + [-1.0] * 8)

    for created_filename in ["MAGMOM", "POSCAR-001", "SPOSCAR", "phonopy_disp.yaml"]:
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def test_create_force_sets():
    """Test phonopy --create-force-sets command."""
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
        file_path = pathlib.Path(cwd_called / created_filename)
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


@pytest.mark.parametrize("load_phonopy_yaml", [False, True])
def test_phonopy_load(load_phonopy_yaml: bool):
    """Test phonopy-load command."""
    pytest.importorskip("symfc")
    # Check sys.exit(0)
    argparse_control = _get_phonopy_args(
        filename=cwd / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
        load_phonopy_yaml=load_phonopy_yaml,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phonopy-load script.
    for created_filename in ("phonopy.yaml",):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def test_phonopy_is_check_symmetry():
    """Test phonopy --symmetry command with phonopy.yaml input structure."""
    # Check sys.exit(0)
    argparse_control = _get_phonopy_args(
        filename=cwd / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
        load_phonopy_yaml=False,
        is_check_symmetry=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phonopy --symmetry command.
    file_path = pathlib.Path(cwd_called / "phonopy_symcells.yaml")
    assert file_path.exists()
    file_path.unlink()


def test_conf_file():
    """Test phonopy CONFILE."""
    argparse_control = _get_phonopy_args(
        filename=cwd / "dim.conf",
        cell_filename=cwd / "POSCAR-unitcell_Cr",
        load_phonopy_yaml=False,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    for created_filename in ("POSCAR-001", "SPOSCAR", "phonopy_disp.yaml"):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def test_config_option():
    """Test phonopy-yaml --config."""
    pytest.importorskip("symfc")
    argparse_control = _get_phonopy_args(
        filename=cwd / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
        conf_filename=cwd / "mesh.conf",
        load_phonopy_yaml=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phonopy-load script.
    for created_filename in ("phonopy.yaml", "mesh.yaml"):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def _get_phonopy_args(
    cell_filename: Optional[Union[str, pathlib.Path]] = None,
    create_force_sets: Optional[list[str]] = None,
    supercell_dimension: Optional[str] = None,
    is_displacement: Optional[bool] = None,
    magmoms: Optional[str] = None,
    load_phonopy_yaml: bool = False,
    is_check_symmetry: bool = False,
    filename: Optional[str] = None,
    conf_filename: Optional[str] = None,
    use_pypolymlp: bool = False,
):
    if filename is None:
        _filename = []
    else:
        _filename = [filename]
    mockargs = MockArgs(
        filename=_filename,
        log_level=1,
        cell_filename=cell_filename,
        create_force_sets=create_force_sets,
        supercell_dimension=supercell_dimension,
        is_displacement=is_displacement,
        magmoms=magmoms,
        is_check_symmetry=is_check_symmetry,
        conf_filename=conf_filename,
        use_pypolymlp=use_pypolymlp,
    )

    if load_phonopy_yaml:
        argparse_control = {
            "fc_symmetry": True,
            "is_nac": True,
            "load_phonopy_yaml": True,
            "args": mockargs,
        }
    else:
        argparse_control = {
            "fc_symmetry": False,
            "is_nac": False,
            "load_phonopy_yaml": False,
            "args": mockargs,
        }
    return argparse_control
