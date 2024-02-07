"""Tests of Phonopy --symmetry."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Optional, Union

import pytest

from phonopy.cui.phonopy_script import main

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    filename: Optional[Sequence[os.PathLike]] = None
    conf_filename: Optional[os.PathLike] = None
    log_level: Optional[int] = None
    fc_symmetry: bool = True
    cell_filename: Optional[str] = None
    is_check_symmetry: Optional[bool] = None
    is_graph_plot: Optional[bool] = None
    is_graph_save: Optional[bool] = None
    is_legend: Optional[bool] = None

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))

    def __contains__(self, item):
        return item in (field.name for field in fields(self))


def test_phonopy_load():
    """Test phonopy-load command."""
    # Check sys.exit(0)
    argparse_control = _get_phonopy_load_args(
        cwd / ".." / "phonopy_params_NaCl-1.00.yaml.xz"
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
    """Test phonopy --symmetry command."""
    # Check sys.exit(0)
    argparse_control = _get_phonopy_load_args(
        cwd / ".." / "phonopy_params_NaCl-1.00.yaml.xz",
        load_phonopy_yaml=False,
        is_check_symmetry=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        main(**argparse_control)
    assert excinfo.value.code == 0

    # Clean files created by phonopy --symmetry command.
    for created_filename in ("BPOSCAR", "PPOSCAR", "phonopy_symcells.yaml"):
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        file_path.unlink()


def _get_phonopy_load_args(
    phonopy_yaml_filepath: Union[str, pathlib.Path],
    load_phonopy_yaml: bool = True,
    is_check_symmetry: bool = False,
):
    # Mock of ArgumentParser.args.
    if load_phonopy_yaml:
        mockargs = MockArgs(
            filename=[phonopy_yaml_filepath],
            log_level=1,
        )
    else:
        mockargs = MockArgs(
            filename=[],
            log_level=1,
            cell_filename=phonopy_yaml_filepath,
            is_check_symmetry=is_check_symmetry,
        )

    # See phonopy-load script.
    argparse_control = {
        "fc_symmetry": load_phonopy_yaml,
        "is_nac": load_phonopy_yaml,
        "load_phonopy_yaml": load_phonopy_yaml,
        "args": mockargs,
    }
    return argparse_control
