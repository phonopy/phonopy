"""Tests for phonopy.interface.calculator."""

import pathlib

import pytest

from phonopy import Phonopy
from phonopy.interface.calculator import write_supercells_with_displacements


def test_write_supercells_without_structure_info(
    ph_nacl: Phonopy, tmp_path: pathlib.Path
):
    """optional_structure_info may be omitted (e.g. for the VASP default).

    API users building PhonopyAtoms programmatically have no
    StructureInfo from read_crystal_structure; interfaces that do not
    need it must work without it.

    """
    cells = ph_nacl.supercells_with_displacements
    assert cells is not None
    pre = str(tmp_path / "POSCAR")
    write_supercells_with_displacements(
        "vasp",
        ph_nacl.supercell,
        cells,
        additional_info={"pre_filename": pre},
    )
    assert (tmp_path / "SPOSCAR").exists()
    assert (tmp_path / "POSCAR-001").exists()


def test_write_supercells_wien2k_requires_structure_info(ph_nacl: Phonopy):
    """Interfaces that need structure information raise a clear error."""
    cells = ph_nacl.supercells_with_displacements
    assert cells is not None
    with pytest.raises(ValueError, match="wien2k requires structure information"):
        write_supercells_with_displacements("wien2k", ph_nacl.supercell, cells)
