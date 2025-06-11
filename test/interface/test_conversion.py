"""Test Conversion between calculator formats."""

import os
import pathlib
import tempfile

import pytest

from phonopy.interface.calculator import calculator_info, convert_crystal_structure

cwd = pathlib.Path(__file__).parent


def test_conversion():
    """Calcs that can use extra info are below."""
    calcs = calculator_info.keys()
    require_extra_info = ["wien2k", "siesta", "cp2k", "crystal", "fleur", "abacus"]
    poscar_file = cwd / "../POSCAR_NaCl"

    for calc in calcs:
        if calc == "turbomole":
            with tempfile.TemporaryDirectory() as td:
                convert_crystal_structure(poscar_file, "vasp", td, calc)
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                name = temp.name
            if calc in require_extra_info:
                with pytest.raises(RuntimeError):
                    # These calcs need additional info to write their input files
                    convert_crystal_structure(poscar_file, "vasp", name, calc)
            os.unlink(name)
