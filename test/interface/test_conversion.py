"""Test Conversion between calculator formats."""

import tempfile, pytest, os
from phonopy.interface.calculator import calculator_info, convert_crystal_structure


def test_conversion():
    """Calcs that can use extra info are below."""
    calcs = calculator_info.keys()
    require_extra_info = ["wien2k", "siesta", "cp2k", "crystal", "fleur", "abacus"]

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(parent_dir)
    temp = tempfile.NamedTemporaryFile()
    td = tempfile.TemporaryDirectory()
    POSCAR = os.path.join(parent_dir, "POSCAR_NaCl")

    for calc in calcs:
        if calc == "turbomole":
            name = td.name
        else:
            name = temp.name
        if calc in require_extra_info:
            with pytest.raises(RuntimeError):
                # These calcs need additional info to write their input files
                convert_crystal_structure(POSCAR, "vasp", name, calc)
        else:
            convert_crystal_structure(POSCAR, "vasp", name, calc)
