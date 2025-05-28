"""Test Conversion between calculator formats"""

import tempfile, pytest
from phonopy.interface.calculator import convert_crystal_structure, calculator_info


def test_conversion():
    """calcs that can use extra info are below"""
    calcs = calculator_info.keys()
    require_extra_info = ["wien2k", "siesta", "cp2k", "crystal", "fleur", "abacus"]
    
    temp = tempfile.NamedTemporaryFile()
    td = tempfile.TemporaryDirectory()
   
    POSCAR = "../POSCAR_NaCl"
    
    for calc in calcs:
        if calc == "turbomole":
            name = td.name
        else:
            name = temp.name
        print(f"Testing conversion for {calc}")
        if calc in require_extra_info:
            with pytest.raises(RuntimeError):
                # These calcs need additional info to write their input files
                convert_crystal_structure(POSCAR, "vasp", name, calc)
        else:
            convert_crystal_structure(POSCAR, "vasp", name, calc)