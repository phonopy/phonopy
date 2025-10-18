"""Test Conversion between calculator formats."""

import os
import pathlib
import sys
import tempfile

import pytest

from phonopy.interface.calculator import calculator_info, convert_crystal_structure
from phonopy.scripts.phonopy_calc_convert import run

cwd = pathlib.Path(__file__).parent


def test_conversion():
    """Calcs that can use extra info are below."""
    poscar_file = cwd / "../POSCAR_NaCl"
    calcs = calculator_info.keys()
    require_extra_info = ["wien2k", "cp2k"]
    expected_warnings = ["qe", "abacus", "crystal", "fleur", "wien2k"]
    for calc in calcs:
        with tempfile.TemporaryDirectory() as td:
            if calc == "turbomole":
                convert_crystal_structure(poscar_file, "vasp", td, calc)
            else:
                name = str(pathlib.Path(td) / f"crystal_structure_{calc}")
                if calc in require_extra_info:
                    with pytest.raises(RuntimeError):
                        # These calcs need additional info to write their input files
                        convert_crystal_structure(poscar_file, "vasp", name, calc)
                else:
                    if calc in expected_warnings:
                        with pytest.warns(UserWarning):
                            convert_crystal_structure(poscar_file, "vasp", name, calc)
                    else:
                        convert_crystal_structure(poscar_file, "vasp", name, calc)


def simulate_calc_convert_script(
    input_file, output_file, calc_in, calc_out, addinfo=None
):
    """Simulate running the phonopy_calc_convert script with given arguments."""
    # Convert PosixPath objects to strings
    input_file = str(input_file)
    output_file = str(output_file)

    # Build the argument list
    args = [
        "phonopy_calc_convert.py",  # Script name
        "-i",
        input_file,  # Input file
        "-o",
        output_file,  # Output file
        "--calcin",
        calc_in,  # Input calculator
        "--calcout",
        calc_out,  # Output calculator
    ]

    # Add additional info if provided
    if addinfo:
        args.extend(["--additional-info"] + [str(info) for info in addinfo])

    # Backup the original sys.argv
    original_argv = sys.argv
    try:
        # Replace sys.argv with the simulated arguments
        sys.argv = args
        # Call the script's run function
        run()
    finally:
        # Restore the original sys.argv
        sys.argv = original_argv


def test_calc_convert():
    """
    Test parsing of additional info kwarg.

    Converts to/from vasp for calculators that accept
    additional info into phonopy-calc-convert script.
    """
    # Simulate command-line arguments
    xtra_inps = {
        "qe": [
            "Na",
            "Na.pbe-spn-kjpaw_psl.0.2.UPF",
            "Cl",
            "Cl.pbe-n-kjpaw_psl.0.1.UPF",
        ],
        "wien2k": ["781 781 781", "1e-05 5e-05 5e-05", "2.5 2.28 2.28"],
        "elk": ["alternate_si_file.in"],
        "crystal": [14, 14],
        "fleur": [13.0, 13.0, 13.1, "Title \n other stuff"],
        "abacus": ["Al", "Al.PD04.PBE.UPF"],
    }
    xtra_files = {
        "qe": cwd / "NaCl-pwscf.in",
        "wien2k": cwd / "BaGa2.struct",
        "elk": cwd / "elk.in",
        "crystal": cwd / "Si-CRYSTAL.o",
        "fleur": cwd / "fleur_inpgen",
        "abacus": cwd / "STRU.in",
    }

    for calc in xtra_inps.keys():
        with tempfile.TemporaryDirectory() as temp_dir:
            vasp_tmp = os.path.join(temp_dir, "vasp_tmp_file")
            simulate_calc_convert_script(xtra_files[calc], vasp_tmp, calc, "vasp")
            with tempfile.TemporaryDirectory() as temp2:
                name = os.path.join(temp2, calc + "_tmp_file")
                simulate_calc_convert_script(
                    vasp_tmp, name, "vasp", calc, addinfo=xtra_inps[calc]
                )
