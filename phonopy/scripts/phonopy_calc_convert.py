# Copyright (C) 2020
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os
from argparse import ArgumentParser, ArgumentTypeError

from phonopy.interface.calculator import calculator_info, convert_crystal_structure


def get_options():
    """Parse options."""
    parser = ArgumentParser(
        description="Phonopy crystal structure converter command-line-tool"
    )
    parser.add_argument(
        "-i",
        dest="filename_in",
        metavar="FILE_IN",
        default=None,
        type=_infile_exist,
        help="Input crystal structure filename",
        required=True,
    )
    parser.add_argument(
        "-o",
        dest="filename_out",
        metavar="FILE_OUT",
        type=_outfile_exist,
        default=None,
        help="Output crystal structure filename",
        required=True,
    )
    parser.add_argument(
        "--calcin",
        dest="calculator_in",
        metavar="CALC_IN",
        default=None,
        type=_calc_check,
        help="Input calculator format",
        required=True,
    )
    parser.add_argument(
        "--calcout",
        dest="calculator_out",
        metavar="CALC_OUT",
        type=_calc_check,
        default=None,
        help="Output calculator format",
        required=True,
    )
    parser.add_argument(
        "--additional-info",
        dest="additional_info",
        metavar="ADDITIONAL_INFO",
        nargs="+",
        default=None,
        help="Additional information for the conversion,"
        " which is required for some calculators."
        " Pass 'help' to this option to see the format"
        " for your desired output format.",
    )
    return parser.parse_args()


def parse_qe(opts):
    """Parse additional info used in some calculators.

    qe:
        additional_info is an alternating list of element symbols followed by
        their corresponding pseudopotential filenames
        Symbol1 PP_file1 Symbol2 PP_file2 ...
        where Symbol1 and Symbol2 are atomic symbols and
        PP_file1 and PP_file2 are the corresponding pseudopotential files.

    """
    if len(opts.additional_info) % 2 != 0:
        raise ArgumentTypeError("Equal number of symbols and pp files expected.")
    keys = opts.additional_info[0::2]
    values = opts.additional_info[1::2]
    pp_files = dict(zip(keys, values))
    return (opts.filename_out, pp_files)


def parse_wien2k(opts):
    """Parse additional info used in wien2k.

    wien2k:
        additional_info is a set of stringified lists in the format
        "npt1 npt2 ..." "r01 r02 ..." "rmts1 rmts2 ..."
        str(ints)       str(floats)    str(floats)
    """
    info = opts.additional_info
    if len(info) != 3:
        raise ArgumentTypeError("Three stringified lists expected.")

    npts = [int(x.strip()) for x in info[0].split()]
    r0s = [float(x.strip()) for x in info[1].split()]
    rmts = [float(x.strip()) for x in info[2].split()]

    return (opts.filename_out, npts, r0s, rmts)


def parse_elk(opts):
    """Parse additional info used in elk.

    elk:
        additional_info is a list in the format
        spfname1 spfname2 ...
        where the elements of the list are the speciesfile names.
    NOTE: This information is only necessary if speciesfile names are not
        in the format of "symbol.in", which is otherwise intuited by the writer.
    """
    spfnames = opts.additional_info
    return (opts.filename_out, spfnames)


def parse_cp2k(opts):
    """Not Implemented."""
    if opts.additional_info is not None:
        raise NotImplementedError()


def parse_fleur(opts):
    r"""Parse additional info used in fleur.

    fleur:
        additional_info is a list of atom lables. They are either Z or Z.x,
        where Z is the atomic number and x is any number of decimal places.
        After this list of lables, A single string of additional lines
        (separated by \n) with job information can be given to be added to output file.

        eg. '13.0 13.0 13.1 "Title \n Additional job info here \n"'
    """
    speci = []
    restlines = None
    for arg in opts.additional_info:
        try:
            float(arg)
            speci.append(float(arg))
        except ValueError:
            restlines = arg.split("\\n")
    if len(speci) == 0:
        speci = None
    return (opts.filename_out, speci, restlines)


def parse_crystal(opts):
    """Parse additional info used in crystal.

    crystal:
        additional_info is a list of CRYSTAL conventional atomic numbers
        eg. 'Ge' -> 32 or 'Ge' -> 232
    """
    atomic_numbers = [int(x.strip()) for x in opts.additional_info]
    return (opts.filename_out, atomic_numbers)


def parse_abacus(opts):
    """Parse additional info used in abacus.

    abacus:
        additional_info is a list in the format
        symbol1 PP_file1 symbol2 PP_file2 ...
        where symbol1 and symbol2 are atomic symbols and
        PP_file1 and PP_file2 are the corresponding pseudopotential files.
    NOTE: This is the same format as qe at the moment because the other
    possible information `atom_basis`, `atom_offsite_basis` is not implemented.
    """
    fname, ppfiles = parse_qe(opts)
    return fname, ppfiles, None, None


def parse_additional_info(opts):
    """Decide how to parse additional_info based on the output calculator type."""
    calcs = {
        "qe": parse_qe,
        "wien2k": parse_wien2k,
        "elk": parse_elk,
        "cp2k": parse_cp2k,
        "crystal": parse_crystal,
        "fleur": parse_fleur,
        "abacus": parse_abacus,
    }
    if opts.additional_info is not None:
        info_parser = calcs[opts.calculator_out]
        if opts.additional_info[0].lower() in ["help", "h"]:
            print(info_parser.__doc__)
            quit()
        additional_info = info_parser(opts)
    else:
        additional_info = None
    return additional_info


def run():
    """Run phonopy-calc-convert."""
    opts = get_options()

    additional_info = parse_additional_info(opts)
    args = (
        opts.filename_in,
        opts.calculator_in,
        opts.filename_out,
        opts.calculator_out,
        additional_info,
    )

    convert_crystal_structure(*args)


def _calc_check(calc_str):
    calc = calc_str.lower()
    if calc not in calculator_info:
        msg = 'Calculator name of "%s" is not supported.' % calc_str
        raise RuntimeError(msg)
    return calc


def _infile_exist(filename):
    if not os.path.isfile(filename):
        msg = 'No such file of "%s"' % filename
        raise FileNotFoundError(msg)
    return filename


def _outfile_exist(filename):
    if os.path.isfile(filename):
        msg = '"%s" exists in the current directory. Use different filename.' % filename
        raise RuntimeError(msg)
    return filename
