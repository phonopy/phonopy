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

from phonopy.interface.calculator import calculator_info, convert_crystal_structure


def get_options():
    """Parse options."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phonopy crystal structure converter command-line-tool"
    )
    parser.add_argument(
        "-i",
        dest="filename_in",
        metavar="FILE_IN",
        default=None,
        help="Input crystal structure filename",
        required=True,
    )
    parser.add_argument(
        "-o",
        dest="filename_out",
        metavar="FILE_OUT",
        default=None,
        help="Output crystal structure filename",
        required=True,
    )
    parser.add_argument(
        "--calcin",
        dest="calculator_in",
        metavar="CALC_IN",
        default=None,
        help="Input calculator format",
        required=True,
    )
    parser.add_argument(
        "--calcout",
        dest="calculator_out",
        metavar="CALC_OUT",
        default=None,
        help="Output calculator format",
        required=True,
    )
    parser.add_argument(
        "--additional-info",
        dest="additional_info",
        metavar="ADDITIONAL_INFO",
        default=None,
        help="Additional information for the conversion,\
            which is required for some calculators.\
                If not provided when required, an error will be raised.",
    )
    return parser.parse_args()


def run():
    """Run phonopy-calc-convert."""
    opts = get_options()
    args = (
        opts.filename_in,
        opts.calculator_in,
        opts.filename_out,
        opts.calculator_out,
        opts.additional_info,
    )

    try:
        _infile_exist(args[0])
        _outfile_exist(args[2])
        _calc_check(args[1])
        _calc_check(args[3])
    except (RuntimeError, FileNotFoundError) as err:
        print("ERROR: %s" % err)

    convert_crystal_structure(*args)


def _calc_check(calc_str):
    if calc_str.lower() not in calculator_info:
        msg = 'Calculator name of "%s" is not supported.' % calc_str
        raise RuntimeError(msg)


def _infile_exist(filename):
    if not os.path.isfile(filename):
        msg = 'No such file of "%s"' % filename
        raise FileNotFoundError(msg)


def _outfile_exist(filename):
    if os.path.isfile(filename):
        msg = '"%s" exists in the current directory. Use different filename.' % filename
        raise RuntimeError(msg)
