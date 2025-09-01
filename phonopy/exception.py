"""Phonopy exceptions."""

# Copyright (C) 2022 Atsushi Togo
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


class ForcesetsNotFoundError(RuntimeError):
    """Exception when forces not found in Phonopy class instance."""

    pass


class ForceCalculatorRequiredError(RuntimeError):
    """Exception when force calculator is required to compute force constants."""

    pass


class ForceConstantsCalculatorNotFoundError(RuntimeError):
    """Exception when force constants calculator is not found."""

    pass


class CellNotFoundError(RuntimeError):
    """Exception when unit cell not found."""

    pass


class MagmomValueError(ValueError):
    """Exception when magnetic moment value is not valid."""

    pass


class PypolymlpFileNotFoundError(RuntimeError):
    """Exception when pypolymlp file is not found."""

    pass


class PypolymlpTrainingDatasetNotFoundError(RuntimeError):
    """Exception when pypolymlp dataset is not found."""

    pass


class PypolymlpRelaxationError(RuntimeError):
    """Exception when relaxation of atomic positions by pypolymlp fails."""

    pass


class PypolymlpDevelopmentError(RuntimeError):
    """Exception when development of pypolymlp fails."""

    pass
