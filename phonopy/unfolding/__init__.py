# Copyright (C) 2015 Atsushi Togo
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

import numpy as np
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import Atoms
from phonopy.structure.cells import get_supercell

class Unfolding:
    def __init__(self,
                 phonon,
                 supercell_matrix):
        self._phonon = phonon
        self._supercell_matrix = supercell_matrix
        self._comm_points = get_commensurate_points(supercell_matrix)
        self._translations = None
        self._set_translations()

    def operator_P(self):
        pass

    def get_translations(self):
        return self._translations

    def get_commensurate_points(self):
        return self._comm_points

    def _set_translations(self):
        pcell = Atoms(numbers=[1],
                      scaled_positions=[[0, 0, 0]],
                      cell=np.diag([1, 1, 1]),
                      pbc=True)
        smat = self._supercell_matrix
        translations = get_supercell(pcell, smat).get_scaled_positions()
        translations -= np.floor(translations)
        self._translations = translations

