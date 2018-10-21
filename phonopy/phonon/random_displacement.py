# Copyright (C) 2018 Atsushi Togo
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
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points_in_integers
from phonopy.structure.cells import determinant
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone

class RandomDisplacement(object):
    """Generate Random displacements by Canonical ensenmble.

    """

    def __init__(self,
                 supercell_matrix,
                 reciprocal_primitive_lattice):
        """

        Parameters
        ----------
        supercell_matrix : array_like
            Supercell matrix.
            shape=(3, 3)
            dtype='intc'
        reciprocal_primitive_lattice : array_like
            Primitive basis vectors by column vectors.
            shape=(3, 3)
            dtype='double'


        # self._dynmat = dynamical_matrix
        points = get_commensurate_points_in_integers(supercell_matrix)
        N = determinant(supercell_matrix)
        ii, ij = self._categorize_points(points, N)
        assert len(ii) + len(ij) * 2 == N
        q_ii = points[ii] / float(N)
        print(q_ii)
        print(get_qpoints_in_Brillouin_zone(rec_prim_lat), q_ii))

    def _categorize_points(self, points, N):
        ii = []
        ij = []
        for i, p in enumerate(points):
            for j, _p in enumerate(points):
                if ((p + _p) % N == 0).all():
                    if i == j:
                        ii.append(i)
                    elif i < j:
                        ij.append(i)
                    break
        return ii, ij
