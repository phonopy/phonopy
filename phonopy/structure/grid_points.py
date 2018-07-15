# Copyright (C) 2014 Atsushi Togo
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
from phonopy.structure.spglib import (get_stabilized_reciprocal_mesh,
                                      relocate_BZ_grid_address)
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone
from phonopy.structure.symmetry import get_lattice_vector_equivalence


def get_qpoints(mesh_numbers,
                reciprocal_lattice,  # column vectors
                q_mesh_shift=None,  # Monkhorst-Pack style grid shift
                is_gamma_center=True,
                is_time_reversal=True,
                fit_in_BZ=True,
                rotations=None,  # Point group operations in real space
                is_mesh_symmetry=True):
    gp = GridPoints(mesh_numbers,
                    reciprocal_lattice,
                    q_mesh_shift=q_mesh_shift,
                    is_gamma_center=is_gamma_center,
                    is_time_reversal=is_time_reversal,
                    fit_in_BZ=fit_in_BZ,
                    rotations=rotations,
                    is_mesh_symmetry=is_mesh_symmetry)

    return gp.qpoints, gp.weights


def extract_ir_grid_points(grid_mapping_table):
    ir_grid_points = np.array(np.unique(grid_mapping_table), dtype='intc')
    weights = np.zeros_like(grid_mapping_table)
    for i, gp in enumerate(grid_mapping_table):
        weights[gp] += 1
    ir_weights = np.array(weights[ir_grid_points], dtype='intc')

    return ir_grid_points, ir_weights


class GridPoints(object):
    """Class to generate irreducible grid points on uniform mesh grids

    Attributes
    ----------
    mesh_numbers: ndarray
       Mesh numbers along a, b, c axes.
       dtype='intc'
       shape=(3,)
    qpoints: ndarray
       q-points in reduced coordinates of reciprocal lattice
       dtype='double'
       shape=(ir-grid points, 3)
    weights: ndarray
       Geometric q-point weights. Its sum is the number of grid points.
       dtype='intc'
       shape=(ir-grid points,)
    grid_address: ndarray
       Addresses of all grid points represented by integers.
       dtype='intc'
       shape=(prod(mesh_numbers), 3)
    ir_grid_points: ndarray
        Indices of irreducibple grid points in grid_address.
        dtype='intc'
        shape=(ir-grid points,)
    grid_mapping_table: ndarray
        Index mapping table from all grid points to ir-grid points.
        dtype='intc'
        shape=(prod(mesh_numbers),)

    """

    def __init__(self,
                 mesh_numbers,
                 reciprocal_lattice,
                 q_mesh_shift=None,  # Monkhorst-Pack style grid shift
                 is_gamma_center=True,
                 is_time_reversal=True,
                 fit_in_BZ=True,
                 rotations=None,  # Point group operations in real space
                 is_mesh_symmetry=True):  # Except for time reversal symmetry
        """

        Note
        ----
        Uniform mesh grids are made according to Monkhorst-Pack scheme, i.e.,
        for odd (even) numbers, centre are (are not) sampled. The Gamma-centre
        sampling is supported by ``is_gamma_center=True``.

        Parameters
        ----------
        mesh_numbers: array_like
            Mesh numbers along a, b, c axes.
            dtype='intc'
            shape=(3, )
        reciprocal_lattice: array_like
            Basis vectors in reciprocal space. a*, b*, c* are given in column
            vectors.
            dtype='double'
            shape=(3, 3)
        q_mesh_shift: array_like, optional, default None (no shift)
            Mesh shifts along a*, b*, c* axes with respect to neighboring grid
            points from the original mesh (Monkhorst-Pack or Gamma center).
            0.5 gives half grid shift. Normally 0 or 0.5 is given.
            Otherwise q-points symmetry search is not performed.
            dtype='double'
            shape=(3, )
        is_gamma_center: bool, default False
            Uniform mesh grids are generated centring at Gamma point but not
            the Monkhorst-Pack scheme.
        is_time_reversal: bool, optional, default True
            Time reversal symmetry is considered in symmetry search. By this,
            inversion symmetry is always included.
        fit_in_BZ: bool, optional, default True
        rotations: array_like, default None (only unitary operation)
            Rotation matrices in real space.
            dtype='intc'
            shape=(rotations, 3, 3)
        is_mesh_symmetry: bool, optional, default True
            Wheather symmetry search is done or not.

        """

        self._mesh = np.array(mesh_numbers, dtype='intc')
        self._rec_lat = reciprocal_lattice
        self._is_shift = self._shift2boolean(q_mesh_shift,
                                             is_gamma_center=is_gamma_center)
        self._is_time_reversal = is_time_reversal
        self._fit_in_BZ = fit_in_BZ
        self._rotations = rotations
        self._is_mesh_symmetry = is_mesh_symmetry

        self._ir_qpoints = None
        self._grid_address = None
        self._ir_grid_points = None
        self._ir_weights = None
        self._grid_mapping_table = None

        if self._is_shift is None:
            self._is_mesh_symmetry = False
            self._is_shift = self._shift2boolean(None)
            self._set_grid_points()
            self._ir_qpoints += q_mesh_shift / self._mesh
            self._fit_qpoints_in_BZ()
        else:
            self._set_grid_points()

    @property
    def grid_address(self):
        return self._grid_address

    def get_grid_address(self):
        return self.grid_address

    @property
    def ir_grid_points(self):
        return self._ir_grid_points

    def get_ir_grid_points(self):
        return self.ir_grid_points

    @property
    def qpoints(self):
        return self._ir_qpoints

    def get_ir_qpoints(self):
        return self.qpoints

    @property
    def weights(self):
        return self._ir_weights

    def get_ir_grid_weights(self):
        return self.weights

    @property
    def grid_mapping_table(self):
        return self._grid_mapping_table

    def get_grid_mapping_table(self):
        return self.grid_mapping_table

    def _set_grid_points(self):
        if self._is_mesh_symmetry and self._has_mesh_symmetry():
            self._set_ir_qpoints(self._rotations,
                                 is_time_reversal=self._is_time_reversal)
        else:
            self._set_ir_qpoints([np.eye(3, dtype='intc')],
                                 is_time_reversal=self._is_time_reversal)

    def _shift2boolean(self,
                       q_mesh_shift,
                       is_gamma_center=False,
                       tolerance=1e-5):
        """
        Tolerance is used to judge zero/half gird shift.
        This value is not necessary to be changed usually.
        """
        if q_mesh_shift is None:
            shift = np.zeros(3, dtype='double')
        else:
            shift = np.array(q_mesh_shift, dtype='double')

        diffby2 = np.abs(shift * 2 - np.rint(shift * 2))

        if (diffby2 < 0.01).all():  # zero or half shift
            diff = np.abs(shift - np.rint(shift))
            if is_gamma_center:
                is_shift = list(diff > 0.1)
            else:  # Monkhorst-pack
                is_shift = list(np.logical_xor((diff > 0.1),
                                               (self._mesh % 2 == 0)) * 1)
        else:
            is_shift = None

        return is_shift

    def _has_mesh_symmetry(self):
        if self._rotations is None:
            return False
        m = self._mesh
        mesh_equiv = [m[1] == m[2], m[2] == m[0], m[0] == m[1]]
        lattice_equiv = get_lattice_vector_equivalence(
            [r.T for r in self._rotations])
        return np.extract(lattice_equiv, mesh_equiv).all()

    def _fit_qpoints_in_BZ(self):
        qpoint_set_in_BZ = get_qpoints_in_Brillouin_zone(self._rec_lat,
                                                         self._ir_qpoints)
        qpoints_in_BZ = np.array([q_set[0] for q_set in qpoint_set_in_BZ],
                                 dtype='double', order='C')
        self._ir_qpoints = qpoints_in_BZ

    def _set_ir_qpoints(self,
                        rotations,
                        is_time_reversal=True):
        grid_mapping_table, grid_address = get_stabilized_reciprocal_mesh(
            self._mesh,
            rotations,
            is_shift=self._is_shift,
            is_time_reversal=is_time_reversal)

        shift = np.array(self._is_shift, dtype='intc') * 0.5

        if self._fit_in_BZ:
            self._grid_address = relocate_BZ_grid_address(
                grid_address,
                self._mesh,
                self._rec_lat,
                is_shift=self._is_shift)[0][:np.prod(self._mesh)]
        else:
            self._grid_address = grid_address

        (self._ir_grid_points,
         self._ir_weights) = extract_ir_grid_points(grid_mapping_table)

        self._ir_qpoints = np.array(
            (self._grid_address[self._ir_grid_points] + shift) / self._mesh,
            dtype='double', order='C')
        self._grid_mapping_table = grid_mapping_table
