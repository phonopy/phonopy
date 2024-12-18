"""Routines to analyze MD velocity data."""

# Copyright (C) 2016 Atsushi Togo
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

import sys

import numpy as np

from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.grid_points import get_qpoints
from phonopy.units import AMU, kb_J


class Velocity:
    """Class to calculate atomic velocities from temporal positions."""

    def __init__(
        self,
        lattice=None,  # column vectors, in angstrom
        positions=None,  # fractional coordinates
        timestep=None,  # in femtosecond
    ):
        """Init method."""
        self._lattice = lattice
        self._positions = positions
        self._timestep = timestep
        self._velocities = None  # in m/s [timestep, atom, 3]

    def run(self, skip_steps=0):
        """Calculate velocities."""
        pos = self._positions
        diff = pos[(skip_steps + 1) :] - pos[skip_steps:-1]
        diff = np.where(diff > 0.5, diff - 1, diff)
        diff = np.where(diff < -0.5, diff + 1, diff)
        self._velocities = np.dot(diff, self._lattice.T * 1e5) / self._timestep

    def get_velocities(self):
        """Return velocities."""
        return self._velocities

    def get_timestep(self):
        """Return time step."""
        return self._timestep


class VelocityQpoints:
    """Class to calculate q-point projected velocity."""

    def __init__(
        self,
        supercell,
        primitive,
        velocities,  # in m/s either real or reciprocal
        symmetry=None,
    ):
        """Init method."""
        if symmetry is not None:
            self._polonggroup_opts = symmetry.get_pointgroup_operations()
        else:
            self._polonggroup_opts = None

        self._supercell = supercell
        self._primitive = primitive
        self._velocities = velocities

        (self._shortest_vectors, self._multiplicity) = primitive.get_smallest_vectors()

        self._qpoints = None
        self._weights = None

        self._velocities_q = None  # [timestep, p_atom, qpoitns, 3]

    def run(self):
        """Calculate q-point projected velocities."""
        self._velocities_q = self._transform(self._qpoints)

    def get_velocities(self):
        """Return q-point projected velocities."""
        return self._velocities_q

    def set_mesh(self, mesh):
        """Set mesh."""
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._qpoints, self._weights = get_qpoints(
            mesh, rec_lat, is_gamma_center=True, rotations=self._polonggroup_opts
        )

    def set_qpoints(self, qpoints):
        """Set q-points."""
        self._weights = np.ones(len(qpoints), dtype="long")
        self._qpoints = qpoints

    def set_commensurate_points(self):
        """Set commensurate points."""
        supercell_matrix = np.rint(
            np.linalg.inv(self._primitive.primitive_matrix)
        ).astype("intc")
        self.set_qpoints(get_commensurate_points(supercell_matrix))

    def get_qpoints(self):
        """Return irreducible q-points and weights."""
        return self._qpoints, self._weights

    def _transform(self, q):
        """Calculate projection.

        exp(i q.r(i)) v(i)

        """
        s2p = self._primitive.s2p_map
        p2s = self._primitive.p2s_map

        num_p = len(self._primitive)
        v = self._velocities

        q_array = np.reshape(q, (-1, 3))
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        v_q = np.zeros((v.shape[0], num_p, len(q_array), 3), dtype=dtype)

        for p_i, s_i in enumerate(p2s):
            for s_j, s2p_j in enumerate(s2p):
                if s2p_j == s_i:
                    for q_i, pf in enumerate(self._get_phase_factor(p_i, s_j, q_array)):
                        v_q[:, p_i, q_i, :] += pf * v[:, s_j, :]
        return v_q

    def _get_phase_factor(self, p_i, s_j, q_array):
        multi = self._multiplicity[s_j, p_i]
        pos = self._shortest_vectors[s_j, p_i, :multi]
        return np.exp(-2j * np.pi * np.dot(q_array, pos.T)).sum(axis=1) / multi


class AutoCorrelation:
    """Class to calculate autocorrelation."""

    def __init__(self, velocities, masses=None, temperature=None):  # in m/s, AMU, K
        """Init method."""
        self._velocities = velocities
        self._masses = masses
        self._temperature = temperature

        self._vv = None
        self._n_elements = 0

    def run(self, num_frequency_points, verbose=False):
        """Calculate autocorrelation."""
        v = self._velocities
        max_lag = num_frequency_points * 2
        n_elem = len(v) - max_lag

        if n_elem < 1:
            return False

        vv = np.zeros((max_lag,) + v.shape[1:], dtype=v.dtype, order="C")

        if np.iscomplexobj(vv):
            v_c = v.conj()

        # Here is the bottle neck.
        d = max_lag / 2
        for i in range(max_lag):
            if verbose:
                if (i + 1) % (max_lag // 100) == 0:
                    sys.stdout.write("\r%d%%" % (((i + 1) * 100) // max_lag))
                    sys.stdout.flush()
            if np.iscomplexobj(vv):
                vv[i - d] = (v[d : (d + n_elem)] * v_c[i : (i + n_elem)]).sum(axis=0)
            else:
                vv[i - d] = (v[d : (d + n_elem)] * v[i : (i + n_elem)]).sum(axis=0)
        if verbose:
            sys.stdout.write("\r    \n")
            sys.stdout.flush()

        self._vv = vv
        if self._masses is not None and self._temperature is not None:
            for i, m in enumerate(self._masses):
                self._vv[:, i] *= m * AMU / (kb_J * self._temperature)

        self._n_elements = n_elem

        return True

    def get_autocorrelation(self):
        """Return autocorrelation."""
        return self._vv

    def get_number_of_elements(self):
        """Return number of elements of autocorrelation array."""
        return self._n_elements
