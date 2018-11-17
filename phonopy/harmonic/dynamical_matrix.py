# Copyright (C) 2011 Atsushi Togo
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
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
import numpy as np


def get_dynamical_matrix(fc2,
                         supercell,
                         primitive,
                         nac_params=None,
                         frequency_scale_factor=None,
                         decimals=None,
                         symprec=1e-5,
                         log_level=0):
    if frequency_scale_factor is None:
        _fc2 = fc2
    else:
        _fc2 = fc2 * frequency_scale_factor ** 2

    if nac_params is None:
        dm = DynamicalMatrix(
            supercell,
            primitive,
            _fc2,
            decimals=decimals,
            symprec=symprec)
    else:
        dm = DynamicalMatrixNAC(
            supercell,
            primitive,
            _fc2,
            decimals=decimals,
            symprec=symprec,
            log_level=log_level)
        dm.set_nac_params(nac_params)
    return dm


class DynamicalMatrix(object):
    """Dynamical matrix class

    When prmitive and supercell lattices are L_p and L_s, respectively,
    frame F is defined by
    L_p = dot(F, L_s), then L_s = dot(F^-1, L_p).
    where lattice matrix is defined by axies a,b,c in Cartesian:
        [ a1 a2 a3 ]
    L = [ b1 b2 b3 ]
        [ c1 c2 c3 ]

    Phase difference in primitive cell unit
    between atoms 1 and 2 in supercell is calculated by, e.g.,
    1j * dot((x_s(2) - x_s(1)), F^-1) * 2pi
    where x_s is reduced atomic coordinate in supercell unit.

    Attributes
    ----------
    primitive: Primitive
        Primitive cell instance. Note that Primitive is inherited from
        PhonopyAtoms.
    supercell: Supercell
        Supercell instance. Note that Supercell is inherited from PhonopyAtoms.
    force_constants: ndarray
        Supercell force constants. Full and compact shapes of arrays are
        supported.
        dtype='double'
        shape=(supercell atoms, supercell atoms, 3, 3) for full array
        shape=(primitive atoms, supercell atoms, 3, 3) for compact array
    dynatmical_matrix: ndarray
        Dynamical matrix at specified q.
        dtype='complex128'
        shape=(primitive atoms * 3, primitive atoms * 3)

    """

    def __init__(self,
                 supercell,
                 primitive,
                 force_constants,
                 decimals=None,
                 symprec=1e-5):
        self._scell = supercell
        self._pcell = primitive
        self._decimals = decimals
        self._symprec = symprec
        self._dynamical_matrix = None
        self._force_constants = None
        self._set_force_constants(force_constants)

        itemsize = self._force_constants.itemsize
        self._dtype_complex = ("c%d" % (itemsize * 2))

        self._p2s_map = primitive.get_primitive_to_supercell_map()
        self._s2p_map = primitive.get_supercell_to_primitive_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        self._s2pp_map = np.array(
            [p2p_map[self._s2p_map[i]] for i in range(len(self._s2p_map))],
            dtype='intc')
        (self._smallest_vectors,
         self._multiplicity) = primitive.get_smallest_vectors()
        # Non analytical term correction
        self._nac = False

    def is_nac(self):
        return self._nac

    def get_dimension(self):
        return self._pcell.get_number_of_atoms() * 3

    def get_decimals(self):
        return self._decimals

    @property
    def supercell(self):
        return self._scell

    def get_supercell(self):
        return self.supercell

    @property
    def primitive(self):
        return self._pcell

    def get_primitive(self):
        return self.primitive

    @property
    def force_constants(self):
        return self._force_constants

    def get_force_constants(self):
        return self.force_constants

    def get_shortest_vectors(self):
        return self._smallest_vectors, self._multiplicity

    @property
    def dynamical_matrix(self):
        dm = self._dynamical_matrix

        if self._dynamical_matrix is None:
            return None

        if self._decimals is None:
            return dm
        else:
            return dm.round(decimals=self._decimals)

    def get_dynamical_matrix(self):
        return self.dynamical_matrix

    def set_dynamical_matrix(self, q):
        self._set_dynamical_matrix(q)

    def _set_dynamical_matrix(self, q):
        try:
            import phonopy._phonopy as phonoc
            self._set_c_dynamical_matrix(q)
        except ImportError:
            self._set_py_dynamical_matrix(q)

    def _set_force_constants(self, fc):
        if (type(fc) is np.ndarray and
            fc.dtype is np.dtype('double') and
            fc.flags.aligned and
            fc.flags.owndata and
            fc.flags.c_contiguous):
            self._force_constants = fc
        else:
            self._force_constants = np.array(fc, dtype='double', order='C')

    def _set_c_dynamical_matrix(self, q):
        import phonopy._phonopy as phonoc

        fc = self._force_constants
        vectors = self._smallest_vectors
        mass = self._pcell.get_masses()
        multiplicity = self._multiplicity
        size_prim = len(mass)
        itemsize = self._force_constants.itemsize
        dm = np.zeros((size_prim * 3, size_prim * 3),
                      dtype=("c%d" % (itemsize * 2)))

        if fc.shape[0] == fc.shape[1]:  # full FC
            phonoc.dynamical_matrix(dm.view(dtype='double'),
                                    fc,
                                    np.array(q, dtype='double'),
                                    vectors,
                                    multiplicity,
                                    mass,
                                    self._s2p_map,
                                    self._p2s_map)
        else:
            phonoc.dynamical_matrix(
                dm.view(dtype='double'),
                fc,
                np.array(q, dtype='double'),
                vectors,
                multiplicity,
                mass,
                self._s2pp_map,
                np.arange(len(self._p2s_map), dtype='intc'))

        # Data of dm array are stored in memory by the C order of
        # (size_prim * 3, size_prim * 3, 2), where the last 2 means
        # real and imaginary parts. This code assumes this memory
        # order is that expected by numpy. Otherwise, numpy complex array
        # should be created as follows:
        #   dm_double = dm.view(dtype='double').reshape(size_prim * 3,
        #                                               size_prim * 3, 2)
        #   dm = dm_double[:, :, 0] + 1j * dm_double[:, :, 1]
        self._dynamical_matrix = dm

    def _set_py_dynamical_matrix(self, q):
        fc = self._force_constants
        vecs = self._smallest_vectors
        multiplicity = self._multiplicity
        num_atom = len(self._p2s_map)
        dm = np.zeros((3 * num_atom, 3 * num_atom), dtype=self._dtype_complex)
        mass = self._pcell.get_masses()

        for i, s_i in enumerate(self._p2s_map):
            for j, s_j in enumerate(self._p2s_map):
                sqrt_mm = np.sqrt(mass[i] * mass[j])
                dm_local = np.zeros((3, 3), dtype=self._dtype_complex)
                # Sum in lattice points
                for k in range(self._scell.get_number_of_atoms()):
                    if s_j == self._s2p_map[k]:
                        multi = multiplicity[k][i]
                        phase = []
                        for l in range(multi):
                            vec = vecs[k][i][l]
                            phase.append(np.vdot(vec, q) * 2j * np.pi)
                        phase_factor = np.exp(phase).sum()
                        dm_local += fc[s_i, k] * phase_factor / sqrt_mm / multi

                dm[(i*3):(i*3+3), (j*3):(j*3+3)] += dm_local

        # Impose Hermisian condition
        self._dynamical_matrix = (dm + dm.conj().transpose()) / 2


# Non analytical term correction (NAC)
# Call this when NAC is required instead of DynamicalMatrix
class DynamicalMatrixNAC(DynamicalMatrix):
    def __init__(self,
                 supercell,
                 primitive,
                 force_constants,
                 nac_params=None,
                 num_G_points=None,  # For Gonze NAC
                 decimals=None,
                 symprec=1e-5,
                 log_level=0):

        DynamicalMatrix.__init__(self,
                                 supercell,
                                 primitive,
                                 force_constants,
                                 decimals=decimals,
                                 symprec=1e-5)

        self._log_level = log_level
        self._method = None

        # For the method by Gonze et al.
        self._Gonze_force_constants = None
        if num_G_points is None:
            self._num_G_points = 300
        else:
            self._num_G_points = num_G_points
        self._G_list = None
        self._G_cutoff = None
        self._Lambda = None  # 4*Lambda**2 is stored.
        self._dd_q0 = None

        self._nac = True
        if nac_params is not None:
            self.set_nac_params(nac_params)

    def get_born_effective_charges(self):
        return self._born

    def get_nac_factor(self):
        return self._unit_conversion * 4.0 * np.pi / self._pcell.get_volume()

    def get_dielectric_constant(self):
        return self._dielectric

    def get_nac_method(self):
        return self._method

    def get_Gonze_nac_dataset(self):
        if self._method == 'gonze':
            return (self._Gonze_force_constants,
                    self._dd_q0,
                    self._G_cutoff,
                    self._G_list,
                    self._Lambda)
        else:
            return None

    def set_nac_params(self, nac_params):
        self._born = np.array(nac_params['born'], dtype='double', order='C')
        self._unit_conversion = nac_params['factor']
        self._dielectric = np.array(nac_params['dielectric'],
                                    dtype='double', order='C')
        if 'method' not in nac_params:
            self._method = 'gonze'
        elif nac_params['method'] == 'gonze':
            self._method = 'gonze'
        elif nac_params['method'] == 'wang':
            self._method = 'wang'
        else:
            self._method = 'gonze'

        if self._method == 'gonze':
            if 'G_cutoff' in nac_params:
                self._G_cutoff = nac_params['G_cutoff']
            else:
                self._G_cutoff = (3 * self._num_G_points / (4 * np.pi) /
                                  self._pcell.get_volume()) ** (1.0 / 3)
            if 'Lambda' in nac_params:
                self._Lambda = nac_params['Lambda']
            else:
                exp_cutoff = 1e-10
                GeG = self._G_cutoff ** 2 * np.trace(self._dielectric) / 3
                self._Lambda = np.sqrt(- GeG / 4 / np.log(exp_cutoff))

    def make_Gonze_nac_dataset(self, verbose=False):
        self._G_list = self._get_G_list(self._G_cutoff)
        if verbose:
            print("NAC by Gonze et al., PRB 50, 13035(R) (1994), "
                  "PRB 55, 10355 (1997):")
            print("  G-cutoff distance: %5.2f" % self._G_cutoff)
            print("  Number of G-points: %d" % len(self._G_list))
            print("  Lambda: %6.2f" % self._Lambda)

        try:
            import phonopy._phonopy as phonoc
            self._set_c_dipole_dipole_q0()
        except ImportError:
            print("Python version of dipole-dipole calculation is not well "
                  "implemented.")
            sys.exit(1)

        self._set_Gonze_force_constants()
        self._Gonze_count = 0

    def set_dynamical_matrix(self, q_red, q_direction=None):
        rec_lat = np.linalg.inv(self._pcell.get_cell())  # column vectors
        if q_direction is None:
            q_norm = np.linalg.norm(np.dot(q_red, rec_lat.T))
        else:
            q_norm = np.linalg.norm(np.dot(q_direction, rec_lat.T))

        if q_norm < self._symprec:
            self._set_dynamical_matrix(q_red)
            return False

        if self._method == 'wang':
            self._set_Wang_dynamical_matrix(q_red, q_direction)
        else:
            if self._Gonze_force_constants is None:
                self.make_Gonze_nac_dataset(self._log_level)
            self._set_Gonze_dynamical_matrix(q_red, q_direction)

    def _set_Wang_dynamical_matrix(self, q_red, q_direction):
        # Wang method (J. Phys.: Condens. Matter 22 (2010) 202201)
        rec_lat = np.linalg.inv(self._pcell.get_cell())  # column vectors
        if q_direction is None:
            q = np.dot(q_red, rec_lat.T)
        else:
            q = np.dot(q_direction, rec_lat.T)

        constant = self._get_constant_factor(q,
                                             self._dielectric,
                                             self._pcell.get_volume(),
                                             self._unit_conversion)
        try:
            import phonopy._phonopy as phonoc
            self._set_c_Wang_dynamical_matrix(q_red, q, constant)
        except ImportError:
            num_atom = self._pcell.get_number_of_atoms()
            fc_backup = self._force_constants.copy()
            nac_q = self._get_charge_sum(num_atom, q, self._born) * constant
            self._set_py_Wang_force_constants(self._force_constants, nac_q)
            self._set_dynamical_matrix(q_red)
            self._force_constants = fc_backup

    def _set_c_Wang_dynamical_matrix(self, q_red, q, factor):
        import phonopy._phonopy as phonoc

        fc = self._force_constants
        vectors = self._smallest_vectors
        mass = self._pcell.get_masses()
        multiplicity = self._multiplicity
        size_prim = len(mass)
        itemsize = fc.itemsize
        dm = np.zeros((size_prim * 3, size_prim * 3),
                      dtype=("c%d" % (itemsize * 2)))

        if fc.shape[0] == fc.shape[1]:  # full fc
            phonoc.nac_dynamical_matrix(dm.view(dtype='double'),
                                        fc,
                                        np.array(q_red, dtype='double'),
                                        vectors,
                                        multiplicity,
                                        mass,
                                        self._s2p_map,
                                        self._p2s_map,
                                        np.array(q, dtype='double'),
                                        self._born,
                                        factor)
        else:
            phonoc.nac_dynamical_matrix(dm.view(dtype='double'),
                                        fc,
                                        np.array(q_red, dtype='double'),
                                        vectors,
                                        multiplicity,
                                        mass,
                                        self._s2pp_map,
                                        np.arange(len(self._p2s_map),
                                                  dtype='intc'),
                                        np.array(q, dtype='double'),
                                        self._born,
                                        factor)

        self._dynamical_matrix = dm

    def _set_py_Wang_force_constants(self, fc, nac_q):
        N = (self._scell.get_number_of_atoms() //
             self._pcell.get_number_of_atoms())
        for s1 in range(self._scell.get_number_of_atoms()):
            # This if-statement is the trick.
            # In contructing dynamical matrix in phonopy
            # fc of left indices with s1 == self._s2p_map[ s1 ] are
            # only used.
            if s1 != self._s2p_map[s1]:
                continue
            p1 = self._s2pp_map[s1]
            for s2 in range(self._scell.get_number_of_atoms()):
                p2 = self._s2pp_map[s2]
                fc[s1, s2] += nac_q[p1, p2] / N

    def _set_Gonze_dynamical_matrix(self, q_red, q_direction):
        if self._log_level > 2:
            print("%d %s" % (self._Gonze_count + 1, q_red))
        self._Gonze_count += 1
        fc = self._force_constants
        self._force_constants = self._Gonze_force_constants
        self._set_dynamical_matrix(q_red)
        self._force_constants = fc
        dm_dd = self._get_Gonze_dipole_dipole(q_red, q_direction)
        self._dynamical_matrix += dm_dd

    def _set_Gonze_force_constants(self):
        fc_shape = self._force_constants.shape
        d2f = DynmatToForceConstants(self._pcell,
                                     self._scell,
                                     is_full_fc=(fc_shape[0] == fc_shape[1]),
                                     symprec=self._symprec)
        dynmat = []
        num_q = len(d2f.get_commensurate_points())
        for i, q_red in enumerate(d2f.get_commensurate_points()):
            if self._log_level > 2:
                print("%d/%d %s" % (i + 1, num_q, q_red))
            self._set_dynamical_matrix(q_red)
            dm_dd = self._get_Gonze_dipole_dipole(q_red, None)
            self._dynamical_matrix -= dm_dd
            dynmat.append(self._dynamical_matrix)
        d2f.set_dynamical_matrices(dynmat=dynmat)
        d2f.run()
        self._Gonze_force_constants = d2f.get_force_constants()

    def _get_Gonze_dipole_dipole(self, q_red, q_direction):
        rec_lat = np.linalg.inv(self._pcell.get_cell())  # column vectors
        q_cart = np.array(np.dot(q_red, rec_lat.T), dtype='double')
        if q_direction is None:
            q_dir_cart = None
        else:
            q_dir_cart = np.array(np.dot(q_direction, rec_lat.T),
                                  dtype='double')

        try:
            import phonopy._phonopy as phonoc
            C = self._get_c_dipole_dipole(q_cart, q_dir_cart)
        except ImportError:
            print("Python version of dipole-dipole calculation is not well "
                  "implemented.")
            sys.exit(1)

        # D-type to C-type conversion and mass weighted
        # mass = self._pcell.get_masses()
        # num_atom = self._pcell.get_number_of_atoms()
        # pos = self._pcell.get_positions()
        # for i in range(num_atom):
        #     dpos = pos - pos[i]
        #     phase_factor = np.exp(2j * np.pi * np.dot(dpos, q))
        #     for j in range(num_atom):
        #         C[i, :, j, :] *= phase_factor[j] / np.sqrt(mass[i] * mass[j])

        # Mass weighted
        mass = self._pcell.get_masses()
        num_atom = self._pcell.get_number_of_atoms()
        for i in range(num_atom):
            for j in range(num_atom):
                C[i, :, j, :] *= 1.0 / np.sqrt(mass[i] * mass[j])

        C_dd = C.reshape(num_atom * 3, num_atom * 3)

        return C_dd

    def _get_c_dipole_dipole(self, q_cart, q_dir_cart):
        import phonopy._phonopy as phonoc

        pos = self._pcell.get_positions()
        num_atom = len(pos)
        volume = self._pcell.get_volume()
        dd = np.zeros((num_atom, 3, num_atom, 3),
                      dtype=self._dtype_complex, order='C')

        phonoc.dipole_dipole(dd.view(dtype='double'),
                             self._dd_q0.view(dtype='double'),
                             self._G_list,
                             q_cart,
                             q_dir_cart,
                             self._born,
                             self._dielectric,
                             np.array(pos, dtype='double', order='C'),
                             self._unit_conversion * 4.0 * np.pi / volume,
                             self._Lambda,
                             self._symprec)
        return dd

    def _set_c_dipole_dipole_q0(self):
        import phonopy._phonopy as phonoc

        pos = self._pcell.get_positions()
        self._dd_q0 = np.zeros((len(pos), 3, 3),
                               dtype=self._dtype_complex, order='C')

        phonoc.dipole_dipole_q0(self._dd_q0.view(dtype='double'),
                                self._G_list,
                                self._born,
                                self._dielectric,
                                np.array(pos, dtype='double', order='C'),
                                self._Lambda,
                                self._symprec)

        # Limiting contribution
        # inv_eps = np.linalg.inv(self._dielectric)
        # sqrt_det_eps = np.sqrt(np.linalg.det(self._dielectric))
        # coef = (4.0 / 3 / np.sqrt(np.pi) * inv_eps
        #         / sqrt_det_eps * self._Lambda ** 3)
        # self._dd_q0 -= coef

    def _get_py_dipole_dipole(self, K_list, q, q_dir_cart):
        pos = self._pcell.get_positions()
        num_atom = self._pcell.get_number_of_atoms()
        volume = self._pcell.get_volume()
        C = np.zeros((num_atom, 3, num_atom, 3),
                     dtype=self._dtype_complex, order='C')

        for q_K in K_list:
            if np.linalg.norm(q_K) < self._symprec:
                if q_dir_cart is None:
                    continue
                else:
                    dq_K = q_dir_cart
            else:
                dq_K = q_K

            Z_mat = (self._get_charge_sum(num_atom, dq_K, self._born) *
                     self._get_constant_factor(dq_K,
                                               self._dielectric,
                                               volume,
                                               self._unit_conversion))
            for i in range(num_atom):
                dpos = - pos + pos[i]
                phase_factor = np.exp(2j * np.pi * np.dot(dpos, q_K))
                for j in range(num_atom):
                    C[i, :,  j, :] += Z_mat[i, j] * phase_factor[j]

        for q_K in K_list:
            q_G = q_K - q
            if np.linalg.norm(q_G) < self._symprec:
                continue
            Z_mat = (self._get_charge_sum(num_atom, q_G, self._born) *
                     self._get_constant_factor(q_G,
                                               self._dielectric,
                                               volume,
                                               self._unit_conversion))
            for i in range(num_atom):
                C_i = np.zeros((3, 3), dtype=self._dtype_complex, order='C')
                dpos = - pos + pos[i]
                phase_factor = np.exp(2j * np.pi * np.dot(dpos, q_G))
                for j in range(num_atom):
                    C_i += Z_mat[i, j] * phase_factor[j]
                C[i, :,  i, :] -= C_i

        return C

    def _get_G_list(self, G_cutoff, g_rad=100):
        rec_lat = np.linalg.inv(self._pcell.get_cell())  # column vectors
        # g_rad must be greater than 0 for broadcasting.
        n = g_rad * 2 + 1
        G = np.zeros((n ** 3, 3), dtype='double', order='C')
        pts = np.arange(-g_rad, g_rad + 1)
        grid = np.meshgrid(pts, pts, pts)
        for i in range(3):
            G[:, i] = grid[i].ravel()
        grid = None
        del grid
        G_vec_list = np.dot(G, rec_lat.T)
        G = None
        del G
        G_norm2 = ((G_vec_list) ** 2).sum(axis=1)
        return np.array(G_vec_list[G_norm2 < G_cutoff ** 2],
                        dtype='double', order='C')

    def _get_charge_sum(self, num_atom, q, born):
        nac_q = np.zeros((num_atom, num_atom, 3, 3), dtype='double', order='C')
        A = np.dot(q, born)
        for i in range(num_atom):
            for j in range(num_atom):
                nac_q[i, j] = np.outer(A[i], A[j])
        return nac_q

    def _get_constant_factor(self, q, dielectric, volume, unit_conversion):
        return (unit_conversion * 4.0 * np.pi / volume /
                np.dot(q.T, np.dot(dielectric, q)))
