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

import numpy as np
from phonopy.structure.cells import get_reduced_bases

def get_dynamical_matrix(fc2,
                         supercell,
                         primitive,
                         nac_params=None,
                         frequency_scale_factor=None,
                         decimals=None,
                         symprec=1e-5):
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
            symprec=symprec)
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
    """

    def __init__(self,
                 supercell,
                 primitive,
                 force_constants,
                 decimals=None,
                 symprec=1e-5):
        self._scell = supercell
        self._pcell = primitive
        self._force_constants = np.array(force_constants,
                                         dtype='double', order='C')
        self._decimals = decimals
        self._symprec = symprec

        self._p2s_map = primitive.get_primitive_to_supercell_map()
        self._s2p_map = primitive.get_supercell_to_primitive_map()
        p2p_map = primitive.get_primitive_to_primitive_map()
        self._p2p_map = [p2p_map[self._s2p_map[i]]
                         for i in range(len(self._s2p_map))]
        self._smallest_vectors, self._multiplicity = \
            get_smallest_vectors(supercell, primitive, symprec)
        self._mass = self._pcell.get_masses()
        # Non analytical term correction
        self._nac = False

    def is_nac(self):
        return self._nac

    def get_dimension(self):
        return self._pcell.get_number_of_atoms() * 3

    def get_decimals(self):
        return self._decimals
    
    def get_supercell(self):
        return self._scell

    def get_primitive(self):
        return self._pcell

    def get_force_constants(self):
        return self._force_constants

    def get_shortest_vectors(self):
        return self._smallest_vectors, self._multiplicity

    def get_primitive_to_supercell_map(self):
        return self._p2s_map

    def get_supercell_to_primitive_map(self):
        return self._s2p_map

    def get_dynamical_matrix(self):
        dm = self._dynamical_matrix

        if self._decimals is None:
            return dm
        else:
            return dm.round(decimals=self._decimals)

    def set_dynamical_matrix(self, q, verbose=False):
        try:
            import phonopy._phonopy as phonoc
            self._set_c_dynamical_matrix(q)
        except ImportError:
            self._set_py_dynamical_matrix(q, verbose=verbose)

    def _set_py_dynamical_matrix(self,
                                 q,
                                 verbose=False):
        fc = self._force_constants
        vecs = self._smallest_vectors
        multiplicity = self._multiplicity
        num_atom = len(self._p2s_map)
        dm = np.zeros((3 * num_atom, 3 * num_atom), dtype=complex)

        for i, s_i in enumerate(self._p2s_map):
            for j, s_j in enumerate(self._p2s_map):
                mass = np.sqrt(self._mass[i] * self._mass[j])
                dm_local = np.zeros((3, 3), dtype=complex)
                # Sum in lattice points                
                for k in range(self._scell.get_number_of_atoms()): 
                    if s_j == self._s2p_map[k]:
                        multi = multiplicity[k][i]
                        phase = []
                        for l in range(multi):
                            vec = vecs[k][i][l]
                            phase.append(np.vdot(vec, q) * 2j * np.pi)
                        phase_factor = np.exp(phase).sum()
                        dm_local += fc[s_i, k] * phase_factor / mass / multi

                dm[(i*3):(i*3+3), (j*3):(j*3+3)] += dm_local

        # Impose Hermisian condition
        self._dynamical_matrix = (dm + dm.conj().transpose()) / 2 

        if verbose:
            self._dynamical_matrix_log()

    def _dynamical_matrix_log(self):
        dm = self._dynamical_matrix
        for i in range(dm.shape[0] // 3):
            for j in range(dm.shape[0] // 3):
                dm_local = dm[(i*3):(i*3+3), (j*3):(j*3+3)]
                for vec in dm_local:
                    re = vec.real
                    im = vec.imag
                    print("dynamical matrix(%3d - %3d) "
                          "%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f" % 
                          (i+1, j+1, re[0], im[0], re[1], im[1], re[2], im[2]))
                print('')

    def _smallest_vectors_log(self):
        r = self._smallest_vectors
        m = self._multiplicity

        print("#%4s %4s %4s %4s %4s %10s" % 
              ("p_i", "p_j", "s_i", "s_j", "mult", "dist"))
        for p_i, s_i in enumerate(self._p2s_map): # run in primitive
            for s_j in range(r.shape[0]): # run in supercell
                for tmp_p_j, tmp_s_j in enumerate(self._p2s_map):
                    if self._s2p_map[s_j] == tmp_s_j:
                        p_j = tmp_p_j
                for k in range(m[s_j][p_i]):
                    print(" %4d %4d %4d %4d %4d %10.5f" %
                          (p_i+1, p_j+1, s_i+1, s_j+1, m[s_j][p_i],
                           np.linalg.norm(np.dot(r[s_j][p_i][k],
                                                 self._pcell.get_cell()))))

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
        phonoc.dynamical_matrix(dm.view(dtype='double'),
                                fc,
                                np.array(q, dtype='double'),
                                vectors,
                                multiplicity,
                                mass,
                                self._s2p_map,
                                self._p2s_map)
        self._dynamical_matrix = dm

# Non analytical term correction (NAC)
# Call this when NAC is required instead of DynamicalMatrix
class DynamicalMatrixNAC(DynamicalMatrix):
    def __init__(self,
                 supercell,
                 primitive,
                 force_constants,
                 nac_params=None,
                 decimals=None,
                 symprec=1e-5):

        DynamicalMatrix.__init__(self,
                                 supercell,
                                 primitive,
                                 force_constants,
                                 decimals=decimals,
                                 symprec=1e-5)
        self._bare_force_constants = self._force_constants.copy()

        self._nac = True
        if nac_params is not None:
            self.set_nac_params(nac_params)

    def get_born_effective_charges(self):
        return self._born

    def get_nac_factor(self):
        return self._unit_conversion * 4.0 * np.pi / self._pcell.get_volume()

    def get_dielectric_constant(self):
        return self._dielectric
    
    def set_nac_params(self, nac_params):
        self._born = np.array(nac_params['born'], dtype='double', order='C')
        self._unit_conversion = nac_params['factor']
        self._dielectric = np.array(nac_params['dielectric'],
                                    dtype='double', order='C')

    def set_dynamical_matrix(self, q_red, q_direction=None, verbose=False):
        num_atom = self._pcell.get_number_of_atoms()

        if q_direction is None:
            q = np.dot(q_red, np.linalg.inv(self._pcell.get_cell()).T)
        else:
            q = np.dot(q_direction, np.linalg.inv(self._pcell.get_cell()).T)

        if (q_direction is None and np.abs(q).sum() < self._symprec) or \
                ((q_direction is not None) and
                 np.abs(q_direction).sum() < self._symprec):
            self._force_constants = self._bare_force_constants.copy()
            DynamicalMatrix.set_dynamical_matrix(self, q_red, verbose)
            return False
    
        volume = self._pcell.get_volume()
        constant = (self._unit_conversion * 4.0 * np.pi / volume
                    / np.dot(q, np.dot(self._dielectric, q)))

        # Wang method (J. Phys.: Condens. Matter 22 (2010) 202201)
        import phonopy._phonopy as phonoc
        try:
            import phonopy._phonopy as phonoc
            self._set_c_nac_dynamical_matrix(q_red, q, constant)
        except ImportError:
            fc = self._bare_force_constants.copy()
            nac_q = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
            for i in range(num_atom):
                A_i = np.dot(q, self._born[i])
                for j in range(num_atom):
                    A_j = np.dot(q, self._born[j])
                    nac_q[i, j] = np.outer(A_i, A_j) * constant
            self._set_NAC_force_constants(fc, nac_q)
            self._force_constants = fc
            DynamicalMatrix.set_dynamical_matrix(self, q_red, verbose)

    def _set_NAC_force_constants(self, fc, nac_q):
        N = (self._scell.get_number_of_atoms() //
             self._pcell.get_number_of_atoms())
        for s1 in range(self._scell.get_number_of_atoms()):
            # This if-statement is the trick.
            # In contructing dynamical matrix in phonopy
            # fc of left indices with s1 == self._s2p_map[ s1 ] are
            # only used.
            if s1 != self._s2p_map[s1]:
                continue
            p1 = self._p2p_map[s1]
            for s2 in range(self._scell.get_number_of_atoms()):            
                p2 = self._p2p_map[s2]
                fc[s1, s2] += nac_q[p1, p2] / N

    def _get_charge_sum(self, num_atom, q):
        charge_sum = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
        for i in range(num_atom):
            for j in range(num_atom):
                for a in (0, 1, 2):
                    for b in (0, 1, 2):
                        charge_sum[i, j, a, b] = \
                            np.dot(q, self._born[i, :, a]) * np.dot(q, self._born[j, :, b])
        return charge_sum

    def _set_c_nac_dynamical_matrix(self, q_red, q, factor):
        import phonopy._phonopy as phonoc

        fc = self._bare_force_constants.copy()
        vectors = self._smallest_vectors
        mass = self._pcell.get_masses()
        multiplicity = self._multiplicity
        size_prim = len(mass)
        itemsize = self._force_constants.itemsize
        dm = np.zeros((size_prim * 3, size_prim * 3),
                      dtype=("c%d" % (itemsize * 2)))
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
        self._dynamical_matrix = dm


# Helper methods
def get_equivalent_smallest_vectors(atom_number_supercell,
                                    atom_number_primitive,
                                    supercell,
                                    primitive_lattice,
                                    symprec):
    distances = []
    differences = []
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    positions = np.dot(supercell.get_positions(), np.linalg.inv(reduced_bases))

    # Atomic positions are confined into the lattice made of reduced bases.
    for pos in positions:
        pos -= np.rint(pos)

    p_pos = positions[atom_number_primitive]
    s_pos = positions[atom_number_supercell]
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                # The vector arrow is from the atom in primitive to
                # the atom in supercell cell plus a supercell lattice
                # point. This is related to determine the phase
                # convension when building dynamical matrix.
                diff = s_pos + [i, j, k] - p_pos
                differences.append(diff)
                vec = np.dot(diff, reduced_bases)
                distances.append(np.linalg.norm(vec))

    minimum = min(distances)
    smallest_vectors = []
    for i in range(27):
        if abs(minimum - distances[i]) < symprec:
            relative_scale = np.dot(reduced_bases,
                                    np.linalg.inv(primitive_lattice))
            smallest_vectors.append(np.dot(differences[i], relative_scale))
            
    return smallest_vectors

def get_smallest_vectors(supercell, primitive, symprec):
    """
    shortest_vectors:

      Shortest vectors from an atom in primitive cell to an atom in
      supercell in the fractional coordinates. If an atom in supercell
      is on the border centered at an atom in primitive and there are
      multiple vectors that have the same distance and different
      directions, several shortest vectors are stored. The
      multiplicity is stored in another array, "multiplicity".
      [atom_super, atom_primitive, multiple-vectors, 3]
      
    multiplicity:
      Number of multiple shortest vectors (third index of "shortest_vectors")
      [atom_super, atom_primitive]
    """

    p2s_map = primitive.get_primitive_to_supercell_map()
    size_super = supercell.get_number_of_atoms()
    size_prim = primitive.get_number_of_atoms()
    shortest_vectors = np.zeros((size_super, size_prim, 27, 3), dtype='double')
    multiplicity = np.zeros((size_super, size_prim), dtype='intc')

    for i in range(size_super): # run in supercell
        for j, s_j in enumerate(p2s_map): # run in primitive
            vectors = get_equivalent_smallest_vectors(i,
                                                      s_j,
                                                      supercell, 
                                                      primitive.get_cell(),
                                                      symprec)
            multiplicity[i][j] = len(vectors)
            for k, elem in enumerate(vectors):
                shortest_vectors[i][j][k] = elem

    return shortest_vectors, multiplicity

