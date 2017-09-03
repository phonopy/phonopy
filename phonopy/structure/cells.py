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
import phonopy.structure.spglib as spg
from phonopy.structure.atoms import PhonopyAtoms as Atoms

def get_supercell(unitcell, supercell_matrix, is_old_style=True, symprec=1e-5):
    return Supercell(unitcell,
                     supercell_matrix,
                     is_old_style=is_old_style,
                     symprec=symprec)

def get_primitive(supercell, primitive_frame, symprec=1e-5):
    return Primitive(supercell, primitive_frame, symprec=symprec)

def trim_cell(relative_axes, cell, symprec):
    """
    relative_axes: relative axes to supercell axes
    Trim positions outside relative axes

    """
    positions = cell.get_scaled_positions()
    numbers = cell.get_atomic_numbers()
    masses = cell.get_masses()
    magmoms = cell.get_magnetic_moments()
    lattice = cell.get_cell()
    trimed_lattice = np.dot(relative_axes.T, lattice)

    trimed_positions = []
    trimed_numbers = []
    if masses is None:
        trimed_masses = None
    else:
        trimed_masses = []
    if magmoms is None:
        trimed_magmoms = None
    else:
        trimed_magmoms = []
    extracted_atoms = []

    positions_in_new_lattice = np.dot(positions, np.linalg.inv(relative_axes).T)
    positions_in_new_lattice -= np.floor(positions_in_new_lattice)
    trimed_positions = np.zeros_like(positions_in_new_lattice)
    num_atom = 0

    mapping_table = np.arange(len(positions), dtype='intc')
    for i, pos in enumerate(positions_in_new_lattice):
        is_overlap = False
        if num_atom > 0:
            diff = trimed_positions[:num_atom] - pos
            diff -= np.rint(diff)
            # Older numpy doesn't support axis argument.
            # distances = np.linalg.norm(np.dot(diff, trimed_lattice), axis=1)
            # overlap_indices = np.where(distances < symprec)[0]
            distances = np.sqrt(
                np.sum(np.dot(diff, trimed_lattice) ** 2, axis=1))
            overlap_indices = np.where(distances < symprec)[0]
            if len(overlap_indices) > 0:
                assert len(overlap_indices) == 1
                is_overlap = True
                mapping_table[i] = extracted_atoms[overlap_indices[0]]

        if not is_overlap:
            trimed_positions[num_atom] = pos
            num_atom += 1
            trimed_numbers.append(numbers[i])
            if masses is not None:
                trimed_masses.append(masses[i])
            if magmoms is not None:
                trimed_magmoms.append(magmoms[i])
            extracted_atoms.append(i)

    trimed_cell = Atoms(numbers=trimed_numbers,
                        masses=trimed_masses,
                        magmoms=trimed_magmoms,
                        scaled_positions=trimed_positions[:num_atom],
                        cell=trimed_lattice,
                        pbc=True)

    return trimed_cell, extracted_atoms, mapping_table

def print_cell(cell, mapping=None, stars=None):
    symbols = cell.get_chemical_symbols()
    masses = cell.get_masses()
    magmoms = cell.get_magnetic_moments()
    lattice = cell.get_cell()
    print("Lattice vectors:")
    print("  a %20.15f %20.15f %20.15f" % tuple(lattice[0]))
    print("  b %20.15f %20.15f %20.15f" % tuple(lattice[1]))
    print("  c %20.15f %20.15f %20.15f" % tuple(lattice[2]))
    print("Atomic positions (fractional):")
    for i, v in enumerate(cell.get_scaled_positions()):
        num = " "
        if stars is not None:
            if i in stars:
                num = "*"
        num += "%d" % (i + 1)
        line = ("%5s %-2s%18.14f%18.14f%18.14f" %
                (num, symbols[i], v[0], v[1], v[2]))
        if masses is not None:
            line += " %7.3f" % masses[i]
        if magmoms is not None:
            line += "  %5.3f" % magmoms[i]
        if mapping is None:
            print(line)
        else:
            print(line + " > %d" % (mapping[i] + 1))

class Supercell(Atoms):
    """Build supercell from supercell matrix
    In this function, unit cell is considered
    [1,0,0]
    [0,1,0]
    [0,0,1].
    Supercell matrix is given by relative ratio, e.g,
    [-1, 1, 1]
    [ 1,-1, 1]  is for FCC from simple cubic.
    [ 1, 1,-1]
    In this case multiplicities of surrounding simple lattice are [2,2,2].

    First, create supercell with surrounding simple lattice.
    Second, trim the surrounding supercell with the target lattice.
    """

    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 is_old_style=True,
                 symprec=1e-5):
        self._is_old_style = is_old_style
        self._s2u_map = None
        self._u2s_map = None
        self._u2u_map = None
        self._supercell_matrix = np.array(supercell_matrix, dtype='intc')
        self._create_supercell(unitcell, symprec)

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def get_supercell_to_unitcell_map(self):
        return self._s2u_map

    def get_unitcell_to_supercell_map(self):
        return self._u2s_map

    def get_unitcell_to_unitcell_map(self):
        return self._u2u_map

    def _create_supercell(self, unitcell, symprec):
        mat = self._supercell_matrix

        if self._is_old_style:
            P = None
            multi = self._get_surrounding_frame(mat)
            # trim_fram is to trim overlapping atoms.
            trim_frame = np.array([mat[0] / float(multi[0]),
                                   mat[1] / float(multi[1]),
                                   mat[2] / float(multi[2])])
        else:
            # In the new style, it is unnecessary to trim atoms,
            if (np.diag(np.diagonal(mat)) != mat).any():
                snf = SNF3x3(mat)
                snf.run()
                P = snf.P
                multi = np.diagonal(snf.A)
            else:
                P = None
                multi = np.diagonal(mat)
            trim_frame = np.eye(3)

        sur_cell, u2sur_map = self._get_simple_supercell(unitcell, multi, P)
        supercell, sur2s_map, mapping_table = trim_cell(trim_frame,
                                                        sur_cell,
                                                        symprec)

        num_satom = supercell.get_number_of_atoms()
        num_uatom = unitcell.get_number_of_atoms()
        N = num_satom // num_uatom

        if N != determinant(self._supercell_matrix):
            print("Supercell creation failed.")
            print("Probably some atoms are overwrapped. "
                  "The mapping table is give below.")
            print(mapping_table)
            Atoms.__init__(self)
        else:
            Atoms.__init__(self,
                           numbers=supercell.get_atomic_numbers(),
                           masses=supercell.get_masses(),
                           magmoms=supercell.get_magnetic_moments(),
                           scaled_positions=supercell.get_scaled_positions(),
                           cell=supercell.get_cell(),
                           pbc=True)
            self._u2s_map = np.arange(num_uatom) * N
            self._u2u_map = dict([(j, i) for i, j in enumerate(self._u2s_map)])
            self._s2u_map = np.array(u2sur_map)[sur2s_map] * N

    def _get_simple_supercell(self, unitcell, multi, P):
        if self._is_old_style:
            mat = np.diag(multi)
        else:
            mat = self._supercell_matrix

        # Scaled positions within the frame, i.e., create a supercell that
        # is made simply to multiply the input cell.
        positions = unitcell.get_scaled_positions()
        numbers = unitcell.get_atomic_numbers()
        masses = unitcell.get_masses()
        magmoms = unitcell.get_magnetic_moments()
        lattice = unitcell.get_cell()

        # Index of a axis runs fastest for creating lattice points.
        # See numpy.meshgrid document for the complicated index order for 3D
        b, c, a = np.meshgrid(range(multi[1]), range(multi[2]), range(multi[0]))
        lattice_points = np.c_[a.ravel(), b.ravel(), c.ravel()]

        if P is not None:
            # If supercell matrix is not a diagonal matrix,
            # Smith normal form is applied to find oblique basis vectors for
            # supercell and primitive cells, where their basis vectos are
            # parallel each other. By this reason, simple construction of
            # supercell becomes possible.
            P_inv = np.rint(np.linalg.inv(P)).astype(int)
            assert determinant(P_inv) == 1
            lattice_points = np.dot(lattice_points, P_inv.T)

        n = len(positions)
        n_l = len(lattice_points)
        # tile: repeat blocks
        # repeat: repeat each element
        positions_multi = np.dot(np.tile(lattice_points, (n, 1)) +
                                 np.repeat(positions, n_l, axis=0),
                                 np.linalg.inv(mat).T)
        numbers_multi = np.repeat(numbers, n_l)
        atom_map = np.repeat(np.arange(n), n_l)
        if masses is None:
            masses_multi = None
        else:
            masses_multi = np.repeat(masses, n_l)
        if magmoms is None:
            magmoms_multi = None
        else:
            magmoms_multi = np.repeat(magmoms, n_l)

        simple_supercell = Atoms(numbers=numbers_multi,
                                 masses=masses_multi,
                                 magmoms=magmoms_multi,
                                 scaled_positions=positions_multi,
                                 cell=np.dot(mat, lattice),
                                 pbc=True)

        return simple_supercell, atom_map

    def _get_surrounding_frame(self, supercell_matrix):
        # Build a frame surrounding supercell lattice
        # For example,
        #  [2,0,0]
        #  [0,2,0] is the frame for FCC from simple cubic.
        #  [0,0,2]

        m = np.array(supercell_matrix)
        axes = np.array([[0, 0, 0],
                         m[:,0],
                         m[:,1],
                         m[:,2],
                         m[:,1] + m[:,2],
                         m[:,2] + m[:,0],
                         m[:,0] + m[:,1],
                         m[:,0] + m[:,1] + m[:,2]])
        frame = [max(axes[:,i]) - min(axes[:,i]) for i in (0,1,2)]
        return frame

class Primitive(Atoms):
    def __init__(self, supercell, primitive_matrix, symprec=1e-5):
        """
        primitive_matrix (3x3 matrix):
        Primitive lattice is given with respect to supercell by
           np.dot(primitive_matrix.T, supercell.get_cell())
        """
        self._primitive_matrix = np.array(primitive_matrix)
        self._symprec = symprec
        self._p2s_map = None
        self._s2p_map = None
        self._p2p_map = None
        self._smallest_vectors = None
        self._multiplicity = None

        self._primitive_cell(supercell)
        self._supercell_to_primitive_map(supercell.get_scaled_positions())
        self._primitive_to_primitive_map()
        self._set_smallest_vectors(supercell)

    def get_primitive_matrix(self):
        return self._primitive_matrix

    def get_primitive_to_supercell_map(self):
        return self._p2s_map

    def get_supercell_to_primitive_map(self):
        return self._s2p_map

    def get_primitive_to_primitive_map(self):
        return self._p2p_map

    def get_smallest_vectors(self):
        return self._smallest_vectors, self._multiplicity

    def _primitive_cell(self, supercell):
        trimed_cell, p2s_map, mapping_table = trim_cell(self._primitive_matrix,
                                                        supercell,
                                                        self._symprec)
        Atoms.__init__(self,
                       numbers=trimed_cell.get_atomic_numbers(),
                       masses=trimed_cell.get_masses(),
                       magmoms=trimed_cell.get_magnetic_moments(),
                       scaled_positions=trimed_cell.get_scaled_positions(),
                       cell=trimed_cell.get_cell(),
                       pbc=True)

        self._p2s_map = np.array(p2s_map, dtype='intc')

    def _supercell_to_primitive_map(self, pos):
        inv_F = np.linalg.inv(self._primitive_matrix)
        s2p_map = []
        for i in range(pos.shape[0]):
            s_pos = np.dot(pos[i], inv_F.T)
            for j in self._p2s_map:
                p_pos = np.dot(pos[j], inv_F.T)
                diff = p_pos - s_pos
                diff -= np.rint(diff)
                if (abs(diff) < self._symprec).all():
                    s2p_map.append(j)
                    break
        self._s2p_map = np.array(s2p_map, dtype='intc')

    def _primitive_to_primitive_map(self):
        """
        Mapping table from supercell index to primitive index
        in primitive cell
        """
        self._p2p_map = dict([(j, i) for i, j in enumerate(self._p2s_map)])

    def _set_smallest_vectors(self, supercell):
        self._smallest_vectors, self._multiplicity = _get_smallest_vectors(
            supercell, self, self._symprec)

#
# Delaunay reduction
#
def get_reduced_bases(lattice,
                      method='delaunay',
                      tolerance=1e-5):
    """Apply reduction to basis vectors

    args:
        basis as row vectors, [a, b, c]^T
    return:
         reduced basin as row vectors, [a_red, b_red, c_red]^T
    """

    if method == 'niggli':
        return spg.niggli_reduce(lattice, eps=tolerance)
    else:
        return spg.delaunay_reduce(lattice, eps=tolerance)

#
# Shortest pairs of atoms in supercell (Wigner-Seitz like)
#
# This is currently no longer used in phonopy, but still used by
# phono3py. In phono3py, this is used to measure the shortest distance
# between arbitrary pair of atoms in supercell. Therefore this method
# may be moved to phono3py, but this way of use can also happen in
# phonopy in the future, so let's keep it for a while.
#
def get_equivalent_smallest_vectors(atom_number_supercell,
                                    atom_number_primitive,
                                    supercell,
                                    primitive_lattice,
                                    symprec):
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    reduced_bases_inv = np.linalg.inv(reduced_bases)
    cart_positions = supercell.get_positions()

    # Atomic positions are confined into the delaunay-reduced superlattice.
    # Their positions will lie in the range -0.5 < x < 0.5, so that vectors
    # drawn between them have components in the range -1 < x < 1.
    def reduced_frac_pos(i):
        vec = np.dot(cart_positions[i], reduced_bases_inv)
        return vec - np.rint(vec)
    p_pos = reduced_frac_pos(atom_number_primitive)
    s_pos = reduced_frac_pos(atom_number_supercell)

    # The vector arrow is from the atom in the primitive cell to the
    # atom in the supercell.
    differences = _get_equivalent_smallest_vectors_simple(s_pos - p_pos,
                                                          reduced_bases,
                                                          symprec)

    # Return fractional coords in the basis of the primitive cell
    #  rather than the supercell.
    relative_scale = reduced_bases.dot(np.linalg.inv(primitive_lattice))
    return differences.dot(relative_scale)

# Given:
#  - A delaunay-reduced lattice (row vectors)
#  - A fractional vector (with respect to that lattice)
#      whose coords lie in the range (-1 < x < 1)
# Produce:
#  - All fractional vectors of shortest length that are translationally
#      equivalent to that vector under the lattice.
def _get_equivalent_smallest_vectors_simple(frac_vector,
                                            reduced_bases, # row vectors
                                            symprec):

    # Try all nearby images of the vector
    lattice_points = np.array([
        [i, j, k] for i in (-1, 0, 1)
                  for j in (-1, 0, 1)
                  for k in (-1, 0, 1)
    ])
    candidates = frac_vector + lattice_points

    # Filter out the best ones by computing cartesian lengths.
    # (A "clever" optimizer might try to skip the square root calculation here,
    #  but he would be wrong; we're comparing a *difference* to the tolerance)
    lengths = np.sqrt(np.sum(np.dot(candidates, reduced_bases)**2, axis=1))
    return candidates[lengths - lengths.min() < symprec]

def _get_smallest_vectors(supercell, primitive, symprec):
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

    # useful data from arguments
    p2s_map = primitive.get_primitive_to_supercell_map()
    size_super = supercell.get_number_of_atoms()
    size_prim = primitive.get_number_of_atoms()
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)

    # Reduce all positions into the cell formed by the reduced bases.
    supercell_fracs = np.dot(supercell.get_positions(),
                             np.linalg.inv(reduced_bases))
    supercell_fracs -= np.rint(supercell_fracs)
    primitive_fracs = supercell_fracs[list(p2s_map)]

    # For each vector, we will need to consider all nearby images in the reduced bases.
    lattice_points = np.array([
        [i, j, k] for i in (-1, 0, 1)
                  for j in (-1, 0, 1)
                  for k in (-1, 0, 1)
    ])

    # Here's where things get interesting.
    # We want to avoid manually iterating over all possible pairings of
    # supercell atoms and primitive atoms, because doing so creates a
    # tight loop in larger structures that is difficult to optimize.
    #
    # Furthermore, it seems wise to call numpy.dot on as large of an array
    # as possible, since numpy can shell out to BLAS to handle the
    # real heavy lifting.

    # For every atom in the supercell and every atom in the primitive cell,
    # we want 27 images of the vector between them.
    #
    # 'None' is used to insert trivial axes to make these arrays broadcast.
    #
    # shape: (size_super, size_prim, 27, 3)
    candidate_fracs = (
        supercell_fracs[:, None, None, :]    # shape: (size_super, 1, 1, 3)
        - primitive_fracs[None, :, None, :]  # shape: (1, size_prim, 1, 3)
        + lattice_points[None, None, :, :]   # shape: (1, 1, 27, 3)
    )

    # To compute the lengths, we want cartesian positions.
    #
    # Conveniently, calling 'numpy.dot' between a 4D array and a 2D array
    # does vector-matrix multiplication on each row vector in the last axis
    # of the 4D array.
    #
    # shape: (size_super, size_prim, 27, 3)
    candidate_carts = np.dot(candidate_fracs, reduced_bases)
    # shape: (size_super, size_prim, 27)
    lengths = np.sqrt(np.sum(candidate_carts**2, axis=-1))

    # Create the output, initially consisting of all candidate vectors scaled
    # by the primitive cell.
    #
    # shape: (size_super, size_prim, 27, 3)
    candidate_vectors = np.dot(
        candidate_fracs,
        reduced_bases.dot(np.linalg.inv(primitive.get_cell())))

    # The last final bits are done in C.
    #
    # We will gather the shortest ones from each list of 27 vectors.
    shortest_vectors = np.zeros_like(candidate_vectors,
                                     dtype='double', order='C')
    multiplicity = np.zeros((size_super, size_prim), dtype='intc', order='C')

    import phonopy._phonopy as phonoc
    phonoc.gsv_copy_smallest_vectors(shortest_vectors,
                                     multiplicity,
                                     candidate_vectors,
                                     lengths,
                                     symprec)

    return shortest_vectors, multiplicity

#
# Other tiny tools
#
def get_angles(lattice):
    a, b, c = get_cell_parameters(lattice)
    alpha = np.arccos(np.vdot(lattice[1], lattice[2]) / b / c) / np.pi * 180
    beta  = np.arccos(np.vdot(lattice[2], lattice[0]) / c / a) / np.pi * 180
    gamma = np.arccos(np.vdot(lattice[0], lattice[1]) / a / b) / np.pi * 180
    return alpha, beta, gamma

def get_cell_parameters(lattice):
    return np.sqrt(np.dot (lattice, lattice.transpose()).diagonal())

def get_cell_matrix(a, b, c, alpha, beta, gamma):
    # These follow 'matrix_lattice_init' in matrix.c of GDIS
    alpha *= np.pi / 180
    beta *= np.pi / 180
    gamma *= np.pi / 180
    a1 = a
    a2 = 0.0
    a3 = 0.0
    b1 = np.cos(gamma)
    b2 = np.sin(gamma)
    b3 = 0.0
    c1 = np.cos(beta)
    c2 = (2 * np.cos(alpha) + b1**2 + b2**2 - 2 * b1 * c1 - 1) / (2 * b2)
    c3 = np.sqrt(1 - c1**2 - c2**2)
    lattice = np.zeros((3, 3), dtype='double')
    lattice[0, 0] = a
    lattice[1] = np.array([b1, b2, b3]) * b
    lattice[2] = np.array([c1, c2, c3]) * c
    return lattice

def determinant(m):
    return (m[0][0] * m[1][1] * m[2][2] -
            m[0][0] * m[1][2] * m[2][1] +
            m[0][1] * m[1][2] * m[2][0] -
            m[0][1] * m[1][0] * m[2][2] +
            m[0][2] * m[1][0] * m[2][1] -
            m[0][2] * m[1][1] * m[2][0])

#
# Smith normal form for 3x3 integer matrix
# This code is maintained at https://github.com/atztogo/snf3x3.
#
class SNF3x3(object):
    def __init__(self, A):
        self._A_orig = np.array(A, dtype='intc')
        self._A = np.array(A, dtype='intc')
        self._Ps = []
        self._Qs = []
        self._L = []
        self._P = None
        self._Q = None
        self._attempt = 0

    def run(self):
        for i in self:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        self._attempt += 1
        if self._first():
            if self._second():
                self._set_PQ()
                raise StopIteration
        return self._attempt

    def next(self):
        self.__next__()

    @property
    def A(self):
        return self._A.copy()

    @property
    def P(self):
        return self._P.copy()

    @property
    def Q(self):
        return self._Q.copy()

    def _set_PQ(self):
        if np.linalg.det(self._A) < 0:
            for i in range(3):
                if self._A[i, i] < 0:
                    self._flip_sign_row(i)
            self._Ps += self._L
            self._L = []

        P = np.eye(3, dtype='intc')
        for _P in self._Ps:
            P = np.dot(_P, P)
        Q = np.eye(3, dtype='intc')
        for _Q in self._Qs:
            Q = np.dot(Q, _Q.T)

        if np.linalg.det(P) < 0:
            P = -P
            Q = -Q

        self._P = P
        self._Q = Q

    def _first(self):
        self._first_one_loop()
        A = self._A
        if A[1, 0] == 0 and A[2, 0] == 0:
            return True
        elif A[1, 0] % A[0, 0] == 0 and A[2, 0] % A[0, 0] == 0:
            self._first_finalize()
            self._Ps += self._L
            self._L = []
            return True
        else:
            return False

    def _first_one_loop(self):
        self._first_column()
        self._Ps += self._L
        self._L = []
        self._A = self._A.T
        self._first_column()
        self._Qs += self._L
        self._L = []
        self._A = self._A.T

    def _first_column(self):
        i = self._search_first_pivot()
        if i > 0:
            self._swap_rows(0, i)

        if self._A[1, 0] != 0:
            self._zero_first_column(1)
        if self._A[2, 0] != 0:
            self._zero_first_column(2)

    def _zero_first_column(self, j):
        if self._A[j, 0] < 0:
            self._flip_sign_row(j)
        A = self._A
        r, s, t = xgcd([A[0, 0], A[j, 0]])
        self._set_zero(0, j, A[0, 0], A[j, 0], r, s, t)

    def _search_first_pivot(self):
        A = self._A
        for i in range(3): # column index
            if A[i, 0] != 0:
                return i

    def _first_finalize(self):
        """Set zeros along the first colomn except for A[0, 0]

        This is possible only when A[1,0] and A[2,0] are dividable by A[0,0].

        """

        A = self._A
        L = np.eye(3, dtype='intc')
        L[1, 0] = -A[1, 0] // A[0, 0]
        L[2, 0] = -A[2, 0] // A[0, 0]
        self._L.append(L.copy())
        self._A = np.dot(L, self._A)

    def _second(self):
        """Find Smith normal form for Right-low 2x2 matrix"""

        self._second_one_loop()
        A = self._A
        if A[2, 1] == 0:
            return True
        elif A[2, 1] % A[1, 1] == 0:
            self._second_finalize()
            self._Ps += self._L
            self._L = []
            return True
        else:
            return False

    def _second_one_loop(self):
        self._second_column()
        self._Ps += self._L
        self._L = []
        self._A = self._A.T
        self._second_column()
        self._Qs += self._L
        self._L = []
        self._A = self._A.T

    def _second_column(self):
        """Right-low 2x2 matrix

        Assume elements in first row and column are all zero except for A[0,0].

        """

        if self._A[1, 1] == 0 and self._A[2, 1] != 0:
            self._swap_rows(1, 2)

        if self._A[2, 1] != 0:
            self._zero_second_column()

    def _zero_second_column(self):
        if self._A[2, 1] < 0:
            self._flip_sign_row(2)
        A = self._A
        r, s, t = xgcd([A[1, 1], A[2, 1]])
        self._set_zero(1, 2, A[1, 1], A[2, 1], r, s, t)

    def _second_finalize(self):
        """Set zero at A[2, 1]

        This is possible only when A[2,1] is dividable by A[1,1].

        """

        A = self._A
        L = np.eye(3, dtype='intc')
        L[2, 1] = -A[2, 1] // A[1, 1]
        self._L.append(L.copy())
        self._A = np.dot(L, self._A)

    def _swap_rows(self, i, j):
        """Swap i and j rows

        As the side effect, determinant flips.

        """

        L = np.eye(3, dtype='intc')
        L[i, i] = 0
        L[j, j] = 0
        L[i, j] = 1
        L[j, i] = 1
        self._L.append(L.copy())
        self._A = np.dot(L, self._A)

    def _flip_sign_row(self, i):
        """Multiply -1 for all elements in row"""

        L = np.eye(3, dtype='intc')
        L[i, i] = -1
        self._L.append(L.copy())
        self._A = np.dot(L, self._A)

    def _set_zero(self, i, j, a, b, r, s, t):
        """Let A[i, j] be zero based on Bezout's identity

           [ii ij]
           [ji jj] is a (k,k) minor of original 3x3 matrix.

        """

        L = np.eye(3, dtype='intc')
        L[i, i] = s
        L[i, j] = t
        L[j, i] = -b // r
        L[j, j] = a // r
        self._L.append(L.copy())
        self._A = np.dot(L, self._A)

def xgcd(vals):
    _xgcd = Xgcd(vals)
    return _xgcd.run()

class Xgcd(object):
    def __init__(self, vals):
        self._vals = np.array(vals, dtype='intc')

    def run(self):
        r0, r1 = self._vals
        s0 = 1
        s1 = 0
        t0 = 0
        t1 = 1
        for i in range(1000):
            r0, r1, s0, s1, t0, t1 = self._step(r0, r1, s0, s1, t0, t1)
            if r1 == 0:
                break
        self._rst = np.array([r0, s0, t0], dtype='intc')
        return self._rst

    def _step(self, r0, r1, s0, s1, t0, t1):
        q, m = divmod(r0, r1)
        r2 = m
        s2 = s0 - q * s1
        t2 = t0 - q * t1
        return r1, r2, s1, s2, t1, t2

    def __str__(self):
        v = self._vals
        r, s, t = self._rst
        return "%d = %d * (%d) + %d * (%d)" % (r, v[0], s, v[1], t)
