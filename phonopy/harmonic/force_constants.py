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
import sys
from phonopy.structure.cells import get_reduced_bases
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors

def get_force_constants(set_of_forces,
                        symmetry,
                        supercell,
                        atom_list=None,
                        decimals=None):

    if atom_list==None:
        force_constants = run_force_constants(
            supercell,
            symmetry,
            set_of_forces,
            range(supercell.get_number_of_atoms()))
    else:
        force_constants = run_force_constants(supercell,
                                              symmetry,
                                              set_of_forces,
                                              atom_list)

    if decimals:
        return force_constants.round(decimals=decimals)
    else:
        return force_constants

def cutoff_force_constants(force_constants,
                           supercell,
                           cutoff_radius,
                           symprec=1e-5):
    num_atom = supercell.get_number_of_atoms()
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)
    positions = np.dot(supercell.get_positions(),
                       np.linalg.inv(reduced_bases))
    for i in range(num_atom):
        pos_i = positions[i]
        for j in range(num_atom):
            pos_j = positions[j]
            min_distance = get_shortest_distance_in_PBC(pos_i,
                                                        pos_j,
                                                        reduced_bases)
            if min_distance > cutoff_radius:
                force_constants[i, j] = 0.0

def get_shortest_distance_in_PBC(pos_i, pos_j, reduced_bases):
    distances = []
    for k in (-1, 0, 1):
        for l in (-1, 0, 1):
            for m in (-1, 0, 1):
                diff = pos_j + np.array([k, l, m]) - pos_i
                distances.append(np.linalg.norm(np.dot(diff, reduced_bases)))
    return np.min(distances)
                        

def symmetrize_force_constants(force_constants, iteration=3):
    for i in range(iteration):
        set_permutation_symmetry(force_constants)
        set_translational_invariance(force_constants)

def run_force_constants(supercell,
                        symmetry,
                        set_of_forces,
                        atom_list):
    """
    Bare force_constants is returned.

    Force constants, Phi, are calculated from sets for forces, F, and
    atomic displacement, d:
      Phi = -F / d
    This is solved by matrix pseudo-inversion.
    Crsytal symmetry is included when creating F and d matrices.

    force_constants[ i, j, a, b ]
      i: Atom index of finitely displaced atom.
      j: Atom index at which force on the atom is measured.
      a, b: Cartesian direction indices = (0, 1, 2) for i and j, respectively
    """
    
    force_constants = np.zeros((supercell.get_number_of_atoms(),
                                supercell.get_number_of_atoms(),
                                3, 3), dtype='double')

    # Fill force_constants[ displaced_atoms, all_atoms_in_supercell ]
    atom_list_done = get_force_constants_disps(force_constants,
                                               supercell,
                                               set_of_forces,
                                               symmetry)

    # Distribute non-equivalent force constants to those equivalent
    for i in atom_list:
        if not (i in atom_list_done):
            distribute_force_constants(force_constants,
                                       i,
                                       atom_list_done,
                                       supercell,
                                       symmetry)

    return force_constants

def distribute_force_constants(force_constants,
                               atom_disp,
                               atom_list_done,
                               supercell,
                               symmetry,
                               is_symmetrize=False):

    positions = supercell.get_scaled_positions()
    lattice = supercell.get_cell()
    symprec = symmetry.get_symmetry_tolerance()
    rotations = symmetry.get_symmetry_operations()['rotations']
    trans = symmetry.get_symmetry_operations()['translations']

    if is_symmetrize:
        map_atom_disps, map_syms = get_all_atom_mappings_by_symmetry(
            atom_list_done,
            atom_disp,
            rotations, 
            trans,
            positions,
            symprec)
    
        for i, pos_i in enumerate(positions):
            for map_atom_disp, map_sym in zip(map_atom_disps, map_syms):
                # L R L^-1
                rot_cartesian = similarity_transformation(lattice.T,
                                                          rotations[map_sym])
                distribute_fc2_part(force_constants,
                                    positions,
                                    atom_disp,
                                    map_atom_disp,
                                    i,
                                    rot_cartesian,
                                    rotations[map_sym],
                                    trans[map_sym],
                                    symprec)
    
            force_constants[atom_disp, i] /= len(map_atom_disps)
    else:
        map_atom_disp, map_sym = get_atom_mapping_by_symmetry(
            atom_list_done,
            atom_disp,
            rotations, 
            trans,
            positions,
            symprec)

        # L R L^-1
        rot_cartesian = similarity_transformation(lattice.T,
                                                  rotations[map_sym])

        distribute_fc2_part(force_constants,
                            positions,
                            atom_disp,
                            map_atom_disp,
                            rot_cartesian,
                            rotations[map_sym],
                            trans[map_sym],
                            symprec)



def distribute_fc2_part(force_constants,
                        positions,
                        atom_disp,
                        map_atom_disp,
                        rot_cartesian,
                        r,
                        t,
                        symprec):

    try:
        import phonopy._phonopy as phonoc
        phonoc.distribute_fc2(force_constants,
                              positions,
                              atom_disp,
                              map_atom_disp,
                              rot_cartesian,
                              r,
                              t,
                              symprec)
    except ImportError:
        for i, pos_i in enumerate(positions):
            rot_pos = np.dot(pos_i, r.T) + t
            rot_atom = -1
            for j, pos_j in enumerate(positions):
                diff = pos_j - rot_pos
                if (abs(diff - np.rint(diff)) < symprec).all():
                    rot_atom = j
                    break
        
            if rot_atom < 0:
                print 'Input forces are not enough to calculate force constants,'
                print 'or something wrong (e.g. crystal structure does not match).'
                raise ValueError
            
            # R^-1 P R (inverse transformation)
            force_constants[atom_disp, i] += similarity_transformation(
                rot_cartesian.T,
                force_constants[map_atom_disp,
                                rot_atom])
    
    
def get_atom_mapping_by_symmetry(atom_list_done,
                                 atom_number,
                                 rotations,
                                 translations,
                                 positions,
                                 symprec=1e-5):
    """
    Find a mapping from an atom to an atom in the atom list done.
    """

    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_number], r.T) + t
        for j in atom_list_done:
            diff = positions[j] - rot_pos
            if (abs(diff - np.rint(diff)) < symprec).all():
                return j, i

    print 'Input forces are not enough to calculate force constants,'
    print 'or something wrong (e.g. crystal structure does not match).'
    raise ValueError

def get_all_atom_mappings_by_symmetry(atom_list_done,
                                      atom_number,
                                      rotations,
                                      translations,
                                      positions,
                                      symprec=1e-5):
    """
    Find mappings from an atom to atoms in the atom list done.
    """

    map_atoms = []
    map_syms = []
    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_number], r.T) + t
        for j in atom_list_done:
            diff = positions[j] - rot_pos
            if (abs(diff - np.rint(diff)) < symprec).all():
                map_atoms.append(j)
                map_syms.append(i)
                break

    if len(map_atoms) == 0:
        print 'Input forces are not enough to calculate force constants,'
        print 'or something wrong (e.g. crystal structure does not match).'
        raise ValueError

    return map_atoms, map_syms

def get_force_constants_disps(force_constants,
                              supercell,
                              set_of_forces,
                              symmetry):
    """
    Phi = -F / d
    """
    
    symprec = symmetry.get_symmetry_tolerance()
    disp_atom_list = np.unique(
        [forces.get_atom_number() for forces in set_of_forces])

    for disp_atom_number in disp_atom_list:
        site_symmetry = symmetry.get_site_symmetry(disp_atom_number)
        symmetry_matrices = get_symmetry_matrices(supercell, site_symmetry)
        rot_disps = []
        row_forces = []

        # Bind several displacements of a displaced atom
        # with symmetry operations
        for forces in set_of_forces:
            if not forces.get_atom_number() == disp_atom_number:
                continue

            displacement = forces.get_displacement()
            # Displacement * Rotation (U * A)
            rot_disps.append(get_rotated_displacements(displacement,
                                                       symmetry_matrices))
            row_forces.append(forces.get_forces())

        # Bind forces for several displacements and symmetry
        # operations of a displaced atom
        rot_disps = np.array(rot_disps).reshape(-1, 9)
        inv = np.linalg.pinv(rot_disps)

        solve_force_constants_disps(force_constants,
                                    supercell,
                                    disp_atom_number,
                                    site_symmetry,
                                    row_forces,
                                    inv,
                                    symprec)

    return disp_atom_list


def solve_force_constants_disps(force_constants,
                                supercell,
                                disp_atom_number,
                                site_symmetry,
                                sets_of_forces,
                                inv_displacements,
                                symprec):
        for i in range(supercell.get_number_of_atoms()):
            # Shift positions according to set disp_atom_number is at origin
            positions = supercell.get_scaled_positions()
            pos_center = positions[disp_atom_number].copy()
            positions -= pos_center
            # Combined forces (F)
            combined_forces = []
            try:
                import phonopy._phonopy as phonoc
                for forces in sets_of_forces:
                    rotated_forces = np.zeros(len(site_symmetry) * 3,
                                              dtype='double')
                    phonoc.rotated_forces(rotated_forces,
                                          positions,
                                          i,
                                          forces,
                                          site_symmetry,
                                          symprec)
                    combined_forces.append(rotated_forces)
            except ImportError:
                for forces in sets_of_forces:
                    combined_forces.append(
                        get_rotated_forces(positions,
                                           i,
                                           forces,
                                           site_symmetry,
                                           symprec))
    
            combined_forces = np.array(combined_forces).reshape(-1, 1)
            # Force constant (P) = -(U * A)^-1 * (F)
            force_constants[disp_atom_number,i] = \
                -np.dot(inv_displacements, combined_forces).reshape(3, 3)


def get_rotated_forces(positions,
                       atom_number,
                       forces,
                       site_symmetry,
                       symprec=1e-5):
    """
    Pack forces on atoms translated by site symmetry
    
    The relation:
    R [ F(r) ] = F(R.r)
    where R is a rotation cetering at displaced atom.
    (This is not the transformation of a function,
     but just the rotation of force vector at r.)
    """
    rot_forces = []
    for sym in site_symmetry:
        rot_pos = np.dot(positions[atom_number], sym.T)

        is_found = False
        for i, p in enumerate(positions):
            diff = p - rot_pos
            if (abs(diff - diff.round()) < symprec).all():
                rot_forces.append(forces[i])
                is_found = True
                break

        if not is_found:
            print "Phonopy encontered symmetry problem"

    return rot_forces


def set_tensor_symmetry(force_constants, supercell, symmetry):
    """
    Full force constants are symmetrized using crystal symmetry.
    This method extracts symmetrically equivalent sets of atomic pairs and
    take sum of their force constants and average the sum.
    
    Since get_force_constants_disps may include crystal symmetry, this method
    is usually meaningless.
    """

    positions = supercell.get_scaled_positions()
    symprec = symmetry.get_symmetry_tolerance()
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']

    fc_bak = force_constants.copy()

    # Create mapping table between an atom and the symmetry operated atom
    # map[ i, j ]
    # i: atom index
    # j: operation index
    mapping = []
    for pos_i in positions:
        map_local = []
        for rot, trans in zip(rotations, translations):
            rot_pos = np.dot(pos_i, rot.T) + trans
            for j, pos_j in enumerate(positions):
                diff = pos_j - rot_pos
                if (abs(diff -diff.round()) < symprec).all():
                    map_local.append(j)
                    break
        mapping.append(map_local)
    mapping = np.array(mapping)

    # Look for the symmetrically equivalent force constant tensors
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            tmp_fc = np.zeros((3, 3), dtype='double')
            for k, rot in enumerate(rotations):
                cart_rot = similarity_transformation(
                    supercell.get_cell().T, rot)

                # Reverse rotation of force constant is summed
                tmp_fc += similarity_transformation(cart_rot.T,
                                                    fc_bak[mapping[i, k],
                                                           mapping[j, k]])
            # Take average and set to new force cosntants
            force_constants[i, j] = tmp_fc / len(rotations)

def set_translational_invariance(force_constants):
    """
    Translational invariance is imposed.  This is quite simple
    implementation, which is just take sum of the force constants in
    an axis and an atom index. The sum has to be zero due to the
    translational invariance. If the sum is not zero, this error is
    uniformly subtracted from force constants.
    """
    for i in range(force_constants.shape[1]):
        for j in range(force_constants.shape[2]):
            for k in range(force_constants.shape[3]):
                force_constants[:,i,j,k] -= np.sum(
                    force_constants[:,i,j,k]) / force_constants.shape[0]

def set_permutation_symmetry(force_constants):
    """
    Phi_ij_ab = Phi_ji_ba
    
    i, j: atom index
    a, b: Cartesian axis index

    This is not necessary for harmonic phonon calculation because this
    condition is imposed when making dynamical matrix Hermite in
    dynamical_matrix.py.
    """
    fc_copy = force_constants.copy()
    for i in range(force_constants.shape[0]):
        for j in range(force_constants.shape[1]):
            force_constants[i, j] = (force_constants[i, j] +
                                     fc_copy[j, i].T) / 2

def rotational_invariance(force_constants,
                          supercell,
                          primitive,
                          symprec=1e-5):
    """
    *** Under development ***
    Just show how force constant is close to the condition of rotational invariance,
    """
    print "Check rotational invariance ..."

    fc = force_constants 
    p2s = primitive.get_primitive_to_supercell_map()

    abc = "xyz"
    
    for pi, p in enumerate(p2s):
        for i in range(3):
            mat = np.zeros((3, 3), dtype='double')
            for s in range(supercell.get_number_of_atoms()):
                vecs = np.array(get_equivalent_smallest_vectors(
                        s, p, supercell, primitive.get_cell(), symprec))
                m = len(vecs)
                v = np.dot(vecs[:,:].sum(axis=0) / m, primitive.get_cell())
                for j in range(3):
                    for k in range(3):
                        mat[j, k] += (fc[p, s, i, j] * v[k] -
                                      fc[p, s, i, k] * v[j])

            print "Atom %d %s" % (p+1, abc[i])
            for vec in mat:
                print "%10.5f %10.5f %10.5f" % tuple(vec)

def force_constants_log(force_constants):
    fs = force_constants
    for i, fs_i in enumerate(fs):
        for j, fs_j in enumerate(fs_i):
            for v in fs_j:
                print "force constant (%d - %d): %10.5f %10.5f %10.5f" % (
                    i + 1, j + 1, v[0], v[1], v[2])


# Shared functions to calculate force constant
def get_symmetry_matrices(cell, site_symmetry):
    """
    Transformation of 2nd order force constant

    In the phonopy implementation (Cartesian coords.)

    (R.F)^T = -(R.U)^T Psi' --> F^T = -U^T.R^T.Psi'.R
    Therefore,
    Psi = R^T.Psi'.R --> Psi' = R.Psi.R^T

    The symmetrical relation between Psi and Psi' can be represented
    by a 9x9 matrix. What we want is transformation matrix A defined
    by

    P' = A.P

    where P' and P are the 9x1 matrices and A is the 9x9 matrices.
    """
    matrices = []
    for reduced_rot in site_symmetry:
        mat = []
        rot = similarity_transformation(cell.get_cell().T, reduced_rot)
        for i in range(3):
            for j in range(3):
                psi = []
                for k in range(3):
                    for l in range(3):
                        psi.append(rot[i, k] * rot[j, l])
                mat.append(psi)
        matrices.append(mat)
    return np.array(matrices)

def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))
                
def get_rotated_displacements(displacement, symmetry_matrices):
    """
    U x A
                                                [123456789]
                                                [2        ]
                                                [3        ]
      [ d_x  0   0   d_y  0   0   d_z  0   0  ] [4        ]
    U [  0  d_x  0    0  d_y  0    0  d_z  0  ] [5   A    ]
      [  0   0  d_x   0   0  d_y   0   0  d_z ] [6        ]
                                                [7        ]
                                                [8        ]
                                                [9        ]
    """
    rot_disps = []
    for sym in symmetry_matrices:
        rot_disps.append(np.dot(expand_displacement(displacement), sym))
    return rot_disps

def expand_displacement(displacement):
    """
    [ d_x  0   0   d_y  0   0   d_z  0   0  ]
    [  0  d_x  0    0  d_y  0    0  d_z  0  ]
    [  0   0  d_x   0   0  d_y   0   0  d_z ]
    """
    d = displacement
    disp = np.hstack((np.eye(3)*d[0], np.eye(3)*d[1], np.eye(3)*d[2]))
    return disp

