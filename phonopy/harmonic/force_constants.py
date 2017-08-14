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
from phonopy.structure.cells import (get_reduced_bases,
                                     get_equivalent_smallest_vectors)

def get_force_constants(set_of_forces,
                        symmetry,
                        supercell,
                        atom_list=None,
                        decimals=None):
    first_atoms = [{'number': x.get_atom_number(),
                    'displacement': x.get_displacement(),
                    'forces': x.get_forces()} for x in set_of_forces]
    dataset = {'natom': supercell.get_number_of_atoms(),
               'first_atoms': first_atoms}
    force_constants = get_fc2(supercell,
                              symmetry,
                              dataset,
                              atom_list=atom_list)
    return force_constants

def get_fc2(supercell,
            symmetry,
            dataset,
            atom_list=None,
            decimals=None,
            computation_algorithm="svd"):
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
    atom_list_done = _get_force_constants_disps(
        force_constants,
        supercell,
        dataset,
        symmetry,
        computation_algorithm=computation_algorithm)

    # Distribute non-equivalent force constants to those equivalent
    symprec = symmetry.get_symmetry_tolerance()
    rotations = symmetry.get_symmetry_operations()['rotations']
    trans = symmetry.get_symmetry_operations()['translations']
    positions = supercell.get_scaled_positions()
    lattice = np.array(supercell.get_cell().T, dtype='double', order='C')

    if atom_list is None:
        distribute_force_constants(force_constants,
                                   range(supercell.get_number_of_atoms()),
                                   atom_list_done,
                                   lattice,
                                   positions,
                                   rotations,
                                   trans,
                                   symprec)
    else:
        distribute_force_constants(force_constants,
                                   atom_list,
                                   atom_list_done,
                                   lattice,
                                   positions,
                                   rotations,
                                   trans,
                                   symprec)

    if decimals:
        return force_constants.round(decimals=decimals)
    else:
        return force_constants

def cutoff_force_constants(force_constants,
                           supercell,
                           cutoff_radius,
                           symprec=1e-5):
    num_atom = supercell.get_number_of_atoms()
    reduced_bases = get_reduced_bases(supercell.get_cell(), tolerance=symprec)
    positions = np.dot(supercell.get_positions(),
                       np.linalg.inv(reduced_bases))
    for i in range(num_atom):
        pos_i = positions[i]
        for j in range(num_atom):
            pos_j = positions[j]
            min_distance = _get_shortest_distance_in_PBC(pos_i,
                                                         pos_j,
                                                         reduced_bases)
            if min_distance > cutoff_radius:
                force_constants[i, j] = 0.0


def symmetrize_force_constants(force_constants, iteration=3):
    for i in range(iteration):
        set_permutation_symmetry(force_constants)
        set_translational_invariance(force_constants)

def distribute_force_constants(force_constants,
                               atom_list,
                               atom_list_done,
                               lattice, # column vectors
                               positions, # scaled (fractional)
                               rotations, # scaled (fractional)
                               trans, # scaled (fractional)
                               symprec):
    if True:
        permutations = _compute_all_sg_permutations(positions,
                                                    rotations,
                                                    trans,
                                                    lattice,
                                                    symprec)

        map_atoms, map_syms = _get_sym_mappings_from_permutations(
            permutations, atom_list_done)

        rots_cartesian = np.array([similarity_transformation(lattice, r)
                                   for r in rotations],
                                  dtype='double', order='C')

        import phonopy._phonopy as phonoc
        phonoc.distribute_fc2_with_mappings(force_constants,
                                            np.array(atom_list, dtype='intc'),
                                            rots_cartesian,
                                            permutations,
                                            np.array(map_atoms, dtype='intc'),
                                            np.array(map_syms, dtype='intc'))

    elif True:
        rots_cartesian = np.array([similarity_transformation(lattice, r)
                                   for r in rotations],
                                  dtype='double', order='C')
        atom_list_copied = []
        for i in atom_list:
            if i not in atom_list_done:
                atom_list_copied.append(i)

        import phonopy._phonopy as phonoc
        phonoc.distribute_fc2_all(force_constants,
                                  lattice,
                                  positions,
                                  np.array(atom_list_copied, dtype='intc'),
                                  np.array(atom_list_done, dtype='intc'),
                                  rots_cartesian,
                                  rotations,
                                  trans,
                                  symprec)
    else:
        for atom_disp in atom_list:
            if atom_disp in atom_list_done:
                continue
    
            map_atom_disp, map_sym = _get_atom_mapping_by_symmetry(
                atom_list_done,
                atom_disp,
                rotations,
                trans,
                lattice,
                positions,
                symprec=symprec)
    
            _distribute_fc2_part(force_constants,
                                 positions,
                                 atom_disp,
                                 map_atom_disp,
                                 lattice,
                                 rotations[map_sym],
                                 trans[map_sym],
                                 symprec)


def solve_force_constants(force_constants,
                          disp_atom_number,
                          displacements,
                          sets_of_forces,
                          supercell,
                          site_symmetry,
                          symprec,
                          computation_algorithm="svd"):
    if computation_algorithm == "regression":
        fc_info = _solve_force_constants_regression(
            force_constants,
            disp_atom_number,
            displacements,
            sets_of_forces,
            supercell,
            site_symmetry,
            symprec)
        return fc_info
    else:
        _solve_force_constants_svd(force_constants,
                                   disp_atom_number,
                                   displacements,
                                   sets_of_forces,
                                   supercell,
                                   site_symmetry,
                                   symprec)
        return None

def get_positions_sent_by_rot_inv(lattice, # column vectors
                                  positions,
                                  site_symmetry,
                                  symprec):
    rot_map_syms = []
    for sym in site_symmetry:
        # inverse permutation of sym;
        # satisfies 'rotated_positions[rot_map] == positions'
        rot_map = _compute_permutation_for_rotation(np.dot(positions, sym.T),
                                                    positions,
                                                    lattice,
                                                    symprec)
        rot_map_syms.append(rot_map)

    return np.array(rot_map_syms, dtype='intc', order='C')

def get_rotated_displacement(displacements, site_sym_cart):
    rot_disps = []
    for u in displacements:
        rot_disps.append([np.dot(sym, u) for sym in site_sym_cart])
    return np.reshape(rot_disps, (-1, 3))

def get_rotated_forces(forces_syms, site_sym_cart):
    rot_forces = []
    for forces, sym_cart in zip(forces_syms, site_sym_cart):
        rot_forces.append(np.dot(forces, sym_cart.T))

    return rot_forces

def set_tensor_symmetry_old(force_constants,
                            lattice, # column vectors
                            positions,
                            symmetry):
    """
    Full force constants are symmetrized using crystal symmetry.
    This method extracts symmetrically equivalent sets of atomic pairs and
    take sum of their force constants and average the sum.

    Since get_force_constants_disps may include crystal symmetry, this method
    is usually meaningless.
    """

    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()

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
                diff -= np.rint(diff)
                diff = np.dot(diff, lattice.T)
                if np.linalg.norm(diff) < symprec:
                    map_local.append(j)
                    break
        mapping.append(map_local)
    mapping = np.array(mapping)

    # Look for the symmetrically equivalent force constant tensors
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            tmp_fc = np.zeros((3, 3), dtype='double')
            for k, rot in enumerate(rotations):
                cart_rot = similarity_transformation(lattice, rot)

                # Reverse rotation of force constant is summed
                tmp_fc += similarity_transformation(cart_rot.T,
                                                    fc_bak[mapping[i, k],
                                                           mapping[j, k]])
            # Take average and set to new force cosntants
            force_constants[i, j] = tmp_fc / len(rotations)

def set_tensor_symmetry(force_constants,
                        lattice, # column vectors
                        positions,
                        symmetry):
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    map_atoms = symmetry.get_map_atoms()
    symprec = symmetry.get_symmetry_tolerance()
    cart_rot = np.array([similarity_transformation(lattice, rot)
                         for rot in rotations])

    mapa = _get_atom_indices_by_symmetry(lattice,
                                         positions,
                                         rotations,
                                         translations,
                                         symprec)
    fc_new = np.zeros_like(force_constants)
    indep_atoms = symmetry.get_independent_atoms()

    for i in indep_atoms:
        fc_combined = np.zeros(force_constants.shape[1:], dtype='double')
        num_equiv_atoms = _combine_force_constants_equivalent_atoms(
            fc_combined,
            force_constants,
            i,
            cart_rot,
            map_atoms,
            mapa)
        num_sitesym = _average_force_constants_by_sitesym(fc_new,
                                                          fc_combined,
                                                          i,
                                                          cart_rot,
                                                          mapa)

        assert num_equiv_atoms * num_sitesym == len(rotations)

    distribute_force_constants(fc_new,
                               range(len(positions)),
                               indep_atoms,
                               lattice,
                               positions,
                               rotations,
                               translations,
                               symprec)

    force_constants[:] = fc_new

def set_tensor_symmetry_PJ(force_constants,
                           lattice,
                           positions,
                           symmetry):
    """
    Full force constants are symmetrized using crystal symmetry.
    This method extracts symmetrically equivalent sets of atomic pairs and
    take sum of their force constants and average the sum.

    Since get_force_constants_disps may include crystal symmetry, this method
    is usually meaningless.
    """

    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()

    N = len(rotations)

    mapa = _get_atom_indices_by_symmetry(lattice,
                                         positions,
                                         rotations,
                                         translations,
                                         symprec)
    cart_rot = np.array([similarity_transformation(lattice, rot).T
                         for rot in rotations])
    cart_rot_inv = np.array([np.linalg.inv(rot) for rot in cart_rot])
    fcm = np.array([force_constants[mapa[n],:,:,:][:,mapa[n],:,:]
                    for n in range(N)])
    s = np.transpose(np.array([np.dot(cart_rot[n],
                                      np.dot(fcm[n], cart_rot_inv[n]))
                               for n in range(N)]), (0, 2, 3, 1, 4))
    force_constants[:] = np.array(np.average(s, axis=0),
                                  dtype='double',
                                  order='C')

def set_translational_invariance(force_constants,
                                 translational_symmetry_type=1):
    """
    Translational invariance is imposed. The type1 is quite simple
    implementation, which is just taking sum of the force constants in
    an axis and an atom index. The sum has to be zero due to the
    translational invariance. If the sum is not zero, this error is
    uniformly subtracted from force constants.
    """
    for i in range(2):
       set_translational_invariance_per_index(
           force_constants,
           index=i,
           translational_symmetry_type=translational_symmetry_type)

def set_translational_invariance_per_index(fc2,
                                           index=0,
                                           translational_symmetry_type=1):
        for i in range(fc2.shape[1 - index]):
            for j, k in list(np.ndindex(3, 3)):
                if translational_symmetry_type == 2: # Type 2
                    if index == 0:
                        fc_abs = np.abs(fc2[:, i, j, k])
                        fc_sum = np.sum(fc2[:, i, j, k])
                        fc_abs_sum = np.sum(fc_abs)
                        fc2[:, i, j, k] -= fc_sum / fc_abs_sum * fc_abs
                    else:
                        fc_abs = np.abs(fc2[i, :, j, k])
                        fc_sum = np.sum(fc2[i, :, j, k])
                        fc_abs_sum = np.sum(fc_abs)
                        fc2[i, :, j, k] -= fc_sum / fc_abs_sum * fc_abs
                else: # Type 1
                    if index == 0:
                        fc2[:, i, j, k] -= np.sum(
                            fc2[:, i, j, k]) / fc2.shape[0]
                    else:
                        fc2[i, :, j, k] -= np.sum(
                            fc2[i, :, j, k]) / fc2.shape[1]

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
    print("Check rotational invariance ...")

    fc = force_constants
    p2s = primitive.get_primitive_to_supercell_map()

    smallest_vectors, multiplicity = primitive.get_smallest_vectors()

    abc = "xyz"

    for pi, p in enumerate(p2s):
        for i in range(3):
            mat = np.zeros((3, 3), dtype='double')
            for s in range(supercell.get_number_of_atoms()):
                m = multiplicity[s, pi]
                vecs = smallest_vectors[s, pi, :m]
                v = np.dot(vecs.sum(axis=0) / m, primitive.get_cell())
                for j in range(3):
                    for k in range(3):
                        mat[j, k] += (fc[p, s, i, j] * v[k] -
                                      fc[p, s, i, k] * v[j])

            print("Atom %d %s" % (p + 1, abc[i]))
            for vec in mat:
                print("%10.5f %10.5f %10.5f" % tuple(vec))

def force_constants_log(force_constants):
    fs = force_constants
    for i, fs_i in enumerate(fs):
        for j, fs_j in enumerate(fs_i):
            for v in fs_j:
                print("force constant (%d - %d): %10.5f %10.5f %10.5f" %
                      (i + 1, j + 1, v[0], v[1], v[2]))


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))

def show_drift_force_constants(force_constants, name="force constants"):
    num_atom = force_constants.shape[0]
    maxval1 = 0
    maxval2 = 0
    jk1 = [0, 0]
    jk2 = [0, 0]
    for i, j, k in list(np.ndindex((num_atom, 3, 3))):
        val1 = force_constants[:, i, j, k].sum()
        val2 = force_constants[i, :, j, k].sum()
        if abs(val1) > abs(maxval1):
            maxval1 = val1
            jk1 = [j, k]
        if abs(val2) > abs(maxval2):
            maxval2 = val2
            jk2 = [j, k]
    print("max drift of %s: %f (%s%s) %f (%s%s)" %
          (name,
           maxval1, "xyz"[jk1[0]], "xyz"[jk1[1]],
           maxval2, "xyz"[jk2[0]], "xyz"[jk2[1]]))


#################
# Local methods #
#################
def _solve_force_constants_svd(force_constants,
                               disp_atom_number,
                               displacements,
                               sets_of_forces,
                               supercell,
                               site_symmetry,
                               symprec):
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    pos_center = positions[disp_atom_number].copy()
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(lattice,
                                                 positions,
                                                 site_symmetry,
                                                 symprec)
    site_sym_cart = [similarity_transformation(lattice, sym)
                     for sym in site_symmetry]
    rot_disps = get_rotated_displacement(displacements, site_sym_cart)
    inv_displacements = np.linalg.pinv(rot_disps)

    for i in range(supercell.get_number_of_atoms()):
        combined_forces = []
        for forces in sets_of_forces:
            combined_forces.append(
                get_rotated_forces(forces[rot_map_syms[:, i]],
                                   site_sym_cart))

        combined_forces = np.reshape(combined_forces, (-1, 3))
        force_constants[disp_atom_number, i] = -np.dot(
            inv_displacements, combined_forces)

# KL(m).
# This is very similar, but instead of using inverse displacement
# and later multiplying it by the force a linear regression is used.
# Force is "plotted" versus displacement and the slope is
# calculated, together with its standard deviation.
def _solve_force_constants_regression(force_constants,
                                      disp_atom_number,
                                      displacements,
                                      sets_of_forces,
                                      supercell,
                                      site_symmetry,
                                      symprec):
    fc_errors = np.zeros((3, 3), dtype='double')
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    pos_center = positions[disp_atom_number].copy()
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(lattice,
                                                 positions,
                                                 site_symmetry,
                                                 symprec)
    site_sym_cart = [similarity_transformation(lattice, sym)
                     for sym in site_symmetry]
    rot_disps = get_rotated_displacement(displacements, site_sym_cart)
    inv_displacements = np.linalg.pinv(rot_disps)

    for i in range(supercell.get_number_of_atoms()):
        combined_forces = []
        for forces in sets_of_forces:
            combined_forces.append(
                get_rotated_forces(forces[rot_map_syms[:, i]],
                                   site_sym_cart))

        combined_forces = np.reshape(combined_forces, (-1, 3))
        # KL(m).
        # We measure the Fi-Xj slope (linear regression), see:
# stackoverflow.com/questions/9990789/how-to-force-zero-interception-in-linear-regression
# en.wikipedia.org/wiki/Simple_linear_regression#Linear_regression_without_the_intercept_term
# http://courses.washington.edu/qsci483/Lectures/20.pdf
        for x in range(3):
            for y in range(3):
                xLin = rot_disps.T[x]
                yLin = combined_forces.T[y]
                force_constants[disp_atom_number,i,x,y] = \
                      -np.dot(xLin,yLin) / np.dot(xLin,xLin)
                if len(xLin)<=1:
                    # no chances for a fitting error, we have just one value
                    err = 0
                else:
                    variance = np.dot(yLin,yLin)/np.dot(xLin,xLin) - \
                                  force_constants[disp_atom_number,i,x,y]**2
                    if variance<0 and variance>-1e-10:
                       # in numerics, it happens. This is "numerical zero"
                       err = 0
                    else:
                       err = np.sqrt(variance) / ( len(xLin)-1 )
                fc_errors[x,y] += err

    return fc_errors

def _get_force_constants_disps(force_constants,
                               supercell,
                               dataset,
                               symmetry,
                               computation_algorithm="svd"):
    """
    Phi = -F / d
    """

    """
    Force constants are obtained by one of the following algorithm.

    svd: Singular value decomposition is used, which is equivalent to
         least square fitting.

    regression:
         The goal is to monitor the quality of computed force constants (FC).
         For that the FC spread is calculated, more precisely its standard
         deviation, i.e. sqrt(variance). Every displacement results in
         slightly different FC (due to numerical errors) -- that is why the
         FCs are spread. Such an FC 'error' is calculated separately for every
         tensor element. At the end we report their average value. We also
         report a maximum value among these tensor-elements-errors.
    """
    symprec = symmetry.get_symmetry_tolerance()
    disp_atom_list = np.unique([x['number'] for x in dataset['first_atoms']])
    for disp_atom_number in disp_atom_list:
        disps = []
        sets_of_forces = []

        for x in dataset['first_atoms']:
            if x['number'] != disp_atom_number:
                continue
            disps.append(x['displacement'])
            sets_of_forces.append(x['forces'])

        site_symmetry = symmetry.get_site_symmetry(disp_atom_number)

        fc_info = solve_force_constants(
            force_constants,
            disp_atom_number,
            disps,
            sets_of_forces,
            supercell,
            site_symmetry,
            symprec,
            computation_algorithm=computation_algorithm)

        if fc_info is not None:
            # KL(m)
            fc_errors = fc_info
            avg_len = len(disp_atom_list) * supercell.get_number_of_atoms()
            if avg_len <= 0:
                print(" (Standard deviation of the force constants not available)")
            else:
                print(" Standard deviation of the force constants, full table:")
                print(fc_errors / avg_len)
                print(" Maximal table element is %f" % (fc_errors.max() / avg_len))

    return disp_atom_list

def _distribute_fc2_part(force_constants,
                         positions,
                         atom_disp,
                         map_atom_disp,
                         lattice, # column vectors
                         r,
                         t,
                         symprec):

    # L R L^-1
    rot_cartesian = np.array(
        similarity_transformation(lattice, r), dtype='double', order='C')

    try:
        import phonopy._phonopy as phonoc
        phonoc.distribute_fc2(force_constants,
                              lattice,
                              positions,
                              atom_disp,
                              map_atom_disp,
                              rot_cartesian,
                              np.array(r, dtype='intc', order='C'),
                              np.array(t, dtype='double'),
                              symprec)
    except ImportError:
        for i, pos_i in enumerate(positions):
            rot_pos = np.dot(pos_i, r.T) + t
            rot_atom = -1
            for j, pos_j in enumerate(positions):
                diff = pos_j - rot_pos
                diff -= np.rint(diff)
                diff = np.dot(diff, lattice.T)
                if np.linalg.norm(diff) < symprec:
                    rot_atom = j
                    break

            if rot_atom < 0:
                print("Input forces are not enough to calculate force constants,")
                print("or something wrong (e.g. crystal structure does not match).")
                raise ValueError

            # R^-1 P R (inverse transformation)
            force_constants[atom_disp, i] += similarity_transformation(
                rot_cartesian.T,
                force_constants[map_atom_disp, rot_atom])

def _combine_force_constants_equivalent_atoms(fc_combined,
                                              force_constants,
                                              i,
                                              cart_rot,
                                              map_atoms,
                                              mapa):
    num_equiv_atoms = 0
    for j, k in enumerate(map_atoms):
        if k != i:
            continue

        num_equiv_atoms += 1
        r_i = (mapa[:, j] == i).nonzero()[0][0]
        for k, l in enumerate(mapa[r_i]):
            fc_combined[l] += similarity_transformation(
                cart_rot[r_i], force_constants[j, k])

    fc_combined /= num_equiv_atoms

    return num_equiv_atoms

def _average_force_constants_by_sitesym(fc_new,
                                        fc_i,
                                        i,
                                        cart_rot,
                                        mapa):
    num_sitesym = 0
    for r_i, mapa_r in enumerate(mapa):
        if mapa_r[i] != i:
            continue
        num_sitesym += 1
        for j in range(fc_i.shape[0]):
            fc_new[i, j] += similarity_transformation(
                cart_rot[r_i].T, fc_i[mapa[r_i, j]])

    fc_new[i] /= num_sitesym

    return num_sitesym

def _get_atom_indices_by_symmetry(lattice,
                                  positions,
                                  rotations,
                                  translations,
                                  symprec):
    # To understand this method, understanding numpy broadcasting is mandatory.

    K = len(positions)
    # positions[K, 3]
    # dot()[K, N, 3] where N is number of sym opts.
    # translation[N, 3] is added to the last two dimenstions after dot().
    rpos = np.dot(positions, np.transpose(rotations, (0, 2, 1))) + translations

    # np.tile(rpos, (K, 1, 1, 1))[K(2), K(1), N, 3]
    # by adding one dimension in front of [K(1), N, 3].
    # np.transpose(np.tile(rpos, (K, 1, 1, 1)), (2, 1, 0, 3))[N, K(1), K(2), 3]
    diff = positions - np.transpose(np.tile(rpos, (K, 1, 1, 1)), (2, 1, 0, 3))
    diff -= np.rint(diff)
    diff = np.dot(diff, lattice.T)
    # m[N, K(1), K(2)]
    m = (np.sqrt(np.sum(diff ** 2, axis=3)) < symprec)
    # index_array[K(1), K(2)]
    index_array = np.tile(np.arange(K, dtype='intc'), (K, 1))
    # Understanding numpy boolean array indexing (extract True elements)
    # mapa[N, K(1)]
    mapa = np.array([index_array[mr] for mr in m])
    return mapa

def _get_shortest_distance_in_PBC(pos_i, pos_j, reduced_bases):
    distances = []
    for k in (-1, 0, 1):
        for l in (-1, 0, 1):
            for m in (-1, 0, 1):
                diff = pos_j + np.array([k, l, m]) - pos_i
                distances.append(np.linalg.norm(np.dot(diff, reduced_bases)))
    return np.min(distances)

def _get_atom_mapping_by_symmetry(atom_list_done,
                                  atom_number,
                                  rotations,
                                  translations,
                                  lattice, # column vectors
                                  positions,
                                  symprec=1e-5):
    """
    Find a mapping from an atom to an atom in the atom list done.
    """

    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_number], r.T) + t
        for j in atom_list_done:
            diff = positions[j] - rot_pos
            diff -= np.rint(diff)
            if np.linalg.norm(np.dot(diff, lattice.T)) < symprec:
                return j, i

    print("Input forces are not enough to calculate force constants,")
    print("or something wrong (e.g. crystal structure does not match).")
    raise ValueError

# Compute a permutation for every space group operation.
# See '_compute_permutation_for_rotation' for more info.
#
# Output has shape (num_rot, num_pos)
def _compute_all_sg_permutations(positions, # scaled positions
                                 rotations, # scaled
                                 translations, # scaled
                                 lattice, # column vectors
                                 symprec):

    out = [] # Finally the shape is fixed as (num_sym, num_pos_of_supercell).
    for (sym, t) in zip(rotations, translations):
        rotated_positions = np.dot(positions, sym.T) + t
        out.append(_compute_permutation_for_rotation(positions,
                                                     rotated_positions,
                                                     lattice,
                                                     symprec))
    return np.array(out, dtype='intc', order='C')

# Get the overall permutation such that
#
#        positions_a[perm[i]] == positions_b[i]   (modulo the lattice)
#
# or in numpy speak,
#
#        positions_a[perm] == positions_b   (modulo the lattice)
#
# This version is optimized for the case where positions_a and positions_b
# are related by a rotation.
def _compute_permutation_for_rotation(positions_a, # scaled positions
                                      positions_b,
                                      lattice, # column vectors
                                      symprec):

    # Sort both sides by some measure which is likely to produce a small
    # maximum value of (sorted_rotated_index - sorted_original_index).
    # The C code is optimized for this case, reducing an O(n^2)
    # search down to ~O(n). (for O(n log n) work overall, including the sort)
    #
    # We choose distance from the nearest bravais lattice point as our measure.
    def sort_by_lattice_distance(fracs):
        carts = np.dot(fracs - np.rint(fracs), lattice.T)
        perm = np.argsort(np.sum(carts**2, axis=1))
        sorted_fracs = np.array(fracs[perm], dtype='double', order='C')
        return perm, sorted_fracs

    (perm_a, sorted_a) = sort_by_lattice_distance(positions_a)
    (perm_b, sorted_b) = sort_by_lattice_distance(positions_b)

    # Call the C code on our conditioned inputs.
    perm_between = _compute_permutation_c(sorted_a,
                                          sorted_b,
                                          lattice,
                                          symprec)

    # Compose all of the permutations for the full permutation.
    #
    # Note the following properties of permutation arrays:
    #
    # 1. Inverse:         if  x[perm] == y  then  x == y[argsort(perm)]
    # 2. Associativity:   x[p][q] == x[p[q]]
    return perm_a[perm_between][np.argsort(perm_b)]

# Version of '_compute_permutation_for_rotation' which just directly calls the C function,
# without any conditioning of the data.
#
# Skipping the conditioning step makes this EXTREMELY slow on large structures.
def _compute_permutation_c(positions_a, # scaled positions
                           positions_b,
                           lattice, # column vectors
                           symprec):

    permutation = np.zeros(shape=(len(positions_a),), dtype='intc')

    def permutation_error():
        raise ValueError("Input forces are not enough to calculate force constants, "
                         "or something wrong (e.g. crystal structure does not match).")

    try:
        import phonopy._phonopy as phonoc
        is_found = phonoc.compute_permutation(permutation,
                                              lattice,
                                              positions_a,
                                              positions_b,
                                              symprec)

        if not is_found:
            permutation_error()

    except ImportError:
        for i, pos_b in enumerate(positions_b):
            diffs = positions_a - pos_b
            diffs -= np.rint(diffs)
            diffs = np.dot(diffs, lattice.T)

            possible_j = np.nonzero(
                np.sqrt(np.sum(diffs**2, axis=1)) < symprec)[0]
            if len(possible_j) != 1:
                permutation_error()

            permutation[i] = possible_j[0]

        if -1 in permutation:
            permutation_error()

    return permutation

# This can be thought of as computing 'map_atom_disp' and 'map_sym' for all atoms,
# except done using permutations instead of by computing overlaps.
#
# Input:
#  * permutations, shape [num_rot][num_pos]
#  * atom_list_done
#
# Output:
#  * map_atoms, shape [num_pos].
#    Maps each atom in the full structure to its equivalent atom in atom_list_done.
#    (each entry will be an integer found in atom_list_done)
#
#  * map_syms, shape [num_pos].
#    For each atom, provides the index of a rotation that maps it into atom_list_done.
#    (there might be more than one such rotation, but only one will be returned)
#    (each entry will be an integer 0 <= i < num_rot)
def _get_sym_mappings_from_permutations(permutations,
                                        atom_list_done):
    assert permutations.ndim == 2
    num_pos = permutations.shape[1]

    # filled with -1
    map_atoms = np.zeros((num_pos,), dtype='intc') - 1
    map_syms  = np.zeros((num_pos,), dtype='intc') - 1

    atom_list_done = set(atom_list_done)
    for atom_todo in range(num_pos):
        for (sym_index, permutation) in enumerate(permutations):
            if permutation[atom_todo] in atom_list_done:
                map_atoms[atom_todo] = permutation[atom_todo]
                map_syms[atom_todo]  = sym_index
                break
        else:
            print("Input forces are not enough to calculate force constants,")
            print("or something wrong (e.g. crystal structure does not match).")
            raise ValueError

    assert set(map_atoms) & set(atom_list_done) == set(map_atoms)
    assert -1 not in map_atoms
    assert -1 not in map_syms
    return map_atoms, map_syms
