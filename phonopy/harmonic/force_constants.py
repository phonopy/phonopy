"""Force constants calculation."""
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

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    compute_permutation_for_rotation,
    get_smallest_vectors,
)


def get_force_constants(
    set_of_forces, symmetry, supercell, atom_list=None, decimals=None
):
    """Calculate force constants from disp-force dataset."""
    first_atoms = [
        {
            "number": x.get_atom_number(),
            "displacement": x.get_displacement(),
            "forces": x.get_forces(),
        }
        for x in set_of_forces
    ]
    dataset = {"natom": supercell.get_number_of_atoms(), "first_atoms": first_atoms}
    force_constants = get_fc2(supercell, symmetry, dataset, atom_list=atom_list)
    return force_constants


def get_fc2(supercell, symmetry, dataset, atom_list=None, decimals=None):
    """Force constants are computed.

    Force constants, Phi, are calculated from sets for forces, F, and
    atomic displacement, d:
      Phi = -F / d
    This is solved by matrix pseudo-inversion.
    Crystal symmetry is included when creating F and d matrices.

    Returns
    -------
    ndarray
        Force constants[ i, j, a, b ]
        i: Atom index of finitely displaced atom.
        j: Atom index at which force on the atom is measured.
        a, b: Cartesian direction indices = (0, 1, 2) for i and j, respectively
        dtype=double
        shape=(len(atom_list),n_satom,3,3),

    """
    if atom_list is None:
        fc_dim0 = len(supercell)
    else:
        fc_dim0 = len(atom_list)

    force_constants = np.zeros(
        (fc_dim0, len(supercell), 3, 3), dtype="double", order="C"
    )

    # Fill force_constants[ displaced_atoms, all_atoms_in_supercell ]
    atom_list_done = _get_force_constants_disps(
        force_constants, supercell, dataset, symmetry, atom_list=atom_list
    )

    rotations = symmetry.symmetry_operations["rotations"]
    lattice = np.array(supercell.cell.T, dtype="double", order="C")
    permutations = symmetry.atomic_permutations
    distribute_force_constants(
        force_constants,
        atom_list_done,
        lattice,
        rotations,
        permutations,
        atom_list=atom_list,
    )

    if decimals:
        force_constants = force_constants.round(decimals=decimals)

    return force_constants


def compact_fc_to_full_fc(phonon, compact_fc, log_level=0):
    """Transform compact fc to full fc."""
    fc = np.zeros(
        (compact_fc.shape[1], compact_fc.shape[1], 3, 3), dtype="double", order="C"
    )
    fc[phonon.primitive.p2s_map] = compact_fc
    distribute_force_constants_by_translations(fc, phonon.primitive, phonon.supercell)
    if log_level:
        print("Force constants were expanded to full format.")

    return fc


def cutoff_force_constants(
    force_constants, supercell, primitive, cutoff_radius, symprec=1e-5
):
    """Set zero to force constants outside of cutoff distance.

    Note
    ----
    `force_constants` is overwritten.

    """
    fc_shape = force_constants.shape

    if fc_shape[0] == fc_shape[1]:
        svecs, multi = get_smallest_vectors(
            supercell.cell,
            supercell.scaled_positions,
            supercell.scaled_positions,
            symprec=symprec,
            store_dense_svecs=primitive.store_dense_svecs,
        )
        lattice = supercell.cell
    else:
        svecs, multi = primitive.get_smallest_vectors()
        lattice = primitive.cell

    if primitive.store_dense_svecs:
        _svecs = svecs[multi[:, :, 1]]
    else:
        _svecs = svecs[:, :, 0, :]

    min_distances = np.sqrt(np.sum(np.dot(_svecs, lattice) ** 2, axis=-1))

    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            if min_distances[j, i] > cutoff_radius:
                force_constants[i, j] = 0.0


def symmetrize_force_constants(force_constants, level=1):
    """Symmetry force constants by translational and permutation symmetries.

    Note
    ----
    Schemes of symmetrization are slightly different between C and
    python implementations. If these give very different results, the
    original force constants are not reliable anyway.

    Parameters
    ----------
    force_constants: ndarray
        Force constants. Symmetrized force constants are overwritten.
        dtype=double
        shape=(n_satom,n_satom,3,3)
    primitive: Primitive
        Primitive cell
    level: int
        Controls the number of times the following steps repeated:
        1) Subtract drift force constants along row and column
        2) Average fc and fc.T

    """
    try:
        import phonopy._phonopy as phonoc

        phonoc.perm_trans_symmetrize_fc(force_constants, level)
    except ImportError:
        for i in range(level):
            set_translational_invariance(force_constants)
            set_permutation_symmetry(force_constants)
        set_translational_invariance(force_constants)


def symmetrize_compact_force_constants(force_constants, primitive, level=1):
    """Symmetry force constants by translational and permutation symmetries.

    Parameters
    ----------
    force_constants: ndarray
        Compact force constants. Symmetrized force constants are overwritten.
        dtype=double
        shape=(n_patom,n_satom,3,3)
    primitive: Primitive
        Primitive cell
    level: int
        Controls the number of times the following steps repeated:
        1) Subtract drift force constants along row and column
        2) Average fc and fc.T

    """
    s2p_map = primitive.s2p_map
    p2s_map = primitive.p2s_map
    p2p_map = primitive.p2p_map
    permutations = primitive.atomic_permutations
    s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map, p2p_map, permutations)
    try:
        import phonopy._phonopy as phonoc

        phonoc.perm_trans_symmetrize_compact_fc(
            force_constants, permutations, s2pp_map, p2s_map, nsym_list, level
        )
    except ImportError:
        text = (
            "Import error at phonoc.perm_trans_symmetrize_compact_fc. "
            "Corresponding pytono code is not implemented."
        )
        raise RuntimeError(text)


def distribute_force_constants(
    force_constants,
    atom_list_done,
    lattice,  # column vectors
    rotations,  # scaled (fractional)
    permutations,
    atom_list=None,
):
    """Fill force constants elements by symmetry."""
    map_atoms, map_syms = _get_sym_mappings_from_permutations(
        permutations, atom_list_done
    )
    rots_cartesian = np.array(
        [similarity_transformation(lattice, r) for r in rotations],
        dtype="double",
        order="C",
    )
    if atom_list is None:
        targets = np.arange(force_constants.shape[1], dtype="intc")
    else:
        targets = np.array(atom_list, dtype="intc")
    import phonopy._phonopy as phonoc

    phonoc.distribute_fc2(
        force_constants,
        targets,
        rots_cartesian,
        permutations,
        np.array(map_atoms, dtype="intc"),
        np.array(map_syms, dtype="intc"),
    )


def distribute_force_constants_by_translations(fc, primitive, supercell):
    """Distribute compact fc data to full fc by pure translations.

    For example, the input fc has to be prepared in the following way
    in advance:

    fc = np.zeros((compact_fc.shape[1], compact_fc.shape[1], 3, 3),
                  dtype='double', order='C')
    fc[primitive.p2s_map] = compact_fc

    """
    s2p = primitive.s2p_map
    p2s = primitive.p2s_map
    positions = supercell.scaled_positions
    lattice = supercell.cell.T
    diff = positions - positions[p2s[0]]
    trans = np.array(diff[np.where(s2p == p2s[0])[0]], dtype="double", order="C")
    rotations = np.array(
        [np.eye(3, dtype="intc")] * len(trans), dtype="intc", order="C"
    )
    permutations = primitive.atomic_permutations
    distribute_force_constants(fc, p2s, lattice, rotations, permutations)


def solve_force_constants(
    force_constants,
    disp_atom_number,
    displacements,
    sets_of_forces,
    supercell,
    site_symmetry,
    symprec,
    atom_list=None,
):
    """Calculate force constants elements of pairs from an atom."""
    if atom_list is None:
        fc_index = disp_atom_number
    else:
        fc_index = np.where(disp_atom_number == atom_list)[0]
        if len(fc_index) == 0:
            raise RuntimeError
        else:
            fc_index = fc_index[0]
    force_constants[fc_index] = _solve_force_constants_svd(
        disp_atom_number,
        displacements,
        sets_of_forces,
        supercell,
        site_symmetry,
        symprec,
    )
    return None


def get_positions_sent_by_rot_inv(
    lattice, positions, site_symmetry, symprec  # column vectors
):
    """Return atom indices of positions sent by inverse site symmetries.

    Rotated_positions[rot_map] == positions.

    Note
    ----
    This method is public because of being used by phono3py.

    """
    rot_map_syms = []
    for sym in site_symmetry:
        rot_map = compute_permutation_for_rotation(
            np.dot(positions, sym.T), positions, lattice, symprec
        )
        rot_map_syms.append(rot_map)

    return np.array(rot_map_syms, dtype="intc", order="C")


def get_rotated_displacement(displacements, site_sym_cart):
    """Rotate displacements by site symmetry.

    Note
    ----
    This method is public because of being used by phono3py.

    """
    rot_disps = []
    for u in displacements:
        rot_disps.append([np.dot(sym, u) for sym in site_sym_cart])
    return np.array(np.reshape(rot_disps, (-1, 3)), dtype="double", order="C")


def set_tensor_symmetry(
    force_constants, lattice, positions, symmetry  # column vectors
):
    """Full force constants are symmetrized using crystal symmetry.

    This method extracts symmetrically equivalent sets of atomic pairs and
    take sum of their force constants and average the sum.

    """
    rotations = symmetry.get_symmetry_operations()["rotations"]
    translations = symmetry.get_symmetry_operations()["translations"]
    map_atoms = symmetry.get_map_atoms()
    symprec = symmetry.tolerance
    cart_rot = np.array([similarity_transformation(lattice, rot) for rot in rotations])

    mapa = _get_atom_indices_by_symmetry(
        lattice, positions, rotations, translations, symprec
    )
    fc_new = np.zeros_like(force_constants)
    indep_atoms = symmetry.get_independent_atoms()

    for i in indep_atoms:
        fc_combined = np.zeros(force_constants.shape[1:], dtype="double")
        num_equiv_atoms = _combine_force_constants_equivalent_atoms(
            fc_combined, force_constants, i, cart_rot, map_atoms, mapa
        )
        num_sitesym = _average_force_constants_by_sitesym(
            fc_new, fc_combined, i, cart_rot, mapa
        )

        assert num_equiv_atoms * num_sitesym == len(rotations)

    permutations = symmetry.atomic_permutations
    distribute_force_constants(fc_new, indep_atoms, lattice, rotations, permutations)

    force_constants[:] = fc_new


def set_tensor_symmetry_PJ(force_constants, lattice, positions, symmetry):
    """Full force constants are symmetrized using crystal symmetry.

    This method extracts symmetrically equivalent sets of atomic pairs and
    take sum of their force constants and average the sum.

    """
    rotations = symmetry.get_symmetry_operations()["rotations"]
    translations = symmetry.get_symmetry_operations()["translations"]
    symprec = symmetry.tolerance

    N = len(rotations)

    mapa = _get_atom_indices_by_symmetry(
        lattice, positions, rotations, translations, symprec
    )
    cart_rot = np.array(
        [similarity_transformation(lattice, rot).T for rot in rotations]
    )
    cart_rot_inv = np.array([np.linalg.inv(rot) for rot in cart_rot])
    fcm = np.array(
        [force_constants[mapa[n], :, :, :][:, mapa[n], :, :] for n in range(N)]
    )
    s = np.transpose(
        np.array(
            [np.dot(cart_rot[n], np.dot(fcm[n], cart_rot_inv[n])) for n in range(N)]
        ),
        (0, 2, 3, 1, 4),
    )
    force_constants[:] = np.array(np.average(s, axis=0), dtype="double", order="C")


def set_translational_invariance(force_constants):
    """Impose translational invariance (Python version).

    The type1 is quite simple implementation, which is just taking sum of the
    force constants in an axis and an atom index. The sum has to be zero due to
    the translational invariance. If the sum is not zero, this error is
    uniformly subtracted from force constants.

    """
    for i in range(2):
        set_translational_invariance_per_index(force_constants, index=i)


def set_translational_invariance_per_index(fc2, index=0):
    """Impose translational invariance per index (Python version)."""
    for i in range(fc2.shape[1 - index]):
        for j, k in list(np.ndindex(3, 3)):
            if index == 0:
                fc2[:, i, j, k] -= np.sum(fc2[:, i, j, k]) / fc2.shape[0]
            else:
                fc2[i, :, j, k] -= np.sum(fc2[i, :, j, k]) / fc2.shape[1]


def set_permutation_symmetry(force_constants):
    """Enforce permutation symmetry to force constants (Python version).

    This is done by

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
            force_constants[i, j] = (force_constants[i, j] + fc_copy[j, i].T) / 2


def similarity_transformation(rot, mat):
    """Similarity transformation by R x M x R^-1."""
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))


def show_drift_force_constants(
    force_constants, primitive=None, name="force constants", values_only=False
):
    """Show force constants drift."""
    if force_constants.shape[0] == force_constants.shape[1]:
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
    else:
        s2p_map = primitive.s2p_map
        p2s_map = primitive.p2s_map
        p2p_map = primitive.p2p_map
        permutations = primitive.atomic_permutations
        s2pp_map, nsym_list = get_nsym_list_and_s2pp(s2p_map, p2p_map, permutations)

        try:
            import phonopy._phonopy as phonoc

            phonoc.transpose_compact_fc(
                force_constants, permutations, s2pp_map, p2s_map, nsym_list
            )
            maxval1, jk1 = _get_drift_per_index(force_constants)
            phonoc.transpose_compact_fc(
                force_constants, permutations, s2pp_map, p2s_map, nsym_list
            )
            maxval2, jk2 = _get_drift_per_index(force_constants)

        except ImportError:
            text = (
                "Import error at phonoc.tranpose_compact_fc. "
                "Corresponding python code is not implemented."
            )
            raise RuntimeError(text)

    if values_only:
        text = ""
    else:
        text = "Max drift of %s: " % name
    text += "%f (%s%s) %f (%s%s)" % (
        maxval1,
        "xyz"[jk1[0]],
        "xyz"[jk1[1]],
        maxval2,
        "xyz"[jk2[0]],
        "xyz"[jk2[1]],
    )
    print(text)


def get_nsym_list_and_s2pp(s2p_map, p2p_map, permutations):
    """Find lattice points corresponding to atoms in s2p_map.

    Parameters
    ----------
    s2p_map : array_like
        See Primitive class.
    p2p_map : dict
        See Primitive class.
    permutations : ndarray
        See Primitive.atomic_permutations.

    Returns
    -------
    s2pp : ndarray
        Atom indices in primitive cell that correspond to supercell atoms.
        shape=(num_atoms_in_supercell, ), dtype='intc'
    nsym_list : ndarray
        Pure translation indices that map atoms in supercell to those in
        primitive cell.
        shape=(num_pure_translation, ), dtype='intc'

    Note
    ----
    This method is public because of being used by phono3py.

    """
    s2pp = np.array([p2p_map[i] for i in s2p_map], dtype="intc")
    nsym_list = np.array(
        [
            np.where(permutations[:, i] == target)[0][0]
            for i, target in enumerate(s2p_map)
        ],
        dtype="intc",
    )
    return s2pp, nsym_list


def get_harmonic_potential_energy(force_constants, displacements):
    """Calculate harmonic potential energy of displacements.

    Parameters
    ----------
    force_constants : ndarray
        Full shape force constants.
        shape=(num_supercell_atoms, num_supercell_atoms, 3, 3), dtype='double'
    displacements : ndarray
        Displacements of atoms.
        shape=(num_supercell_atoms, 3) or shape=(N, num_supercell_atoms, 3),
        dtype='double'
        N in shape means number of snapshots.

    Returns
    -------
    float or list of float
        Increase of harmonic potential energy by displacements.

    Note
    ----
    This is not directly used in phonopy, but is kept useful.

    """
    if force_constants.shape[0] != force_constants.shape[1]:
        raise RuntimeError("Full shape force constants are necessary.")

    def _get_harm_pot(fc, d):
        return np.dot(d, np.dot(fc, d)) / 2

    n = force_constants.shape[0]
    fc = np.swapaxes(force_constants, 1, 2).reshape(n * 3, n * 3)
    if displacements.ndim == 3:
        return [_get_harm_pot(fc, d.ravel()) for d in displacements]
    elif displacements.ndim == 2:
        d = displacements.ravel()
        return _get_harm_pot(fc, d)
    else:
        raise RuntimeError("Array shape of displacements is wrong.")


def _get_rotated_forces(forces_syms, site_sym_cart):
    rot_forces = []
    for forces, sym_cart in zip(forces_syms, site_sym_cart):
        rot_forces.append(np.dot(forces, sym_cart.T))

    return rot_forces


def _get_drift_per_index(force_constants):
    num_atom = force_constants.shape[0]
    maxval = 0
    jk = [0, 0]
    for i, j, k in list(np.ndindex((num_atom, 3, 3))):
        val = force_constants[i, :, j, k].sum()
        if abs(val) > abs(maxval):
            maxval = val
            jk = [j, k]
    return maxval, jk


def _solve_force_constants_svd(
    disp_atom_number,
    displacements,
    sets_of_forces,
    supercell: PhonopyAtoms,
    site_symmetry,
    symprec,
):
    lattice = supercell.cell.T
    positions = supercell.scaled_positions
    pos_center = positions[disp_atom_number]
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(
        lattice, positions, site_symmetry, symprec
    )
    site_sym_cart = [similarity_transformation(lattice, sym) for sym in site_symmetry]
    rot_disps = get_rotated_displacement(displacements, site_sym_cart)
    inv_displacements = np.linalg.pinv(rot_disps)

    fc = np.zeros((len(supercell), 3, 3), dtype="double", order="C")
    for i in range(len(supercell)):
        combined_forces = []
        for forces in sets_of_forces:
            combined_forces.append(
                _get_rotated_forces(forces[rot_map_syms[:, i]], site_sym_cart)
            )

        combined_forces = np.reshape(combined_forces, (-1, 3))
        fc[i] = -np.dot(inv_displacements, combined_forces)
    return fc


def _get_force_constants_disps(
    force_constants, supercell, dataset, symmetry, atom_list=None
):
    """Calculate force constants Phi = -F / d.

    Force constants are obtained by one of the following algorithm.

    Parameters
    ----------
    force_constants: ndarray
        Force constants
        shape=(len(atom_list),n_satom,3,3)
        dtype=double
    supercell: PhonopyAtoms
        Supercell
    dataset: dict
        Distplacement dataset. Forces are also stored.
    symmetry: Symmetry
        Symmetry information of supercell
    atom_list: list
        List of atom indices corresponding to the first index of force
        constants. None assigns all atoms in supercell.

    """
    symprec = symmetry.tolerance
    disp_atom_list = np.unique([x["number"] for x in dataset["first_atoms"]])
    for disp_atom_number in disp_atom_list:
        disps = []
        sets_of_forces = []

        for x in dataset["first_atoms"]:
            if x["number"] != disp_atom_number:
                continue
            disps.append(x["displacement"])
            sets_of_forces.append(x["forces"])

        site_symmetry = symmetry.get_site_symmetry(disp_atom_number)

        solve_force_constants(
            force_constants,
            disp_atom_number,
            disps,
            sets_of_forces,
            supercell,
            site_symmetry,
            symprec,
            atom_list=atom_list,
        )

    return disp_atom_list


def _combine_force_constants_equivalent_atoms(
    fc_combined, force_constants, i, cart_rot, map_atoms, mapa
):
    num_equiv_atoms = 0
    for j, k in enumerate(map_atoms):
        if k != i:
            continue

        num_equiv_atoms += 1
        r_i = (mapa[:, j] == i).nonzero()[0][0]
        for k, l in enumerate(mapa[r_i]):
            fc_combined[l] += similarity_transformation(
                cart_rot[r_i], force_constants[j, k]
            )

    fc_combined /= num_equiv_atoms

    return num_equiv_atoms


def _average_force_constants_by_sitesym(fc_new, fc_i, i, cart_rot, mapa):
    num_sitesym = 0
    for r_i, mapa_r in enumerate(mapa):
        if mapa_r[i] != i:
            continue
        num_sitesym += 1
        for j in range(fc_i.shape[0]):
            fc_new[i, j] += similarity_transformation(
                cart_rot[r_i].T, fc_i[mapa[r_i, j]]
            )

    fc_new[i] /= num_sitesym

    return num_sitesym


def _get_atom_indices_by_symmetry(lattice, positions, rotations, translations, symprec):
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
    m = np.sqrt(np.sum(diff**2, axis=3)) < symprec
    # index_array[K(1), K(2)]
    index_array = np.tile(np.arange(K, dtype="intc"), (K, 1))
    # Understanding numpy boolean array indexing (extract True elements)
    # mapa[N, K(1)]
    mapa = np.array([index_array[mr] for mr in m])
    return mapa


def _get_shortest_distance_in_PBC(pos_i, pos_j, reduced_bases):
    distances = []
    for k in (-1, 0, 1):
        for ll in (-1, 0, 1):
            for m in (-1, 0, 1):
                diff = pos_j + np.array([k, ll, m]) - pos_i
                distances.append(np.linalg.norm(np.dot(diff, reduced_bases)))
    return np.min(distances)


def _get_sym_mappings_from_permutations(permutations, atom_list_done):
    """Find atomic indices where force constants have not yet calculated.

    This can be thought of as computing 'map_atom_disp' and 'map_sym'
    for all atoms, except done using permutations instead of by
    computing overlaps.

    Parameters
    ----------
    permutations : ndarray
        Atomic index permutation table by space group operations.
        shape=(operations, positions)
    atom_list_done : array_like
         Atomic indices where force constants (first index) were already
         calculated.

    Returns
    -------
    map_atoms : ndarray
        Maps each atom in the full structure to its equivalent atom in
        atom_list_done.  (each entry will be an integer found in
        atom_list_done)
        shape=(positions, ), dtype='intc'
    map_syms : ndarray
        For each atom, provides the index of a rotation that maps it
        into atom_list_done.  (there might be more than one such
        rotation, but only one will be returned) (each entry will be
        an integer 0 <= i < num_rot)
        shape=(positions, ), dtype='intc'

    """
    assert permutations.ndim == 2
    num_pos = permutations.shape[1]

    # filled with -1
    map_atoms = np.zeros((num_pos,), dtype="intc") - 1
    map_syms = np.zeros((num_pos,), dtype="intc") - 1

    atom_list_done = set(atom_list_done)
    for atom_todo in range(num_pos):
        for (sym_index, permutation) in enumerate(permutations):
            if permutation[atom_todo] in atom_list_done:
                map_atoms[atom_todo] = permutation[atom_todo]
                map_syms[atom_todo] = sym_index
                break
        else:
            text = (
                "Input forces are not enough to calculate force constants,"
                "or something wrong (e.g. crystal structure does not "
                "match)."
            )
            raise ValueError(text)

    assert set(map_atoms) & set(atom_list_done) == set(map_atoms)
    assert -1 not in map_atoms
    assert -1 not in map_syms
    return map_atoms, map_syms
