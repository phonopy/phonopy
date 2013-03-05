import numpy as np


def get_fc2_reciprocal(shortest_vectors,
                       multiplicity,
                       q,
                       p2s,
                       s2p,
                       fc2,
                       symprec=1e-5):

    return get_py_fc2_reciprocal(shortest_vectors,
                                 multiplicity,
                                 q,
                                 p2s,
                                 s2p,
                                 fc2,
                                 symprec=1e-5)

# This method assumes q1+q2+q3=G and uses only q2 and q3.
#
# r2q_TI_index is only used in c code and gives which index is set as
# translational invariant. This is specified by 0, 1, or 2 and which
# correspond to the most left to right indices of atomic position,
# respectively. 0 is default.
def get_fc3_reciprocal(shortest_vectors,
                       multiplicity,
                       q_set,
                       p2s_map,
                       s2p_map,
                       fc3,
                       symprec=1e-5,
                       r2q_TI_index=0):
    try:
        import anharmonic._phono3py as phono3c
        return get_c_fc3_reciprocal(shortest_vectors,
                                    multiplicity,
                                    q_set,
                                    p2s_map,
                                    s2p_map,
                                    fc3,
                                    r2q_TI_index,
                                    symprec)
    except ImportError:
        return get_py_fc3_reciprocal(shortest_vectors,
                                     multiplicity,
                                     q_set,
                                     p2s_map,
                                     s2p_map,
                                     fc3,
                                     symprec)

def get_py_fc2_reciprocal(shortest_vectors,
                          multiplicity,
                          q,
                          p2s,
                          s2p,
                          fc2,
                          symprec=1e-5):

    num_atom_prim = len(p2s)
    fc2_rec = np.zeros((num_atom_prim, num_atom_prim, 3, 3),
                       dtype=complex)
    for i in range(num_atom_prim):
        for j in range(num_atom_prim):
            fc2_rec[i, j] = get_fc2_sum_in_supercell(
                q,
                i,
                j,
                fc2,
                shortest_vectors,
                multiplicity,
                p2s,
                s2p)

    return fc2_rec

def get_fc2_sum_in_supercell(q,
                             i,
                             j,
                             fc2,
                             shortest_vectors,
                             multiplicity,
                             p2s,
                             s2p):
    """
    i, j are the indices of atoms in primitive cell.
    """

    tensor2 = np.zeros((3, 3), dtype=complex)
    vecs = shortest_vectors
    multi = multiplicity

    # Due to translational invariance, no summation for l
    l = s2p.index(p2s[i])

    for m in range(len(s2p)):
        if not s2p[m] == p2s[j]:
            continue

        # Average phase factors for equivalent atomic pairs
        phase = np.exp(2j * np.pi * np.dot(
                vecs[m, i, :multi[m, i]], q)).sum() / multi[m, i]
        tensor2 += fc2[l, m] * phase

    return tensor2


def get_c_fc3_reciprocal(shortest_vectors,
                         multiplicity,
                         q_set,
                         p2s_map,
                         s2p_map,
                         fc3,
                         r2q_TI_index,
                         symprec=1e-5):

    import anharmonic._phono3py as phono3c

    num_atom = len(p2s_map)
    fc3_q = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3),
                     dtype=complex)
    
    phono3c.fc3_reciprocal(fc3_q,
                           shortest_vectors,
                           multiplicity,
                           q_set,
                           np.array(p2s_map),
                           np.array(s2p_map),
                           fc3,
                           r2q_TI_index,
                           symprec)

    return fc3_q

def get_py_fc3_reciprocal(shortest_vectors,
                          multiplicity,
                          q_set,
                          p2s_map,
                          s2p_map,
                          fc3,
                          symprec=1e-5):

    num_atom = len(p2s_map)
    fc3_q = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3),
                     dtype=complex)
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                fc3_q[i, j, k] = get_fc3_sum_in_supercell(
                    shortest_vectors,
                    multiplicity,
                    q_set,
                    p2s_map,
                    s2p_map,
                    i,
                    j,
                    k,
                    fc3)

        s_i = p2s_map[i]
        phase = np.exp(2j * np.pi * np.dot(shortest_vectors[s_i, 0, 0],
                                           q_set.sum(axis=0)))
        fc3_q[i, :, :] *= phase

    return fc3_q

def get_fc3_sum_in_supercell(shortest_vectors,
                             multiplicity,
                             q_set,
                             p2s_map,
                             s2p_map,
                             i,
                             j,
                             k,
                             fc3):
    """
    i, j, k are the indices of atoms in primitive cell.
    """

    tensor3 = np.zeros((3, 3, 3), dtype=complex)
    r = shortest_vectors

    # Due to translational invariance, no summation for l (1)
    l = s2p_map.index(p2s_map[i])

    for m in range(len(s2p_map)):
        if not s2p_map[m] == p2s_map[j]:
            continue

        # Average phase factors for equivalent atomic pairs (2)
        phase_m = np.exp(2j * np.pi *
                         np.dot(r[m, i, :multiplicity[m, i]],
                                q_set[1])).sum() / multiplicity[m, i]
        
        for n in range(len(s2p_map)):
            if not s2p_map[n] == p2s_map[k]:
                continue

            # Average phase factors for equivalent atomic pairs (3)
            phase_n = np.exp(2j * np.pi *
                             np.dot(r[n, i, :multiplicity[n, i]],
                                    q_set[2])).sum() / multiplicity[n, i]
            tensor3 += fc3[l, m, n] * phase_m * phase_n

    return tensor3
                            

def print_fc3_q(num_atom, fc3_q, qpoints3):
    for q_index, tensor in enumerate(fc3_q):
        for i in range(num_atom):
            for j in range(num_atom):
                for k in range(num_atom):
                    print "q1(%4.2f,%4.2f,%4.2f), q2(%4.2f,%4.2f,%4.2f), q3(%4.2f,%4.2f,%4.2f)" % tuple(qpoints3[q_index])
                    print "atom index:", i+1, j+1, k+1
                    for mat in tensor[i,j,k]*(Bohr**3/Rydberg):
                        for vec in mat:
                            print "%10.5f "*6 % (vec[0].real, vec[0].imag, 
                                                 vec[1].real, vec[1].imag, 
                                                 vec[2].real, vec[2].imag)
                    print

