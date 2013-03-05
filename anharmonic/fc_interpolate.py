import sys
import numpy as np
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors
from anharmonic.r2q import get_fc3_reciprocal, get_fc2_reciprocal
from phonopy.structure.cells import get_supercell, Primitive, print_cell

def get_fc_interpolation( fc3,
                          fc2,
                          supercell,
                          primitive,
                          q_mesh,
                          symprec=1e-5 ):

    vecs, multi = get_smallest_vectors( supercell, primitive, symprec )
    p2s = primitive.get_primitive_to_supercell_map()
    s2p = primitive.get_supercell_to_primitive_map()
    q_mesh = np.array( q_mesh )
    mesh_points = get_mesh_points( q_mesh )

    num_atom_prim = len( p2s )
    
    supercell_intpl = get_supercell( primitive, np.diag( q_mesh ), symprec )
    primitive_intpl = Primitive( supercell_intpl, np.diag(1.0/q_mesh), symprec )
    print_cell( supercell_intpl )

    fc3_interpolation = get_fc3_interpolation( fc3,
                                               supercell_intpl,
                                               primitive_intpl,
                                               mesh_points,
                                               vecs,
                                               multi,
                                               p2s,
                                               s2p,
                                               symprec )

    fc2_interpolation = get_fc2_interpolation( fc2,
                                               supercell_intpl,
                                               primitive_intpl,
                                               mesh_points,
                                               vecs,
                                               multi,
                                               p2s,
                                               s2p,
                                               symprec )

    return fc2_interpolation.real, fc3_interpolation.real
    

def get_fc2_interpolation( fc2,
                           supercell, primitive, mesh_points,
                           vecs_orig, multi_orig, p2s_orig, s2p_orig,
                           symprec ):

    num_atom_super = supercell.get_number_of_atoms()
    num_atom_prim = primitive.get_number_of_atoms()
    
    fc2_intpl = np.zeros( ( num_atom_prim,
                            num_atom_super, 3, 3 ), dtype=complex )
    vecs, multi = get_smallest_vectors( supercell, primitive, symprec )
    s2p = primitive.get_supercell_to_primitive_map()
    p2p = primitive.get_primitive_to_primitive_map()
    s2p_special = np.array( [ p2p[x] for x in s2p ] )

    for q in mesh_points:
        fc2_rec = get_fc2_reciprocal( vecs_orig,
                                      multi_orig,
                                      q,
                                      p2s_orig,
                                      s2p_orig,
                                      fc2,
                                      symprec )
        
        fc2_intpl += get_py_fc2_realspace( fc2_rec,
                                           q,
                                           s2p, p2p,
                                           vecs, multi,
                                           symprec )

    return fc2_intpl / len( mesh_points )


def get_fc3_interpolation( fc3,
                           supercell, primitive, mesh_points,
                           vecs_orig, multi_orig, p2s_orig, s2p_orig,
                           symprec ):

    q1 = np.array([0,0,0])
    num_atom_super = supercell.get_number_of_atoms()
    num_atom_prim = primitive.get_number_of_atoms()
    
    fc3_intpl = np.zeros( ( num_atom_prim,
                            num_atom_super,
                            num_atom_super, 3, 3, 3 ), dtype=complex )
    vecs, multi = get_smallest_vectors( supercell, primitive, symprec )
    s2p = primitive.get_supercell_to_primitive_map()
    p2p = primitive.get_primitive_to_primitive_map()
    s2p_special = np.array( [ p2p[x] for x in s2p ] )

    for i, q2 in enumerate( mesh_points ):
        for j, q3 in enumerate( mesh_points ):
            sys.stdout.flush()
            print q2, q3, j+1+i*len(mesh_points), "/", len(mesh_points)**2
            fc3_rec = get_fc3_reciprocal( vecs_orig,
                                          multi_orig,
                                          np.array([ q1, q2, q3 ]),
                                          p2s_orig,
                                          s2p_orig,
                                          fc3,
                                          symprec=symprec )

            try:
                import anharmonic._phono3py as phono3c
                phono3c.fc3_realspace( fc3_intpl,
                                       vecs,
                                       multi,
                                       np.array( [ q1, q2, q3 ] ),
                                       s2p_special,
                                       fc3_rec,
                                       symprec )
            except ImportError:
                fc3_intpl += get_py_fc3_realspace( fc3_rec,
                                                   q2, q3, 
                                                   s2p_intpl, p2p_intpl,
                                                   vecs, multi,
                                                   symprec )

    return fc3_intpl / len( mesh_points ) ** 2


def get_py_fc2_realspace( fc2_rec, q, s2p, p2p, vecs, multi, symprec=1e-5 ):
    num_atom_prim = len( p2p )
    num_atom_super = len( s2p )
    fc2_intpl_q = np.zeros( ( num_atom_prim,
                              num_atom_super, 3, 3 ), dtype=complex )
    for mu in range( num_atom_prim ):
        for Nnu in range( num_atom_super ):
            phase = np.exp( -2j * np.pi * np.dot( 
                    vecs[Nnu,mu,:multi[Nnu,mu],:], q ) ).sum() / multi[Nnu,mu]
            fc2_intpl_q[mu,Nnu] = \
                fc2_rec[mu, p2p[s2p[Nnu]] ] * phase

    return fc2_intpl_q

def get_py_fc3_realspace( fc3_rec, q2, q3, s2p, p2p, vecs, multi, symprec=1e-5 ):
    num_atom_prim = len( p2p )
    num_atom_super = len( s2p )
    fc3_intpl_qset = np.zeros( ( num_atom_prim,
                                 num_atom_super,
                                 num_atom_super,
                                 3, 3, 3 ), dtype=complex )
    for mu in range( num_atom_prim ):
        for Nnu in range( num_atom_super ):
            phase2 = np.exp( -2j * np.pi * np.dot( 
                    vecs[Nnu,mu,:multi[Nnu,mu],:], q2 ) ).sum() / multi[Nnu,mu]
            for Ppi in range( num_atom_super ):
                phase3 = np.exp( -2j * np.pi * np.dot( 
                        vecs[Ppi,mu,:multi[Ppi,mu],:], q3 ) ).sum() / multi[Ppi,mu]
                fc3_intpl_qset[mu,Nnu,Ppi] = \
                    fc3_rec[mu, p2p[s2p[Nnu]], p2p[s2p[Ppi]] ] * phase2 * phase3

    return fc3_intpl_qset
            

def get_mesh_points( q_mesh ):
    mesh_points = []
    for i in range( q_mesh[0] ):
        for j in range( q_mesh[1] ):
            for k in range( q_mesh[2] ):
                mesh_points.append( [ i, j, k ] )

    mesh_points = np.array( mesh_points ) - q_mesh * ( mesh_points > q_mesh / 2 )
    mesh_points = mesh_points.astype(float) / q_mesh

    return mesh_points
    
