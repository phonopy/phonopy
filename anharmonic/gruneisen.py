import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from anharmonic.shortest_distance import get_shortest_vectors
from anharmonic.fc_interpolate import get_fc_interpolation
from phonopy.structure.cells import get_supercell, Primitive, print_cell
from anharmonic.file_IO import write_fc3_dat, write_fc2_dat
from anharmonic.fc_tools import expand_fc2, expand_fc3

class Gruneisen:
    def __init__( self,
                  fc2,
                  fc3,
                  supercell,
                  primitive,
                  mesh,
                  symprec=1e-5 ):

        self.fc2 = fc2
        self.fc3 = fc3
        self.scell = supercell
        self.pcell = primitive
        self.symprec = symprec
        self.dm = DynamicalMatrix( self.scell,
                                   self.pcell,
                                   self.fc2,
                                   self.symprec )
        self.shortest_vectors, self.multiplicity = \
            get_shortest_vectors( self.scell, self.pcell, self.symprec )

        self.X = None
        self.__set_X()

        self.dPhidu = None
        self.__set_dPhidu()

        k = [0,0,0]
        for i in range( 3, self.pcell.get_number_of_atoms()*3 ):
            g = self.get_gamma(k,i)
            print g, np.trace( g )

        # mesh = np.array(mesh)
        # fc2_intpl, fc3_intpl = get_fc_interpolation( self.fc3,
        #                                              self.fc2,
        #                                              self.scell,
        #                                              self.pcell,
        #                                              mesh,
        #                                              self.symprec )

        # self.scell = get_supercell( primitive, np.diag( mesh ), symprec )
        # self.pcell = Primitive( self.scell, np.diag( 1.0 / mesh ), symprec )
        # self.fc2 = expand_fc2( fc2_intpl, self.scell, self.pcell,
        #                        symprec=self.symprec )
        # self.fc3 = expand_fc3( fc3_intpl, self.scell, self.pcell,
        #                        symprec=self.symprec )
        # write_fc2_dat( self.fc2, "fc2_intpl-%d%d%d.dat" % tuple( mesh ) )
        # write_fc3_dat( self.fc3, "fc3_intpl-%d%d%d.dat" % tuple( mesh ) )
        # self.dm = DynamicalMatrix( self.scell,
        #                            self.pcell,
        #                            self.fc2,
        #                            self.symprec )
        
        # self.shortest_vectors, self.multiplicity = \
        #     get_shortest_vectors( self.scell, self.pcell, self.symprec )

        # self.X = None
        # self.__set_X()

        # self.dPhidu = None
        # self.__set_dPhidu()

        # k = [0,0,0]
        # for i in range( 3, self.pcell.get_number_of_atoms()*3 ):
        #     g = self.get_gamma(k,i)
        #     print g, np.trace( g )

    def get_gamma( self, q, s ):
        self.dm.set_dynamical_matrix( q )
        omega2, w = np.linalg.eigh( self.dm.get_dynamical_matrix() )
        g = np.zeros( ( 3, 3 ), dtype=float )
        num_atom_prim = self.pcell.get_number_of_atoms()
        dDdu = self.__get_dDdu( q )

        for i in range( 3 ):
            for j in range( 3 ):
                for nu in range( num_atom_prim ):
                    for pi in range( num_atom_prim ):
                        g += w[nu*3+i,s].conjugate() * \
                            dDdu[nu,pi,i,j] * w[pi*3+j,s]

        g *= -1.0/2/omega2[s]

        return g

    def __get_dDdu( self, q ):
        num_atom_prim = self.pcell.get_number_of_atoms()
        num_atom_super = self.scell.get_number_of_atoms()
        p2s = self.pcell.get_primitive_to_supercell_map()
        s2p = self.pcell.get_supercell_to_primitive_map()
        vecs = self.shortest_vectors
        multi = self.multiplicity
        m = self.pcell.get_masses()
        dPhidu = self.dPhidu
        dDdu = np.zeros( ( num_atom_prim, num_atom_prim, 3, 3, 3, 3 ), dtype=float )
        
        for nu in range( num_atom_prim ):
            for pi, p in enumerate( p2s ):
                for Ppi, s in enumerate( s2p ):
                    if not s==p:
                        continue
                    phase = np.exp( 2j * np.pi * np.dot( 
                            vecs[Ppi,nu,:multi[Ppi,nu],:], q ) ).sum() / multi[Ppi,nu]
                    dDdu[nu,pi] += phase * dPhidu[nu,Ppi]
                dDdu[nu,pi] /= np.sqrt( m[nu] * m[pi] )

        return dDdu
                                    
    def __set_dPhidu( self ):
        fc3 = self.fc3
        num_atom_prim = self.pcell.get_number_of_atoms()
        num_atom_super = self.scell.get_number_of_atoms()
        p2s = self.pcell.get_primitive_to_supercell_map()
        dPhidu = np.zeros( ( num_atom_prim, num_atom_super, 3, 3, 3, 3 ), dtype=float )

        for nu in range( num_atom_prim ):
            Y = self.__get_Y( nu )
            for pi in range( num_atom_super ):
                for i in range( 3 ):
                    for j in range( 3 ):
                        for k in range( 3 ):
                            for l in range( 3 ):
                                for m in range( 3 ):
                                    dPhidu[nu,pi,i,j,k,l] = \
                                        (fc3[p2s[nu],pi,:,i,j,:] * \
                                             Y[:,:,k,l] ).sum()
                                             # ( Y[:,:,k,l] + Y[:,:,l,k] ) / 2 ).sum() # Symmetrization?

        self.dPhidu = dPhidu

    def __get_Y( self, nu ):
        P = self.fc2
        X = self.X
        vecs = self.shortest_vectors
        multi = self.multiplicity
        lat = self.pcell.get_cell()
        num_atom_super = self.scell.get_number_of_atoms()
        R = np.array( [ np.dot(
                vecs[Npi,nu,:multi[Npi,nu],:].sum(axis=0) / multi[Npi,nu],
                lat ) for Npi in range( num_atom_super ) ] )

        p2s = self.pcell.get_primitive_to_supercell_map()
        s2p = self.pcell.get_supercell_to_primitive_map()
        p2p = self.pcell.get_primitive_to_primitive_map()

        Y = np.zeros( ( num_atom_super, 3, 3, 3 ), dtype=float )

        for Mmu in range( num_atom_super ):
            for i in range( 3 ):
                Y[ Mmu, i, i, : ] = R[ Mmu, : ]
            Y[Mmu] += X[p2p[s2p[Mmu]]]
            
        return Y

    def __set_X( self ):
        num_atom_super = self.scell.get_number_of_atoms()
        num_atom_prim = self.pcell.get_number_of_atoms()
        p2s = self.pcell.get_primitive_to_supercell_map()
        lat = self.pcell.get_cell()
        vecs = self.shortest_vectors
        multi = self.multiplicity
        X = np.zeros( ( num_atom_prim, 3, 3, 3 ), dtype=float )
        G = self.__get_Gamma()
        P = self.fc2

        for mu in range( num_atom_prim ):
            for nu in range( num_atom_prim ):
                R = np.array(
                    [ np.dot( vecs[Npi,nu,:multi[Npi,nu],:].sum(axis=0) \
                                / multi[Npi,nu], lat ) \
                          for Npi in range( num_atom_super ) ])
                for i in range( 3 ):
                    for j in range( 3 ):
                        for k in range( 3 ):
                            for l in range( 3 ):
                                X[mu,i,j,k] -= G[mu,nu,i,l] * \
                                    np.dot( P[p2s[nu],:,l,j], R[:,k] )

        self.X = X

    def __get_Gamma( self ):
        num_atom_prim = self.pcell.get_number_of_atoms()
        m = self.pcell.get_masses()
        self.dm.set_dynamical_matrix( [0,0,0] )
        vals, vecs = np.linalg.eigh( self.dm.get_dynamical_matrix().real )
        G = np.zeros( ( num_atom_prim, num_atom_prim, 3, 3 ), dtype=float )

        for pi in range( num_atom_prim ):
            for mu in range( num_atom_prim ):
                for k in range( 3 ):
                    for i in range( 3 ):
                        G[ pi, mu, k, i ] = \
                            1.0 / np.sqrt( m[pi] * m[mu] ) * \
                            ( vecs[pi*3+k,3:] * vecs[mu*3+i,3:] / vals[3:] ).sum()
        return G

            
        
