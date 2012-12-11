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
from phonopy.units import *
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors
# np.seterr(invalid='raise')

class ThermalMotion:
    def __init__( self,
                  eigenvalues,
                  eigenvectors,
                  weights,
                  masses,
                  factor=VaspToTHz,
                  cutoff_eigenvalue=None ):

        if cutoff_eigenvalue==None:
            self.cutoff_eigenvalue = 0
        else:
            self.cutoff_eigenvalue = cutoff_eigenvalue
            
        self.distances = None
        self.displacements = None
        self.eigenvalues = eigenvalues
        self.p_eigenvectors = None
        self.eigenvectors = eigenvectors
        self.factor = factor
        self.masses = masses
        self.masses3 = np.array( [ [m] * 3 for m in masses ] ).flatten()
        self.weights = weights

    def __get_population( self, omega, t ):
        if t < 1: # temperatue less than 1 K is approximated as 0 K.
            return 0
        else:
            return 1.0 / ( np.exp( omega * THzToEv / ( Kb * t ) ) - 1 )

    def get_Q2( self, omega, t ):
        return Hbar * EV / Angstrom ** 2 * ( self.__get_population( omega, t ) + 0.5 ) / ( omega * 1e12 * 2 * np.pi )

    def set_temperature_range( self, t_min=0, t_max=1000, t_step=10 ):
        if t_min < 0:
            t_min = 0
        if t_step < 0:
            t_step = 0
        temps = []
        t = t_min
        while t < t_max + t_step / 2.0:
            temps.append( t )
            t += t_step
        self.temperatures = np.array( temps )

    def project_eigenvectors( self, direction, lattice=None ):
        """
        direction
        
        Without specifying lattice:
          Projection direction in Cartesian coordinates
        With lattice:
          Projection direction in fractional coordinates
        """
        if not lattice==None:
            projector = np.dot( direction, lattice )
        else:
            projector = np.array( direction, dtype=float )
        projector /= np.linalg.norm( projector )
        
        self.p_eigenvectors = []
        for vecs_q in self.eigenvectors:
            p_vecs_q = []
            for vecs in vecs_q.T:
                p_vecs_q.append( np.dot( vecs.reshape( -1, 3 ), projector ) )
            self.p_eigenvectors.append( np.array( p_vecs_q ).T )
        self.p_eigenvectors = np.array( self.p_eigenvectors )

class ThermalDisplacements( ThermalMotion ):
    def __init__( self,
                  eigenvalues,
                  eigenvectors,
                  weights,
                  masses,
                  factor=VaspToTHz,
                  cutoff_eigenvalue=None):

        ThermalMotion.__init__( self,
                                eigenvalues,
                                eigenvectors,
                                weights,
                                masses,
                                factor=VaspToTHz,
                                cutoff_eigenvalue=None )
        
    def set_thermal_displacements( self ):
        disps = np.zeros( ( len( self.temperatures ),
                            len( self.eigenvalues[0] ) ), dtype=float )
        for eigs, vecs2, w in zip( self.eigenvalues,
                                   abs( self.eigenvectors ) ** 2,
                                   self.weights ):
            for e, v2 in zip( eigs, vecs2.T * w ):
                if e > self.cutoff_eigenvalue:
                    omega = np.sqrt( e ) * self.factor # To THz
                    c = v2 / self.masses3 / AMU
                    for i, t in enumerate( self.temperatures ):
                        disps[i] += self.get_Q2( omega, t ) * c
            
        self.displacements = np.array( disps ) / self.weights.sum()

    def write_yaml( self ):
        natom = len( self.eigenvalues[0] )/3
        file = open('thermal_displacements.yaml', 'w')
        file.write("# Thermal displacements\n")
        file.write("natom: %5d\n" % ( natom ))

        file.write("thermal_displacements:\n")
        for t, u in zip( self.temperatures, self.displacements ):
            file.write("- temperature:   %15.7f\n" % t)
            file.write("  displacements:\n")
            
            for i in range( natom ):
                file.write("  - [ %10.7f, %10.7f, %10.7f ] # atom %d\n" 
                           % ( u[i*3], u[i*3+1], u[i*3+2], i+1 ) )
        
    def plot_thermal_displacements( self, is_legend=False ):
        import matplotlib.pyplot as plt

        plots = []
        labels = []
        xyz = [ 'x', 'y', 'z' ]
        for i, u in enumerate( self.displacements.transpose() ):
            plots.append( plt.plot( self.temperatures, u ) )
            labels.append( "%d-%s" % ( i//3 + 1, xyz[i % 3] ) )
        
        if is_legend:
            plt.legend( plots, labels, loc='upper left' )
            
        return plt


class ThermalDistances( ThermalMotion ):
    def __init__( self,
                  eigenvalues,
                  eigenvectors,
                  weights,
                  supercell,
                  primitive,
                  qpoints,
                  symprec=1e-5,
                  factor=VaspToTHz,
                  cutoff_eigenvalue=None):

        self.primitive = primitive
        self.supercell = supercell
        self.qpoints = qpoints
        self.symprec = symprec

        ThermalMotion.__init__( self,
                                eigenvalues,
                                eigenvectors,
                                weights,
                                primitive.get_masses(),
                                factor=VaspToTHz,
                                cutoff_eigenvalue=None )

    def __get_cross( self, v, delta_r, q, atom1, atom2 ):
        phase = np.exp( 2j * np.pi * np.dot( delta_r, q ) )
        cross_val = v[atom1]*phase*v[atom2].conjugate()
        return -2*(cross_val).real

    def set_thermal_distances( self, atom_pairs ):
        s2p = self.primitive.get_supercell_to_primitive_map()
        p2p = self.primitive.get_primitive_to_primitive_map()
        dists = np.zeros( ( len( self.temperatures), len( atom_pairs ) ), dtype=float )
        for i, ( atom1, atom2 ) in enumerate( atom_pairs ):
            patom1 = p2p[s2p[atom1]]
            patom2 = p2p[s2p[atom2]]
            delta_r = ( get_equivalent_smallest_vectors( atom2,
                                                         atom1,
                                                         self.supercell,
                                                         self.primitive.get_cell(),
                                                         self.symprec ) )[0]

            self.project_eigenvectors( delta_r, self.primitive.get_cell() )
            
            for eigs, vecs, q, w in zip( self.eigenvalues,
                                         self.p_eigenvectors,
                                         self.qpoints,
                                         self.weights ):
                c_cross = w / ( np.sqrt( self.masses[patom1] * self.masses[patom2] ) * AMU )
                c1 = w / ( self.masses[patom1] * AMU )
                c2 = w / ( self.masses[patom2] * AMU )

                for e, v in zip( eigs, vecs.T ):
                    cross_term = self.__get_cross( v, delta_r, q, patom1, patom2 )
                    v2 = abs(v)**2
                    if e > self.cutoff_eigenvalue:
                        omega = np.sqrt( e ) * self.factor # To THz
                        for j, t in enumerate( self.temperatures ):
                            dists[j,i] += self.get_Q2( omega, t ) * ( v2[patom1] * c1 + cross_term * c_cross + v2[patom2] * c2 )
            
        self.atom_pairs = atom_pairs
        self.distances = dists / self.weights.sum()
                             
    def write_yaml( self ):
        natom = len( self.eigenvalues[0] )/3
        file = open('thermal_distances.yaml', 'w')
        file.write("natom: %5d\n" % ( natom ))

        file.write("thermal_distances:\n")
        for t, u in zip( self.temperatures, self.distances ):
            file.write("- temperature:   %15.7f\n" % t)
            file.write("  distance:\n")
            for i, (atom1, atom2) in enumerate( self.atom_pairs ):
                file.write("  - %10.7f # atom pair %d-%d\n"
                           % ( u[i], atom1+1, atom2+1 ) )


