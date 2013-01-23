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
    def __init__(self,
                 eigenvalues,
                 eigenvectors,
                 weights,
                 masses,
                 factor=VaspToTHz,
                 cutoff_eigenvalue=None):

        if cutoff_eigenvalue==None:
            self._cutoff_eigenvalue = 0
        else:
            self._cutoff_eigenvalue = cutoff_eigenvalue
            
        self._distances = None
        self._displacements = None
        self._eigenvalues = eigenvalues
        self._p_eigenvectors = None
        self._eigenvectors = eigenvectors
        self._factor = factor
        self._masses = masses
        self._masses3 = np.array([[m] * 3 for m in masses]).flatten()
        self._temperatures = None
        self._weights = weights

    def _get_population(self, omega, t):
        if t < 1: # temperatue less than 1 K is approximated as 0 K.
            return 0
        else:
            return 1.0 / (np.exp(omega * THzToEv / (Kb * t)) - 1)

    def get_Q2(self, omega, t):
        return Hbar * EV / Angstrom ** 2 * (
            (self._get_population(omega, t) + 0.5) / (omega * 1e12 * 2 * np.pi))

    def set_temperature_range(self, t_min=0, t_max=1000, t_step=10):
        if t_min < 0:
            t_min = 0
        if t_step < 0:
            t_step = 0
        temps = []
        t = t_min
        while t < t_max + t_step / 2.0:
            temps.append(t)
            t += t_step
        self._temperatures = np.array(temps)

    def project_eigenvectors(self, direction, lattice=None):
        """
        direction
        
        Without specifying lattice:
          Projection direction in Cartesian coordinates
        With lattice:
          Projection direction in fractional coordinates
        """

        if not lattice==None:
            projector = np.dot(direction, lattice)
        else:
            projector = np.array(direction, dtype=float)
        projector /= np.linalg.norm(projector)
        
        self._p_eigenvectors = []
        for vecs_q in self._eigenvectors:
            p_vecs_q = []
            for vecs in vecs_q.T:
                p_vecs_q.append(np.dot(vecs.reshape(-1, 3), projector))
            self._p_eigenvectors.append(np.transpose(p_vecs_q))
        self._p_eigenvectors = np.array(self._p_eigenvectors)

class ThermalDisplacements(ThermalMotion):
    def __init__(self,
                 eigenvalues,
                 eigenvectors,
                 weights,
                 masses,
                 factor=VaspToTHz,
                 cutoff_eigenvalue=None):

        ThermalMotion.__init__(self,
                               eigenvalues,
                               eigenvectors,
                               weights,
                               masses,
                               factor=VaspToTHz,
                               cutoff_eigenvalue=None)

        self._displacements = None
        
    def get_thermal_displacements(self):
        return (self._temperatures, self._displacements)

    def set_thermal_displacements(self):
        eigvals = self._eigenvalues
        temps = self._temperatures
        weights = self._weights
        if self._p_eigenvectors is not None:
            masses = self._masses
            eigvecs = self._p_eigenvectors
        else:
            masses = self._masses3
            eigvecs = self._eigenvectors

        disps = np.zeros((len(temps), len(masses)), dtype=float)
        for eigs, vecs2, w in zip(eigvals, abs(eigvecs) ** 2, weights):
            for e, v2 in zip(eigs, vecs2.T * w):
                if e > self._cutoff_eigenvalue:
                    omega = np.sqrt(e) * self._factor # To THz
                    c = v2 / masses / AMU
                    for i, t in enumerate(temps):
                        disps[i] += self.get_Q2(omega, t) * c
            
        self._displacements = np.array(disps) / weights.sum()

    def write_yaml(self):
        natom = len(self._eigenvalues[0])/3
        f = open('thermal_displacements.yaml', 'w')
        f.write("# Thermal displacements\n")
        f.write("natom: %5d\n" % (natom))

        f.write("thermal_displacements:\n")
        for t, u in zip(self._temperatures, self._displacements):
            f.write("- temperature:   %15.7f\n" % t)
            f.write("  displacements:\n")
            for i, elems in enumerate(np.reshape(u, (natom, -1))):
                f.write("  - [ %10.7f" % elems[0])
                for j in range(len(elems) - 1):
                    f.write(", %10.7f" % elems[j + 1])
                f.write(" ] # atom %d\n" % (i + 1))
        
    def plot_thermal_displacements(self, is_legend=False):
        import matplotlib.pyplot as plt

        plots = []
        labels = []
        xyz = ['x', 'y', 'z']
        for i, u in enumerate(self._displacements.transpose()):
            plots.append(plt.plot(self._temperatures, u ))
            labels.append("%d-%s" % ( i//3 + 1, xyz[i % 3]))
        
        if is_legend:
            plt.legend(plots, labels, loc='upper left')
            
        return plt


class ThermalDistances(ThermalMotion):
    def __init__(self,
                 eigenvalues,
                 eigenvectors,
                 weights,
                 supercell,
                 primitive,
                 qpoints,
                 symprec=1e-5,
                 factor=VaspToTHz,
                 cutoff_eigenvalue=None):

        self._primitive = primitive
        self._supercell = supercell
        self._qpoints = qpoints
        self._symprec = symprec

        ThermalMotion.__init__(self,
                               eigenvalues,
                               eigenvectors,
                               weights,
                               primitive.get_masses(),
                               factor=VaspToTHz,
                               cutoff_eigenvalue=None)

    def _get_cross(self, v, delta_r, q, atom1, atom2):
        phase = np.exp(2j * np.pi * np.dot(delta_r, q))
        cross_val = v[atom1]*phase*v[atom2].conjugate()
        return -2*(cross_val).real

    def set_thermal_distances(self, atom_pairs):
        s2p = self._primitive.get_supercell_to_primitive_map()
        p2p = self._primitive.get_primitive_to_primitive_map()
        dists = np.zeros((len(self._temperatures), len(atom_pairs)), dtype=float)
        for i, (atom1, atom2) in enumerate(atom_pairs):
            patom1 = p2p[s2p[atom1]]
            patom2 = p2p[s2p[atom2]]
            delta_r = get_equivalent_smallest_vectors(atom2,
                                                      atom1,
                                                      self._supercell,
                                                      self._primitive.get_cell(),
                                                      self._symprec)[0]

            self._project_eigenvectors(delta_r, self._primitive.get_cell())
            
            for eigs, vecs, q, w in zip(self._eigenvalues,
                                        self._p_eigenvectors,
                                        self._qpoints,
                                        self._weights):
                c_cross = w / (
                    np.sqrt(self._masses[patom1] * self._masses[patom2]) * AMU)
                c1 = w / (self._masses[patom1] * AMU)
                c2 = w / (self._masses[patom2] * AMU)

                for e, v in zip(eigs, vecs.T):
                    cross_term = self._get_cross(v, delta_r, q, patom1, patom2)
                    v2 = abs(v)**2
                    if e > self._cutoff_eigenvalue:
                        omega = np.sqrt(e) * self._factor # To THz
                        for j, t in enumerate(self._temperatures):
                            dists[j, i] += self.get_Q2(omega, t) * (
                                v2[patom1] * c1 + cross_term * c_cross + v2[patom2] * c2)
            
        self._atom_pairs = atom_pairs
        self._distances = dists / self._weights.sum()
                             
    def write_yaml(self):
        natom = len(self._eigenvalues[0]) / 3
        f = open('thermal_distances.yaml', 'w')
        f.write("natom: %5d\n" % (natom))

        f.write("thermal_distances:\n")
        for t, u in zip(self._temperatures, self._distances):
            f.write("- temperature:   %15.7f\n" % t)
            f.write("  distance:\n")
            for i, (atom1, atom2) in enumerate(self._atom_pairs):
                f.write("  - %10.7f # atom pair %d-%d\n"
                        % (u[i], atom1 + 1, atom2 + 1))


