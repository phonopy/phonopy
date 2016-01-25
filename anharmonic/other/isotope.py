# Copyright (C) 2015 Atsushi Togo
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
from phonopy.phonon.solver import set_phonon_c, set_phonon_py
from anharmonic.phonon3.triplets import get_bz_grid_address, gaussian
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phonopy.phonon.tetrahedron_mesh import get_tetrahedra_frequencies
from phonopy.units import VaspToTHz
from phonopy.structure.atoms import isotope_data

def get_mass_variances(primitive):
    symbols = primitive.get_chemical_symbols()
    mass_variances = []
    for s in symbols:
        masses = np.array([x[1] for x in isotope_data[s]])
        fractions = np.array([x[2] for x in isotope_data[s]])
        m_ave = np.dot(masses, fractions)
        g = np.dot(fractions, (1 - masses / m_ave) ** 2)
        mass_variances.append(g)

    return np.array(mass_variances, dtype='double')

class Isotope:
    def __init__(self,
                 mesh,
                 primitive,
                 mass_variances=None, # length of list is num_atom.
                 band_indices=None,
                 sigma=None,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._mesh = np.array(mesh, dtype='intc')

        if mass_variances is None:
            self._mass_variances = get_mass_variances(primitive)
        else:
            self._mass_variances = np.array(mass_variances, dtype='double')
        self._primitive = primitive
        self._band_indices = band_indices
        self._sigma = sigma
        self._symprec = symprec
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._nac_q_direction = None
        
        self._grid_address = None
        self._bz_map = None
        self._grid_points = None

        self._frequencies = None
        self._eigenvectors = None
        self._phonon_done = None
        self._dm = None
        self._band_indices = None
        self._grid_point = None
        self._gamma = None
        self._tetrahedron_method = None
        
    def set_grid_point(self, grid_point):
        self._grid_point = grid_point
        num_band = self._primitive.get_number_of_atoms() * 3
        if self._band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.array(self._band_indices, dtype='intc')
            
        self._grid_points = np.arange(np.prod(self._mesh), dtype='intc')
        
        if self._grid_address is None:
            primitive_lattice = np.linalg.inv(self._primitive.get_cell())
            self._grid_address, self._bz_map = get_bz_grid_address(
                self._mesh, primitive_lattice, with_boundary=True)
        
        if self._phonon_done is None:
            self._allocate_phonon()

    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    def run(self):
        self._run_c()

    def get_gamma(self):
        return self._gamma
        
    def get_grid_address(self):
        return self._grid_address

    def get_mass_variances(self):
        return self._mass_variances
        
    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done
    
    def set_phonons(self, frequencies, eigenvectors, phonon_done, dm=None):
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._phonon_done = phonon_done
        if dm is not None:
            self._dm = dm

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._primitive = primitive
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=self._symprec)

    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def _run_c(self):
        self._set_phonon_c(self._grid_points)
        import anharmonic._phono3py as phono3c
        gamma = np.zeros(len(self._band_indices), dtype='double')
        if self._sigma is None:
            self._set_integration_weights()
            weights = np.ones(len(self._grid_points), dtype='intc')
            phono3c.thm_isotope_strength(gamma,
                                         self._grid_point,
                                         self._grid_points,
                                         weights,
                                         self._mass_variances,
                                         self._frequencies,
                                         self._eigenvectors,
                                         self._band_indices,
                                         self._integration_weights,
                                         self._cutoff_frequency)
        else:
            phono3c.isotope_strength(gamma,
                                     self._grid_point,
                                     self._mass_variances,
                                     self._frequencies,
                                     self._eigenvectors,
                                     self._band_indices,
                                     np.prod(self._mesh),
                                     self._sigma,
                                     self._cutoff_frequency)
            
        self._gamma = gamma / np.prod(self._mesh)

    def _set_integration_weights(self):
        primitive_lattice = np.linalg.inv(self._primitive.get_cell())
        thm = TetrahedronMethod(primitive_lattice, mesh=self._mesh)
        num_grid_points = len(self._grid_points)
        num_band = self._primitive.get_number_of_atoms() * 3
        self._integration_weights = np.zeros(
            (num_grid_points, len(self._band_indices), num_band), dtype='double')
        self._set_integration_weights_c(thm)

    def _set_integration_weights_c(self, thm):
        import anharmonic._phono3py as phono3c
        unique_vertices = thm.get_unique_tetrahedra_vertices()
        neighboring_grid_points = np.zeros(
            len(unique_vertices) * len(self._grid_points), dtype='intc')
        phono3c.neighboring_grid_points(
            neighboring_grid_points,
            self._grid_points,
            unique_vertices,
            self._mesh,
            self._grid_address,
            self._bz_map)
        self._set_phonon_c(np.unique(neighboring_grid_points))
        freq_points = np.array(
            self._frequencies[self._grid_point, self._band_indices],
            dtype='double', order='C')
        phono3c.integration_weights(
            self._integration_weights,
            freq_points,
            thm.get_tetrahedra(),
            self._mesh,
            self._grid_points,
            self._frequencies,
            self._grid_address,
            self._bz_map)
        
    def _set_integration_weights_py(self, thm):
        for i, gp in enumerate(self._grid_points):
            tfreqs = get_tetrahedra_frequencies(
                gp,
                self._mesh,
                [1, self._mesh[0], self._mesh[0] * self._mesh[1]],
                self._grid_address,
                thm.get_tetrahedra(),
                self._grid_points,
                self._frequencies)
            
            for bi, frequencies in enumerate(tfreqs):
                thm.set_tetrahedra_omegas(frequencies)
                thm.run(self._frequencies[self._grid_point, self._band_indices])
                iw = thm.get_integration_weight()
                self._integration_weights[i, :, bi] = iw
                
    def _run_py(self):
        for gp in self._grid_points:
            self._set_phonon_py(gp)

        if self._sigma is None:
            self._set_integration_weights()
            
        t_inv = []
        for bi in self._band_indices:
            vec0 = self._eigenvectors[self._grid_point][:, bi].conj()
            f0 = self._frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for i, gp in enumerate(self._grid_points):
                for j, (f, vec) in enumerate(
                        zip(self._frequencies[i], self._eigenvectors[i].T)):
                    if f < self._cutoff_frequency:
                        continue
                    ti_sum_band = np.sum(
                        np.abs((vec * vec0).reshape(-1, 3).sum(axis=1)) ** 2
                        * self._mass_variances)
                    if self._sigma is None:
                        ti_sum += ti_sum_band * self._integration_weights[
                            i, bi, j]
                    else:
                        ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
            t_inv.append(np.pi / 2 / np.prod(self._mesh) * f0 ** 2 * ti_sum)

        self._gamma = np.array(t_inv, dtype='double') / 2
            
    def _set_phonon_c(self, grid_points):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
                     self._phonon_done,
                     grid_points,
                     self._grid_address,
                     self._mesh,
                     self._frequency_factor_to_THz,
                     self._nac_q_direction,
                     self._lapack_zheev_uplo)

    def _set_phonon_py(self, grid_point):
        set_phonon_py(grid_point,
                      self._phonon_done,
                      self._frequencies,
                      self._eigenvectors,
                      self._grid_address,
                      self._mesh,
                      self._dm,
                      self._frequency_factor_to_THz,                  
                      self._lapack_zheev_uplo)

    def _allocate_phonon(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid = len(self._grid_address)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        itemsize = self._frequencies.itemsize
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype=("c%d" % (itemsize * 2)))
        
