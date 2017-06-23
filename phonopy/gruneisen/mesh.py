# Copyright (C) 2012 Atsushi Togo
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
from phonopy.structure.grid_points import get_qpoints
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.gruneisen import Gruneisen
from phonopy.units import THzToEv

class Mesh(object):
    def __init__(self,
                 phonon,
                 phonon_plus,
                 phonon_minus,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_gamma_center=False,
                 is_mesh_symmetry=True):
        self._phonon = phonon
        self._mesh = np.array(mesh, dtype='intc')
        self._factor = phonon.get_unit_conversion_factor(),
        primitive = phonon.get_primitive()
        primitive_symmetry = phonon.get_primitive_symmetry()
        gruneisen = Gruneisen(phonon.get_dynamical_matrix(),
                              phonon_plus.get_dynamical_matrix(),
                              phonon_minus.get_dynamical_matrix())
        self._qpoints, self._weights = get_qpoints(
            self._mesh,
            np.linalg.inv(primitive.get_cell()),
            q_mesh_shift=shift,
            is_time_reversal=is_time_reversal,
            is_gamma_center=is_gamma_center,
            rotations=primitive_symmetry.get_pointgroup_operations(),
            is_mesh_symmetry=is_mesh_symmetry)
        gruneisen.set_qpoints(self._qpoints)
        self._gamma = gruneisen.get_gruneisen()
        self._gamma_prime = gruneisen.get_gamma_prime()
        self._eigenvalues = gruneisen.get_eigenvalues()
        self._frequencies = np.sqrt(
            abs(self._eigenvalues)) * np.sign(self._eigenvalues) * self._factor

    def get_gruneisen(self):
        return self._gamma

    def get_gamma_prime(self):
        return self._gamma_prime

    def get_mesh_numbers(self):
        return self._mesh

    def get_phonon(self):
        return self._phonon

    def get_qpoints(self):
        return self._qpoints

    def get_weights(self):
        return self._weights

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_frequencies(self):
        return self._frequencies

    def get_eigenvectors(self):
        """
        See the detail of array shape in phonopy.phonon.mesh.
        """
        return self._eigenvectors


    def write_yaml(self, filename="gruneisen.yaml"):
        f = open(filename, 'w')
        f.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
        f.write("nqpoint: %d\n" % len(self._qpoints))
        f.write("phonon:\n")
        for q, w, gs, freqs in zip(self._qpoints,
                                   self._weights,
                                   self._gamma,
                                   self._frequencies):
            f.write("- q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(q))
            f.write("  multiplicity: %d\n" % w)
            f.write("  band:\n")
            for j, (g, freq) in enumerate(zip(gs, freqs)):
                f.write("  - # %d\n" % (j + 1))
                f.write("    gruneisen: %15.10f\n" % g)
                f.write("    frequency: %15.10f\n" % freq)
            f.write("\n")
        f.close()

    def write_hdf5(self, filename="gruneisen.hdf5"):
        import h5py
        w = h5py.File(filename, 'w')
        w.create_dataset('mesh', data=self._mesh)
        w.create_dataset('gruneisen', data=self._gamma)
        w.create_dataset('weight', data=self._weights)
        w.create_dataset('frequency', data=self._frequencies)
        w.create_dataset('qpoint', data=self._qpoints)
        w.close()

    def plot(self,
             cutoff_frequency=None,
             color_scheme=None,
             marker='o',
             markersize=None):
        import matplotlib.pyplot as plt
        n = len(self._gamma.T) - 1
        for i, (g, freqs) in enumerate(zip(self._gamma.T,
                                           self._frequencies.T)):
            if cutoff_frequency:
                g = np.extract(freqs > cutoff_frequency, g)
                freqs = np.extract(freqs > cutoff_frequency, freqs)

            if color_scheme == 'RB':
                color = (1. / n * i, 0, 1./ n * (n - i))
                if markersize:
                    plt.plot(freqs, g, marker,
                             color=color, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker, color=color)
            elif color_scheme == 'RG':
                color = (1. / n * i, 1./ n * (n - i), 0)
                if markersize:
                    plt.plot(freqs, g, marker,
                             color=color, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker, color=color)
            elif color_scheme == 'RGB':
                color = (max(2./ n * (i - n / 2.), 0),
                         min(2./ n * i, 2./ n * (n - i)),
                         max(2./ n * (n / 2. - i), 0))
                if markersize:
                    plt.plot(freqs, g, marker,
                             color=color, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker, color=color)
            else:
                if markersize:
                    plt.plot(freqs, g, marker, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker)

        return plt


def get_thermodynamic_Gruneisen_parameter(gammas,
                                          frequencies,
                                          multiplicities,
                                          t):
    if t > 0:
        conditions = (frequencies > 0)
        freq_temp = np.where(conditions, x, 1)
        cv_temp = mode_cv(t, frequencies * THzToEv)
        cv = np.where(conditions, x, 0)
        return (np.dot(multiplicities, cv * gammas).sum() /
                np.dot(multiplicities, cv).sum())
    else:
        return 0.

def get_thermal_expansion_coefficient(gammas,
                                      frequencies,
                                      multiplicities,
                                      t):
    if t > 0:
        return np.dot(multiplicities,
                      mode_cv(t, frequencies * THzToEv) * gammas).sum()
    else:
        return 0.
