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

from phonopy.gruneisen import GruneisenMesh
from phonopy.gruneisen import GruneisenBandStructure
from phonopy.gruneisen import GruneisenThermalProperties

class PhonopyGruneisen(object):
    def __init__(self,
                 phonon,
                 phonon_plus,
                 phonon_minus):
        self._phonon = phonon
        self._phonon_plus = phonon_plus
        self._phonon_minus = phonon_minus

        self._mesh = None
        self._band_structure = None
        self._thermal_properties = None

    def get_phonon(self):
        return self._phonon

    def set_mesh(self,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_gamma_center=False,
                 is_mesh_symmetry=True):
        for phonon in (self._phonon, self._phonon_plus, self._phonon_minus):
            if phonon.get_dynamical_matrix() is None:
                print("Warning: Dynamical matrix has not yet built.")
                return False

        symmetry = phonon.get_primitive_symmetry()
        rotations = symmetry.get_pointgroup_operations()
        self._mesh = GruneisenMesh(
            self._phonon.get_dynamical_matrix(),
            self._phonon_plus.get_dynamical_matrix(),
            self._phonon_minus.get_dynamical_matrix(),
            mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_gamma_center=is_gamma_center,
            is_mesh_symmetry=is_mesh_symmetry,
            rotations=rotations,
            factor=self._phonon.get_unit_conversion_factor())
        return True

    def get_mesh(self):
        if self._mesh is None:
            return None
        else:
            return (self._mesh.get_qpoints(),
                    self._mesh.get_weights(),
                    self._mesh.get_frequencies(),
                    self._mesh.get_eigenvectors(),
                    self._mesh.get_gruneisen())

    def write_yaml_mesh(self):
        self._mesh.write_yaml()

    def write_hdf5_mesh(self):
        self._mesh.write_hdf5()

    def plot_mesh(self,
                  cutoff_frequency=None,
                  color_scheme=None,
                  marker='o',
                  markersize=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        self._mesh.plot(plt,
                        cutoff_frequency=cutoff_frequency,
                        color_scheme=color_scheme,
                        marker=marker,
                        markersize=markersize)
        return plt

    def set_band_structure(self, bands):
        self._band_structure = GruneisenBandStructure(
            bands,
            self._phonon.get_dynamical_matrix(),
            self._phonon_plus.get_dynamical_matrix(),
            self._phonon_minus.get_dynamical_matrix(),
            factor=self._phonon.get_unit_conversion_factor())

    def get_band_structure(self):
        band = self._band_structure
        return (band.get_qpoints(),
                band.get_distances(),
                band.get_frequencies(),
                band.get_eigenvectors(),
                band.get_gruneisen())

    def write_yaml_band_structure(self):
        self._band_structure.write_yaml()

    def plot_band_structure(self,
                            epsilon=1e-4,
                            color_scheme=None):
        import matplotlib.pyplot as plt
        fig, axarr = plt.subplots(2, 1)
        for ax in axarr:
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_tick_params(which='both', direction='in')
            ax.yaxis.set_tick_params(which='both', direction='in')
            self._band_structure.plot(axarr,
                                      epsilon=epsilon,
                                      color_scheme=color_scheme)
        return plt

    def set_thermal_properties(self,
                               volumes,
                               t_step=2,
                               t_max=2004,
                               t_min=0,
                               cutoff_frequency=None):
        self._thermal_properties  = GruneisenThermalProperties(
            self._mesh,
            volumes,
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            cutoff_frequency=cutoff_frequency)

    def get_thermal_properties(self):
        return self._thermal_properties

    def write_yaml_thermal_properties(self, filename='thermal_properties'):
        self._thermal_properties.write_yaml(filename=filename)
