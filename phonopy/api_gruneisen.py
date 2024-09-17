"""API of mode Grueneisen parameter calculation."""

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

from phonopy.gruneisen.band_structure import GruneisenBandStructure
from phonopy.gruneisen.mesh import GruneisenMesh


class PhonopyGruneisen:
    """Class to calculate mode Grueneisen parameters."""

    def __init__(self, phonon, phonon_plus, phonon_minus, delta_strain=None):
        """Init method.

        Parameters
        ----------
        phonon, phonon_plus, phonon_minus : Phonopy
            Phonopy instances of the same crystal with differet volumes,
            V_0, V_0 + dV, V_0 - dV.
        delta_strain : float, optional
            Default is None, which gives dV / V_0.

        """
        self._phonon = phonon
        self._phonon_plus = phonon_plus
        self._phonon_minus = phonon_minus
        self._delta_strain = delta_strain

        self._mesh = None
        self._band_structure = None

    def get_phonon(self):
        """Return Phonopy class instance at dV=0."""
        return self._phonon

    def set_mesh(
        self,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_gamma_center=False,
        is_mesh_symmetry=True,
    ):
        """Set sampling mesh."""
        for phonon in (self._phonon, self._phonon_plus, self._phonon_minus):
            if phonon.dynamical_matrix is None:
                print("Warning: Dynamical matrix has not yet built.")
                return False

        symmetry = phonon.primitive_symmetry
        rotations = symmetry.pointgroup_operations
        self._mesh = GruneisenMesh(
            self._phonon.dynamical_matrix,
            self._phonon_plus.dynamical_matrix,
            self._phonon_minus.dynamical_matrix,
            mesh,
            delta_strain=self._delta_strain,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_gamma_center=is_gamma_center,
            is_mesh_symmetry=is_mesh_symmetry,
            rotations=rotations,
            factor=self._phonon.unit_conversion_factor,
        )
        return True

    def get_mesh(self):
        """Return mode Grueneisen parameters calculated on sampling mesh."""
        if self._mesh is None:
            return None
        else:
            return (
                self._mesh.get_qpoints(),
                self._mesh.get_weights(),
                self._mesh.get_frequencies(),
                self._mesh.get_eigenvectors(),
                self._mesh.get_gruneisen(),
            )

    def write_yaml_mesh(self, filename="gruneisen_mesh.yaml"):
        """Write mesh sampling calculation results to file in yaml."""
        self._mesh.write_yaml(filename=filename)

    def write_hdf5_mesh(self, filename="gruneisen_mesh.hdf5"):
        """Write mesh sampling calculation results to file in hdf5."""
        self._mesh.write_hdf5(filename=filename)

    def plot_mesh(
        self, cutoff_frequency=None, color_scheme=None, marker="o", markersize=None
    ):
        """Return pyplot of mesh sampling calculation results."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")
        self._mesh.plot(
            plt,
            cutoff_frequency=cutoff_frequency,
            color_scheme=color_scheme,
            marker=marker,
            markersize=markersize,
        )
        return plt

    def set_band_structure(self, bands):
        """Set band structure paths."""
        self._band_structure = GruneisenBandStructure(
            bands,
            self._phonon.dynamical_matrix,
            self._phonon_plus.dynamical_matrix,
            self._phonon_minus.dynamical_matrix,
            delta_strain=self._delta_strain,
            factor=self._phonon.unit_conversion_factor,
        )

    def get_band_structure(self):
        """Return band structure calculation results."""
        band = self._band_structure
        return (
            band.get_qpoints(),
            band.get_distances(),
            band.get_frequencies(),
            band.get_eigenvectors(),
            band.get_gruneisen(),
        )

    def write_yaml_band_structure(self, filename="gruneisen_band.yaml"):
        """Write band structure calculation results to file in yaml."""
        self._band_structure.write_yaml(filename=filename)

    def plot_band_structure(self, epsilon=1e-4, color_scheme=None):
        """Return pyplot of band structure calculation results."""
        import matplotlib.pyplot as plt

        fig, axarr = plt.subplots(2, 1)
        for ax in axarr:
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in")
            ax.yaxis.set_tick_params(which="both", direction="in")
            self._band_structure.plot(axarr, epsilon=epsilon, color_scheme=color_scheme)
        return plt
