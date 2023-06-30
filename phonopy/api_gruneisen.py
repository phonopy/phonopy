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
import warnings

import numpy as np

from phonopy.gruneisen.band_structure import GruneisenBandStructure
from phonopy.gruneisen.mesh import GruneisenMesh
from phonopy.phonon.band_structure import BandStructure
from phonopy.structure.grid_points import length2mesh


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

    def run_mesh(
        self,
        mesh=100.0,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        is_gamma_center=False,
    ):
        """

        Parameters
        ----------
        mesh: array_like or float, optional
            Mesh numbers along a, b, c axes when array_like object is given.
            dtype='intc', shape=(3,)
            When float value is given, uniform mesh is generated following
            VASP convention by
                N = max(1, nint(l * |a|^*))
            where 'nint' is the function to return the nearest integer. In this
            case, it is forced to set is_gamma_center=True.
            Default value is 100.0.
        shift: array_like, optional
            Mesh shifts along a*, b*, c* axes with respect to neighboring grid
            points from the original mesh (Monkhorst-Pack or Gamma center).
            0.5 gives half grid shift. Normally 0 or 0.5 is given.
            Otherwise, q-points symmetry search is not performed.
            Default is None (no additional shift).
            dtype='double', shape=(3, )
        is_time_reversal: bool, optional
            Time reversal symmetry is considered in symmetry search. By this,
            inversion symmetry is always included. Default is True.
        is_mesh_symmetry: bool, optional
            Wheather symmetry search is done or not. Default is True
        is_gamma_center: bool, default False
            Uniform mesh grids are generated centring at Gamma point but not
            the Monkhorst-Pack scheme. When type(mesh) is float, this parameter
            setting is ignored and it is forced to set is_gamma_center=True.

        Eigenvectors will always be computed.

        Returns
        -------

        """
        for phonon in (self._phonon, self._phonon_plus, self._phonon_minus):
            if phonon.dynamical_matrix is None:
                print("Warning: Dynamical matrix has not yet built.")
                return False
        _mesh = np.array(mesh)
        mesh_nums = None
        if _mesh.shape:
            if _mesh.shape == (3,):
                mesh_nums = mesh
                _is_gamma_center = is_gamma_center
        else:
            if self._phonon._primitive_symmetry is not None:
                rots = self._phonon._primitive_symmetry.pointgroup_operations
                mesh_nums = length2mesh(
                    mesh, self._phonon._primitive.cell, rotations=rots
                )
            else:
                mesh_nums = length2mesh(mesh, self._phonon._primitive.cell)
            _is_gamma_center = True

        if mesh_nums is None:
            msg = "mesh has inappropriate type."
            raise TypeError(msg)

        symmetry = phonon.primitive_symmetry
        rotations = symmetry.pointgroup_operations
        self._mesh = GruneisenMesh(
            self._phonon.dynamical_matrix,
            self._phonon_plus.dynamical_matrix,
            self._phonon_minus.dynamical_matrix,
            mesh_nums,
            delta_strain=self._delta_strain,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_gamma_center=is_gamma_center,
            is_mesh_symmetry=is_mesh_symmetry,
            rotations=rotations,
            factor=self._phonon.unit_conversion_factor,
        )

    def set_mesh(
        self,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_gamma_center=False,
        is_mesh_symmetry=True,
    ):
        """Set sampling mesh."""
        warnings.warn(
            "PhonopyGruneisen.set_mesh() is deprecated. "
            "Use PhonopyGruneisen.run_mesh().",
            DeprecationWarning,
        )
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

    def write_yaml_mesh(self):
        """Write mesh sampling calculation results to file in yaml."""
        self._mesh.write_yaml()

    def write_hdf5_mesh(self):
        """Write mesh sampling calculation results to file in hdf5."""
        self._mesh.write_hdf5()

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

    def run_band_structure(
        self,
        paths,
        with_eigenvectors=False,
        is_band_connection=False,
        path_connections=None,
        labels=None,
    ):
        """Run phonon band structure calculation.

        Parameters
        ----------
        paths : List of array_like
            Sets of qpoints that can be passed to phonopy.set_band_structure().
            Numbers of qpoints can be different.
            shape of each array_like : (qpoints, 3)
        with_eigenvectors : bool, optional
            Flag whether eigenvectors are calculated or not. Default is False.
        is_band_connection : bool, optional
            Flag whether each band is connected or not. This is achieved by
            comparing similarity of eigenvectors of neghboring poins. Sometimes
            this fails. Default is False.
        path_connections : List of bool, optional
            This is only used in graphical plot of band structure and gives
            whether each path is connected to the next path or not,
            i.e., if False, there is a jump of q-points. Number of elements is
            the same at that of paths. Default is None.
        labels : List of str, optional
            This is only used in graphical plot of band structure and gives
            labels of end points of each path. The number of labels is equal
            to (2 - np.array(path_connections)).sum().


        """
        self._band_structure = GruneisenBandStructure(
            paths,
            self._phonon.dynamical_matrix,
            self._phonon_plus.dynamical_matrix,
            self._phonon_minus.dynamical_matrix,
            with_eigenvectors=with_eigenvectors,
            is_band_connection=is_band_connection,
            path_connections=path_connections,
            labels=labels,
            delta_strain=self._delta_strain,
            factor=self._phonon.unit_conversion_factor,
        )

    def set_band_structure(self, bands):
        """Set band structure paths."""
        warnings.warn(
            "PhonopyGruneisen.set_band_structure() is deprecated. "
            "Use PhonopyGruneisen.run_band_structure().",
            DeprecationWarning,
        )
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

    # TODO: make this more flexible
    def write_yaml_band_structure(self):
        """Write band structure calculation results to file in yaml."""
        self._band_structure.write_yaml()

    # TODO  improve plot including labels
    def plot_band_structure(self, epsilon=1e-4, color_scheme=None, legacy_plot=False):
        """Return pyplot of band structure calculation results."""
        import matplotlib.pyplot as plt

        # if legacy_plot:
        fig, axarr = plt.subplots(2, 1)
        for ax in axarr:
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in")
            ax.yaxis.set_tick_params(which="both", direction="in")
            self._band_structure.plot(axarr, epsilon=epsilon, color_scheme=color_scheme)
        return plt
        # else:
        #    pass
