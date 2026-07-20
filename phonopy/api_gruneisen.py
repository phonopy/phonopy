# SPDX-License-Identifier: BSD-3-Clause
"""API of mode Grueneisen parameter calculation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phonopy.api_phonopy import Phonopy
from phonopy.gruneisen.band_structure import GruneisenBandStructure
from phonopy.gruneisen.mesh import GruneisenMesh


class PhonopyGruneisen:
    """Class to calculate mode Grueneisen parameters."""

    def __init__(
        self,
        phonon: Phonopy,
        phonon_plus: Phonopy,
        phonon_minus: Phonopy,
        delta_strain: float | None = None,
    ) -> None:
        """Init method.

        Parameters
        ----------
        phonon, phonon_plus, phonon_minus : Phonopy
            Phonopy instances of the same crystal with different volumes,
            V_0, V_0 + dV, V_0 - dV.
        delta_strain : float, optional
            Default is None, which gives dV / V_0.

        """
        self._phonon = phonon
        self._phonon_plus = phonon_plus
        self._phonon_minus = phonon_minus
        self._delta_strain = delta_strain

        self._mesh: GruneisenMesh | None = None
        self._band_structure: GruneisenBandStructure | None = None

    def get_phonon(self) -> Phonopy:
        """Return Phonopy class instance at dV=0."""
        return self._phonon

    def set_mesh(
        self,
        mesh: Sequence[int] | NDArray[np.int64],
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_gamma_center: bool = False,
        is_mesh_symmetry: bool = True,
    ) -> bool:
        """Set sampling mesh."""
        for phonon in (self._phonon, self._phonon_plus, self._phonon_minus):
            if phonon.dynamical_matrix is None:
                print("Warning: Dynamical matrix has not yet built.")
                return False

        assert self._phonon.dynamical_matrix is not None
        assert self._phonon_plus.dynamical_matrix is not None
        assert self._phonon_minus.dynamical_matrix is not None
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
            primitive_symmetry=phonon.primitive_symmetry,
            factor=self._phonon.unit_conversion_factor,
        )
        return True

    def get_mesh(
        self,
    ) -> (
        tuple[
            NDArray[np.double],
            NDArray[np.int64],
            NDArray[np.double],
            NDArray[np.cdouble],
            NDArray[np.double],
        ]
        | None
    ):
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

    def write_yaml_mesh(
        self, filename: str | os.PathLike = "gruneisen_mesh.yaml"
    ) -> None:
        """Write mesh sampling calculation results to file in yaml."""
        if self._mesh is None:
            raise RuntimeError("Mesh has not been set.")
        self._mesh.write_yaml(filename=filename)

    def write_hdf5_mesh(
        self, filename: str | os.PathLike = "gruneisen_mesh.hdf5"
    ) -> None:
        """Write mesh sampling calculation results to file in hdf5."""
        if self._mesh is None:
            raise RuntimeError("Mesh has not been set.")

        self._mesh.write_hdf5(filename=filename)

    def plot_mesh(
        self,
        cutoff_frequency: float | None = None,
        color_scheme: str | None = None,
        marker: str = "o",
        markersize: float | None = None,
    ) -> Any:
        """Return pyplot of mesh sampling calculation results."""
        import matplotlib.pyplot as plt

        if self._mesh is None:
            raise RuntimeError("Mesh has not been set.")

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

    def set_band_structure(
        self,
        bands: Sequence[Sequence[Sequence[float]]] | Sequence[NDArray[np.double]],
    ) -> None:
        """Set band structure paths."""
        assert self._phonon.dynamical_matrix is not None
        assert self._phonon_plus.dynamical_matrix is not None
        assert self._phonon_minus.dynamical_matrix is not None
        self._band_structure = GruneisenBandStructure(
            bands,
            self._phonon.dynamical_matrix,
            self._phonon_plus.dynamical_matrix,
            self._phonon_minus.dynamical_matrix,
            delta_strain=self._delta_strain,
            factor=self._phonon.unit_conversion_factor,
        )

    def get_band_structure(
        self,
    ) -> tuple[
        list[NDArray[np.double]],
        list[NDArray[np.double]],
        list[NDArray[np.double]],
        list[NDArray[np.cdouble]],
        list[NDArray[np.double]],
    ]:
        """Return band structure calculation results."""
        self._assert_band_structure()
        assert self._band_structure is not None

        band = self._band_structure
        return (
            band.get_qpoints(),
            band.get_distances(),
            band.get_frequencies(),
            band.get_eigenvectors(),
            band.get_gruneisen(),
        )

    def write_yaml_band_structure(
        self, filename: str | os.PathLike = "gruneisen_band.yaml"
    ) -> None:
        """Write band structure calculation results to file in yaml."""
        self._assert_band_structure()
        assert self._band_structure is not None
        self._band_structure.write_yaml(filename=filename)

    def plot_band_structure(
        self, epsilon: float | None = 1e-4, color_scheme: str | None = None
    ) -> Any:
        """Return pyplot of band structure calculation results."""
        import matplotlib.pyplot as plt

        self._assert_band_structure()
        assert self._band_structure is not None

        fig, axarr = plt.subplots(2, 1)
        for ax in axarr:
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in")
            ax.yaxis.set_tick_params(which="both", direction="in")
            self._band_structure.plot(axarr, epsilon=epsilon, color_scheme=color_scheme)
        return plt

    def _assert_band_structure(self) -> None:
        if self._band_structure is None:
            raise RuntimeError("Band structure has not been set.")
