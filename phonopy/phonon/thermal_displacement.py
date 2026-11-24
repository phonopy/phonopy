# SPDX-License-Identifier: BSD-3-Clause
"""Calculations of thermal displacements."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from phonopy.interface.cif import write_cif_P1
from phonopy.phonon.mesh import IterMesh, Mesh
from phonopy.phonon.thermal_properties import GammaAcousticWarning, bose_einstein_dist
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms


class ThermalDisplacementMatricesDict(TypedDict):
    """Return type of Phonopy.get_thermal_displacement_matrices_dict."""

    temperatures: NDArray[np.double] | None
    thermal_displacement_matrices: NDArray[np.double] | None
    thermal_displacement_matrices_cif: NDArray[np.double] | None


class ThermalMotion:
    """Base class of thermal displacement calculation classes."""

    def __init__(
        self,
        iter_mesh: IterMesh | Mesh,
        freq_min: float | None = None,
        freq_max: float | None = None,
        exclude_gamma_acoustic: bool = False,
    ) -> None:
        """Init method."""
        self._iter_mesh = iter_mesh
        self._exclude_gamma_acoustic = exclude_gamma_acoustic
        if freq_min is None:
            self._fmin: float = 0.0
        elif freq_min < 0:
            raise ValueError(
                "Thermal displacements are defined only for positive frequencies, "
                f"but freq_min={freq_min} was given."
            )
        else:
            self._fmin = freq_min
        if freq_max is None:
            self._fmax = None
        else:
            self._fmax = freq_max

        masses = iter_mesh.dynamical_matrix.primitive.masses
        assert masses is not None
        self._masses = masses * get_physical_units().AMU
        self._temperatures: NDArray[np.double] | None = None
        # Frequencies, shape=(modes,), and eigenvectors, shape=(bands, modes),
        # of the acoustic modes at Gamma that entered the sums.
        self._gamma_acoustic_included: (
            tuple[NDArray[np.double], NDArray[np.cdouble]] | None
        ) = None

    @property
    def exclude_gamma_acoustic(self) -> bool:
        """Return whether the acoustic modes at Gamma are excluded."""
        return self._exclude_gamma_acoustic

    def _get_valid_bands(
        self, i_q: int, freqs: NDArray[np.double], eigvecs: NDArray[np.cdouble]
    ) -> NDArray[np.bool_]:
        """Return the mask of the bands included at the i_q-th q-point.

        At Gamma, the acoustic modes that enter the sums are recorded for
        the warning.

        Parameters
        ----------
        i_q : int
            Index of the q-point in the mesh.
        freqs : ndarray
            Frequencies at the q-point. shape=(bands,), dtype='double'
        eigvecs : ndarray
            Eigenvectors at the q-point in columns.
            shape=(bands, bands), dtype='cdouble'

        Returns
        -------
        ndarray
            shape=(bands,), dtype=bool

        """
        valid = freqs > self._fmin
        if self._fmax is not None:
            valid &= freqs < self._fmax
        if i_q == self._iter_mesh.gamma_index:
            acoustic = np.argsort(np.abs(freqs))[:3]
            if self._exclude_gamma_acoustic:
                valid[acoustic] = False
            else:
                included = acoustic[valid[acoustic]]
                self._gamma_acoustic_included = (
                    freqs[included],
                    eigvecs[:, included],
                )
        return valid

    def _sum_disp_matrices(self) -> tuple[NDArray[np.cdouble], int]:
        """Return the sum of displacement matrices over the mesh.

        The sum is divided by the number of q-points to obtain the thermal
        displacement matrices.

        Returns
        -------
        disp_matrices : ndarray
            shape=(temperatures, atoms, 3, 3), dtype='cdouble'
        num_qpoints : int
            Number of q-points in the mesh.

        """
        temps = self._temperatures
        assert temps is not None
        num_qpoints = len(self._iter_mesh.qpoints)
        assert np.prod(self._iter_mesh.mesh_numbers) == num_qpoints
        disps = np.zeros((len(temps), len(self._masses), 3, 3), dtype="cdouble")
        for i_q, (freqs, eigvecs) in enumerate(self._iter_mesh):
            assert eigvecs is not None
            valid = self._get_valid_bands(i_q, freqs, eigvecs)
            disps += self._get_disp_matrices_of_modes(
                freqs[valid], eigvecs[:, valid], temps
            )
        self._warn_gamma_acoustic_included()
        return disps, num_qpoints

    def _get_disp_matrices_of_modes(
        self,
        freqs: NDArray[np.double],
        eigvecs: NDArray[np.cdouble],
        temps: NDArray[np.double],
    ) -> NDArray[np.cdouble]:
        """Return the sum of displacement matrices of the given modes.

        The matrix of atom a of mode m is Q2[m] e[a, m] e[a, m]^H / M[a],
        where e[a, m] is the three elements of the eigenvector of atom a.

        Parameters
        ----------
        freqs : ndarray
            Frequencies in THz. shape=(modes,), dtype='double'
        eigvecs : ndarray
            Eigenvectors in columns. shape=(bands, modes), dtype='cdouble'
        temps : ndarray
            Temperatures in K. shape=(temperatures,), dtype='double'

        Returns
        -------
        ndarray
            shape=(temperatures, atoms, 3, 3), dtype='cdouble'

        """
        Q2 = self._get_Q2(freqs, temps)
        num_atoms = len(self._masses)
        vecs = (
            eigvecs.T.reshape(len(freqs), num_atoms, 3) / np.sqrt(self._masses)[:, None]
        )
        outer = vecs[:, :, :, None] * vecs.conj()[:, :, None, :]
        return np.einsum("mt,maij->taij", Q2, outer)

    def _warn_gamma_acoustic_included(self) -> None:
        """Warn when acoustic modes at Gamma enter the sums, with what they add."""
        included = self._gamma_acoustic_included
        self._gamma_acoustic_included = None
        if included is None or len(included[0]) == 0:
            return
        assert self._temperatures is not None
        freqs, eigvecs = included
        t_max = float(np.max(self._temperatures))
        matrices = self._get_disp_matrices_of_modes(
            freqs, eigvecs, np.array([t_max], dtype="double")
        )[0]
        added = np.diagonal(matrices.real, axis1=1, axis2=2)
        added = added / np.prod(self._iter_mesh.mesh_numbers)
        warnings.warn(
            f"{len(freqs)} acoustic mode(s) at Gamma, at "
            + ", ".join(f"{f:.3e}" for f in freqs)
            + " THz, entered the thermal-displacement sums. Their frequencies "
            "are nearly zero, and they add up to "
            f"{added.max():.3e} Angstrom^2 to a mean square displacement at "
            f"{t_max:g} K. Set exclude_gamma_acoustic=True "
            "(EXCLUDE_GAMMA_ACOUSTIC = .TRUE., --exclude-gamma-acoustic) to "
            "exclude them.",
            GammaAcousticWarning,
            stacklevel=5,
        )

    def _get_Q2(
        self, freqs: NDArray[np.double], temps: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return hbar (n + 1/2) / omega in kg Angstrom^2.

        Parameters
        ----------
        freqs : ndarray
            Frequencies in THz. shape=(modes,), dtype='double'
        temps : ndarray
            Temperatures in K. shape=(temperatures,), dtype='double'

        Returns
        -------
        ndarray
            shape=(modes, temperatures), dtype='double'

        """
        units = get_physical_units()
        population = np.asarray(bose_einstein_dist(freqs[:, None], temps[None, :]))
        return (
            units.Hbar
            * units.EV
            / units.Angstrom**2
            * (population + 0.5)
            / (freqs[:, None] * 1e12 * 2 * np.pi)
        )

    @property
    def temperatures(self) -> NDArray[np.double] | None:
        """Setter and getter of temperatures."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures: Sequence[float] | NDArray[np.double]) -> None:
        t_array = np.array(temperatures, dtype="double")
        condition = np.logical_not(t_array < 0)
        self._temperatures = np.extract(condition, t_array).astype("double")

    def set_temperature_range(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        t_step: float | None = None,
    ) -> None:
        """Set temperatures by range."""
        if t_min is None:
            _t_min: float = 10.0
        elif t_min < 0:
            _t_min = 0.0
        else:
            _t_min = t_min

        if t_max is None:
            _t_max: float = 1000.0
        elif t_max > _t_min:
            _t_max = t_max
        else:
            _t_max = _t_min

        if t_step is None:
            _t_step: float = 10.0
        elif t_step > 0:
            _t_step = t_step
        else:
            _t_step = 10.0

        self._temperatures = np.arange(
            _t_min, _t_max + _t_step / 2.0, _t_step, dtype="double"
        )


class ThermalDisplacements(ThermalMotion):
    """Class to calculate thermal displacements (mean square displacements).

    Attributes
    ----------
    thermal_displacements : ndarray
        With `projection_direction` is `None`
            shape=(temps, natoms * 3), dtype=float
        otherwise
            shape=(temps, natoms), dtype=float

    """

    def __init__(
        self,
        iter_mesh: IterMesh | Mesh,
        projection_direction: NDArray[np.double] | Sequence[float] | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        exclude_gamma_acoustic: bool = False,
    ) -> None:
        """Init method.

        Parameters
        ----------
        iter_mesh:
            Mesh or IterMesh instance. Grid points must not be reduced by
            symmetry, i.e., IterMesh instance has to be create
            ``is_mesh_symmetry=False``.
        projection_direction:
            Eigenvector projection direction in Cartesian
            coordinates. If None, eigenvector is not projected.
        freq_min:
            Minimum phonon frequency to determine whether include or not.
        freq_max:
            Maximum phonon frequency to determine whether include or not.
        exclude_gamma_acoustic:
            If True, exclude the three modes at Gamma with the smallest
            absolute frequencies.

        """
        super().__init__(
            iter_mesh,
            freq_min=freq_min,
            freq_max=freq_max,
            exclude_gamma_acoustic=exclude_gamma_acoustic,
        )
        if projection_direction is None:
            self._projection_direction = None
        else:
            projection_direction = np.array(projection_direction, dtype="double")
            self._projection_direction = projection_direction / np.linalg.norm(
                projection_direction
            )
        self._displacements: NDArray[np.double] | None = None

    @property
    def thermal_displacements(self) -> NDArray[np.double] | None:
        """Return thermal displacements."""
        return self._displacements

    def run(self) -> None:
        """Calculate thermal displacements.

        They are the diagonal elements of the thermal displacement matrices,
        or the matrices projected along ``projection_direction``.

        """
        disp_matrices, num_qpoints = self._sum_disp_matrices()
        if self._projection_direction is None:
            disps = np.diagonal(disp_matrices.real, axis1=2, axis2=3)
            disps = disps.reshape(len(disp_matrices), -1)
        else:
            direction = self._projection_direction
            disps = np.einsum("i,taij,j->ta", direction, disp_matrices, direction).real
        self._displacements = disps / num_qpoints

    def write_yaml(
        self, filename: str | os.PathLike = "thermal_displacements.yaml"
    ) -> None:
        """Write results to file in yaml."""
        assert self._temperatures is not None
        assert self._displacements is not None
        natom = len(self._masses)
        lines = []
        lines.append("# Thermal displacements")
        lines.append("natom: %5d" % (natom))
        lines.append("freq_min: %f" % self._fmin)

        lines.append("thermal_displacements:")
        for t, u in zip(self._temperatures, self._displacements, strict=True):
            lines.append("- temperature:   %15.7f" % t)
            lines.append("  displacements:")
            for i, elems in enumerate(np.reshape(u, (natom, -1))):
                text = "  - [ %10.7f" % elems[0]
                for j in range(len(elems) - 1):
                    text += ", %10.7f" % elems[j + 1]
                text += " ] # atom %d" % (i + 1)
                lines.append(text)

        with open(filename, "w") as w:
            w.write("\n".join(lines))

    def plot(self, ax: Any, is_legend: bool = False) -> None:
        """Plot thermal displacements into matplotlib axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Single Matplotlib Axes object.
        is_legend : bool, optional
            Show legend when True. Default is False.

        """
        assert self._displacements is not None
        assert self._temperatures is not None
        xyz = ["x", "y", "z"]
        for i, u in enumerate(self._displacements.transpose()):
            ax.plot(self._temperatures, u, label=("%d-%s" % (i // 3 + 1, xyz[i % 3])))

        if is_legend:
            ax.legend(loc="upper left")


class ThermalDisplacementMatrices(ThermalMotion):
    """Class to calculate thermal displacement (mean square displacement) matrices."""

    def __init__(
        self,
        iter_mesh: IterMesh | Mesh,
        freq_min: float | None = None,
        freq_max: float | None = None,
        lattice: NDArray[np.double] | None = None,
        exclude_gamma_acoustic: bool = False,
    ) -> None:
        """Init method.

        Parameters
        ----------
        iter_mesh:
            Mesh or IterMesh instance. Grid points must not be reduced by
            symmetry, i.e., IterMesh instance has to be create
            ``is_mesh_symmetry=False``.
        freq_min: float
            Minimum phonon frequency to determine whether include or not.
        freq_max: float
            Maximum phonon frequency to determine whether include or not.
        lattice: array_like
            Lattice parameters (column vectors) in real space
            dtype='double', shape=(3, 3)
        exclude_gamma_acoustic: bool
            If True, exclude the three modes at Gamma with the smallest
            absolute frequencies.

        """
        super().__init__(
            iter_mesh,
            freq_min=freq_min,
            freq_max=freq_max,
            exclude_gamma_acoustic=exclude_gamma_acoustic,
        )
        self._disp_matrices: NDArray[np.double] | None = None
        self._disp_matrices_cif: NDArray[np.double] | None = None

        self._ANinv: NDArray[np.double] | None
        if lattice is not None:
            A = lattice
            N = np.diag([np.linalg.norm(x) for x in np.linalg.inv(A)])
            self._ANinv = np.linalg.inv(np.dot(A, N))
        else:
            self._ANinv = None

    @property
    def thermal_displacement_matrices(self) -> NDArray[np.double] | None:
        """Return thermal displacement matrices."""
        return self._disp_matrices

    @property
    def thermal_displacement_matrices_cif(self) -> NDArray[np.double] | None:
        """Return thermal displacement matrices in cif definition."""
        return self._disp_matrices_cif

    def run(self) -> None:
        """Calculate thermal displacement matrices."""
        disps, num_qpoints = self._sum_disp_matrices()
        assert (abs(disps.imag) < 1e-10).all()
        self._disp_matrices = disps.real / num_qpoints

        if self._ANinv is not None:
            self._disp_matrices_cif = np.einsum(
                "ik,takl,jl->taij", self._ANinv, self._disp_matrices, self._ANinv
            )

    def write_cif(
        self,
        cell: PhonopyAtoms,
        temperature_index: int,
        filename: str | os.PathLike = "tdispmat.cif",
    ) -> None:
        """Write results to file in P1 symmetry CIF format."""
        assert self._disp_matrices_cif is not None
        write_cif_P1(
            cell,
            U_cif=self._disp_matrices_cif[temperature_index],
            filename=filename,
        )

    def write_yaml(
        self, filename: str | os.PathLike = "thermal_displacement_matrices.yaml"
    ) -> None:
        """Write results to file in yaml."""
        assert self._temperatures is not None
        assert self._disp_matrices is not None
        natom = len(self._masses)
        lines = []

        lines.append("# Thermal displacement_matrices")
        lines.append("natom: %5d" % (natom))
        lines.append("freq_min: %f" % self._fmin)
        lines.append("thermal_displacement_matrices:")
        for i, t in enumerate(self._temperatures):
            matrices = self._disp_matrices[i]
            lines.append("- temperature:   %15.7f" % t)
            lines.append("  displacement_matrices:")
            for j, mat in enumerate(matrices):
                # For checking imaginary part that should be zero
                # lines.append("  - # atom %d" % (i + 1))
                # for v in mat:
                #     lines.append(
                #         "    [ %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f ]"
                #         % (tuple(v.real) + tuple(v.imag)))
                m = mat
                lines.append(
                    ("  - [ " + "%8.5f, " * 5 + "%8.5f ] # atom %d")
                    % (m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1], j + 1)
                )
            if self._ANinv is not None:
                assert self._disp_matrices_cif is not None
                matrices_cif = self._disp_matrices_cif[i]
                lines.append("  displacement_matrices_cif:")
                for j, mat_cif in enumerate(matrices_cif):
                    m = mat_cif
                    lines.append(
                        ("  - [ " + "%8.5f, " * 5 + "%8.5f ] # atom %d")
                        % (m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1], j + 1)
                    )

        with open(filename, "w") as w:
            w.write("\n".join(lines))
