"""Calculation of density of states."""

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

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from phonopy.phonon.grid import BZGrid
from phonopy.phonon.mesh import Mesh
from phonopy.phonon.spectrum import TetrahedronDOSAccumulator


class TotalDosDict(TypedDict):
    """Return type of Phonopy.get_total_dos_dict."""

    frequency_points: NDArray[np.double]
    total_dos: NDArray[np.double]


class ProjectedDosDict(TypedDict):
    """Return type of Phonopy.get_projected_dos_dict."""

    frequency_points: NDArray[np.double]
    projected_dos: NDArray[np.double] | None


class NormalDistribution:
    """Class to represent normal distribution."""

    def __init__(self, sigma: float) -> None:
        """Init method."""
        self._sigma = sigma

    def calc(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Return normal distribution."""
        return (
            1.0
            / np.sqrt(2 * np.pi)
            / self._sigma
            * np.exp(-(x**2) / 2.0 / self._sigma**2)
        )


class CauchyDistribution:
    """Class to represent Cauchy distribution."""

    def __init__(self, gamma: float) -> None:
        """Init method."""
        self._gamma = gamma

    def calc(self, x: NDArray[np.double]) -> NDArray[np.double]:
        """Return Cauchy distribution."""
        return self._gamma / np.pi / (x**2 + self._gamma**2)


class Dos:
    """Base class to calculate density of states."""

    def __init__(
        self,
        mesh_object: Mesh,
        sigma: float | None = None,
        use_tetrahedron_method: bool = False,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        mesh_object : Mesh
            Mesh object. IterMesh object cannot be used currently since
            pre-computed frequencies and weights are used.
        sigma : float, optional
            Sigma for smearing method. When set, the smearing method is
            used regardless of ``use_tetrahedron_method``.
        lang : {"C", "Rust"}, optional
            Backend selector for the tetrahedron-method kernels.  Default
            is "C".

        """
        self._mesh_object = mesh_object
        self._frequencies = mesh_object.frequencies
        self._weights = mesh_object.weights
        # A given sigma selects the smearing method; the tetrahedron
        # method applies only without sigma.
        self._use_tetrahedron_method = use_tetrahedron_method and sigma is None
        self._frequency_points: NDArray[np.double]
        self._sigma = sigma
        self._lang: Literal["C", "Rust"] = lang

        if self._use_tetrahedron_method:
            self.set_draw_area()
        else:
            self._sigma = self.set_draw_area()
            self.set_smearing_function("Normal")

    @property
    def frequency_points(self) -> NDArray[np.double]:
        """Return frequency points."""
        return self._frequency_points

    def set_smearing_function(self, function_name: Literal["Normal", "Cauchy"]) -> None:
        """Set function form for smearing method.

        Parameters
        ----------
        function_name : str
            'Normal': smearing is done by normal distribution.
            'Cauchy': smearing is done by Cauchy distribution.

        """
        assert self._sigma is not None
        if function_name == "Cauchy":
            self._smearing_function = CauchyDistribution(self._sigma)
        else:
            self._smearing_function = NormalDistribution(self._sigma)

    def set_sigma(self, sigma: float) -> None:
        """Set sigma."""
        self._sigma = sigma

    def set_draw_area(
        self,
        freq_min: float | None = None,
        freq_max: float | None = None,
        freq_pitch: float | None = None,
    ) -> float:
        """Set frequency points."""
        f_min = self._frequencies.min()
        f_max = self._frequencies.max()

        sigma = self._sigma
        if sigma is None:
            sigma = (f_max - f_min) / 100.0

        if freq_min is None:
            f_min -= sigma * 10
        else:
            f_min = freq_min

        if freq_max is None:
            f_max += sigma * 10
        else:
            f_max = freq_max

        if freq_pitch is None:
            f_delta = (f_max - f_min) / 200.0
        else:
            f_delta = freq_pitch
        self._frequency_points = np.arange(
            f_min, f_max + f_delta * 0.1, f_delta, dtype="double"
        )

        return sigma


def _bzgrid_and_full_grid_frequencies(
    mesh_object: Mesh,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[BZGrid, NDArray[np.double]]:
    """Return a BZGrid covering every regular-grid point with frequencies on it.

    The Mesh stores ``frequencies`` only at ir-grid points but the Mesh's
    symmetry resolution may differ from BZGrid's (e.g. NAC, slightly different
    rotation reduction).  To avoid an alignment headache the BZGrid here is
    built without point-group reduction; the per-mode frequencies are
    replicated from the Mesh's ir-grid via ``grid_mapping_table`` so every GR
    grid point gets its symmetry-equivalent frequency.  Numerically equivalent
    to the legacy ``tetrahedron_method_dos`` C kernel which also iterated over
    all grid points.

    """
    bzgrid = BZGrid(
        mesh_object.mesh_numbers,
        lattice=mesh_object.dynamical_matrix.primitive.cell,
        is_shift=mesh_object.is_shift,
        is_time_reversal=False,
        lang=lang,
    )
    ir_position = {int(gp): i for i, gp in enumerate(mesh_object.ir_grid_points)}
    positions = np.array(
        [ir_position[int(gp)] for gp in mesh_object.grid_mapping_table],
        dtype="int64",
    )
    frequencies_full = mesh_object.frequencies[positions]
    return bzgrid, frequencies_full


class TotalDos(Dos):
    """Class to calculate total DOS."""

    def __init__(
        self,
        mesh_object: Mesh,
        sigma: float | None = None,
        use_tetrahedron_method: bool = False,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method."""
        super().__init__(
            mesh_object,
            sigma=sigma,
            use_tetrahedron_method=use_tetrahedron_method,
            lang=lang,
        )
        self._dos: NDArray[np.double] | None = None
        self._freq_Debye: float | None = None
        self._Debye_fit_coef = None

    def run(self) -> None:
        """Calculate total DOS."""
        if self._use_tetrahedron_method:
            self._run_tetrahedron_method_dos()
        else:
            self._dos = np.array(
                [self._get_density_of_states_at_freq(f) for f in self._frequency_points]
            )

    @property
    def dos(self) -> NDArray[np.double] | None:
        """Return total DOS."""
        return self._dos

    def get_Debye_frequency(self) -> float | None:
        """Return a kind of Debye frequency."""
        return self._freq_Debye

    def set_Debye_frequency(
        self, num_atoms: int, freq_max_fit: float | None = None
    ) -> None:
        """Calculate a kind of Debye frequency."""
        try:
            from scipy.optimize import curve_fit
        except ImportError as exc:
            raise ModuleNotFoundError("You need to install python-scipy.") from exc

        if self._dos is None:
            raise RuntimeError("Run total DOS calculation first.")

        def Debye_dos(freq, a):
            return a * freq**2

        freq_min = self._frequency_points.min()
        freq_max = self._frequency_points.max()

        if freq_max_fit is None:
            N_fit = int(len(self._frequency_points) / 4.0)  # Hard coded
        else:
            N_fit = int(
                freq_max_fit / (freq_max - freq_min) * len(self._frequency_points)
            )
        popt, pcov = curve_fit(
            Debye_dos, self._frequency_points[0:N_fit], self._dos[0:N_fit]
        )
        a2 = popt[0]
        self._freq_Debye = (3 * 3 * num_atoms / a2) ** (1.0 / 3)
        self._Debye_fit_coef = a2

    def plot(
        self,
        ax,
        xlabel: str | None = None,
        ylabel: str | None = None,
        draw_grid: bool = True,
        flip_xy: bool = False,
    ) -> None:
        """Plot total DOS."""
        if self._dos is None:
            raise RuntimeError("Run total DOS calculation first.")

        if flip_xy:
            _xlabel = "Density of states"
            _ylabel = "Frequency"
        else:
            _xlabel = "Frequency"
            _ylabel = "Density of states"

        if xlabel is not None:
            _xlabel = xlabel
        if ylabel is not None:
            _ylabel = ylabel

        plot_total_dos(
            ax,
            self._frequency_points,
            self._dos,
            freq_Debye=self._freq_Debye,
            Debye_fit_coef=self._Debye_fit_coef,
            xlabel=_xlabel,
            ylabel=_ylabel,
            draw_grid=draw_grid,
            flip_xy=flip_xy,
        )

    def write(self, filename: str | os.PathLike = "total_dos.dat") -> None:
        """Write total DOS to total_dos.dat."""
        if self._dos is None:
            raise RuntimeError("Run total DOS calculation first.")

        if self._use_tetrahedron_method:
            comment = "Tetrahedron method"
        else:
            comment = "Sigma = %f" % self._sigma

        write_total_dos(
            self._frequency_points, self._dos, comment=comment, filename=filename
        )

    def _run_tetrahedron_method_dos(self) -> None:
        bzgrid, freqs_full = _bzgrid_and_full_grid_frequencies(
            self._mesh_object, lang=self._lang
        )
        res = TetrahedronDOSAccumulator(
            freqs_full,
            bzgrid,
            sampling_points=self._frequency_points,
            lang=self._lang,
        ).result
        # res.density shape: (1, n_sampling, 1) for plain DOS.
        self._dos = res.density[0, :, 0]

    def _get_density_of_states_at_freq(self, f: float) -> np.double:
        return np.sum(
            np.dot(self._weights, self._smearing_function.calc(self._frequencies - f))
        ) / np.sum(self._weights)


class ProjectedDos(Dos):
    """Class to calculate projected DOS.

    Attributes
    ----------
    projected_dos : ndarray
        Projected DOS.
        shape=(pdos, frequency_points), dtype="double"
        The first dimension depends on init argument of `xyz_projection`.
        With it True, the length is the number of atoms times 3, otherwise
        the number of atoms.

    """

    def __init__(
        self,
        mesh_object: Mesh,
        sigma: float | None = None,
        use_tetrahedron_method: bool = False,
        direction: Sequence[float] | NDArray[np.double] | None = None,
        xyz_projection: bool = False,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method."""
        super().__init__(
            mesh_object,
            sigma=sigma,
            use_tetrahedron_method=use_tetrahedron_method,
            lang=lang,
        )
        if self._mesh_object.eigenvectors is None:
            raise ValueError("Mesh object does not have eigenvectors.")
        self._eigenvectors = self._mesh_object.eigenvectors
        self._projected_dos = None

        if xyz_projection:
            self._eigvecs2 = np.abs(self._eigenvectors) ** 2
        else:
            num_atom = self._frequencies.shape[1] // 3
            i_x = np.arange(num_atom, dtype="int") * 3
            i_y = np.arange(num_atom, dtype="int") * 3 + 1
            i_z = np.arange(num_atom, dtype="int") * 3 + 2
            if direction is None:
                self._eigvecs2 = np.abs(self._eigenvectors[:, i_x, :]) ** 2
                self._eigvecs2 += np.abs(self._eigenvectors[:, i_y, :]) ** 2
                self._eigvecs2 += np.abs(self._eigenvectors[:, i_z, :]) ** 2
            else:
                d = np.array(direction, dtype="double")
                d /= np.linalg.norm(direction)
                proj_eigvecs = self._eigenvectors[:, i_x, :] * d[0]
                proj_eigvecs += self._eigenvectors[:, i_y, :] * d[1]
                proj_eigvecs += self._eigenvectors[:, i_z, :] * d[2]
                self._eigvecs2 = np.abs(proj_eigvecs) ** 2

    @property
    def projected_dos(self) -> NDArray[np.double] | None:
        """Return projected DOS."""
        return self._projected_dos

    def run(self) -> None:
        """Calculate projected DOS."""
        if self._use_tetrahedron_method:
            self._run_tetrahedron_method_dos()
        else:
            if self._frequency_points is None:
                raise RuntimeError("Run projected DOS calculation first.")
            self._run_smearing_method()

    def plot(
        self,
        ax,
        indices: Sequence[Sequence[int]] | None,
        legend: Sequence[str] | None = None,
        legend_prop: dict | None = None,
        legend_frameon: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        draw_grid: bool = True,
        flip_xy: bool = False,
    ) -> None:
        """Plot projected DOS."""
        if self._projected_dos is None:
            raise RuntimeError("Run projected DOS calculation first.")

        if flip_xy:
            _xlabel = "Partial density of states"
            _ylabel = "Frequency"
        else:
            _xlabel = "Frequency"
            _ylabel = "Partial density of states"

        if xlabel is not None:
            _xlabel = xlabel
        if ylabel is not None:
            _ylabel = ylabel

        plot_projected_dos(
            ax,
            self._frequency_points,
            self._projected_dos,
            indices=indices,
            legend=legend,
            legend_prop=legend_prop,
            legend_frameon=legend_frameon,
            xlabel=_xlabel,
            ylabel=_ylabel,
            draw_grid=draw_grid,
            flip_xy=flip_xy,
        )

    def write(self, filename: str | os.PathLike = "projected_dos.dat") -> None:
        """Write projected DOS to projected_dos.dat."""
        if self._frequency_points is None or self._projected_dos is None:
            raise RuntimeError("Run projected DOS calculation first.")

        if self._use_tetrahedron_method:
            comment = "Tetrahedron method"
        else:
            comment = "Sigma = %f" % self._sigma

        write_projected_dos(
            self._frequency_points,
            self._projected_dos,
            comment=comment,
            filename=filename,
        )

    def _run_smearing_method(self) -> None:
        assert self._frequency_points is not None
        num_pdos = self._eigvecs2.shape[1]
        num_freqs = len(self._frequency_points)
        self._projected_dos = np.zeros((num_pdos, num_freqs), dtype="double")
        weights = self._weights / float(np.sum(self._weights))
        for i, freq in enumerate(self._frequency_points):
            amplitudes = self._smearing_function.calc(self._frequencies - freq)
            for j in range(self._projected_dos.shape[0]):
                self._projected_dos[j, i] = np.dot(
                    weights, self._eigvecs2[:, j, :] * amplitudes
                ).sum()

    def _run_tetrahedron_method_dos(self) -> None:
        bzgrid, freqs_full = _bzgrid_and_full_grid_frequencies(
            self._mesh_object, lang=self._lang
        )
        # Replicate per-mode eigvecs2 to every grid point via the same ir
        # mapping that _bzgrid_and_full_grid_frequencies uses.
        ir_position = {
            int(gp): i for i, gp in enumerate(self._mesh_object.ir_grid_points)
        }
        positions = np.array(
            [ir_position[int(gp)] for gp in self._mesh_object.grid_mapping_table],
            dtype="int64",
        )
        # eigvecs2 shape: (n_ir, num_pdos, n_band).  Replicate to (n_grid,
        # num_pdos, n_band) then reshape to (1, n_grid, n_band, num_pdos)
        # for TetrahedronDOSAccumulator.
        mode_property = np.ascontiguousarray(
            self._eigvecs2[positions].transpose(0, 2, 1)[None]
        )
        res = TetrahedronDOSAccumulator(
            freqs_full,
            bzgrid,
            mode_property=mode_property,
            sampling_points=self._frequency_points,
            lang=self._lang,
        ).result
        # res.density shape: (1, n_sampling, num_pdos) -> (num_pdos, n_sampling).
        self._projected_dos = res.density[0].T


def write_total_dos(
    frequency_points: NDArray[np.double],
    total_dos: NDArray[np.double],
    comment: str | None = None,
    filename: str | os.PathLike = "total_dos.dat",
) -> None:
    """Write total_dos.dat."""
    with open(filename, "w") as fp:
        if comment is not None:
            fp.write("# %s\n" % comment)

        for freq, dos in zip(frequency_points, total_dos, strict=True):
            fp.write("%20.10f%20.10f\n" % (freq, dos))


def write_projected_dos(
    frequency_points: NDArray[np.double],
    projected_dos: NDArray[np.double],
    comment: str | None = None,
    filename: str | os.PathLike = "projected_dos.dat",
) -> None:
    """Write projected_dos.dat."""
    with open(filename, "w") as fp:
        if comment is not None:
            fp.write("# %s\n" % comment)

        for freq, pdos in zip(frequency_points, projected_dos.T, strict=True):
            fp.write("%20.10f" % freq)
            fp.write(("%20.10f" * len(pdos)) % tuple(pdos))
            fp.write("\n")


def plot_total_dos(
    ax,
    frequency_points: NDArray[np.double],
    total_dos: NDArray[np.double],
    freq_Debye: float | None = None,
    Debye_fit_coef: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    draw_grid: bool = True,
    flip_xy: bool = False,
    linestyle: str = "solid",
    color: str = "red",
    linewidth: float = 1.0,
    linestyle_Debye: str = "solid",
    color_Debye: str = "blue",
) -> None:
    """Plot total DOS."""
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    if freq_Debye is not None:
        freq_pitch = frequency_points[1] - frequency_points[0]
        num_points = int(freq_Debye / freq_pitch)
        freqs = np.linspace(0, freq_Debye, num_points + 1)

    if flip_xy:
        ax.plot(
            total_dos,
            frequency_points,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
        )
        if freq_Debye is not None and Debye_fit_coef is not None:
            ax.plot(
                np.append(Debye_fit_coef * freqs**2, 0),
                np.append(freqs, freq_Debye),
                linestyle=linestyle_Debye,
                color=color_Debye,
                linewidth=1,
            )
    else:
        ax.plot(
            frequency_points,
            total_dos,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
        )
        if freq_Debye is not None and Debye_fit_coef is not None:
            ax.plot(
                np.append(freqs, freq_Debye),
                np.append(Debye_fit_coef * freqs**2, 0),
                linestyle=linestyle_Debye,
                color=color_Debye,
                linewidth=1,
            )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(draw_grid)


def plot_projected_dos(
    ax,
    frequency_points: NDArray[np.double],
    projected_dos: NDArray[np.double],
    indices: Sequence[Sequence[int]] | None = None,
    legend: Sequence[str] | None = None,
    legend_prop: dict | None = None,
    legend_frameon: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    draw_grid: bool = True,
    flip_xy: bool = False,
) -> None:
    """Plot projected DOS."""
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    plots = []
    num_pdos = len(projected_dos)

    if indices is None:
        indices = []
        for i in range(num_pdos):
            indices.append([i])

    for set_for_sum in indices:
        pdos_sum = np.zeros_like(frequency_points)
        for i in set_for_sum:
            if i > num_pdos - 1:
                print("Index number '%d' is specified," % (i + 1))
                print("but it is not allowed to be larger than the number of atoms.")
                raise ValueError
            if i < 0:
                print(
                    "Index number '%d' is specified, but it must be positive." % (i + 1)
                )
                raise ValueError
            pdos_sum += projected_dos[i]
        if flip_xy:
            plots.append(ax.plot(pdos_sum, frequency_points, linewidth=1))
        else:
            plots.append(ax.plot(frequency_points, pdos_sum, linewidth=1))

    if legend is not None:
        ax.legend(legend, prop=legend_prop, frameon=legend_frameon)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(draw_grid)


def get_dos_frequency_range(
    freqs: NDArray[np.double], dos: NDArray[np.double]
) -> tuple[float, float]:
    """Return reasonable frequency range."""
    i_min = 0
    i_max = 1000

    for i, (_, d) in enumerate(zip(freqs, dos, strict=True)):
        if d > 1e-5:
            i_min = i
            break

    for i, (_, d) in enumerate(zip(freqs[::-1], dos[::-1], strict=True)):
        if d > 1e-5:
            i_max = len(freqs) - 1 - i
            break

    f_min = freqs[i_min]
    if f_min > 0:
        f_min = 0

    f_max = freqs[i_max]
    f_max += (f_max - f_min) * 0.05

    return f_min, f_max
