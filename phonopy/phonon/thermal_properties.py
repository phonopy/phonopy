# SPDX-License-Identifier: BSD-3-Clause
"""Phonon thermal properties at constant volume."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from phonopy.phonon.mesh import Mesh
from phonopy.physical_units import get_physical_units


class ThermalPropertiesDict(TypedDict):
    """Return type of Phonopy.get_thermal_properties_dict."""

    temperatures: NDArray[np.double]
    free_energy: NDArray[np.double]
    entropy: NDArray[np.double]
    heat_capacity: NDArray[np.double]


def mode_cv(
    temps: float | NDArray[np.double],
    freqs: float | NDArray[np.double],
    classical: bool = False,
) -> float | NDArray[np.double]:  # freqs (eV)
    """Return mode heat capacity.

    Parameters
    ----------
    temp : float
        Temperature in K.
    freqs : float or ndarray
        Phonon frequency in eV.
    classical : bool
        If True use classical statistics.
        If False use quantum statistics.

    Returns
    -------
    float or ndarray
        Mode heat capacity in eV/K.

    """
    if classical:
        if isinstance(freqs, np.ndarray) or isinstance(temps, np.ndarray):
            return (
                np.ones(shape=np.atleast_1d(freqs).shape + np.atleast_1d(temps).shape)
                * get_physical_units().KB
            )
        else:
            return get_physical_units().KB
    else:
        if isinstance(freqs, np.ndarray) or isinstance(temps, np.ndarray):
            x = np.divide.outer(freqs, get_physical_units().KB * temps).T
        else:
            x = freqs / (get_physical_units().KB * temps)
        expVal = np.exp(x)
        return get_physical_units().KB * x**2 * expVal / (expVal - 1.0) ** 2


class GammaAcousticWarning(UserWarning):
    """Issued about the acoustic modes at Gamma in the thermal-property sums.

    Two cases. The acoustic frequencies at Gamma are far from zero, which
    means the force constants do not satisfy translational invariance. Or
    ``exclude_gamma_acoustic`` is off and an acoustic mode at Gamma with a
    frequency that should be zero is a small positive number, so it enters
    the sums with a contribution that depends on that small value.

    """


# Acoustic frequencies at Gamma above this, in THz, are reported as a sign
# that the force constants break translational invariance.
GAMMA_ACOUSTIC_LARGE_FREQUENCY = 0.1


def gamma_acoustic_bands(mesh: Mesh) -> tuple[int | None, NDArray[np.int64] | None]:
    """Return the index of Gamma in the mesh and the band indices of its acoustic modes.

    The acoustic modes are the three modes at Gamma with the smallest
    absolute frequencies. An imaginary mode is stored as a negative frequency,
    so a large imaginary mode is not taken for an acoustic one. The pair is
    ``(None, None)`` when the mesh does not contain Gamma.

    """
    index = mesh.gamma_index
    if index is None:
        return None, None
    abs_frequencies = np.abs(mesh.frequencies[index])
    acoustic_bands = np.sort(np.argsort(abs_frequencies)[:3])
    return index, acoustic_bands


class ThermalPropertiesBase:
    """Base class of thermal property calculation."""

    def __init__(
        self,
        mesh: Mesh,
        cutoff_frequency: float | None = None,
        pretend_real: bool = False,
        band_indices: Sequence[Sequence[int]] | None = None,
        classical: bool = False,
        exclude_gamma_acoustic: bool = False,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Note
        ----
        Physical unit of Phonon frequency is eV throughout this class, for
        which phonon frequencies and cutoff frequency are stored as class
        instance variables by being transformed to have eV unit.

        Parameters
        ----------
         See Phonopy.run_thermal_properties().

        """
        self._band_indices = None
        self._classical = classical
        self._lang: Literal["C", "Rust"] = lang

        if cutoff_frequency is None or cutoff_frequency < 0:
            self._cutoff_frequency = 0.0
        else:
            self._cutoff_frequency = cutoff_frequency * get_physical_units().THzToEv

        if band_indices is not None:
            bi = np.hstack(band_indices).astype("int64")  # type: ignore[arg-type]
            self._band_indices = bi
            self._frequencies = np.array(
                mesh.frequencies[:, bi], dtype="double", order="C"
            )
        else:
            self._frequencies = mesh.frequencies

        if pretend_real:
            self._frequencies = abs(self._frequencies)
        self._frequencies = (
            np.array(self._frequencies, dtype="double", order="C")
            * get_physical_units().THzToEv
        )
        self._weights = mesh.weights
        self._num_modes = self._frequencies.shape[1] * self._weights.sum()

        self._exclude_gamma_acoustic = exclude_gamma_acoustic
        # Frequencies in eV of the acoustic modes at Gamma that enter the sums
        # with exclude_gamma_acoustic off, and the weight of Gamma.
        self._gamma_acoustic_included = np.zeros(0, dtype="double")
        self._gamma_weight = 0
        self._treat_gamma_acoustic(mesh)

        # Precompute masked (frequency, weight) pairs once. The cutoff mask
        # does not depend on temperature, so each thermal property reduces to a
        # single weighted sum over the surviving modes (see
        # _run_py_thermal_properties).
        cond = self._frequencies > self._cutoff_frequency
        self._masked_freqs = self._frequencies[cond]
        weights_per_mode = np.broadcast_to(
            self._weights[:, None], self._frequencies.shape
        )
        self._masked_weights = np.array(weights_per_mode[cond], dtype="double")
        self._num_integrated_modes = np.sum(self._weights * cond.sum(axis=1))

        self._num_formula_units = mesh.dynamical_matrix.primitive.Z

        # When self._weights.dtype is 'uint', the number is casted to float.
        # In future version, self._weights.dtype will be 'int_', so this
        # treatment will be unnecessary.
        if isinstance(self._num_modes, float):
            self._num_modes = int(self._num_modes)
        if isinstance(self._num_integrated_modes, float):
            self._num_integrated_modes = int(self._num_integrated_modes)

    @property
    def cutoff_frequency(self) -> float:
        """Return cutoff frequency in eV."""
        return self._cutoff_frequency

    @property
    def exclude_gamma_acoustic(self) -> bool:
        """Return whether the acoustic modes at Gamma are excluded from the sums."""
        return self._exclude_gamma_acoustic

    def _treat_gamma_acoustic(self, mesh: Mesh) -> None:
        """Set the acoustic modes at Gamma to zero frequency, or record them.

        The frequencies of the three acoustic modes at Gamma should be zero.
        The computed frequencies are, however, small numbers of either sign,
        even after the force constants are symmetrized. Like the other modes,
        one enters the sums when its frequency is larger than the cutoff. With
        the default cutoff of zero, the signs of these small values decide
        which of them enter, and a mode that enters adds about
        kT ln(h nu / kT) divided by the number of grid points. With
        ``exclude_gamma_acoustic``, the three frequencies are set to zero
        here, so that they fail the cutoff and the zero-point sum in every
        implementation. Only the copy used for the sums is changed; the mesh
        keeps the computed frequencies.

        """
        index, bands = gamma_acoustic_bands(mesh)
        if index is None or bands is None:
            return
        frequencies_thz = mesh.frequencies[index, bands]
        if np.abs(frequencies_thz).max() > GAMMA_ACOUSTIC_LARGE_FREQUENCY:
            warnings.warn(
                "The acoustic frequencies at Gamma are "
                + ", ".join(f"{f:.3e}" for f in frequencies_thz)
                + " THz, far from zero. The force constants do not satisfy "
                "translational invariance. Symmetrize them, e.g., by "
                "Phonopy.symmetrize_force_constants(use_symfc_projector=True).",
                GammaAcousticWarning,
                stacklevel=4,
            )

        if self._band_indices is None:
            columns = bands
        else:
            columns = np.flatnonzero(np.isin(self._band_indices, bands))
        if len(columns) == 0:
            return

        if self._exclude_gamma_acoustic:
            self._frequencies[index, columns] = 0.0
        else:
            values = self._frequencies[index, columns]
            self._gamma_acoustic_included = values[values > self._cutoff_frequency]
            self._gamma_weight = int(self._weights[index])

    def _warn_gamma_acoustic_included(
        self, temperatures: NDArray[np.double] | None
    ) -> None:
        """Warn when acoustic modes at Gamma enter the sums, with what they add."""
        if len(self._gamma_acoustic_included) == 0 or temperatures is None:
            return
        units = get_physical_units()
        t_max = float(np.max(temperatures))
        freqs = self._gamma_acoustic_included
        if t_max > 0:
            kt = units.KB * t_max
            if self._classical:
                fe_modes = kt * np.log(freqs / kt)
            else:
                fe_modes = kt * np.log(1 - np.exp(-freqs / kt)) + freqs / 2
        else:
            fe_modes = np.zeros_like(freqs) if self._classical else freqs / 2
        added = fe_modes.sum() * self._gamma_weight / np.sum(self._weights)
        warnings.warn(
            f"{len(freqs)} acoustic mode(s) at Gamma, at "
            + ", ".join(f"{f / units.THzToEv:.3e}" for f in freqs)
            + " THz, entered the thermal-property sums. Their frequencies are "
            "nearly zero, and they add "
            f"{added:.3e} eV per unit cell to the free energy at {t_max:g} K. "
            "Set exclude_gamma_acoustic=True (EXCLUDE_GAMMA_ACOUSTIC = .TRUE., "
            "--exclude-gamma-acoustic) to exclude them.",
            GammaAcousticWarning,
            stacklevel=4,
        )


class ThermalProperties(ThermalPropertiesBase):
    """Phonon thermal property calculation."""

    def __init__(
        self,
        mesh: Mesh,
        cutoff_frequency: float | None = None,
        pretend_real: bool = False,
        band_indices: Sequence[Sequence[int]] | None = None,
        classical: bool = False,
        exclude_gamma_acoustic: bool = False,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Note
        ----
        Physical unit of Phonon frequency is eV throughout this class, for
        which phonon frequencies and cutoff frequency are stored as class
        instance variables by being transformed to have eV unit.

        Parameters
        ----------
        See Phonopy.run_thermal_properties().

        """
        super().__init__(
            mesh,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            classical=classical,
            exclude_gamma_acoustic=exclude_gamma_acoustic,
            lang=lang,
        )
        self._thermal_properties = None
        self._temperatures = None
        self._zero_point_energy = None

        zp_energy = 0.0
        if classical:
            self._zero_point_energy = 0.0
        else:
            for freqs, w in zip(self._frequencies, self._weights, strict=True):
                positive_fs = np.extract(freqs > 0.0, freqs)
                zp_energy += np.sum(positive_fs) * w / 2
            self._zero_point_energy = (
                zp_energy / np.sum(self._weights) * get_physical_units().EvTokJmol
            )

    @property
    def temperatures(self) -> NDArray | None:
        """Setter and getter of temperatures in K."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures: Sequence[float] | NDArray[np.double]) -> None:
        t_array = np.array(temperatures, dtype="double")
        self._temperatures = np.array(
            np.extract(np.invert(t_array < 0), t_array), dtype="double"
        )

    @property
    def thermal_properties(
        self,
    ) -> (
        tuple[
            NDArray[np.double],
            NDArray[np.double],
            NDArray[np.double],
            NDArray[np.double],
        ]
        | None
    ):
        """Return thermal properties.

        Returns
        -------
        tuple :
            Temperatures in K,
            Helmholtz free energies in kJmol,
            Entropies in J/K/mol,
            Heat capacities. in J/K/mol.

        """
        return self._thermal_properties  # type: ignore[return-value]

    @property
    def free_energy(self) -> NDArray[np.double] | None:
        """Return Helmholtz free energies in kJ/mol.

        None is returned before running the calculation.

        """
        if self._thermal_properties is None:
            return None
        return self._thermal_properties[1]

    @property
    def entropy(self) -> NDArray[np.double] | None:
        """Return entropies in J/K/mol.

        None is returned before running the calculation.

        """
        if self._thermal_properties is None:
            return None
        return self._thermal_properties[2]

    @property
    def heat_capacity(self) -> NDArray[np.double] | None:
        """Return heat capacities in J/K/mol.

        None is returned before running the calculation.

        """
        if self._thermal_properties is None:
            return None
        return self._thermal_properties[3]

    @property
    def zero_point_energy(self) -> float | None:
        """Return zero point energy in kJ/mol."""
        return self._zero_point_energy  # type: ignore[return-value]

    @property
    def number_of_integrated_modes(self) -> int:
        """Return number of phonon modes integrated on mesh sampling grid."""
        return self._num_integrated_modes

    @property
    def number_of_modes(self) -> int:
        """Return total number of phonon modes on mesh sampling grid."""
        return self._num_modes

    def set_temperature_range(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        t_step: float | None = None,
    ) -> None:
        """Set temperature range where thermal properties are calculated."""
        if t_min is None:
            _t_min = 10
        elif t_min < 0:
            _t_min = 0
        else:
            _t_min = t_min

        if t_max is None:
            _t_max = 1000
        elif t_max > _t_min:
            _t_max = t_max
        else:
            _t_max = _t_min

        if t_step is None:
            _t_step = 10
        elif t_step > 0:
            _t_step = t_step
        else:
            _t_step = 10

        self._temperatures = np.arange(
            _t_min, _t_max + _t_step / 2.0, _t_step, dtype="double"
        )

    def plot(
        self,
        ax: Any,
        xlabel: str | None = None,
        ylabel: str | None = None,
        with_grid: bool = True,
        divide_by_Z: bool = False,
        legend_style: str | None = "normal",
    ) -> None:
        """Plot thermal properties using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Single Matplotlib Axes object.
        xlabel : str, optional
            Label used for x-axis.
        ylabel : str, optional
            Label used for y-axis.
        with_grid : bool, optional
            With grid or not. Default is True.
        divide_by_Z : bool, optional
            Divide thermal properties by number of formula units of primitive
            cell. Default is False.
        legend_style : str, optional
            "normal", "compact", None. None will not show legend.

        """
        if xlabel is None:
            if legend_style == "compact":
                _xlabel = "Temperature (K)"
            else:
                _xlabel = "Temperature [K]"
        else:
            _xlabel = xlabel

        if divide_by_Z:
            z_num = self._num_formula_units
        else:
            z_num = 1

        temps, fe, entropy, cv = self._thermal_properties  # type: ignore[misc]

        ax.plot(temps, fe / z_num, "r-")
        ax.plot(temps, entropy / z_num, "b-")
        ax.plot(temps, cv / z_num, "g-")

        if legend_style == "compact":
            ax.legend(
                (
                    "Free energy (kJ/mol)",
                    "Entropy (J/K/mol)",
                    r"$C_\mathrm{V}$ (J/K/mol)",
                ),
                loc="best",
                prop={"size": 8.5},
                frameon=False,
            )
        elif legend_style == "normal":
            ax.legend(
                (
                    "Free energy [kJ/mol]",
                    "Entropy [J/K/mol]",
                    r"C$_\mathrm{V}$ [J/K/mol]",
                ),
                loc="best",
            )

        ax.grid(with_grid)
        if not with_grid:
            ax.axhline(y=0, linestyle=":", linewidth=0.5, color="k")
        ax.set_xlabel(_xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    def run(
        self,
        t_step: float | None = None,
        t_max: float | None = None,
        t_min: float | None = None,
        lang: Literal["C", "Python", "Rust"] | None = None,
    ) -> None:
        """Run thermal property calculation.

        ``lang="Rust"`` is accepted for API consistency but routes to the
        pure-Python implementation; the kernel is fast enough in numpy that
        no Rust port exists.  When ``lang=None`` (default), the value is
        taken from the ``lang`` argument passed at construction.

        """
        if t_step is not None or t_max is not None or t_min is not None:
            warnings.warn(
                "keywords for this method are depreciated. "
                "Use 'set_temperature_range' or "
                "'set_temperature_range' method instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.set_temperature_range(t_min=t_min, t_max=t_max, t_step=t_step)

        self._warn_gamma_acoustic_included(self._temperatures)
        _lang = lang if lang is not None else self._lang
        if _lang == "C":
            self._run_c_thermal_properties()
        else:
            self._run_py_thermal_properties()

    def write_yaml(
        self,
        filename: str | os.PathLike = "thermal_properties.yaml",
        volume: float | None = None,
    ) -> None:
        """Write thermal properties in yaml file."""
        lines = self._get_tp_yaml_lines(volume=volume)
        with open(filename, "w") as w:
            w.write("\n".join(lines))

    def _run_c_thermal_properties(self) -> None:
        import phonopy._phonopy as phonoc

        props = np.zeros((len(self._temperatures), 3), dtype="double", order="C")  # type: ignore[arg-type]
        phonoc.thermal_properties(
            props,
            self._temperatures,
            self._frequencies,
            self._weights,
            self._cutoff_frequency,
            get_physical_units().KB,
            self._classical,
        )

        props /= np.sum(self._weights)
        fe = props[:, 0] * get_physical_units().EvTokJmol + self._zero_point_energy  # type: ignore[operator]
        entropy = props[:, 1] * get_physical_units().EvTokJmol * 1000
        cv = props[:, 2] * get_physical_units().EvTokJmol * 1000
        self._thermal_properties = (self._temperatures, fe, entropy, cv)

    def _run_py_thermal_properties(self) -> None:
        # Free energy, entropy and heat capacity are computed over the whole
        # (temperature x mode) grid at once, reusing the precomputed masked
        # frequencies and weights. The per-mode formulas are inlined here so
        # free energy / entropy / heat capacity can be evaluated in a single
        # vectorized pass without a Python loop over temperatures.
        temps = np.asarray(self._temperatures, dtype="double")
        freqs = self._masked_freqs
        weights = self._masked_weights
        KB = get_physical_units().KB
        EvTokJmol = get_physical_units().EvTokJmol
        sumw = np.sum(self._weights)

        fe = np.zeros(len(temps), dtype="double")
        entropy = np.zeros(len(temps), dtype="double")
        cv = np.zeros(len(temps), dtype="double")

        # t = 0: free energy is the zero-point energy (quantum) or 0
        # (classical); entropy and heat capacity vanish.
        if not self._classical:
            fe[temps <= 0] = np.dot(freqs / 2, weights)

        # t > 0: evaluate in temperature blocks to bound peak memory (a single
        # (n_temp, n_mode) array can be large for dense non-symmetric meshes).
        pos_idx = np.flatnonzero(temps > 0)
        block = max(1, 4_000_000 // max(1, freqs.size))
        f = freqs[None, :]
        for lo in range(0, len(pos_idx), block):
            idx = pos_idx[lo : lo + block]
            T = temps[idx][:, None]
            if self._classical:
                ln = np.log(f / (KB * T))
                fe[idx] = (KB * T * ln) @ weights
                entropy[idx] = (KB - KB * ln) @ weights
                cv[idx] = KB * weights.sum()
            else:
                x = f / (KB * T)
                ex = np.exp(x)
                fe[idx] = (KB * T * np.log(1.0 - np.exp(-x)) + f / 2) @ weights
                v = x / 2
                sinh_v = np.sinh(v)
                entropy[idx] = (
                    KB * v * np.cosh(v) / sinh_v - KB * np.log(2 * sinh_v)
                ) @ weights
                cv[idx] = (KB * x**2 * ex / (ex - 1.0) ** 2) @ weights

        self._thermal_properties = (
            self._temperatures,
            fe / sumw * EvTokJmol,
            entropy / sumw * EvTokJmol * 1000,
            cv / sumw * EvTokJmol * 1000,
        )

    def _get_tp_yaml_lines(self, volume: float | None = None) -> list[str]:
        lines = []
        lines.append("# Thermal properties / unit cell (natom)")
        lines.append("")
        lines.append("unit:")
        lines.append("  temperature:   K")
        lines.append("  free_energy:   kJ/mol")
        lines.append("  entropy:       J/K/mol")
        lines.append("  heat_capacity: J/K/mol")
        lines.append("")
        lines.append("natom: %-5d" % (self._frequencies[0].shape[0] // 3))
        if volume is not None:
            lines.append("volume: %-20.10f" % volume)
        lines.append(
            "cutoff_frequency: %.5f"
            % (self._cutoff_frequency / get_physical_units().THzToEv)
        )
        lines.append(
            "exclude_gamma_acoustic: %s"
            % ("true" if self._exclude_gamma_acoustic else "false")
        )
        lines.append("num_modes: %d" % self._num_modes)
        lines.append("num_integrated_modes: %d" % self._num_integrated_modes)
        if self._band_indices is not None:
            bi = self._band_indices + 1
            lines.append(
                "band_index: [ "
                + ("%d, " * (len(bi) - 1)) % tuple(bi[:-1])
                + ("%d ]" % bi[-1])
            )
        lines.append("")
        lines.append("zero_point_energy: %15.7f" % self._zero_point_energy)
        lines.append("")
        lines.append("thermal_properties:")
        temperatures, fe, entropy, cv = self._thermal_properties  # type: ignore[misc]
        for i, t in enumerate(temperatures):  # type: ignore[arg-type]
            lines.append("- temperature:   %15.7f" % t)
            lines.append("  free_energy:   %15.7f" % fe[i])
            lines.append("  entropy:       %15.7f" % entropy[i])
            # Sometimes 'nan' of C_V is returned at low temperature.
            if np.isnan(cv[i]):
                lines.append("  heat_capacity: %15.7f" % 0)
            else:
                lines.append("  heat_capacity: %15.7f" % cv[i])
            lines.append("  energy:        %15.7f" % (fe[i] + entropy[i] * t / 1000))
            lines.append("")
        return lines
