"""Phonon thermal properties at constant volume."""

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

import warnings
from typing import Optional, Union

import numpy as np

from phonopy.phonon.mesh import Mesh
from phonopy.units import EvTokJmol, Kb, THzToEv


def mode_cv(
    temp: float, freqs: Union[float, np.ndarray], classical: bool = False
) -> Union[float, np.ndarray]:  # freqs (eV)
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
        return np.array(len(freqs) * [Kb])
    else:
        x = freqs / Kb / temp
        expVal = np.exp(x)
        return Kb * x**2 * expVal / (expVal - 1.0) ** 2


def mode_F(
    temp: float, freqs: Union[float, np.ndarray], classical: bool = False
) -> Union[float, np.ndarray]:
    """Return mode Helmholtz free energy.

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
        Mode Helmholtz free energy in eV.

    """
    if classical:
        return Kb * temp * np.log(freqs / (Kb * temp))
    else:
        return Kb * temp * np.log(1.0 - np.exp((-freqs) / (Kb * temp))) + freqs / 2


def mode_S(
    temp: float, freqs: Union[float, np.ndarray], classical: bool = False
) -> Union[float, np.ndarray]:
    """Return mode entropy.

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
        Mode entropy in eV/K.

    """
    if classical:
        return Kb - Kb * np.log(freqs / (Kb * temp))
    else:
        val = freqs / (2 * Kb * temp)
        return 1 / (2 * temp) * freqs * np.cosh(val) / np.sinh(val) - Kb * np.log(
            2 * np.sinh(val)
        )


def mode_ZPE(
    temp: float, freqs: Union[float, np.ndarray], classical: bool = False
) -> Union[float, np.ndarray]:
    """Return half of phonon frequency as mode zero point energy.

    Parameters
    ----------
    temp : float
        Dummy parameter. This is not used but needed to exist.
    freqs : float or ndarray
        Phonon frequency in eV.
    classical : bool
        If True use classical statistics.
        If False use quantum statistics.

    Returns
    -------
    float or ndarray
        Half of phonon frequency as mode zero point energy in eV.

    """
    if classical:
        return np.array(len(freqs) * [0.0])
    else:
        return freqs / 2


def mode_zero(
    temp: float, freqs: Union[float, np.ndarray], classical: bool = False
) -> Union[float, np.ndarray]:
    """Return zero.

    Parameters
    ----------
    temp : float
        Dummy parameter. This is not used but needed to exist.
    freqs : float or ndarray
        Dummy parameter. This is not used except for determining the array shape.
    classical : bool
        Dummy parameter. This is not used but needed to exist.

    Returns
    -------
    float or ndarray
        0 or an array of zero.

    """
    if isinstance(freqs, np.ndarray):
        return np.zeros_like(freqs)
    else:
        return 0.0


class ThermalPropertiesBase:
    """Base class of thermal property calculation."""

    def __init__(
        self,
        mesh: Mesh,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
        classical=False,
    ):
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
        self._is_projection = is_projection
        self._band_indices = None
        self._classical = classical

        if cutoff_frequency is None or cutoff_frequency < 0:
            self._cutoff_frequency = 0.0
        else:
            self._cutoff_frequency = cutoff_frequency * THzToEv

        if band_indices is not None:
            bi = np.hstack(band_indices).astype("intc")
            self._band_indices = bi
            self._frequencies = np.array(
                mesh.frequencies[:, bi], dtype="double", order="C"
            )
            if mesh.eigenvectors is not None:
                self._eigenvectors = np.array(
                    mesh.eigenvectors[:, :, bi], dtype="double", order="C"
                )
        else:
            self._frequencies = mesh.frequencies
            self._eigenvectors = mesh.eigenvectors

        if pretend_real:
            self._frequencies = abs(self._frequencies)
        self._frequencies = (
            np.array(self._frequencies, dtype="double", order="C") * THzToEv
        )
        self._weights = mesh.weights
        self._num_modes = self._frequencies.shape[1] * self._weights.sum()
        self._num_integrated_modes = np.sum(
            self._weights * (self._frequencies > self._cutoff_frequency).sum(axis=1)
        )

        self._num_formula_units = mesh.dynamical_matrix.primitive.Z

        # When self._weights.dtype is 'uint', the number is casted to float.
        # In future version, self._weights.dtype will be 'int_', so this
        # treatment will be unnecessary.
        if isinstance(self._num_modes, float):
            self._num_modes = int(self._num_modes)
        if isinstance(self._num_integrated_modes, float):
            self._num_integrated_modes = int(self._num_integrated_modes)

    @property
    def cutoff_frequency(self):
        """Return cutoff frequency in eV."""
        return self._cutoff_frequency

    def run_free_energy(self, t):
        """Calculate mode Helmholtz free energy in kJ/mol."""
        if t > 0:
            free_energy = self._calculate_thermal_property(mode_F, t)
        else:
            free_energy = self._calculate_thermal_property(mode_ZPE, None)
        return free_energy / np.sum(self._weights) * EvTokJmol

    def run_heat_capacity(self, t):
        """Calculate mode heat capacity in kJ/K/mol."""
        if t > 0:
            cv = self._calculate_thermal_property(mode_cv, t)
        else:
            cv = self._calculate_thermal_property(mode_zero, None)
        return cv / np.sum(self._weights) * EvTokJmol

    def run_entropy(self, t):
        """Calculate mode entropy in kJ/K/mol."""
        if t > 0:
            entropy = self._calculate_thermal_property(mode_S, t)
        else:
            entropy = self._calculate_thermal_property(mode_zero, None)
        return entropy / np.sum(self._weights) * EvTokJmol

    def _calculate_thermal_property(self, func, t):
        if not self._is_projection:
            t_property = 0.0
            for freqs, w in zip(self._frequencies, self._weights):
                cond = freqs > self._cutoff_frequency
                t_property += (
                    np.sum(func(t, freqs[cond], classical=self._classical)) * w
                )
            return t_property
        else:
            t_property = np.zeros(len(self._frequencies[0]), dtype="double")
            for freqs, eigvecs2, w in zip(
                self._frequencies, np.abs(self._eigenvectors) ** 2, self._weights
            ):
                cond = freqs > self._cutoff_frequency
                t_property += (
                    np.dot(
                        eigvecs2[:, cond],
                        func(t, freqs[cond], classical=self._classical),
                    )
                    * w
                )
            return t_property


class ThermalProperties(ThermalPropertiesBase):
    """Phonon thermal property calculation."""

    def __init__(
        self,
        mesh,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
        classical=False,
    ):
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
            is_projection=is_projection,
            classical=classical,
        )
        self._thermal_properties = None
        self._temperatures = None
        self._zero_point_energy = None
        self._projected_thermal_properties = None

        zp_energy = 0.0
        if classical:
            self._zero_point_energy = 0.0
        else:
            for freqs, w in zip(self._frequencies, self._weights):
                positive_fs = np.extract(freqs > 0.0, freqs)
                zp_energy += np.sum(positive_fs) * w / 2
            self._zero_point_energy = zp_energy / np.sum(self._weights) * EvTokJmol

    @property
    def temperatures(self):
        """Setter and getter of temperatures in K."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        t_array = np.array(temperatures, dtype="double")
        self._temperatures = np.array(
            np.extract(np.invert(t_array < 0), t_array), dtype="double"
        )

    def get_temperatures(self):
        """Return temperatures."""
        warnings.warn(
            "ThermalProperties.get_temperatures is deprecated."
            "Use temperatures attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.temperatures

    def set_temperatures(self, temperatures):
        """Set temperatures."""
        warnings.warn(
            "ThermalProperties.set_temperatures is deprecated."
            "Use temperatures attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.temperatures = temperatures

    @property
    def thermal_properties(self):
        """Return thermal properties.

        Returns
        -------
        tuple :
            Temperatures in K,
            Helmholtz free energies in kJmol,
            Entropies in J/K/mol,
            Heat capacities. in J/K/mol.

        """
        return self._thermal_properties

    def get_thermal_properties(self):
        """Return thermal properties."""
        warnings.warn(
            "ThermalProperties.get_thermal_properties is deprecated."
            "Use thermal_properties attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.thermal_properties

    @property
    def zero_point_energy(self):
        """Return zero point energy in kJ/mol."""
        return self._zero_point_energy

    def get_zero_point_energy(self):
        """Return zero point energy in kJ/mol."""
        warnings.warn(
            "ThermalProperties.get_zero_point_energy is deprecated."
            "Use zero_point_energy attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.zero_point_energy

    @property
    def number_of_integrated_modes(self):
        """Return number of phonon modes integrated on mesh sampling grid."""
        return self._num_integrated_modes

    def get_number_of_integrated_modes(self):
        """Return number of phonon modes integrated on mesh sampling grid."""
        warnings.warn(
            "ThermalProperties.get_number_of_integrated_modes is "
            "deprecated. Use number_of_integrated_modes attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.number_of_integrated_modes

    @property
    def number_of_modes(self):
        """Return total number of phonon modes on mesh sampling grid."""
        return self._num_modes

    def get_number_of_modes(self):
        """Return total number of phonon modes on mesh sampling grid."""
        warnings.warn(
            "ThermalProperties.get_number_of_modes is "
            "deprecated. Use number_of_modes attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.number_of_modes

    def set_temperature_range(self, t_min=None, t_max=None, t_step=None):
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
        ax,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        with_grid: bool = True,
        divide_by_Z: bool = False,
        legend_style: Optional[str] = "normal",
    ):
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

        temps, fe, entropy, cv = self._thermal_properties

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

    def run(self, t_step=None, t_max=None, t_min=None, lang="C"):
        """Run thermal property calculation."""
        if t_step is not None or t_max is not None or t_min is not None:
            warnings.warn(
                "keywords for this method are depreciated. "
                "Use 'set_temperature_range' or "
                "'set_temperature_range' method instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.set_temperature_range(t_min=t_min, t_max=t_max, t_step=t_step)

        if lang == "C":
            self._run_c_thermal_properties()
        else:
            self._run_py_thermal_properties()

        if self._is_projection:
            fe = []
            entropy = []
            cv = []
            for t in self._temperatures:
                fe.append(self.run_free_energy(t))
                entropy.append(self.run_entropy(t) * 1000)
                cv.append(self.run_heat_capacity(t) * 1000)

            self._projected_thermal_properties = (
                self._temperatures,
                np.array(fe, dtype="double"),
                np.array(entropy, dtype="double"),
                np.array(cv, dtype="double"),
            )

    def write_yaml(self, filename="thermal_properties.yaml", volume=None):
        """Write thermal properties in yaml file."""
        lines = self._get_tp_yaml_lines(volume=volume)
        if self._is_projection:
            lines += self._get_projected_tp_yaml_lines()
        with open(filename, "w") as w:
            w.write("\n".join(lines))

    def _run_c_thermal_properties(self):
        import phonopy._phonopy as phonoc

        props = np.zeros((len(self._temperatures), 3), dtype="double", order="C")
        phonoc.thermal_properties(
            props,
            self._temperatures,
            self._frequencies,
            self._weights,
            self._cutoff_frequency,
            self._classical,
        )
        # for f, w in zip(self._frequencies, self._weights):
        #     phonoc.thermal_properties(
        #         props,
        #         self._temperatures,
        #         np.array(f, dtype='double', order='C')[None, :],
        #         np.array([w], dtype='intc'),
        #         cutoff_frequency)

        props /= np.sum(self._weights)
        fe = props[:, 0] * EvTokJmol + self._zero_point_energy
        entropy = props[:, 1] * EvTokJmol * 1000
        cv = props[:, 2] * EvTokJmol * 1000
        self._thermal_properties = (self._temperatures, fe, entropy, cv)

    def _run_py_thermal_properties(self):
        fe = []
        entropy = []
        cv = []
        for t in self._temperatures:
            props = self._get_py_thermal_properties(t)
            fe.append(props[0])
            entropy.append(props[1] * 1000)
            cv.append(props[2] * 1000)
        self._thermal_properties = (
            self._temperatures,
            np.array(fe, dtype="double"),
            np.array(entropy, dtype="double"),
            np.array(cv, dtype="double"),
        )

    def _get_tp_yaml_lines(self, volume=None):
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
        lines.append("cutoff_frequency: %.5f" % (self._cutoff_frequency / THzToEv))
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
        temperatures, fe, entropy, cv = self._thermal_properties
        for i, t in enumerate(temperatures):
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

    def _get_projected_tp_yaml_lines(self):
        lines = []
        lines.append("projected_thermal_properties:")
        temperatures, fe, entropy, cv = self._projected_thermal_properties
        for i, t in enumerate(temperatures):
            lines.append("- temperature:   %13.7f" % t)
            line = "  free_energy:   [ "
            line += ", ".join(["%13.7f" % x for x in fe[i]])
            line += " ] # %13.7f" % np.sum(fe[i])
            lines.append(line)
            line = "  entropy:       [ "
            line += ", ".join(["%13.7f" % x for x in entropy[i]])
            line += " ] # %13.7f" % np.sum(entropy[i])
            lines.append(line)
            # Sometimes 'nan' of C_V is returned at low temperature.
            line = "  heat_capacity: [ "
            sum_cv = 0.0
            for j, cv_i in enumerate(cv[i]):
                if np.isnan(cv_i):
                    line += "%13.7f" % 0
                else:
                    sum_cv += cv_i
                    line += "%13.7f" % cv_i
                if j < len(cv[i]) - 1:
                    line += ", "
                else:
                    line += " ]"
            line += " # %13.7f" % sum_cv
            lines.append(line)
            energy = fe[i] + entropy[i] * t / 1000
            line = "  energy:        [ "
            line += ", ".join(["%13.7f" % x for x in energy])
            line += " ] # %13.7f" % np.sum(energy)
            lines.append(line)
        return lines

    def _get_py_thermal_properties(self, t):
        return (self.run_free_energy(t), self.run_entropy(t), self.run_heat_capacity(t))
