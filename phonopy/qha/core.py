"""Core routines for QHA."""

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

import warnings

import numpy as np

from phonopy.qha.eos import fit_to_eos, get_eos
from phonopy.units import Avogadro, EVAngstromToGPa, EvTokJmol


class BulkModulus:
    """Bulk modulus class.

    This class is used to calculate bulk modulus only from temperature
    independent energy input.

    """

    def __init__(self, volumes, energies, pressure=None, eos="vinet"):
        """Init method.

        volumes : array_like
            Unit cell volumes where energies are obtained.
            shape=(volumes, ), dtype='double'.
        energies : array_like
            Energies obtained at volumes and optionally temperatures.
            shape=(volumes, ), or shape=(temperatures, volumes) dtype='double'.
        pressure: float, optional
            Pressure in GPa that is added to energy as PV term.
        eos : str, optional
            Identifier of equation of states function.

        """
        self._volumes = np.array(volumes)
        self._energies = np.array(energies)
        self._eos_name = eos
        self._eos = get_eos(self._eos_name)

        if pressure is not None:
            self._energies += self._volumes * pressure / EVAngstromToGPa

        if self._energies.ndim == 1:
            self._energy = None
            self._bulk_modulus = None
            self._b_prime = None
            self._equiv_volume = None
            (
                self._energy,
                self._bulk_modulus,
                self._b_prime,
                self._equiv_volume,
            ) = self.fit_to_eos(self._energies)
        elif self._energies.ndim == 2:
            self._energy = np.zeros(len(self._energies), dtype="double")
            self._bulk_modulus = np.zeros_like(self._energy)
            self._b_prime = np.zeros_like(self._energy)
            self._equiv_volume = np.zeros_like(self._energy)
            for i, energies_at_T in enumerate(self._energies):
                e, b, bp, ev = self.fit_to_eos(energies_at_T)
                self._energy[i] = e
                self._bulk_modulus[i] = b
                self._b_prime[i] = bp
                self._equiv_volume[i] = ev
        else:
            raise TypeError("Array shape of energies is wrong.")

    def fit_to_eos(self, energies):
        """Fit energy-volume to EOS."""
        try:
            e, b, bp, ev = fit_to_eos(self._volumes, energies, self._eos)
        except TypeError as exc:
            msg = ['Failed to fit to "%s" equation of states.' % self._eos_name]
            if len(self._volumes) < 4:
                msg += ["At least 4 volume points are needed for the fitting."]
            msg += ["Careful choice of volume points is recommended."]
            raise RuntimeError("\n".join(msg)) from exc
        return e, b, bp, ev

    @property
    def bulk_modulus(self):
        """Return bulk modulus."""
        return self._bulk_modulus

    def get_bulk_modulus(self):
        """Return bulk modulus."""
        warnings.warn(
            "BulkModulus.get_bulk_modulus() is deprecated."
            "Use BulkModulus.bulk_modulus attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bulk_modulus

    @property
    def equilibrium_volume(self):
        """Return volume at equilibrium."""
        return self._equiv_volume

    def get_equilibrium_volume(self):
        """Return volume at equilibrium."""
        warnings.warn(
            "BulkModulus.get_equilibrium_volume() is deprecated."
            "Use BulkModulus.equilibrium_volume attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.equilibrium_volume

    @property
    def b_prime(self):
        """Return fitted parameter B'."""
        return self._b_prime

    def get_b_prime(self):
        """Return fitted parameter B'."""
        warnings.warn(
            "BulkModulus.get_b_prime() is deprecated."
            "Use BulkModulus.b_prime attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._b_prime

    @property
    def energy(self):
        """Return fitted parameter of energy."""
        return self._energy

    def get_energy(self):
        """Return fitted parameter of energy."""
        warnings.warn(
            "BulkModulus.get_energy() is deprecated."
            "Use BulkModulus.energy attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._energy

    def get_parameters(self):
        """Return fitted parameters."""
        return (self._energy, self._bulk_modulus, self._b_prime, self._equiv_volume)

    def get_eos(self):
        """Return EOS function as a python method."""
        warnings.warn(
            "BulkModulus.get_eos() is deprecated.", DeprecationWarning, stacklevel=2
        )
        return self._eos

    def plot(self, thin_number=10):
        """Plot fitted EOS curve."""
        import matplotlib.pyplot as plt

        vols = self._volumes
        volume_points = np.linspace(min(vols), max(vols), 201)
        _, ax = plt.subplots()
        parameters = self.get_parameters()
        if self._energies.ndim == 1:
            ax.plot(volume_points, self._eos(volume_points, *parameters), "r-")
            ax.plot(vols, self._energies, "bo", markersize=4)
        elif self._energies.ndim == 2:
            for i, (e_t, b_t, bp_t, ev_t) in enumerate(zip(*parameters)):
                if i % thin_number == 0:
                    ep = (e_t, b_t, bp_t, ev_t)
                    ax.plot(volume_points, self._eos(volume_points, *ep), "-")
                    ax.plot(vols, self._energies[i], "bo", markersize=4)
        return plt


class QHA:
    """Quasi harmonic approximation class."""

    def __init__(
        self,
        volumes,  # angstrom^3
        electronic_energies,  # eV
        temperatures,  # K
        cv,  # J/K/mol
        entropy,  # J/K/mol
        fe_phonon,  # kJ/mol
        pressure=None,
        eos="vinet",
        t_max=None,
        energy_plot_factor=None,
    ):
        """Init method.

        Parameters
        ----------
        volumes: array_like
            Unit cell volumes (V) in angstrom^3.
            dtype='double'
            shape=(volumes,)
        electronic_energies: array_like
            Electronic energies (U_el) or electronic free energies (F_el) in eV.
            It is assumed as formar if ndim==1 and latter if ndim==2.
            dtype='double'
            shape=(volumes,) or (temperatuers, volumes)
        temperatures: array_like
            Temperatures ascending order (T) in K.
            dtype='double'
            shape=(temperatures,)
        cv: array_like
            Phonon Heat capacity at constant volume in J/K/mol.
            dtype='double'
            shape=(temperatuers, volumes)
        entropy: array_like
            Phonon entropy at constant volume (S_ph) in J/K/mol.
            dtype='double'
            shape=(temperatuers, volumes)
        fe_phonon: array_like
            Phonon Helmholtz free energy (F_ph) in kJ/mol.
            dtype='double'
            shape=(temperatuers, volumes)
        pressure: float, optional
            Pressure in GPa that is added to energy as PV term.
        eos: str, optional
            Equation of state used for fitting F vs V.
            'vinet', 'murnaghan' or 'birch_murnaghan'.
        t_max: float
            Maximum temperature to be calculated. This has to be not
            greater than the temperature of the third element from the
            end of 'temperatre' elements. If max_t=None, the temperature
            of the third element from the end is used.
        energy_plot_factor: float
            This value is multiplied to energy like values only in plotting.

        """
        self._volumes = np.array(volumes)
        self._electronic_energies = np.array(electronic_energies)
        if pressure is not None:
            self._electronic_energies += self._volumes * pressure / EVAngstromToGPa
        self._all_temperatures = np.array(temperatures)
        self._cv = np.array(cv)
        self._entropy = np.array(entropy)
        self._fe_phonon = np.array(fe_phonon) / EvTokJmol

        self._eos = get_eos(eos)
        self._t_max = t_max
        self._energy_plot_factor = energy_plot_factor

        self._temperatures = None
        self._equiv_volumes = None
        self._equiv_energies = None
        self._equiv_bulk_modulus = None
        self._equiv_parameters = None
        self._free_energies = None
        self._num_elems = None

        self._thermal_expansions = None
        self._cp_numerical = None
        self._volume_entropy_parameters = None
        self._volume_cv_parameters = None
        self._volume_entropy = None
        self._volume_cv = None
        self._cp_polyfit = None
        self._dsdv = None
        self._gruneisen_parameters = None
        self._len = None

    @property
    def thermal_expansion(self):
        """Return volumetric thermal expansion coefficients at temperatures."""
        return self._thermal_expansions[: self._len]

    @property
    def helmholtz_volume(self):
        """Return Helmholtz free energies at temperatures and volumes."""
        return self._free_energies[: self._len]

    @property
    def volume_temperature(self):
        """Return equilibrium volumes at temperatures."""
        return self._equiv_volumes[: self._len]

    @property
    def gibbs_temperature(self):
        """Return Gibbs free energies at temperatures."""
        return self._equiv_energies[: self._len]

    @property
    def bulk_modulus_temperature(self):
        """Return bulk modulus vs temperature data."""
        return self._equiv_bulk_modulus[: self._len]

    @property
    def heat_capacity_P_numerical(self):
        """Return heat capacities at constant pressure at temperatures.

        Values are computed by numerical derivative of Gibbs free energy.

        """
        return self._cp_numerical[: self._len]

    @property
    def heat_capacity_P_polyfit(self):
        """Return heat capacities at constant pressure at temperatures.

        Volumes are computed in another way to heat_capacity_P_numerical
        for the better numerical behaviour. But this does not work
        when temperature dependent electronic_energies is supplied.

        """
        if self._electronic_energies.ndim == 1:
            return self._cp_polyfit[: self._len]
        else:
            raise NotImplementedError(
                "QHA.heat_capacity_P_polyfit() unavailable with electronic free energy"
            )

    @property
    def gruneisen_temperature(self):
        """Return Gruneisen parameters at temperatures."""
        return self._gruneisen_parameters[: self._len]

    def run(self, verbose=False):
        """Fit parameters to EOS at temperatures.

        Even if fitting failed, simply omit the volume point. In this case,
        the failed temperature point doesn't exist in the returned arrays.

        """
        if verbose:
            print(("#%11s" + "%14s" * 4) % ("T", "E_0", "B_0", "B'_0", "V_0"))

        # Plus one temperature point is necessary for computing e.g. beta.
        num_elems = self._get_num_elems(self._all_temperatures) + 1
        if num_elems > len(self._all_temperatures):
            num_elems -= 1

        temperatures = []
        parameters = []
        free_energies = []

        for i in range(num_elems):  # loop over temperaturs
            if self._electronic_energies.ndim == 1:
                el_energy = self._electronic_energies
            else:
                el_energy = self._electronic_energies[i]
            fe = [ph_e + el_e for ph_e, el_e in zip(self._fe_phonon[i], el_energy)]

            try:
                ep = fit_to_eos(self._volumes, fe, self._eos)
            except TypeError:
                print("Fitting failure at T=%.1f" % self._all_temperatures[i])

            if ep is None:
                # Simply omit volume point where the fitting failed.
                continue
            else:
                [ee, eb, ebp, ev] = ep
                t = self._all_temperatures[i]
                temperatures.append(t)
                parameters.append(ep)
                free_energies.append(fe)

                if verbose:
                    print(
                        ("%14.6f" * 5)
                        % (t, ep[0], ep[1] * EVAngstromToGPa, ep[2], ep[3])
                    )

        self._free_energies = np.array(free_energies)
        self._temperatures = np.array(temperatures)
        self._equiv_parameters = np.array(parameters)
        self._equiv_volumes = np.array(self._equiv_parameters[:, 3])
        self._equiv_energies = np.array(self._equiv_parameters[:, 0])
        self._equiv_bulk_modulus = np.array(
            self._equiv_parameters[:, 1] * EVAngstromToGPa
        )

        self._num_elems = len(self._temperatures)

        # For computing following values at temperatures, finite difference
        # method is used. Therefore number of temperature points are needed
        # larger than self._num_elems that nearly equals to the temparature
        # point we expect.
        self._set_thermal_expansion()
        self._set_heat_capacity_P_numerical()
        self._set_heat_capacity_P_polyfit()
        self._set_gruneisen_parameter()  # To be run after thermal expansion.

        self._len = len(self._thermal_expansions)
        assert self._len + 1 == self._num_elems

    def plot(self, thin_number=10, volume_temp_exp=None):
        """Plot three figures.

        - Helmholtz free energy at volumes and temperatures.
        - Equilibrium volumes at temperatures.
        - Thermal expansion coefficients at temperatures.

        """
        import matplotlib.pyplot as plt

        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["font.family"] = "serif"

        fig, axs = plt.subplots(1, 3, figsize=(7, 3.5))
        axs[0].xaxis.set_ticks_position("both")
        axs[0].yaxis.set_ticks_position("both")
        axs[0].xaxis.set_tick_params(which="both", direction="in")
        axs[0].yaxis.set_tick_params(which="both", direction="in")
        self._plot_helmholtz_volume(axs[0], thin_number=thin_number)
        axs[1].xaxis.set_ticks_position("both")
        axs[1].yaxis.set_ticks_position("both")
        axs[1].xaxis.set_tick_params(which="both", direction="in")
        axs[1].yaxis.set_tick_params(which="both", direction="in")
        self._plot_volume_temperature(axs[1], exp_data=volume_temp_exp)
        axs[2].xaxis.set_ticks_position("both")
        axs[2].yaxis.set_ticks_position("both")
        axs[2].xaxis.set_tick_params(which="both", direction="in")
        axs[2].yaxis.set_tick_params(which="both", direction="in")
        self._plot_thermal_expansion(axs[2])
        plt.tight_layout()
        return plt

    def get_helmholtz_volume(self):
        """Return Helmholtz free energy at volumes and temperatures."""
        warnings.warn(
            "QHA.get_helmholtz_volume() is deprecated."
            "Use helmholtz_volume attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.helmholtz_volume

    def plot_helmholtz_volume(
        self, thin_number=10, xlabel=r"Volume $(\AA^3)$", ylabel="Free energy"
    ):
        """Return pyplot of Helmholtz free energes vs volume at temperatures."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_helmholtz_volume(
            ax, thin_number=thin_number, xlabel=xlabel, ylabel=ylabel
        )
        return plt

    def plot_pdf_helmholtz_volume(
        self, thin_number=10, filename="helmholtz-volume.pdf"
    ):
        """Plot Helmholtz free energes vs volume at temperatures in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_helmholtz_volume(ax, thin_number=thin_number)
        plt.savefig(filename)
        plt.close()

    def write_helmholtz_volume(self, filename="helmholtz-volume.dat"):
        """Write Helmholtz free energy vs volume in file."""
        w = open(filename, "w")
        for i, (t, ep, fe) in enumerate(
            zip(self._temperatures, self._equiv_parameters, self._free_energies)
        ):
            if i == self._len:
                break

            w.write("# Temperature: %f\n" % t)
            w.write("# Parameters: %f %f %f %f\n" % tuple(ep))
            for j, v in enumerate(self._volumes):
                w.write("%20.15f %25.15f\n" % (v, fe[j]))
            w.write("\n\n")
        w.close()

    def write_helmholtz_volume_fitted(
        self, thin_number, filename="helholtz-volume_fitted.dat"
    ):
        """Write Helmholtz free energy (fitted) vs volume in file."""
        if self._energy_plot_factor is None:
            _energy_plot_factor = 1
        else:
            _energy_plot_factor = self._energy_plot_factor

        volume_points = np.linspace(min(self._volumes), max(self._volumes), 201)
        selected_volumes = []
        selected_energies = []

        for i, _ in enumerate(self._temperatures[: self._len]):
            if i % thin_number == 0:
                selected_volumes.append(self._equiv_volumes[i])
                selected_energies.append(self._equiv_energies[i])

        e0 = 0
        for i, t in enumerate(self._temperatures[: self._len]):
            if t >= 298:
                if i > 0:
                    de = self._equiv_energies[i] - self._equiv_energies[i - 1]
                    dt = t - self._temperatures[i - 1]
                    e0 = (
                        298 - self._temperatures[i - 1]
                    ) / dt * de + self._equiv_energies[i - 1]
                else:
                    e0 = 0
                break
        e0 *= _energy_plot_factor
        _data_vol_points = []
        _data_eos = []
        for i, _ in enumerate(self._temperatures[: self._len]):
            if i % thin_number == 0:
                _data_vol_points.append(
                    np.array(self._free_energies[i]) * _energy_plot_factor - e0
                )
                _data_eos.append(
                    self._eos(volume_points, *self._equiv_parameters[i])
                    * _energy_plot_factor
                    - e0
                )

        data_eos = np.array(_data_eos).T
        data_vol_points = np.array(_data_vol_points).T
        data_min = np.array(selected_energies) * _energy_plot_factor - e0

        with open(filename, "w") as w:
            w.write("# Volume points\n")
            for j, k in zip(self._volumes, data_vol_points):
                w.write("%10.5f " % j)
                for ll in k:
                    w.write("%10.5f" % ll)
                w.write("\n")
            w.write("\n# Fitted data\n")

            for m, n in zip(volume_points, data_eos):
                w.write("%10.5f " % m)
                for ll in n:
                    w.write("%10.5f" % ll)
                w.write("\n")
            w.write("\n# Minimas\n")
            for a, b in zip(selected_volumes, data_min):
                w.write("%10.5f %10.5f %s" % (a, b, "\n"))
            w.write("\n")

    def get_volume_temperature(self):
        """Return equilibrium volumes at temperatures."""
        warnings.warn(
            "QHA.get_volume_temperature() is deprecated."
            "Use volume_temperature attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.volume_temperature

    def plot_volume_temperature(self, exp_data=None):
        """Return pyplot of volume vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_volume_temperature(ax, exp_data=exp_data)

        return plt

    def plot_pdf_volume_temperature(
        self, exp_data=None, filename="volume-temperature.pdf"
    ):
        """Plot volume vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_volume_temperature(ax, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_volume_temperature(self, filename="volume-temperature.dat"):
        """Write volume vs temperature in file."""
        w = open(filename, "w")
        for i in range(self._len):
            w.write(
                "%25.15f %25.15f\n" % (self._temperatures[i], self._equiv_volumes[i])
            )
        w.close()

    def get_thermal_expansion(self):
        """Return thermal expansion coefficients at temperatures."""
        warnings.warn(
            "QHA.get_thermal_expansion() is deprecated."
            "Use thermal_expansion attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.thermal_expansion

    def plot_thermal_expansion(self):
        """Return pyplot of thermal expansion vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_thermal_expansion(ax)

        return plt

    def plot_pdf_thermal_expansion(self, filename="thermal_expansion.pdf"):
        """Plot thermal expansion vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_thermal_expansion(ax)
        plt.savefig(filename)
        plt.close()

    def write_thermal_expansion(self, filename="thermal_expansion.dat"):
        """Write thermal expansion vs temperature in file."""
        w = open(filename, "w")
        for i in range(self._len):
            w.write(
                "%25.15f %25.15f\n"
                % (self._temperatures[i], self._thermal_expansions[i])
            )
        w.close()

    def get_gibbs_temperature(self):
        """Return Gibbs free energies at temperatures."""
        return self.gibbs_temperature

    def plot_gibbs_temperature(
        self, xlabel="Temperature (K)", ylabel="Gibbs free energy"
    ):
        """Return pyplot Gibbs free energy vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_gibbs_temperature(ax, xlabel=xlabel, ylabel=ylabel)

        return plt

    def plot_pdf_gibbs_temperature(self, filename="gibbs-temperature.pdf"):
        """Plot Gibbs free energy vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_gibbs_temperature(ax)
        plt.savefig(filename)
        plt.close()

    def write_gibbs_temperature(self, filename="gibbs-temperature.dat"):
        """Write Gibbs free energy vs temperature in file."""
        w = open(filename, "w")
        for i in range(self._len):
            w.write(
                "%20.15f %25.15f\n" % (self._temperatures[i], self._equiv_energies[i])
            )
        w.close()

    def get_bulk_modulus_temperature(self):
        """Return bulk moduli at temperatures."""
        return self.bulk_modulus_temperature

    def plot_bulk_modulus_temperature(
        self, xlabel="Temperature (K)", ylabel="Bulk modulus"
    ):
        """Return pyplot of bulk modulus vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_bulk_modulus_temperature(ax, xlabel=xlabel, ylabel=ylabel)
        return plt

    def plot_pdf_bulk_modulus_temperature(
        self, filename="bulk_modulus-temperature.pdf"
    ):
        """Plot bulk modulus vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_bulk_modulus_temperature(ax)
        plt.savefig(filename)
        plt.close()

    def write_bulk_modulus_temperature(self, filename="bulk_modulus-temperature.dat"):
        """Write bulk modulus vs temperature in file."""
        w = open(filename, "w")
        for i in range(self._len):
            w.write(
                "%20.15f %25.15f\n"
                % (self._temperatures[i], self._equiv_bulk_modulus[i])
            )
        w.close()

    def get_heat_capacity_P_numerical(self):
        """Return C_P by numerical differenciation at temperatures."""
        return self.heat_capacity_P_numerical

    def plot_heat_capacity_P_numerical(self, Z=1, exp_data=None):
        """Return pyplot of C_P by numerical difference vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_heat_capacity_P_numerical(ax, Z=Z, exp_data=exp_data)

        return plt

    def plot_pdf_heat_capacity_P_numerical(
        self, exp_data=None, filename="Cp-temperature.pdf"
    ):
        """Plot C_P by numerical difference vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_heat_capacity_P_numerical(ax, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_heat_capacity_P_numerical(self, filename="Cp-temperature.dat"):
        """Write C_P by numerical difference vs temperature in file."""
        w = open(filename, "w")
        for i in range(self._len):
            w.write(
                "%20.15f %20.15f\n" % (self._temperatures[i], self._cp_numerical[i])
            )
        w.close()

    def get_heat_capacity_P_polyfit(self):
        """Return C_P by fitting at temperatures."""
        return self.heat_capacity_P_polyfit

    def plot_heat_capacity_P_polyfit(self, Z=1, exp_data=None):
        """Return pyplot of C_P by fittings vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_heat_capacity_P_polyfit(ax, Z=Z, exp_data=exp_data)

        return plt

    def plot_pdf_heat_capacity_P_polyfit(
        self, exp_data=None, filename="Cp-temperature_polyfit.pdf"
    ):
        """Plot C_P by fittings vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_heat_capacity_P_polyfit(ax, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_heat_capacity_P_polyfit(
        self,
        filename="Cp-temperature_polyfit.dat",
        filename_ev="entropy-volume.dat",
        filename_cvv="Cv-volume.dat",
        filename_dsdvt="dsdv-temperature.dat",
    ):
        """Write C_P by fittings vs temperature in file."""
        wve = open(filename_ev, "w")
        wvcv = open(filename_cvv, "w")
        for i in range(1, self._len):
            t = self._temperatures[i]
            wve.write("# temperature %20.15f\n" % t)
            wve.write(
                "# %20.15f %20.15f %20.15f %20.15f %20.15f\n"
                % tuple(self._volume_entropy_parameters[i - 1])
            )
            wvcv.write("# temperature %20.15f\n" % t)
            wvcv.write(
                "# %20.15f %20.15f %20.15f %20.15f %20.15f\n"
                % tuple(self._volume_cv_parameters[i - 1])
            )
            for ve, vcv in zip(self._volume_entropy[i - 1], self._volume_cv[i - 1]):
                wve.write("%20.15f %20.15f\n" % tuple(ve))
                wvcv.write("%20.15f %20.15f\n" % tuple(vcv))
            wve.write("\n\n")
            wvcv.write("\n\n")
        wve.close()
        wvcv.close()

        w = open(filename, "w")
        for i in range(self._len):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i], self._cp_polyfit[i]))
        w.close()

        w = open(filename_dsdvt, "w")  # GPa
        for i in range(self._len):
            w.write(
                "%20.15f %20.15f\n"
                % (self._temperatures[i], self._dsdv[i] * 1e21 / Avogadro)
            )
        w.close()

    def get_gruneisen_temperature(self):
        """Return Grueneisen parameters at temperatures."""
        return self.gruneisen_temperature

    def plot_gruneisen_temperature(self):
        """Return pyplot of Grueneisen parameter vs temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._plot_gruneisen_temperature(ax)

        return plt

    def plot_pdf_gruneisen_temperature(self, filename="gruneisen-temperature.pdf"):
        """Plot Grueneisen parameter vs temperature in pdf."""
        import matplotlib.pyplot as plt

        self._set_rcParams(plt)

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._plot_gruneisen_temperature(ax)
        plt.savefig(filename)
        plt.close()

    def write_gruneisen_temperature(self, filename="gruneisen-temperature.dat"):
        """Write Grueneisen parameter vs temperature in file."""
        w = open(filename, "w")
        for i in range(self._len):
            w.write(
                "%20.15f %25.15f\n"
                % (self._temperatures[i], self._gruneisen_parameters[i])
            )
        w.close()

    def _plot_helmholtz_volume(
        self, ax, thin_number=10, xlabel=r"Volume $(\AA^3)$", ylabel="Free energy"
    ):
        if self._energy_plot_factor is None:
            _energy_plot_factor = 1
            _ylabel = ylabel + " (eV)"
        else:
            _energy_plot_factor = self._energy_plot_factor
            _ylabel = ylabel

        volume_points = np.linspace(min(self._volumes), max(self._volumes), 201)
        selected_volumes = []
        selected_energies = []

        thin_index = 0
        for i, _ in enumerate(self._temperatures[: self._len]):
            if i % thin_number == 0:
                selected_volumes.append(self._equiv_volumes[i])
                selected_energies.append(self._equiv_energies[i])

        e0 = 0
        for i, t in enumerate(self._temperatures[: self._len]):
            if t >= 298:
                if i > 0:
                    de = self._equiv_energies[i] - self._equiv_energies[i - 1]
                    dt = t - self._temperatures[i - 1]
                    e0 = (
                        298 - self._temperatures[i - 1]
                    ) / dt * de + self._equiv_energies[i - 1]
                else:
                    e0 = 0
                break
        e0 *= _energy_plot_factor

        for i, _ in enumerate(self._temperatures[: self._len]):
            if i % thin_number == 0:
                ax.plot(
                    self._volumes,
                    np.array(self._free_energies[i]) * _energy_plot_factor - e0,
                    "bo",
                    markeredgecolor="b",
                    markersize=3,
                )
                ax.plot(
                    volume_points,
                    self._eos(volume_points, *self._equiv_parameters[i])
                    * _energy_plot_factor
                    - e0,
                    "b-",
                )
                thin_index = i

        for i, j in enumerate((0, thin_index)):
            ax.text(
                self._volumes[-2],
                (self._free_energies[j, -1] + (1 - i * 2) * 0.1 - 0.05)
                * _energy_plot_factor
                - e0,
                "%dK" % int(self._temperatures[j]),
                fontsize=8,
            )

        ax.plot(
            selected_volumes,
            np.array(selected_energies) * _energy_plot_factor - e0,
            "ro-",
            markeredgecolor="r",
            markersize=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_ylabel)

    def _plot_volume_temperature(
        self, ax, exp_data=None, xlabel="Temperature (K)", ylabel=r"Volume $(\AA^3)$"
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(self._temperatures[: self._len], self._equiv_volumes[: self._len], "r-")
        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])
        # exp
        if exp_data:
            ax.plot(exp_data[0], exp_data[1], "ro")

    def _plot_thermal_expansion(
        self,
        ax,
        xlabel="Temperature (K)",
        ylabel=r"Thermal expansion $(\mathrm{K}^{-1})$",
    ):
        from matplotlib.ticker import ScalarFormatter

        class FixedScaledFormatter(ScalarFormatter):
            def __init__(self):
                super().__init__(useMathText=True)

            def _set_orderOfMagnitude(self, range):
                self.orderOfMagnitude = -6

        ax.yaxis.set_major_formatter(FixedScaledFormatter())
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        beta = np.array(self._thermal_expansions)
        ax.plot(self._temperatures[: self._len], beta[: self._len], "r-")
        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _plot_gibbs_temperature(
        self, ax, xlabel="Temperature (K)", ylabel="Gibbs free energy (eV)"
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(
            self._temperatures[: self._len], self._equiv_energies[: self._len], "r-"
        )
        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])

    def _plot_bulk_modulus_temperature(
        self, ax, xlabel="Temperature (K)", ylabel="Bulk modulus (GPa)"
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(
            self._temperatures[: self._len], self._equiv_bulk_modulus[: self._len], "r-"
        )
        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])

    def _plot_heat_capacity_P_numerical(
        self,
        ax,
        Z=1,
        exp_data=None,
        xlabel="Temperature (K)",
        ylabel=r"$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$",
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(
            self._temperatures[: self._len],
            np.array(self._cp_numerical[: self._len]) / Z,
            "r-",
        )

        # exp
        if exp_data:
            ax.plot(exp_data[0], exp_data[1], "ro")

        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])

    def _plot_heat_capacity_P_polyfit(
        self,
        ax,
        Z=1,
        exp_data=None,
        xlabel="Temperature (K)",
        ylabel=r"$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$",
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(
            self._temperatures[: self._len],
            np.array(self._cp_polyfit[: self._len]) / Z,
            "r-",
        )

        # exp
        if exp_data:
            ax.plot(exp_data[0], exp_data[1], "ro")

        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])

    def _plot_gruneisen_temperature(
        self, ax, xlabel="Temperature (K)", ylabel="Gruneisen parameter"
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.plot(
            self._temperatures[: self._len],
            self._gruneisen_parameters[: self._len],
            "r-",
        )
        ax.set_xlim(self._temperatures[0], self._temperatures[self._len - 1])

    def _set_thermal_expansion(self):
        beta = [0.0]
        for i in range(1, self._num_elems - 1):
            dt = self._temperatures[i + 1] - self._temperatures[i - 1]
            dv = self._equiv_volumes[i + 1] - self._equiv_volumes[i - 1]
            beta.append(dv / dt / self._equiv_volumes[i])

        self._thermal_expansions = beta

    def _set_heat_capacity_P_numerical(self):
        cp = []
        g = np.array(self._equiv_energies) * EvTokJmol * 1000
        cp.append(0.0)

        for i in range(1, self._num_elems - 1):
            t = self._temperatures[i]
            parameters = np.polyfit(
                self._temperatures[i - 1 : i + 2], g[i - 1 : i + 2], 2
            )
            cp.append(-(2 * parameters[0]) * t)

        self._cp_numerical = cp

    def _set_heat_capacity_P_polyfit(self):
        cp = [0.0]
        dsdv = [0.0]
        self._volume_entropy_parameters = []
        self._volume_cv_parameters = []
        self._volume_entropy = []
        self._volume_cv = []

        for j in range(1, self._num_elems - 1):
            t = self._temperatures[j]
            x = self._equiv_volumes[j]

            try:
                parameters = np.polyfit(self._volumes, self._cv[j], 4)
            except np.lib.polynomial.RankWarning as exc:
                msg = ["Failed to fit heat capacities to polynomial of degree 4."]
                if len(self._volumes) < 5:
                    msg += ["At least 5 volume points are needed for the fitting."]
                raise RuntimeError("\n".join(msg)) from exc

            cv_p = np.dot(parameters, np.array([x**4, x**3, x**2, x, 1]))
            self._volume_cv_parameters.append(parameters)

            try:
                parameters = np.polyfit(self._volumes, self._entropy[j], 4)
            except np.lib.polynomial.RankWarning as exc:
                msg = ["Failed to fit entropies to polynomial of degree 4."]
                if len(self._volumes) < 5:
                    msg += ["At least 5 volume points are needed for the fitting."]
                raise RuntimeError("\n".join(msg)) from exc

            dsdv_t = np.dot(parameters[:4], np.array([4 * x**3, 3 * x**2, 2 * x, 1]))
            self._volume_entropy_parameters.append(parameters)

            try:
                parameters = np.polyfit(
                    self._temperatures[j - 1 : j + 2],
                    self._equiv_volumes[j - 1 : j + 2],
                    2,
                )
            except np.lib.polynomial.RankWarning as exc:
                msg = (
                    "Failed to fit equilibrium volumes vs T to "
                    "polynomial of degree 2."
                )
                raise RuntimeError(msg) from exc
            dvdt = parameters[0] * 2 * t + parameters[1]

            cp.append(cv_p + t * dvdt * dsdv_t)
            dsdv.append(dsdv_t)

            self._volume_cv.append(np.array([self._volumes, self._cv[j]]).T)
            self._volume_entropy.append(np.array([self._volumes, self._entropy[j]]).T)

        self._cp_polyfit = cp
        self._dsdv = dsdv

    def _set_gruneisen_parameter(self):
        gamma = [0]
        for i in range(1, self._num_elems - 1):
            v = self._equiv_volumes[i]
            kt = self._equiv_bulk_modulus[i]
            beta = self._thermal_expansions[i]
            try:
                parameters = np.polyfit(self._volumes, self._cv[i], 4)
            except np.lib.polynomial.RankWarning as exc:
                msg = ["Failed to fit heat capacities to polynomial of degree 4."]
                if len(self._volumes) < 5:
                    msg += ["At least 5 volume points are needed for the fitting."]
                raise RuntimeError("\n".join(msg)) from exc
            cv = (
                np.dot(parameters, [v**4, v**3, v**2, v, 1])
                / v
                / 1000
                / EvTokJmol
                * EVAngstromToGPa
            )
            if cv < 1e-10:
                gamma.append(0.0)
            else:
                gamma.append(beta * kt / cv)
        self._gruneisen_parameters = gamma

    def _get_num_elems(self, temperatures):
        if self._t_max is None:
            return len(temperatures)
        else:
            i = np.argmin(np.abs(temperatures - self._t_max))
            return i + 1

    def _set_rcParams(self, plt):
        plt.rcParams["backend"] = "PDF"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["axes.labelsize"] = 18
        plt.rcParams["figure.subplot.left"] = 0.25
        plt.rcParams["figure.subplot.bottom"] = 0.15
        plt.rcParams["figure.figsize"] = 4, 6
