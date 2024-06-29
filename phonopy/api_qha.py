"""API for QHA calculation."""

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

from phonopy.qha.core import QHA, BulkModulus


class PhonopyQHA:
    """PhonopyQHA API."""

    def __init__(
        self,
        volumes=None,
        electronic_energies=None,
        temperatures=None,
        free_energy=None,
        cv=None,
        entropy=None,
        pressure=None,
        eos="vinet",
        t_max=None,
        energy_plot_factor=None,
        verbose=False,
    ):
        """Init method.

        Notes
        -----
        The first two parameters have to be in this order for the backward
        compatibility.

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
        pressure: float,
            Pressure in GPa that is added to energy as PV term.
        free_energy: array_like
            Phonon Helmholtz free energy (F_ph) in kJ/mol.
            dtype='double'
            shape=(temperatuers, volumes)
        cv: array_like
            Phonon heat capacity at constant volume in J/K/mol.
            dtype='double'
            shape=(temperatuers, volumes)
        entropy: array_like
            Phonon entropy at constant volume (S_ph) in J/K/mol.
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
        verbose: boolean
            Show log or not.

        """
        self._bulk_modulus = BulkModulus(
            volumes, electronic_energies, pressure=pressure, eos=eos
        )

        if temperatures is not None:
            self._qha = QHA(
                volumes,
                electronic_energies,
                temperatures,
                cv,
                entropy,
                free_energy,
                pressure=pressure,
                eos=eos,
                t_max=t_max,
                energy_plot_factor=energy_plot_factor,
            )
            self._qha.run(verbose=verbose)

    @property
    def bulk_modulus(self):
        """Return bulk modulus computed without phonon contribution.

        Returns
        -------
        float
            Bulk modulus calculated without phonon contribution.

        """
        return self._bulk_modulus.bulk_modulus

    @property
    def thermal_expansion(self):
        """Return thermal expansion coefficients at temperatures.

        Returns
        -------
        list
            Thermal expansion coefficients at temperatues.
            shape=(temperatures, )

        """
        return self._qha.thermal_expansion

    @property
    def helmholtz_volume(self):
        """Return free_energies at volumes.

        Returns
        -------
        ndarray
            Helmholtz free energies calculated at temperatures and volumes.
            shape=(temperatures, volumes)

        """
        return self._qha.helmholtz_volume

    @property
    def volume_temperature(self):
        """Return volumes at temperatures.

        Returns
        -------
        ndarray
            Equilibrium volumes at temperatures
            shape=(temperatures,), dtype=float

        """
        return self._qha.volume_temperature

    @property
    def gibbs_temperature(self):
        """Return Gibbs free energies at temperatures.

        Returns
        -------
        ndarray
            Gibbs free energies at temperatures.
            shape=(temperatures, ), dtype=float

        """
        return self._qha.gibbs_temperature

    @property
    def bulk_modulus_temperature(self):
        """Return bulk modulus at temperatures.

        Returns
        -------
        ndarray
            Bulk modulus at constant pressure and temperatures
            shape=(temperatures, ), dtype=float

        """
        return self._qha.bulk_modulus_temperature

    @property
    def heat_capacity_P_numerical(self):
        """Return heat capacities at constant pressure at temperatures.

        These values are calculated by -T*d^2G/dT^2.

        Returns
        -------
        list
            Heat capacity at constant pressure and temperatures.
            shape=(temperatures, )

        """
        return self._qha.get_heat_capacity_P_numerical()

    @property
    def heat_capacity_P_polyfit(self):
        """Return heat capacities at constant pressure at temperatures.

        Note
        ----
        This does not work when temperature dependent electronic_energies
        is supplied.

        Returns
        -------
        list
            Heat capacities at constant pressure at temperatures, which are
            calculated from the values obtained by polynomial fittings of
            Cv and S.

            shape=(temperatures, )

        """
        return self._qha.heat_capacity_P_polyfit

    @property
    def gruneisen_temperature(self):
        """Return Gruneisen parameters at temperatures.

        Returns
        -------
        list
            Thermodynamic Gruneisen parameters at temperatures.
            shape=(temperatures, )

        """
        return self._qha.gruneisen_temperature

    def get_bulk_modulus_parameters(self):
        """Return temperature independent bulk modulus EOS fitting parameters.

        These values are those computed without phonon free energy.

        (lowest energy, bulk modulus, b_prime, equilibrium volume)

        """
        return self._bulk_modulus.get_parameters()

    def write_helmholtz_volume(self, filename="helmholtz-volume.dat"):
        """Write Helmholtz free energy vs volume in file."""
        self._qha.write_helmholtz_volume(filename=filename)

    def write_helmholtz_volume_fitted(
        self, thin_number, filename="helmholtz-volume_fitted.dat"
    ):
        """Write Helmholtz free energy (fitted) vs volume in file."""
        self._qha.write_helmholtz_volume_fitted(thin_number, filename=filename)

    def write_volume_temperature(self, filename="volume-temperature.dat"):
        """Write volume vs temperature in file."""
        self._qha.write_volume_temperature(filename=filename)

    def write_thermal_expansion(self, filename="thermal_expansion.dat"):
        """Write thermal expansion vs temperature in file."""
        self._qha.write_thermal_expansion(filename=filename)

    def write_gibbs_temperature(self, filename="gibbs-temperature.dat"):
        """Write Gibbs free energy vs temperature in file."""
        self._qha.write_gibbs_temperature(filename=filename)

    def write_bulk_modulus_temperature(self, filename="bulk_modulus-temperature.dat"):
        """Write bulk modulus vs temperature in file."""
        self._qha.write_bulk_modulus_temperature(filename=filename)

    def plot_bulk_modulus(self, thin_number=10):
        """Return pyplot of bulk modulus fitting curve."""
        return self._bulk_modulus.plot(thin_number=thin_number)

    def plot_qha(self, thin_number=10, volume_temp_exp=None):
        """Return pyplot of QHA fitting curves at temperatures."""
        return self._qha.plot(thin_number=thin_number, volume_temp_exp=volume_temp_exp)

    def plot_helmholtz_volume(
        self, thin_number=10, xlabel=r"Volume $(\AA^3)$", ylabel="Free energy"
    ):
        """Return pyplot of Helmholtz free energes vs volume at temperatures."""
        return self._qha.plot_helmholtz_volume(
            thin_number=thin_number, xlabel=xlabel, ylabel=ylabel
        )

    def plot_pdf_helmholtz_volume(
        self, thin_number=10, filename="helmholtz-volume.pdf"
    ):
        """Plot Helmholtz free energes vs volume at temperatures in pdf."""
        self._qha.plot_pdf_helmholtz_volume(thin_number=thin_number, filename=filename)

    def plot_volume_temperature(self, exp_data=None):
        """Return pyplot of volume vs temperature."""
        return self._qha.plot_volume_temperature(exp_data=exp_data)

    def plot_pdf_volume_temperature(
        self, exp_data=None, filename="volume-temperature.pdf"
    ):
        """Plot volume vs temperature in pdf."""
        self._qha.plot_pdf_volume_temperature(exp_data=exp_data, filename=filename)

    def plot_thermal_expansion(self):
        """Return pyplot of thermal expansion vs temperature."""
        return self._qha.plot_thermal_expansion()

    def plot_pdf_thermal_expansion(self, filename="thermal_expansion.pdf"):
        """Plot thermal expansion vs temperature in pdf."""
        self._qha.plot_pdf_thermal_expansion(filename=filename)

    def plot_gibbs_temperature(
        self, xlabel="Temperature (K)", ylabel="Gibbs free energy"
    ):
        """Return pyplot of Gibbs free energy vs temperature."""
        return self._qha.plot_gibbs_temperature(xlabel=xlabel, ylabel=ylabel)

    def plot_pdf_gibbs_temperature(self, filename="gibbs-temperature.pdf"):
        """Plot Gibbs free energy vs temperature in pdf."""
        self._qha.plot_pdf_gibbs_temperature(filename=filename)

    def plot_bulk_modulus_temperature(
        self, xlabel="Temperature (K)", ylabel="Bulk modulus"
    ):
        """Return pyplot of bulk modulus vs temperature."""
        return self._qha.plot_bulk_modulus_temperature(xlabel=xlabel, ylabel=ylabel)

    def plot_pdf_bulk_modulus_temperature(
        self, filename="bulk_modulus-temperature.pdf"
    ):
        """Plot bulk modulus vs temperature in pdf."""
        self._qha.plot_pdf_bulk_modulus_temperature(filename=filename)

    def plot_heat_capacity_P_numerical(self, Z=1, exp_data=None):
        """Return pyplot of C_P by numerical difference vs temperature."""
        return self._qha.plot_heat_capacity_P_numerical(Z=Z, exp_data=exp_data)

    def plot_pdf_heat_capacity_P_numerical(
        self, exp_data=None, filename="Cp-temperature.pdf"
    ):
        """Plot C_P by numerical difference vs temperature in pdf."""
        self._qha.plot_pdf_heat_capacity_P_numerical(
            exp_data=exp_data, filename=filename
        )

    def write_heat_capacity_P_numerical(self, filename="Cp-temperature.dat"):
        """Write C_P by numerical difference vs temperature in file."""
        self._qha.write_heat_capacity_P_numerical(filename=filename)

    def plot_heat_capacity_P_polyfit(self, exp_data=None, Z=1):
        """Return pyplot of C_P by fittings vs temperature."""
        return self._qha.plot_heat_capacity_P_polyfit(Z=Z, exp_data=exp_data)

    def plot_pdf_heat_capacity_P_polyfit(
        self, exp_data=None, filename="Cp-temperature_polyfit.pdf"
    ):
        """Plot C_P by fittings vs temperature in pdf."""
        self._qha.plot_pdf_heat_capacity_P_polyfit(exp_data=exp_data, filename=filename)

    def write_heat_capacity_P_polyfit(
        self,
        filename="Cp-temperature_polyfit.dat",
        filename_ev="entropy-volume.dat",
        filename_cvv="Cv-volume.dat",
        filename_dsdvt="dsdv-temperature.dat",
    ):
        """Write C_P by fittings vs temperature in file."""
        self._qha.write_heat_capacity_P_polyfit(
            filename=filename,
            filename_ev=filename_ev,
            filename_cvv=filename_cvv,
            filename_dsdvt=filename_dsdvt,
        )

    def plot_gruneisen_temperature(self):
        """Return pyplot of Grueneisen parameter vs temperature."""
        return self._qha.plot_gruneisen_temperature()

    def plot_pdf_gruneisen_temperature(self, filename="gruneisen-temperature.pdf"):
        """Plot Grueneisen parameter vs temperature in pdf."""
        self._qha.plot_pdf_gruneisen_temperature(filename=filename)

    def write_gruneisen_temperature(self, filename="gruneisen-temperature.dat"):
        """Write Grueneisen parameter vs temperature in file."""
        self._qha.write_gruneisen_temperature(filename=filename)

    def get_bulk_modulus(self):
        """Return bulk moduli with no phonon contribution."""
        warnings.warn(
            "PhonopyQHA.get_bulk_modulus() is deprecated."
            "Use bulk_modulus attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bulk_modulus

    def get_helmholtz_volume(self):
        """Return Helmholtz free energies at temperatures and volumes."""
        warnings.warn(
            "PhonopyQHA.get_helmholtz_volume() is deprecated."
            "Use helmholtz_volume attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.helmholtz_volume

    def get_volume_temperature(self):
        """Return equilibrium volumes at temperatures."""
        warnings.warn(
            "PhonopyQHA.get_volume_temperature() is deprecated."
            "Use volume_temperature attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.volume_temperature

    def get_thermal_expansion(self):
        """Return thermal expansion coefficients at temperatures."""
        warnings.warn(
            "PhonopyQHA.get_thermal_expansion() is deprecated."
            "Use thermal_expansion attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.thermal_expansion

    def get_gibbs_temperature(self):
        """Return Gibbs free energies at temperatures."""
        warnings.warn(
            "PhonopyQHA.get_gibbs_temperature() is deprecated."
            "Use gibbs_temperature attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gibbs_temperature

    def get_bulk_modulus_temperature(self):
        """Return bulk moduli at temperatures."""
        warnings.warn(
            "PhonopyQHA.get_bulk_modulus_temperature() is deprecated."
            "Use bulk_modulus_temperature attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bulk_modulus_temperature

    def get_heat_capacity_P_numerical(self):
        """Return C_P calculated by numerical differentiation."""
        warnings.warn(
            "PhonopyQHA.get_heat_capacity_P_numerical() is deprecated."
            "Use heat_capacity_P_numerical attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.heat_capacity_P_numerical

    def get_heat_capacity_P_polyfit(self):
        """Return C_P calculated by fittings."""
        warnings.warn(
            "PhonopyQHA.get_heat_capacity_P_polyfit() is deprecated."
            "Use heat_capacity_P_polyfit attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.heat_capacity_P_polyfit

    def get_gruneisen_temperature(self):
        """Return Grueneisen paramters at temperatures."""
        warnings.warn(
            "PhonopyQHA.get_gruneisen_temperature() is deprecated."
            "Use gruneisen_temperature attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.gruneisen_temperature
