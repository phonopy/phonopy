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

from phonopy.qha import *
from phonopy.units import EvTokJmol, EVAngstromToGPa

class PhonopyQHA(object):
    def __init__(self,
                 volumes,
                 electronic_energies,
                 eos='vinet',
                 temperatures=None,
                 free_energy=None,
                 cv=None,
                 entropy=None,
                 t_max=None,
                 verbose=False):
        """
        The following two have the same number of elements
          volumes: Unit cell volumes (V) in Angstrom^3
          electronic_energies: Electronic energies (U) in eV

        The following four have the same number of elements
          temperatures: Temperatures ascending order (T) in K
          cv: Heat capacity at constant volume in J/K/mol
          entropy: Entropy at constant volume (S) J/K/mol
          free_energy: Helmholtz free energy (F) kJ/mol

        eos: Equation of state used for fitting F vs V
             'vinet', 'murnaghan' or 'birch_murnaghan'
        tmax: Maximum temperature to be calculated. This has to be not
              greater than the temperature of the third element from the
              end of 'temperatre' elements. If max_t=None, the temperature
              of the third element from the end is used.
        """
        self._bulk_modulus = BulkModulus(volumes,
                                         electronic_energies,
                                         eos=eos)

        if temperatures is not None:
            self._qha = QHA(volumes,
                            electronic_energies,
                            temperatures,
                            cv,
                            entropy,
                            free_energy,
                            eos=eos,
                            t_max=t_max)
            self._qha.run(verbose=verbose)

    def get_bulk_modulus(self):
        return self._bulk_modulus.get_bulk_modulus()
            
    def get_bulk_modulus_parameters(self):
        """Returns bulk modulus
        (lowest energy,
         bulk modulus,
         b_prime,
         equilibrium volume)
        """
        return self._bulk_modulus.get_parameters()
    
    def plot_bulk_modulus(self):
        return self._bulk_modulus.plot()

    def get_eos(self):
        return self._qha.get_eos()

    def plot_qha(self, thin_number=10, volume_temp_exp=None):
        return self._qha.plot(thin_number=thin_number,
                              volume_temp_exp=volume_temp_exp)

    def get_helmholtz_volume(self):
        """Returns free_energies
        
        free_energies: Free energies calculated at temperatures and volumes
                       [temperatures][volumes]
        """
        return self._qha.get_helmholtz_volume()

    def plot_helmholtz_volume(self,
                              thin_number=10,
                              plt=None,
                              xlabel=r'Volume $(\AA^3)$',
                              ylabel='Free energy'):
        return self._qha.plot_helmholtz_volume(thin_number=thin_number,
                                               plt=plt,
                                               xlabel=xlabel,
                                               ylabel=ylabel)

    def plot_pdf_helmholtz_volume(self,
                                  thin_number=10,
                                  filename='helmholtz-volume.pdf'):
        self._qha.plot_pdf_helmholtz_volume(thin_number=thin_number,
                                            filename=filename)

    def write_helmholtz_volume(self, filename='helmholtz-volume.dat'):
        self._qha.write_helmholtz_volume(filename=filename)

    def get_volume_temperature(self):
        """Returns volumes at temperatures"""
        return self._qha.get_volume_temperature()

    def plot_volume_temperature(self, exp_data=None, plt=None):
        return self._qha.plot_volume_temperature(exp_data=exp_data, plt=plt)

    def plot_pdf_volume_temperature(self,
                                    exp_data=None,
                                    filename='volume-temperature.pdf'):
        self._qha.plot_pdf_volume_temperature(exp_data=exp_data,
                                              filename=filename)

    def write_volume_temperature(self, filename='volume-temperature.dat'):
        self._qha.write_volume_temperature(filename=filename)

    def get_thermal_expansion(self):
        """Returns thermal expansion coefficients at temperatures"""
        return self._qha.get_thermal_expansion()

    def plot_thermal_expansion(self, plt=None):
        return self._qha.plot_thermal_expansion(plt=plt)

    def plot_pdf_thermal_expansion(self,
                                   filename='thermal_expansion.pdf'):
        self._qha.plot_pdf_thermal_expansion(filename=filename)

    def write_thermal_expansion(self,
                                filename='thermal_expansion.dat'):
        self._qha.write_thermal_expansion(filename=filename)

    def get_volume_expansion(self):
        """Return volume expansions at temperatures"""
        return self._qha.get_volume_expansion()

    def plot_volume_expansion(self, exp_data=None, symbol='o', plt=None):
        return self._qha.plot_volume_expansion(exp_data=exp_data,
                                               symbol=symbol,
                                               plt=plt)

    def plot_pdf_volume_expansion(self,
                                  exp_data=None,
                                  symbol='o',
                                  filename='volume_expansion.pdf'):
        self._qha.plot_pdf_volume_expansion(exp_data=exp_data,
                                            symbol=symbol,
                                            filename=filename)

    def write_volume_expansion(self, filename='volume_expansion.dat'):
        self._qha.write_volume_expansion(filename=filename)

    def get_gibbs_temperature(self):
        """Returns Gibbs free energies at temperatures"""
        return self._qha.get_gibbs_temperature()

    def plot_gibbs_temperature(self,
                               plt=None,
                               xlabel='Temperature (K)',
                               ylabel='Gibbs free energy'):
        return self._qha.plot_gibbs_temperature(plt=plt,
                                                xlabel=xlabel,
                                                ylabel=ylabel)

    def plot_pdf_gibbs_temperature(self, filename='gibbs-temperature.pdf'):
        self._qha.plot_pdf_gibbs_temperature(filename=filename)

    def write_gibbs_temperature(self, filename='gibbs-temperature.dat'):
        self._qha.write_gibbs_temperature(filename=filename)

    def get_bulk_modulus_temperature(self):
        """Returns bulk modulus at temperatures"""
        return self._qha.get_bulk_modulus_temperature()

    def plot_bulk_modulus_temperature(self,
                                      plt=None,
                                      xlabel='Temperature (K)',
                                      ylabel='Bulk modulus'):
        return self._qha.plot_bulk_modulus_temperature(plt=plt,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel)

    def plot_pdf_bulk_modulus_temperature(self,
                                          filename='bulk_modulus-temperature.pdf'):
        self._qha.plot_pdf_bulk_modulus_temperature(filename=filename)

    def write_bulk_modulus_temperature(self,
                                       filename='bulk_modulus-temperature.dat'):
        self._qha.write_bulk_modulus_temperature(filename=filename)

    def get_heat_capacity_P_numerical(self):
        """Returns heat capacities at constant pressure at temperatures
        
        These values are calculated by -T*d^2G/dT^2.
        """
        return self._qha.get_heat_capacity_P_numerical()

    def plot_heat_capacity_P_numerical(self, Z=1, exp_data=None, plt=None):
        return self._qha.plot_heat_capacity_P_numerical(Z=Z,
                                                        exp_data=exp_data,
                                                        plt=plt)

    def plot_pdf_heat_capacity_P_numerical(self,
                                           exp_data=None,
                                           filename='Cp-temperature.pdf'):
        self._qha.plot_pdf_heat_capacity_P_numerical(exp_data=exp_data,
                                                     filename=filename)

    def write_heat_capacity_P_numerical(self, filename='Cp-temperature.dat'):
        self._qha.write_heat_capacity_P_numerical(filename=filename)

    def get_heat_capacity_P_polyfit(self):
        """Returns heat capacities at constant pressure at temperatures

        These values are calculated from the values obtained by polynomial
        fittings of Cv and S.
        """
        return self._qha.get_heat_capacity_P_polyfit()

    def plot_heat_capacity_P_polyfit(self, exp_data=None, Z=1, plt=None):
        return self._qha.plot_heat_capacity_P_polyfit(Z=Z,
                                                      exp_data=exp_data,
                                                      plt=plt)

    def plot_pdf_heat_capacity_P_polyfit(self,
                                         exp_data=None,
                                         filename='Cp-temperature_polyfit.pdf'):
        self._qha.plot_pdf_heat_capacity_P_polyfit(exp_data=exp_data,
                                                   filename=filename)

    def write_heat_capacity_P_polyfit(self,
                                      filename='Cp-temperature_polyfit.dat',
                                      filename_ev='entropy-volume.dat',
                                      filename_cvv='Cv-volume.dat',
                                      filename_dsdvt='dsdv-temperature.dat'):
        self._qha.write_heat_capacity_P_polyfit(filename=filename,
                                                filename_ev=filename_ev,
                                                filename_cvv=filename_cvv,
                                                filename_dsdvt=filename_dsdvt)

    def get_gruneisen_temperature(self):
        """Returns Gruneisen parameters at temperatures"""
        return self._qha.get_gruneisen_temperature()

    def plot_gruneisen_temperature(self, plt=None):
        return self._qha.plot_gruneisen_temperature(plt=plt)

    def plot_pdf_gruneisen_temperature(self,
                                       filename='gruneisen-temperature.pdf'):
        self._qha.plot_pdf_gruneisen_temperature(filename=filename)

    def write_gruneisen_temperature(self, filename='gruneisen-temperature.dat'):
        self._qha.write_gruneisen_temperature(filename=filename)
