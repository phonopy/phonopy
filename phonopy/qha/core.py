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

import numpy as np
from phonopy.units import Avogadro, EvTokJmol, EVAngstromToGPa
from phonopy.qha.eos import get_eos, fit_to_eos


class BulkModulus(object):
    def __init__(self,
                 volumes,
                 electronic_energies,
                 eos='vinet'):
        self._volumes = volumes
        if electronic_energies.ndim == 1:
            self._electronic_energies = electronic_energies
        else:
            self._electronic_energies = electronic_energies[0]
        self._eos = get_eos(eos)

        (self._energy,
         self._bulk_modulus,
         self._b_prime,
         self._volume) = fit_to_eos(volumes,
                                    self._electronic_energies,
                                    self._eos)

    def get_bulk_modulus(self):
        return self._bulk_modulus

    def get_equilibrium_volume(self):
        return self._volume

    def get_b_prime(self):
        return self._b_prime

    def get_energy(self):
        return self._energy

    def get_parameters(self):
        return (self._energy,
                self._bulk_modulus,
                self._b_prime,
                self._volume)

    def get_eos(self):
        return self._eos

    def plot(self):
        import matplotlib.pyplot as plt
        ep = self.get_parameters()
        vols = self._volumes
        volume_points = np.linspace(min(vols), max(vols), 201)
        plt.plot(volume_points, self._eos(volume_points, *ep), 'r-')
        plt.plot(vols, self._electronic_energies, 'bo', markersize=4)
        return plt


class QHA(object):
    def __init__(self,
                 volumes,              # Angstrom^3
                 electronic_energies,  # eV
                 temperatures,
                 cv,                   # J/K/mol
                 entropy,              # J/K/mol
                 fe_phonon,            # kJ/mol
                 eos='vinet',
                 t_max=None,
                 energy_plot_factor=None):
        self._volumes = np.array(volumes)
        self._electronic_energies = np.array(electronic_energies)

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

    def run(self, verbose=False):
        """Fit parameters to EOS at temperatures

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
            fe = [ph_e + el_e
                  for ph_e, el_e in zip(self._fe_phonon[i], el_energy)]
            ep = fit_to_eos(self._volumes, fe, self._eos)

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
                    print(("%14.6f" * 5) %
                          (t, ep[0], ep[1] * EVAngstromToGPa, ep[2], ep[3]))

        self._free_energies = np.array(free_energies)
        self._temperatures = np.array(temperatures)
        self._equiv_parameters = np.array(parameters)
        self._equiv_volumes = np.array(self._equiv_parameters[:, 3])
        self._equiv_energies = np.array(self._equiv_parameters[:, 0])
        self._equiv_bulk_modulus = np.array(
            self._equiv_parameters[:, 1] * EVAngstromToGPa)

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
        assert(self._len + 1 == self._num_elems)

    def plot(self, thin_number=10, volume_temp_exp=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.5))
        ax = plt.subplot(1, 3, 1)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        self._plot_helmholtz_volume(plt, thin_number=thin_number)
        ax = plt.subplot(1, 3, 2)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        self._plot_volume_temperature(plt, exp_data=volume_temp_exp)
        ax = plt.subplot(1, 3, 3)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        self._plot_thermal_expansion(plt)
        plt.tight_layout()
        return plt

    def get_helmholtz_volume(self):
        return self._free_energies[:self._len]

    def plot_helmholtz_volume(self,
                              thin_number=10,
                              plt=None,
                              xlabel=r'Volume $(\AA^3)$',
                              ylabel='Free energy'):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_helmholtz_volume(_plt,
                                        thin_number=thin_number,
                                        xlabel=xlabel,
                                        ylabel=ylabel)
            return _plt
        else:
            self._plot_helmholtz_volume(plt,
                                        thin_number=thin_number,
                                        xlabel=xlabel,
                                        ylabel=ylabel)

    def plot_pdf_helmholtz_volume(self,
                                  thin_number=10,
                                  filename='helmholtz-volume.pdf'):
        import matplotlib.pyplot as plt

        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.25
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 4, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_helmholtz_volume(plt, thin_number=thin_number)
        plt.savefig(filename)
        plt.close()

    def write_helmholtz_volume(self, filename='helmholtz-volume.dat'):
        w = open(filename, 'w')
        for i, (t, ep, fe) in enumerate(zip(self._temperatures,
                                            self._equiv_parameters,
                                            self._free_energies)):
            if i == self._len:
                break

            w.write("# Temperature: %f\n" % t)
            w.write("# Parameters: %f %f %f %f\n" % tuple(ep))
            for j, v in enumerate(self._volumes):
                w.write("%20.15f %25.15f\n" % (v, fe[j]))
            w.write("\n\n")
        w.close()

    def get_volume_temperature(self):
        return self._equiv_volumes[:self._len]

    def plot_volume_temperature(self, exp_data=None, plt=None):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_volume_temperature(_plt, exp_data=exp_data)
            return _plt
        else:
            self._plot_volume_temperature(plt, exp_data=exp_data)

    def plot_pdf_volume_temperature(self,
                                    exp_data=None,
                                    filename='volume-temperature.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_volume_temperature(plt, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_volume_temperature(self, filename='volume-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%25.15f %25.15f\n" % (self._temperatures[i],
                                           self._equiv_volumes[i]))
        w.close()

    def get_thermal_expansion(self):
        return self._thermal_expansions[:self._len]

    def plot_thermal_expansion(self, plt=None):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_thermal_expansion(_plt)
            return _plt
        else:
            self._plot_thermal_expansion(plt)

    def plot_pdf_thermal_expansion(self, filename='thermal_expansion.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_thermal_expansion(plt)
        plt.savefig(filename)
        plt.close()

    def write_thermal_expansion(self, filename='thermal_expansion.dat'):
        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%25.15f %25.15f\n" % (self._temperatures[i],
                                           self._thermal_expansions[i]))
        w.close()

    def get_gibbs_temperature(self):
        return self._equiv_energies[:self._len]

    def plot_gibbs_temperature(self,
                               plt=None,
                               xlabel='Temperature (K)',
                               ylabel='Gibbs free energy'):

        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_gibbs_temperature(_plt, xlabel=xlabel, ylabel=ylabel)
            return _plt
        else:
            self._plot_gibbs_temperature(plt, xlabel=xlabel, ylabel=ylabel)

    def plot_pdf_gibbs_temperature(self, filename='gibbs-temperature.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_gibbs_temperature(plt)
        plt.savefig(filename)
        plt.close()

    def write_gibbs_temperature(self, filename='gibbs-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                           self._equiv_energies[i]))
        w.close()

    def get_bulk_modulus_temperature(self):
        return self._equiv_bulk_modulus[:self._len]

    def plot_bulk_modulus_temperature(self,
                                      plt=None,
                                      xlabel='Temperature (K)',
                                      ylabel='Bulk modulus'):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_bulk_modulus_temperature(_plt,
                                                xlabel=xlabel,
                                                ylabel=ylabel)
            return _plt
        else:
            self._plot_bulk_modulus_temperature(plt,
                                                xlabel=xlabel,
                                                ylabel=ylabel)

    def plot_pdf_bulk_modulus_temperature(
            self,
            filename='bulk_modulus-temperature.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_bulk_modulus_temperature(plt)
        plt.savefig(filename)
        plt.close()

    def write_bulk_modulus_temperature(
            self,
            filename='bulk_modulus-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                           self._equiv_bulk_modulus[i]))
        w.close()

    def get_heat_capacity_P_numerical(self, exp_data=None):
        return self._cp_numerical[:self._len]

    def plot_heat_capacity_P_numerical(self, Z=1, exp_data=None, plt=None):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_heat_capacity_P_numerical(_plt, Z=Z, exp_data=exp_data)
            return _plt
        else:
            import matplotlib.pyplot as plt

    def plot_pdf_heat_capacity_P_numerical(self,
                                           exp_data=None,
                                           filename='Cp-temperature.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_heat_capacity_P_numerical(plt, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_heat_capacity_P_numerical(self, filename='Cp-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i],
                                           self._cp_numerical[i]))
        w.close()

    def get_heat_capacity_P_polyfit(self):
        return self._cp_polyfit[:self._len]

    def plot_heat_capacity_P_polyfit(self, Z=1, exp_data=None, plt=None):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_heat_capacity_P_polyfit(_plt, Z=Z, exp_data=exp_data)
            return _plt
        else:
            self._plot_heat_capacity_P_polyfit(plt, Z=Z, exp_data=exp_data)

    def plot_pdf_heat_capacity_P_polyfit(
            self,
            exp_data=None,
            filename='Cp-temperature_polyfit.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_heat_capacity_P_polyfit(plt,
                                           exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_heat_capacity_P_polyfit(self,
                                      filename='Cp-temperature_polyfit.dat',
                                      filename_ev='entropy-volume.dat',
                                      filename_cvv='Cv-volume.dat',
                                      filename_dsdvt='dsdv-temperature.dat'):
        wve = open(filename_ev, 'w')
        wvcv = open(filename_cvv, 'w')
        for i in range(1, self._len):
            t = self._temperatures[i]
            wve.write("# temperature %20.15f\n" % t)
            wve.write("# %20.15f %20.15f %20.15f %20.15f %20.15f\n" %
                      tuple(self._volume_cv_parameters[i - 1]))
            wvcv.write("# temperature %20.15f\n" % t)
            wvcv.write("# %20.15f %20.15f %20.15f %20.15f %20.15f\n" %
                       tuple(self._volume_entropy_parameters[i - 1]))
            for ve, vcv in zip(self._volume_entropy[i - 1],
                               self._volume_cv[i - 1]):
                wve.write("%20.15f %20.15f\n" % tuple(ve))
                wvcv.write("%20.15f %20.15f\n" % tuple(vcv))
            wve.write("\n\n")
            wvcv.write("\n\n")
        wve.close()
        wvcv.close()

        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i],
                                           self._cp_polyfit[i]))
        w.close()

        w = open(filename_dsdvt, 'w')  # GPa
        for i in range(self._len):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i],
                                           self._dsdv[i] * 1e21 / Avogadro))
        w.close()

    def get_gruneisen_temperature(self):
        return self._gruneisen_parameters[:self._len]

    def plot_gruneisen_temperature(self, plt=None):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_gruneisen_temperature(_plt)
            return _plt
        else:
            self._plot_gruneisen_temperature(plt)

    def plot_pdf_gruneisen_temperature(self,
                                       filename='gruneisen-temperature.pdf'):
        import matplotlib.pyplot as plt
        plt.rcParams['backend'] = 'PDF'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['figure.subplot.left'] = 0.15
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.figsize'] = 8, 6

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._plot_gruneisen_temperature(plt)
        plt.savefig(filename)
        plt.close()

    def write_gruneisen_temperature(self,
                                    filename='gruneisen-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._len):
            w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                           self._gruneisen_parameters[i]))
        w.close()

    def _plot_helmholtz_volume(self,
                               plt,
                               thin_number=10,
                               xlabel=r'Volume $(\AA^3)$',
                               ylabel='Free energy'):
        if self._energy_plot_factor is None:
            _energy_plot_factor = 1
            _ylabel = ylabel + ' (eV)'
        else:
            _energy_plot_factor = self._energy_plot_factor
            _ylabel = ylabel

        volume_points = np.linspace(min(self._volumes),
                                    max(self._volumes),
                                    201)
        selected_volumes = []
        selected_energies = []

        thin_index = 0
        for i, t in enumerate(self._temperatures[:self._len]):
            if i % thin_number == 0:
                selected_volumes.append(self._equiv_volumes[i])
                selected_energies.append(self._equiv_energies[i])

        for i, t in enumerate(self._temperatures[:self._len]):
            if t >= 298:
                if i > 0:
                    de = self._equiv_energies[i] - self._equiv_energies[i - 1]
                    dt = t - self._temperatures[i - 1]
                    e0 = ((298 - self._temperatures[i - 1]) / dt * de +
                          self._equiv_energies[i - 1])
                else:
                    e0 = 0
                break
        e0 *= _energy_plot_factor

        for i, t in enumerate(self._temperatures[:self._len]):
            if i % thin_number == 0:
                plt.plot(self._volumes,
                         np.array(self._free_energies[i]) * _energy_plot_factor
                         - e0,
                         'bo', markeredgecolor='b', markersize=3)
                plt.plot(volume_points,
                         self._eos(volume_points, * self._equiv_parameters[i])
                         * _energy_plot_factor - e0, 'b-')
                thin_index = i

        for i, j in enumerate((0, thin_index)):
            plt.text(self._volumes[-2],
                     (self._free_energies[j, -1] + (1 - i * 2) * 0.1 - 0.05) *
                     _energy_plot_factor - e0,
                     "%dK" % int(self._temperatures[j]),
                     fontsize=8)

        plt.plot(selected_volumes,
                 np.array(selected_energies) * _energy_plot_factor - e0,
                 'ro-', markeredgecolor='r', markersize=3)
        plt.xlabel(xlabel)
        plt.ylabel(_ylabel)

    def _plot_volume_temperature(self,
                                 plt,
                                 exp_data=None,
                                 xlabel='Temperature (K)',
                                 ylabel=r'Volume $(\AA^3)$'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._len],
                 self._equiv_volumes[:self._len],
                 'r-')
        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])
        # exp
        if exp_data:
            plt.plot(exp_data[0], exp_data[1], 'ro')

    def _plot_thermal_expansion(
            self,
            plt,
            xlabel='Temperature (K)',
            ylabel=r'Thermal expansion $(10^{-6} \mathrm{K}^{-1})$'):

        beta = np.array(self._thermal_expansions) * 1e6
        plt.plot(self._temperatures[:self._len],
                 beta[:self._len],
                 'r-')
        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def _plot_gibbs_temperature(self,
                                plt,
                                xlabel='Temperature (K)',
                                ylabel='Gibbs free energy (eV)'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._len],
                 self._equiv_energies[:self._len],
                 'r-')
        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])

    def _plot_bulk_modulus_temperature(self,
                                       plt,
                                       xlabel='Temperature (K)',
                                       ylabel='Bulk modulus (GPa)'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._len],
                 self._equiv_bulk_modulus[:self._len],
                 'r-')
        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])

    def _plot_heat_capacity_P_numerical(
            self,
            plt,
            Z=1,
            exp_data=None,
            xlabel='Temperature (K)',
            ylabel=r'$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._len],
                 np.array(self._cp_numerical[:self._len]) / Z,
                 'r-')

        # exp
        if exp_data:
            plt.plot(exp_data[0], exp_data[1], 'ro')

        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])

    def _plot_heat_capacity_P_polyfit(
            self,
            plt,
            Z=1,
            exp_data=None,
            xlabel='Temperature (K)',
            ylabel=r'$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._len],
                 np.array(self._cp_polyfit[:self._len]) / Z,
                 'r-')

        # exp
        if exp_data:
            plt.plot(exp_data[0], exp_data[1], 'ro')

        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])

    def _plot_gruneisen_temperature(self,
                                    plt,
                                    xlabel='Temperature (K)',
                                    ylabel='Gruneisen parameter'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._len],
                 self._gruneisen_parameters[:self._len],
                 'r-')
        plt.xlim(self._temperatures[0],
                 self._temperatures[self._len - 1])

    def _set_thermal_expansion(self):
        beta = [0.]
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
            parameters = np.polyfit(self._temperatures[i - 1:i + 2],
                                    g[i - 1: i + 2], 2)
            cp.append(- (2 * parameters[0]) * t)

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

            parameters = np.polyfit(self._volumes, self._cv[j], 4)
            cv_p = np.dot(parameters, np.array([x**4, x**3, x**2, x, 1]))
            self._volume_cv_parameters.append(parameters)

            parameters = np.polyfit(self._volumes, self._entropy[j], 4)
            dsdv_t = np.dot(parameters[:4], np.array(
                [4 * x**3, 3 * x**2, 2 * x, 1]))
            self._volume_entropy_parameters.append(parameters)

            parameters = np.polyfit(self._temperatures[j - 1:j + 2],
                                    self._equiv_volumes[j - 1: j + 2], 2)
            dvdt = parameters[0] * t + parameters[1]

            cp.append(cv_p + t * dvdt * dsdv_t)
            dsdv.append(dsdv_t)

            self._volume_cv.append(np.array([self._volumes, self._cv[j]]).T)
            self._volume_entropy.append(np.array([self._volumes,
                                                  self._entropy[j]]).T)

        self._cp_polyfit = cp
        self._dsdv = dsdv

    def _set_gruneisen_parameter(self):
        gamma = [0]
        for i in range(1, self._num_elems - 1):
            v = self._equiv_volumes[i]
            kt = self._equiv_bulk_modulus[i]
            beta = self._thermal_expansions[i]
            parameters = np.polyfit(self._volumes, self._cv[i], 4)
            cv = (np.dot(parameters, [v**4, v**3, v**2, v, 1]) /
                  v / 1000 / EvTokJmol * EVAngstromToGPa)
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
