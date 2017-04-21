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
        self._electronic_energies = electronic_energies
        self._eos = get_eos(eos)
        (self._energy,
         self._bulk_modulus,
         self._b_prime,
         self._volume) = fit_to_eos(volumes,
                                    electronic_energies,
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
        plt.plot(vols, self._electronic_energies, 'bo', markersize=4)
        volume_points = np.linspace(min(vols), max(vols), 201)
        plt.plot(volume_points, self._eos(volume_points, *ep), 'r-')
        return plt


class QHA(object):
    def __init__(self,
                 volumes, # Angstrom^3
                 electronic_energies, # eV
                 temperatures,
                 cv,        # J/K/mol
                 entropy,   # J/K/mol
                 fe_phonon, # kJ/mol
                 eos='vinet',
                 t_max=None):
        self._volumes = np.array(volumes)
        self._electronic_energies = np.array(electronic_energies)

        self._all_temperatures = np.array(temperatures)
        self._cv = np.array(cv)
        self._entropy = np.array(entropy)
        self._fe_phonon = np.array(fe_phonon) / EvTokJmol

        self._eos = get_eos(eos)
        self._t_max = t_max

        self._temperatures = []
        self._equiv_volumes = []
        self._equiv_energies = []
        self._equiv_bulk_modulus = []
        self._equiv_parameters = []
        self._free_energies = []

        self._thermal_expansions = None
        self._volume_expansions = None
        self._cp_numerical = None
        self._volume_entropy_parameters = None
        self._volume_cv_parameters = None
        self._volume_entropy = None
        self._volume_cv = None
        self._cp_polyfit = None
        self._dsdv = None
        self._gruneisen_parameters = None

    def run(self, verbose=False):
        if verbose:
            print(("#%11s" + "%14s" * 4) % ("T", "E_0", "B_0", "B'_0", "V_0"))

        max_t_index = self._get_max_t_index(self._all_temperatures)

        for i in range(max_t_index + 2):
            t = self._all_temperatures[i]
            fe = []
            for j, e in enumerate(self._electronic_energies):
                fe.append(e + self._fe_phonon[i][j])
            self._free_energies.append(fe)

            ee, eb, ebp, ev = fit_to_eos(self._volumes, fe, self._eos)
            if ee is None:
                continue
            else:
                ep = [ee, eb, ebp, ev]
                self._temperatures.append(t)
                self._equiv_volumes.append(ev)
                self._equiv_energies.append(ee)
                self._equiv_bulk_modulus.append(eb * EVAngstromToGPa)
                self._equiv_parameters.append(ep)

                if verbose:
                    print(("%14.6f" * 5) %
                          (t, ep[0], ep[1] * EVAngstromToGPa, ep[2], ep[3]))

        self._max_t_index = self._get_max_t_index(self._temperatures)
        self._set_volume_expansion()
        self._set_thermal_expansion() # len = len(t) - 1
        self._set_heat_capacity_P_numerical() # len = len(t) - 2
        self._set_heat_capacity_P_polyfit()
        self._set_gruneisen_parameter() # To be run after thermal expansion.

    def plot(self, thin_number=10, volume_temp_exp=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.5))
        plt.subplot(1, 3, 1)
        self._plot_helmholtz_volume(plt, thin_number=thin_number)
        plt.subplot(1, 3, 2)
        self._plot_volume_temperature(plt, exp_data=volume_temp_exp)
        plt.subplot(1, 3, 3)
        self._plot_thermal_expansion(plt)
        plt.tight_layout()
        return plt

    def get_eos(self):
        return self._eos

    def get_helmholtz_volume(self):
        return self._free_energies[:self._max_t_index]

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

        self._plot_helmholtz_volume(plt, thin_number=thin_number)
        plt.savefig(filename)
        plt.close()

    def write_helmholtz_volume(self, filename='helmholtz-volume.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("# Temperature: %f\n" % self._temperatures[i])
            w.write("# Parameters: %f %f %f %f\n" %
                    tuple(self._equiv_parameters[i]))
            for j, v in enumerate(self._volumes):
                w.write("%20.15f %25.15f\n" % (v, self._free_energies[i][j]))
            w.write("\n\n")
        w.close()

    def get_volume_temperature(self):
        return self._equiv_volumes[:self._max_t_index]

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

        self._plot_volume_temperature(plt, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_volume_temperature(self, filename='volume-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("%25.15f %25.15f\n" % (self._temperatures[i],
                                           self._equiv_volumes[i]))
        w.close()

    def get_thermal_expansion(self):
        return self._thermal_expansions[:self._max_t_index]

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

        self._plot_thermal_expansion(plt)
        plt.savefig(filename)
        plt.close()

    def write_thermal_expansion(self, filename='thermal_expansion.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("%25.15f %25.15f\n" % (self._temperatures[i],
                                           self._thermal_expansions[i]))
        w.close()

    def get_volume_expansion(self):
        return self._volume_expansions[:self._max_t_index]

    def plot_volume_expansion(self, exp_data=None, symbol='o', _plt=None):
        if self._temperatures[self._max_t_index] > 300:
            if plt is None:
                import matplotlib.pyplot as _plt
                self._plot_volume_expansion(_plt,
                                            exp_data=exp_data,
                                            symbol=symbol)
                return plt
            else:
                self._plot_volume_expansion(plt,
                                            exp_data=exp_data,
                                            symbol=symbol)
        else:
            return None

    def plot_pdf_volume_expansion(self,
                                  exp_data=None,
                                  symbol='o',
                                  filename='volume_expansion.pdf'):
        if self._temperatures[self._max_t_index] > 300:
            import matplotlib.pyplot as plt
            plt.rcParams['backend'] = 'PDF'
            plt.rcParams['pdf.fonttype'] = 3
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['axes.labelsize'] = 18
            plt.rcParams['figure.subplot.left'] = 0.15
            plt.rcParams['figure.subplot.bottom'] = 0.15
            plt.rcParams['figure.figsize'] = 8, 6

            self._plot_volume_expansion(plt,
                                        exp_data=exp_data,
                                        symbol=symbol)
            plt.savefig(filename)
            plt.close()

    def write_volume_expansion(self, filename='volume_expansion.dat'):
        if self._temperatures[self._max_t_index] > 300:
            w = open(filename, 'w')
            for i in range(self._max_t_index):
                w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                               self._volume_expansions[i]))
            w.close()

    def get_gibbs_temperature(self):
        return self._equiv_energies[:self._max_t_index]

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

        self._plot_gibbs_temperature(plt)
        plt.savefig(filename)
        plt.close()

    def write_gibbs_temperature(self, filename='gibbs-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                           self._equiv_energies[i]))
        w.close()

    def get_bulk_modulus_temperature(self):
        return self._equiv_bulk_modulus[:self._max_t_index]

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

        self._plot_bulk_modulus_temperature(plt)
        plt.savefig(filename)
        plt.close()

    def write_bulk_modulus_temperature(self,
                                       filename='bulk_modulus-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                           self._equiv_bulk_modulus[i]))
        w.close()

    def get_heat_capacity_P_numerical(self, exp_data=None):
        return self._cp_numerical[:self._max_t_index]

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

        self._plot_heat_capacity_P_numerical(plt, exp_data=exp_data)
        plt.savefig(filename)
        plt.close()

    def write_heat_capacity_P_numerical(self, filename='Cp-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i],
                                           self._cp_numerical[i]))
        w.close()

    def get_heat_capacity_P_polyfit(self):
        return self._cp_polyfit[:self._max_t_index]

    def plot_heat_capacity_P_polyfit(self, Z=1, exp_data=None, plt=None):
        if plt is None:
            import matplotlib.pyplot as _plt
            self._plot_heat_capacity_P_polyfit(_plt, Z=Z, exp_data=exp_data)
            return _plt
        else:
            self._plot_heat_capacity_P_polyfit(plt, Z=Z, exp_data=exp_data)

    def plot_pdf_heat_capacity_P_polyfit(self,
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
        for i in range(1, self._max_t_index):
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
        for i in range(self._max_t_index):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i],
                                           self._cp_polyfit[i]))
        w.close()

        w = open(filename_dsdvt, 'w') # GPa
        for i in range(self._max_t_index):
            w.write("%20.15f %20.15f\n" % (self._temperatures[i],
                                           self._dsdv[i] * 1e21 / Avogadro))
        w.close()

    def get_gruneisen_temperature(self):
        return self._gruneisen_parameters[:self._max_t_index]

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

        self._plot_gruneisen_temperature(plt)
        plt.savefig(filename)
        plt.close()

    def write_gruneisen_temperature(self, filename='gruneisen-temperature.dat'):
        w = open(filename, 'w')
        for i in range(self._max_t_index):
            w.write("%20.15f %25.15f\n" % (self._temperatures[i],
                                           self._gruneisen_parameters[i]))
        w.close()

    def _plot_helmholtz_volume(self,
                               plt,
                               thin_number=10,
                               xlabel=r'Volume $(\AA^3)$',
                               ylabel='Free energy (eV)'):
        volume_points = np.linspace(min(self._volumes),
                                    max(self._volumes),
                                    201)
        selected_volumes = []
        selected_energies = []

        thin_index = 0
        for i, t in enumerate(self._temperatures[:self._max_t_index]):
            if i % thin_number == 0:
                selected_volumes.append(self._equiv_volumes[i])
                selected_energies.append(self._equiv_energies[i])
                plt.plot(self._volumes,
                         self._free_energies[i],
                         'bo', markeredgecolor='b', markersize=3)
                plt.plot(volume_points,
                         self._eos(volume_points,
                                   *self._equiv_parameters[i]), 'b-')
                thin_index = i

        for i, j in enumerate((0, thin_index)):
            plt.text(self._volumes[-2],
                     self._free_energies[j][-1] + (1 - i * 2) * 0.1 - 0.05,
                     "%dK" % int(self._temperatures[j]),
                     fontsize=8)

        plt.plot(selected_volumes,
                 selected_energies,
                 'ro-', markeredgecolor='r', markersize=3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def _plot_volume_temperature(self,
                                 plt,
                                 exp_data=None,
                                 xlabel='Temperature (K)',
                                 ylabel=r'Volume $(\AA^3)$'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._max_t_index],
                 self._equiv_volumes[:self._max_t_index],
                 'r-')
        # exp
        if exp_data:
            plt.plot(exp_data[0], exp_data[1], 'ro')

    def _plot_thermal_expansion(
            self,
            plt,
            xlabel='Temperature (K)',
            ylabel=r'Thermal expansion $\times 10^6 (\mathrm{K}^{-1})$'):

        beta = np.array(self._thermal_expansions) * 1e6
        plt.plot(self._temperatures[:self._max_t_index],
                 beta[:self._max_t_index],
                 'r-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def _plot_volume_expansion(
            self,
            plt,
            exp_data=None,
            symbol='o',
            xlabel='Temperature (K)',
            ylabel=r'Volume expansion $\Delta L/L_0 \, (L=V^{\,1/3})$'):
        plt.plot(self._temperatures[:self._max_t_index],
                 self._volume_expansions[:self._max_t_index],
                 'r-')

        if exp_data:
            plt.plot(exp_data[0],
                     (exp_data[1] / exp_data[1][0]) ** (1.0 / 3) - 1,
                     symbol)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(self._temperatures[0],
                 self._temperatures[self._max_t_index])

    def _plot_gibbs_temperature(self,
                                plt,
                                xlabel='Temperature (K)',
                                ylabel='Gibbs free energy (eV)'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._max_t_index],
                 self._equiv_energies[:self._max_t_index],
                 'r-')

    def _plot_bulk_modulus_temperature(self,
                                       plt,
                                       xlabel='Temperature (K)',
                                       ylabel='Bulk modulus (GPa)'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._max_t_index],
                 self._equiv_bulk_modulus[:self._max_t_index],
                 'r-')

    def _plot_heat_capacity_P_numerical(
            self,
            plt,
            Z=1,
            exp_data=None,
            xlabel='Temperature (K)',
            ylabel=r'$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._max_t_index],
                 np.array(self._cp_numerical[:self._max_t_index]) / Z,
                 'r-')

        # exp
        if exp_data:
            plt.plot(exp_data[0], exp_data[1], 'ro')

    def _plot_heat_capacity_P_polyfit(
            self,
            plt,
            Z=1,
            exp_data=None,
            xlabel='Temperature (K)',
            ylabel=r'$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._max_t_index],
                 np.array(self._cp_polyfit[:self._max_t_index]) / Z,
                 'r-')

        # exp
        if exp_data:
            plt.plot(exp_data[0], exp_data[1], 'ro')

    def _plot_gruneisen_temperature(self,
                                    plt,
                                    xlabel='Temperature (K)',
                                    ylabel='Gruneisen parameter'):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self._temperatures[:self._max_t_index],
                 self._gruneisen_parameters[:self._max_t_index],
                 'r-')

    def _set_thermal_expansion(self):
        beta = [0.]
        dt = self._temperatures[1] - self._temperatures[0]
        for i in range(self._max_t_index):
            beta.append((self._equiv_volumes[i + 2] - self._equiv_volumes[i]) /
                        (2 * dt) / self._equiv_volumes[i + 1])

        self._thermal_expansions = beta

    def _set_volume_expansion(self):
        if self._temperatures[self._max_t_index] > 300:
            l = np.array(self._equiv_volumes) ** (1.0 / 3)
            for i in range(self._max_t_index):
                t = self._temperatures[i]
                if (abs(t - 300) <
                    (self._temperatures[1] - self._temperatures[0]) / 10):
                    l_0 = (self._equiv_volumes[i]) ** (1.0 / 3)
                    break

            self._volume_expansions = l / l_0 - 1

    def _set_heat_capacity_P_numerical(self):
        cp = []
        g = np.array(self._equiv_energies) * EvTokJmol * 1000
        cp.append(0.0)
        cp.append(0.0)
        dt = self._temperatures[1] - self._temperatures[0]
        for i in range(2, self._max_t_index):
            cp.append(-(g[i + 2] - 2 * g[i] + g[i - 2]) /
                       (dt ** 2) / 4 * self._temperatures[i])
        self._cp_numerical = cp

    def _set_heat_capacity_P_polyfit(self):
        cp = [0.0]
        dsdv = [0.0]
        self._volume_entropy_parameters = []
        self._volume_cv_parameters = []
        self._volume_entropy = []
        self._volume_cv = []

        dt = self._temperatures[1] - self._temperatures[0]
        for j in range(1, self._max_t_index):
            t = self._temperatures[j]
            x = self._equiv_volumes[j]

            parameters = np.polyfit(self._volumes, self._cv[j], 4)
            cv_p = np.dot(parameters, np.array([x**4, x**3, x**2, x, 1]))
            self._volume_cv_parameters.append(parameters)

            parameters = np.polyfit(self._volumes, self._entropy[j], 4)
            dsdv_t = np.dot(parameters[:4], np.array(
                [4 * x**3, 3 * x**2, 2 * x, 1]))
            self._volume_entropy_parameters.append(parameters)

            dvdt = (self._equiv_volumes[j + 1] -
                    self._equiv_volumes[j - 1]) / dt / 2

            cp.append(cv_p + t * dvdt * dsdv_t)
            dsdv.append(dsdv_t)

            self._volume_cv.append(np.array([self._volumes, self._cv[j]]).T)
            self._volume_entropy.append(np.array([self._volumes,
                                                  self._entropy[j]]).T)

        self._cp_polyfit = cp
        self._dsdv = dsdv

    def _set_gruneisen_parameter(self):
        gamma = [0]
        for i in range(1, self._max_t_index):
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

    def _get_max_t_index(self, temperatures):
        if self._t_max is None:
            return len(self._all_temperatures) - 3
        else:
            max_t_index = 0

            for i, t in enumerate(temperatures):
                if self._t_max + 1e-5 < t:
                    max_t_index = i + 1
                    break

            if (max_t_index > len(temperatures) - 3 or
                max_t_index < 2):
                max_t_index = len(temperatures) - 3

            return max_t_index
