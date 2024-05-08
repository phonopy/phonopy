"""Collection of physical units."""

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

from math import pi, sqrt

kb_J = 1.3806504e-23  # [J/K]
PlanckConstant = 4.13566733e-15  # [eV s]
Hbar = PlanckConstant / (2 * pi)  # [eV s]
Avogadro = 6.02214179e23
SpeedOfLight = 299792458  # [m/s]
AMU = 1.6605402e-27  # [kg]
Newton = 1.0  # [kg m / s^2]
Joule = 1.0  # [kg m^2 / s^2]
EV = 1.60217733e-19  # [J]
Angstrom = 1.0e-10  # [m]
THz = 1.0e12  # [/s]
Mu0 = 4.0e-7 * pi  # [Hartree/m]
Epsilon0 = 1.0 / Mu0 / SpeedOfLight**2  # [C^2 / N m^2]
Me = 9.10938215e-31

Bohr = 4e10 * pi * Epsilon0 * Hbar**2 / Me  # Bohr radius [A] 0.5291772
Hartree = Me * EV / 16 / pi**2 / Epsilon0**2 / Hbar**2  # Hartree [eV] 27.211398
Rydberg = Hartree / 2  # Rydberg [eV] 13.6056991

THzToEv = PlanckConstant * 1e12  # [eV]
Kb = kb_J / EV  # [eV/K] 8.6173383e-05
THzToCm = 1.0e12 / (SpeedOfLight * 100)  # [cm^-1] 33.356410
CmToEv = THzToEv / THzToCm  # [eV] 1.2398419e-4
VaspToEv = sqrt(EV / AMU) / Angstrom / (2 * pi) * PlanckConstant  # [eV] 6.46541380e-2
VaspToTHz = sqrt(EV / AMU) / Angstrom / (2 * pi) / 1e12  # [THz] 15.633302
VaspToCm = VaspToTHz * THzToCm  # [cm^-1] 521.47083
EvTokJmol = EV / 1000 * Avogadro  # [kJ/mol] 96.4853910
Wien2kToTHz = (
    sqrt(Rydberg / 1000 * EV / AMU) / (Bohr * 1e-10) / (2 * pi) / 1e12
)  # [THz] 3.44595837
AbinitToTHz = sqrt(EV / (AMU * Bohr)) / Angstrom / (2 * pi) / 1e12  # [THz] 21.49068
PwscfToTHz = (
    sqrt(Rydberg * EV / AMU) / (Bohr * 1e-10) / (2 * pi) / 1e12
)  # [THz] 108.97077
ElkToTHz = (
    sqrt(Hartree * EV / AMU) / (Bohr * 1e-10) / (2 * pi) / 1e12
)  # [THz] 154.10794
SiestaToTHz = sqrt(EV / (AMU * Bohr)) / Angstrom / (2 * pi) / 1e12  # [THz] 21.49068
CP2KToTHz = (
    sqrt(Hartree * EV / (AMU * Bohr)) / Angstrom / (2 * pi) / 1e12
)  # CP2K uses a.u. for forces but Angstrom for distances
CrystalToTHz = VaspToTHz
CastepToTHz = VaspToTHz
DftbpToTHz = (
    sqrt(Hartree * EV / AMU) / (Bohr * 1e-10) / (2 * pi) / 1e12
)  # [THz] 154.10794344
dftbpToBohr = 0.188972598857892e01
TurbomoleToTHz = ElkToTHz  # Turbomole uses atomic units (Hartree/Bohr)
EVAngstromToGPa = EV * 1e21
FleurToTHz = ElkToTHz  # Fleur uses atomic units (Hartree/Bohr)
