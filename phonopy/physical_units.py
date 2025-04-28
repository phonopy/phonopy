"""Collection of physical units."""

# Copyright (C) 2025 Atsushi Togo
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

from dataclasses import dataclass
from math import pi, sqrt


@dataclass
class PhysicalUnits:
    """Physical units for phonon calculations.

    Attributes
    ----------
    kb_J: float
        Boltzmann constant in J/K.
    PlanckConstant: float
        Planck constant in eV s.
    Avogadro: float
        Avogadro's number.
    SpeedOfLight: float
        Speed of light in m/s.
    AMU: float
        Atomic mass unit in kg.
    EV: float
        Electron volt in J.
    Me: float
        Electron mass in kg.

    THz: float
        Terahertz frequency in /s.
    Angstrom: float
        Angstrom in m.
    Newton: float
        Newton in kg m / s^2.
    Joule: float
        Joule in kg m^2 / s^2.
    Hbar: float
        Reduced Planck constant in eV s.
    Epsilon0: float
        Vacuum permittivity in C^2 / N m^2.
    Bohr: float
        Bohr radius in A.
    Hartree: float
        Hartree energy in eV.
    Rydberg: float
        Rydberg energy in eV.
    Kb: float
        Boltzmann constant in eV/K.
    THzToEv: float
        Conversion factor from THz to eV.
    CmToEv: float
        Conversion factor from cm^-1 to eV.
    EVAngstromToGPa: float
        Conversion factor from eV/Angstrom to GPa.
    EvTokJmol: float
        Conversion factor from eV to kJ/mol.

    defaultToTHz: float
        Default conversion factor to THz.

    """

    kb_J: float  # [J/K]
    PlanckConstant: float  # [eV s]
    Avogadro: float
    SpeedOfLight: float  # [m/s]
    AMU: float  # [kg]
    EV: float  # [J]
    Me: float  # [kg]

    def __post_init__(self):
        """Initialize derived physical constants."""
        Mu0 = 4.0e-7 * pi  # [Hartree/m]
        self.THz = 1.0e12  # [/s]
        self.Angstrom = 1.0e-10  # [m]
        self.Newton = 1.0  # [kg m / s^2]
        self.Joule = 1.0  # [kg m^2 / s^2]
        self.Hbar = self.PlanckConstant / (2 * pi)  # [eV s]
        self.Epsilon0 = 1.0 / Mu0 / self.SpeedOfLight**2  # [C^2 / N m^2]
        self.Bohr = (
            4e10 * pi * self.Epsilon0 * self.Hbar**2 / self.Me
        )  # Bohr radius [A] 0.5291772
        self.Hartree = (
            self.Me * self.EV / 16 / pi**2 / self.Epsilon0**2 / self.Hbar**2
        )  # Hartree [eV] 27.211398
        self.Rydberg = self.Hartree / 2  # Rydberg [eV] 13.6056991

        self.THzToEv = self.PlanckConstant * 1e12  # [eV]
        self.Kb = self.kb_J / self.EV  # [eV/K] 8.6173383e-05
        self.THzToCm = 1.0e12 / (self.SpeedOfLight * 100)  # [cm^-1] 33.356410
        self.CmToEv = self.THzToEv / self.THzToCm  # [eV] 1.2398419e-4
        self.EVAngstromToGPa = self.EV * 1e21
        self.EvTokJmol = self.EV / 1000 * self.Avogadro  # [kJ/mol] 96.4853910

        self.defaultToTHz = (
            sqrt(self.EV / self.AMU) / self.Angstrom / (2 * pi) / 1e12
        )  # [THz] 15.633302


physical_units = PhysicalUnits(
    kb_J=1.3806504e-23,  # [J/K]
    PlanckConstant=4.13566733e-15,  # [eV s]
    Avogadro=6.02214179e23,
    SpeedOfLight=299792458,  # [m/s]
    AMU=1.6605402e-27,  # [kg]
    EV=1.60217733e-19,  # [J]
    Me=9.10938215e-31,  # [kg],
)
