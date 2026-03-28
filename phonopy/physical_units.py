"""Collection of physical units.

Use get_physical_units() for getting the physical units.
To overwrite the physical units, use set_physical_units().

"""

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

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from math import pi, sqrt
from typing import Any


@dataclass(frozen=True)
class CalculatorPhysicalUnits:
    """Physical units for calculator interfaces.

    Dict-like access is supported for backward compatibility but deprecated.
    """

    factor: float
    nac_factor: float
    distance_to_A: float
    force_to_eVperA: float
    energy_to_eV: float
    force_constants_unit: str
    length_unit: str
    force_unit: str
    energy_unit: str

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return supported attribute names without dict-like access."""
        return tuple(cls.__dataclass_fields__.keys())

    def __getitem__(self, key: str) -> Any:
        """Return attribute by name (deprecated dict-like usage)."""
        warnings.warn(
            "Dict-like access to CalculatorPhysicalUnits is deprecated. "
            "Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        """Return attribute by name with a default (deprecated dict-like usage)."""
        warnings.warn(
            "Dict-like access to CalculatorPhysicalUnits is deprecated. "
            "Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        """Return supported attribute names (deprecated dict-like usage)."""
        warnings.warn(
            "Dict-like access to CalculatorPhysicalUnits is deprecated. "
            "Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.field_names()

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Return (name, value) pairs (deprecated dict-like usage).

        warnings.warn is not written in this method because it is emitted at
        self.keys().

        """
        return tuple((key, getattr(self, key)) for key in self.keys())

    def values(self) -> tuple[Any, ...]:
        """Return values in key order (deprecated dict-like usage).

        warnings.warn is not written in this method because it is emitted at
        self.keys().

        """
        return tuple(getattr(self, key) for key in self.keys())


@dataclass
class PhysicalUnitsGenerator:
    """Physical units for phonon calculations."""

    KB_J: float  # [J/K]
    PlanckConstant: float  # [eV s]
    Avogadro: float
    SpeedOfLight: float  # [m/s]
    AMU: float  # [kg]
    EV: float  # [J]
    Me: float  # [kg]
    THz: float | None = None  # [/s]
    Angstrom: float | None = None  # [m]
    Newton: float | None = None  # [kg m / s^2]
    Joule: float | None = None  # [kg m^2 / s^2]
    Hbar: float | None = None  # [eV s]
    Epsilon0: float | None = None  # [C^2 / N m^2]
    Bohr: float | None = None  # Bohr radius [A]
    Hartree: float | None = None  # Hartree [eV]
    Rydberg: float | None = None  # Rydberg [eV]
    THzToEv: float | None = None  # [eV]
    KB: float | None = None  # [eV/K]
    THzToCm: float | None = None  # [cm^-1]
    CmToEv: float | None = None  # [eV]
    EVAngstromToGPa: float | None = None
    EvTokJmol: float | None = None  # [kJ/mol]
    DefaultToTHz: float | None = None  # [THz]

    def __post_init__(self) -> None:
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
        self.Rydberg = self.Hartree / 2  # Rydberg [eV]

        self.THzToEv = self.PlanckConstant * 1e12  # [eV]
        self.KB = self.KB_J / self.EV  # [eV/K]
        self.THzToCm = 1.0e12 / (self.SpeedOfLight * 100)  # [cm^-1]
        self.CmToEv = self.THzToEv / self.THzToCm  # [eV]
        self.EVAngstromToGPa = self.EV * 1e21
        self.EvTokJmol = self.EV / 1000 * self.Avogadro  # [kJ/mol]

        self.DefaultToTHz = (
            sqrt(self.EV / self.AMU) / self.Angstrom / (2 * pi) / 1e12
        )  # [THz]


@dataclass(frozen=True)
class PhysicalUnits:
    """Immutable physical units for phonon calculations.

    Attributes
    ----------
    KB_J: float
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
    KB: float
        Boltzmann constant in eV/K.
    THzToEv: float
        Conversion factor from THz to eV.
    CmToEv: float
        Conversion factor from cm^-1 to eV.
    EVAngstromToGPa: float
        Conversion factor from eV/Angstrom to GPa.
    EvTokJmol: float
        Conversion factor from eV to kJ/mol.

    DefaultToTHz: float
        Default conversion factor to THz.

    """

    KB_J: float  # [J/K]
    PlanckConstant: float  # [eV s]
    Avogadro: float
    SpeedOfLight: float  # [m/s]
    AMU: float  # [kg]
    EV: float  # [J]
    Me: float  # [kg]
    THz: float  # [/s]
    Angstrom: float  # [m]
    Newton: float  # [kg m / s^2]
    Joule: float  # [kg m^2 / s^2]
    Hbar: float  # [eV s]
    Epsilon0: float  # [C^2 / N m^2]
    Bohr: float  # Bohr radius [A]
    Hartree: float  # Hartree [eV]
    Rydberg: float  # Rydberg [eV]
    THzToEv: float  # [eV]
    KB: float  # [eV/K]
    THzToCm: float  # [cm^-1]
    CmToEv: float  # [eV]
    EVAngstromToGPa: float
    EvTokJmol: float  # [kJ/mol]
    DefaultToTHz: float  # [THz]


def set_physical_units(
    KB_J: float = 1.3806504e-23,  # [J/K]
    PlanckConstant: float = 4.13566733e-15,  # [eV s]
    Avogadro: float = 6.02214179e23,
    SpeedOfLight: float = 299792458,  # [m/s]
    AMU: float = 1.6605402e-27,  # [kg]
    EV: float = 1.60217733e-19,  # [J]
    Me: float = 9.10938215e-31,  # [kg],
) -> None:
    """Set physical units used globally.

    Default values are:

    KB_J : 1.3806504e-23 [J/K]
    PlanckConstant : 4.13566733e-15 [eV s]
    Avogadro : 6.02214179e+23
    SpeedOfLight : 299792458 [m/s]
    AMU : 1.6605402e-27 [kg]
    EV : 1.60217733e-19 [J]
    Me : 9.10938215e-31 [kg]
    THz : 1e12 [/s]
    Angstrom : 1e-10 [m]
    Newton : 1.0 [kg m / s^2]
    Joule : 1.0 [kg m^2 / s^2]
    Hbar : 6.582118985531608e-16 [eV s]
    Epsilon0 : 8.85418781762039e-12 [C^2 / N m^2]
    Bohr : 0.529177207423948 [A]
    Hartree : 27.211398230887998 [eV]
    Rydberg : 13.605699115443999 [eV]
    THzToEv : 0.00413566733 [eV]
    KB : 8.617338256808316e-05 [eV/K]
    THzToCm : 33.3564095198152 [cm^-1]
    CmToEv : 0.00012398418743309975 [eV]
    EVAngstromToGPa : 160.21773299999998 [GPa]
    EvTokJmol : 96.4853905398362 [kJ/mol]
    DefaultToTHz : 15.633302300230191 [THz]

    """
    global _physical_units
    physical_units = PhysicalUnitsGenerator(
        KB_J=KB_J,
        PlanckConstant=PlanckConstant,
        Avogadro=Avogadro,
        SpeedOfLight=SpeedOfLight,
        AMU=AMU,
        EV=EV,
        Me=Me,
    )
    _physical_units = PhysicalUnits(**asdict(physical_units))


def get_physical_units() -> PhysicalUnits:
    """Get physical units used globally."""
    return _physical_units


def get_calculator_physical_units(
    interface_mode: str | None = None,
) -> CalculatorPhysicalUnits:
    """Return physical units of each calculator.

    Physical units: energy,  distance,  atomic mass, force,        force constants
    vasp          : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    wien2k        : Ry,      au(=borh), AMU,         mRy/au,       mRy/au^2
    abinit        : hartree, au,        AMU,         eV/angstrom,  eV/angstrom.au
    elk           : hartree, au,        AMU,         hartree/au,   hartree/au^2
    qe            : Ry,      au,        AMU,         Ry/au,        Ry/au^2
    siesta        : eV,      au,        AMU,         eV/angstrom,  eV/angstrom.au
    CRYSTAL       : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    DFTB+         : hartree, au,        AMU          hartree/au,   hartree/au^2
    TURBOMOLE     : hartree, au,        AMU,         hartree/au,   hartree/au^2
    CP2K          : hartree, angstrom,  AMU,         hartree/au,   hartree/angstrom.au
    FHI-aims      : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    castep        : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    fleur         : hartree, au,        AMU,         hartree/au,   hartree/au^2
    abacus        : eV,      au,        AMU,         eV/angstrom,  eV/angstrom.au
    lammps        : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2
    qlm           : Ry,      au,        AMU,         Ry/au,        Ry/au^2
    pwmat         : eV,      angstrom,  AMU,         eV/angstrom,  eV/angstrom^2

    units['force_constants_unit'] is used in
    the 'get_force_constant_conversion_factor' method.

    """
    physical_units = get_physical_units()

    if interface_mode is None or interface_mode in (
        "vasp",
        "aims",
        "lammps",
        "pwmat",
        "crystal",
        "castep",
    ):
        VaspToTHz = physical_units.DefaultToTHz  # [THz] 15.633302
        units = CalculatorPhysicalUnits(
            factor=VaspToTHz,
            nac_factor=physical_units.Hartree * physical_units.Bohr,
            distance_to_A=1.0,
            force_to_eVperA=1.0,
            energy_to_eV=1.0,
            force_constants_unit="eV/angstrom^2",
            length_unit="angstrom",
            force_unit="eV/angstrom",
            energy_unit="eV",
        )
    elif interface_mode == "abinit":
        AbinitToTHz = (
            sqrt(physical_units.EV / (physical_units.AMU * physical_units.Bohr))
            / physical_units.Angstrom
            / (2 * pi)
            / 1e12
        )  # [THz] 21.49068
        units = CalculatorPhysicalUnits(
            factor=AbinitToTHz,
            nac_factor=physical_units.Hartree / physical_units.Bohr,
            distance_to_A=physical_units.Bohr,
            force_to_eVperA=1.0,
            energy_to_eV=physical_units.Hartree,
            force_constants_unit="eV/angstrom.au",
            length_unit="au",
            force_unit="eV/angstrom",
            energy_unit="hartree",
        )
    elif interface_mode in ("qe", "qlm"):
        PwscfToTHz = (
            sqrt(physical_units.Rydberg * physical_units.EV / physical_units.AMU)
            / (physical_units.Bohr * 1e-10)
            / (2 * pi)
            / 1e12
        )  # [THz] 108.97077
        units = CalculatorPhysicalUnits(
            factor=PwscfToTHz,
            nac_factor=2.0,
            distance_to_A=physical_units.Bohr,
            force_to_eVperA=physical_units.Rydberg / physical_units.Bohr,
            energy_to_eV=physical_units.Rydberg,
            force_constants_unit="Ry/au^2",
            length_unit="au",
            force_unit="Ry/au",
            energy_unit="Ry",
        )
    elif interface_mode == "wien2k":
        Wien2kToTHz = (
            sqrt(physical_units.Rydberg / 1000 * physical_units.EV / physical_units.AMU)
            / (physical_units.Bohr * 1e-10)
            / (2 * pi)
            / 1e12
        )  # [THz] 3.44595837
        units = CalculatorPhysicalUnits(
            factor=Wien2kToTHz,
            nac_factor=2000.0,
            distance_to_A=physical_units.Bohr,
            force_to_eVperA=0.001 * physical_units.Rydberg / physical_units.Bohr,
            energy_to_eV=physical_units.Rydberg,
            force_constants_unit="mRy/au^2",
            length_unit="au",
            force_unit="mRy/au",
            energy_unit="Ry",
        )
    elif interface_mode in ("elk", "dftbp", "turbomole", "fleur"):
        ElkToTHz = (
            sqrt(physical_units.Hartree * physical_units.EV / physical_units.AMU)
            / (physical_units.Bohr * 1e-10)
            / (2 * pi)
            / 1e12
        )  # [THz] 154.10794
        units = CalculatorPhysicalUnits(
            factor=ElkToTHz,
            nac_factor=physical_units.Hartree * physical_units.Bohr,
            distance_to_A=physical_units.Bohr,
            force_to_eVperA=physical_units.Hartree / physical_units.Bohr,
            energy_to_eV=physical_units.Hartree,
            force_constants_unit="hartree/au^2",
            length_unit="au",
            force_unit="hartree/au",
            energy_unit="hartree",
        )
    elif interface_mode in ["siesta", "abacus"]:
        SiestaToTHz = (
            sqrt(physical_units.EV / (physical_units.AMU * physical_units.Bohr))
            / physical_units.Angstrom
            / (2 * pi)
            / 1e12
        )  # [THz] 21.49068
        units = CalculatorPhysicalUnits(
            factor=SiestaToTHz,
            nac_factor=physical_units.Hartree / physical_units.Bohr,
            distance_to_A=physical_units.Bohr,
            force_to_eVperA=1.0,
            energy_to_eV=1.0,
            force_constants_unit="eV/angstrom.au",
            length_unit="au",
            force_unit="eV/angstrom",
            energy_unit="eV",
        )
    elif interface_mode == "cp2k":
        CP2KToTHz = (  # CP2K uses a.u. for forces but Angstrom for distances
            sqrt(
                physical_units.Hartree
                * physical_units.EV
                / (physical_units.AMU * physical_units.Bohr)
            )
            / physical_units.Angstrom
            / (2 * pi)
            / 1e12
        )
        units = CalculatorPhysicalUnits(
            factor=CP2KToTHz,
            nac_factor=physical_units.Bohr**2,
            distance_to_A=1.0,
            force_to_eVperA=physical_units.Hartree / physical_units.Bohr,
            energy_to_eV=physical_units.Hartree,
            force_constants_unit="hartree/angstrom.au",
            length_unit="angstrom",
            force_unit="hartree/au",
            energy_unit="hartree",
        )
    else:
        msg = f"Unknown calculator interface: {interface_mode}"
        raise ValueError(msg)

    return units


# Global variable _physical_units is initialized by set_physical_units() below.
_physical_units: PhysicalUnits
set_physical_units()
