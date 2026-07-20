# SPDX-License-Identifier: BSD-3-Clause
"""Collection of physical units."""

import warnings
from math import pi, sqrt

from phonopy.physical_units import get_physical_units

warnings.warn(
    "phonopy.units.* is deprecated. "
    "Use phonopy.physical_units.get_physical_units() instead.",
    DeprecationWarning,
    stacklevel=2,
)

kb_J = get_physical_units().KB_J
PlanckConstant = get_physical_units().PlanckConstant
Avogadro = get_physical_units().Avogadro
SpeedOfLight = get_physical_units().SpeedOfLight
AMU = get_physical_units().AMU
EV = get_physical_units().EV
Me = get_physical_units().Me
THz = get_physical_units().THz
Angstrom = get_physical_units().Angstrom
Newton = get_physical_units().Newton
Joule = get_physical_units().Joule
Hbar = get_physical_units().Hbar
Epsilon0 = get_physical_units().Epsilon0
Bohr = get_physical_units().Bohr
Hartree = get_physical_units().Hartree
Rydberg = get_physical_units().Rydberg
Kb = get_physical_units().KB
THzToCm = get_physical_units().THzToCm
THzToEv = get_physical_units().THzToEv
CmToEv = get_physical_units().CmToEv
EVAngstromToGPa = get_physical_units().EVAngstromToGPa
EvTokJmol = get_physical_units().EvTokJmol

VaspToTHz = get_physical_units().DefaultToTHz  # [THz] 15.633302
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
TurbomoleToTHz = ElkToTHz  # Turbomole uses atomic units (Hartree/Bohr)
FleurToTHz = ElkToTHz  # Fleur uses atomic units (Hartree/Bohr)
QlmToTHz = (
    sqrt(Rydberg * EV / AMU) / (Bohr * 1e-10) / (2 * pi) / 1e12
)  # [THz] 108.97077
