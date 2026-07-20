# SPDX-License-Identifier: BSD-3-Clause
"""File writers for anisotropic QHA results.

All functions take an AnisotropicQHAResult as the first argument and write
one temperature-indexed quantity to a text file.

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phonopy.qha.anisotropic import AnisotropicQHAResult


def write_lattice_parameters_temperature(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "lattice_parameters-temperature.dat",
) -> None:
    """Write equilibrium lattice parameters vs temperature in file."""
    with open(filename, "w") as w:
        w.write("# temperature (K), a, b, c (angstrom)\n")
        for t, abc in zip(
            result.temperatures,
            result.equilibrium_lattice_parameters,
            strict=True,
        ):
            w.write("%20.15f %25.15f %25.15f %25.15f\n" % (t, *abc))


def write_axial_thermal_expansion(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "axial_thermal_expansion.dat",
) -> None:
    """Write axial thermal expansion coefficients vs temperature in file."""
    with open(filename, "w") as w:
        w.write(
            "# temperature (K), alpha_a, alpha_b, alpha_c, "
            "alpha_a+alpha_b+alpha_c (1/K)\n"
        )
        for t, alpha in zip(
            result.temperatures,
            result.axial_thermal_expansions,
            strict=True,
        ):
            w.write(
                "%20.15f %25.15f %25.15f %25.15f %25.15f\n" % (t, *alpha, alpha.sum())
            )


def write_volume_temperature(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "volume-temperature.dat",
) -> None:
    """Write equilibrium volume vs temperature in file."""
    with open(filename, "w") as w:
        for t, v in zip(result.temperatures, result.equilibrium_volumes, strict=True):
            w.write("%25.15f %25.15f\n" % (t, v))


def write_free_energy_temperature(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "free_energy-temperature.dat",
) -> None:
    """Write the minimized free energy vs temperature in file.

    The minimized free energy is the Helmholtz free energy, or the Gibbs
    free energy when a pressure was given to run_anisotropic_qha.

    """
    with open(filename, "w") as w:
        for t, g in zip(result.temperatures, result.gibbs_free_energies, strict=True):
            w.write("%20.15f %25.15f\n" % (t, g))
