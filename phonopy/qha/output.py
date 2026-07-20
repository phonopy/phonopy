# SPDX-License-Identifier: BSD-3-Clause
"""File writers for QHA results.

All functions take a QHAResult as the first argument. File formats of the
quantities shared with the legacy QHA implementation are kept identical.

"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from phonopy.physical_units import get_physical_units
from phonopy.qha.eos import get_eos

if TYPE_CHECKING:
    from phonopy.qha.qha import QHAResult


def write_helmholtz_volume(
    result: QHAResult, filename: str | os.PathLike = "helmholtz-volume.dat"
) -> None:
    """Write Helmholtz free energy vs volume in file."""
    with open(filename, "w") as w:
        for t, ep, fe in zip(
            result.temperatures,
            result.eos_parameters,
            result.helmholtz_volume,
            strict=True,
        ):
            w.write("# Temperature: %f\n" % t)
            w.write("# Parameters: %f %f %f %f\n" % tuple(ep))
            for v, e in zip(result.volumes, fe, strict=True):
                w.write("%20.15f %25.15f\n" % (v, e))
            w.write("\n\n")


def write_helmholtz_volume_fitted(
    result: QHAResult,
    thin_number: int = 10,
    filename: str | os.PathLike = "helmholtz-volume_fitted.dat",
    energy_plot_factor: float | None = None,
) -> None:
    """Write Helmholtz free energy (fitted) vs volume in file."""
    if energy_plot_factor is None:
        _energy_plot_factor = 1.0
    else:
        _energy_plot_factor = energy_plot_factor

    eos = get_eos(result.eos_name)
    temperatures = result.temperatures
    equiv_volumes = result.equilibrium_volumes
    equiv_energies = result.gibbs_free_energies

    volume_points = np.linspace(min(result.volumes), max(result.volumes), 201)
    selected_volumes = []
    selected_energies = []
    for i, _ in enumerate(temperatures):
        if i % thin_number == 0:
            selected_volumes.append(equiv_volumes[i])
            selected_energies.append(equiv_energies[i])

    e0 = 0.0
    for i, t in enumerate(temperatures):
        if t >= 298:
            if i > 0:
                de = equiv_energies[i] - equiv_energies[i - 1]
                dt = t - temperatures[i - 1]
                e0 = (298 - temperatures[i - 1]) / dt * de + equiv_energies[i - 1]
            else:
                e0 = 0.0
            break
    e0 *= _energy_plot_factor

    _data_vol_points = []
    _data_eos = []
    for i, _ in enumerate(temperatures):
        if i % thin_number == 0:
            _data_vol_points.append(
                np.array(result.helmholtz_volume[i]) * _energy_plot_factor - e0
            )
            _data_eos.append(
                eos(volume_points, result.eos_parameters[i]) * _energy_plot_factor - e0
            )

    data_eos = np.array(_data_eos).T
    data_vol_points = np.array(_data_vol_points).T
    data_min = np.array(selected_energies) * _energy_plot_factor - e0

    with open(filename, "w") as w:
        w.write("# Volume points\n")
        for j, k in zip(result.volumes, data_vol_points, strict=True):
            w.write("%10.5f " % j)
            for ll in k:
                w.write("%10.5f" % ll)
            w.write("\n")
        w.write("\n# Fitted data\n")

        for m, n in zip(volume_points, data_eos, strict=True):
            w.write("%10.5f " % m)
            for ll in n:
                w.write("%10.5f" % ll)
            w.write("\n")
        w.write("\n# Minimas\n")
        for a, b in zip(selected_volumes, data_min, strict=True):
            w.write("%10.5f %10.5f %s" % (a, b, "\n"))
        w.write("\n")


def write_volume_temperature(
    result: QHAResult, filename: str | os.PathLike = "volume-temperature.dat"
) -> None:
    """Write volume vs temperature in file."""
    with open(filename, "w") as w:
        for t, v in zip(result.temperatures, result.equilibrium_volumes, strict=True):
            w.write("%25.15f %25.15f\n" % (t, v))


def write_thermal_expansion(
    result: QHAResult, filename: str | os.PathLike = "thermal_expansion.dat"
) -> None:
    """Write thermal expansion vs temperature in file."""
    with open(filename, "w") as w:
        for t, beta in zip(result.temperatures, result.thermal_expansion, strict=True):
            w.write("%25.15f %25.15f\n" % (t, beta))


def write_gibbs_temperature(
    result: QHAResult, filename: str | os.PathLike = "gibbs-temperature.dat"
) -> None:
    """Write Gibbs free energy vs temperature in file."""
    with open(filename, "w") as w:
        for t, g in zip(result.temperatures, result.gibbs_free_energies, strict=True):
            w.write("%20.15f %25.15f\n" % (t, g))


def write_bulk_modulus_temperature(
    result: QHAResult, filename: str | os.PathLike = "bulk_modulus-temperature.dat"
) -> None:
    """Write bulk modulus vs temperature in file."""
    with open(filename, "w") as w:
        for t, b in zip(result.temperatures, result.bulk_moduli, strict=True):
            w.write("%20.15f %25.15f\n" % (t, b))


def write_heat_capacity_P(
    result: QHAResult,
    filename: str | os.PathLike = "Cp-temperature.dat",
    filename_ev: str | os.PathLike = "entropy-volume.dat",
    filename_cvv: str | os.PathLike = "Cv-volume.dat",
    filename_dsdvt: str | os.PathLike = "dsdv-temperature.dat",
) -> None:
    """Write C_P and its polynomial-fit details in files."""
    cp_data = result.heat_capacity_P
    temperatures = result.temperatures

    with open(filename_ev, "w") as wve, open(filename_cvv, "w") as wvcv:
        for i in range(1, len(temperatures)):
            t = temperatures[i]
            wve.write("# temperature %20.15f\n" % t)
            wve.write(
                "# %20.15f %20.15f %20.15f %20.15f %20.15f\n"
                % tuple(cp_data.volume_entropy_parameters[i - 1])
            )
            wvcv.write("# temperature %20.15f\n" % t)
            wvcv.write(
                "# %20.15f %20.15f %20.15f %20.15f %20.15f\n"
                % tuple(cp_data.volume_cv_parameters[i - 1])
            )
            for ve, vcv in zip(
                cp_data.volume_entropy[i - 1], cp_data.volume_cv[i - 1], strict=True
            ):
                wve.write("%20.15f %20.15f\n" % tuple(ve))
                wvcv.write("%20.15f %20.15f\n" % tuple(vcv))
            wve.write("\n\n")
            wvcv.write("\n\n")

    with open(filename, "w") as w:
        for t, cp in zip(temperatures, cp_data.heat_capacities, strict=True):
            w.write("%20.15f %20.15f\n" % (t, cp))

    with open(filename_dsdvt, "w") as w:  # GPa
        for t, dsdv in zip(temperatures, cp_data.dsdv, strict=True):
            w.write(
                "%20.15f %20.15f\n" % (t, dsdv * 1e21 / get_physical_units().Avogadro)
            )


def write_gruneisen_temperature(
    result: QHAResult, filename: str | os.PathLike = "gruneisen-temperature.dat"
) -> None:
    """Write Gruneisen parameter vs temperature in file."""
    with open(filename, "w") as w:
        for t, gamma in zip(
            result.temperatures, result.gruneisen_parameters, strict=True
        ):
            w.write("%20.15f %25.15f\n" % (t, gamma))


def write_lattice_parameters_temperature(
    result: QHAResult,
    filename: str | os.PathLike = "lattice_parameters-temperature.dat",
) -> None:
    """Write lattice parameters vs temperature in file."""
    if result.lattice is None:
        raise RuntimeError("Lattice parameters are not available.")
    with open(filename, "w") as w:
        w.write("# temperature (K), a, b, c (angstrom)\n")
        for t, abc in zip(
            result.temperatures, result.lattice.lattice_parameters, strict=True
        ):
            w.write("%20.15f %25.15f %25.15f %25.15f\n" % (t, *abc))


def write_axial_thermal_expansion(
    result: QHAResult,
    filename: str | os.PathLike = "axial_thermal_expansion.dat",
) -> None:
    """Write axial thermal expansion coefficients vs temperature in file."""
    if result.lattice is None:
        raise RuntimeError("Axial thermal expansions are not available.")
    with open(filename, "w") as w:
        w.write(
            "# temperature (K), alpha_a, alpha_b, alpha_c, "
            "alpha_a+alpha_b+alpha_c (1/K)\n"
        )
        for t, alpha in zip(
            result.temperatures,
            result.lattice.axial_thermal_expansions,
            strict=True,
        ):
            w.write(
                "%20.15f %25.15f %25.15f %25.15f %25.15f\n" % (t, *alpha, alpha.sum())
            )
