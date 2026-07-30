# SPDX-License-Identifier: BSD-3-Clause
"""File writers for anisotropic QHA results.

All functions take an AnisotropicQHAResult as the first argument and write
one temperature-indexed quantity to a text file.

Every file opens with a provenance line recording the settings that produced
the numbers, followed by the column legend. Without it two runs of the same
grid are indistinguishable after the fact, and the settings that matter are
not cosmetic: the q-mesh and the electronic free energy each move the axial
thermal expansions by tens of percent while leaving the volumetric expansion
nearly unchanged.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, TextIO

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from phonopy.qha.anisotropic import AnisotropicQHAResult


def _format_mesh(mesh: float | Sequence[int] | NDArray[np.int64]) -> str:
    """Return a compact string for the mesh setting, scalar or explicit."""
    if isinstance(mesh, (float, int, np.generic)):
        return f"{float(mesh):g}"
    return " ".join(str(int(v)) for v in mesh)


def _format_temperatures(temperatures: NDArray[np.double]) -> str:
    """Return the temperature range, with the step when the grid is uniform."""
    if len(temperatures) == 0:
        return "none"
    if len(temperatures) == 1:
        return f"{temperatures[0]:g} K"
    steps = np.diff(temperatures)
    span = f"{temperatures[0]:g}-{temperatures[-1]:g} K"
    if np.allclose(steps, steps[0]):
        return f"{span} step {steps[0]:g}"
    return f"{span}, {len(temperatures)} points"


def format_provenance(
    result: AnisotropicQHAResult, provenance: Sequence[str] | None = None
) -> str:
    """Return the one-line record of the settings that produced the result.

    The settings come from the result itself, so a caller cannot omit them.
    Information the result does not carry -- the input dataset, the
    force-constant calculator -- is appended through ``provenance``. The
    returned line carries no comment marker; the caller adds one.

    """
    items = []
    if result.mesh is not None:
        items.append(f"mesh={_format_mesh(result.mesh)}")
    items.append(f"surface_degree={result.surface_degree}")
    items.append(f"F_el={'on' if result.with_electronic else 'off'}")
    if result.pressure is not None:
        items.append(f"pressure={result.pressure:g} GPa")
    items.append(f"grid_points={result.lattice_lengths.shape[0]}")
    items.append(f"temperatures={_format_temperatures(result.temperatures)}")
    if provenance:
        items.extend(provenance)
    return "anisotropic QHA: " + ", ".join(items)


def _write_header(
    w: TextIO,
    result: AnisotropicQHAResult,
    columns: str,
    provenance: Sequence[str] | None = None,
) -> None:
    """Write the provenance line and the column legend as "#" comments."""
    w.write(f"# {format_provenance(result, provenance)}\n")
    w.write(f"# {columns}\n")


def write_lattice_parameters_temperature(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "lattice_parameters-temperature.dat",
    provenance: Sequence[str] | None = None,
) -> None:
    """Write equilibrium lattice parameters vs temperature in file."""
    with open(filename, "w") as w:
        _write_header(w, result, "temperature (K), a, b, c (angstrom)", provenance)
        for t, abc in zip(
            result.temperatures,
            result.equilibrium_lattice_parameters,
            strict=True,
        ):
            w.write("%20.15f %25.15f %25.15f %25.15f\n" % (t, *abc))


def write_axial_thermal_expansion(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "axial_thermal_expansion.dat",
    provenance: Sequence[str] | None = None,
) -> None:
    """Write axial thermal expansion coefficients vs temperature in file."""
    with open(filename, "w") as w:
        _write_header(
            w,
            result,
            "temperature (K), alpha_a, alpha_b, alpha_c, alpha_a+alpha_b+alpha_c (1/K)",
            provenance,
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
    provenance: Sequence[str] | None = None,
) -> None:
    """Write equilibrium volume vs temperature in file."""
    with open(filename, "w") as w:
        _write_header(
            w,
            result,
            "temperature (K), volume (angstrom^3, primitive cell)",
            provenance,
        )
        for t, v in zip(result.temperatures, result.equilibrium_volumes, strict=True):
            w.write("%25.15f %25.15f\n" % (t, v))


def write_free_energy_temperature(
    result: AnisotropicQHAResult,
    filename: str | os.PathLike = "free_energy-temperature.dat",
    provenance: Sequence[str] | None = None,
) -> None:
    """Write the minimized free energy vs temperature in file.

    The minimized free energy is the Helmholtz free energy, or the Gibbs
    free energy when a pressure was given to run_anisotropic_qha.

    """
    label = "Gibbs" if result.pressure is not None else "Helmholtz"
    with open(filename, "w") as w:
        _write_header(
            w,
            result,
            f"temperature (K), {label} free energy (eV, primitive cell)",
            provenance,
        )
        for t, g in zip(result.temperatures, result.gibbs_free_energies, strict=True):
            w.write("%20.15f %25.15f\n" % (t, g))
