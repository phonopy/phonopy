# SPDX-License-Identifier: BSD-3-Clause
"""Output of SSCHA calculations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

import numpy as np

from phonopy import __version__
from phonopy.sscha.core import MLPSSCHA

_HEADER_COMMENT = """\
# SSCHA free energies, one entry per iteration.
#
# The free energy of an iteration is that of the force constants the iteration
# sampled ("sampled_force_constants"), not of the ones it produced from that
# sample ("produced_force_constants"): the harmonic part and the ensemble
# averaged for the anharmonic part then belong to the same force constants.
# The initialization step (iteration 0) is absent, its displacements being
# drawn at a fixed distance rather than from a canonical ensemble.
#
# Energies are per primitive cell. "free_energy" is the sum of "harmonic" and
# "anharmonic". "free_energy_error" is the standard error of the mean of the
# anharmonic part over the sampled supercells, the harmonic part carrying no
# sampling noise. It is conditional on the force constants and does not cover
# their own stochastic uncertainty.
"""

_UNIT = "meV"
_EV_TO_UNIT = 1000.0


def write_sscha_yaml(
    sscha: MLPSSCHA,
    force_constants_filenames: Mapping[int, str] | None = None,
    filename: str | os.PathLike = "sscha_free_energies.yaml",
) -> str:
    """Write SSCHA settings and per-iteration free energies into a yaml file.

    Parameters
    ----------
    sscha : MLPSSCHA
        The SSCHA instance whose settings and history are written.
    force_constants_filenames : Mapping[int, str], optional
        Files the force constants of the iterations were written into, keyed
        by iteration number. Iteration i produces the force constants sampled
        by iteration i + 1, and the two file names are reported accordingly.
        The default is None, which writes no file names.
    filename : str or os.PathLike, optional
        Output file name, by default "sscha_free_energies.yaml".

    Returns
    -------
    str
        The file name written.

    """
    with open(filename, "w") as w:
        w.write("\n".join(get_sscha_yaml_lines(sscha, force_constants_filenames)))
        w.write("\n")
    return str(filename)


def get_sscha_yaml_lines(
    sscha: MLPSSCHA,
    force_constants_filenames: Mapping[int, str] | None = None,
) -> list[str]:
    """Return lines of the yaml file written by ``write_sscha_yaml``."""
    filenames: Mapping[int, str] = force_constants_filenames or {}
    lines = _HEADER_COMMENT.splitlines()
    lines.append("")
    lines += _versions_lines()
    lines.append("")
    lines += _settings_lines(sscha)
    lines.append("")
    lines.append(f"free_energy_unit: {_UNIT}")
    lines += _iterations_lines(sscha, filenames)
    return lines


def _versions_lines() -> list[str]:
    lines = [f'phonopy_version: "{__version__}"']
    for package in ("pypolymlp", "symfc"):
        try:
            version = _package_version(package)
        except PackageNotFoundError:
            continue
        lines.append(f'{package}_version: "{version}"')
    return lines


def _settings_lines(sscha: MLPSSCHA) -> list[str]:
    lines = [
        "sscha:",
        f"  temperature: {sscha.temperature:.5f}",
        f"  number_of_snapshots: {_yaml_value(sscha.number_of_snapshots)}",
        f"  max_iterations: {sscha.max_iterations}",
        f"  mesh: {_yaml_value(sscha.mesh)}",
        f"  random_seed: {_yaml_value(sscha.random_seed)}",
        f"  fc_calculator: {sscha.fc_calculator}",
    ]
    if sscha.initial_force_constants_provided:
        lines.append("  initial_force_constants: provided")
    else:
        lines.append("  initial_force_constants: generated")
        lines.append(f"  distance: {sscha.distance}")
    return lines


def _iterations_lines(sscha: MLPSSCHA, filenames: Mapping[int, str]) -> list[str]:
    if not sscha.history:
        return ["iterations: []"]

    lines = ["iterations:"]
    for result in sscha.history:
        sampled = filenames.get(result.iteration - 1)
        produced = filenames.get(result.iteration)
        lines.append(f"- iteration: {result.iteration}")
        lines.append(f"  sampled_force_constants: {_yaml_value(sampled)}")
        lines.append(f"  produced_force_constants: {_yaml_value(produced)}")
        lines.append(f"  free_energy: {result.free_energy * _EV_TO_UNIT:.6f}")
        lines.append(
            f"  free_energy_error: {result.free_energy_error * _EV_TO_UNIT:.6f}"
        )
        lines.append(f"  harmonic: {result.harmonic * _EV_TO_UNIT:.6f}")
        lines.append(f"  anharmonic: {result.anharmonic * _EV_TO_UNIT:.6f}")
    return lines


def _yaml_value(value) -> str:
    """Return a yaml scalar or flow sequence for a settings value.

    Sequences are nested rather than flattened, so that a mesh given as a 3x3
    matrix is written as one.

    """
    if value is None:
        return "null"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    return str(value)
