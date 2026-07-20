# SPDX-License-Identifier: BSD-3-Clause
"""Phonon calculation code: Phonopy."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.api_phonopy import Phonopy
from phonopy.api_qha import PhonopyQHA
from phonopy.cui.load import load
from phonopy.qha.anisotropic import AnisotropicQHAResult, run_anisotropic_qha
from phonopy.qha.qha import QHAResult, run_qha

try:
    __version__ = _package_version("phonopy")
except PackageNotFoundError:  # running from a source tree without an install
    try:
        from phonopy._version import __version__
    except ImportError:
        __version__ = "0.0.0"

__all__ = [
    "AnisotropicQHAResult",
    "PhonopyGruneisen",
    "Phonopy",
    "PhonopyQHA",
    "QHAResult",
    "load",
    "run_anisotropic_qha",
    "run_qha",
    "__version__",
]
