# SPDX-License-Identifier: BSD-3-Clause
"""Pytest fixtures shared by QHA tests."""

from __future__ import annotations

import numpy as np
import pytest
from qha_utils import (
    MESH,
    TEMPERATURES,
    internal_energies,
    scaled_phonopy,
    thermal_properties,
)

from phonopy import Phonopy, PhonopyQHA, run_qha
from phonopy.qha.qha import QHAResult


@pytest.fixture(scope="session")
def nacl_qha_phonopys(ph_nacl: Phonopy) -> list[Phonopy]:
    """NaCl series scaled anisotropically along a tetragonal path.

    The conventional cubic cells scaled this way are detected as
    tetragonal (I4/mmm), so lattice-parameter fitting applies.

    """
    phonopys = []
    for s in np.linspace(0.98, 1.04, 7):
        s_c = s * (1.0 + 0.5 * (s - 1.0))
        phonopys.append(scaled_phonopy(ph_nacl, np.array([s, s, s_c])))
    return phonopys


@pytest.fixture(scope="session")
def qha_result_nacl(nacl_qha_phonopys: list[Phonopy]) -> QHAResult:
    """QHAResult of the scaled NaCl series without pressure."""
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    return run_qha(
        nacl_qha_phonopys,
        TEMPERATURES,
        internal_energies=internal_energies(volumes),
        mesh=MESH,
    )


@pytest.fixture(scope="session")
def qha_ref_nacl(nacl_qha_phonopys: list[Phonopy]) -> PhonopyQHA:
    """Legacy PhonopyQHA built from the same data as qha_result_nacl."""
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    fe_phonon, entropy, cv = thermal_properties(nacl_qha_phonopys)
    return PhonopyQHA(
        volumes=volumes,
        electronic_energies=internal_energies(volumes),
        eos="vinet",
        temperatures=TEMPERATURES,
        free_energy=fe_phonon,
        cv=cv,
        entropy=entropy,
    )
