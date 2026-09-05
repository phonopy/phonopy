# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.physical_units."""

from __future__ import annotations

import pytest

from phonopy.physical_units import get_calculator_physical_units

# e^2 / (4 pi eps0) = 14.3996454 eV.Angstrom (CODATA 2022 elementary charge and
# vacuum electric permittivity), which is 1 hartree*bohr, 2 Ry*bohr and
# 2000 mRy*bohr.
E2_OVER_4PIEPS0_EV_ANGSTROM = 14.3996454

# nac_factor is used in dynamical_matrix.py as
#   nac_factor * 4*pi / volume * (q.Z)(q.Z) / (q.eps.q)
# and added to the force constants, and the cell keeps the calculator's own
# length unit, so nac_factor is e^2/(4 pi eps0) written in
# force_constants_unit * length_unit^3.  These are the values documented in
# doc/interfaces.md.
EXPECTED_NAC_FACTOR = {
    "vasp": E2_OVER_4PIEPS0_EV_ANGSTROM,  # eV/angstrom^2, angstrom
    "qe": 2.0,  # Ry/au^2, au
    "wien2k": 2000.0,  # mRy/au^2, au
    "elk": 1.0,  # hartree/au^2, au
    "turbomole": 1.0,  # hartree/au^2, au
    "fleur": 1.0,  # hartree/au^2, au
    "octopus": 1.0,  # hartree/au^2, au
}


@pytest.mark.parametrize("interface_mode", EXPECTED_NAC_FACTOR)
def test_nac_factor(interface_mode: str):
    """Test nac_factor is e^2/(4 pi eps0) in the calculator's own units."""
    units = get_calculator_physical_units(interface_mode)
    assert units.nac_factor == pytest.approx(
        EXPECTED_NAC_FACTOR[interface_mode], rel=1e-5
    )


def test_nac_factor_dftbp_unchanged():
    """Test DFTB+ keeps the value it has had since 2021.

    This pins current behaviour rather than deriving it; doc/interfaces.md
    documents 14.399652 for DFTB+.
    """
    units = get_calculator_physical_units("dftbp")
    assert units.nac_factor == pytest.approx(E2_OVER_4PIEPS0_EV_ANGSTROM, rel=1e-5)


def test_nac_factor_agrees_across_identical_unit_systems():
    """Test branches declaring the same units agree on nac_factor."""
    reference = get_calculator_physical_units("octopus")
    for interface_mode in ("elk", "turbomole", "fleur"):
        units = get_calculator_physical_units(interface_mode)
        assert units.force_constants_unit == reference.force_constants_unit
        assert units.length_unit == reference.length_unit
        assert units.nac_factor == reference.nac_factor
