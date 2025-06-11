"""Tests for the QLM calculator interface."""

import sys
import tempfile

import numpy as np
import pytest

from phonopy.interface.qlm import (
    get_qlm_structure,
    parse_set_of_forces,
    read_qlm,
)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="does not run on windows")
def test_parse_set_of_forces():
    """Test parse_set_of_forces."""
    force_ref = """% rows 2 cols 3 real
    0.00000000   -0.00406659   -0.00406659
   -0.00000000    0.00406659    0.00406659"""

    with tempfile.NamedTemporaryFile() as tfl:
        tfl.write(force_ref.encode())
        tfl.flush()

        frs = parse_set_of_forces(2, (tfl.name,), verbose=False)

        np.testing.assert_allclose(
            frs,
            [
                np.array(
                    [
                        [0.00000000, -0.00406659, -0.00406659],
                        [-0.00000000, 0.00406659, 0.00406659],
                    ]
                )
            ],
            atol=1e-7,
        )


@pytest.mark.skipif(sys.platform.startswith("win"), reason="does not run on windows")
def test_cell2struct_and_read_qlm():
    """Test read_qlm and get_qlm_structure."""
    sitex_ref = (
        """% site-data vn=3.0 xpos fast io=15 nbas=8 alat=10.7531113565"""
        + """ plat= 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
#                            pos
 Na        0.0000000000   0.0000000000   0.0000000000
 Na        0.0000000000   0.5000000000   0.5000000000
 Na        0.5000000000   0.0000000000   0.5000000000
 Na        0.5000000000   0.5000000000   0.0000000000
 Cl        0.5000000000   0.5000000000   0.5000000000
 Cl        0.5000000000   0.0000000000   0.0000000000
 Cl        0.0000000000   0.5000000000   0.0000000000
 Cl        0.0000000000   0.0000000000   0.5000000000
"""
    )

    with tempfile.NamedTemporaryFile() as tfl:
        tfl.write(sitex_ref.encode())
        tfl.flush()

        cell1 = read_qlm(tfl.name)

        cell1_s = get_qlm_structure(cell1)

        tfl.seek(0)
        tfl.write(cell1_s.encode())
        tfl.flush()

        cell2 = read_qlm(tfl.name)

        np.testing.assert_allclose(cell1.cell, cell2.cell, atol=1e-7)
        np.testing.assert_allclose(
            cell1.scaled_positions, cell2.scaled_positions, atol=1e-7
        )
        assert cell1.symbols == cell2.symbols
