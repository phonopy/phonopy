"""Tests for the QLM calculator interface."""

import os
import tempfile

import numpy as np

from phonopy.interface.qlm import (
    parse_set_of_forces,
    read_qlm,
)


def test_parse_set_of_forces():
    """Test parse_set_of_forces."""
    force_ref = """% rows 2 cols 3 real
    0.00000000   -0.00406659   -0.00406659
   -0.00000000    0.00406659    0.00406659"""

    # when min py v3.12+ move from try..finally to "with" and delete_on_close=False
    # for windows

    try:
        tfl = tempfile.NamedTemporaryFile(delete=False)
        tfl.write(force_ref.encode())
        tfl.close()

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

    finally:
        os.unlink(tfl.name)


def test_cell2struct_and_read_qlm():
    """Test read_qlm and get_qlm_structure."""
    sitex_ref = (
        """% site-data vn=3.0 xpos fast io=15 nbas=8"""
        + """ alat=10.7531113565 plat= 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
#                            pos
 Na        0.0000000000   0.0000000000   0.0000000000
 Na        0.0000000000   0.5000000000   0.5000000000
 Na2       0.5000000000   0.0000000000   0.5000000000
 Na3       0.5000000000   0.5000000000   0.0000000000
 Cl        0.5000000000   0.5000000000   0.5000000000
 Clu       0.5000000000   0.0000000000   0.0000000000
 Cl        0.0000000000   0.5000000000   0.0000000000
 Cld       0.0000000000   0.0000000000   0.5000000000
"""
    )

    try:
        fl1 = tempfile.NamedTemporaryFile(delete=False)
        fl1.write(sitex_ref.encode())
        fl1.close()

        cell1, (inst1,) = read_qlm(fl1.name)

        fl2 = tempfile.NamedTemporaryFile(delete=False)
        fl2.write(inst1.to_site_str(cell1).encode())
        fl2.close()

        cell2, _ = read_qlm(fl2.name)

        np.testing.assert_allclose(cell1.cell, cell2.cell, atol=1e-7)
        np.testing.assert_allclose(
            cell1.scaled_positions, cell2.scaled_positions, atol=1e-7
        )
        assert cell1.symbols == cell2.symbols

    finally:
        os.unlink(fl1.name)
        os.unlink(fl2.name)
