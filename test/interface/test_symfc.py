"""Tests for symfc interface."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
    full_fc_to_compact_fc,
)
from phonopy.interface.symfc import SymfcFCSolver, symmetrize_by_projector


@pytest.mark.parametrize("cutoff", [None, {3: 5.0}])
def test_symfc_cutoff(ph_nacl: Phonopy, cutoff: Optional[dict]):
    """Test symfc interface with cutoff distance."""
    ph = ph_nacl
    symfc_solver = SymfcFCSolver(
        ph.supercell,
        symmetry=ph.symmetry,
        options={"cutoff": cutoff},
        log_level=1,
    )
    symfc_solver.compute_basis_set(
        orders=[2, 3],
    )
    basis_set_fc3 = symfc_solver.basis_set[3]
    if cutoff is None:
        assert basis_set_fc3.basis_set.shape == (
            786,
            758,
        )
    else:
        assert basis_set_fc3.basis_set.shape == (
            80,
            67,
        )
        nonzero_elems = symfc_solver.get_nonzero_atomic_indices_fc3()
        assert nonzero_elems is not None
        assert nonzero_elems.size == len(ph.supercell) ** 3
        assert nonzero_elems.sum() == 21952


def test_symmetrize_by_projector(
    ph_zr3n4_nofcsym: Phonopy, ph_zr3n4_nofcsym_compact_fc: Phonopy
):
    """Test symmetrization by projector."""
    for i, ph in enumerate((ph_zr3n4_nofcsym, ph_zr3n4_nofcsym_compact_fc)):
        fc_sym = symmetrize_by_projector(
            ph.supercell,
            ph.force_constants,
            2,
            primitive=ph.primitive,
            log_level=2,
        )
        diff = ph.force_constants - fc_sym
        assert diff.max() == pytest.approx(0.001016, rel=1e-5)

        if i == 1:
            fc_sym = compact_fc_to_full_fc(ph.primitive, fc_sym)

        for i, j, k in list(np.ndindex((len(ph.supercell), 3, 3))):
            assert fc_sym[:, i, j, k].sum() == pytest.approx(0)
            assert fc_sym[i, :, j, k].sum() == pytest.approx(0)


def test_symmetrize_by_projector_with_inconsistent_p2s(
    ph_tio2_nofcsym: Phonopy,
):
    """Test symmetrization by projector with special shape of compact fc.

    This is not an ideal situation, but user might encounter it when a
    non-primitive cell is used as the phonopy's primitive cell.

    """
    ph = ph_tio2_nofcsym
    ph_copy = Phonopy(unitcell=ph.unitcell, supercell_matrix=ph.supercell_matrix)
    fc = full_fc_to_compact_fc(
        ph_copy.primitive,
        ph.force_constants,
    )
    with pytest.warns(
        UserWarning,
        match="p2s_map of primitive cell does not match with p2s_map of symfc.",
    ):
        fc_sym = symmetrize_by_projector(
            ph_copy.supercell,
            fc,
            2,
            primitive=ph_copy.primitive,
            log_level=2,
        )
    assert (fc - fc_sym).max() == pytest.approx(0.0010540, rel=1e-5)
