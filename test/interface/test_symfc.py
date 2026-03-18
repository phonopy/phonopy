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
from phonopy.interface.symfc import (
    SymfcFCSolver,
    estimate_symfc_memory_usage,
    parse_symfc_options,
    symmetrize_by_projector,
    update_symfc_cutoff_by_memsize,
)


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


# ---------------------------------------------------------------------------
# parse_symfc_options — no symfc import needed
# ---------------------------------------------------------------------------


def test_parse_symfc_options_none():
    """None input returns empty dict."""
    assert parse_symfc_options(None, 2) == {}


def test_parse_symfc_options_dict_passthrough():
    """Dict input is returned as-is."""
    opts = {"cutoff": {2: 5.0}}
    assert parse_symfc_options(opts, 2) == {"cutoff": {2: 5.0}}


def test_parse_symfc_options_str_cutoff():
    """String 'cutoff = 5.0' parsed into {cutoff: {order: 5.0}}."""
    result = parse_symfc_options("cutoff = 5.0", 2)
    assert result == {"cutoff": {2: 5.0}}


def test_parse_symfc_options_str_memsize():
    """String 'memsize = 2.0' parsed into {memsize: {order: 2.0}}."""
    result = parse_symfc_options("memsize = 2.0", 3)
    assert result == {"memsize": {3: 2.0}}


def test_parse_symfc_options_str_use_mkl():
    """'use_mkl = true' parsed as bool True."""
    result = parse_symfc_options("use_mkl = true", 2)
    assert result == {"use_mkl": True}


def test_parse_symfc_options_str_multiple():
    """Multiple options separated by comma are all parsed."""
    result = parse_symfc_options("cutoff = 5.0, use_mkl = true", 2)
    assert result == {"cutoff": {2: 5.0}, "use_mkl": True}


def test_parse_symfc_options_type_error():
    """Non-str/dict/None raises TypeError."""
    with pytest.raises(TypeError, match="options must be str or dict"):
        parse_symfc_options(123, 2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SymfcFCSolver integration tests (require symfc)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nacl_symfc_data(ph_nacl_rd: Phonopy):
    """Return (supercell, displacements, forces, symmetry) for NaCl."""
    pytest.importorskip("symfc")
    ph = ph_nacl_rd
    return ph.supercell, ph.displacements, ph.forces, ph.symmetry


def test_symfc_fc_shape_full(nacl_symfc_data):
    """Full FC shape is (N, N, 3, 3)."""
    supercell, disps, forces, symmetry = nacl_symfc_data
    solver = SymfcFCSolver(
        supercell,
        displacements=disps,
        forces=forces,
        symmetry=symmetry,
        orders=[2],
        is_compact_fc=False,
    )
    fc = solver.force_constants[2]
    n = len(supercell)
    assert fc.shape == (n, n, 3, 3)


def test_symfc_fc_shape_compact(nacl_symfc_data, ph_nacl_rd: Phonopy):
    """Compact FC shape is (n_prim, N, 3, 3)."""
    supercell, disps, forces, symmetry = nacl_symfc_data
    solver = SymfcFCSolver(
        supercell,
        displacements=disps,
        forces=forces,
        symmetry=symmetry,
        orders=[2],
        is_compact_fc=True,
    )
    fc = solver.force_constants[2]
    n = len(supercell)
    n_prim = len(ph_nacl_rd.primitive)
    assert fc.shape == (n_prim, n, 3, 3)


def test_symfc_force_constants_before_run_raises(nacl_symfc_data):
    """Accessing force_constants before run() raises RuntimeError."""
    supercell, disps, forces, symmetry = nacl_symfc_data
    solver = SymfcFCSolver(supercell, symmetry=symmetry)
    with pytest.raises(RuntimeError, match="Run SymfcFCSolver.run"):
        _ = solver.force_constants


def test_symfc_estimate_basis_size(nacl_symfc_data):
    """estimate_basis_size returns dict with int key and positive int value."""
    supercell, _, _, symmetry = nacl_symfc_data
    solver = SymfcFCSolver(supercell, symmetry=symmetry)
    basis_sizes = solver.estimate_basis_size(orders=[2])
    assert 2 in basis_sizes
    assert basis_sizes[2] > 0


def test_symfc_estimate_numbers_of_supercells(nacl_symfc_data):
    """estimate_numbers_of_supercells returns dict with positive int value."""
    supercell, _, _, symmetry = nacl_symfc_data
    solver = SymfcFCSolver(supercell, symmetry=symmetry)
    n_scells = solver.estimate_numbers_of_supercells(orders=[2])
    assert 2 in n_scells
    assert n_scells[2] > 0


def test_estimate_symfc_memory_usage(ph_nacl_rd: Phonopy):
    """estimate_symfc_memory_usage returns two non-negative floats."""
    pytest.importorskip("symfc")
    ph = ph_nacl_rd
    memsize, memsize2 = estimate_symfc_memory_usage(
        ph.supercell, ph.symmetry, order=2, cutoff=5.0
    )
    assert memsize >= 0.0
    assert memsize2 >= 0.0


def test_update_symfc_cutoff_by_memsize_no_memsize(ph_nacl_rd: Phonopy):
    """update_symfc_cutoff_by_memsize is no-op when 'memsize' not in options."""
    pytest.importorskip("symfc")
    options = {"cutoff": {2: 5.0}}
    update_symfc_cutoff_by_memsize(
        options, ph_nacl_rd.supercell, ph_nacl_rd.primitive, ph_nacl_rd.symmetry
    )
    assert options == {"cutoff": {2: 5.0}}


def test_update_symfc_cutoff_by_memsize_large_memsize(ph_nacl_rd: Phonopy):
    """update_symfc_cutoff_by_memsize sets cutoff=None when memsize is huge."""
    pytest.importorskip("symfc")
    ph = ph_nacl_rd
    options: dict = {"memsize": {2: 1000.0}}
    update_symfc_cutoff_by_memsize(options, ph.supercell, ph.primitive, ph.symmetry)
    assert "memsize" not in options
    assert options.get("cutoff") is None
