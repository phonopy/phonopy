# SPDX-License-Identifier: BSD-3-Clause
"""Tests for symfc interface."""

from __future__ import annotations

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
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture


@pytest.mark.parametrize("cutoff", [None, {3: 5.0}])
def test_symfc_cutoff(ph_nacl: Phonopy, cutoff: dict | None):
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


def test_symfc_auto_fallback_to_full_fc_on_primitive_mismatch(
    agno2_cell: PhonopyAtoms, capsys: pytest.CaptureFixture[str]
):
    """When the input primitive is non-primitive, symfc auto-falls back to full FC.

    AgNO2 (Imm2 #44) is body-centered with a 4-atom primitive cell.  When
    the user supplies the 8-atom conventional cell as the "primitive"
    (``primitive_matrix=I``), symfc detects a smaller primitive cell than
    phonopy's, and the compact FC arrays cannot be stored in phonopy's
    compact slots without re-mapping.  ``FCSolver`` should detect this and
    silently fall back to full FC (with an informative WARNING).

    """
    ph = Phonopy(agno2_cell, supercell_matrix=[1, 1, 1], primitive_matrix=np.eye(3))
    assert len(ph.primitive.p2s_map) == 8
    ph.generate_displacements(distance=0.01, number_of_snapshots=2, random_seed=1)
    n_atoms = len(ph.supercell)
    ph.forces = np.zeros((ph.displacements.shape[0], n_atoms, 3), dtype="double")
    ph.produce_force_constants(
        fc_calculator="symfc",
        calculate_full_force_constants=False,  # request compact
        fc_calculator_log_level=1,
    )
    # Shape must be the full (n_satom, n_satom, 3, 3) form, not compact
    # (n_patom_phonopy, n_satom, 3, 3).
    assert ph.force_constants.shape == (n_atoms, n_atoms, 3, 3)
    captured = capsys.readouterr()
    assert "Falling back to full force constants" in captured.out
    assert "default primitive matrix" in captured.out


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
    # When phonopy's primitive disagrees with symfc's, ``symmetrize_by_projector``
    # transparently expands the compact FC to full, runs the full-FC projection,
    # and re-compacts in phonopy's primitive view.
    fc_sym = symmetrize_by_projector(
        ph_copy.supercell,
        fc,
        2,
        primitive=ph_copy.primitive,
        log_level=2,
    )
    assert (fc - fc_sym).max() == pytest.approx(0.0010540, rel=1e-5)


def _random_fc(ph: Phonopy) -> np.ndarray:
    n_atoms = len(ph.supercell)
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_atoms, n_atoms, 3, 3))


def _get_rocksalt_with_layered_afm() -> PhonopyAtoms:
    """Return rocksalt conventional cell with antiferromagnetic layers along c."""
    return PhonopyAtoms(
        symbols=["Mn"] * 4 + ["O"] * 4,
        cell=np.eye(3) * 4.4,
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0, 0, 0.5],
            [0, 0.5, 0],
            [0.5, 0, 0],
        ],
        magnetic_moments=[2, 2, -2, -2, 0, 0, 0, 0],
    )


def test_symmetrize_force_constants_with_ferrimagnetic_order():
    """Test that the projector does not equate two magnetic sublattices.

    The two atoms of the bcc conventional cell have different magnetic
    moments. The body-centering translation is not a symmetry of this cell,
    so the on-site force constants of the two atoms have to stay different.

    """
    cell = PhonopyAtoms(
        symbols=["Fe", "Fe"],
        cell=np.eye(3) * 2.87,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=[2, 1],
    )
    ph = Phonopy(cell, supercell_matrix=[2, 2, 2], primitive_matrix=np.eye(3))
    ph.force_constants = _random_fc(ph)
    ph.symmetrize_force_constants(use_symfc_projector=True)
    fc = ph.force_constants
    i, j = ph.primitive.p2s_map
    assert np.abs(fc[i, i] - fc[j, j]).max() > 1e-3


@pytest.mark.parametrize("is_compact_fc", [False, True])
def test_symmetrize_force_constants_with_layered_afm(is_compact_fc: bool):
    """Test that force constants of a magnetic space group are kept.

    The magnetic moments of this cell lower the point group from cubic to
    tetragonal. The force constants made below have only the tetragonal
    symmetry, and projecting them again has to change nothing.

    The primitive cell of symfc has 2 atoms and the one of phonopy has 8
    atoms in this test. Compact force constants therefore go through the
    expansion to full force constants in ``symmetrize_by_projector``.

    """
    ph = Phonopy(
        _get_rocksalt_with_layered_afm(),
        supercell_matrix=[2, 2, 2],
        primitive_matrix=np.eye(3),
    )
    assert len(ph.symmetry.symmetry_operations["rotations"]) == 512
    fc_ref = symmetrize_by_projector(
        ph.supercell, _random_fc(ph), 2, symmetry=ph.symmetry
    )
    if is_compact_fc:
        fc_ref = full_fc_to_compact_fc(ph.primitive, fc_ref)
    ph.force_constants = fc_ref.copy()
    ph.symmetrize_force_constants(use_symfc_projector=True)
    np.testing.assert_allclose(ph.force_constants, fc_ref, atol=1e-10)


def test_symmetrize_force_constants_with_is_symmetry_false(ph_nacl: Phonopy):
    """Test that the projector does not impose the space group.

    The site symmetry of NaCl is m-3m and it would make the on-site force
    constants proportional to the unit matrix. With is_symmetry=False, only
    the sum rule, permutation and lattice translations are imposed.

    """
    ph = Phonopy(
        ph_nacl.unitcell,
        supercell_matrix=[2, 2, 2],
        primitive_matrix=ph_nacl.primitive_matrix,
        is_symmetry=False,
    )
    ph.force_constants = _random_fc(ph)
    ph.symmetrize_force_constants(use_symfc_projector=True)
    fc = ph.force_constants
    onsite = fc[0, 0]
    assert np.abs(onsite - np.eye(3) * np.trace(onsite) / 3).max() > 1e-3
    np.testing.assert_allclose(fc.sum(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(fc, fc.transpose(1, 0, 3, 2), atol=1e-10)


def test_symfc_with_large_symprec(ph_nacl: Phonopy):
    """Test symfc with a structure distorted within a large symprec.

    Symmetry operations are found with the large symprec. Symfc has to use
    the same tolerance, otherwise it fails to find the atomic permutations.

    """
    cell = ph_nacl.unitcell.copy()
    rng = np.random.default_rng(1)
    cell.scaled_positions = (
        cell.scaled_positions + rng.uniform(-1, 1, (len(cell), 3)) * 2e-4
    )
    ph = Phonopy(
        cell,
        supercell_matrix=[2, 2, 2],
        primitive_matrix=ph_nacl.primitive_matrix,
        symprec=1e-2,
    )
    assert len(ph.symmetry.symmetry_operations["rotations"]) == 1536

    solver = SymfcFCSolver(ph.supercell, symmetry=ph.symmetry)
    solver.compute_basis_set(orders=[2])

    ph.force_constants = _random_fc(ph)
    ph.symmetrize_force_constants(use_symfc_projector=True)
    onsite = ph.force_constants[0, 0]
    np.testing.assert_allclose(onsite, np.eye(3) * onsite[0, 0], atol=1e-10)


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
    )
    solver.run(orders=[2], is_compact_fc=False)
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
    )
    solver.run(orders=[2], is_compact_fc=True)
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


def test_symfc_force_constants_GeSn_mixture():
    """Symfc fc_calculator builds FC on a GeSn 50/50 mixture supercell.

    Mixed-species cells raise RuntimeError on ``cell.numbers`` because a
    mixture has no single atomic number. The symfc interface must fall
    back to ``species_ids`` (which already encodes the same "same species
    -> same opaque integer" contract that symfc / spglib expect from the
    ``numbers`` argument) so that the default fc_calculator path works
    end-to-end on site-mixture cells.

    """
    pytest.importorskip("symfc")

    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    weighted = apply_site_mixture(cell, [0.5, 0.5, 0.5, 0.5])
    ph = Phonopy(weighted, supercell_matrix=np.diag([2, 2, 2]))
    ph.generate_displacements(distance=0.01)

    n_sites = len(ph.supercell)
    assert ph.unmerged_supercell is not None
    n_expanded = len(ph.unmerged_supercell)
    assert n_expanded == 2 * n_sites

    rng = np.random.default_rng(seed=42)
    dataset = ph.dataset
    assert dataset is not None
    for entry in dataset["first_atoms"]:
        forces = rng.standard_normal((n_expanded, 3)) * 1e-3
        forces -= forces.mean(axis=0)
        entry["forces"] = forces
    ph.dataset = dataset

    ph.produce_force_constants(fc_calculator="symfc")

    fc = ph.force_constants
    assert fc is not None
    assert fc.shape == (n_sites, n_sites, 3, 3)
    assert ph.dataset["first_atoms"][0]["forces"].shape == (n_expanded, 3)
