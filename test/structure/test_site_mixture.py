"""Tests for the non-merge site-mixture scheme (apply_site_mixture)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from phonopy import Phonopy
from phonopy.structure.atoms import (
    PhonopyAtoms,
    build_species_table_from_mixtures,
    parse_cell_dict,
)
from phonopy.structure.cells import apply_site_mixture
from phonopy.structure.symmetry import Symmetry

_a = 5.789
_zincblende_lattice = [[0, _a / 2, _a / 2], [_a / 2, 0, _a / 2], [_a / 2, _a / 2, 0]]


def _make_GeSn_co_located_cell() -> PhonopyAtoms:
    """Return a zincblende cell with Ge and Sn co-located on both sites."""
    return PhonopyAtoms(
        symbols=["Ge", "Sn", "Ge", "Sn"],
        scaled_positions=[
            [0, 0, 0],
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25],
        ],
        cell=_zincblende_lattice,
    )


def test_apply_site_mixture_basic():
    """apply_site_mixture attaches weights without merging atoms."""
    cell = _make_GeSn_co_located_cell()
    vca = apply_site_mixture(cell, weights=[0.5, 0.5, 0.5, 0.5])
    assert len(vca) == len(cell)
    assert vca.symbols == cell.symbols
    np.testing.assert_array_equal(vca.numbers, cell.numbers)
    np.testing.assert_allclose(vca.masses, cell.masses)
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5, 0.5, 0.5])
    assert vca.has_weighted_species
    assert not vca.has_mixtures
    # Both Ge atoms share one weighted species; likewise both Sn atoms.
    assert len(vca.species_table) == 2
    np.testing.assert_array_equal(vca.species_ids, [0, 1, 0, 1])
    # The input cell is not modified.
    assert not cell.has_weighted_species


def test_apply_site_mixture_isolated_atom_keeps_species():
    """An isolated atom with weight 1.0 keeps its unweighted species."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn", "Si"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        cell=_zincblende_lattice,
    )
    vca = apply_site_mixture(cell, weights=[0.5, 0.5, 1.0])
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5, 1.0])
    si = vca.species_table[int(vca.species_ids[2])]
    assert si.symbol == "Si"
    assert si.weight is None


def test_apply_site_mixture_all_unity_weights_on_normal_cell():
    """A cell without overlaps and all-1.0 weights stays an ordinary cell."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=_zincblende_lattice,
    )
    vca = apply_site_mixture(cell, weights=[1.0, 1.0])
    assert not vca.has_weighted_species
    assert vca.mixture_weights is None
    assert vca.species_table == cell.species_table


def test_apply_site_mixture_degenerate_same_species_allowed():
    """Co-located atoms of the same (element, weight) are accepted."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Ge"],
        scaled_positions=[[0, 0, 0], [0, 0, 0]],
        cell=_zincblende_lattice,
    )
    vca = apply_site_mixture(cell, weights=[0.5, 0.5])
    assert len(vca.species_table) == 1
    np.testing.assert_array_equal(vca.species_ids, [0, 0])
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5])


def test_apply_site_mixture_yaml_roundtrip():
    """Weighted species serialize a per-atom weight and round-trip via YAML."""
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])

    data = yaml.safe_load(str(vca))
    # Each point carries its concentration weight (not a merged mixture).
    assert all("mixture" not in p for p in data["points"])
    np.testing.assert_allclose([p["weight"] for p in data["points"]], 0.5)

    restored = parse_cell_dict(data)
    assert restored is not None
    assert restored.has_weighted_species
    assert not restored.has_mixtures
    assert restored.symbols == vca.symbols
    np.testing.assert_array_equal(restored.species_ids, vca.species_ids)
    np.testing.assert_allclose(restored.mixture_weights, vca.mixture_weights)
    np.testing.assert_allclose(restored.masses, vca.masses)
    np.testing.assert_array_equal(restored.numbers, vca.numbers)


def test_apply_site_mixture_length_mismatch():
    """Length of weights must match the number of atoms."""
    cell = _make_GeSn_co_located_cell()
    with pytest.raises(ValueError):
        apply_site_mixture(cell, weights=[0.5, 0.5])


def test_apply_site_mixture_isolated_atom_weight_error():
    """An isolated atom must carry weight 1.0."""
    cell = PhonopyAtoms(
        symbols=["Ge"],
        scaled_positions=[[0, 0, 0]],
        cell=_zincblende_lattice,
    )
    with pytest.raises(ValueError):
        apply_site_mixture(cell, weights=[0.5])


def test_apply_site_mixture_group_sum_error():
    """Weights of a co-located group must sum to 1.0."""
    cell = _make_GeSn_co_located_cell()
    with pytest.raises(ValueError):
        apply_site_mixture(cell, weights=[0.6, 0.6, 0.5, 0.5])


def test_apply_site_mixture_rejects_weighted_cell():
    """apply_site_mixture cannot be applied twice."""
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        apply_site_mixture(vca, weights=[0.5, 0.5, 0.5, 0.5])


def test_apply_site_mixture_rejects_merge_cell():
    """apply_site_mixture cannot be applied to a merge-style mixture cell."""
    species, ids = build_species_table_from_mixtures([[("Ge", 0.5), ("Sn", 0.5)]])
    merge_cell = PhonopyAtoms(
        cell=_zincblende_lattice,
        scaled_positions=[[0, 0, 0]],
        species_table=species,
        species_ids=ids,
    )
    with pytest.raises(ValueError):
        apply_site_mixture(merge_cell, weights=[1.0])


def test_apply_site_mixture_rejects_magnetic_cell():
    """apply_site_mixture does not support cells carrying magnetic moments."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [0, 0, 0]],
        cell=_zincblende_lattice,
        magnetic_moments=[1.0, -1.0],
    )
    with pytest.raises(ValueError):
        apply_site_mixture(cell, weights=[0.5, 0.5])


def test_apply_site_mixture_symprec_controls_grouping():
    """Overlap detection follows the symprec tolerance."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [1e-4, 0, 0]],
        cell=_zincblende_lattice,
    )
    # Within a loose tolerance the two atoms form one group.
    vca = apply_site_mixture(cell, weights=[0.5, 0.5], symprec=1e-3)
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5])
    # With the default tolerance they are isolated atoms, whose weights
    # must be 1.0.
    with pytest.raises(ValueError):
        apply_site_mixture(cell, weights=[0.5, 0.5])


def test_symmetry_GeSn_50_50_co_located():
    """Diamond symmetry of the 50/50 co-located cell is found by spglib."""
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    symmetry = Symmetry(vca)
    assert symmetry.dataset.number == 227  # Fd-3m
    assert len(symmetry.symmetry_operations["rotations"]) == 48
    # Ge@A is equivalent to Ge@B; likewise for Sn. One independent atom
    # per species.
    np.testing.assert_array_equal(symmetry.get_map_atoms(), [0, 1, 0, 1])
    np.testing.assert_array_equal(symmetry.get_independent_atoms(), [0, 1])


def test_symmetry_permutations_do_not_mix_species():
    """Atomic permutations keep co-located Ge and Sn within their species.

    Position-only matching is ambiguous when Ge and Sn share a site, so
    the type-aware matcher is required. Every operation must map each
    atom onto an atom of the same species (Ge ids 0/2, Sn ids 1/3).

    """
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    symmetry = Symmetry(vca)
    perms = symmetry.atomic_permutations
    species = np.array(vca.species_ids)
    for perm in perms:
        np.testing.assert_array_equal(species[perm], species)


def test_symmetry_distinct_concentrations_lower_symmetry():
    """Sites with different concentrations are not symmetry-equivalent."""
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.9, 0.1, 0.5, 0.5])
    symmetry = Symmetry(vca)
    assert symmetry.dataset.number == 216  # F-43m, no A<->B swap
    assert len(vca.species_table) == 4


def test_phonopy_construction_and_displacements_co_located():
    """Phonopy builds a co-located species-resolved cell, one displacement each.

    The supercell keeps every constituent atom (no merging), weights
    propagate through the species table, and the symmetry-reduced
    displacements move one Ge and one Sn independently.

    """
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    phonon = Phonopy(vca, supercell_matrix=np.diag([2, 2, 2]), primitive_matrix="auto")
    supercell = phonon.supercell
    assert len(supercell) == 32  # 4 atoms x 8, nothing merged away
    assert len(phonon.primitive) == 4
    np.testing.assert_allclose(supercell.mixture_weights, 0.5)

    # Permutations (supercell symmetry and primitive translations) never
    # map an atom onto a different species.
    species = np.array(supercell.species_ids)
    for perm in phonon.symmetry.atomic_permutations:
        np.testing.assert_array_equal(species[perm], species)
    for perm in phonon.primitive.atomic_permutations:
        np.testing.assert_array_equal(species[perm], species)

    phonon.generate_displacements(distance=0.01)
    first_atoms = phonon.dataset["first_atoms"]
    displaced_species = sorted(
        int(supercell.species_ids[d["number"]]) for d in first_atoms
    )
    # One independent Ge (species 0) and one independent Sn (species 1).
    assert displaced_species == [0, 1]


# ---------------------------------------------------------------------------
# Eq64: VCA effective mass x_i * M_i in the dynamical matrix (non-merge only).
# ---------------------------------------------------------------------------


def _symmetric_fc(natom: int, seed: int) -> np.ndarray:
    """Return a synthetic full force-constant array fc[i,j,a,b].

    The array is made symmetric under (i,a) <-> (j,b) so the dynamical
    matrix is Hermitian. The values are arbitrary; only consistency
    between runs matters for these comparison tests.

    """
    rng = np.random.default_rng(seed)
    fc = rng.standard_normal((natom, natom, 3, 3))
    return 0.5 * (fc + fc.transpose(1, 0, 3, 2))


def _phonon_from_cell(cell: PhonopyAtoms, lang: str = "C") -> Phonopy:
    """Build a 2x2x2 Phonopy with the unit cell as primitive."""
    return Phonopy(
        cell,
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix=np.eye(3),
        lang=lang,
    )


def test_scaled_masses_property_non_merge():
    """scaled_masses scales each mass by its concentration weight."""
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.9, 0.1, 0.9, 0.1])
    phonon = _phonon_from_cell(vca)
    phonon.force_constants = np.zeros(
        (len(phonon.supercell), len(phonon.supercell), 3, 3)
    )
    dm = phonon.dynamical_matrix
    primitive = dm.primitive
    np.testing.assert_allclose(
        dm.scaled_masses, primitive.masses * primitive.mixture_weights
    )
    # The reported masses are left untouched.
    np.testing.assert_allclose(primitive.masses, vca.masses[:2].tolist() * 2)


def test_scaled_masses_property_normal_cell():
    """For an ordinary cell scaled_masses equals the reported masses."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=_zincblende_lattice,
    )
    phonon = _phonon_from_cell(cell)
    phonon.force_constants = np.zeros(
        (len(phonon.supercell), len(phonon.supercell), 3, 3)
    )
    dm = phonon.dynamical_matrix
    assert dm.primitive.mixture_weights is None
    np.testing.assert_allclose(dm.scaled_masses, dm.primitive.masses)


def test_scaled_masses_property_merge_cell():
    """For a merge-style mixture cell scaled_masses equals the averaged masses."""
    species, ids = build_species_table_from_mixtures([[("Ge", 0.5), ("Sn", 0.5)]])
    merge = PhonopyAtoms(
        cell=_zincblende_lattice,
        scaled_positions=[[0, 0, 0]],
        species_table=species,
        species_ids=ids,
    )
    assert merge.has_mixtures
    phonon = _phonon_from_cell(merge)
    phonon.force_constants = np.zeros(
        (len(phonon.supercell), len(phonon.supercell), 3, 3)
    )
    dm = phonon.dynamical_matrix
    assert dm.primitive.mixture_weights is None
    np.testing.assert_allclose(dm.scaled_masses, dm.primitive.masses)


@pytest.mark.parametrize("lang", ["C", "Rust"])
def test_eq64_frequencies_depend_only_on_scaled_mass(lang):
    """Frequencies and group velocities are driven by x_i * M_i, not M_i.

    Two non-merge cells share one structure and one force constant array
    but use different (weight, mass) pairs chosen so the products x_i * M_i
    coincide. Eq64 makes both dynamical matrices identical, so frequencies
    and group velocities must match. With the pre-Eq64 behaviour (bare
    masses) the two runs would differ because their reported masses differ.

    """
    cell = _make_GeSn_co_located_cell()

    phonon_a = _phonon_from_cell(apply_site_mixture(cell, [0.5, 0.5, 0.5, 0.5]), lang)
    fc = _symmetric_fc(len(phonon_a.supercell), seed=0)
    phonon_a.force_constants = fc

    phonon_b = _phonon_from_cell(
        apply_site_mixture(cell, [0.25, 0.75, 0.25, 0.75]), lang
    )
    phonon_b.force_constants = fc
    # Pick masses so that mass * weight equals phonon_a's scaled masses.
    scaled_a = phonon_a.dynamical_matrix.scaled_masses
    phonon_b.masses = scaled_a / phonon_b.primitive.mixture_weights

    # The engineered effective masses coincide while the weights differ.
    np.testing.assert_allclose(
        phonon_b.dynamical_matrix.scaled_masses, scaled_a, atol=1e-12
    )

    qpoints = [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.5, 0.0, 0.0]]
    phonon_a.run_qpoints(qpoints, with_group_velocities=True)
    phonon_b.run_qpoints(qpoints, with_group_velocities=True)
    res_a = phonon_a.qpoints
    res_b = phonon_b.qpoints
    np.testing.assert_allclose(res_a.frequencies, res_b.frequencies, atol=1e-8)
    np.testing.assert_allclose(
        res_a.group_velocities, res_b.group_velocities, atol=1e-8
    )


def test_eq64_weights_change_frequencies():
    """Changing the concentration weights changes the frequencies."""
    cell = _make_GeSn_co_located_cell()

    phonon_a = _phonon_from_cell(apply_site_mixture(cell, [0.5, 0.5, 0.5, 0.5]))
    fc = _symmetric_fc(len(phonon_a.supercell), seed=1)
    phonon_a.force_constants = fc

    phonon_c = _phonon_from_cell(apply_site_mixture(cell, [0.9, 0.1, 0.9, 0.1]))
    phonon_c.force_constants = fc

    qpoints = [[0.1, 0.2, 0.3]]
    phonon_a.run_qpoints(qpoints)
    phonon_c.run_qpoints(qpoints)
    freqs_a = phonon_a.qpoints.frequencies
    freqs_c = phonon_c.qpoints.frequencies
    assert not np.allclose(freqs_a, freqs_c)
