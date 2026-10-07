# SPDX-License-Identifier: BSD-3-Clause
"""Tests for cells with weighted species of site mixture.

Weighted species are attached by apply_site_mixture and merged into sites by
merge_weighted_species.

"""

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
from phonopy.structure.cells import (
    apply_site_mixture,
    build_mixture_cell,
    get_atom_order,
    get_primitive,
    get_supercell,
    isclose,
    merge_weighted_species,
)
from phonopy.structure.symmetry import (
    Symmetry,
    _get_mapping_between_cells,
    symmetrize_borns_and_epsilon,
)

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


_Atom = tuple[str, list[float], float]
_rocksalt_fcc = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
_rocksalt_cl = [[0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]


def _make_NaKCl_cell(order: str) -> tuple[PhonopyAtoms, list[float]]:
    """Return conventional NaCl with K on the Na sites (Na 0.9, K 0.1).

    ``order`` is the order of the input atoms: "interleaved" (Na and K of one
    site next to each other), "by_element" (Na, K, Cl as POSCAR rows), or
    "Na_Cl_K".

    """
    na: list[_Atom] = [("Na", p, 0.9) for p in _rocksalt_fcc]
    k: list[_Atom] = [("K", p, 0.1) for p in _rocksalt_fcc]
    cl: list[_Atom] = [("Cl", p, 1.0) for p in _rocksalt_cl]
    if order == "interleaved":
        atoms = [a for pair in zip(na, k, strict=True) for a in pair] + cl
    elif order == "by_element":
        atoms = na + k + cl
    else:
        atoms = na + cl + k
    cell = PhonopyAtoms(
        cell=np.eye(3) * 5.69,
        symbols=[a[0] for a in atoms],
        scaled_positions=[a[1] for a in atoms],
    )
    return cell, [a[2] for a in atoms]


def _make_NaK_CsCl_cell(order: str) -> tuple[PhonopyAtoms, list[float]]:
    """Return a CsCl-type cell with sites Na0.9K0.1 and Na0.5K0.5.

    ``order`` is "by_site" (Na_A K_A Na_B K_B) or "by_element"
    (Na_A Na_B K_A K_B, as POSCAR rows).

    """
    a: list[float] = [0.0, 0.0, 0.0]
    b: list[float] = [0.5, 0.5, 0.5]
    if order == "by_site":
        symbols, positions, weights = (
            ["Na", "K", "Na", "K"],
            [a, a, b, b],
            [0.9, 0.1, 0.5, 0.5],
        )
    else:
        symbols, positions, weights = (
            ["Na", "Na", "K", "K"],
            [a, b, a, b],
            [0.9, 0.5, 0.1, 0.5],
        )
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0, symbols=symbols, scaled_positions=positions
    )
    return cell, weights


def _get_sites(cell: PhonopyAtoms) -> list[tuple]:
    """Return (species, scaled position modulo 1, mass) of each atom."""
    sites = []
    for sid, pos, mass in zip(
        cell.species_ids, cell.scaled_positions, cell.masses, strict=True
    ):
        sp = cell.species_table[sid]
        species = sp.mixture if sp.mixture is not None else sp.symbol
        sites.append((species, pos, mass))
    return sites


def _assert_same_sites(cell_a: PhonopyAtoms, cell_b: PhonopyAtoms):
    sites_a, sites_b = _get_sites(cell_a), _get_sites(cell_b)
    assert len(sites_a) == len(sites_b)
    for (sp_a, pos_a, mass_a), (sp_b, pos_b, mass_b) in zip(
        sites_a, sites_b, strict=True
    ):
        assert sp_a == sp_b
        diff = pos_a - pos_b
        diff -= np.rint(diff)
        np.testing.assert_allclose(diff, 0, atol=1e-8)
        assert mass_a == pytest.approx(mass_b)


_mixture_cells = [
    (_make_NaKCl_cell, "interleaved"),
    (_make_NaKCl_cell, "by_element"),
    (_make_NaKCl_cell, "Na_Cl_K"),
    (_make_NaK_CsCl_cell, "by_site"),
    (_make_NaK_CsCl_cell, "by_element"),
]


@pytest.mark.parametrize("make_cell,order", _mixture_cells)
def test_merge_weighted_species_equals_build_mixture_cell(make_cell, order):
    """Merging a weighted cell gives the cell of build_mixture_cell."""
    cell, weights = make_cell(order)
    site_cell, site_indices = merge_weighted_species(apply_site_mixture(cell, weights))
    _assert_same_sites(site_cell, build_mixture_cell(cell, weights))
    # Each atom is at the position of its site.
    diff = cell.scaled_positions - site_cell.scaled_positions[site_indices]
    diff -= np.rint(diff)
    np.testing.assert_allclose(diff, 0, atol=1e-8)


def test_merge_weighted_species_site_indices():
    """Site indices follow the first atom of each site."""
    cell, weights = _make_NaK_CsCl_cell("by_element")
    site_cell, site_indices = merge_weighted_species(apply_site_mixture(cell, weights))
    np.testing.assert_array_equal(site_indices, [0, 1, 0, 1])
    assert site_cell.species_table[site_cell.species_ids[0]].mixture == (
        ("Na", 0.9),
        ("K", 0.1),
    )
    assert site_cell.species_table[site_cell.species_ids[1]].mixture == (
        ("Na", 0.5),
        ("K", 0.5),
    )


@pytest.mark.parametrize("make_cell,order", _mixture_cells)
@pytest.mark.parametrize(
    "supercell_matrix",
    [
        np.diag([2, 2, 2]),
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 2]],
    ],
)
def test_merge_weighted_species_commutes_with_supercell(
    make_cell, order, supercell_matrix
):
    """Merging the supercell of a weighted cell gives the merged supercell.

    The supercell of the weighted cell, which is written for the calculator,
    and the supercell of the site cell, which phonopy calculates with, then
    have their sites in the same order.

    """
    cell, weights = make_cell(order)
    weighted = apply_site_mixture(cell, weights)
    site_cell, _ = merge_weighted_species(weighted)
    merged_supercell, _ = merge_weighted_species(
        get_supercell(weighted, supercell_matrix)
    )
    _assert_same_sites(merged_supercell, get_supercell(site_cell, supercell_matrix))


@pytest.mark.parametrize("order", ["interleaved", "by_element", "Na_Cl_K"])
def test_merge_weighted_species_commutes_with_primitive(order):
    """Merging the primitive cell of a weighted cell gives the merged one."""
    cell, weights = _make_NaKCl_cell(order)
    weighted = apply_site_mixture(cell, weights)
    site_cell, _ = merge_weighted_species(weighted)
    pmat = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    merged_primitive, _ = merge_weighted_species(get_primitive(weighted, pmat))
    _assert_same_sites(merged_primitive, get_primitive(site_cell, pmat))


def test_merge_weighted_species_ordinary_cell():
    """An ordinary cell is merged into itself."""
    cell, _ = _make_NaKCl_cell("by_element")
    cell = PhonopyAtoms(
        cell=cell.cell,
        symbols=cell.symbols[:4] + cell.symbols[8:],
        scaled_positions=np.vstack(
            [cell.scaled_positions[:4], cell.scaled_positions[8:]]
        ),
    )
    site_cell, site_indices = merge_weighted_species(cell)
    assert site_cell.symbols == cell.symbols
    np.testing.assert_array_equal(site_indices, np.arange(len(cell)))


def test_merge_weighted_species_rejects_merge_cell():
    """A cell with mixed-species sites cannot be merged again."""
    cell, weights = _make_NaK_CsCl_cell("by_site")
    with pytest.raises(ValueError):
        merge_weighted_species(build_mixture_cell(cell, weights))


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
    phonon = Phonopy(
        vca,
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="auto",
        site_mixture_scheme="split",
    )
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
    """Build a 2x2x2 Phonopy with the unit cell as primitive, split scheme."""
    return Phonopy(
        cell,
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix=np.eye(3),
        lang=lang,
        site_mixture_scheme="split",
    )


def test_normalization_masses_property_non_merge():
    """normalization_masses scales each mass by its concentration weight."""
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.9, 0.1, 0.9, 0.1])
    phonon = _phonon_from_cell(vca)
    phonon.force_constants = np.zeros(
        (len(phonon.supercell), len(phonon.supercell), 3, 3)
    )
    dm = phonon.dynamical_matrix
    primitive = dm.primitive
    np.testing.assert_allclose(
        dm.normalization_masses, primitive.masses * primitive.mixture_weights
    )
    # The reported masses are left untouched.
    np.testing.assert_allclose(primitive.masses, vca.masses[:2].tolist() * 2)


def test_normalization_masses_property_normal_cell():
    """For an ordinary cell normalization_masses equals the reported masses."""
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
    np.testing.assert_allclose(dm.normalization_masses, dm.primitive.masses)


def test_normalization_masses_property_merge_cell():
    """For a merge-style mixture cell normalization_masses equals averaged masses."""
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
    np.testing.assert_allclose(dm.normalization_masses, dm.primitive.masses)


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
    scaled_a = phonon_a.dynamical_matrix.normalization_masses
    phonon_b.masses = scaled_a / phonon_b.primitive.mixture_weights

    # The engineered effective masses coincide while the weights differ.
    np.testing.assert_allclose(
        phonon_b.dynamical_matrix.normalization_masses, scaled_a, atol=1e-12
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


def test_get_mapping_between_cells_co_located():
    """Mapping resolves co-located atoms of a site mixture by species.

    Position-only matching is ambiguous when Ge and Sn share a site, so
    the same position has two candidates. The mapping must disambiguate by
    species instead of raising "Index matching didn't go well."

    """
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    # Mapping a cell onto itself returns the identity order.
    np.testing.assert_array_equal(_get_mapping_between_cells(vca, vca), [0, 1, 2, 3])

    # Swapping the two co-located atoms at each site (Ge <-> Sn) is resolved
    # by species, recovering the permutation rather than a positional tie.
    swapped = apply_site_mixture(
        PhonopyAtoms(
            symbols=["Sn", "Ge", "Sn", "Ge"],
            scaled_positions=[
                [0, 0, 0],
                [0, 0, 0],
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25],
            ],
            cell=_zincblende_lattice,
        ),
        weights=[0.5, 0.5, 0.5, 0.5],
    )
    np.testing.assert_array_equal(
        _get_mapping_between_cells(vca, swapped), [1, 0, 3, 2]
    )


def test_get_atom_order_co_located_by_species():
    """Arbitrary-order matching resolves co-located atoms by species.

    A position-only match returns two candidates for a co-located site and
    previously bailed out; the species check now recovers the atom order.

    """
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    swapped = apply_site_mixture(
        PhonopyAtoms(
            symbols=["Sn", "Ge", "Sn", "Ge"],
            scaled_positions=[
                [0, 0, 0],
                [0, 0, 0],
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25],
            ],
            cell=_zincblende_lattice,
        ),
        weights=[0.5, 0.5, 0.5, 0.5],
    )
    order = get_atom_order(vca, swapped)
    np.testing.assert_array_equal(order, [1, 0, 3, 2])


def test_isclose_weights_matched_within_tolerance():
    """Site-mixture weights of different origin match within tolerance.

    Species weights are floats; cells built along different paths may carry
    weights that differ by rounding. isclose compares them with a tolerance
    rather than exact equality, so physically equal cells stay close.

    """
    base = _make_GeSn_co_located_cell()
    vca = apply_site_mixture(base, weights=[0.5, 0.5, 0.5, 0.5])
    perturbed = apply_site_mixture(base, weights=[0.5 + 1e-12, 0.5 - 1e-12, 0.5, 0.5])
    # Same order and arbitrary order both treat the cells as equivalent.
    assert isclose(vca, perturbed)
    np.testing.assert_array_equal(get_atom_order(vca, perturbed), [0, 1, 2, 3])
    # A genuinely different concentration is still rejected.
    distinct = apply_site_mixture(base, weights=[0.9, 0.1, 0.5, 0.5])
    assert not isclose(vca, distinct)


def test_symmetrize_borns_co_located_keeps_species():
    """Born symmetrization does not mix co-located Ge and Sn charges.

    The symmetry-operation pre-image of an atom is found by position, which
    is ambiguous at a co-located site. Selecting the wrong species would
    average Ge into Sn (and vice versa). Each species must keep its own
    Born effective charge.

    """
    vca = apply_site_mixture(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    eye = np.eye(3)
    # Ge (ids 0, 2) and Sn (ids 1, 3) carry opposite, isotropic charges that
    # already obey the acoustic sum rule, so symmetrization is the identity
    # only if species are not mixed.
    borns = np.array([2.0 * eye, -2.0 * eye, 2.0 * eye, -2.0 * eye])
    borns_, _ = symmetrize_borns_and_epsilon(borns, eye, vca)
    np.testing.assert_allclose(borns_[[0, 2]], [2.0 * eye, 2.0 * eye], atol=1e-8)
    np.testing.assert_allclose(borns_[[1, 3]], [-2.0 * eye, -2.0 * eye], atol=1e-8)


def _merged_phonon(
    make_cell=_make_NaK_CsCl_cell, order: str = "by_element"
) -> tuple[Phonopy, PhonopyAtoms]:
    """Return Phonopy of the merge scheme and the input weighted unit cell."""
    cell, weights = make_cell(order)
    weighted = apply_site_mixture(cell, weights)
    phonon = Phonopy(
        weighted, supercell_matrix=np.diag([2, 2, 2]), primitive_matrix="P"
    )
    return phonon, weighted


@pytest.mark.parametrize("make_cell,order", _mixture_cells)
def test_phonopy_merge_scheme_cells(make_cell, order):
    """With the merge scheme, unitcell is of sites and unmerged_* of atoms.

    The sites of the unmerged cells are those of the cells of phonopy, in the
    same order.

    """
    phonon, weighted = _merged_phonon(make_cell, order)
    assert phonon.site_mixture_scheme == "merge"
    assert phonon.unitcell.has_mixtures
    site_cell, _ = merge_weighted_species(weighted)
    _assert_same_sites(phonon.unitcell, site_cell)
    unmerged_unitcell = phonon.unmerged_unitcell
    assert unmerged_unitcell is not None
    assert unmerged_unitcell.symbols == weighted.symbols
    np.testing.assert_allclose(
        unmerged_unitcell.scaled_positions, weighted.scaled_positions
    )
    for cell, unmerged in (
        (phonon.supercell, phonon.unmerged_supercell),
        (phonon.primitive, phonon.unmerged_primitive),
    ):
        assert unmerged is not None
        assert len(unmerged) == len(cell) * len(weighted) // len(site_cell)
        _assert_same_sites(cell, merge_weighted_species(unmerged)[0])


def test_phonopy_unmerged_cells_none_without_merge():
    """unmerged_* are None with the split scheme and for an ordinary cell."""
    cell, weights = _make_NaK_CsCl_cell("by_element")
    split = Phonopy(
        apply_site_mixture(cell, weights),
        supercell_matrix=np.diag([2, 2, 2]),
        site_mixture_scheme="split",
    )
    ordinary = Phonopy(
        PhonopyAtoms(
            cell=np.eye(3) * 4.0,
            symbols=["Na", "Cl"],
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        ),
        supercell_matrix=np.diag([2, 2, 2]),
    )
    for phonon in (split, ordinary):
        assert phonon.unmerged_unitcell is None
        assert phonon.unmerged_primitive is None
        assert phonon.unmerged_supercell is None
    assert len(split.supercell) == 32


def test_phonopy_rejects_unknown_site_mixture_scheme():
    """site_mixture_scheme is "merge" or "split"."""
    cell, weights = _make_NaK_CsCl_cell("by_element")
    with pytest.raises(ValueError):
        Phonopy(
            apply_site_mixture(cell, weights),
            site_mixture_scheme="merged",  # type: ignore[arg-type]
        )


def test_phonopy_merge_scheme_displacements():
    """The displacement of a site is given to every atom of the site."""
    phonon, _ = _merged_phonon()
    phonon.generate_displacements()
    unmerged_supercell = phonon.unmerged_supercell
    assert unmerged_supercell is not None
    _, site_indices = merge_weighted_species(unmerged_supercell)
    dataset = phonon.dataset
    assert dataset is not None
    cells = phonon.supercells_with_displacements
    assert cells is not None
    assert len(cells) == len(dataset["first_atoms"]) == 2
    for disp, cell in zip(dataset["first_atoms"], cells, strict=True):
        assert cell.symbols == unmerged_supercell.symbols
        diff = cell.positions - unmerged_supercell.positions
        moved = site_indices == disp["number"]
        assert moved.sum() == 2
        np.testing.assert_allclose(diff[moved], [disp["displacement"]] * 2)
        np.testing.assert_allclose(diff[~moved], 0, atol=1e-12)


def test_phonopy_merge_scheme_force_constants():
    """Forces on the atoms are summed per site for the force constants.

    The forces on the atoms of each site are the site force divided by the
    weights. The force constants are compared with those from the site
    forces given to the cell of mixed-species sites.

    """
    phonon, weighted = _merged_phonon()
    phonon.generate_displacements()
    unmerged_supercell = phonon.unmerged_supercell
    assert unmerged_supercell is not None
    _, site_indices = merge_weighted_species(unmerged_supercell)
    weights = unmerged_supercell.mixture_weights
    assert weights is not None

    rng = np.random.default_rng(7)
    site_forces = rng.standard_normal((2, len(phonon.supercell), 3))
    site_forces -= site_forces.mean(axis=1, keepdims=True)
    phonon.forces = site_forces[:, site_indices] * weights[None, :, None]
    phonon.produce_force_constants()

    cell, cell_weights = _make_NaK_CsCl_cell("by_element")
    reference = Phonopy(
        build_mixture_cell(cell, cell_weights),
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="P",
    )
    reference.dataset = phonon.dataset
    reference.forces = site_forces
    reference.produce_force_constants()
    assert phonon.force_constants is not None
    assert reference.force_constants is not None
    np.testing.assert_allclose(
        phonon.force_constants, reference.force_constants, atol=1e-10
    )


def test_phonopy_merge_scheme_replicate():
    """Replicate keeps the scheme and the unmerged unit cell."""
    phonon, weighted = _merged_phonon()
    replica = phonon.replicate()
    assert replica.site_mixture_scheme == "merge"
    unmerged_unitcell = replica.unmerged_unitcell
    assert unmerged_unitcell is not None
    assert unmerged_unitcell.symbols == weighted.symbols
    _assert_same_sites(replica.supercell, phonon.supercell)
