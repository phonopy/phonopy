# SPDX-License-Identifier: BSD-3-Clause
"""Tests of Octopus calculator interface."""

import math
import sys

import numpy as np
import pytest

from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.octopus import (
    get_born_octopus,
    get_cell_from_octopus_lines,
    get_octopus_structure_lines,
    is_octopus_geometry,
    parse_octopus_born_charges,
    parse_octopus_epsilon,
    read_octopus,
    write_octopus,
)
from phonopy.interface.vasp import write_vasp
from phonopy.physical_units import get_physical_units
from phonopy.scripts.phonopy_octopus_eigenmodes import OctopusPhononModes
from phonopy.structure.atoms import PhonopyAtoms

# Real Octopus em_resp output for rock-salt NaCl (static response).
_EPSILON_FILE = """\
# Real part of dielectric constant
            2.588021            0.000000            0.000000
            0.000000            2.588021            0.000000
            0.000000            0.000000            2.588021
Isotropic average            2.588021

# Imaginary part of dielectric constant
            0.000000            0.000000            0.000000
            0.000000            0.000000            0.000000
            0.000000            0.000000            0.000000
Isotropic average            0.000000
"""

_BORN_FILE = """\
# (Frequency-dependent) Born effective charge tensors
Index:     1   Label:    Na   Ionic charge:     1.0000
            1.141799            0.000000           -0.000000
            0.000000            1.141799            0.000000
            0.000000            0.000000            1.141799
Isotropic average            1.141799

Index:     2   Label:    Cl   Ionic charge:     7.0000
           -1.141799            0.000000            0.000000
            0.000000           -1.141799           -0.000000
            0.000000           -0.000000           -1.141799
Isotropic average           -1.141799

# Discrepancy of Born effective charges from acoustic sum rule
           -0.025775            0.000000           -0.000000
            0.000000           -0.025775            0.000000
            0.000000            0.000000           -0.025775
Isotropic average           -0.025775
"""


def _nacl_cell() -> PhonopyAtoms:
    a = 5.64
    return PhonopyAtoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=(a / 2) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype="double"),
    )


def test_octopus_structure_lines_roundtrip():
    """get_cell_from_octopus_lines inverts get_octopus_structure_lines."""
    cell = _nacl_cell()
    result = get_cell_from_octopus_lines(get_octopus_structure_lines(cell))

    assert list(result.symbols) == list(cell.symbols)
    np.testing.assert_allclose(result.cell, cell.cell, atol=1e-6)
    np.testing.assert_allclose(
        result.scaled_positions, cell.scaled_positions, atol=1e-8
    )


def test_octopus_structure_lines_roundtrip_triclinic():
    """The writer must be exact for cells with unequal, non-orthogonal vectors.

    Octopus reconstructs lattice vector i as LatticeParameters[i] * row i of
    %LatticeVectors, so each row must be normalized by its own length.
    """
    cell = PhonopyAtoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.array([[3.0, 0.0, 0.0], [1.0, 4.0, 0.0], [1.0, 2.0, 5.0]]),
    )
    result = get_cell_from_octopus_lines(get_octopus_structure_lines(cell))

    np.testing.assert_allclose(result.cell, cell.cell, atol=1e-6)
    np.testing.assert_allclose(
        result.scaled_positions, cell.scaled_positions, atol=1e-8
    )


def test_get_cell_from_octopus_lines_fixed_reference():
    """Parse a known 2x2x2 Si supercell block embedded among ignored lines.

    Non-block lines (e.g. ``CalculationMode = gs``) and comments must be
    ignored, and lattice vector i must be scaled by lattice parameter i.
    """
    lines = [
        "CalculationMode = gs   # ignored",
        "",
        "%LatticeParameters",
        " 14.511546484 | 14.511546484 | 14.511546484",
        "%",
        "# a comment",
        "%LatticeVectors",
        "  0.000000000 | 0.707106781 | 0.707106781",
        "  0.707106781 | 0.000000000 | 0.707106781",
        "  0.707106781 | 0.707106781 | 0.000000000",
        "%",
        "%ReducedCoordinates",
        '  "Si" | 0.000689106 | 0.000000000 | 0.000000000',
        '  "Si" | 0.500000000 | 0.000000000 | 0.000000000',
        "%",
    ]
    cell = get_cell_from_octopus_lines(lines)

    off = 14.511546484 * 0.707106781
    expected_cell = np.array([[0, off, off], [off, 0, off], [off, off, 0]])
    assert list(cell.symbols) == ["Si", "Si"]
    np.testing.assert_allclose(cell.cell, expected_cell, atol=1e-6)
    np.testing.assert_allclose(
        cell.scaled_positions,
        [[0.000689106, 0.0, 0.0], [0.5, 0.0, 0.0]],
        atol=1e-9,
    )


def test_get_cell_from_octopus_lines_rejects_non_numeric():
    """A geometry with unresolved variables (not the canonical format) fails."""
    lines = [
        "%LatticeParameters",
        " a | a | a",
        "%",
        "%LatticeVectors",
        " 0 | 0.5 | 0.5",
        " 0.5 | 0 | 0.5",
        " 0.5 | 0.5 | 0",
        "%",
        "%ReducedCoordinates",
        ' "Na" | 0 | 0 | 0',
        "%",
    ]
    with pytest.raises(ValueError):
        get_cell_from_octopus_lines(lines)


def test_read_octopus_file_roundtrip(tmp_path):
    """read_octopus reads back what write_octopus wrote."""
    cell = _nacl_cell()
    filename = tmp_path / "geometry-unitcell"
    write_octopus(filename, cell)
    result = read_octopus(filename)

    assert list(result.symbols) == list(cell.symbols)
    np.testing.assert_allclose(result.cell, cell.cell, atol=1e-6)
    np.testing.assert_allclose(
        result.scaled_positions, cell.scaled_positions, atol=1e-8
    )


def test_parse_octopus_epsilon(tmp_path):
    """parse_octopus_epsilon reads the real part of the dielectric tensor."""
    filename = tmp_path / "epsilon"
    filename.write_text(_EPSILON_FILE)
    epsilon = parse_octopus_epsilon(filename)

    assert epsilon.shape == (3, 3)
    np.testing.assert_allclose(epsilon, 2.588021 * np.eye(3), atol=1e-8)


def test_parse_octopus_born_charges(tmp_path):
    """parse_octopus_born_charges reads Z* and ignores the ASR discrepancy block."""
    filename = tmp_path / "born_charges"
    filename.write_text(_BORN_FILE)
    borns, labels = parse_octopus_born_charges(filename)

    assert borns.shape == (2, 3, 3)  # discrepancy block must not be parsed as an atom
    assert labels == ["Na", "Cl"]
    np.testing.assert_allclose(borns[0], 1.141799 * np.eye(3), atol=1e-8)
    np.testing.assert_allclose(borns[1], -1.141799 * np.eye(3), atol=1e-8)


def test_get_born_octopus_end_to_end(tmp_path):
    """get_born_octopus ties the cell, epsilon and born_charges together."""
    write_octopus(tmp_path / "geometry-unitcell", _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE)

    borns, epsilon, atom_indices = get_born_octopus(
        tmp_path / "geometry-unitcell",
        tmp_path / "epsilon",
        tmp_path / "born_charges",
    )

    # NaCl: Na and Cl are symmetry-independent, so both survive.
    assert atom_indices.tolist() == [0, 1]
    assert borns.shape == (2, 3, 3)
    np.testing.assert_allclose(epsilon, 2.588021 * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(borns[0], 1.141799 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(borns[1], -1.141799 * np.eye(3), atol=1e-5)


def test_is_octopus_geometry(tmp_path):
    """A geometry include file is detected; a POSCAR is not."""
    cell = _nacl_cell()
    geom = tmp_path / "geometry-unitcell"
    poscar = tmp_path / "POSCAR"
    write_octopus(geom, cell)
    write_vasp(poscar, cell)

    assert is_octopus_geometry(geom) is True
    assert is_octopus_geometry(poscar) is False


def test_read_crystal_structure_octopus_accepts_geometry(tmp_path):
    """--octopus reads an Octopus geometry file (atomic units; no conversion)."""
    cell = _nacl_cell()
    geom = tmp_path / "geometry-unitcell"
    write_octopus(geom, cell)
    unitcell, _ = read_crystal_structure(geom, interface_mode="octopus")

    assert list(unitcell.symbols) == ["Na", "Cl"]
    np.testing.assert_allclose(unitcell.cell, cell.cell, atol=1e-6)


def test_read_crystal_structure_octopus_accepts_poscar(tmp_path):
    """--octopus still reads a POSCAR, converting the lattice Angstrom -> Bohr."""
    cell = _nacl_cell()  # lattice numbers treated as Angstrom in a POSCAR
    poscar = tmp_path / "POSCAR"
    write_vasp(poscar, cell)
    unitcell, _ = read_crystal_structure(poscar, interface_mode="octopus")

    bohr_angstrom = 1.0 / get_physical_units().Bohr
    assert list(unitcell.symbols) == ["Na", "Cl"]
    np.testing.assert_allclose(unitcell.cell, cell.cell * bohr_angstrom, atol=1e-6)


def test_get_born_octopus_accepts_poscar(tmp_path):
    """get_born_octopus also accepts the unit cell as a POSCAR."""
    write_vasp(tmp_path / "POSCAR", _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE)

    borns, epsilon, atom_indices = get_born_octopus(
        tmp_path / "POSCAR",
        tmp_path / "epsilon",
        tmp_path / "born_charges",
    )

    assert atom_indices.tolist() == [0, 1]
    np.testing.assert_allclose(epsilon, 2.588021 * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(borns[0], 1.141799 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(borns[1], -1.141799 * np.eye(3), atol=1e-5)


# Complex (frequency-dependent) Born charges as printed by Octopus when
# write_real is false: "Real:"/"Imaginary:" sub-blocks after each "Index:".
_BORN_FILE_COMPLEX = """\
# (Frequency-dependent) Born effective charge tensors
Index:     1   Label:    Na   Ionic charge:     1.0000
Real:
            1.141799            0.000000           -0.000000
            0.000000            1.141799            0.000000
            0.000000            0.000000            1.141799
Imaginary:
            0.010000            0.000000            0.000000
            0.000000            0.010000            0.000000
            0.000000            0.000000            0.010000
"""

# Born charges violating the cubic site symmetry of rock-salt NaCl by far
# more than the 0.1 threshold of symmetrize_borns_and_epsilon.
_BORN_FILE_BROKEN_SYMMETRY = _BORN_FILE.replace(
    "            1.141799            0.000000           -0.000000",
    "            2.141799            0.000000           -0.000000",
    1,
)


def test_parse_octopus_born_charges_complex_rejected(tmp_path):
    """Frequency-dependent (complex) Born charges give a clear error."""
    filename = tmp_path / "born_charges"
    filename.write_text(_BORN_FILE_COMPLEX)
    with pytest.raises(ValueError, match="complex"):
        parse_octopus_born_charges(filename)


def test_get_born_octopus_warns_on_broken_symmetry(tmp_path):
    """Symmetry-broken Born charges emit a UserWarning."""
    write_octopus(tmp_path / "geometry-unitcell", _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE_BROKEN_SYMMETRY)

    with pytest.warns(UserWarning):
        get_born_octopus(
            tmp_path / "geometry-unitcell",
            tmp_path / "epsilon",
            tmp_path / "born_charges",
        )


def test_phonopy_octopus_born_script_symmetry_broken(tmp_path, monkeypatch, capsys):
    """The CLI reports '# Symmetry broken' instead of printing a BORN file."""
    from phonopy.scripts.phonopy_octopus_born import run

    geom = tmp_path / "geometry-unitcell"
    write_octopus(geom, _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE_BROKEN_SYMMETRY)

    monkeypatch.setattr(sys, "argv", ["phonopy-octopus-born", str(geom), str(tmp_path)])
    with pytest.raises(SystemExit):
        run()
    assert "# Symmetry broken" in capsys.readouterr().out


def _parse_phonon_modes_file(filename):
    """Parse a phonon modes file following the Octopus phonon_modes.F90 reads.

    Mimics the Fortran parser (format 1.0): header of ``Version:``,
    ``Nmodes:``, ``Natoms:``, ``Np:`` and a ``Masses:`` block, then per mode
    a ``frequency:`` line, one line of 3 floats per supercell atom, and an
    ``alpha:`` line. ``#`` comment lines between blocks are skipped.
    """
    with open(filename) as f:
        lines = [
            line.rstrip("\n")
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    assert lines[0].split() == ["Version:", "1.0"]
    assert lines[1].split()[0] == "Nmodes:"
    num_modes = int(lines[1].split()[1])
    assert lines[2].split()[0] == "Natoms:"
    num_atoms = int(lines[2].split()[1])
    assert lines[3].split()[0] == "Np:"
    num_super = int(lines[3].split()[1])
    assert lines[4].split() == ["Masses:"]
    i = 5
    masses = []
    while len(masses) < num_atoms:
        masses.extend(float(x) for x in lines[i].split())
        i += 1
    assert len(masses) == num_atoms

    modes = []
    while i < len(lines):
        assert lines[i].split()[0] == "frequency:", f"line: {lines[i]!r}"
        freq = float(lines[i].split()[1])
        i += 1
        vec = []
        while not lines[i].startswith("alpha:"):
            vals = [float(x) for x in lines[i].split()]
            assert len(vals) == 3
            vec.append(vals)
            i += 1
        assert len(vec) == num_atoms
        alpha = float(lines[i].split()[1])
        i += 1
        modes.append({"frequency": freq, "vec": np.array(vec), "alpha": alpha})

    return num_modes, num_atoms, num_super, np.array(masses), modes


def test_octopus_phonon_modes_file(ph_nacl_nonac, tmp_path):
    """The phonon modes file encodes correct, complete supercell eigenmodes.

    NaCl 2x2x2 with an F-centered primitive matrix: 64 supercell atoms,
    2-atom primitive cell, Np = 32 (exercises primitive != unit cell).
    """
    phonon = ph_nacl_nonac
    modes_obj = OctopusPhononModes(phonon)
    filename = tmp_path / "phonon_modes.txt"
    modes_obj.write_phonon_file(str(filename))

    num_modes, natoms_header, np_header, file_masses, modes = _parse_phonon_modes_file(
        filename
    )
    n_super = len(phonon.supercell)
    n_prim = len(phonon.primitive)
    Np = n_super // n_prim

    assert natoms_header == n_super
    assert np_header == Np
    np.testing.assert_allclose(file_masses, phonon.supercell.masses, atol=1e-6)
    # Complete supercell mode set minus the 3 excluded acoustic Gamma modes.
    assert num_modes == 3 * n_super - 3
    assert len(modes) == num_modes
    assert all(m["vec"].shape == (n_super, 3) for m in modes)
    assert all(m["frequency"] > 0 for m in modes)  # acoustic Gamma excluded

    # alpha convention: 1/sqrt(2) for region A, 1/2 for region B.
    alphas = sorted({round(m["alpha"], 8) for m in modes})
    assert alphas == [0.5, round(1 / math.sqrt(2.0), 8)]

    # Norms: |W|^2 = g*Np with g = 1 (region A) and g = 2 (region B),
    # i.e. alpha = (2*g)^{-1/2} pairs with |W|^2 = Np/(2*alpha^2).
    vecs = np.array([m["vec"].ravel() for m in modes])
    norms2 = (vecs**2).sum(axis=1)
    for mode, n2 in zip(modes, norms2, strict=True):
        expected = Np / (2 * mode["alpha"] ** 2)
        assert abs(n2 - expected) < 0.05

    # All real supercell modes are mutually orthogonal.
    gram = vecs @ vecs.T
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.abs(off_diagonal).max() < 2e-2

    # Region-A Bloch factors are exactly +-1: per-atom displacement norms
    # are constant within each set of primitive-cell images.
    s2p = np.array(modes_obj.s2p_index)
    is_region_A = [abs(m["alpha"] - 1 / math.sqrt(2.0)) < 1e-6 for m in modes]
    for mode, in_A in zip(modes, is_region_A, strict=True):
        if not in_A:
            continue
        row_norms = np.linalg.norm(mode["vec"], axis=1)
        for p in range(n_prim):
            group = row_norms[s2p == p]
            assert group.max() - group.min() < 1e-4

    # Gamma is the first commensurate q-point (region A) and its acoustic
    # modes are excluded, so the first three modes are the Gamma optical
    # modes: sum_kappa sqrt(m_kappa) u_kappa = 0, identical in every cell.
    sqrt_m = np.sqrt(phonon.primitive.masses)  # [Na, Cl]
    for mode in modes[:3]:
        com = (mode["vec"] * sqrt_m[s2p][:, None]).sum(axis=0)
        assert np.abs(com).max() < 1e-2
        for p in range(n_prim):
            rows = mode["vec"][s2p == p]
            assert np.abs(rows - rows[0]).max() < 1e-4

    # Frequency unit: angular frequency in atomic units (the 2*pi is
    # supplied by h = 2*pi*hbar in the THz -> Hartree conversion).
    phonon.run_qpoints([[0, 0, 0]])
    freqs_thz = phonon.qpoints.frequencies[0]
    assert abs(modes[2]["frequency"] - freqs_thz[5] * modes_obj.THz_to_Ha) < 1e-7


def test_octopus_phonon_modes_variance_sum(ph_nacl_nonac, tmp_path):
    """T=0 variance consistency (interface notes, consistency check 1).

    The mode-sum of the sampled displacement variance computed from the file
    contents, sum_m alpha_m^2 * W_{l kappa alpha; m}^2 / omega_m (with
    sigma~^2 = 1/2 and l~^2 = 2/omega), must equal the analytic harmonic
    result sum_{q nu} |W_kappa_alpha(q nu)|^2 / (2 omega_{q nu}) summed over
    all commensurate q-points, per atom and Cartesian component (acoustic
    Gamma modes excluded on both sides).
    """
    phonon = ph_nacl_nonac
    modes_obj = OctopusPhononModes(phonon)
    filename = tmp_path / "phonon_modes.txt"
    modes_obj.write_phonon_file(str(filename))
    _, _, _, _, modes = _parse_phonon_modes_file(filename)

    # File side: sum over effective real modes.
    file_var = np.zeros((modes_obj.num_atoms_super, 3))
    for m in modes:
        file_var += m["alpha"] ** 2 * m["vec"] ** 2 / m["frequency"]

    # Analytic side: sum over all commensurate (q, nu), same units (Ha).
    s2p = np.array(modes_obj.s2p_index)
    analytic = np.zeros((modes_obj.num_atoms_super, 3))
    for iq in range(modes_obj.nq):
        for nu in range(modes_obj.num_eigenvectors):
            freq = modes_obj.frequencies[iq, nu] * modes_obj.THz_to_Ha
            if abs(freq) < 1e-8:  # excluded acoustic Gamma modes
                continue
            w2 = np.abs(modes_obj.eigenvectors[iq][:, nu].reshape(-1, 3)) ** 2
            analytic += w2[s2p] / (2 * freq)

    np.testing.assert_allclose(file_var, analytic, rtol=1e-3, atol=1e-6)


def test_octopus_phonon_modes_gauge_invariance(ph_nacl_nonac, tmp_path):
    """Gauge invariance (interface notes, consistency check 4).

    Multiplying the phonopy eigenvectors by random phases and mixing
    degenerate subspaces by random unitaries must leave every physical
    content of the file unchanged: alpha factors, frequencies, mode norms,
    the T=0 variance mode-sum, and the subspace spanned by the modes of each
    degenerate frequency group.
    """
    phonon = ph_nacl_nonac

    ref_obj = OctopusPhononModes(phonon)
    ref_obj.write_phonon_file(str(tmp_path / "ref.txt"))

    gauge_obj = OctopusPhononModes(phonon)
    gauge_obj._run()
    # detach from the shared phonon.qpoints arrays before perturbing
    gauge_obj.eigenvectors = np.array(gauge_obj.eigenvectors)
    gauge_obj.frequencies = np.array(gauge_obj.frequencies)
    rng = np.random.default_rng(42)
    for iq in range(gauge_obj.nq):
        for group in gauge_obj._degenerate_groups(iq):
            d = len(group)
            # random unitary (a random phase for d = 1)
            mat = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            unitary, _ = np.linalg.qr(mat)
            gauge_obj.eigenvectors[iq][:, group] = (
                gauge_obj.eigenvectors[iq][:, group] @ unitary
            )
    gauge_obj.write_phonon_file(str(tmp_path / "gauge.txt"))

    _, _, _, _, ref_modes = _parse_phonon_modes_file(tmp_path / "ref.txt")
    _, _, _, _, gauge_modes = _parse_phonon_modes_file(tmp_path / "gauge.txt")

    assert len(ref_modes) == len(gauge_modes)
    for r, g in zip(ref_modes, gauge_modes, strict=True):
        assert r["frequency"] == pytest.approx(g["frequency"], abs=1e-10)
        assert r["alpha"] == pytest.approx(g["alpha"], abs=1e-12)
        assert np.linalg.norm(r["vec"]) == pytest.approx(
            np.linalg.norm(g["vec"]), abs=1e-3
        )

    # T=0 variance mode-sum is basis-invariant within degenerate subspaces.
    def variance_sum(modes):
        var = np.zeros_like(modes[0]["vec"])
        for m in modes:
            var += m["alpha"] ** 2 * m["vec"] ** 2 / m["frequency"]
        return var

    np.testing.assert_allclose(
        variance_sum(ref_modes), variance_sum(gauge_modes), rtol=1e-3, atol=1e-6
    )

    # The span of the modes of each degenerate frequency group is invariant:
    # compare the orthogonal projectors built from the normalized modes.
    def projectors(modes):
        groups = {}
        for m in modes:
            groups.setdefault(round(m["frequency"], 6), []).append(
                m["vec"].ravel() / np.linalg.norm(m["vec"])
            )
        return {f: sum(np.outer(v, v) for v in vecs) for f, vecs in groups.items()}

    p_ref = projectors(ref_modes)
    p_gauge = projectors(gauge_modes)
    assert p_ref.keys() == p_gauge.keys()
    for f in p_ref:
        np.testing.assert_allclose(p_ref[f], p_gauge[f], atol=1e-4)


def test_octopus_phonon_modes_vs_phonopy_sampler(ph_nacl_nonac, tmp_path):
    """Cross-check vs phonopy's sampler (interface notes, consistency check 3).

    phonopy's RandomDisplacements implements the displacement half of the
    same sampling scheme independently (phonopy paper eqs. (118)-(122)). The
    per-atom displacement variances predicted from the phonon file contents
    (as sampled by Octopus, sigma~^2 = coth(omega/2kT)/2) must agree with the
    empirical variances of RandomDisplacements snapshots for the same force
    constants and temperature.
    """
    from phonopy.phonon.random_displacements import RandomDisplacements

    phonon = ph_nacl_nonac
    temperature = 300.0
    nsnapshots = 20000

    modes_obj = OctopusPhononModes(phonon)
    filename = tmp_path / "phonon_modes.txt"
    modes_obj.write_phonon_file(str(filename))
    _, _, np_header, file_masses, modes = _parse_phonon_modes_file(filename)

    # File-side prediction of <u^2> per atom and component, in Angstrom^2:
    # u = xi * alpha * sqrt(2/(amu_au*omega)) * W / sqrt(m_amu * Np), with
    # <xi^2> = coth(omega/(2 kB T))/2 and everything in atomic units.
    units = get_physical_units()
    kbt_ha = units.KB * temperature / units.Hartree  # kB T in Hartree
    amu_au = 1.0 / 5.485799110e-4
    var_au = np.zeros((len(file_masses), 3))
    for m in modes:
        sigma2 = 0.5 / np.tanh(m["frequency"] / (2.0 * kbt_ha))
        var_au += (
            sigma2
            * 2.0
            * m["alpha"] ** 2
            * m["vec"] ** 2
            / (amu_au * m["frequency"] * file_masses[:, None] * np_header)
        )
    var_file = var_au * units.Bohr**2  # -> Angstrom^2

    # Independent sampler: phonopy's RandomDisplacements (u in Angstrom).
    # The frequency conversion factor must match the one the file writer
    # inherited from the phonon object (the NaCl fixture carries VASP-unit
    # force constants; an octopus-calculator dataset carries its own factor).
    rd = RandomDisplacements(
        phonon.supercell,
        phonon.primitive,
        phonon.force_constants,
        factor=phonon.unit_conversion_factor,
    )
    rd.run(temperature, number_of_snapshots=nsnapshots, random_seed=12345)
    var_rd = (rd.u**2).mean(axis=0)

    # 5-sigma statistical tolerance on each variance estimate, plus a tight
    # check on the global mean.
    se = np.sqrt(2.0 / nsnapshots)
    np.testing.assert_allclose(var_rd, var_file, rtol=5 * se)
    assert var_rd.mean() == pytest.approx(var_file.mean(), rel=0.01)


def test_octopus_phonon_modes_nac_q_direction(ph_nacl, tmp_path):
    """NAC q-direction follows the phonopy convention.

    Default (None): no direction-dependent non-analytic term at Gamma, so the
    optical modes of NaCl stay threefold degenerate (analytic q = 0 limit).
    An explicit direction imposes the LO/TO splitting along that approach.
    """
    modes_obj = OctopusPhononModes(ph_nacl)
    modes_obj.write_phonon_file(str(tmp_path / "none.txt"))
    _, _, _, _, modes_none = _parse_phonon_modes_file(tmp_path / "none.txt")
    f_none = [m["frequency"] for m in modes_none[:3]]  # Gamma optical triplet
    assert max(f_none) - min(f_none) < 1e-6

    modes_obj = OctopusPhononModes(ph_nacl, nac_q_direction=[1, 0, 0])
    modes_obj.write_phonon_file(str(tmp_path / "split.txt"))
    _, _, _, _, modes_split = _parse_phonon_modes_file(tmp_path / "split.txt")
    f_split = sorted(m["frequency"] for m in modes_split[:3])
    assert f_split[2] > 1.2 * f_split[0]  # LO pushed above the TO doublet
    assert f_split[0] == pytest.approx(f_none[0], abs=1e-6)  # TO unchanged
