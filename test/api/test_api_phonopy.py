"""Tests of Phonopy API."""

from pathlib import Path

import numpy as np

from phonopy import Phonopy
from phonopy.structure.dataset import get_displacements_and_forces

cwd = Path(__file__).parent


def test_displacements_setter_NaCl(ph_nacl: Phonopy):
    """Test Phonopy.displacements setter and getter."""
    ph_in = ph_nacl
    displacements, _ = get_displacements_and_forces(ph_in.dataset)
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.displacements = displacements
    np.testing.assert_allclose(displacements, ph.displacements)


def test_forces_setter_NaCl_type1(ph_nacl: Phonopy):
    """Test Phonopy.forces setter and getter (type1 dataset)."""
    ph_in = ph_nacl
    forces = ph_in.forces
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    ph.forces = forces
    np.testing.assert_allclose(ph.forces, forces)


def test_forces_setter_NaCl_type2(ph_nacl: Phonopy):
    """Test Phonopy.forces setter and getter (type2 dataset)."""
    ph_in = ph_nacl
    displacements, forces = get_displacements_and_forces(ph_in.dataset)
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.displacements = displacements
    ph.forces = forces
    np.testing.assert_allclose(ph.forces, forces)


def test_energies_setter_NaCl_type1(ph_nacl: Phonopy):
    """Test Phonopy.energies setter and getter (type1 dataset)."""
    ph_in = ph_nacl
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    energies = [-216.82820693, -216.82817843]
    ph.energies = energies
    np.testing.assert_allclose(ph.energies, energies)


def test_energies_setter_NaCl_type2(ph_nacl: Phonopy):
    """Test Phonopy.energies setter and getter (type2 dataset)."""
    ph_in = ph_nacl
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.generate_displacements(number_of_snapshots=2)
    energies = [-216.82820693, -216.82817843]
    ph.energies = energies
    np.testing.assert_allclose(ph.energies, energies)
