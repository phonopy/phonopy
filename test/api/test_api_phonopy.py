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


def test_energies_setter_NaCl_type1(ph_nacl_fd: Phonopy):
    """Test Phonopy.energies setter and getter (type1 dataset)."""
    ph_in = ph_nacl_fd
    energies = ph_in.supercell_energies
    ref_supercell_energies = [-216.82820693, -216.82817843]
    np.testing.assert_allclose(energies, ref_supercell_energies)

    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    np.testing.assert_allclose(energies, ph.supercell_energies)
    ph.supercell_energies = ph_in.supercell_energies + 1
    np.testing.assert_allclose(ph.supercell_energies, ph_in.supercell_energies + 1)


def test_energies_setter_NaCl_type2(ph_nacl_rd: Phonopy):
    """Test Phonopy.energies setter and getter (type2 dataset)."""
    ph_in = ph_nacl_rd
    energies = ph_in.supercell_energies
    ref_supercell_energies = [
        -216.84472784,
        -216.84225045,
        -216.84473283,
        -216.84431749,
        -216.84276813,
        -216.84204234,
        -216.84150156,
        -216.84438529,
        -216.84339285,
        -216.84005085,
    ]
    np.testing.assert_allclose(energies, ref_supercell_energies)

    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    np.testing.assert_allclose(energies, ph.supercell_energies)
    ph.supercell_energies = ph_in.supercell_energies + 1
    np.testing.assert_allclose(ph.supercell_energies, ph_in.supercell_energies + 1)
