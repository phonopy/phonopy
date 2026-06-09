"""Tests for derived-state invalidation in the Phonopy class.

Regression tests for the Lane A2 invalidation policy (see
V5_MIGRATION.md): every input mutation clears the derived state that
depends on it, the eager dynamical-matrix rebuild on fc / nac / masses
change is preserved, and group velocity is no longer auto-rebuilt.

"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy

cwd = pathlib.Path(__file__).parent

# Derived-state properties populated by _run_analyses. All other
# analyses (_pdos, _moment, ...) are cleared by the same code path in
# Phonopy._invalidate_derived, so this representative subset is enough.
ANALYSIS_PROPERTIES = (
    "group_velocity",
    "band_structure",
    "mesh",
    "total_dos",
    "thermal_properties",
    "qpoints",
)


@pytest.fixture
def ph() -> Phonopy:
    """Return a fresh NaCl Phonopy instance with fc and nac_params set."""
    return phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=False,
        log_level=0,
        produce_fc=True,
    )


def _run_analyses(ph: Phonopy) -> None:
    """Populate a representative set of derived results."""
    ph.run_mesh([2, 2, 2], with_group_velocities=True)
    ph.run_band_structure([[[0, 0, 0], [0.25, 0, 0.25], [0.5, 0, 0.5]]])
    ph.run_total_dos()
    ph.run_thermal_properties(t_min=0, t_max=100, t_step=50)
    ph.run_qpoints([[0.1, 0.1, 0.1]])
    for name in ANALYSIS_PROPERTIES:
        assert getattr(ph, name) is not None


def _assert_analyses_cleared(ph: Phonopy) -> None:
    for name in ANALYSIS_PROPERTIES:
        assert getattr(ph, name) is None


def test_force_constants_setter(ph: Phonopy) -> None:
    """Setting fc clears all DM-derived state and eagerly rebuilds DM."""
    _run_analyses(ph)
    assert ph.force_constants is not None
    ph.force_constants = ph.force_constants.copy()
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None  # eager rebuild preserved


def test_force_constants_setter_none(ph: Phonopy) -> None:
    """Setting fc to None clears DM and all DM-derived state."""
    _run_analyses(ph)
    ph.force_constants = None
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is None


def test_nac_params_setter(ph: Phonopy) -> None:
    """Setting nac_params clears all DM-derived state and rebuilds DM."""
    _run_analyses(ph)
    ph.nac_params = copy.deepcopy(ph.nac_params)
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None


def test_masses_setter(ph: Phonopy) -> None:
    """Setting masses clears all DM-derived state and rebuilds DM."""
    _run_analyses(ph)
    ph.masses = ph.masses * 1.01
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None


def test_set_force_constants_zero_with_radius(ph: Phonopy) -> None:
    """In-place fc mutation clears all DM-derived state and rebuilds DM."""
    _run_analyses(ph)
    ph.set_force_constants_zero_with_radius(3.0)
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None


def test_produce_force_constants(ph: Phonopy) -> None:
    """Recomputing fc clears all DM-derived state and rebuilds DM."""
    _run_analyses(ph)
    ph.produce_force_constants(show_drift=False)
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None


@pytest.mark.parametrize("use_symfc_projector", [False, True])
def test_symmetrize_force_constants(ph: Phonopy, use_symfc_projector: bool) -> None:
    """Symmetrizing fc clears all DM-derived state and rebuilds DM."""
    if use_symfc_projector:
        pytest.importorskip("symfc")
    _run_analyses(ph)
    ph.symmetrize_force_constants(
        show_drift=False, use_symfc_projector=use_symfc_projector
    )
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None


def test_symmetrize_force_constants_by_space_group(ph: Phonopy) -> None:
    """Symmetrizing fc by space group clears DM-derived state, rebuilds DM."""
    _run_analyses(ph)
    ph.symmetrize_force_constants_by_space_group(show_drift=False)
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is not None


def test_unit_conversion_factor_setter(ph: Phonopy) -> None:
    """Setting the factor clears analyses but preserves the DM instance."""
    _run_analyses(ph)
    dm = ph.dynamical_matrix
    assert dm is not None
    ph.unit_conversion_factor = ph.unit_conversion_factor * 2
    _assert_analyses_cleared(ph)
    assert ph.dynamical_matrix is dm  # DM does not depend on the factor


def test_dataset_setter(ph: Phonopy) -> None:
    """Replacing the dataset clears fc, DM, and all derived state."""
    _run_analyses(ph)
    ph.dataset = copy.deepcopy(ph.dataset)
    _assert_analyses_cleared(ph)
    assert ph.force_constants is None
    assert ph.dynamical_matrix is None


def test_displacements_setter(ph: Phonopy) -> None:
    """Replacing displacements clears fc, DM, and all derived state."""
    fc = ph.force_constants
    assert fc is not None
    ph.dataset = None  # displacements setter requires a type-2 dataset
    ph.force_constants = fc  # restore fc; DM rebuilt eagerly
    _run_analyses(ph)
    ph.displacements = np.zeros((1, len(ph.supercell), 3))
    _assert_analyses_cleared(ph)
    assert ph.force_constants is None
    assert ph.dynamical_matrix is None


def test_forces_setter(ph: Phonopy) -> None:
    """Replacing forces clears fc, DM, and all derived state."""
    _run_analyses(ph)
    ph.forces = ph.forces.copy()
    _assert_analyses_cleared(ph)
    assert ph.force_constants is None
    assert ph.dynamical_matrix is None


def test_supercell_energies_setter(ph: Phonopy) -> None:
    """Replacing supercell energies clears fc, DM, and all derived state."""
    _run_analyses(ph)
    ph.supercell_energies = np.zeros(len(ph.displacements))
    _assert_analyses_cleared(ph)
    assert ph.force_constants is None
    assert ph.dynamical_matrix is None


def test_group_velocity_not_auto_rebuilt(ph: Phonopy) -> None:
    """GV is cleared like any derived state, never rebuilt eagerly (Q2)."""
    _run_analyses(ph)
    assert ph.group_velocity is not None
    assert ph.force_constants is not None
    ph.force_constants = ph.force_constants.copy()
    assert ph.dynamical_matrix is not None  # DM eagerly rebuilt
    assert ph.group_velocity is None  # GV is not

    # GV is rebuilt on demand by the next analysis run.
    ph.run_mesh([2, 2, 2], with_group_velocities=True)
    assert ph.group_velocity is not None


def test_derived_state_cleared_on_fc_change(ph: Phonopy) -> None:
    """Derived results are set to None after a force_constants change (Q4)."""
    _run_analyses(ph)
    assert ph.force_constants is not None
    ph.force_constants = ph.force_constants.copy()
    assert ph.band_structure is None
    assert ph.mesh is None
    assert ph.thermal_properties is None
