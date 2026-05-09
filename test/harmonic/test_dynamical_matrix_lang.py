"""Parity tests comparing Rust and C dispatch paths in DynamicalMatrix.

Covers the dipole-dipole and Wang-NAC charge-sum dispatch sites:

* ``DynamicalMatrixGL._run_c_recip_dipole_dipole_q0``
* ``DynamicalMatrixGL._get_c_recip_dipole_dipole``
* ``DynamicalMatrixWang._get_charge_sum``

The whole module is skipped when phonors is not importable.

"""

from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import phonopy
from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL, DynamicalMatrixWang

pytest.importorskip("phonors")
pytest.importorskip("phonopy._phonopy")

cwd = pathlib.Path(__file__).parent.parent


def _load_nacl(lang: Literal["C", "Rust"]) -> Phonopy:
    return phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=False,
        log_level=0,
        produce_fc=True,
        lang=lang,
    )


def _load_gl(lang: Literal["C", "Rust"]) -> tuple[Phonopy, DynamicalMatrixGL]:
    ph = _load_nacl(lang)
    dm = ph.dynamical_matrix
    assert isinstance(dm, DynamicalMatrixGL)
    return ph, dm


def _load_wang(lang: Literal["C", "Rust"]) -> tuple[Phonopy, DynamicalMatrixWang]:
    ph = _load_nacl(lang)
    nac = ph.nac_params
    assert nac is not None
    nac["method"] = "wang"
    ph.nac_params = nac
    dm = ph.dynamical_matrix
    assert isinstance(dm, DynamicalMatrixWang)
    return ph, dm


@pytest.fixture(scope="module")
def gl_c() -> tuple[Phonopy, DynamicalMatrixGL]:
    """NaCl Phonopy with Gonze-Lee NAC and lang='C'."""
    return _load_gl("C")


@pytest.fixture(scope="module")
def gl_rust() -> tuple[Phonopy, DynamicalMatrixGL]:
    """NaCl Phonopy with Gonze-Lee NAC and lang='Rust'."""
    return _load_gl("Rust")


@pytest.fixture(scope="module")
def wang_c() -> tuple[Phonopy, DynamicalMatrixWang]:
    """NaCl Phonopy with Wang NAC and lang='C'."""
    return _load_wang("C")


@pytest.fixture(scope="module")
def wang_rust() -> tuple[Phonopy, DynamicalMatrixWang]:
    """NaCl Phonopy with Wang NAC and lang='Rust'."""
    return _load_wang("Rust")


def test_recip_dipole_dipole_q0_rust_matches_c(
    gl_c: tuple[Phonopy, DynamicalMatrixGL],
    gl_rust: tuple[Phonopy, DynamicalMatrixGL],
) -> None:
    """phonors.recip_dipole_dipole_q0 must agree with the C kernel."""
    _, dm_c = gl_c
    _, dm_r = gl_rust
    dm_c.make_Gonze_nac_dataset()
    dm_r.make_Gonze_nac_dataset()
    _, dd_q0_c, *_ = dm_c.Gonze_nac_dataset
    _, dd_q0_r, *_ = dm_r.Gonze_nac_dataset
    np.testing.assert_allclose(dd_q0_c, dd_q0_r, atol=1e-13)


@pytest.mark.parametrize(
    "q_red",
    [
        (0.1, 0.2, 0.3),
        (0.5, 0.5, 0.0),
        (0.25, 0.25, 0.25),
        (0.0, 0.5, 0.0),
    ],
)
@pytest.mark.parametrize("with_q_dir", [False, True])
def test_recip_dipole_dipole_rust_matches_c(
    gl_c: tuple[Phonopy, DynamicalMatrixGL],
    gl_rust: tuple[Phonopy, DynamicalMatrixGL],
    q_red: tuple[float, float, float],
    with_q_dir: bool,
) -> None:
    """phonors.recip_dipole_dipole must agree with the C kernel.

    Both ``q_dir_cart=None`` and an explicit direction array are
    exercised because the two Rust paths differ
    (``Option<arr>`` matching at the FFI boundary).

    """
    _, dm_c = gl_c
    _, dm_r = gl_rust
    dm_c.make_Gonze_nac_dataset()
    dm_r.make_Gonze_nac_dataset()
    rec_lat = np.linalg.inv(dm_c.primitive.cell)
    q_cart = np.array(np.dot(q_red, rec_lat.T), dtype="double")
    q_dir_cart: NDArray[np.double] | None = (
        np.array([1.0, 0.0, 0.0]) if with_q_dir else None
    )
    dd_c = dm_c._get_c_recip_dipole_dipole(q_cart, q_dir_cart)
    dd_r = dm_r._get_c_recip_dipole_dipole(q_cart, q_dir_cart)
    np.testing.assert_allclose(dd_c, dd_r, atol=1e-13)


def test_gonze_force_constants_rust_matches_c(
    gl_c: tuple[Phonopy, DynamicalMatrixGL],
    gl_rust: tuple[Phonopy, DynamicalMatrixGL],
) -> None:
    """The whole Gonze-Lee dataset (built across commensurate q) must agree."""
    _, dm_c = gl_c
    _, dm_r = gl_rust
    dm_c.make_Gonze_nac_dataset()
    dm_r.make_Gonze_nac_dataset()
    np.testing.assert_allclose(
        dm_c.short_range_force_constants,
        dm_r.short_range_force_constants,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "q_red",
    [(0.1, 0.2, 0.3), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0)],
)
def test_dynmat_at_q_rust_matches_c(
    gl_c: tuple[Phonopy, DynamicalMatrixGL],
    gl_rust: tuple[Phonopy, DynamicalMatrixGL],
    q_red: tuple[float, float, float],
) -> None:
    """End-to-end Gonze NAC dynamical matrix at q must agree."""
    ph_c, _ = gl_c
    ph_r, _ = gl_rust
    dm_c = ph_c.get_dynamical_matrix_at_q(q_red)
    dm_r = ph_r.get_dynamical_matrix_at_q(q_red)
    np.testing.assert_allclose(dm_c, dm_r, atol=1e-13)


def test_dynmat_gamma_with_q_direction_rust_matches_c(
    gl_c: tuple[Phonopy, DynamicalMatrixGL],
    gl_rust: tuple[Phonopy, DynamicalMatrixGL],
) -> None:
    """Gamma point with q_direction reaches the q_dir_cart != None branch."""
    _, dm_c = gl_c
    _, dm_r = gl_rust
    q_dir = [1.0, 0.0, 0.0]
    dm_c.run([0, 0, 0], q_direction=q_dir)
    dm_r.run([0, 0, 0], q_direction=q_dir)
    np.testing.assert_allclose(dm_c.dynamical_matrix, dm_r.dynamical_matrix, atol=1e-13)


def test_with_full_terms_rust_matches_c(
    gl_c: tuple[Phonopy, DynamicalMatrixGL],
) -> None:
    """The seldom-used with_full_terms=True branch must also agree."""
    _, dm_base = gl_c
    dm_c = DynamicalMatrixGL(
        dm_base.supercell,
        dm_base.primitive,
        dm_base.force_constants,
        nac_params=dm_base.nac_params,
        with_full_terms=True,
        lang="C",
    )
    dm_r = DynamicalMatrixGL(
        dm_base.supercell,
        dm_base.primitive,
        dm_base.force_constants,
        nac_params=dm_base.nac_params,
        with_full_terms=True,
        lang="Rust",
    )
    dm_c.run([0.25, 0.25, 0.25])
    dm_r.run([0.25, 0.25, 0.25])
    np.testing.assert_allclose(dm_c.dynamical_matrix, dm_r.dynamical_matrix, atol=1e-13)


@pytest.mark.parametrize(
    "q_cart",
    [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ],
)
def test_wang_charge_sum_rust_matches_c(
    wang_c: tuple[Phonopy, DynamicalMatrixWang],
    wang_rust: tuple[Phonopy, DynamicalMatrixWang],
    q_cart: NDArray[np.double],
) -> None:
    """phonors.charge_sum must agree with the Python loop element-wise."""
    _, dm_c = wang_c
    _, dm_r = wang_rust
    born = dm_c.born
    num_atom = len(born)
    cs_c = dm_c._get_charge_sum(num_atom, q_cart, born)
    cs_r = dm_r._get_charge_sum(num_atom, q_cart, born)
    np.testing.assert_array_equal(cs_c, cs_r)


@pytest.mark.parametrize(
    "q_red",
    [(0.1, 0.2, 0.3), (0.5, 0.5, 0.0)],
)
def test_wang_dynmat_at_q_rust_matches_c(
    wang_c: tuple[Phonopy, DynamicalMatrixWang],
    wang_rust: tuple[Phonopy, DynamicalMatrixWang],
    q_red: tuple[float, float, float],
) -> None:
    """End-to-end Wang NAC dynamical matrix at q must agree."""
    ph_c, _ = wang_c
    ph_r, _ = wang_rust
    dm_c = ph_c.get_dynamical_matrix_at_q(q_red)
    dm_r = ph_r.get_dynamical_matrix_at_q(q_red)
    np.testing.assert_allclose(dm_c, dm_r, atol=1e-13)


@pytest.mark.parametrize(
    "q_red",
    [(0.1, 0.2, 0.3), (0.5, 0.5, 0.0), (0.25, 0.25, 0.25)],
)
def test_wang_py_compute_dynamical_matrix_matches_c(
    wang_c: tuple[Phonopy, DynamicalMatrixWang],
    q_red: tuple[float, float, float],
) -> None:
    """Pure-Python Wang reference path must agree with the dispatched kernel.

    ``DynamicalMatrixWang._run_py_compute_dynamical_matrix`` is kept as a
    readable reference for the C / Rust kernels.  This locks it against
    drift.

    """
    _, dm = wang_c
    dm.run(q_red)
    expected = np.array(dm.dynamical_matrix, copy=True)
    dm._run_py_compute_dynamical_matrix(np.array(q_red, dtype="double"), None)
    np.testing.assert_allclose(dm.dynamical_matrix, expected, atol=1e-13)
