"""Pytest configuration."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

cwd = pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def ph_si() -> Phonopy:
    """Return Phonopy class instance of Si-prim 2x2x2."""
    yaml_filename = cwd / "phonopy_params_Si.yaml"
    return phonopy.load(
        yaml_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    born_filename = cwd / "BORN_NaCl"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl_wang() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    born_filename = cwd / "BORN_NaCl"
    ph = phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )
    nac_params = ph.nac_params
    nac_params["method"] = "wang"
    ph.nac_params = nac_params
    return ph


@pytest.fixture(scope="session")
def ph_nacl_nofcsym() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 without symmetrizing fc2."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    born_filename = cwd / "BORN_NaCl"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        symmetrize_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl_compact_fcsym() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 with compact fc2."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    born_filename = cwd / "BORN_NaCl"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        is_compact_fc=True,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl_nonac() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 without NAC."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        is_nac=False,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl_nonac_compact_fc() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 without NAC with compact fc2."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        is_nac=False,
        is_compact_fc=True,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl_nonac_dense_svecs() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 without NAC with dense svecs."""
    yaml_filename = cwd / "phonopy_disp_NaCl.yaml"
    force_sets_filename = cwd / "FORCE_SETS_NaCl"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        is_nac=False,
        is_compact_fc=True,
        store_dense_svecs=True,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_nacl_rd() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 with RD results.

    This data contains supercell energies and NAC params.

    """
    yaml_filename = cwd / "phonopy_params_NaCl-rd.yaml.xz"
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False)


@pytest.fixture(scope="session")
def ph_nacl_rd_symfc() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 with RD results.

    This data contains supercell energies and NAC params. Symfc is used to
    compute force constants.

    """
    pytest.importorskip("symfc")
    pytest.importorskip("spglib", minversion="2.5")

    yaml_filename = cwd / "phonopy_params_NaCl-rd.yaml.xz"
    return phonopy.load(
        yaml_filename, log_level=1, fc_calculator="symfc", is_compact_fc=True
    )


@pytest.fixture(scope="session")
def ph_nacl_fd() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 with RD results.

    This data contains supercell energies and NAC params.

    """
    yaml_filename = cwd / "phonopy_params_NaCl-fd.yaml.xz"
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False)


@pytest.fixture(scope="session")
def ph_sno2() -> Phonopy:
    """Return Phonopy class instance of rutile SnO2 2x2x3."""
    yaml_filename = cwd / "phonopy_disp_SnO2.yaml"
    force_sets_filename = cwd / "FORCE_SETS_SnO2"
    born_filename = cwd / "BORN_SnO2"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_tio2() -> Phonopy:
    """Return Phonopy class instance of anataze TiO2 3x3x1."""
    yaml_filename = cwd / "phonopy_disp_TiO2.yaml"
    force_sets_filename = cwd / "FORCE_SETS_TiO2"
    born_filename = cwd / "BORN_TiO2"
    return phonopy.load(
        yaml_filename,
        force_sets_filename=force_sets_filename,
        born_filename=born_filename,
        is_compact_fc=False,
        log_level=1,
        produce_fc=True,
    )


@pytest.fixture(scope="session")
def ph_zr3n4() -> Phonopy:
    """Return Phonopy class instance of anataze Zr3N4 1x1x1."""
    yaml_filename = cwd / "phonopy_params_Zr3N4.yaml"
    return phonopy.load(
        yaml_filename, is_compact_fc=False, log_level=1, produce_fc=True
    )


@pytest.fixture(scope="session")
def ph_tipn3() -> Phonopy:
    """Return Phonopy class instance of anataze TiPN3 4x2x1."""
    yaml_filename = cwd / "phonopy_params_TiPN3.yaml.xz"
    return phonopy.load(
        yaml_filename, is_compact_fc=False, log_level=1, produce_fc=True
    )


@pytest.fixture(scope="session")
def ph_srtio3() -> Phonopy:
    """Return Phonopy class instance of anataze SrTiO3 3x3x3."""
    yaml_filename = cwd / "phonopy_SrTiO3.yaml.xz"
    return phonopy.load(
        yaml_filename, is_compact_fc=False, log_level=1, produce_fc=True
    )


@pytest.fixture(scope="session")
def ph_kcl() -> Phonopy:
    """Return Phonopy class instance of KCl for mlpsscha_KCl-120.pmlp."""
    yaml_filename = cwd / "phonopy_KCl.yaml"
    return phonopy.load(yaml_filename, produce_fc=False, log_level=1)


@pytest.fixture(scope="session")
def ph_nacl_gruneisen() -> tuple[Phonopy, Phonopy, Phonopy]:
    """Return Phonopy class instances of NaCl 2x2x2 at three volumes."""
    ph0 = phonopy.load(
        cwd / "phonopy_params_NaCl-1.00.yaml.xz",
        log_level=1,
        produce_fc=True,
    )
    ph_minus = phonopy.load(
        cwd / "phonopy_params_NaCl-0.995.yaml.xz",
        log_level=1,
        produce_fc=True,
    )
    ph_plus = phonopy.load(
        cwd / "phonopy_params_NaCl-1.005.yaml.xz",
        log_level=1,
        produce_fc=True,
    )
    return ph0, ph_minus, ph_plus


@pytest.fixture(scope="session")
def convcell_sio2() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of rutile SiO2."""
    symbols = ["Si"] * 2 + ["O"] * 4
    lattice = [[4.65, 0, 0], [0, 4.75, 0], [0, 0, 3.25]]
    points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.0],
        [0.7, 0.7, 0.0],
        [0.2, 0.8, 0.5],
        [0.8, 0.2, 0.5],
    ]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def convcell_al2o3() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of rutile Al2O3."""
    symbols = ["Al"] * 12 + ["O"] * 18
    lattice = [
        [4.7742263452989944, 0.0000000000000000, 0.0000000000000000],
        [-2.3871131726494972, 4.1346012984458653, 0.0000000000000000],
        [-0.0000000000000000, 0.0000000000000000, 13.0113593913272219],
    ]
    points = [
        [0.3333333333333357, 0.6666666666666643, 0.0187603745624614],
        [0.3333333333333357, 0.6666666666666643, 0.3145729587708743],
        [0.0000000000000000, 0.0000000000000000, 0.1479062921042101],
        [0.6666666666666643, 0.3333333333333357, 0.1854270412291256],
        [0.0000000000000000, 0.0000000000000000, 0.3520937078957900],
        [0.0000000000000000, 0.0000000000000000, 0.6479062921042100],
        [0.6666666666666643, 0.3333333333333357, 0.4812396254375386],
        [0.3333333333333357, 0.6666666666666643, 0.5187603745624614],
        [0.6666666666666643, 0.3333333333333357, 0.6854270412291257],
        [0.6666666666666643, 0.3333333333333357, 0.9812396254375386],
        [0.3333333333333357, 0.6666666666666643, 0.8145729587708743],
        [0.0000000000000000, 0.0000000000000000, 0.8520937078957900],
        [0.3058086537880128, 0.0000000000000000, 0.2500000000000000],
        [0.3608580128786515, 0.3333333333333357, 0.0833333333333357],
        [0.0000000000000000, 0.3058086537880128, 0.2500000000000000],
        [0.6666666666666643, 0.0275246795453229, 0.0833333333333357],
        [0.6941913462119872, 0.6941913462119872, 0.2500000000000000],
        [0.9724753204546771, 0.6391419871213485, 0.0833333333333357],
        [0.9724753204546771, 0.3333333333333357, 0.5833333333333357],
        [0.0275246795453229, 0.6666666666666643, 0.4166666666666643],
        [0.6666666666666643, 0.6391419871213485, 0.5833333333333357],
        [0.3333333333333357, 0.3608580128786515, 0.4166666666666643],
        [0.3608580128786515, 0.0275246795453229, 0.5833333333333357],
        [0.6391419871213485, 0.9724753204546771, 0.4166666666666643],
        [0.6391419871213485, 0.6666666666666643, 0.9166666666666643],
        [0.6941913462119872, 0.0000000000000000, 0.7500000000000000],
        [0.3333333333333357, 0.9724753204546771, 0.9166666666666643],
        [0.0000000000000000, 0.6941913462119872, 0.7500000000000000],
        [0.0275246795453229, 0.3608580128786515, 0.9166666666666643],
        [0.3058086537880128, 0.3058086537880128, 0.7500000000000000],
    ]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def primcell_si() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of primitive cell of Si."""
    symbols = ["Si"] * 2
    lattice = [[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]]
    points = [[0.75, 0.75, 0.75], [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def nacl_unitcell_order1() -> PhonopyAtoms:
    """Return Phonopy class instance of only NaCl unitcell with order-1."""
    yaml_filename = cwd / "phonopy_NaCl_unitcell1.yaml"
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False).unitcell


@pytest.fixture(scope="session")
def nacl_unitcell_order2() -> PhonopyAtoms:
    """Return Phonopy class instance of only NaCl unitcell with order-2."""
    yaml_filename = cwd / "phonopy_NaCl_unitcell2.yaml"
    return phonopy.load(yaml_filename, log_level=1, produce_fc=False).unitcell


@pytest.fixture(scope="session")
def primcell_nacl() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of primitive cell of NaCl."""
    symbols = ["Na", "Cl"]
    x = 5.6903014761756712 / 2
    lattice = [[0, x, x], [x, 0, x], [x, x, 0]]
    points = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def convcell_cr() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of primitive cell of Cr."""
    symbols = ["Cr"] * 2
    a = 2.812696943681890
    lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def helper_methods() -> Callable:
    """Return methods to compare cells."""

    class HelperMethods:
        @classmethod
        def compare_cells_with_order(
            cls, cell: PhonopyAtoms, cell_ref: PhonopyAtoms, symprec=1e-5
        ) -> None:
            """Compare two cells with the same orders of positions."""
            np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=symprec)
            cls.compare_positions_with_order(
                cell.scaled_positions, cell_ref.scaled_positions, cell.cell
            )
            np.testing.assert_array_equal(cell.numbers, cell_ref.numbers)
            np.testing.assert_allclose(cell.masses, cell_ref.masses, atol=symprec)
            if cell.magnetic_moments is None:
                assert cell_ref.magnetic_moments is None
            else:
                np.testing.assert_allclose(
                    cell.magnetic_moments, cell_ref.magnetic_moments, atol=symprec
                )

        @classmethod
        def compare_positions_with_order(
            cls, pos, pos_ref, lattice, symprec=1e-5
        ) -> None:
            """Compare two lists of positions and orders.

            lattice :
                Basis vectors in row vectors.

            """
            diff = pos - pos_ref
            diff -= np.rint(diff)
            dist = (np.dot(diff, lattice) ** 2).sum(axis=1)
            assert (dist < symprec).all()

        @classmethod
        def compare_cells(
            cls, cell: PhonopyAtoms, cell_ref: PhonopyAtoms, symprec=1e-5
        ) -> None:
            """Compare two cells where position orders can be different."""
            np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=symprec)

            indices = cls.compare_positions_in_arbitrary_order(
                cell.scaled_positions, cell_ref.scaled_positions, cell.cell
            )
            np.testing.assert_array_equal(cell.numbers, cell_ref.numbers[indices])
            np.testing.assert_allclose(
                cell.masses, cell_ref.masses[indices], atol=symprec
            )
            if cell.magnetic_moments is None:
                assert cell_ref.magnetic_moments is None
            else:
                np.testing.assert_allclose(
                    cell.magnetic_moments,
                    cell_ref.magnetic_moments[indices],
                    atol=symprec,
                )

        @classmethod
        def compare_positions_in_arbitrary_order(
            cls, pos_in, pos_ref, lattice, symprec=1e-5
        ) -> list:
            """Compare two sets of positions irrespective of orders.

            lattice :
                Basis vectors in row vectors.

            """
            indices = []
            for pos in pos_in:
                diff = pos_ref - pos
                diff -= np.rint(diff)
                dist = (np.dot(diff, lattice) ** 2).sum(axis=1)
                matches = np.where(dist < symprec)[0]
                assert len(matches) == 1
                indices.append(matches[0])
            return indices

    return HelperMethods
