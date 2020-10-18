import os
import pytest
import phonopy
from phonopy.structure.atoms import PhonopyAtoms

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def ph_nacl():
    yaml_filename = os.path.join(current_dir, "phonopy_disp_NaCl.yaml")
    force_sets_filename = os.path.join(current_dir, "FORCE_SETS_NaCl")
    born_filename = os.path.join(current_dir, "BORN_NaCl")
    return phonopy.load(yaml_filename,
                        force_sets_filename=force_sets_filename,
                        born_filename=born_filename,
                        is_compact_fc=False,
                        log_level=1, produce_fc=True)


@pytest.fixture(scope='session')
def ph_nacl_nonac():
    yaml_filename = os.path.join(current_dir, "phonopy_disp_NaCl.yaml")
    force_sets_filename = os.path.join(current_dir, "FORCE_SETS_NaCl")
    return phonopy.load(yaml_filename,
                        force_sets_filename=force_sets_filename,
                        is_nac=False,
                        is_compact_fc=False,
                        log_level=1, produce_fc=True)


@pytest.fixture(scope='session')
def ph_sno2():
    yaml_filename = os.path.join(current_dir, "phonopy_disp_SnO2.yaml")
    force_sets_filename = os.path.join(current_dir, "FORCE_SETS_SnO2")
    born_filename = os.path.join(current_dir, "BORN_SnO2")
    return phonopy.load(yaml_filename,
                        force_sets_filename=force_sets_filename,
                        born_filename=born_filename,
                        is_compact_fc=False,
                        log_level=1, produce_fc=True)


@pytest.fixture(scope='session')
def convcell_sio2():
    symbols = ['Si'] * 2 + ['O'] * 4
    lattice = [[4.65, 0, 0],
               [0, 4.75, 0],
               [0, 0, 3.25]]
    points = [[0.0, 0.0, 0.0],
              [0.5, 0.5, 0.5],
              [0.3, 0.3, 0.0],
              [0.7, 0.7, 0.0],
              [0.2, 0.8, 0.5],
              [0.8, 0.2, 0.5]]
    return PhonopyAtoms(cell=lattice,
                        scaled_positions=points,
                        symbols=symbols)


@pytest.fixture(scope='session')
def primcell_si():
    symbols = ['Si'] * 2
    lattice = [[0, 2.73, 2.73],
               [2.73, 0, 2.73],
               [2.73, 2.73, 0]]
    points = [[0.75, 0.75, 0.75],
              [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice,
                        scaled_positions=points,
                        symbols=symbols)


@pytest.fixture(scope='session')
def convcell_nacl():
    symbols = ['Na'] * 4 + ['Cl'] * 4
    a = 5.6903014761756712
    lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    points = [[0.0, 0.0, 0.0],
              [0.0, 0.5, 0.5],
              [0.5, 0.0, 0.5],
              [0.5, 0.5, 0.0],
              [0.5, 0.5, 0.5],
              [0.5, 0.0, 0.0],
              [0.0, 0.5, 0.0],
              [0.0, 0.0, 0.5]]
    return PhonopyAtoms(cell=lattice,
                        scaled_positions=points,
                        symbols=symbols)


@pytest.fixture(scope='session')
def primcell_nacl():
    symbols = ['Na', 'Cl']
    x = 5.6903014761756712 / 2
    lattice = [[0, x, x], [x, 0, x], [x, x, 0]]
    points = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice,
                        scaled_positions=points,
                        symbols=symbols)


@pytest.fixture(scope='session')
def convcell_cr():
    symbols = ['Cr'] * 2
    a = 2.812696943681890
    lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    points = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice,
                        scaled_positions=points,
                        symbols=symbols)
