import os
import pytest
import phonopy

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def ph_nacl():
    yaml_filename = os.path.join(current_dir, "phonopy_disp_NaCl.yaml")
    force_sets_filename = os.path.join(current_dir, "FORCE_SETS_NaCl")
    born_filename = os.path.join(current_dir, "BORN_NaCl")
    return phonopy.load(yaml_filename,
                        force_sets_filename=force_sets_filename,
                        born_filename=born_filename,
                        log_level=1, produce_fc=True)
