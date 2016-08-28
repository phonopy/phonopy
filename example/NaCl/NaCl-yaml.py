from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.interface.phonopy_yaml import PhonopyYaml

unitcell = read_vasp("POSCAR")
phonon = Phonopy(unitcell,
                 [[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]],
                 primitive_matrix=[[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]])
force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()
primitive = phonon.get_primitive()

nac_params = parse_BORN(primitive, filename="BORN")
phonon.set_nac_params(nac_params)

phpy_yaml = PhonopyYaml(calculator='vasp',
                        show_force_constants=True)
phpy_yaml.set_phonon_info(phonon)
print(phpy_yaml)
