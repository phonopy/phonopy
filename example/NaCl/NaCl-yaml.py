from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.phonopy_yaml import PhonopyYaml

bulk = read_vasp("POSCAR")
phonon = Phonopy(bulk,
                 [[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]],
                 primitive_matrix=[[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]],
                 is_auto_displacements=False)
force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()

born = [[[1.08703, 0, 0],
         [0, 1.08703, 0],
         [0, 0, 1.08703]],
        [[-1.08672, 0, 0],
         [0, -1.08672, 0],
         [0, 0, -1.08672]]]
epsilon = [[2.43533967, 0, 0],
           [0, 2.43533967, 0],
           [0, 0, 2.43533967]]
factors = 14.400
phonon.set_nac_params({'born': born,
                       'factor': factors,
                       'dielectric': epsilon})

phpy_yaml = PhonopyYaml(calculator='vasp',
                        show_force_constants=True)
phpy_yaml.set(phonon)
print(phpy_yaml)
