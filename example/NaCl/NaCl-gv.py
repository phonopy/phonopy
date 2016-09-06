import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
import matplotlib.pyplot as plt

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
phonon.set_group_velocity()
phonon.set_mesh([31, 31, 31])
qpoints, weights, frequencies, _ = phonon.get_mesh()
group_velocity = phonon.get_group_velocity()
gv_norm = np.sqrt((group_velocity ** 2).sum(axis=2))
for i, (f, g) in enumerate(zip(frequencies.T, gv_norm.T)):
    plt.plot(f, g, 'o', label=('band%d' % (i + 1)))
plt.legend()
plt.xlabel("Frequency (THz)")
plt.ylabel("|group-velocity| (THz.A)")
plt.show()
