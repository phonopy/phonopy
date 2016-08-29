from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
import numpy as np

def get_frequency(poscar_filename, force_sets_filename):
    unitcell = read_vasp(poscar_filename)
    volume = unitcell.get_volume()
    phonon = Phonopy(unitcell,
                     [[2, 0, 0],
                      [0, 2, 0],
                      [0, 0, 2]],
                     primitive_matrix=[[0, 0.5, 0.5],
                                       [0.5, 0, 0.5],
                                       [0.5, 0.5, 0]])
    force_sets = parse_FORCE_SETS(filename=force_sets_filename)
    phonon.set_displacement_dataset(force_sets)
    phonon.produce_force_constants()
    return phonon.get_frequencies([0.5, 0.5, 0]), volume

frequencies = []
volumes = []
for i in range(-10, 6):
    poscar_filename = "POSCAR-%d" % i
    force_sets_filename = "FORCE_SETS-%d" % i
    fs, v = get_frequency(poscar_filename, force_sets_filename)
    frequencies.append(fs)
    volumes.append(v)

import matplotlib.pyplot as plt
for freq_at_X in np.array(frequencies).T:
    freq_squared = freq_at_X ** 2 * np.sign(freq_at_X)
    # np.sign is used to treat imaginary mode, since
    # imaginary frequency is returned as a negative value from phonopy.
    plt.plot(volumes, freq_squared, "o-")
    # for v, f2 in zip(volumes, freq_squared):
    #      print("%f %f" % (v, f2))
    # print('')
    # print('')
plt.title("Frequeny squared (THz^2) at X-point vs volume (A^3)")
plt.show()
