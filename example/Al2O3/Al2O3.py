from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
import numpy as np

cell = read_vasp("POSCAR")

# Initialize phonon. Supercell matrix has to have the shape of (3, 3)
phonon = Phonopy(cell,
                 np.diag([2, 2, 1]),
                 primitive_matrix=[[2./3, -1./3, -1./3],
                                   [1./3, 1./3, -2./3],
                                   [1./3, 1./3, 1./3]])

symmetry = phonon.get_symmetry()
print("Space group: %s" % symmetry.get_international_table())

force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()

born = parse_BORN(phonon.get_primitive())
phonon.set_nac_params(born)

# Example to obtain dynamical matrix
dmat = phonon.get_dynamical_matrix_at_q([0,0,0])
print(dmat)

# Example of band structure calculation
bands = []
q_start = np.array([1./3, 1./3, 0])
q_end = np.array([0, 0, 0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

q_start = np.array([0, 0, 0])
q_end = np.array([1./3, 1./3, 1./2])
band = []
for i in range(51):
    band.append(q_start + ( q_end - q_start ) / 50 * i)
bands.append(band)

#*********************
# Matplotlib required
#*********************
print("\nPhonon dispersion:")
phonon.set_band_structure(bands,
                          is_eigenvectors=True)
band_plot = phonon.plot_band_structure(["X", "$\Gamma$", "L"])
band_plot.show()

bands = phonon.get_band_structure()
distances = bands[1]
frequencies = bands[2]
qpoints = bands[0]

for (qs_at_segments,
     dists_at_segments,
     freqs_at_segments) in zip(qpoints, distances, frequencies):

    for q, d, f in zip(qs_at_segments,
                       dists_at_segments,
                       freqs_at_segments):
        print("# %f %f %f" % tuple(q))
        print(("%s " + "%f " * len(f)) % ((d,) + tuple(f)))

# If you just want to plot along q-points of all band segments, the
# following is easier.
        
# all_freqs = np.vstack(frequencies)
# all_qs = np.vstack(qpoints)
# all_dists = np.hstack(distances)
# all_eigvecs = np.concatenate(eigvecs)
# print("# shape of eigvecs %s" % all_eigvecs.shape)
# for d, q, f in zip(all_dists, all_qs, all_freqs):
#     print(("%f " * (4 + len(f))) % ((d,) + tuple(q) + tuple(f)))
