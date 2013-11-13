from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
import numpy as np

cell = read_vasp("POSCAR")

# Initialize phonon. Supercell matrix has to have the shape of (3, 3)
phonon = Phonopy(cell, np.diag([2, 2, 1]))

symmetry = phonon.get_symmetry()
print "Space group:", symmetry.get_international_table()

# Read and convert forces and displacements
force_sets = parse_FORCE_SETS(cell.get_number_of_atoms() * 4)
# Sets of forces have to be set before phonon.set_post_process or
# at phonon.set_post_process(..., sets_of_forces=sets_of_forces, ...).
phonon.set_force_sets(force_sets)

# To activate non-analytical term correction, 'is_nac=True' has to be set here.
phonon.set_post_process(primitive_matrix=[[2./3, -1./3, -1./3],
                                          [1./3, 1./3, -2./3],
                                          [1./3, 1./3, 1./3]],
                        is_nac=True)

# Parameters for non-analytical term correction can be set
# also after phonon.set_post_process
born = parse_BORN(phonon.get_primitive())
phonon.set_nac_params(born)

# Example to obtain dynamical matrix
dmat = phonon.get_dynamical_matrix_at_q([0,0,0])
print dmat

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
print "\nPhonon dispersion:"
phonon.set_band_structure(bands,
                          is_eigenvectors=True)
phonon.plot_band_structure(["X", "$\Gamma$", "L"]).show()

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
        print "# %f %f %f" % tuple(q)
        print d, ("%f " * len(f)) % tuple(f)

# If you just want to plot along q-points of all band segments, the
# following is easier.
        
# all_freqs = np.vstack(frequencies)
# all_qs = np.vstack(qpoints)
# all_dists = np.hstack(distances)
# all_eigvecs = np.concatenate(eigvecs)
# print "# shape of eigvecs", all_eigvecs.shape
# for d, q, f in zip(all_dists, all_qs, all_freqs):
#     print ("%f " * (4 + len(f))) % ((d,) + tuple(q) + tuple(f))
