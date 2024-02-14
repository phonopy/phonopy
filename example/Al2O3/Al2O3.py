"""Example by corundum Al2O3."""

import numpy as np

import phonopy

phonon = phonopy.load(
    unitcell_filename="POSCAR-unitcell", supercell_matrix=[2, 2, 1], log_level=1
)
print("Space group: %s" % phonon.symmetry.get_international_table())

# Example to obtain dynamical matrix
dmat = phonon.get_dynamical_matrix_at_q([0, 0, 0])
print(dmat)

# Example of band structure calculation
bands = []
q_start = np.array([1.0 / 3, 1.0 / 3, 0])
q_end = np.array([0, 0, 0])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

q_start = np.array([0, 0, 0])
q_end = np.array([1.0 / 3, 1.0 / 3, 1.0 / 2])
band = []
for i in range(51):
    band.append(q_start + (q_end - q_start) / 50 * i)
bands.append(band)

print("\nPhonon dispersion:")
phonon.run_band_structure(bands, with_eigenvectors=True, labels=["X", r"$\Gamma$", "L"])
band_plot = phonon.plot_band_structure()
band_plot.show()

bs = phonon.get_band_structure_dict()
distances = bs["distances"]
frequencies = bs["frequencies"]
qpoints = bs["qpoints"]

for qs_at_segments, dists_at_segments, freqs_at_segments in zip(
    qpoints, distances, frequencies
):
    for q, d, f in zip(qs_at_segments, dists_at_segments, freqs_at_segments):
        print("# %f %f %f" % tuple(q))
        print(("%s " + "%f " * len(f)) % ((d,) + tuple(f)))
