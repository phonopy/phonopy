"""Example of NaCl calculation."""

import numpy as np

from phonopy import Phonopy
from phonopy.file_IO import parse_BORN, parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp

# from phonopy.structure.atoms import PhonopyAtoms


def _append_band(bands, q_start, q_end):
    band = []
    for i in range(51):
        band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / 50 * i)
    bands.append(band)


# NaCl crystal structure is read from POSCAR.
unitcell = read_vasp("POSCAR-unitcell")
# This can be given via a PhonopyAtoms class as follows:
# unitcell = PhonopyAtoms(symbols=(['Na'] * 4 + ['Cl'] * 4),
#                         cell=(np.eye(3) * 5.6903014761756712),
#                         scaled_positions=[[0, 0, 0],
#                                           [0, 0.5, 0.5],
#                                           [0.5, 0, 0.5],
#                                           [0.5, 0.5, 0],
#                                           [0.5, 0.5, 0.5],
#                                           [0.5, 0, 0],
#                                           [0, 0.5, 0],
#                                           [0, 0, 0.5]])

phonon = Phonopy(
    unitcell,
    [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
)

symmetry = phonon.symmetry
print("Space group: %s" % symmetry.get_international_table())

force_sets = parse_FORCE_SETS()
phonon.dataset = force_sets
phonon.produce_force_constants()
primitive = phonon.primitive

# Born effective charges and dielectric constants are read from BORN file.
nac_params = parse_BORN(primitive, filename="BORN")
# Or it can be of course given by hand as follows:
# born = [[[1.08703, 0, 0],
#          [0, 1.08703, 0],
#          [0, 0, 1.08703]],
#         [[-1.08672, 0, 0],
#          [0, -1.08672, 0],
#          [0, 0, -1.08672]]]
# epsilon = [[2.43533967, 0, 0],
#            [0, 2.43533967, 0],
#            [0, 0, 2.43533967]]
# factors = 14.400
# nac_params = {'born': born,
#               'factor': factors,
#               'dielectric': epsilon}
phonon.nac_params = nac_params

# BAND = 0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5
bands = []
_append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
_append_band(bands, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0])
_append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
_append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
phonon.run_band_structure(bands)
bs = phonon.band_structure
assert bs is not None
q_points = bs.qpoints
distances = bs.distances
frequencies = bs.frequencies
eigvecs = bs.eigenvectors
for q_path, d_path, freq_path in zip(q_points, distances, frequencies, strict=True):
    for q, d, freq in zip(q_path, d_path, freq_path, strict=True):
        print(
            ("%10.5f  %5.2f %5.2f %5.2f " + (" %7.3f" * len(freq)))
            % ((d, q[0], q[1], q[2]) + tuple(freq))
        )

phonon.plot_band_structure().show()

# Mesh sampling 20x20x20
phonon.run_mesh(mesh=[20, 20, 20])
phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)

# DOS
phonon.run_total_dos(sigma=0.1)
total_dos = phonon.total_dos
assert total_dos is not None
for omega, dos in zip(total_dos.frequency_points, total_dos.dos, strict=True):
    print("%15.7f%15.7f" % (omega, dos))
phonon.plot_total_dos().show()

# Thermal properties
tp = phonon.thermal_properties
assert tp is not None

for t, free_energy, entropy, cv in zip(
    tp.temperatures,
    tp.free_energy,
    tp.entropy,
    tp.heat_capacity,
    strict=True,
):
    print(("%12.3f " + "%15.7f" * 3) % (t, free_energy, entropy, cv))
phonon.plot_thermal_properties().show()

# PDOS
phonon.run_mesh(mesh=[10, 10, 10], is_mesh_symmetry=False, with_eigenvectors=True)
phonon.run_projected_dos(use_tetrahedron_method=True)
pdos = phonon.projected_dos
assert pdos is not None
omegas = pdos.frequency_points
pdos_array = pdos.projected_dos
pdos_indices = [[0], [1]]
phonon.plot_projected_dos(pdos_indices=pdos_indices, legend=pdos_indices).show()
