"""Example of band structure and group velocity calculation by NaCl."""

import numpy as np

import phonopy


def _append_band(bands, q_start, q_end):
    band = []
    nq = 51
    for i in range(nq):
        band.append(
            np.array(q_start) + (np.array(q_end) - np.array(q_start)) / (nq - 1) * i
        )
    bands.append(band)


phonon = phonopy.load(
    unitcell_filename="POSCAR-unitcell",
    born_filename="BORN",
    force_sets_filename="FORCE_SETS",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
)

bands = []
_append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
_append_band(bands, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0])
_append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
_append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
bs = phonon.run_band_structure(bands, with_group_velocities=True)
frequencies = bs.frequencies
group_velocities = bs.group_velocities
print(len(frequencies))
print(frequencies[0].shape)
print(len(group_velocities))
print(group_velocities[0].shape)
