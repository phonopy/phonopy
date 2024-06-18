"""Example to calculate mode Grueneisen parameters."""

from __future__ import annotations

import numpy as np

from phonopy import Phonopy, PhonopyGruneisen
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp


def _append_band(bands: list[list], q_start, q_end):
    band = []
    for i in range(51):
        points = np.array(q_start) + (np.array(q_end) - np.array(q_start)) / 50 * i
        band.append(points.tolist())
    bands.append(band)


phonons = {}
for vol in ("orig", "plus", "minus"):
    unitcell = read_vasp("%s/POSCAR-unitcell" % vol)
    phonon = Phonopy(
        unitcell,
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    force_sets = parse_FORCE_SETS(filename="%s/FORCE_SETS" % vol)
    phonon.set_displacement_dataset(force_sets)
    phonon.produce_force_constants()
    phonons[vol] = phonon

gruneisen = PhonopyGruneisen(phonons["orig"], phonons["plus"], phonons["minus"])

gruneisen.set_mesh([2, 2, 2])
q_points, _, frequencies, _, gammas = gruneisen.get_mesh()
for q, freq, g in zip(q_points, frequencies, gammas):
    print(
        ("%5.2f %5.2f %5.2f " + (" %7.3f" * len(freq)))
        % ((q[0], q[1], q[2]) + tuple(freq))
    )
    print(((" " * 18) + (" %7.3f" * len(g))) % tuple(g))

bands: list[list] = []
_append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
_append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
gruneisen.set_band_structure(bands)

q_points, distances, frequencies, _, gammas = gruneisen.get_band_structure()
for q_path, d_path, freq_path, g_path in zip(q_points, distances, frequencies, gammas):
    for q, d, freq, g in zip(q_path, d_path, freq_path, g_path):
        print(
            ("%10.5f  %5.2f %5.2f %5.2f " + (" %7.3f" * len(freq)))
            % ((d, q[0], q[1], q[2]) + tuple(freq))
        )
        print(((" " * 30) + (" %7.3f" * len(g))) % tuple(g))
