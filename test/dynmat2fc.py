#!/usr/bin/env python

import numpy as np
from phonopy import Phonopy
from phonopy.units import VaspToTHz
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS

from phonopy.harmonic.dynmat_to_fc import get_commensurate_points, DynmatToForceConstants

def append_band(bands, q_start, q_end):
    band = []
    for i in range(51):
        band.append(np.array(q_start) +
                    (np.array(q_end) - np.array(q_start)) / 50 * i)
    bands.append(band)

bulk = read_vasp("POSCAR")
phonon = Phonopy(bulk,
                 [[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]],
                 primitive_matrix=[[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]],
                 is_auto_displacements=False)

primitive = phonon.get_primitive()
supercell = phonon.get_supercell()
dynmat2fc = DynmatToForceConstants(primitive, supercell)
com_points = dynmat2fc.get_commensurate_points()

print "Commensurate points"
for i, q in enumerate(com_points):
    print i + 1, q

force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()
phonon.set_qpoints_phonon(com_points,
                          is_eigenvectors=True)
frequencies, eigenvectors = phonon.get_qpoints_phonon()
dynmat2fc.set_dynamical_matrices(frequencies / VaspToTHz, eigenvectors)
dynmat2fc.run()
fc = dynmat2fc.get_force_constants()

phonon2 = Phonopy(bulk,
                  [[2, 0, 0],
                   [0, 2, 0],
                   [0, 0, 2]],
                  primitive_matrix=[[0, 0.5, 0.5],
                                    [0.5, 0, 0.5],
                                    [0.5, 0.5, 0]],
                  is_auto_displacements=False)
phonon2.set_force_constants(fc)
bands = []
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
append_band(bands, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0])
append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
phonon2.set_band_structure(bands)
phonon2.plot_band_structure().show()

 # Artificially increase phonon frequency of a highest band at Gamma
frequencies[0, 5] = 7.5
dynmat2fc.set_dynamical_matrices(frequencies / VaspToTHz, eigenvectors)
dynmat2fc.run()
fc = dynmat2fc.get_force_constants()
phonon3 = Phonopy(bulk,
                  [[2, 0, 0],
                   [0, 2, 0],
                   [0, 0, 2]],
                  primitive_matrix=[[0, 0.5, 0.5],
                                    [0.5, 0, 0.5],
                                    [0.5, 0.5, 0]],
                  is_auto_displacements=False)
phonon3.set_force_constants(fc)
phonon3.set_band_structure(bands)
phonon3.plot_band_structure().show()


