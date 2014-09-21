#!/usr/bin/env python

from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
import numpy as np

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

force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()

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
# phonon.set_nac_params({'born': born,
#                        'factor': factors,
#                        'dielectric': epsilon})

phonon.set_qpoints_phonon(com_points,
                          is_eigenvectors=True)
frequencies, eigenvectors = phonon.get_qpoints_phonon()

dynmat2fc.set_dynamical_matrices(frequencies, eigenvectors)
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
# BAND = 0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5
bands = []
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
append_band(bands, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0])
append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
phonon2.set_band_structure(bands)
phonon2.plot_band_structure().show()


