# Copyright (C) 2018 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from phonopy.api_phonopy import Phonopy
from phonopy.units import VaspToTHz
from phonopy.interface import (read_crystal_structure,
                               get_default_physical_units)
from phonopy.file_IO import (parse_BORN, parse_FORCE_SETS,
                             read_force_constants_hdf5,
                             parse_FORCE_CONSTANTS)


def load(supercell_matrix,
         primitive_matrix=None,
         nac_params=None,
         unitcell=None,
         calculator="vasp",
         unitcell_filename=None,
         born_filename=None,
         force_sets_filename=None,
         force_constants_filename=None,
         factor=VaspToTHz,
         frequency_scale_factor=None,
         symprec=1e-5,
         is_symmetry=True,
         log_level=0):

    if unitcell is None:
        _unitcell, _ = read_crystal_structure(filename=unitcell_filename,
                                              interface_mode=calculator)
    else:
        _unitcell = unitcell

    # units keywords: factor, nac_factor, distance_to_A
    units = get_default_physical_units(calculator)
    phonon = Phonopy(_unitcell,
                     supercell_matrix,
                     primitive_matrix=primitive_matrix,
                     factor=units['factor'])

    if nac_params is None:
        if born_filename is None:
            _nac_params = None
        else:
            _nac_params = parse_BORN(phonon.primitive, filename=born_filename)
    else:
        _nac_params = nac_params

    if _nac_params is not None:
        if _nac_params['factor'] is None:
            _nac_params['factor'] = units['nac_factor']
        phonon.set_nac_params(_nac_params)

    if force_constants_filename is not None:
        dot_split = force_constants_filename.split('.')
        p2s_map = phonon.primitive.get_primitive_to_supercell_map()
        if len(dot_split) > 1 and dot_split[-1] == 'hdf5':
            fc = read_force_constants_hdf5(filename=force_constants_filename,
                                           p2s_map=p2s_map)
        else:
            fc = parse_FORCE_CONSTANTS(filename=force_constants_filename,
                                       p2s_map=p2s_map)
        phonon.set_force_constants(fc)
    elif force_sets_filename is not None:
        force_sets = parse_FORCE_SETS(filename=force_sets_filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()

    return phonon
