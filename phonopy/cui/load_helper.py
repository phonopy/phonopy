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

import os
import numpy as np
from phonopy.interface import read_crystal_structure
from phonopy.structure.cells import get_primitive_matrix_by_centring
from phonopy.file_IO import (parse_BORN, parse_FORCE_SETS,
                             read_force_constants_hdf5,
                             parse_FORCE_CONSTANTS)
from phonopy.structure.atoms import PhonopyAtoms


def get_cell_settings(phonopy_yaml=None,
                      supercell_matrix=None,
                      primitive_matrix=None,
                      unitcell=None,
                      supercell=None,
                      unitcell_filename=None,
                      supercell_filename=None,
                      calculator=None,
                      symprec=1e-5):
    if unitcell_filename is not None:
        cell, filename = read_crystal_structure(
            filename=unitcell_filename, interface_mode=calculator)
        smat = _get_supercell_matrix(supercell_matrix)
        pmat = primitive_matrix
    elif supercell_filename is not None:
        cell, filename = read_crystal_structure(
            filename=supercell_filename, interface_mode=calculator)
        smat = np.eye(3, dtype='intc', order='C')
        if primitive_matrix is None or primitive_matrix == "auto":
            pmat = 'auto'
    elif unitcell is not None:
        cell = PhonopyAtoms(atoms=unitcell)
        smat = _get_supercell_matrix(supercell_matrix)
        pmat = primitive_matrix
    elif supercell is not None:
        cell = PhonopyAtoms(atoms=supercell)
        smat = np.eye(3, dtype='intc', order='C')
        if primitive_matrix is None or primitive_matrix == "auto":
            pmat = 'auto'
    else:
        raise RuntimeError("Cell has to be specified.")

    if cell is None:
        msg = "'%s' could not be found." % filename
        raise FileNotFoundError(msg)

    pmat = _get_primitive_matrix(pmat, cell, symprec)

    return cell, smat, pmat


def set_nac_params(phonon, nac_params, born_filename, is_nac, nac_factor):
    _nac_params = None
    if nac_params is not None:
        _nac_params = nac_params
    elif born_filename is not None:
        _nac_params = parse_BORN(phonon.primitive, filename=born_filename)
    elif is_nac is True:
        if os.path.isfile("BORN"):
            _nac_params = parse_BORN(phonon.primitive, filename="BORN")

    if _nac_params is not None:
        if _nac_params['factor'] is None:
            _nac_params['factor'] = nac_factor
        phonon.nac_params = _nac_params


def set_force_constants(
        phonon,
        dataset=None,
        force_constants_filename=None,
        force_sets_filename=None,
        use_alm=False):
    natom = phonon.supercell.get_number_of_atoms()

    _dataset = None
    if dataset is not None:
        _dataset = dataset
    elif force_constants_filename is not None:
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
        _dataset = parse_FORCE_SETS(natom=natom,
                                    filename=force_sets_filename)
    elif os.path.isfile("FORCE_SETS"):
        _dataset = parse_FORCE_SETS(natom=natom)

    if _dataset:
        phonon.set_displacement_dataset(_dataset)
        phonon.produce_force_constants(
            calculate_full_force_constants=False,
            use_alm=use_alm)


def _get_supercell_matrix(smat):
    if smat is None:
        _smat = np.eye(3, dtype='intc', order='C')
    elif len(np.ravel(smat)) == 3:
        _smat = np.diag(smat)
    elif len(np.ravel(smat)) == 9:
        _smat = np.reshape(smat, (3, 3))
    else:
        msg = "supercell_matrix shape has to be (3,) or (3, 3)"
        raise RuntimeError(msg)
    return _smat


def _get_primitive_matrix(pmat, unitcell, symprec):
    if type(pmat) is str and pmat in ('F', 'I', 'A', 'C', 'R', 'auto'):
        if pmat == 'auto':
            _pmat = pmat
        else:
            _pmat = get_primitive_matrix_by_centring(pmat)
    elif pmat is None:
        _pmat = None
    elif len(np.ravel(pmat)) == 9:
        matrix = np.reshape(pmat, (3, 3))
        if matrix.dtype.kind in ('i', 'u', 'f'):
            det = np.linalg.det(matrix)
            if symprec < det and det < 1 + symprec:
                _pmat = matrix
            else:
                msg = ("Determinant of primitive_matrix has to be larger "
                       "than 0")
                raise RuntimeError(msg)
    else:
        msg = ("primitive_matrix has to be a 3x3 matrix, None, 'auto', "
               "'F', 'I', 'A', 'C', or 'R'")
        raise RuntimeError(msg)

    return _pmat
