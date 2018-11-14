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
from phonopy.api_phonopy import Phonopy
from phonopy.units import VaspToTHz
from phonopy.structure.cells import (guess_primitive_matrix,
                                     get_primitive_matrix_by_centring)
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
         use_alm=False,
         factor=VaspToTHz,
         frequency_scale_factor=None,
         symprec=1e-5,
         is_symmetry=True,
         log_level=0):
    """Create Phonopy instance from parameters and/or input files.

    When unitcell and unitcell_filename are not given, file name that is
    default for the chosen calculator is looked for in the current directory
    as the default behaviour.

    When force_sets_filename and force_constants_filename are not given,
    'FORCE_SETS' is looked for in the current directory as the default
    behaviour.

    Parameters
    ----------
    supercell_matrix : array_like
        Supercell matrix multiplied to input cell basis vectors.
        shape=(3, ) or (3, 3), where the former is considered a diagonal
        matrix.
        dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is
        the identity matrix.
        shape=(3, 3)
        dtype=float
        When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix,
        the primitive matrix defined at
        https://atztogo.github.io/spglib/definition.html
        is used.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default is
        None.
        {'born': Born effective charges
                 (array_like, shape=(primitive cell atoms, 3, 3), dtype=float),
         'dielectric': Dielectric constant matrix
                       (array_like, shape=(3, 3), dtype=float),
         'factor': unit conversion facotr (float)}
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None.
    calculator : str, optional.
        Calculator used for computing forces. This is used to switch the set
        of physical units. Default is 'vasp'.
    unitcell_filename : str, optional
        Input unit cell filename. Default is None.
    born_filename : str, optional
        Filename corresponding to 'BORN', a file contains non-analytical term
        correction parameters.
    force_sets_filename : str, optional
        Filename of a file corresponding to 'FORCE_SETS', a file contains sets
        of forces and displacements. Default is None.
    force_constants_filename : str, optional
        Filename of a file corresponding to 'FORCE_CONSTANTS' or
        'force_constants.hdf5', a file contains force constants.
        Default is None.
    use_alm : bool, optional
        Default is False.
    factor : float, optional
        Phonon frequency unit conversion factor. Default is
        phonopy.units.VaspToTHz.
    frequency_scale_factor : float, optional
        Factor multiplied to calculated phonon frequency. Default is None,
        i.e., effectively 1.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    is_symmetry : bool, optional
        Setting False, crystal symmetry except for lattice translation is not
        considered. Default is True.
    log_level : int, optional
        Verbosity control. Default is 0.

    """

    if unitcell is None:
        _unitcell, _ = read_crystal_structure(filename=unitcell_filename,
                                              interface_mode=calculator)
    else:
        _unitcell = unitcell

    if len(np.ravel(supercell_matrix)) == 3:
        _supercell_matrix = np.diag(supercell_matrix)
    elif len(np.ravel(supercell_matrix)) == 9:
        _supercell_matrix = np.reshape(supercell_matrix, (3, 3))
    else:
        msg = "supercell_matrix shape has to be (3,) or (3, 3)"
        raise RuntimeError(msg)

    if primitive_matrix in (None, 'F', 'I', 'A', 'C', 'R', 'auto'):
        if primitive_matrix == 'auto':
            _primitive_matrix = guess_primitive_matrix(_unitcell,
                                                       symprec=symprec)
        else:
            _primitive_matrix = get_primitive_matrix_by_centring(
                primitive_matrix)
    elif len(np.ravel(primitive_matrix)) == 9:
        matrix = np.reshape(primitive_matrix, (3, 3))
        if matrix.dtype.kind in ('i', 'u', 'f'):
            det = np.linalg.det(matrix)
            if symprec < det and det < 1 + symprec:
                _primitive_matrix = matrix
            else:
                msg = ("Determinant of primitive_matrix has to be larger "
                       "than 0")
                raise RuntimeError(msg)
    else:
        msg = ("primitive_matrix has to be a 3x3 matrix, None, 'auto', "
               "'F', 'I', 'A', 'C', or 'R'")
        raise RuntimeError(msg)

    # units keywords: factor, nac_factor, distance_to_A
    units = get_default_physical_units(calculator)
    phonon = Phonopy(_unitcell,
                     _supercell_matrix,
                     primitive_matrix=_primitive_matrix,
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
        natom = phonon.supercell.get_number_of_atoms()
        force_sets = parse_FORCE_SETS(natom=natom,
                                      filename=force_sets_filename)
        if force_sets:
            phonon.set_displacement_dataset(force_sets)
            phonon.produce_force_constants(
                calculate_full_force_constants=False,
                use_alm=use_alm)
    elif os.path.isfile("FORCE_SETS"):
        natom = phonon.supercell.get_number_of_atoms()
        force_sets = parse_FORCE_SETS(natom=natom)
        if force_sets:
            phonon.set_displacement_dataset(force_sets)
            phonon.produce_force_constants(
                calculate_full_force_constants=False,
                use_alm=use_alm)

    return phonon
