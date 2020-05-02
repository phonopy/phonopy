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
from phonopy.file_IO import parse_FORCE_SETS, parse_FORCE_CONSTANTS
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.calculator import get_default_physical_units
import phonopy.cui.load_helper as load_helper


def load(phonopy_yaml=None,  # phonopy.yaml-like must be the first argument.
         supercell_matrix=None,
         primitive_matrix=None,
         is_nac=True,
         calculator=None,
         unitcell=None,
         supercell=None,
         nac_params=None,
         unitcell_filename=None,
         supercell_filename=None,
         born_filename=None,
         force_sets_filename=None,
         force_constants_filename=None,
         fc_calculator=None,
         factor=None,
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
    phonopy_yaml : str, optional
        Filename of "phonopy.yaml"-like file. If this is given, the data
        in the file are parsed. Default is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to input cell basis vectors.
        shape=(3, ) or (3, 3), where the former is considered a diagonal
        matrix. Default is the unit matrix.
        dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is
        the identity matrix.
        shape=(3, 3)
        dtype=float
        When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix,
        the primitive matrix defined at
        https://spglib.github.io/spglib/definition.html
        is used.
    is_nac : bool, optional
        If True, look for 'BORN' file. If False, NAS is turned off.
        The priority for NAC is nac_params > born_filename > is_nac ('BORN').
        Default is True.
    calculator : str, optional.
        Calculator used for computing forces. This is used to switch the set
        of physical units. Default is None, which is equivalent to "vasp".
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None. The priority for cell is
        unitcell_filename > supercell_filename > unitcell > supercell.
    supercell : PhonopyAtoms, optional
        Input supercell cell. Default value of primitive_matrix is set to
        'auto' (can be overwitten). supercell_matrix is ignored. Default is
        None. The priority for cell is
        unitcell_filename > supercell_filename > unitcell > supercell.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default is
        None. The priority for NAC is nac_params > born_filename > is_nac.
        {'born': Born effective charges
                 (array_like, shape=(primitive cell atoms, 3, 3), dtype=float),
         'dielectric': Dielectric constant matrix
                       (array_like, shape=(3, 3), dtype=float),
         'factor': unit conversion facotr (float)}
    unitcell_filename : str, optional
        Input unit cell filename. Default is None. The priority for cell is
        unitcell_filename > supercell_filename > unitcell > supercell.
    supercell_filename : str, optional
        Input supercell filename. When this is specified, supercell_matrix is
        ignored. Default is None. The priority for cell is
        1. unitcell_filename (with supercell_matrix)
        2. supercell_filename
        3. unitcell (with supercell_matrix)
        4. supercell.
    born_filename : str, optional
        Filename corresponding to 'BORN', a file contains non-analytical term
        correction parameters.
        The priority for NAC is nac_params > born_filename > is_nac ('BORN').
    force_sets_filename : str, optional
        Filename of a file corresponding to 'FORCE_SETS', a file contains sets
        of forces and displacements. Default is None.
        The priority for force constants is
        force_constants_filename > force_sets_filename > 'FORCE_SETS'.
    force_constants_filename : str, optional
        Filename of a file corresponding to 'FORCE_CONSTANTS' or
        'force_constants.hdf5', a file contains force constants.
        Default is None.
        The priority for force constants is
        force_constants_filename > force_sets_filename > 'FORCE_SETS'.
    fc_calculator : str, optional
        Force constants calculator. Currently only 'alm'. Default is None.
    factor : float, optional
        Phonon frequency unit conversion factor. Unless specified, default
        unit conversion factor for each calculator is used.
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

    if phonopy_yaml is None:
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=calculator,
            symprec=symprec)
        _nac_params = nac_params
        _dataset = None
        _fc = None
    else:
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(phonopy_yaml)
        cell = phpy_yaml.unitcell
        smat = phpy_yaml.supercell_matrix
        if smat is None:
            smat = np.eye(3, dtype='intc', order='C')
        if primitive_matrix == 'auto':
            pmat = 'auto'
        else:
            pmat = phpy_yaml.primitive_matrix
        if is_nac:
            _nac_params = phpy_yaml.nac_params
        else:
            _nac_params = None
        _dataset = phpy_yaml.dataset
        _fc = phpy_yaml.force_constants

    # units keywords: factor, nac_factor, distance_to_A
    units = get_default_physical_units(calculator)
    if factor is None:
        _factor = units['factor']
    else:
        _factor = factor
    phonon = Phonopy(cell,
                     smat,
                     primitive_matrix=pmat,
                     factor=_factor,
                     frequency_scale_factor=frequency_scale_factor,
                     symprec=symprec,
                     is_symmetry=is_symmetry,
                     calculator=calculator,
                     log_level=log_level)
    _nac_params = load_helper.get_nac_params(phonon.primitive,
                                             _nac_params,
                                             born_filename,
                                             is_nac,
                                             units['nac_factor'])
    if _dataset is not None:
        phonon.dataset = _dataset

    if _nac_params is not None:
        phonon.nac_params = _nac_params

    if _fc is not None:
        phonon.force_constants = _fc
    else:
        _compute_force_constants(
            phonon,
            dataset=_dataset,
            force_constants_filename=force_constants_filename,
            force_sets_filename=force_sets_filename,
            calculator=calculator,
            fc_calculator=fc_calculator)

    return phonon


def _compute_force_constants(
        phonon,
        dataset=None,
        force_constants_filename=None,
        force_sets_filename=None,
        calculator=None,
        fc_calculator=None):
    natom = phonon.supercell.get_number_of_atoms()

    _dataset = None
    if dataset is not None:
        _dataset = dataset
    elif force_constants_filename is not None:
        dot_split = force_constants_filename.split('.')
        p2s_map = phonon.primitive.p2s_map
        if len(dot_split) > 1 and dot_split[-1] == 'hdf5':
            fc = load_helper.read_force_constants_from_hdf5(
                filename=force_constants_filename,
                p2s_map=p2s_map,
                calculator=calculator)
        else:
            fc = parse_FORCE_CONSTANTS(filename=force_constants_filename,
                                       p2s_map=p2s_map)
        phonon.set_force_constants(fc)
    elif force_sets_filename is not None:
        _dataset = parse_FORCE_SETS(natom=natom,
                                    filename=force_sets_filename)
    elif os.path.isfile("FORCE_SETS"):
        _dataset = parse_FORCE_SETS(natom=natom)

    if _dataset is not None:
        phonon.dataset = _dataset
        _produce_force_constants(phonon, fc_calculator)


def _produce_force_constants(phonon, fc_calculator):
    try:
        phonon.produce_force_constants(
            calculate_full_force_constants=False,
            fc_calculator=fc_calculator)
    except RuntimeError:
        pass
