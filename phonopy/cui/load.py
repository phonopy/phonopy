"""Phonopy loader."""

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

from __future__ import annotations

import io
import os
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

import phonopy.cui.load_helper as load_helper
from phonopy.api_phonopy import Phonopy
from phonopy.interface.calculator import get_calculator_physical_units
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive_matrix
from phonopy.structure.dataset import forces_in_dataset


def load(
    phonopy_yaml: str
    | os.PathLike
    | io.IOBase
    | None = None,  # phonopy.yaml-like must be the first argument.
    supercell_matrix: ArrayLike | None = None,
    primitive_matrix: ArrayLike | str | None = None,
    is_nac: bool = True,
    calculator: str | None = None,
    unitcell: PhonopyAtoms | None = None,
    supercell: PhonopyAtoms | None = None,
    nac_params: dict | None = None,
    unitcell_filename: os.PathLike | str | None = None,
    supercell_filename: os.PathLike | str | None = None,
    born_filename: os.PathLike | str | None = None,
    force_sets_filename: os.PathLike | str | None = None,
    force_constants_filename: os.PathLike | str | None = None,
    fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
    fc_calculator_options: str | None = None,
    factor: float | None = None,
    produce_fc: bool = True,
    is_symmetry: bool = True,
    symmetrize_fc: bool = True,
    is_compact_fc: bool = True,
    use_pypolymlp: bool = False,
    mlp_params: dict | None = None,
    store_dense_svecs: bool = True,
    use_SNF_supercell: bool = False,
    symprec: float = 1e-5,
    log_level: int = 0,
) -> Phonopy:
    """Create Phonopy instance from parameters and/or input files.

    "phonopy_yaml"-like file is parsed unless crystal structure information is
    given by unitcell_filename, supercell_filename, unitcell
    (PhonopyAtoms-like), or supercell (PhonopyAtoms-like). Even when
    "phonopy_yaml"-like file is parse, parameters except for crystal structure
    can be overwritten.

    Phonopy default files of 'FORCE_SETS' and 'BORN' are parsed when they are
    found in current directory and those data are not yet provided by other
    means.

    Crystal structure
    -----------------
    Means to provide crystal structure(s) and their priority:
        1. unitcell_filename (with supercell_matrix)
        2. supercell_filename
        3. unitcell (with supercell_matrix)
        4. supercell.
        5. phonopy_yaml

    Force sets or force constants
    -----------------------------
    Optional. Means to provide information to generate force constants and their
    priority:
        1. force_constants_filename
        2. force_sets_filename
        3. phonopy_yaml if force constants are found in phonoy_yaml.
        4. phonopy_yaml if forces are found in phonoy_yaml.dataset.
        5. 'FORCE_CONSTANTS' is searched in current directory.
        6. 'force_constants.hdf5' is searched in current directory.
        7. 'FORCE_SETS' is searched in current directory.
    When both of 3 and 4 are satisfied but not others, force constants and
    dataset are stored in Phonopy instance, but force constants are not produced
    from dataset.

    Parameters for non-analytical term correctiion (NAC)
    ----------------------------------------------------
    Optional. Means to provide NAC parameters and their priority:
        1. born_filename
        2. nac_params
        3. phonopy_yaml.nac_params if existed and is_nac=True.
        4. 'BORN' is searched in current directory when is_nac=True.

    Parameters
    ----------
    phonopy_yaml : str, os.PathLike, typing.IO, optional
        Filename of "phonopy.yaml"-like file for str or bytes, otherwise file
        pointer-like. If this is given, the data in the file are parsed. Default
        is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to input cell basis vectors. shape=(3, ) or
        (3, 3), where the former is considered a diagonal matrix. Default is the
        unit matrix. dtype=int
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to input cell basis vectors. Default is
        None, which is equivalent to 'auto'. For array_like, shape=(3, 3),
        dtype=float. When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3
        matrix, the primitive matrix for the character found at
        https://spglib.github.io/spglib/definition.html is used.
    is_nac : bool, optional
        If True, look for 'BORN' file. If False, NAS is turned off. Default is
        True.
    calculator : str, optional.
        Calculator used for computing forces. This is used to switch the set of
        physical units. Default is None, which is equivalent to "vasp".
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None.
    supercell : PhonopyAtoms, optional
        Input supercell. With given, default value of primitive_matrix is set to
        'auto' (can be overwitten). supercell_matrix is ignored. Default is
        None.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default is None.
        {'born': Born effective charges
                 (array_like, shape=(primitive cell atoms, 3, 3), dtype=float),
         'dielectric': Dielectric constant matrix
                       (array_like, shape=(3, 3), dtype=float),
         'factor': unit conversion facotr (float)}
    unitcell_filename : str, optional
        Input unit cell filename. Default is None.
    supercell_filename : str, optional
        Input supercell filename. When this is specified, supercell_matrix is
        ignored. Default is None.
    born_filename : str, optional
        Filename corresponding to 'BORN', a file contains non-analytical term
        correction parameters.
    force_sets_filename : str, optional
        Filename of a file corresponding to 'FORCE_SETS', a file contains sets
        of forces and displacements. Default is None.
    force_constants_filename : str, optional
        Filename of a file corresponding to 'FORCE_CONSTANTS' or
        'force_constants.hdf5', a file contains force constants. Default is
        None.
    fc_calculator : str, optional
        Force constants calculator. Currently only 'alm'. Default is None.
    fc_calculator_options : str, optional
        Optional parameters that are passed to the external fc-calculator. This
        is given as one text string. How to parse this depends on the
        fc-calculator. For alm, each parameter is splitted by comma ',', and
        each set of key and value pair is written in 'key = value'.
    factor : float, optional
        Deprecated. Conversion factor is selected based off of `calculator`
    produce_fc : bool, optional
        Setting False, force constants are not calculated from dataset of
        displacements and forces even if the dataset exists. Default is True.
    is_symmetry : bool, optional
        Setting False, crystal symmetry except for lattice translation is not
        considered. Default is True.
    symmetrize_fc : bool, optional
        Setting False, force constants are not symmetrized when creating force
        constants from displacements and forces. Default is True.
    is_compact_fc : bool, optional
        Force constants are produced in the array whose shape is
            True: (primitive, supecell, 3, 3) False: (supercell, supecell, 3, 3)
        where 'supercell' and 'primitive' indicate number of atoms in these
        cells. Default is True.
    use_pypolymlp : bool, optional
        Use pypolymlp for generating force constants. Default is False.
    mlp_params : dict, optional
        A set of parameters used by machine learning potentials.
    store_dense_svecs : bool, optional
        Deprected. Dataset of shortest vectors between atoms in primitive
        cell and supercell is stored in the dense format when this is True.
        Default is True. In phonopy v3 or later version, False will not be
        supported.
    use_SNF_supercell : bool, optional
        Supercell is built by SNF algorithm when True. Default is False. SNF
        algorithm is faster than the original one, but the order of atoms in the
        supercell can be different from the original one. So the backward
        compatibility to the old data (e.g., force constants) may not be
        preserved.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    log_level : int, optional
        Verbosity control. Default is 0.

    """
    if (
        supercell is not None
        or supercell_filename is not None
        or unitcell is not None
        or unitcell_filename is not None
    ):
        if primitive_matrix is None:
            _primitive_matrix = "auto"
        else:
            _primitive_matrix = primitive_matrix
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=_primitive_matrix,
            unitcell=unitcell,
            supercell=supercell,
            unitcell_filename=unitcell_filename,
            supercell_filename=supercell_filename,
            calculator=calculator,
            symprec=symprec,
            log_level=log_level,
        )
        _calculator = calculator
        _nac_params = nac_params
        _dataset = None
        _fc = None
    elif phonopy_yaml is not None:
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(phonopy_yaml)
        cell = phpy_yaml.unitcell
        smat = phpy_yaml.supercell_matrix
        if smat is None:
            smat = np.eye(3, dtype="intc", order="C")
        if primitive_matrix is not None:
            pmat = get_primitive_matrix(primitive_matrix, symprec=symprec)
        else:
            pmat = phpy_yaml.primitive_matrix
        if nac_params is not None:
            _nac_params = nac_params
        elif is_nac:
            _nac_params = phpy_yaml.nac_params
        else:
            _nac_params = None
        _dataset = phpy_yaml.dataset
        _fc = phpy_yaml.force_constants
        if calculator is None:
            _calculator = phpy_yaml.calculator
        else:
            _calculator = calculator

    if cell is None:
        msg = "Cell information could not found. Phonopy instance loading failed."
        raise RuntimeError(msg)

    if log_level and _calculator is not None:
        print(f'Set "{_calculator}" mode.')

    phonon = Phonopy(
        cell,
        smat,
        primitive_matrix=pmat,
        factor=factor,
        symprec=symprec,
        is_symmetry=is_symmetry,
        store_dense_svecs=store_dense_svecs,
        use_SNF_supercell=use_SNF_supercell,
        calculator=_calculator,
        set_factor_by_calculator=True,
        log_level=log_level,
    )

    units = get_calculator_physical_units(_calculator)
    # NAC params
    if born_filename is not None or _nac_params is not None or is_nac:
        ret_nac_params = load_helper.get_nac_params(
            primitive=phonon.primitive,
            nac_params=_nac_params,
            born_filename=born_filename,
            is_nac=is_nac,
            nac_factor=units["nac_factor"],
            log_level=log_level,
        )
        if ret_nac_params is not None:
            phonon.nac_params = ret_nac_params

    dataset = load_helper.select_and_load_dataset(
        len(phonon.supercell),
        _dataset,
        phonopy_yaml_filename=phonopy_yaml,
        force_sets_filename=force_sets_filename,
        log_level=log_level,
    )
    if dataset is not None:
        phonon.dataset = dataset

    fc = load_helper.select_and_extract_force_constants(
        phonon,
        force_constants=_fc,
        force_constants_filename=force_constants_filename,
        is_compact_fc=is_compact_fc,
        log_level=log_level,
    )
    if fc is not None:
        phonon.force_constants = fc

    if use_pypolymlp and dataset is not None:
        phonon.mlp_dataset = dataset
        phonon.dataset = None
        load_helper.develop_or_load_pypolymlp(
            phonon, mlp_params=mlp_params, log_level=log_level
        )
    if (
        phonon.force_constants is None
        and produce_fc
        and forces_in_dataset(phonon.dataset)
    ):
        load_helper.produce_force_constants(
            phonon,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            symmetrize_fc=symmetrize_fc,
            is_compact_fc=is_compact_fc,
            use_symfc_projector=True,
            log_level=log_level,
        )

    return phonon
