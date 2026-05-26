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

import os
import typing
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import phonopy.cui.load_helper as load_helper
from phonopy._lang import resolve_lang
from phonopy.api_phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.physical_units import get_calculator_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.dataset import forces_in_dataset


def load(
    phonopy_yaml: str
    | os.PathLike
    | typing.IO
    | None = None,  # phonopy.yaml-like must be the first argument.
    supercell_matrix: (
        Sequence[int] | Sequence[Sequence[int]] | NDArray[np.int64] | None
    ) = None,
    primitive_matrix: Sequence[Sequence[float]]
    | Literal["P", "F", "I", "A", "C", "R", "auto"]
    | NDArray[np.double]
    | None = "auto",
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
    use_SNF_supercell: bool = False,
    symprec: float = 1e-5,
    log_level: int = 0,
    lang: Literal["C", "Rust"] = "Rust",
) -> Phonopy:
    """Create a ``Phonopy`` instance from parameters and/or input files.

    A "phonopy_yaml"-like file is parsed unless crystal structure
    information is given through ``unitcell_filename``,
    ``supercell_filename``, ``unitcell`` (``PhonopyAtoms``-like), or
    ``supercell`` (``PhonopyAtoms``-like). Even when a "phonopy_yaml"-like
    file is parsed, parameters other than crystal structure can be
    overwritten.

    Phonopy's default ``FORCE_SETS`` and ``BORN`` files are parsed when
    they are found in the current directory and the corresponding data
    are not yet provided by other means.

    **Crystal structure**

    Means to provide crystal structure(s), in order of priority:

    1. ``unitcell_filename`` (with ``supercell_matrix``)
    2. ``supercell_filename``
    3. ``unitcell`` (with ``supercell_matrix``)
    4. ``supercell``
    5. ``phonopy_yaml``

    **Force sets or force constants**

    Optional. Means to provide information used to generate force
    constants, in order of priority:

    1. ``force_constants_filename``
    2. ``force_sets_filename``
    3. ``phonopy_yaml`` if force constants are found in it.
    4. ``phonopy_yaml`` if forces are found in ``phonopy_yaml.dataset``.
    5. ``FORCE_CONSTANTS`` is searched in the current directory.
    6. ``force_constants.hdf5`` is searched in the current directory.
    7. ``FORCE_SETS`` is searched in the current directory.

    When both 3 and 4 are satisfied but not the others, force constants
    and dataset are both stored on the ``Phonopy`` instance, but force
    constants are not produced from the dataset.

    **Parameters for non-analytical term correction (NAC)**

    Optional. Means to provide NAC parameters, in order of priority:

    1. ``born_filename``
    2. ``nac_params``
    3. ``phonopy_yaml.nac_params`` if present and ``is_nac=True``.
    4. ``BORN`` is searched in the current directory when ``is_nac=True``.

    Parameters
    ----------
    phonopy_yaml : str, os.PathLike, typing.IO, optional
        Filename of a "phonopy.yaml"-like file (``str`` / ``bytes``) or
        a file-pointer-like object. If given, the data in the file are
        parsed. Default is None.
    supercell_matrix : array_like, optional
        Supercell matrix multiplied to the input cell basis vectors.
        ``shape=(3,)`` or ``(3, 3)``; the former is interpreted as a
        diagonal matrix. ``dtype=int``. Default is the identity matrix.
    primitive_matrix : array_like or str, optional
        Primitive matrix multiplied to the input cell basis vectors.
        Default is ``'auto'``, which guesses the primitive matrix from
        crystal symmetry. ``None`` is treated the same as ``'auto'``.
        To use the unit cell as the primitive cell (identity
        transformation), pass ``'P'``. For array_like input,
        ``shape=(3, 3)``, ``dtype=float``. When ``'F'``, ``'I'``,
        ``'A'``, ``'C'``, or ``'R'`` is given instead of a 3x3 matrix,
        the primitive matrix for the character found at
        https://spglib.github.io/spglib/definition.html is used. When a
        "phonopy.yaml"-like file is loaded and it carries a
        ``primitive_matrix``, that value takes priority over the
        default ``'auto'``.
    is_nac : bool, optional
        If True, look for a ``BORN`` file. If False, NAC is turned off.
        Default is True.
    calculator : str, optional
        Calculator used for computing forces. This is used to switch
        the set of physical units. Default is None, which is equivalent
        to ``"vasp"``.
    unitcell : PhonopyAtoms, optional
        Input unit cell. Default is None.
    supercell : PhonopyAtoms, optional
        Input supercell. When given, the default value of
        ``primitive_matrix`` is set to ``'auto'`` (can be overwritten),
        and ``supercell_matrix`` is ignored. Default is None.
    nac_params : dict, optional
        Parameters required for non-analytical term correction. Default
        is None. Expected structure::

            {'born': Born effective charges
                     (array_like, shape=(primitive cell atoms, 3, 3),
                      dtype=float),
             'dielectric': Dielectric constant matrix
                           (array_like, shape=(3, 3), dtype=float),
             'factor': unit conversion factor (float)}

    unitcell_filename : str, optional
        Input unit cell filename. Default is None.
    supercell_filename : str, optional
        Input supercell filename. When specified, ``supercell_matrix``
        is ignored. Default is None.
    born_filename : str, optional
        Filename of a ``BORN``-format file containing non-analytical
        term correction parameters.
    force_sets_filename : str, optional
        Filename of a file in ``FORCE_SETS`` format containing sets of
        forces and displacements. Default is None.
    force_constants_filename : str, optional
        Filename of a file in ``FORCE_CONSTANTS`` or
        ``force_constants.hdf5`` format containing force constants.
        Default is None.
    fc_calculator : {"traditional", "symfc", "alm", None}, optional
        Force constants calculator backend. Default is None.
    fc_calculator_options : str, optional
        Optional parameters passed to the external fc-calculator as a
        single text string. Parsing rules depend on the calculator.
        For ``alm``, each parameter is split by ``','`` and each
        key-value pair is written as ``'key = value'``.
    factor : float, optional
        Deprecated. The conversion factor is selected based on
        ``calculator``.
    produce_fc : bool, optional
        When False, force constants are not calculated from the dataset
        of displacements and forces even if the dataset exists. Default
        is True.
    is_symmetry : bool, optional
        When False, crystal symmetry (except for lattice translation)
        is not considered. Default is True.
    symmetrize_fc : bool, optional
        When False, force constants are not symmetrized when creating
        them from displacements and forces. Default is True.
    is_compact_fc : bool, optional
        When True, force constants are produced with
        ``shape=(primitive, supercell, 3, 3)``; when False, with
        ``shape=(supercell, supercell, 3, 3)``. ``supercell`` and
        ``primitive`` indicate the number of atoms in those cells.
        Default is True.
    use_pypolymlp : bool, optional
        Use pypolymlp for generating force constants. Default is False.
    mlp_params : dict, optional
        A set of parameters used by machine-learning potentials.
    use_SNF_supercell : bool, optional
        Build the supercell with the SNF algorithm when True. Default
        is False. The SNF algorithm is faster than the original one,
        but the order of atoms in the supercell can be different.
        Backward compatibility with old data (e.g., force constants)
        is therefore not guaranteed.
    symprec : float, optional
        Tolerance used to find crystal symmetry. Default is 1e-5.
    log_level : int, optional
        Verbosity control. Default is 0.
    lang : Literal["C", "Rust"], optional
        Backend implementation for compute-heavy kernels. ``"C"`` uses
        the existing C extension; ``"Rust"`` selects the experimental
        phonors backend. Default is ``"Rust"``.

    """
    lang = resolve_lang(lang)
    if primitive_matrix is None:
        primitive_matrix = "auto"
    if (
        supercell is not None
        or supercell_filename is not None
        or unitcell is not None
        or unitcell_filename is not None
    ):
        cell, smat, pmat = load_helper.get_cell_settings(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
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
            smat = np.eye(3, dtype="int64", order="C")
        # When the caller leaves primitive_matrix at the default "auto",
        # a value stored in the yaml takes priority (preserves the cell
        # transformation that was used originally).
        if primitive_matrix == "auto" and phpy_yaml.primitive_matrix is not None:
            pmat = phpy_yaml.primitive_matrix
        else:
            pmat = primitive_matrix
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
        use_SNF_supercell=use_SNF_supercell,
        calculator=_calculator,
        log_level=log_level,
        lang=lang,
    )

    units = get_calculator_physical_units(_calculator)
    # NAC params
    if born_filename is not None or _nac_params is not None or is_nac:
        ret_nac_params = load_helper.get_nac_params(
            primitive=phonon.primitive,
            nac_params=_nac_params,
            born_filename=born_filename,
            is_nac=is_nac,
            nac_factor=units.nac_factor,
            log_level=log_level,
            lang=lang,
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
        load_helper.move_force_dataset_to_mlp_dataset(phonon)
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
