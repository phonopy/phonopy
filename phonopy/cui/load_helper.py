"""Helper methods of phonopy loader."""

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

import dataclasses
import os
import pathlib
import typing
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phonopy import Phonopy
from phonopy.exception import (
    ForcesetsNotFoundError,
    PypolymlpDevelopmentError,
    PypolymlpFileNotFoundError,
    PypolymlpTrainingDatasetNotFoundError,
)
from phonopy.file_IO import (
    parse_BORN,
    parse_FORCE_CONSTANTS,
    parse_FORCE_SETS,
    read_force_constants_hdf5,
)
from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
    full_fc_to_compact_fc,
)
from phonopy.interface.calculator import (
    get_force_constant_conversion_factor,
    read_crystal_structure,
)
from phonopy.interface.pypolymlp import (
    PypolymlpParams,
    parse_mlp_params,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, get_primitive_matrix
from phonopy.structure.dataset import forces_in_dataset


def get_cell_settings(
    supercell_matrix: ArrayLike | None = None,
    primitive_matrix: ArrayLike | str | None = None,
    unitcell: PhonopyAtoms | None = None,
    supercell: PhonopyAtoms | None = None,
    unitcell_filename: str | os.PathLike | None = None,
    supercell_filename: str | os.PathLike | None = None,
    calculator: str | None = None,
    symprec: float = 1e-5,
    log_level: int = 0,
) -> tuple[PhonopyAtoms, ArrayLike | None, str | NDArray | None]:
    """Return crystal structures."""
    optional_structure_info = None
    if primitive_matrix is None or (
        isinstance(primitive_matrix, str) and primitive_matrix == "auto"
    ):
        pmat = "auto"
    else:
        pmat = primitive_matrix

    if unitcell_filename is not None:
        cell, optional_structure_info = _read_crystal_structure(
            filename=unitcell_filename, interface_mode=calculator
        )
        smat = supercell_matrix
        if log_level:
            print(
                'Unit cell structure was read from "%s".' % optional_structure_info[0]
            )
    elif supercell_filename is not None:
        cell, optional_structure_info = read_crystal_structure(
            filename=supercell_filename, interface_mode=calculator
        )
        smat = np.eye(3, dtype="intc", order="C")
        if log_level:
            print(
                'Supercell structure was read from "%s".' % optional_structure_info[0]
            )
    elif unitcell is not None:
        cell = unitcell.copy()
        smat = supercell_matrix
    elif supercell is not None:
        cell = supercell.copy()
        smat = np.eye(3, dtype="intc", order="C")
    else:
        raise RuntimeError("Cell has to be specified.")

    if optional_structure_info is not None and cell is None:
        filename = optional_structure_info[0]
        msg = "'%s' could not be found." % filename
        raise FileNotFoundError(msg)

    pmat = get_primitive_matrix(pmat, symprec=symprec)

    return cell, smat, pmat


def get_nac_params(
    primitive: Primitive | None = None,
    nac_params: dict | None = None,
    born_filename: str | os.PathLike | None = None,
    is_nac: bool = True,
    nac_factor: float | None = None,
    log_level: int = 0,
) -> dict | None:
    """Look for and return NAC parameters.

    Parameters
    ----------
    primitive : Primitive
        Primitive cell.
    nac_params : dict
        NAC parameters.
    born_filename : str
        Filename of BORN file.
    is_nac : bool
        Whether to read NAC parameters from BORN file.
    nac_factor : float
        Unit conversion factor for non-analytical term correction. This value is
        set only when the NAC factor is unset by other means.
    log_level : int
        Log level.

    Returns
    -------
    dict
        Parameters used for non-analytical term correction 'born': ndarray
            Born effective charges. shape=(primitive cell atoms, 3, 3),
            dtype='double', order='C'
        'dielectric': ndarray
            Dielectric constant tensor. shape=(3, 3), dtype='double', order='C'
        'factor': float, optional
            Unit conversion factor.

    """
    if born_filename is not None:
        if primitive is None:
            raise ValueError(
                "Primitive cell has to be specified when born_filename is given."
            )
        _nac_params = parse_BORN(primitive, filename=born_filename)
        if log_level:
            print('NAC parameters were read from "%s".' % born_filename)
    elif nac_params is not None:  # nac_params input or phonopy_yaml.nac_params
        _nac_params = nac_params
    elif is_nac and pathlib.Path("BORN").exists():
        if primitive is None:
            raise ValueError(
                "Primitive cell has to be specified when born_filename is given."
            )
        _nac_params = parse_BORN(primitive, filename="BORN")
        if log_level:
            print('NAC params were read from "BORN".')
    else:
        _nac_params = None

    if _nac_params and "factor" not in _nac_params and nac_factor is not None:
        _nac_params["factor"] = nac_factor

    return _nac_params


def read_force_constants_from_hdf5(
    filename: str | os.PathLike = "force_constants.hdf5",
    p2s_map: ArrayLike | None = None,
    calculator: str | None = None,
) -> NDArray:
    """Convert force constants physical unit.

    Each calculator interface has own default force constants physical unit.
    This method reads 'physical_unit' in force constants hdf5 file and
    if this is different from the one for 'calculator', the force constants
    are converted to have the physical unit of the calculator.

    Note
    ----
    This method is also used from phonopy script.

    """
    fc, fc_unit = read_force_constants_hdf5(
        filename=filename, p2s_map=p2s_map, return_physical_unit=True
    )
    if fc_unit is None:
        return fc
    else:
        factor = get_force_constant_conversion_factor(fc_unit, calculator)
        return fc * factor


def select_and_load_dataset(
    nsatom: int,
    dataset: dict | None = None,
    phonopy_yaml_filename: str | os.PathLike | typing.IO | None = None,
    force_sets_filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> dict | None:
    """Set displacement-force dataset."""
    _dataset = None
    _force_sets_filename = None
    if forces_in_dataset(dataset):
        _dataset = dataset
        if isinstance(phonopy_yaml_filename, (str, os.PathLike)):
            _force_sets_filename = phonopy_yaml_filename
    elif force_sets_filename is not None:
        _dataset = parse_FORCE_SETS(natom=nsatom, filename=force_sets_filename)
        _force_sets_filename = force_sets_filename
    elif pathlib.Path("FORCE_SETS").exists():
        _dataset = parse_FORCE_SETS(natom=nsatom)
        _force_sets_filename = "FORCE_SETS"
    else:
        _dataset = dataset
        if isinstance(phonopy_yaml_filename, (str, os.PathLike)):
            _force_sets_filename = phonopy_yaml_filename

    if log_level:
        if forces_in_dataset(_dataset):
            print(f'Displacement-force dataset was read from "{_force_sets_filename}".')
        elif _dataset is not None:
            print(f'Displacement dataset was read from "{_force_sets_filename}".')

    return _dataset


def select_and_extract_force_constants(
    phonon: Phonopy,
    phonopy_yaml_filename: str | os.PathLike | None = None,
    force_constants: NDArray | None = None,
    force_constants_filename: str | os.PathLike | None = None,
    is_compact_fc: bool = True,
    log_level: int = 0,
) -> NDArray | None:
    """Extract force constants.

    1. From fc (ndarray) in phonopy_yaml.
    2. From FORCE_CONSTANTS file or force_constants.hdf5 file.

    """
    _fc = None
    _force_constants_filename = None
    if force_constants is not None:  # 1
        _fc = force_constants
        _force_constants_filename = phonopy_yaml_filename
    elif force_constants_filename is not None:  # 2
        _fc = _read_force_constants_file(phonon, force_constants_filename)
        _force_constants_filename = force_constants_filename
    elif phonon.force_constants is None:
        # unless provided these from phonopy_yaml.
        if pathlib.Path("FORCE_CONSTANTS").exists():
            _fc = _read_force_constants_file(phonon, "FORCE_CONSTANTS")
            _force_constants_filename = "FORCE_CONSTANTS"
        elif pathlib.Path("force_constants.hdf5").exists():
            _fc = _read_force_constants_file(phonon, "force_constants.hdf5")
            _force_constants_filename = "force_constants.hdf5"

    if _fc is not None:
        if log_level and _force_constants_filename is not None:
            print(f'Force constants were read from "{_force_constants_filename}".')
        if not is_compact_fc and _fc.shape[0] != _fc.shape[1]:
            _fc = compact_fc_to_full_fc(phonon.primitive, _fc, log_level=log_level)
        elif is_compact_fc and _fc.shape[0] == _fc.shape[1]:
            _fc = full_fc_to_compact_fc(phonon.primitive, _fc, log_level=log_level)

    return _fc


def produce_force_constants(
    phonon: Phonopy,
    fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
    fc_calculator_options: str | None = None,
    symmetrize_fc: bool = True,
    is_compact_fc: bool = True,
    use_symfc_projector: bool = False,
    log_level: int = 0,
):
    """Produce force constants."""
    try:
        phonon.produce_force_constants(
            calculate_full_force_constants=(not is_compact_fc),
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
        )
        if symmetrize_fc:
            if fc_calculator is None:
                phonon.symmetrize_force_constants(
                    show_drift=True, use_symfc_projector=use_symfc_projector
                )
            elif fc_calculator == "traditional":
                phonon.symmetrize_force_constants(
                    show_drift=True, use_symfc_projector=False
                )
    except ForcesetsNotFoundError:
        if log_level:
            print("Displacement-force dataset was not found. ")


def check_nac_params(nac_params: dict, unitcell: PhonopyAtoms, pmat: np.ndarray):
    """Check number of Born effective charges."""
    borns = nac_params["born"]
    if len(borns) != np.rint(len(unitcell) * np.linalg.det(pmat)).astype(int):
        msg = "Number of Born effective charges is not consistent with the cell."
        raise ValueError(msg)


def develop_or_load_pypolymlp(
    phonon: Phonopy,
    mlp_params: str | dict | PypolymlpParams | None = None,
    mlp_filename: str | os.PathLike | None = None,
    log_level: int = 0,
):
    """Run pypolymlp to compute forces."""
    if log_level:
        print("-" * 29 + " pypolymlp start " + "-" * 30)
        _show_pypolymlp_header(mlp_params=mlp_params)

    _load_pypolymlp(phonon, mlp_filename=mlp_filename, log_level=log_level)

    if phonon.mlp is None:
        _develop_and_save_pypolymlp(
            phonon,
            mlp_params=mlp_params,
            mlp_filename=mlp_filename,
            log_level=log_level,
        )

    if log_level:
        print("-" * 30 + " pypolymlp end " + "-" * 31, flush=True)


def _show_pypolymlp_header(mlp_params: str | dict | PypolymlpParams | None = None):
    """Show pypolymlp header."""
    import pypolymlp

    print(f"Pypolymlp version {pypolymlp.__version__}")
    print("Pypolymlp is a generator of polynomial machine learning potentials.")
    print("Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).")
    print("Pypolymlp is developed at https://github.com/sekocha/pypolymlp.")


def _load_pypolymlp(
    phonon: Phonopy,
    mlp_filename: str | os.PathLike | None = None,
    log_level: int = 0,
):
    """Load MLPs from polymlp.yaml or phonopy.pmlp."""
    _mlp_filename = None
    if mlp_filename is None:
        for default_mlp_filename in ["polymlp.yaml", "phonopy.pmlp", "phono3py.pmlp"]:
            _mlp_filename_list = list(pathlib.Path().glob(f"{default_mlp_filename}*"))
            if _mlp_filename_list:
                _mlp_filename = _mlp_filename_list[0]
                if _mlp_filename.suffix not in [
                    ".yaml",
                    ".pmlp",
                    ".xz",
                    ".gz",
                    ".bz2",
                    "lzma",
                ]:
                    continue
                if log_level and "pmlp" in default_mlp_filename:
                    print(f'Loading MLPs from "{_mlp_filename}" is obsolete.')
                break
    else:
        _mlp_filename = mlp_filename

    if _mlp_filename is None:
        return None

    if log_level:
        print(f'Load MLPs from "{_mlp_filename}".')

    if not pathlib.Path(_mlp_filename).exists():
        raise PypolymlpFileNotFoundError(f'"{_mlp_filename}" is not found.')

    phonon.load_mlp(_mlp_filename)


def _develop_and_save_pypolymlp(
    phonon: Phonopy,
    mlp_params: str | dict | PypolymlpParams | None = None,
    mlp_filename: str | os.PathLike | None = None,
    log_level: int = 0,
):
    """Develop MLPs by pypolymlp and save them into polymlp.yaml."""
    if forces_in_dataset(phonon.mlp_dataset):
        if log_level:
            if mlp_params is None:
                pmlp_params = PypolymlpParams()
            else:
                pmlp_params = parse_mlp_params(mlp_params)
            print("Parameters:")
            for k, v in dataclasses.asdict(pmlp_params).items():
                if v is not None:
                    print(f"  {k}: {v}")
            print("Developing MLPs by pypolymlp...", flush=True)

        try:
            phonon.develop_mlp(params=mlp_params)
        except PypolymlpDevelopmentError as e:
            if log_level:
                print("-" * 30 + " pypolymlp end " + "-" * 31, flush=True)
            raise PypolymlpDevelopmentError(str(e)) from e

        if mlp_filename is None:
            _mlp_filename = "polymlp.yaml"
        else:
            _mlp_filename = mlp_filename
        phonon.save_mlp(filename=_mlp_filename)
        if log_level:
            print(f'MLPs were written into "{_mlp_filename}"', flush=True)
    else:
        raise PypolymlpTrainingDatasetNotFoundError(
            "Pypolymlp training dataset is not found."
        )


def prepare_dataset_by_pypolymlp(
    phonon: Phonopy,
    displacement_distance: float | None = None,
    number_of_snapshots: int | Literal["auto"] | None = None,
    rd_number_estimation_factor: float | None = None,
    random_seed: int | None = None,
    log_level: int = 0,
):
    """Generate displacements and evaluate forces by pypolymlp."""
    if displacement_distance is None:
        _displacement_distance = 0.01
    else:
        _displacement_distance = displacement_distance

    if log_level:
        if number_of_snapshots:
            print("Generate random displacements")
            print(
                "  Twice of number of snapshots will be generated "
                "for plus-minus displacements."
            )
        else:
            print("Generate displacements")
        print(
            f"  Displacement distance: {_displacement_distance:.5f}".rstrip("0").rstrip(
                "."
            )
        )
    phonon.generate_displacements(
        distance=_displacement_distance,
        is_plusminus=True,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
        number_estimation_factor=rd_number_estimation_factor,
    )
    assert phonon.supercells_with_displacements is not None

    if log_level and number_of_snapshots == "auto":
        print(
            "  Number of generated supercells with random displacements: "
            f"{len(phonon.supercells_with_displacements)}",
        )

    if log_level:
        print(
            f"Evaluate forces in {len(phonon.displacements)} supercells by pypolymlp",
            flush=True,
        )

    phonon.evaluate_mlp()


def _read_force_constants_file(phonon: Phonopy, force_constants_filename) -> NDArray:
    dot_split = force_constants_filename.split(".")
    p2s_map = phonon.primitive.p2s_map
    if len(dot_split) > 1 and dot_split[-1] == "hdf5":
        _fc = read_force_constants_from_hdf5(
            filename=force_constants_filename,
            p2s_map=p2s_map,
            calculator=phonon.calculator,
        )
    else:
        _fc = parse_FORCE_CONSTANTS(filename=force_constants_filename, p2s_map=p2s_map)

    return _fc


def _read_crystal_structure(filename=None, interface_mode=None):
    try:
        return read_crystal_structure(filename=filename, interface_mode=interface_mode)
    except FileNotFoundError:
        raise
    except Exception as exc:
        msg = [
            "============================ phonopy.load ============================",
            "  Reading crystal structure file failed in phonopy.load.",
            "  Maybe phonopy.load(..., calculator='<calculator name>') expected?",
            "============================ phonopy.load ============================",
        ]
        raise RuntimeError("\n".join(msg)) from exc
