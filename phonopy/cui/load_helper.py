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

import pathlib
from dataclasses import asdict
from typing import Optional, Union

import numpy as np

from phonopy import Phonopy
from phonopy.exception import ForcesetsNotFoundError
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
    supercell_matrix=None,
    primitive_matrix=None,
    unitcell=None,
    supercell=None,
    unitcell_filename=None,
    supercell_filename=None,
    calculator=None,
    symprec=1e-5,
    log_level=0,
):
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
    primitive: Optional[Primitive] = None,
    nac_params: Optional[dict] = None,
    born_filename: Optional[str] = None,
    is_nac: bool = True,
    nac_factor: Optional[float] = None,
    log_level: int = 0,
) -> dict:
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
        _nac_params = parse_BORN(primitive, filename=born_filename)
        if log_level:
            print('NAC parameters were read from "%s".' % born_filename)
    elif nac_params is not None:  # nac_params input or phonopy_yaml.nac_params
        _nac_params = nac_params
    elif is_nac and pathlib.Path("BORN").exists():
        _nac_params = parse_BORN(primitive, filename="BORN")
        if log_level:
            print('NAC params were read from "BORN".')
    else:
        _nac_params = None

    if _nac_params and "factor" not in _nac_params and nac_factor is not None:
        _nac_params["factor"] = nac_factor

    return _nac_params


def read_force_constants_from_hdf5(
    filename="force_constants.hdf5", p2s_map=None, calculator=None
):
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


def set_dataset_and_force_constants(
    phonon: Phonopy,
    dataset: Optional[dict],
    phonopy_yaml_filename: Optional[str] = None,
    fc: Optional[np.ndarray] = None,  # From phonopy_yaml
    force_constants_filename: Optional[str] = None,
    force_sets_filename: Optional[str] = None,
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[str] = None,
    produce_fc: bool = True,
    symmetrize_fc: bool = True,
    is_compact_fc: bool = True,
    use_pypolymlp: bool = False,
    mlp_params: Optional[dict] = None,
    displacement_distance: Optional[float] = None,
    number_of_snapshots: Optional[int] = None,
    random_seed: Optional[int] = None,
    evaluating_forces: bool = False,
    log_level: int = 0,
):
    """Set displacement-force dataset and force constants."""
    _set_dataset(
        phonon,
        dataset,
        phonopy_yaml_filename,
        force_sets_filename,
        log_level,
    )

    _set_force_constants(
        phonon,
        phonopy_yaml_filename,
        fc,
        force_constants_filename,
        is_compact_fc,
        log_level,
    )

    if use_pypolymlp:
        phonon.mlp_dataset = phonon.dataset
        phonon.dataset = None
        _run_pypolymlp(phonon, mlp_params, log_level=log_level)
        if evaluating_forces:
            _generate_displacements_and_forces_by_pypolymlp(
                phonon,
                displacement_distance=displacement_distance,
                number_of_snapshots=number_of_snapshots,
                random_seed=random_seed,
                log_level=log_level,
            )

    if (
        phonon.force_constants is None
        and produce_fc
        and forces_in_dataset(phonon.dataset)
    ):
        _produce_force_constants(
            phonon,
            fc_calculator,
            fc_calculator_options,
            symmetrize_fc,
            is_compact_fc,
            log_level,
        )


def check_nac_params(nac_params: dict, unitcell: PhonopyAtoms, pmat: np.ndarray):
    """Check number of Born effective charges."""
    borns = nac_params["born"]
    if len(borns) != np.rint(len(unitcell) * np.linalg.det(pmat)).astype(int):
        msg = "Number of Born effective charges is not consistent with the cell."
        raise ValueError(msg)


def _set_dataset(
    phonon: Phonopy,
    dataset: Optional[dict],
    phonopy_yaml_filename: Optional[str] = None,
    force_sets_filename: Optional[str] = None,
    log_level: int = 0,
):
    natom = len(phonon.supercell)
    _dataset = None
    _force_sets_filename = None
    if forces_in_dataset(dataset):
        _dataset = dataset
        _force_sets_filename = phonopy_yaml_filename
    elif force_sets_filename is not None:
        _dataset = parse_FORCE_SETS(natom=natom, filename=force_sets_filename)
        _force_sets_filename = force_sets_filename
    elif pathlib.Path("FORCE_SETS").exists():
        _dataset = parse_FORCE_SETS(natom=natom)
        _force_sets_filename = "FORCE_SETS"
    else:
        _dataset = dataset

    if log_level:
        if forces_in_dataset(_dataset):
            print(f'Displacement-force dataset was read from "{_force_sets_filename}".')
        elif _dataset is not None:
            print(f'Displacement dataset was read from "{_force_sets_filename}".')

    phonon.dataset = _dataset


def _run_pypolymlp(
    phonon: Phonopy,
    mlp_params: Union[str, dict, PypolymlpParams],
    mlp_filename: str = "phonopy.pmlp",
    log_level: int = 0,
):
    """Run pypolymlp to compute forces."""
    if log_level:
        print("-" * 29 + " pypolymlp start " + "-" * 30)
        print("Pypolymlp is a generator of polynomial machine learning potentials.")
        print("Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).")
        print("Pypolymlp is developed at https://github.com/sekocha/pypolymlp.")
        if mlp_params:
            print("Parameters:")
            for k, v in asdict(parse_mlp_params(mlp_params)).items():
                if v is not None:
                    print(f"  {k}: {v}")

    _mlp_filename_list = list(pathlib.Path().glob(f"{mlp_filename}*"))
    if _mlp_filename_list:
        _mlp_filename = _mlp_filename_list[0]
        if log_level:
            print(f'Load MLPs from "{_mlp_filename}".')
        phonon.load_mlp(_mlp_filename)
        phonon.mlp_dataset = None
    elif forces_in_dataset(phonon.mlp_dataset):
        if log_level:
            print("Developing MLPs by pypolymlp...", flush=True)
        phonon.develop_mlp(params=mlp_params)
        phonon.save_mlp(filename=mlp_filename)
        if log_level:
            print(f'MLPs were written into "{mlp_filename}"', flush=True)
    else:
        raise RuntimeError(f'"{mlp_filename}" is not found.')

    if log_level:
        print("-" * 30 + " pypolymlp end " + "-" * 31, flush=True)


def _generate_displacements_and_forces_by_pypolymlp(
    phonon: Phonopy,
    displacement_distance: Optional[float] = None,
    number_of_snapshots: Optional[int] = None,
    random_seed: Optional[int] = None,
    log_level: int = 0,
):
    if displacement_distance is None:
        _displacement_distance = 0.001
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
    )

    if log_level:
        print(
            f"Evaluate forces in {len(phonon.displacements)} supercells "
            "by pypolymlp",
            flush=True,
        )

    if phonon.supercells_with_displacements is None:
        raise RuntimeError("Displacements are not set. Run generate_displacements.")

    phonon.evaluate_mlp()


def _set_force_constants(
    phonon: Phonopy,
    phonopy_yaml_filename: Optional[str] = None,
    fc: Optional[np.ndarray] = None,  # From phonopy_yaml
    force_constants_filename: Optional[str] = None,
    is_compact_fc: bool = True,
    log_level: int = 0,
):
    _fc = None
    _force_constants_filename = None
    if fc is not None:
        _fc = fc
        _force_constants_filename = phonopy_yaml_filename
    elif force_constants_filename is not None:
        _fc = _read_force_constants_file(
            phonon,
            force_constants_filename,
            is_compact_fc=is_compact_fc,
            log_level=log_level,
        )
        _force_constants_filename = force_constants_filename
    elif phonon.force_constants is None:
        # unless provided these from phonopy_yaml.
        if pathlib.Path("FORCE_CONSTANTS").exists():
            _fc = _read_force_constants_file(
                phonon,
                "FORCE_CONSTANTS",
                is_compact_fc=is_compact_fc,
                log_level=log_level,
            )
            _force_constants_filename = "FORCE_CONSTANTS"
        elif pathlib.Path("force_constants.hdf5").exists():
            _fc = _read_force_constants_file(
                phonon,
                "force_constants.hdf5",
                is_compact_fc=is_compact_fc,
                log_level=log_level,
            )
            _force_constants_filename = "force_constants.hdf5"

    if _fc is not None:
        if not is_compact_fc and _fc.shape[0] != _fc.shape[1]:
            _fc = compact_fc_to_full_fc(phonon.primitive, _fc, log_level=log_level)
        elif is_compact_fc and _fc.shape[0] == _fc.shape[1]:
            _fc = full_fc_to_compact_fc(phonon.primitive, _fc, log_level=log_level)
        phonon.force_constants = _fc
        if log_level and _force_constants_filename is not None:
            print(f'Force constants were read from "{_force_constants_filename}".')


def _read_force_constants_file(
    phonon: Phonopy, force_constants_filename, is_compact_fc=True, log_level=0
):
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

    if is_compact_fc and _fc.shape[0] == _fc.shape[1]:
        _fc = full_fc_to_compact_fc(phonon.primitive, _fc, log_level=log_level)
    elif not is_compact_fc and _fc.shape[0] != _fc.shape[1]:
        _fc = compact_fc_to_full_fc(phonon.primitive, _fc, log_level=log_level)

    return _fc


def _produce_force_constants(
    phonon: Phonopy,
    fc_calculator: Optional[str],
    fc_calculator_options: Optional[str],
    symmetrize_fc: bool,
    is_compact_fc: bool,
    log_level: int,
):
    try:
        phonon.produce_force_constants(
            calculate_full_force_constants=(not is_compact_fc),
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
        )
        if symmetrize_fc:
            phonon.symmetrize_force_constants(show_drift=(log_level > 0))
            if log_level:
                print("Force constants were symmetrized.")
    except ForcesetsNotFoundError:
        if log_level:
            print("Displacement-force datast was not found. ")


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
