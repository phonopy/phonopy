"""Routines to collect crystal structure information."""

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
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy.cui.settings import Settings
from phonopy.exception import CellNotFoundError, MagmomValueError
from phonopy.file_IO import is_file_phonopy_yaml
from phonopy.interface.calculator import (
    get_default_cell_filename,
    read_crystal_structure,
)
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.vasp import read_vasp
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive_matrix_with_auto

_FallbackReason = Literal[
    "load_phonopy_yaml mode",
    "read_vasp parsing failed",
    "default file not found",
    "no supercell matrix given",
]


@dataclasses.dataclass
class CellInfoResult:
    """Dataclass to hold the result of collect_cell_info."""

    unitcell: PhonopyAtoms
    optional_structure_info: tuple
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None
    primitive_matrix: (
        Sequence[Sequence[float]]
        | NDArray[np.double]
        | Literal["P", "F", "I", "A", "C", "R", "auto"]
        | None
    ) = None
    interface_mode: str | None = None


@dataclasses.dataclass
class PhonopyCellInfoResult(CellInfoResult):
    """Phono3py cell info result.

    This is a subclass of CellInfoResult.

    """

    phonopy_yaml: PhonopyYaml | None = None


def get_cell_info(
    settings: Settings,
    cell_filename: str | os.PathLike | None,
    log_level: int = 0,
    load_phonopy_yaml: bool = False,
    phonopy_yaml_cls: type[PhonopyYaml] = PhonopyYaml,
    enforce_primitive_matrix_auto: bool = False,
) -> PhonopyCellInfoResult:
    """Return calculator interface and crystal structure information.

    CellNotFoundError and MagmomValueError can be raised by collect_cell_info
    and set_magnetic_moments functions, respectively.

    """
    cell_info = collect_cell_info(
        supercell_matrix=settings.supercell_matrix,
        primitive_matrix=settings.primitive_matrix,
        interface_mode=settings.calculator,
        cell_filename=cell_filename,
        chemical_symbols=settings.chemical_symbols,
        enforce_primitive_matrix_auto=enforce_primitive_matrix_auto,
        phonopy_yaml_cls=phonopy_yaml_cls,
        load_phonopy_yaml=load_phonopy_yaml,
    )

    # Show primitive matrix overwrite message
    phpy_yaml = cell_info.phonopy_yaml
    if phpy_yaml is not None:
        assert phpy_yaml.unitcell is not None
        yaml_filename = cell_info.optional_structure_info[0]
        pmat_in_settings = get_primitive_matrix_with_auto(
            phpy_yaml.unitcell, cell_info.primitive_matrix
        )
        if pmat_in_settings is None:
            pmat_in_settings = np.eye(3, dtype="double", order="C")
        pmat_in_phpy_yaml = get_primitive_matrix_with_auto(
            phpy_yaml.unitcell, phpy_yaml.primitive_matrix
        )
        if pmat_in_phpy_yaml is None:
            pmat_in_phpy_yaml = np.eye(3, dtype="double", order="C")
        if log_level and not np.allclose(
            pmat_in_phpy_yaml, pmat_in_settings, atol=1e-5
        ):
            if phpy_yaml.primitive_matrix is None:
                print(f'Primitive matrix is not specified in "{yaml_filename}".')
            else:
                print(f'Primitive matrix in "{yaml_filename}" is')
                for v in pmat_in_phpy_yaml:
                    print(f"  {v}")
            print("But it is overwritten by")
            for v in pmat_in_settings:
                print(f"  {v}")
            print("")

    set_magnetic_moments(cell_info.unitcell, settings.magnetic_moments, log_level)

    return cell_info


def collect_cell_info(
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None = None,
    interface_mode: str | None = None,
    cell_filename: str | os.PathLike | None = None,
    chemical_symbols: Sequence[str] | None = None,
    enforce_primitive_matrix_auto: bool = False,
    phonopy_yaml_cls: type[PhonopyYaml] = PhonopyYaml,
    load_phonopy_yaml: bool = False,
) -> PhonopyCellInfoResult:
    """Collect crystal structure information from input file and parameters.

    This function returns crystal structure information obtained from an input
    file and parameters. Although this function is convenient, this function
    tends to be complicated and error-prone since phonopy has to support various
    file formats.

    How crystal structure is recognized
    -----------------------------------
    Phonopy supports various crystal structure file formats. Most of them are
    those from force calculators (e.g., VASP, QE). To let phonopy know which
    calculator format is chosen, usually the calculator interface name is
    simultaneously given. Otherwise, it is considered as the default interface,
    i.e., VASP-like format. When a calculator interface is specified, phonopy
    goes into the calculator interface mode. In this calculator interface mode,
    when none of crystal structure is provided, the default file name for each
    calculator interface mode is assumed and the file name is searched in the
    current directory. The crystal structure information found in this way is
    recognized as the unit cell. The supercell and primitive matrices
    information are given by the parameters of this function.

    When srystal structure search explained above failed, phonopy.yaml like file
    is searched in the current directory. phonopy.yaml like file name can be
    specified as the input crystal structure. Since phonopy.yaml like file
    contains supercell and primitive cell matrices information, these
    parameter inputs of this function are ignored.

    Parameters
    ----------
    supercell_matrix : array_like or None
        3x3 transformation matrix or when it is a diagonal matrix, three
        diagonal elements. Default is None. See also shape_supercell_matrix.
    primitive_matrix : array_like, str, or None
        3x3 transformation matrix or a character representing centring or None.
        Default is None. See also get_primitive_matrix.
    interface_mode : str or None
        Force calculator or crystal structure format name.
    cell_filename : str or None
        Input cell filename.
    chemical_symbols : list of str or None
        List of chemical symbols or unit cell.
    enforce_primitive_matrix_auto : bool
        Enforce primitive_matrix='auto' when True. Default is False.
    phonopy_yaml_cls : Class object, optional
        PhonopyYaml like class name. This is used to return its instance when
        needed. Default is None, which means PhonopyYaml class type. This can be
        Phono3pyYaml class type. Default is None.
    load_phonopy_yaml : bool
        True means phonopy-load mode.

    """
    fallback_reason = _compute_fallback_reason(
        supercell_matrix, interface_mode, cell_filename, load_phonopy_yaml
    )
    _interface_mode, _cell_filename = _resolve_interface_and_filename(
        fallback_reason, interface_mode, cell_filename, phonopy_yaml_cls
    )

    unitcell, optional_structure_info = read_crystal_structure(
        filename=_cell_filename,
        interface_mode=_interface_mode,
        chemical_symbols=chemical_symbols,
        phonopy_yaml_cls=phonopy_yaml_cls,
    )

    if unitcell is None:
        err_msg = _get_error_message(
            optional_structure_info,
            fallback_reason,
            cell_filename,
            phonopy_yaml_cls,
        )
        raise CellNotFoundError(err_msg)

    interface_mode_out, supercell_matrix_out, primitive_matrix_out = (
        _resolve_cell_parameters(
            _interface_mode,
            optional_structure_info,
            interface_mode,
            supercell_matrix,
            primitive_matrix,
        )
    )

    _validate_cell(
        unitcell,
        supercell_matrix_out,
        _interface_mode,
        optional_structure_info,
        phonopy_yaml_cls,
        _cell_filename,
        interface_mode_out,
    )

    if enforce_primitive_matrix_auto:
        primitive_matrix_out = "auto"

    phpy_yaml = (
        optional_structure_info[1] if _interface_mode == "phonopy_yaml" else None
    )

    return PhonopyCellInfoResult(
        unitcell=unitcell,
        supercell_matrix=supercell_matrix_out,
        primitive_matrix=primitive_matrix_out,
        optional_structure_info=optional_structure_info,
        interface_mode=interface_mode_out,
        phonopy_yaml=phpy_yaml,
    )


def set_magnetic_moments(
    unitcell: PhonopyAtoms, magnetic_moments: Sequence | None, log_level: int
) -> None:
    """Set magnetic moments to unitcell."""
    # Set magnetic moments
    magmoms = magnetic_moments
    if magmoms is not None:
        if len(magmoms) in (len(unitcell), len(unitcell) * 3):
            unitcell.magnetic_moments = magmoms
        else:
            raise MagmomValueError("Invalid MAGMOM setting")


def _compute_fallback_reason(
    supercell_matrix: Sequence[Sequence[int]] | NDArray | None,
    interface_mode: str | None,
    cell_filename: str | os.PathLike | None,
    load_phonopy_yaml: bool,
) -> _FallbackReason | None:
    """Return the reason for falling back to phonopy_yaml mode, or None."""
    if load_phonopy_yaml:
        return "load_phonopy_yaml mode"
    return _fallback_to_phonopy_yaml(supercell_matrix, interface_mode, cell_filename)


def _resolve_interface_and_filename(
    fallback_reason: _FallbackReason | None,
    interface_mode: str | None,
    cell_filename: str | os.PathLike | None,
    phonopy_yaml_cls: type[PhonopyYaml],
) -> tuple[str | None, str | os.PathLike | None]:
    """Resolve interface mode and cell filename from fallback state."""
    if fallback_reason is not None:
        _interface_mode = "phonopy_yaml"
        keyword = phonopy_yaml_cls.command_name
        if cell_filename is None:
            _cell_filename = None
        elif is_file_phonopy_yaml(cell_filename, yaml_dict_keys=[keyword]):
            # Readable as a phonopy.yaml
            _cell_filename = cell_filename
        elif is_file_phonopy_yaml(cell_filename):
            # Readable as yaml
            _cell_filename = cell_filename
        else:
            # Not readable as yaml. Proceed to look for default file names.
            _cell_filename = None
    elif interface_mode is None:
        _interface_mode = None
        _cell_filename = cell_filename
    else:
        _interface_mode = interface_mode.lower()
        _cell_filename = cell_filename

    return _interface_mode, _cell_filename


def _validate_cell(
    unitcell: PhonopyAtoms,
    supercell_matrix_out: Sequence[Sequence[int]] | NDArray[np.int64] | None,
    resolved_interface_mode: str | None,
    optional_structure_info: tuple,
    phonopy_yaml_cls: type[PhonopyYaml],
    _cell_filename: str | os.PathLike | None,
    interface_mode_out: str | None,
) -> None:
    """Validate the crystal cell parameters and raise CellNotFoundError if invalid."""
    err_msg = []
    unitcell_filename = optional_structure_info[0]
    if supercell_matrix_out is None:
        if resolved_interface_mode == "phonopy_yaml":
            err_msg.append(f"'supercell_matrix' not found  in \"{unitcell_filename}\".")
        else:
            err_msg.append("Supercell matrix information (DIM or --dim) was not found.")
        if _cell_filename is None and (
            unitcell_filename == get_default_cell_filename(interface_mode_out)
        ):
            err_msg += [
                "",
                "Phonopy read the crystal structure from the file having the default "
                "filename ",
                "of each calculator. In this case, supercell matrix has to be "
                "specified.",
                "Because this is the old style way of using %s,"
                % phonopy_yaml_cls.command_name,
            ]
            filenames = [
                '"%s"' % name for name in phonopy_yaml_cls.default_filenames[:-1]
            ]
            err_msg += [
                '"%s" was read being preferred to files such as ' % unitcell_filename,
                '%s, or "%s".'
                % (", ".join(filenames), phonopy_yaml_cls.default_filenames[-1]),
            ]
            err_msg += [
                "",
                'If crystal structure is expected to be read from some "*.yaml" file,',
                'Please rename "%s" to something else.' % unitcell_filename,
            ]
    if np.linalg.det(unitcell.cell) < 0.0:
        err_msg.append("Lattice vectors have to follow the right-hand rule.")
    if err_msg:
        raise CellNotFoundError(
            'Crystal structure was read from "%s".\n' % unitcell_filename
            + "\n".join(err_msg)
        )


def _fallback_to_phonopy_yaml(
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None,
    interface_mode: str | None,
    cell_filename: str | os.PathLike | None,
) -> _FallbackReason | None:
    """Find possibility to fallback to phonopy.yaml mode.

    Fallback happens in any of the following cases.

    1. Parsing crystal structure file in the VASP POSCAR-style failed
    2. Default VASP POSCAR-style file is not found.
    3. supercell_matrix is not given along with (1) or (2).

    Parameters
    ----------
    supercell_matrix : array_like or None
        None is given when phonopy.yaml mode is expected.
    interface_mode : str or None
        None is the default mode, i.e., VASP like.
    cell_filename : str or None
        File name of VASP style crystal structure. None means the default
        file name, "POSCAR".

    Returns
    -------
    fallback_reason : str or None
        This provides information how to handle after the fallback.
        None means fallback to phonopy.yaml mode will not happen.

    """
    fallback_reason: _FallbackReason | None = None

    if interface_mode is None:
        fallback_reason = _poscar_failed(cell_filename)

    if fallback_reason is not None:
        if supercell_matrix is None:
            fallback_reason = "no supercell matrix given"

    return fallback_reason


def _poscar_failed(cell_filename: str | os.PathLike | None) -> _FallbackReason | None:
    """Determine if fall back happens.

    1) read_vasp (parsing POSCAR-style file) is failed. --> fallback

    ValueError is raised by read_vasp when the POSCAR-style format
    is broken. By this way, we assume the input crystal structure
    is not in the POSCAR-style format and is in the phonopy.yaml
    type.

    2) The file given by get_default_cell_filename('vasp') is not
       found at the current directory.  --> fallback

    This is the trigger to look for the phonopy.yaml type file.

    3) The given file with cell_filename is not found.  --> not fallback

    This case will not invoke phonopy.yaml mode and here nothing
    is done, i.e., fallback_reason = None.
    This error will be caught in the following part again be
    handled properly (read_crystal_structure).

    """
    fallback_reason: _FallbackReason | None = None
    try:
        if cell_filename is None:
            read_vasp(get_default_cell_filename("vasp"))
        else:
            read_vasp(cell_filename)
    except ValueError:
        # (1) see above
        fallback_reason = "read_vasp parsing failed"
    except FileNotFoundError:
        if cell_filename is None:
            # (2) see above
            fallback_reason = "default file not found"
        else:
            # (3) see above
            pass
    return fallback_reason


def _resolve_cell_parameters(
    resolved_interface_mode: str | None,
    optional_structure_info: tuple,
    interface_mode: str | None,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None = None,
) -> tuple[
    str | None,
    Sequence[Sequence[int]] | NDArray[np.int64] | None,
    Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None,
]:
    if (
        resolved_interface_mode == "phonopy_yaml"
        and optional_structure_info[1] is not None
    ):
        phpy: PhonopyYaml = optional_structure_info[1]
        interface_mode_out = (
            phpy.calculator if phpy.calculator is not None else interface_mode
        )
        supercell_matrix_out = (
            phpy.supercell_matrix
            if phpy.supercell_matrix is not None
            else supercell_matrix
        )
        primitive_matrix_out = (
            primitive_matrix if primitive_matrix is not None else phpy.primitive_matrix
        )
    else:
        interface_mode_out = resolved_interface_mode
        supercell_matrix_out = supercell_matrix
        primitive_matrix_out = primitive_matrix

    return interface_mode_out, supercell_matrix_out, primitive_matrix_out


def _get_error_message(
    optional_structure_info: tuple,
    fallback_reason: _FallbackReason | None,
    cell_filename: str | os.PathLike | None,
    phonopy_yaml_cls: type[PhonopyYaml],
) -> str:
    """Show error message for failure of getting crystal structure."""
    final_cell_filename = optional_structure_info[0]

    # No fallback to phonopy_yaml mode.
    if fallback_reason is None:
        msg_list = []
        if cell_filename is None:
            msg_list += [
                "Crystal structure file was not specified.",
                "Tried to find default crystal structure file.",
            ]
        msg_list.append(
            f'Crystal structure file "{final_cell_filename}" was not found.'
        )
        return "\n".join(msg_list)

    ####################################
    # Must be phonopy_yaml mode below. #
    ####################################

    msg_list = []
    if fallback_reason in ["default file not found", "read_vasp parsing failed"]:
        if cell_filename:
            vasp_filename = cell_filename
        else:
            vasp_filename = get_default_cell_filename("vasp")

        if fallback_reason == "read_vasp parsing failed":
            msg_list += [
                f'Parsing crystal structure file "{vasp_filename}" '
                "as in VASP format failed.",
                "(Calculator option is needed for parsing different crystal "
                "structure format.)",
            ]
        else:
            msg_list.append(f'Crystal structure file "{vasp_filename}" was not found.')

    elif fallback_reason == "no supercell matrix given":
        msg_list.append("Supercell matrix (DIM or --dim) was not explicitly specified.")

    msg_list.append(f"Switched on {phonopy_yaml_cls.command_name}-yaml mode.")

    if final_cell_filename is None:  # No phonopy*.yaml file was found.
        filenames = [f'"{name}"' for name in phonopy_yaml_cls.default_filenames]
        if len(filenames) == 1:
            text = filenames[0]
        elif len(filenames) == 2:
            text = " or ".join(filenames)
        else:
            tail = " or ".join(filenames[-2:])
            head = ", ".join(filenames[:-2])
            text = head + ", " + tail
        msg_list.append(f"{text} could not be found.")
        return "\n".join(msg_list)

    phpy = optional_structure_info[1]
    if phpy is None:  # Failed to parse phonopy*.yaml.
        msg_list.append(f'But parsing "{final_cell_filename}" failed.')

    return "\n".join(msg_list)
