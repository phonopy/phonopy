"""Phonopy command user interface."""

# Copyright (C) 2020 Atsushi Togo
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

import datetime
import os
import pathlib
import sys
from typing import Optional, Union

import numpy as np
import spglib

from phonopy import Phonopy, __version__
from phonopy.cui.collect_cell_info import collect_cell_info
from phonopy.cui.create_force_sets import create_FORCE_SETS
from phonopy.cui.load_helper import (
    get_nac_params,
    read_force_constants_from_hdf5,
    set_dataset_and_force_constants,
)
from phonopy.cui.phonopy_argparse import get_parser, show_deprecated_option_warnings
from phonopy.cui.settings import PhonopyConfParser, PhonopySettings
from phonopy.cui.show_symmetry import check_symmetry
from phonopy.file_IO import (
    get_born_parameters,
    get_supported_file_extensions_for_compression,
    is_file_phonopy_yaml,
    parse_FORCE_CONSTANTS,
    parse_FORCE_SETS,
    parse_QPOINTS,
    write_FORCE_CONSTANTS,
    write_force_constants_to_hdf5,
)
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixNAC
from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
    full_fc_to_compact_fc,
)
from phonopy.interface.calculator import (
    get_default_displacement_distance,
    get_default_physical_units,
    write_supercells_with_displacements,
)
from phonopy.interface.fc_calculator import fc_calculator_names
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.vasp import create_FORCE_CONSTANTS
from phonopy.phonon.band_structure import get_band_qpoints, get_band_qpoints_by_seekpath
from phonopy.phonon.dos import get_pdos_indices
from phonopy.sscha.core import MLPSSCHA
from phonopy.structure.atoms import atom_data, symbol_map
from phonopy.structure.cells import (
    get_primitive_matrix,
    guess_primitive_matrix,
    print_cell,
)
from phonopy.structure.cells import isclose as cells_isclose
from phonopy.structure.dataset import forces_in_dataset
from phonopy.units import THzToEv


# AA is created at http://www.network-science.de/ascii/ with standard.
def _print_phonopy():
    """Show phonopy logo."""
    print(
        r"""        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/"""
    )
    print_version(__version__)
    print_time()


def _print_phonopy_end():
    print_time()
    print_end()


def print_version(version, package_name="phonopy", rjust_length=44):
    """Show phonopy version number."""
    try:
        version_text = version.rjust(rjust_length)
        import importlib.metadata

        if importlib.metadata.version(package_name):
            ver = importlib.metadata.version(package_name).split(".")
            if len(ver) > 3:
                rev = ver[3]
                version_text = ("%s-%s" % (version, rev)).rjust(44)
    except (ImportError, importlib.metadata.PackageNotFoundError):
        pass
    finally:
        print(version_text)
        print("")


def print_end():
    """Show end banner."""
    print(
        r"""                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
"""
    )


def print_error():
    """Show error banner."""
    print(
        r"""  ___ _ __ _ __ ___  _ __
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |
 \___|_|  |_|  \___/|_|
"""
    )


def print_warning():
    """Show warning banner."""
    print(
        r"""                          _
__      ____ _ _ __ _ __ (_)_ __   __ _
\ \ /\ / / _` | '__| '_ \| | '_ \ / _` |
 \ V  V / (_| | |  | | | | | | | | (_| |
  \_/\_/ \__,_|_|  |_| |_|_|_| |_|\__, |
                                  |___/
"""
    )


def print_attention(attention_text):
    """Show attentinal information."""
    print("*" * 67)
    print(attention_text)
    print("*" * 67)
    print("")


def print_error_message(message):
    """Show error message."""
    print("")
    print(message)


def print_time():
    """Print current time."""
    print(
        "-------------------------"
        f'[time {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'
        "-------------------------"
    )


def file_exists(
    filename: Union[str, os.PathLike],
    log_level: int = 0,
    is_any: bool = False,
    check_file_extensions: bool = False,
) -> Optional[str]:
    """Check existence of file.

    Parameters
    ----------
    is_any : bool
        When False, the error message is shown and the program is terminated.
    check_file_extensions : bool
        By True, the existence of files having file
        extensions of compression extensions is checked.

    Returns
    -------
    str or None
        If the file exists, the filename is returned. Otherwise, None is
        returned.

    """
    if check_file_extensions:
        file_extensions = get_supported_file_extensions_for_compression()
    else:
        file_extensions = [""]
    for ext in file_extensions:
        _filename = pathlib.Path(filename) / ext
        if _filename.exists():
            return str(_filename)

    if is_any:
        return None
    else:
        error_text = '"%s" was not found.' % filename
        print_error_message(error_text)
        if log_level > 0:
            print_error()
        sys.exit(1)


def files_exist(
    filename_list: list[Union[str, os.PathLike]],
    log_level: int = 0,
    is_any: bool = False,
    check_file_extensions: bool = True,
) -> list[str]:
    """Check existence of files.

    Parameters
    ----------
    check_file_extensions : bool
        By True, the existence of files having file extensions of compression
        extensions is checked.

    Returns
    -------
    list[str]
        If the files exist, a list of the filenames that are found in
        considiering file extensions is returned.

    """
    filenames = []
    for filename in filename_list:
        _filename = file_exists(
            filename,
            log_level=log_level,
            is_any=is_any,
            check_file_extensions=check_file_extensions,
        )
        if _filename is not None:
            filenames.append(_filename)
            break

    if filenames:
        return filenames

    all_filenames = ", ".join(['"%s"' % fn for fn in filename_list[:-1]])
    all_filenames += ' or "%s"' % filename_list[-1]
    error_text = "Any of %s was not found." % all_filenames
    print_error_message(error_text)
    if log_level > 0:
        print_error()
    sys.exit(1)


def _finalize_phonopy(
    log_level,
    settings: PhonopySettings,
    confs,
    phonon: Phonopy,
    filename="phonopy.yaml",
):
    """Finalize phonopy."""
    units = get_default_physical_units(phonon.calculator)

    if phonon.mlp_dataset is not None:
        mlp_eval_filename = "phonopy_mlp_eval_dataset.yaml"
        if log_level:
            print(
                f'Dataset generated using MMLPs was written in "{mlp_eval_filename}".'
            )
        phonon.save(mlp_eval_filename)

    if settings.save_params:
        exists_fc_only = (
            not forces_in_dataset(phonon.dataset) and phonon.force_constants is not None
        )
        yaml_settings = {
            "displacements": not exists_fc_only,
            "force_sets": not exists_fc_only,
            "force_constants": exists_fc_only,
            "born_effective_charge": True,
            "dielectric_constant": True,
        }
        _filename = "phonopy_params.yaml"
    else:
        yaml_settings = {
            "force_sets": settings.include_force_sets,
            "force_constants": settings.include_force_constants,
            "born_effective_charge": settings.include_nac_params,
            "dielectric_constant": settings.include_nac_params,
            "displacements": settings.include_displacements,
        }
        _filename = filename

    phpy_yaml = PhonopyYaml(
        configuration=confs, physical_units=units, settings=yaml_settings
    )
    phpy_yaml.set_phonon_info(phonon)
    with open(_filename, "w") as w:
        w.write(str(phpy_yaml))

    if log_level > 0:
        print("")
        if settings.save_params:
            print(
                "Summary of calculation and parameters were written "
                'in "%s".' % _filename
            )
        else:
            print('Summary of calculation was written in "%s".' % _filename)
        _print_phonopy_end()
    sys.exit(0)


def _print_cells(phonon: Phonopy):
    """Show cells."""
    supercell = phonon.supercell
    unitcell = phonon.unitcell
    primitive = phonon.primitive
    p2p_map = primitive.p2p_map
    mapping = np.array([p2p_map[x] for x in primitive.s2p_map], dtype="intc")
    s_indep_atoms = phonon.symmetry.get_independent_atoms()
    p_indep_atoms = mapping[s_indep_atoms]
    u2s_map = supercell.u2s_map
    print("-" * 30 + " primitive cell " + "-" * 30)
    print_cell(primitive, stars=p_indep_atoms)
    print("-" * 32 + " unit cell " + "-" * 33)  # 32 + 11 + 33 = 76
    u2u_map = supercell.u2u_map
    u_indep_atoms = [u2u_map[x] for x in s_indep_atoms]
    print_cell(unitcell, mapping=mapping[u2s_map], stars=u_indep_atoms)
    print("-" * 32 + " super cell " + "-" * 32)
    print_cell(supercell, mapping=mapping, stars=s_indep_atoms)
    print("-" * 76)


def _print_settings(
    settings: PhonopySettings,
    phonon: Phonopy,
    is_primitive_axes_auto: bool,
    unitcell_filename: str,
    load_phonopy_yaml: bool,
):
    """Show setting info."""
    primitive_matrix = phonon.primitive_matrix
    supercell_matrix = phonon.supercell_matrix
    interface_mode = phonon.calculator
    run_mode = settings.run_mode
    if interface_mode:
        print(f"Calculator interface: {interface_mode}")
    print(f'Crystal structure was read from "{unitcell_filename}".')
    if (
        settings.cell_filename is not None
        and settings.cell_filename != unitcell_filename
    ):
        print(f'("{settings.cell_filename}" was not used though specified.)')
    physical_units = get_default_physical_units(interface_mode)
    print("Unit of length: %s" % physical_units["length_unit"])
    if _is_band_auto(settings) and not is_primitive_axes_auto:
        print(
            "Automatic band structure mode forced automatic choice "
            "of primitive axes."
        )
    if run_mode == "band":
        if _is_band_auto(settings):
            print("Band structure mode (Auto)")
        else:
            print("Band structure mode")
    if run_mode == "mesh":
        print("Mesh sampling mode")
    if run_mode == "band_mesh":
        print("Band structure and mesh sampling mode")
    if run_mode == "anime":
        print("Animation mode")
    if run_mode == "modulation":
        print("Modulation mode")
    if run_mode == "irreps":
        print("Ir-representation mode")
    if run_mode == "qpoints":
        if settings.write_dynamical_matrices:
            print("QPOINTS mode (dynamical matrices written out)")
        else:
            print("QPOINTS mode")
    if (
        run_mode == "band" or run_mode == "mesh" or run_mode == "qpoints"
    ) and settings.is_group_velocity:  # noqa 129
        gv_delta_q = settings.group_velocity_delta_q
        if gv_delta_q is not None:
            print("  With group velocity calculation (dq=%3.1e)" % gv_delta_q)
        else:
            print("")
    if settings.create_displacements or settings.random_displacements:
        print("Displacements creation mode")
        if not settings.is_plusminus_displacement == "auto":
            if settings.is_plusminus_displacement:
                print("  Plus Minus displacement: full plus minus directions")
            else:
                print("  Plus Minus displacement: only one direction")
        if not settings.is_diagonal_displacement:
            print("  Diagonal displacement: off")

        if settings.random_displacements is not None:
            print(
                "  Number of supercells with random displacements: %d"
                % settings.random_displacements
            )
            if settings.random_displacement_temperature is not None:
                print(
                    "  Temperatuere to generate random displacements: "
                    f"{settings.random_displacement_temperature}"
                )
            else:
                if settings.displacement_distance_max is None:
                    if settings.displacement_distance is not None:
                        print(
                            f"  Displacement distance: {settings.displacement_distance}"
                        )
                else:
                    if settings.displacement_distance is not None:
                        print(
                            "  Min displacement distance: "
                            f"{settings.displacement_distance}"
                        )
                    print(
                        "  Max displacement distance: "
                        f"{settings.displacement_distance_max}"
                    )
            if settings.random_seed is not None:
                print("  Random seed: %d" % settings.random_seed)
        elif settings.displacement_distance is not None:
            print(f"Displacement distance: {settings.displacement_distance}")

    print("Settings:")
    if load_phonopy_yaml:
        if not settings.is_nac:
            print("  Non-analytical term correction (NAC): off")
    else:
        if settings.is_nac:
            print("  Non-analytical term correction (NAC): on")
            if settings.nac_q_direction is not None:
                print("  NAC q-direction: %s" % settings.nac_q_direction)
    if settings.fc_spg_symmetry:
        print("  Enforce space group symmetry to force constants: on")
    if load_phonopy_yaml:
        if not settings.fc_symmetry:
            print("  Force constants symmetrization: off")
    else:
        if settings.fc_symmetry:
            print("  Force constants symmetrization: on")
    if settings.symmetry_tolerance is not None:
        print("  Symmetry tolerance: %5.2e" % settings.symmetry_tolerance)
    if run_mode == "mesh" or run_mode == "band_mesh":
        mesh = settings.mesh_numbers
        if isinstance(mesh, float):
            print("  Length for sampling mesh: %.1f" % mesh)
        elif mesh is not None:
            print("  Sampling mesh: %s" % np.array(mesh))
        if settings.is_thermal_properties:
            cutoff_freq = settings.cutoff_frequency
            if cutoff_freq is None:
                pass
            else:
                print(
                    "  - Phonon frequencies > %5.3f are used to calculate "
                    "thermal properties." % cutoff_freq
                )
            if settings.classical:
                print(
                    "  - Classical statistics are used to calculate"
                    " thermodynamic properties."
                )
        elif (
            settings.is_thermal_displacements
            or settings.is_thermal_displacement_matrices
        ):
            fmin = settings.min_frequency
            fmax = settings.max_frequency
            text = None
            if (fmin is not None) and (fmax is not None):
                text = "  - Phonon frequency from %5.3f to %5.3f" % (fmin, fmax)
                text += " are used to calculate\n"
                text += "    thermal displacements."
            elif (fmin is None) and (fmax is not None):
                text = "Phonon frequency < %5.3f" % fmax
                text = "  - %s are used to calculate thermal displacements." % text
            elif (fmin is not None) and (fmax is None):
                text = "Phonon frequency > %5.3f" % fmin
                text = "  - %s are used to calculate thermal displacements." % text
            if text:
                print(text)
    if (np.diag(np.diag(supercell_matrix)) - supercell_matrix).any():
        print("  Supercell matrix:")
        for v in supercell_matrix:
            print("    %s" % v)
    else:
        print("  Supercell: %s" % np.diag(supercell_matrix))
    if is_primitive_axes_auto or _is_band_auto(settings):
        print("  Primitive matrix (Auto):")
        for v in primitive_matrix:
            print("    %s" % v)
    elif primitive_matrix is not None:
        print("  Primitive matrix:")
        for v in primitive_matrix:
            print("    %s" % v)


def _write_displacements_files_then_exit(
    phonon: Phonopy,
    settings: PhonopySettings,
    confs: dict,
    optional_structure_info: Optional[tuple],
    log_level: int,
    disp_filename: str = "phonopy_disp.yaml",
):
    """Write supercells with displacements and displacement dataset.

    Note
    ----
    From phonopy v1.15.0, displacement dataset is written into
    phonopy_disp.yaml.

    """
    cells_with_disps = phonon.supercells_with_displacements
    additional_info = {"supercell_matrix": phonon.supercell_matrix}
    write_supercells_with_displacements(
        phonon.calculator,
        phonon.supercell,
        cells_with_disps,
        optional_structure_info=optional_structure_info,
        additional_info=additional_info,
    )

    if log_level > 0:
        print('"phonopy_disp.yaml" and supercells have been created.')

    settings.set_include_displacements(True)
    settings.set_include_nac_params(True)
    _finalize_phonopy(log_level, settings, confs, phonon, filename=disp_filename)


def _create_FORCE_SETS_from_settings(
    settings: PhonopySettings,
    cell_filename: Optional[str],
    symprec: float,
    log_level: int,
):
    """Create FORCE_SETS."""
    if settings.create_force_sets:
        filenames = settings.create_force_sets
        force_sets_zero_mode = False
    elif settings.create_force_sets_zero:
        filenames = settings.create_force_sets_zero
        force_sets_zero_mode = True
    else:
        print_error_message("Something wrong for parsing arguments.")
        sys.exit(0)

    if cell_filename is None:
        disp_filename_candidates = []
    else:
        disp_filename_candidates = [cell_filename]
    disp_filename_candidates += ["phonopy_disp.yaml", "disp.yaml"]

    disp_filenames = files_exist(
        disp_filename_candidates, log_level=log_level, is_any=True
    )
    disp_filename = disp_filenames[0]

    interface_mode = settings.calculator
    phpy_yaml = None
    if disp_filename != "disp.yaml":
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(disp_filename)
        if phpy_yaml.calculator is not None:
            interface_mode = phpy_yaml.calculator  # overwrite interface_mode
        physical_units = get_default_physical_units(interface_mode)
        if phpy_yaml.physical_units is None or all(
            [
                physical_units.get(key, None) == val
                for key, val in phpy_yaml.physical_units.items()
            ]
        ):
            phpy_yaml.physical_units = physical_units

    files_exist(filenames, log_level=log_level)
    create_FORCE_SETS(
        interface_mode,
        filenames,
        phpy_yaml=phpy_yaml,
        symmetry_tolerance=symprec,
        force_sets_zero_mode=force_sets_zero_mode,
        disp_filename=disp_filename,
        save_params=settings.save_params,
        log_level=log_level,
    )


def _produce_force_constants_load_phonopy_yaml(
    phonon: Phonopy,
    settings: PhonopySettings,
    phpy_yaml: PhonopyYaml,
    unitcell_filename: str,
    log_level: int,
):
    is_full_fc = settings.fc_spg_symmetry or settings.is_full_fc
    (fc_calculator, fc_calculator_options) = _get_fc_calculator_params(settings)

    try:
        set_dataset_and_force_constants(
            phonon,
            phpy_yaml.dataset,
            phonopy_yaml_filename=unitcell_filename,
            fc=phpy_yaml.force_constants,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            produce_fc=True,
            symmetrize_fc=False,
            is_compact_fc=(not is_full_fc),
            use_pypolymlp=settings.use_pypolymlp,
            mlp_params=settings.mlp_params,
            displacement_distance=settings.displacement_distance,
            number_of_snapshots=settings.random_displacements,
            random_seed=settings.random_seed,
            evaluating_forces=not settings.sscha_iterations,
            log_level=log_level,
        )
    except (RuntimeError, ValueError) as e:
        print_error_message(str(e))
        if log_level:
            print_error()
        sys.exit(1)


def _produce_force_constants(
    phonon: Phonopy,
    settings: PhonopySettings,
    phpy_yaml: dict,
    unitcell_filename: str,
    log_level: int,
):
    """Run force constants calculation (non-phonopy-yaml mode)."""
    num_satom = len(phonon.supercell)
    is_full_fc = settings.fc_spg_symmetry or settings.is_full_fc

    if settings.read_force_constants:
        _read_force_constants_from_file(
            settings, phonon, unitcell_filename, is_full_fc, log_level
        )
    else:

        def read_force_sets_from_phonopy_yaml(phpy_yaml):
            if phpy_yaml.dataset is not None and (
                "forces" in phpy_yaml.dataset
                or (
                    "first_atoms" in phpy_yaml.dataset
                    and "forces" in phpy_yaml.dataset["first_atoms"][0]
                )
            ):
                return phpy_yaml.dataset
            else:
                return None

        force_sets = None

        if phpy_yaml is not None:
            force_sets = read_force_sets_from_phonopy_yaml(phpy_yaml)
            if log_level:
                if force_sets is None:
                    print(f'Force sets were not found in "{unitcell_filename}".')
                else:
                    print(
                        'Forces and displacements were read from "%s".'
                        % unitcell_filename
                    )

        if force_sets is None:
            file_exists("FORCE_SETS", log_level=log_level)
            force_sets = parse_FORCE_SETS(natom=num_satom)
            if log_level:
                print('Forces and displacements were read from "%s".' % "FORCE_SETS")

        if log_level and force_sets is not None and "displacements" in force_sets:
            print("%d snapshots were found." % len(force_sets["displacements"]))

        if "natom" in force_sets:
            natom = force_sets["natom"]
        else:
            natom = force_sets["forces"].shape[1]
        if natom != num_satom:
            error_text = "Number of atoms in supercell is not consistent with "
            error_text += "the data in FORCE_SETS.\n"
            error_text += (
                "Please carefully check DIM, FORCE_SETS," " and %s"
            ) % unitcell_filename
            print_error_message(error_text)
            if log_level:
                print_error()
            sys.exit(1)

        (fc_calculator, fc_calculator_options) = _get_fc_calculator_params(
            settings, load_phonopy_yaml=False
        )

        phonon.dataset = force_sets
        if log_level:
            if fc_calculator is not None:
                print(
                    "Force constants calculation using "
                    f"{fc_calculator_names[fc_calculator]} starts."
                )
            else:
                print("Computing force constants...")

        try:
            if is_full_fc:
                # Need to calculate full force constant tensors
                phonon.produce_force_constants(
                    fc_calculator=fc_calculator,
                    fc_calculator_options=fc_calculator_options,
                )
            else:
                # Only force constants between atoms in primitive cell and
                # supercell
                phonon.produce_force_constants(
                    calculate_full_force_constants=False,
                    fc_calculator=fc_calculator,
                    fc_calculator_options=fc_calculator_options,
                )
        except RuntimeError as e:
            print_error_message(str(e))
            if log_level:
                print_error()
            sys.exit(1)


def _read_force_constants_from_file(
    settings,
    phonon: Phonopy,
    unitcell_filename: str,
    is_full_fc: bool,
    log_level: Union[bool, int],
):
    num_satom = len(phonon.supercell)
    p2s_map = phonon.primitive.p2s_map
    if settings.is_hdf5 or settings.readfc_format == "hdf5":
        try:
            import h5py  # noqa F401
        except ImportError:
            print_error_message("You need to install python-h5py.")
            if log_level:
                print_error()
            sys.exit(1)

        file_exists("force_constants.hdf5", log_level=log_level)
        fc = read_force_constants_from_hdf5(
            filename="force_constants.hdf5",
            p2s_map=p2s_map,
            calculator=phonon.calculator,
        )
        fc_filename = "force_constants.hdf5"
    else:
        file_exists("FORCE_CONSTANTS", log_level=log_level)
        fc = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS", p2s_map=p2s_map)
        fc_filename = "FORCE_CONSTANTS"

    if log_level:
        print('Force constants are read from "%s".' % fc_filename)

    if fc.shape[1] != num_satom:
        error_text = "\n".join(
            [
                f"Number of atoms in supercell ({num_satom}) is not consistent with "
                "the matrix shape of ",
                f"force constants {fc.shape[:2]} read from ",
            ]
        )
        if settings.is_hdf5 or settings.readfc_format == "hdf5":
            error_text += "force_constants.hdf5.\n"
        else:
            error_text += "FORCE_CONSTANTS.\n"
        error_text += (
            "Please carefully check DIM, FORCE_CONSTANTS, " "and %s."
        ) % unitcell_filename
        print_error_message(error_text)
        if log_level:
            print_error()
        sys.exit(1)

    # Compact fc is expanded to full fc when full fc is required.
    if is_full_fc and fc.shape[0] != fc.shape[1]:
        fc = compact_fc_to_full_fc(phonon.primitive, fc, log_level=log_level)
    elif not is_full_fc and fc.shape[0] == fc.shape[1]:
        fc = full_fc_to_compact_fc(phonon.primitive, fc, log_level=log_level)

    if log_level:
        print(f"Array shape of force constants: {fc.shape}")

    phonon.force_constants = fc


def _store_force_constants(
    phonon: Phonopy,
    settings: PhonopySettings,
    phpy_yaml: PhonopyYaml,
    unitcell_filename: str,
    load_phonopy_yaml: bool,
    log_level: int,
) -> bool:
    """Calculate or read force constants.

    Return True if force constants are created.

    """
    physical_units = get_default_physical_units(phonon.calculator)
    p2s_map = phonon.primitive.p2s_map

    if load_phonopy_yaml:
        _produce_force_constants_load_phonopy_yaml(
            phonon, settings, phpy_yaml, unitcell_filename, log_level
        )
        if settings.use_pypolymlp and settings.sscha_iterations:
            if log_level:
                print(
                    "------------------------------- SSCHA start "
                    "--------------------------------"
                )

            sscha = MLPSSCHA(
                phonon,
                phonon.mlp,
                temperature=settings.random_displacement_temperature,
                number_of_snapshots=settings.random_displacements,
                max_iterations=settings.sscha_iterations,
                log_level=log_level,
            )
            fc_unit = physical_units["force_constants_unit"]
            for iter_num in sscha:
                ph = sscha.phonopy
                out_filename = ph.save(
                    filename=f"phonopy_sscha_fc_{iter_num}.yaml",
                    settings={
                        "force_sets": False,
                        "displacements": False,
                        "force_constants": True,
                    },
                    compression=True,
                )
                if log_level:
                    sscha.calculate_free_energy()
                    print(f"SSCHA free energy: {sscha.free_energy * 1000:.3f} meV")
                    if iter_num == 0:
                        print("Initial ", end="")
                    else:
                        print("SSCHA ", end="")
                    print(f'force constants are written into "{out_filename}".')
                    print("", flush=True)

            phonon.force_constants = ph.force_constants

            if log_level:
                print(
                    "-------------------------------- SSCHA end "
                    "---------------------------------"
                )
    else:
        _produce_force_constants(
            phonon, settings, phpy_yaml, unitcell_filename, log_level
        )

    if phonon.force_constants is None:
        return False

    # Impose cutoff radius on force constants
    cutoff_radius = settings.cutoff_radius
    if cutoff_radius:
        phonon.set_force_constants_zero_with_radius(cutoff_radius)

    # This enforces space group symmetry to force constants.
    # The force constants are supposed to be read from a file since otherwise
    # the force constants are considered to follow space group symmetry.
    if settings.fc_spg_symmetry:
        if log_level:
            print("Force constants are symmetrized by space group operations.")
            print("This may take some time...")
        phonon.symmetrize_force_constants_by_space_group()
        if not load_phonopy_yaml:
            write_FORCE_CONSTANTS(
                phonon.get_force_constants(), filename="FORCE_CONSTANTS_SPG"
            )
            if log_level:
                print(
                    "Symmetrized force constants are written into "
                    '"FORCE_CONSTANTS_SPG".'
                )

    # Imporse translational invariance and index permulation symmetry to
    # force constants
    fc_calculator, _ = _get_fc_calculator_params(
        settings, load_phonopy_yaml=load_phonopy_yaml
    )
    if settings.fc_symmetry and fc_calculator == "traditional":
        phonon.symmetrize_force_constants()

    # Write FORCE_CONSTANTS
    if settings.write_force_constants:
        if settings.is_hdf5 or settings.writefc_format == "hdf5":
            fc_unit = physical_units["force_constants_unit"]
            write_force_constants_to_hdf5(
                phonon.force_constants,
                p2s_map=p2s_map,
                physical_unit=fc_unit,
                compression=settings.hdf5_compression,
            )
            if log_level:
                print('Force constants are written into "force_constants.hdf5".')
        else:
            fc = phonon.force_constants
            write_FORCE_CONSTANTS(fc, p2s_map=p2s_map)
            if log_level:
                print('Force constants are written into "FORCE_CONSTANTS".')
                print("  Array shape of force constants is %s." % str(fc.shape))
                if fc.shape[0] != fc.shape[1]:
                    print(
                        "  Use --full-fc option for full array of force " "constants."
                    )

    if log_level:
        print("")

    return True


def _create_random_displacements_at_finite_temperature(
    phonon: Phonopy,
    settings: PhonopySettings,
    confs: dict,
    optional_structure_info: Optional[tuple],
    log_level: int,
):
    if log_level:
        print(
            "Generate random displacements at "
            f"T={settings.random_displacement_temperature}."
        )

    default_disp_filename = "phonopy_disp.yaml"
    if file_exists(default_disp_filename, log_level=log_level, is_any=True):
        if log_level:
            print(
                f'"{default_disp_filename}" is already existing in the '
                "current directory."
            )
            disp_filename = "phonopy_rd_disp.yaml"
            if file_exists(disp_filename, log_level=log_level, is_any=True):
                print(
                    f'"{disp_filename}" is already existing in the current directory.'
                )
                print(f'Please rename "{default_disp_filename}" or "{disp_filename}".')
                print_error()
                sys.exit(1)
            print(
                f"Random displacements at T={settings.random_displacement_temperature} "
                "were not generated."
            )
            print(
                f'Please rename "{disp_filename}" to "{default_disp_filename}" '
                "when using"
            )
            print("generated displacements.")
            print("")
    else:
        disp_filename = default_disp_filename

    phonon.generate_displacements(
        number_of_snapshots=settings.random_displacements,
        random_seed=settings.random_seed,
        temperature=settings.random_displacement_temperature,
        cutoff_frequency=settings.cutoff_frequency,
    )

    if log_level:
        rd_comm_points = phonon.random_displacements.qpoints
        rd_integrated_modes = phonon.random_displacements.integrated_modes
        rd_frequencies = phonon.random_displacements.frequencies
        print(
            "Sampled q-points for generating displacements "
            "(number of integrated modes):"
        )
        for q, integrated_modes, freqs in zip(
            rd_comm_points, rd_integrated_modes, rd_frequencies
        ):
            print(f"{q} ({integrated_modes.sum()})")
            if log_level > 1:
                print("  ", " ".join([f"{f:.3f}" for f in freqs]))
        if np.prod(rd_integrated_modes.shape) - rd_integrated_modes.sum() != 3:
            msg_lines = [
                "*****************************************************************",
                "* Tiny frequencies can induce unexpectedly large displacements. *",
                "* Please check force constants symmetry, e.g., --sym-fc option. *",
                "*****************************************************************",
            ]
            print("\n".join(msg_lines))
        if log_level < 2:
            print('Phonon frequencies can be shown by "-v" option.')
        print()

    _write_displacements_files_then_exit(
        phonon,
        settings,
        confs,
        optional_structure_info,
        log_level,
        disp_filename=disp_filename,
    )


def store_nac_params(
    phonon,
    settings,
    phpy_yaml,
    unitcell_filename,
    log_level,
    nac_factor=None,
    load_phonopy_yaml=False,
):
    """Calculate or read NAC params."""
    if nac_factor is None:
        physical_units = get_default_physical_units(phonon.calculator)
        _nac_factor = physical_units["nac_factor"]
    else:
        _nac_factor = nac_factor

    def read_BORN(phonon):
        with open("BORN") as f:
            return get_born_parameters(f, phonon.primitive, phonon.primitive_symmetry)

    nac_params = None

    if load_phonopy_yaml:
        nac_params = get_nac_params(
            primitive=phonon.primitive,
            nac_params=phpy_yaml.nac_params,
            log_level=log_level,
        )
        if phpy_yaml.nac_params is not None and log_level:
            print('NAC parameters were read from "%s".' % unitcell_filename)
    else:
        if phpy_yaml:
            nac_params = phpy_yaml.nac_params
            if log_level:
                if nac_params is None:
                    print('NAC parameters were not found in "%s".' % unitcell_filename)
                else:
                    print('NAC parameters were read from "%s".' % unitcell_filename)

        if nac_params is None and file_exists("BORN", log_level=log_level):
            nac_params = read_BORN(phonon)
            if nac_params is not None and log_level:
                print('NAC parameters were read from "%s".' % "BORN")

            if not nac_params:
                error_text = "BORN file could not be read correctly."
                print_error_message(error_text)
                if log_level:
                    print_error()
                sys.exit(1)

    if nac_params is not None:
        if "factor" not in nac_params or nac_params["factor"] is None:
            nac_params["factor"] = _nac_factor
        if settings.nac_method is not None:
            nac_params["method"] = settings.nac_method
        phonon.nac_params = nac_params
        if log_level:
            dm = phonon.dynamical_matrix
            if dm is not None:
                if isinstance(dm, DynamicalMatrixNAC):
                    dm.show_nac_message()
                print("")

        if log_level > 1:
            print("-" * 27 + " Dielectric constant " + "-" * 28)
            for v in nac_params["dielectric"]:
                print("         %12.7f %12.7f %12.7f" % tuple(v))
            print("-" * 26 + " Born effective charges " + "-" * 26)
            symbols = phonon.primitive.symbols
            for i, (z, s) in enumerate(zip(nac_params["born"], symbols)):
                for j, v in enumerate(z):
                    if j == 0:
                        text = "%5d %-2s" % (i + 1, s)
                    else:
                        text = "        "
                    print("%s %12.7f %12.7f %12.7f" % ((text,) + tuple(v)))
            print("-" * 76)


def _run_calculation(phonon: Phonopy, settings: PhonopySettings, plot_conf, log_level):
    """Run phonon calculations."""
    interface_mode = phonon.calculator
    physical_units = get_default_physical_units(interface_mode)
    run_mode = settings.run_mode

    #
    # QPOINTS mode
    #
    if run_mode == "qpoints":
        if settings.read_qpoints:
            q_points = parse_QPOINTS()
            if log_level:
                print("Frequencies at q-points given by QPOINTS:")
        elif settings.qpoints:
            q_points = settings.qpoints
            if log_level:
                print("Q-points that will be calculated at:")
                for q in q_points:
                    print("    %s" % q)
        else:
            print_error_message("Q-points are not properly specified.")
            if log_level:
                print_error()
            sys.exit(1)
        phonon.run_qpoints(
            q_points,
            with_eigenvectors=settings.is_eigenvectors,
            with_group_velocities=settings.is_group_velocity,
            with_dynamical_matrices=settings.write_dynamical_matrices,
            nac_q_direction=settings.nac_q_direction,
        )

        if settings.is_hdf5 or settings.qpoints_format == "hdf5":
            phonon.write_hdf5_qpoints_phonon()
        else:
            phonon.write_yaml_qpoints_phonon()

    #
    # Band structure
    #
    if run_mode == "band" or run_mode == "band_mesh":
        if settings.band_points is None:
            npoints = 51
        else:
            npoints = settings.band_points
        band_paths = settings.band_paths

        if _is_band_auto(settings):
            print("SeeK-path is used to generate band paths.")
            print(
                "  About SeeK-path https://seekpath.readthedocs.io/ "
                "(citation there-in)"
            )
            is_legacy_plot = False
            bands, labels, path_connections = get_band_qpoints_by_seekpath(
                phonon.primitive,
                npoints,
                is_const_interval=settings.is_band_const_interval,
            )
        else:
            is_legacy_plot = settings.is_legacy_plot
            if settings.is_band_const_interval:
                reclat = np.linalg.inv(phonon.primitive.cell)
                bands = get_band_qpoints(
                    band_paths, npoints=npoints, rec_lattice=reclat
                )
            else:
                bands = get_band_qpoints(band_paths, npoints=npoints)
            path_connections = []
            for paths in band_paths:
                path_connections += [
                    True,
                ] * (len(paths) - 2)
                path_connections.append(False)
            labels = settings.band_labels

        if log_level:
            print("Reciprocal space paths in reduced coordinates:")
            for band in bands:
                print(
                    "[%6.3f %6.3f %6.3f] --> [%6.3f %6.3f %6.3f]"
                    % (tuple(band[0]) + tuple(band[-1]))
                )

        phonon.run_band_structure(
            bands,
            with_eigenvectors=settings.is_eigenvectors,
            with_group_velocities=settings.is_group_velocity,
            is_band_connection=settings.is_band_connection,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=is_legacy_plot,
        )
        if interface_mode is None:
            comment = None
        else:
            comment = {
                "calculator": interface_mode,
                "length_unit": physical_units["length_unit"],
            }

        if settings.is_hdf5 or settings.band_format == "hdf5":
            phonon.write_hdf5_band_structure(comment=comment)
        else:
            phonon.write_yaml_band_structure(comment=comment)

        if plot_conf["plot_graph"] and run_mode != "band_mesh":
            plot = phonon.plot_band_structure()
            if plot_conf["save_graph"]:
                plot.savefig("band.pdf")
            else:
                plot.show()

    #
    # mesh sampling
    #
    if run_mode == "mesh" or run_mode == "band_mesh":
        mesh_numbers = settings.mesh_numbers
        if mesh_numbers is None:
            mesh_numbers = 50.0
        mesh_shift = settings.mesh_shift
        t_symmetry = settings.is_time_reversal_symmetry
        q_symmetry = settings.is_mesh_symmetry
        is_gamma_center = settings.is_gamma_center

        if (
            settings.is_thermal_displacements
            or settings.is_thermal_displacement_matrices
        ):
            if settings.cutoff_frequency is not None:
                if log_level:
                    print_error_message(
                        "Use FMIN (--fmin) instead of CUTOFF_FREQUENCY "
                        "(--cutoff-freq)."
                    )
                    print_error()
                sys.exit(1)

            phonon.init_mesh(
                mesh=mesh_numbers,
                shift=mesh_shift,
                is_time_reversal=t_symmetry,
                is_mesh_symmetry=q_symmetry,
                with_eigenvectors=settings.is_eigenvectors,
                is_gamma_center=is_gamma_center,
                use_iter_mesh=True,
            )
            if log_level:
                print("Mesh numbers: %s" % phonon.mesh_numbers)
        else:
            phonon.init_mesh(
                mesh=mesh_numbers,
                shift=mesh_shift,
                is_time_reversal=t_symmetry,
                is_mesh_symmetry=q_symmetry,
                with_eigenvectors=settings.is_eigenvectors,
                with_group_velocities=settings.is_group_velocity,
                is_gamma_center=is_gamma_center,
            )
            if log_level:
                print("Mesh numbers: %s" % phonon.mesh_numbers)
                weights = phonon.mesh.weights
                if q_symmetry:
                    print(
                        "Number of irreducible q-points on sampling mesh: "
                        "%d/%d" % (weights.shape[0], np.prod(phonon.mesh_numbers))
                    )
                else:
                    print("Number of q-points on sampling mesh: %d" % weights.shape[0])
                print("Calculating phonons on sampling mesh...")

            phonon.mesh.run()

            if settings.write_mesh:
                if settings.is_hdf5 or settings.mesh_format == "hdf5":
                    phonon.write_hdf5_mesh()
                else:
                    phonon.write_yaml_mesh()

        #
        # Thermal property
        #
        if settings.is_thermal_properties:
            if log_level:
                if settings.is_projected_thermal_properties:
                    print("Calculating projected thermal properties...")
                else:
                    print("Calculating thermal properties...")
            t_step = settings.temperature_step
            t_max = settings.max_temperature
            t_min = settings.min_temperature
            phonon.run_thermal_properties(
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                cutoff_frequency=settings.cutoff_frequency,
                pretend_real=settings.pretend_real,
                band_indices=settings.band_indices,
                is_projection=settings.is_projected_thermal_properties,
                classical=settings.classical,
            )
            phonon.write_yaml_thermal_properties()

            if log_level:
                cutoff_freq = phonon.thermal_properties.cutoff_frequency
                cutoff_freq /= THzToEv
                print("Cutoff frequency: %.5f" % cutoff_freq)
                num_ignored_modes = (
                    phonon.thermal_properties.number_of_modes
                    - phonon.thermal_properties.number_of_integrated_modes
                )
                print(
                    "Number of phonon frequencies less than cutoff "
                    "frequency: %d/%d"
                    % (num_ignored_modes, phonon.thermal_properties.number_of_modes)
                )
                print(
                    "#%11s %15s%15s%15s%15s"
                    % (
                        "T [K]",
                        "F [kJ/mol]",
                        "S [J/K/mol]",
                        "C_v [J/K/mol]",
                        "E [kJ/mol]",
                    )
                )
                tp = phonon.get_thermal_properties_dict()
                temps = tp["temperatures"]
                fe = tp["free_energy"]
                entropy = tp["entropy"]
                heat_capacity = tp["heat_capacity"]
                for T, F, S, CV in zip(temps, fe, entropy, heat_capacity):
                    print(("%12.3f " + "%15.7f" * 4) % (T, F, S, CV, F + T * S / 1000))

            if plot_conf["plot_graph"]:
                plot = phonon.plot_thermal_properties()
                if plot_conf["save_graph"]:
                    plot.savefig("thermal_properties.pdf")
                else:
                    plot.show()

        #
        # Thermal displacements
        #
        elif settings.is_thermal_displacements and run_mode in ("mesh", "band_mesh"):
            p_direction = settings.projection_direction
            if log_level and p_direction is not None:
                c_direction = np.dot(p_direction, phonon.primitive.cell)
                c_direction /= np.linalg.norm(c_direction)
                print(
                    "Projection direction: [%6.3f %6.3f %6.3f] "
                    "(fractional)" % tuple(p_direction)
                )
                print(
                    "                      [%6.3f %6.3f %6.3f] "
                    "(Cartesian)" % tuple(c_direction)
                )
            if log_level:
                print("Calculating thermal displacements...")
            t_step = settings.temperature_step
            t_max = settings.max_temperature
            t_min = settings.min_temperature
            phonon.run_thermal_displacements(
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                direction=p_direction,
                freq_min=settings.min_frequency,
                freq_max=settings.max_frequency,
            )
            phonon.write_yaml_thermal_displacements()

            if plot_conf["plot_graph"]:
                plot = phonon.plot_thermal_displacements(plot_conf["with_legend"])
                if plot_conf["save_graph"]:
                    plot.savefig("thermal_displacement.pdf")
                else:
                    plot.show()

        #
        # Thermal displacement matrices
        #
        elif settings.is_thermal_displacement_matrices and run_mode in (
            "mesh",
            "band_mesh",
        ):
            if log_level:
                print("Calculating thermal displacement matrices...")
            t_step = settings.temperature_step
            t_max = settings.max_temperature
            t_min = settings.min_temperature
            t_cif = settings.thermal_displacement_matrix_temperatue
            if t_cif is None:
                temperatures = None
            else:
                temperatures = [
                    t_cif,
                ]
            phonon.run_thermal_displacement_matrices(
                t_step=t_step,
                t_max=t_max,
                t_min=t_min,
                temperatures=temperatures,
                freq_min=settings.min_frequency,
                freq_max=settings.max_frequency,
            )
            phonon.write_yaml_thermal_displacement_matrices()
            if t_cif is not None:
                phonon.write_thermal_displacement_matrix_to_cif(0)

        #
        # Projected DOS
        #
        elif settings.pdos_indices is not None and run_mode in ("mesh", "band_mesh"):
            p_direction = settings.projection_direction
            if log_level and p_direction is not None and not settings.xyz_projection:
                c_direction = np.dot(p_direction, phonon.primitive.cell)
                c_direction /= np.linalg.norm(c_direction)
                print(
                    "Projection direction: [%6.3f %6.3f %6.3f] "
                    "(fractional)" % tuple(p_direction)
                )
                print(
                    "                      [%6.3f %6.3f %6.3f] "
                    "(Cartesian)" % tuple(c_direction)
                )
            if log_level:
                print("Calculating projected DOS...")

            phonon.run_projected_dos(
                sigma=settings.sigma,
                freq_min=settings.min_frequency,
                freq_max=settings.max_frequency,
                freq_pitch=settings.frequency_pitch,
                use_tetrahedron_method=settings.is_tetrahedron_method,
                direction=p_direction,
                xyz_projection=settings.xyz_projection,
            )
            phonon.write_projected_dos()

            if plot_conf["plot_graph"]:
                _pdos_indices, legend = _get_pdos_indices_and_legend(settings, phonon)
                if run_mode != "band_mesh":
                    plot = phonon.plot_projected_dos(
                        pdos_indices=_pdos_indices, legend=legend
                    )
                    if plot_conf["save_graph"]:
                        plot.savefig("partial_dos.pdf")
                    else:
                        plot.show()

        #
        # Total DOS
        #
        elif (
            (plot_conf["plot_graph"] or settings.is_dos_mode)
            and not _is_pdos_auto(settings)
            and run_mode in ("mesh", "band_mesh")
        ):
            phonon.run_total_dos(
                sigma=settings.sigma,
                freq_min=settings.min_frequency,
                freq_max=settings.max_frequency,
                freq_pitch=settings.frequency_pitch,
                use_tetrahedron_method=settings.is_tetrahedron_method,
            )

            if log_level:
                print("Calculating DOS...")

            if settings.fits_Debye_model:
                phonon.set_Debye_frequency()
                if log_level:
                    debye_freq = phonon.get_Debye_frequency()
                    print("Debye frequency: %10.5f" % debye_freq)
            phonon.write_total_dos()

            if plot_conf["plot_graph"] and run_mode != "band_mesh":
                plot = phonon.plot_total_dos()
                if plot_conf["save_graph"]:
                    plot.savefig("total_dos.pdf")
                else:
                    plot.show()

        #
        # Momemt
        #
        elif settings.is_moment and run_mode in ("mesh", "band_mesh"):
            freq_min = settings.min_frequency
            freq_max = settings.max_frequency
            if log_level:
                text = "Calculating moment of phonon states distribution"
                if freq_min is None and freq_max is None:
                    text += "..."
                elif freq_min is None and freq_max is not None:
                    text += "\nbelow frequency %.3f..." % freq_max
                elif freq_min is not None and freq_max is None:
                    text += "\nabove frequency %.3f..." % freq_min
                elif freq_min is not None and freq_max is not None:
                    text += "\nbetween frequencies %.3f and %.3f..." % (
                        freq_min,
                        freq_max,
                    )
            print(text)
            print("")
            print("Order|   Total   |   Projected to atoms")
            if settings.moment_order is not None:
                phonon.run_moment(
                    order=settings.moment_order,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    is_projection=False,
                )
                total_moment = phonon.get_moment()
                phonon.run_moment(
                    order=settings.moment_order,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    is_projection=True,
                )
                text = " %3d |%10.5f | " % (settings.moment_order, total_moment)
                for m in phonon.get_moment():
                    text += "%10.5f " % m
                print(text)
            else:
                for i in range(3):
                    phonon.run_moment(
                        order=i,
                        freq_min=freq_min,
                        freq_max=freq_max,
                        is_projection=False,
                    )
                    total_moment = phonon.get_moment()
                    phonon.run_moment(
                        order=i,
                        freq_min=freq_min,
                        freq_max=freq_max,
                        is_projection=True,
                    )
                    text = " %3d |%10.5f | " % (i, total_moment)
                    for m in phonon.get_moment():
                        text += "%10.5f " % m
                    print(text)

        #
        # Band structure and DOS are plotted simultaneously.
        #
        if (
            run_mode == "band_mesh"
            and plot_conf["plot_graph"]
            and not settings.is_thermal_properties
            and not settings.is_thermal_displacements
            and not settings.is_thermal_displacement_matrices
            and not settings.is_thermal_distances
        ):
            if settings.pdos_indices is not None:
                _pdos_indices, legend = _get_pdos_indices_and_legend(settings, phonon)
                plot = phonon.plot_band_structure_and_dos(pdos_indices=_pdos_indices)
            else:
                plot = phonon.plot_band_structure_and_dos()
            if plot_conf["save_graph"]:
                plot.savefig("band_dos.pdf")
            else:
                plot.show()

    #
    # Animation
    #
    elif run_mode == "anime":
        anime_type = settings.anime_type
        if anime_type == "v_sim":
            q_point = settings.anime_qpoint
            amplitude = settings.anime_amplitude
            fname_out = phonon.write_animation(
                q_point=q_point, anime_type="v_sim", amplitude=amplitude
            )
            if log_level:
                print("Animation type: v_sim")
                print("q-point: [%6.3f %6.3f %6.3f]" % tuple(q_point))
        else:
            amplitude = settings.anime_amplitude
            band_index = settings.anime_band_index
            division = settings.anime_division
            shift = settings.anime_shift
            fname_out = phonon.write_animation(
                anime_type=anime_type,
                band_index=band_index,
                amplitude=amplitude,
                num_div=division,
                shift=shift,
            )
            if log_level:
                print("Animation type: %s" % anime_type)
                print("amplitude: %f" % amplitude)
                if anime_type != "jmol":
                    print("band index: %d" % band_index)
                    print("Number of images: %d" % division)
        if log_level:
            print('Animation was written in "%s". ' % fname_out)

    #
    # Modulation
    #
    elif run_mode == "modulation":
        mod_setting = settings.modulation
        phonon_modes = mod_setting["modulations"]
        dimension = mod_setting["dimension"]
        if "delta_q" in mod_setting:
            delta_q = mod_setting["delta_q"]
        else:
            delta_q = None
        derivative_order = mod_setting["order"]
        num_band = len(phonon.primitive) * 3

        if log_level:
            if len(phonon_modes) == 1:
                print(
                    "Modulated structure with %s multiplicity was created." % dimension
                )
            else:
                print(
                    "Modulated structures with %s multiplicity were created."
                    % dimension
                )

        error_indices = []
        for i, ph_mode in enumerate(phonon_modes):
            if ph_mode[1] < 0 or ph_mode[1] >= num_band:
                error_indices.append(i)
            if log_level:
                text = "%d: q=%s, band index=%d, amplitude=%f" % (
                    i + 1,
                    ph_mode[0],
                    ph_mode[1] + 1,
                    ph_mode[2],
                )
                if len(ph_mode) > 3:
                    text += ", phase=%f" % ph_mode[3]
                print(text)

        if error_indices:
            if log_level:
                lines = [
                    "Band index of modulation %d is out of range." % (i + 1)
                    for i in error_indices
                ]
                print_error_message("\n".join(lines))
            print_error()
            sys.exit(1)

        phonon.run_modulations(
            dimension,
            phonon_modes,
            delta_q=delta_q,
            derivative_order=derivative_order,
            nac_q_direction=settings.nac_q_direction,
        )

        phonon.write_modulations()
        phonon.write_yaml_modulations()

    #
    # Ir-representation
    #
    elif run_mode == "irreps":
        if phonon.set_irreps(
            settings.irreps_q_point,
            is_little_cogroup=settings.is_little_cogroup,
            nac_q_direction=settings.nac_q_direction,
            degeneracy_tolerance=settings.irreps_tolerance,
        ):
            phonon.show_irreps(settings.show_irreps)
            phonon.write_yaml_irreps(settings.show_irreps)


def _start_phonopy(**argparse_control):
    """Parse arguments and set some basic parameters."""
    parser, deprecated = get_parser(**argparse_control)
    args = parser.parse_args()

    # Set log level
    log_level = 1
    if args.verbose:
        log_level = 2
    if args.quiet or args.is_check_symmetry:
        log_level = 0
    if args.loglevel is not None:
        log_level = args.loglevel

    if args.is_graph_save:
        import matplotlib

        matplotlib.use("Agg")

    # Show phonopy logo
    if log_level:
        _print_phonopy()

        import phonopy._phonopy as phonoc

        max_threads = phonoc.omp_max_threads()
        if max_threads > 0:
            print(f"Compiled with OpenMP support (max {max_threads} threads).")

        if argparse_control.get("load_phonopy_yaml", False):
            print("Running in phonopy.load mode.")
        print("Python version %d.%d.%d" % sys.version_info[:3])
        try:  # spglib.get_version() is deprecated.
            print(f"Spglib version {spglib.spg_get_version()}")
        except AttributeError:
            print("Spglib version %d.%d.%d" % spglib.get_version())

        print("")

        if deprecated:
            show_deprecated_option_warnings(deprecated)

    return args, log_level


def _read_phonopy_settings(
    args, argparse_control, log_level
) -> tuple[PhonopySettings, dict, Optional[str]]:
    """Read phonopy settings.

    Returns
    -------
    tuple
        settings : PhonopySettings
            Configurations set by user to run phonopy.
        confs : dict
            Raw phonopy configurations in str (value) for each configuration
            tag (key).
        cell_filename : str or None
            Filename that contains crystal structure information. When
            unspecified in command line tool, this is None.

    """
    load_phonopy_yaml = argparse_control.get("load_phonopy_yaml", False)
    conf_filename = None

    if load_phonopy_yaml:
        if args.conf_filename:
            conf_filename = args.conf_filename
            phonopy_conf_parser = PhonopyConfParser(
                filename=args.conf_filename,
                args=args,
                default_settings=argparse_control,
            )
        else:
            phonopy_conf_parser = PhonopyConfParser(
                args=args, default_settings=argparse_control
            )
        if len(args.filename) > 0:
            file_exists(args.filename[0], log_level=log_level)
            cell_filename = args.filename[0]
        else:
            cell_filename = phonopy_conf_parser.settings.cell_filename
    else:
        if len(args.filename) > 0:
            file_exists(args.filename[0], log_level=log_level)
            if is_file_phonopy_yaml(args.filename[0]):
                phonopy_conf_parser = PhonopyConfParser(args=args)
                cell_filename = args.filename[0]
            else:
                conf_filename = args.filename[0]
                phonopy_conf_parser = PhonopyConfParser(
                    filename=args.filename[0], args=args
                )
                cell_filename = phonopy_conf_parser.settings.cell_filename
        else:
            phonopy_conf_parser = PhonopyConfParser(args=args)
            cell_filename = phonopy_conf_parser.settings.cell_filename

    confs = phonopy_conf_parser.confs.copy()
    settings = phonopy_conf_parser.settings

    if log_level > 0 and conf_filename is not None:
        print(f'"{conf_filename}" was read as phonopy configuration file.')

    return settings, confs, cell_filename


def _is_band_auto(settings):
    """Check whether automatic band paths setting or not."""
    return isinstance(settings.band_paths, str) and settings.band_paths == "auto"


def _is_pdos_auto(settings):
    """Check whether automatic PDOS setting or not."""
    return settings.pdos_indices == "auto"


def _get_pdos_indices_and_legend(settings, phonon: Phonopy):
    """Return pdos_indices and legend from settings."""
    pdos_indices = settings.pdos_indices
    if settings.xyz_projection:
        legend = []
        if _is_pdos_auto(settings):
            pdos_indices = get_pdos_indices(phonon.primitive_symmetry)
        _pdos_indices = []
        for index_set in pdos_indices:
            xyz_set = []
            for idx in index_set:
                xyz_set += list(range(idx * 3, (idx + 1) * 3))
            xyz_set = np.array(xyz_set)
            legend.append(xyz_set + 1)
            _pdos_indices.append(xyz_set)
    elif _is_pdos_auto(settings):
        _pdos_indices = get_pdos_indices(phonon.primitive_symmetry)
        legend = [phonon.primitive.symbols[x[0]] for x in _pdos_indices]
    else:
        legend = [np.array(x) + 1 for x in pdos_indices]
        _pdos_indices = pdos_indices
    return _pdos_indices, legend


def _auto_primitive_axes(primitive_matrix):
    """Check whether automatic primitive matrix setting or not."""
    return isinstance(primitive_matrix, str) and primitive_matrix == "auto"


def _get_fc_calculator_params(settings, load_phonopy_yaml=True):
    """Return fc_calculator and fc_calculator_params from settings."""
    fc_calculator = None
    if settings.fc_calculator is not None:
        if settings.fc_calculator.lower() in fc_calculator_names:
            fc_calculator = settings.fc_calculator.lower()
    else:
        if settings.fc_symmetry:
            if load_phonopy_yaml:
                fc_calculator = "symfc"
            else:
                fc_calculator = "traditional"
        else:
            fc_calculator = "traditional"

    fc_calculator_options = None
    if settings.fc_calculator_options is not None:
        fc_calculator_options = settings.fc_calculator_options

    return fc_calculator, fc_calculator_options


def _get_cell_info(
    settings: PhonopySettings,
    cell_filename: str,
    log_level: int = 0,
    load_phonopy_yaml: bool = False,
) -> dict:
    """Return calculator interface and crystal structure information."""
    cell_info = collect_cell_info(
        supercell_matrix=settings.supercell_matrix,
        primitive_matrix=settings.primitive_matrix,
        interface_mode=settings.calculator,
        cell_filename=cell_filename,
        chemical_symbols=settings.chemical_symbols,
        enforce_primitive_matrix_auto=_is_band_auto(settings),
        load_phonopy_yaml=load_phonopy_yaml,
    )

    # Show primitive matrix overwrite message
    phpy_yaml: PhonopyYaml = cell_info.get("phonopy_yaml")
    if phpy_yaml is not None:
        yaml_filename = cell_info["optional_structure_info"][0]
        pmat_in_settings = _get_primitive_matrix(
            cell_info["primitive_matrix"], phpy_yaml.unitcell
        )
        pmat_in_phpy_yaml = _get_primitive_matrix(
            phpy_yaml.primitive_matrix, phpy_yaml.unitcell
        )
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

    if "error_message" in cell_info:
        print_error_message(cell_info["error_message"])
        if log_level:
            print_error()
        sys.exit(1)

    set_magnetic_moments(cell_info, settings, log_level)

    return cell_info


def _get_primitive_matrix(
    pmat: Optional[Union[str, np.ndarray]], unitcell: Phonopy, symprec: float = 1e-5
) -> np.ndarray:
    _pmat = get_primitive_matrix(pmat)
    if isinstance(_pmat, str) and _pmat == "auto":
        _pmat = guess_primitive_matrix(unitcell, symprec=symprec)
    if _pmat is None:
        _pmat = np.eye(3, dtype="double")
    return _pmat


def set_magnetic_moments(cell_info: dict, settings: PhonopySettings, log_level):
    """Set magnetic moments to unitcell in cell_info."""
    # Set magnetic moments
    magmoms = settings.magnetic_moments
    if magmoms is not None:
        unitcell = cell_info["unitcell"]
        if len(magmoms) in (len(unitcell), len(unitcell) * 3):
            unitcell.magnetic_moments = magmoms
        else:
            error_text = "Invalid MAGMOM setting"
            print_error_message(error_text)
            if log_level:
                print_error()
            sys.exit(1)


def _show_symmetry_info_then_exit(cell_info, symprec):
    """Show crystal structure information in yaml style."""
    phonon = Phonopy(
        cell_info["unitcell"],
        np.eye(3, dtype=int),
        primitive_matrix=cell_info["primitive_matrix"],
        symprec=symprec,
        calculator=cell_info["interface_mode"],
        log_level=0,
    )
    check_symmetry(phonon, cell_info)
    sys.exit(0)


def _check_supercell_in_yaml(cell_info, ph, log_level):
    """Check supercell size consistency."""
    if (
        cell_info["phonopy_yaml"] is not None
        and cell_info["phonopy_yaml"].supercell is not None
    ):
        if not cells_isclose(cell_info["phonopy_yaml"].supercell, ph.supercell):
            if log_level:
                print(
                    "Generated Supercell is inconsistent with that "
                    'in "%s".' % cell_info["optional_structure_info"][0]
                )
                print_error()
            sys.exit(1)


def _init_phonopy(settings, cell_info, symprec, log_level):
    """Prepare phonopy object."""
    if (
        settings.create_displacements
        and settings.random_displacement_temperature is None
    ):
        phonon = Phonopy(
            cell_info["unitcell"],
            cell_info["supercell_matrix"],
            primitive_matrix=cell_info["primitive_matrix"],
            symprec=symprec,
            is_symmetry=settings.is_symmetry,
            store_dense_svecs=settings.store_dense_svecs,
            calculator=cell_info["interface_mode"],
            log_level=log_level,
        )
    else:  # Read FORCE_SETS, FORCE_CONSTANTS, or force_constants.hdf5
        # Overwrite frequency unit conversion factor
        if settings.frequency_conversion_factor is not None:
            freq_factor = settings.frequency_conversion_factor
        else:
            physical_units = get_default_physical_units(cell_info["interface_mode"])
            freq_factor = physical_units["factor"]

        phonon = Phonopy(
            cell_info["unitcell"],
            cell_info["supercell_matrix"],
            primitive_matrix=cell_info["primitive_matrix"],
            factor=freq_factor,
            frequency_scale_factor=settings.frequency_scale_factor,
            dynamical_matrix_decimals=settings.dm_decimals,
            force_constants_decimals=settings.fc_decimals,
            group_velocity_delta_q=settings.group_velocity_delta_q,
            symprec=symprec,
            is_symmetry=settings.is_symmetry,
            store_dense_svecs=settings.store_dense_svecs,
            calculator=cell_info["interface_mode"],
            log_level=log_level,
        )

        _check_supercell_in_yaml(cell_info, phonon, log_level)

    # Set atomic masses of primitive cell
    if settings.masses is not None:
        phonon.masses = settings.masses

    # Atomic species without mass case
    symbols_with_no_mass = []
    if phonon.primitive.masses is None:
        for s in phonon.primitive.symbols:
            if atom_data[symbol_map[s]][3] is None and s not in symbols_with_no_mass:
                symbols_with_no_mass.append(s)
                print_error_message(
                    "Atomic mass of '%s' is not implemented in phonopy." % s
                )
                print_error_message("MASS tag can be used to set atomic masses.")

    if len(symbols_with_no_mass) > 0:
        if log_level:
            _print_phonopy_end()
        sys.exit(1)

    return phonon


def main(**argparse_control):
    """Start phonopy.

    phonopy command:
        argparse_control = {
            "fc_symmetry": False,
            "is_nac": False,
            "load_phonopy_yaml": False,
        }
    phonopy-load command:
        argparse_control = {
            "fc_symmetry": True,
            "is_nac": True,
            "load_phonopy_yaml": True,
        }

    """
    # import warnings

    # warnings.simplefilter("error")

    ############################################
    # Parse phonopy conf and crystal structure #
    ############################################
    load_phonopy_yaml = argparse_control.get("load_phonopy_yaml", False)

    if "args" in argparse_control:  # For pytest
        args = argparse_control["args"]
        log_level = args.log_level
    else:
        args, log_level = _start_phonopy(**argparse_control)

    plot_conf = {
        "plot_graph": args.is_graph_plot,
        "save_graph": args.is_graph_save,
        "with_legend": args.is_legend,
    }

    settings, confs, cell_filename = _read_phonopy_settings(
        args, argparse_control, log_level
    )

    # phonopy --symmetry
    run_symmetry_info = args.is_check_symmetry

    # -----------------------------------------------------------------------
    # ----------------- 'args' should not be used below. --------------------
    # -----------------------------------------------------------------------

    ###########################################################
    # Symmetry tolerance. Distance unit depends on interface. #
    ###########################################################
    if settings.symmetry_tolerance is None:
        symprec = 1e-5
    else:
        symprec = settings.symmetry_tolerance

    ##########################################
    # Create FORCE_SETS (-f or --force_sets) #
    ##########################################
    if settings.create_force_sets or settings.create_force_sets_zero:
        _create_FORCE_SETS_from_settings(settings, cell_filename, symprec, log_level)
        if log_level > 0:
            _print_phonopy_end()
        sys.exit(0)

    ####################################################################
    # Create FORCE_CONSTANTS (--fc or --force_constants) only for VASP #
    ####################################################################
    if settings.create_force_constants:
        filename = settings.create_force_constants
        file_exists(filename, log_level=log_level)
        write_hdf5 = settings.is_hdf5 or settings.writefc_format == "hdf5"
        is_error = create_FORCE_CONSTANTS(filename, write_hdf5, log_level)
        if log_level:
            _print_phonopy_end()
        sys.exit(is_error)

    #################################################################
    # Parse crystal structure and optionally phonopy.yaml-like file #
    #################################################################
    cell_info = _get_cell_info(
        settings,
        cell_filename,
        log_level=log_level,
        load_phonopy_yaml=load_phonopy_yaml,
    )

    unitcell_filename = cell_info["optional_structure_info"][0]

    if cell_info["unitcell"].magnetic_moments is not None and _auto_primitive_axes(
        cell_info["primitive_matrix"]
    ):
        print_error_message(f'Unit cell was read from "{unitcell_filename}".')

        if cell_info["phonopy_yaml"] is None:
            print_error_message(
                "'PRIMITIVE_AXES = auto' and 'BAND = auto' "
                "are not allowed using with MAGMOM."
            )
        else:
            print_error_message(str(cell_info["phonopy_yaml"].unitcell))
            print_error_message("")
            print_error_message(
                "'PRIMITIVE_AXES = auto' and 'BAND = auto' "
                "are not allowed using with magnetic_moments."
            )
        if log_level:
            print_error()
        sys.exit(1)

    ###########################################################
    # Show crystal symmetry information and exit (--symmetry) #
    ###########################################################
    if run_symmetry_info:
        _show_symmetry_info_then_exit(cell_info, symprec)

    ######################
    # Initialize phonopy #
    ######################
    phonon = _init_phonopy(settings, cell_info, symprec, log_level)

    ################################################
    # Show phonopy settings and crystal structures #
    ################################################
    if log_level:
        _print_settings(
            settings,
            phonon,
            _auto_primitive_axes(cell_info["primitive_matrix"]),
            unitcell_filename,
            load_phonopy_yaml,
        )
        if phonon.unitcell.magnetic_moments is None:
            print("Spacegroup: %s" % phonon.symmetry.get_international_table())
        elif phonon.symmetry.dataset is not None:
            uni_number = phonon.symmetry.dataset.uni_number
            msg_type = phonon.symmetry.dataset.msg_type
            print(f"Magnetic space group UNI number: {uni_number}")
            print(f"Type-{msg_type} magnetic space group")
        print(
            "Number of symmetry operations in supercell: %d"
            % len(phonon.symmetry.symmetry_operations["rotations"])
        )
        if log_level > 1:
            _print_cells(phonon)
        else:
            print(
                "Use -v option to watch primitive cell, unit cell, "
                "and supercell structures."
            )
        if log_level == 1:
            print("")

    ##################################
    # Non-analytical term correction #
    ##################################

    if settings.is_nac or (
        (settings.create_displacements or settings.random_displacements)
        and file_exists("BORN", is_any=True)
    ):
        store_nac_params(
            phonon,
            settings,
            cell_info["phonopy_yaml"],
            unitcell_filename,
            log_level,
            load_phonopy_yaml=load_phonopy_yaml,
        )

    ################################################################
    # Create non-temperature dependent displacements and then exit #
    ################################################################
    # settings.use_pypolymlp=True case is handled in _store_force_constants.
    if (
        (settings.create_displacements or settings.random_displacements)
        and settings.random_displacement_temperature is None
        and not settings.use_pypolymlp
    ):
        if settings.displacement_distance is None:
            displacement_distance = get_default_displacement_distance(phonon.calculator)
        else:
            displacement_distance = settings.displacement_distance

        phonon.generate_displacements(
            distance=displacement_distance,
            is_plusminus=settings.is_plusminus_displacement,
            is_diagonal=settings.is_diagonal_displacement,
            is_trigonal=settings.is_trigonal_displacement,
            number_of_snapshots=settings.random_displacements,
            random_seed=settings.random_seed,
            max_distance=settings.displacement_distance_max,
        )
        _write_displacements_files_then_exit(
            phonon, settings, confs, cell_info["optional_structure_info"], log_level
        )

    ###################
    # Force constants #
    ###################
    if not _store_force_constants(
        phonon,
        settings,
        cell_info["phonopy_yaml"],
        unitcell_filename,
        load_phonopy_yaml,
        log_level,
    ):
        if log_level:
            print_error()
        sys.exit(1)

    ###################################################################
    # Create random displacements at finite temperature and then exit #
    ###################################################################
    if (
        not settings.sscha_iterations
        and settings.random_displacements
        and settings.random_displacement_temperature is not None
    ):
        _create_random_displacements_at_finite_temperature(
            phonon, settings, confs, cell_info["optional_structure_info"], log_level
        )

    ######################
    # Additional message #
    ######################
    if log_level and not load_phonopy_yaml:
        print("*" * 76)

        print(
            ' The "phonopy" command for running phonon calculations'
            "will be phased out in "
        )
        print(
            ' the future. It is recommended to use the "phonopy-load" command instead.'
        )
        print(" For more details, please refer to this link.")
        print(" https://phonopy.github.io/phonopy/phonopy-load.html")
        print("*" * 76)

    #######################
    # Phonon calculations #
    #######################
    if settings.run_mode not in (
        "band",
        "mesh",
        "band_mesh",
        "anime",
        "modulation",
        "irreps",
        "qpoints",
    ):
        print("-" * 76)
        print(
            " One of the following run modes may be specified for phonon "
            "calculations."
        )
        for mode in [
            "Mesh sampling (MESH, --mesh)",
            "Q-points (QPOINTS, --qpoints)",
            "Band structure (BAND, --band)",
            "Animation (ANIME, --anime)",
            "Modulation (MODULATION, --modulation)",
            "Characters of Irreps (IRREPS, --irreps)",
            "Create displacements (CREATE_DISPLACEMENTS, -d)",
        ]:
            print(" - %s" % mode)
        print("-" * 76)

    _run_calculation(phonon, settings, plot_conf, log_level)

    ########################
    # Phonopy finalization #
    ########################
    _finalize_phonopy(log_level, settings, confs, phonon)
