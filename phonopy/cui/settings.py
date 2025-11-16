"""Phonopy input and command option tools."""

# Copyright (C) 2011 Atsushi Togo
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

import argparse
import os
import sys
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray


def fracval(frac):
    """Return floating point value from rational."""
    if frac.find("/") == -1:
        return float(frac)
    else:
        x = frac.split("/")
        return float(x[0]) / float(x[1])


class Settings:
    """Phonopy settings container.

    This works almost like a dictionary.
    Method names without 'set_' and 'get_' and keys of self._v have to be same.

    """

    def __init__(self, load_phonopy_yaml: bool = False):
        """Init method."""
        self.band_indices: list | None = None
        self.band_paths: list[NDArray] | None = None
        self.band_points: int | None = None
        self.cell_filename: str | None = None
        self.chemical_symbols: list[str] | None = None
        self.classical: bool = False
        self.cutoff_frequency = None
        self.displacement_distance = None
        self.displacement_distance_max = None
        self.dm_decimals: int | None = None
        self.calculator = None
        self.create_displacements = False
        self.fc_calculator = None
        self.fc_calculator_options = None
        self.fc_decimals: int | None = None
        if load_phonopy_yaml:
            self.fc_symmetry = True
        else:
            self.fc_symmetry = False
        self.frequency_pitch = None
        self.frequency_conversion_factor = None
        self.frequency_scale_factor = None
        self.group_velocity_delta_q = None
        self.hdf5_compression = "gzip"
        self.is_band_const_interval = False
        self.is_diagonal_displacement = True
        self.is_eigenvectors = False
        self.is_mesh_symmetry = True
        if load_phonopy_yaml:
            self.is_nac = True
        else:
            self.is_nac = False
        self.is_plusminus_displacement: Literal["auto"] | bool = "auto"
        self.is_symmetry = True
        self.is_tetrahedron_method = True
        self.is_time_reversal_symmetry = True
        self.is_trigonal_displacement = False
        self.magnetic_moments = None
        self.masses = None
        self.mesh_numbers = None
        self.mlp_params = None
        self.nac_method = None
        self.nac_q_direction = None
        self.num_frequency_points = None
        self.relax_atomic_positions = False
        self.primitive_matrix = None
        self.qpoints: list | None = None
        self.random_displacements: Literal["auto"] | int | None = None
        self.random_seed = None
        self.rd_number_estimation_factor = None
        self.read_qpoints = False
        self.save_params = False
        self.sigma = None
        self.supercell_matrix = None
        self.symmetry_tolerance = None
        self.max_temperature = 1000
        self.min_temperature = 0
        self.temperature_step = 10
        self.use_pypolymlp = False


# Parse phonopy setting filen
class ConfParser:
    """Phonopy conf file parser."""

    def __init__(self):
        """Init method."""
        self._confs = {}
        self._parameters = {}

    @property
    def confs(self):
        """Return configuration dict."""
        return self._confs

    def setting_error(self, message):
        """Show error message."""
        print(message)
        print("Please check the setting tags and options.")
        sys.exit(1)

    def _read_file(self, filename: str | os.PathLike):
        """Read conf file."""
        with open(filename, "r") as file:
            is_continue = False
            left = None

            for line in file:
                if line.strip() == "":
                    is_continue = False
                    continue

                if line.strip()[0] == "#":
                    is_continue = False
                    continue

                if is_continue and left is not None:
                    self._confs[left] += line.strip()
                    self._confs[left] = self._confs[left].replace("+++", " ")
                    is_continue = False

                if line.find("=") != -1:
                    left, right = [x.strip() for x in line.split("=")]
                    self._confs[left.lower()] = right

                if line.find("+++") != -1:
                    is_continue = True

    def _read_options(self, args: argparse.Namespace):
        """Read options from ArgumentParser class instance.

        This is the interface layer to make settings from command options to be
        consistent with settings from conf file.

        """
        arg_list = vars(args)
        if "band_indices" in arg_list:
            band_indices = args.band_indices
            if band_indices is not None:
                if isinstance(band_indices, list):
                    self._confs["band_indices"] = " ".join(band_indices)
                else:
                    self._confs["band_indices"] = band_indices

        if "band_paths" in arg_list:
            if args.band_paths is not None:
                if isinstance(args.band_paths, list):
                    self._confs["band"] = " ".join(args.band_paths)
                else:
                    self._confs["band"] = args.band_paths

        if "band_points" in arg_list:
            if args.band_points is not None:
                self._confs["band_points"] = args.band_points

        if "cell_filename" in arg_list:
            if args.cell_filename is not None:
                self._confs["cell_filename"] = args.cell_filename

        if "classical" in arg_list:
            if args.classical:
                self._confs["classical"] = ".true."
            elif args.classical is False:
                self._confs["classical"] = ".false."

        if "cutoff_frequency" in arg_list:
            if args.cutoff_frequency:
                self._confs["cutoff_frequency"] = args.cutoff_frequency

        if "displacement_distance" in arg_list:
            if args.displacement_distance is not None:
                self._confs["displacement_distance"] = args.displacement_distance

        if "displacement_distance_max" in arg_list:
            if args.displacement_distance_max is not None:
                self._confs["displacement_distance_max"] = (
                    args.displacement_distance_max
                )

        if "dynamical_matrix_decimals" in arg_list:
            if args.dynamical_matrix_decimals:
                self._confs["dm_decimals"] = args.dynamical_matrix_decimals

        if "calculator" in arg_list:
            if args.calculator:
                self._confs["calculator"] = args.calculator

        if "fc_calculator" in arg_list:
            if args.fc_calculator:
                self._confs["fc_calculator"] = args.fc_calculator

        if "fc_calculator_options" in arg_list:
            fc_calc_opt = args.fc_calculator_options
            if fc_calc_opt:
                self._confs["fc_calculator_options"] = fc_calc_opt

        if "fc_symmetry" in arg_list:
            if args.fc_symmetry:
                self._confs["fc_symmetry"] = ".true."
            elif args.fc_symmetry is False:
                self._confs["fc_symmetry"] = ".false."

        if "force_constants_decimals" in arg_list:
            if args.force_constants_decimals:
                self._confs["fc_decimals"] = args.force_constants_decimals

        if "fpitch" in arg_list:
            if args.fpitch:
                self._confs["fpitch"] = args.fpitch

        if "frequency_conversion_factor" in arg_list:
            freq_factor = args.frequency_conversion_factor
            if freq_factor:
                self._confs["frequency_conversion_factor"] = freq_factor

        if "frequency_scale_factor" in arg_list:
            freq_scale = args.frequency_scale_factor
            if freq_scale is not None:
                self._confs["frequency_scale_factor"] = freq_scale

        if "gv_delta_q" in arg_list:
            if args.gv_delta_q:
                self._confs["gv_delta_q"] = args.gv_delta_q

        if "hdf5_compression" in arg_list:
            if args.hdf5_compression:
                self._confs["hdf5_compression"] = args.hdf5_compression

        if "is_band_const_interval" in arg_list:
            if args.is_band_const_interval:
                self._confs["band_const_interval"] = ".true."
            elif args.is_band_const_interval is False:
                self._confs["band_const_interval"] = ".false."

        if "is_displacement" in arg_list:
            if args.is_displacement:
                self._confs["create_displacements"] = ".true."
            elif args.is_displacement is False:
                self._confs["create_displacements"] = ".false."

        if "is_eigenvectors" in arg_list:
            if args.is_eigenvectors:
                self._confs["eigenvectors"] = ".true."
            elif args.is_eigenvectors is False:
                self._confs["eigenvectors"] = ".false."

        if "is_nac" in arg_list:
            if args.is_nac:
                self._confs["nac"] = ".true."
            elif args.is_nac is False:
                self._confs["nac"] = ".false."

        if "is_nodiag" in arg_list:
            if args.is_nodiag:
                self._confs["diag"] = ".false."
            elif args.is_nodiag is False:
                self._confs["diag"] = ".true."

        if "is_nomeshsym" in arg_list:
            if args.is_nomeshsym:
                self._confs["mesh_symmetry"] = ".false."
            elif args.is_nomeshsym is False:
                self._confs["mesh_symmetry"] = ".true."

        if "is_nosym" in arg_list:
            if args.is_nosym:
                self._confs["symmetry"] = ".false."
            elif args.is_nosym is False:
                self._confs["symmetry"] = ".true."

        # Default is "auto".
        if "is_plusminus_displacements" in arg_list:
            if args.is_plusminus_displacements:
                self._confs["pm"] = ".true."

        if "is_trigonal_displacements" in arg_list:
            if args.is_trigonal_displacements:
                self._confs["trigonal"] = ".true."
            elif args.is_trigonal_displacements is False:
                self._confs["trigonal"] = ".false."

        if "masses" in arg_list:
            if args.masses is not None:
                if isinstance(args.masses, list):
                    self._confs["mass"] = " ".join(args.masses)
                else:
                    self._confs["mass"] = args.masses

        if "magmoms" in arg_list:
            if args.magmoms is not None:
                if isinstance(args.magmoms, list):
                    self._confs["magmom"] = " ".join(args.magmoms)
                else:
                    self._confs["magmom"] = args.magmoms

        if "mesh_numbers" in arg_list:
            mesh = args.mesh_numbers
            if mesh is not None:
                if isinstance(mesh, list):
                    self._confs["mesh_numbers"] = " ".join(mesh)
                else:
                    self._confs["mesh_numbers"] = mesh

        if "mlp_params" in arg_list:
            mlp_params = args.mlp_params
            if mlp_params:
                self._confs["mlp_params"] = mlp_params

        if "nac_q_direction" in arg_list:
            q_dir = args.nac_q_direction
            if q_dir is not None:
                if isinstance(q_dir, list):
                    self._confs["q_direction"] = " ".join(q_dir)
                else:
                    self._confs["q_direction"] = q_dir

        if "nac_method" in arg_list:
            if args.nac_method is not None:
                self._confs["nac_method"] = args.nac_method

        if "num_frequency_points" in arg_list:
            opt_num_freqs = args.num_frequency_points
            if opt_num_freqs:
                self._confs["num_frequency_points"] = opt_num_freqs

        # For backword compatibility
        if "primitive_axis" in arg_list:
            if args.primitive_axis is not None:
                if isinstance(args.primitive_axis, list):
                    primitive_axes = " ".join(args.primitive_axis)
                    self._confs["primitive_axes"] = primitive_axes
                else:
                    self._confs["primitive_axes"] = args.primitive_axis

        if "primitive_axes" in arg_list:
            if args.primitive_axes:
                if isinstance(args.primitive_axes, list):
                    primitive_axes = " ".join(args.primitive_axes)
                    self._confs["primitive_axes"] = primitive_axes
                else:
                    self._confs["primitive_axes"] = args.primitive_axes

        if "qpoints" in arg_list:
            if args.qpoints is not None:
                if isinstance(args.qpoints, list):
                    self._confs["qpoints"] = " ".join(args.qpoints)
                else:
                    self._confs["qpoints"] = args.qpoints

        if "random_displacements" in arg_list:
            nrand = args.random_displacements
            if nrand:
                self._confs["random_displacements"] = nrand

        if "random_seed" in arg_list:
            if args.random_seed:
                seed = args.random_seed
                if np.issubdtype(type(seed), np.integer) and seed >= 0 and seed < 2**32:
                    self._confs["random_seed"] = seed

        if "rd_number_estimation_factor" in arg_list:
            if args.rd_number_estimation_factor is not None:
                factor = args.rd_number_estimation_factor
                self._confs["rd_number_estimation_factor"] = factor

        if "read_qpoints" in arg_list:
            if args.read_qpoints:
                self._confs["read_qpoints"] = ".true."
            elif args.read_qpoints is False:
                self._confs["read_qpoints"] = ".false."

        if "relax_atomic_positions" in arg_list:
            if args.relax_atomic_positions:
                self._confs["relax_atomic_positions"] = ".true."
            elif args.relax_atomic_positions is False:
                self._confs["relax_atomic_positions"] = ".false."

        if "save_params" in arg_list:
            if args.save_params:
                self._confs["save_params"] = ".true."
            elif args.save_params is False:
                self._confs["save_params"] = ".false."

        if "supercell_dimension" in arg_list:
            dim = args.supercell_dimension
            if dim is not None:
                if isinstance(dim, list):
                    self._confs["dim"] = " ".join(dim)
                else:
                    self._confs["dim"] = dim

        if "sigma" in arg_list:
            if args.sigma is not None:
                if isinstance(args.sigma, list):
                    self._confs["sigma"] = " ".join(args.sigma)
                else:
                    self._confs["sigma"] = args.sigma

        if "symmetry_tolerance" in arg_list:
            if args.symmetry_tolerance:
                symtol = args.symmetry_tolerance
                self._confs["symmetry_tolerance"] = symtol

        if "tmax" in arg_list:
            if args.tmax:
                self._confs["tmax"] = args.tmax

        if "tmin" in arg_list:
            if args.tmin:
                self._confs["tmin"] = args.tmin

        if "tstep" in arg_list:
            if args.tstep:
                self._confs["tstep"] = args.tstep

        from phonopy.interface.calculator import get_interface_mode

        calculator = get_interface_mode(arg_list)
        if calculator:
            self._confs["calculator"] = calculator

        if "use_alm" in arg_list:
            if args.use_alm:
                self._confs["fc_calculator"] = "alm"

        if "use_symfc" in arg_list:
            if args.use_symfc:
                self._confs["fc_calculator"] = "symfc"

        if "use_pypolymlp" in arg_list:
            if args.use_pypolymlp:
                self._confs["use_pypolymlp"] = ".true."

    def _parse_conf(self):
        """Add treatments to settings from conf file or command options.

        The results are stored in ``self._parameters[key] = val``.

        """
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == "atom_name":
                self._set_parameter(
                    "atom_name", [x.capitalize() for x in confs["atom_name"].split()]
                )

            if conf_key == "band":
                bands = []
                if confs["band"].strip().lower() == "auto":
                    self._set_parameter("band_paths", "auto")
                else:
                    for section in confs["band"].split(","):
                        points = [fracval(x) for x in section.split()]
                        if len(points) % 3 != 0 or len(points) < 6:
                            self.setting_error("BAND is incorrectly set.")
                            break
                        bands.append(np.array(points).reshape(-1, 3))
                    self._set_parameter("band_paths", bands)

            if conf_key == "band_const_interval":
                if confs["band_const_interval"].lower() == ".false.":
                    self._set_parameter("is_band_const_interval", False)
                elif confs["band_const_interval"].lower() == ".true.":
                    self._set_parameter("is_band_const_interval", True)

            if conf_key == "band_indices":
                vals = []
                for sum_set in confs["band_indices"].split(","):
                    vals.append([int(x) - 1 for x in sum_set.split()])
                self._set_parameter("band_indices", vals)

            if conf_key == "band_points":
                self._set_parameter("band_points", int(confs["band_points"]))

            if conf_key == "calculator":
                self._set_parameter("calculator", confs["calculator"])

            if conf_key == "cell_filename":
                self._set_parameter("cell_filename", confs["cell_filename"])

            if conf_key == "classical":
                if confs["classical"].lower() == ".false.":
                    self._set_parameter("classical", False)
                elif confs["classical"].lower() == ".true.":
                    self._set_parameter("classical", True)

            if conf_key == "create_displacements":
                if confs["create_displacements"].lower() == ".true.":
                    self._set_parameter("create_displacements", True)
                elif confs["create_displacements"].lower() == ".false.":
                    self._set_parameter("create_displacements", False)

            if conf_key == "cutoff_frequency":
                val = float(confs["cutoff_frequency"])
                self._set_parameter("cutoff_frequency", val)

            if conf_key == "diag":
                if confs["diag"].lower() == ".false.":
                    self._set_parameter("diag", False)
                elif confs["diag"].lower() == ".true.":
                    self._set_parameter("diag", True)

            if conf_key == "dim":
                matrix = [int(x) for x in confs["dim"].split()]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of DIM tag has to be 3 or 9."
                    )

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            "Determinant of supercell matrix has to be positive."
                        )
                    else:
                        self._set_parameter("supercell_matrix", matrix)

            if conf_key == "displacement_distance":
                self._set_parameter(
                    "displacement_distance", float(confs["displacement_distance"])
                )

            if conf_key == "displacement_distance_max":
                self._set_parameter(
                    "displacement_distance_max",
                    float(confs["displacement_distance_max"]),
                )

            if conf_key == "dm_decimals":
                self._set_parameter("dm_decimals", confs["dm_decimals"])

            if conf_key == "eigenvectors":
                if confs["eigenvectors"].lower() == ".false.":
                    self._set_parameter("is_eigenvectors", False)
                elif confs["eigenvectors"].lower() == ".true.":
                    self._set_parameter("is_eigenvectors", True)

            if conf_key == "fc_calculator":
                self._set_parameter("fc_calculator", confs["fc_calculator"])

            if conf_key == "fc_calculator_options":
                self._set_parameter(
                    "fc_calculator_options", confs["fc_calculator_options"]
                )

            if conf_key == "fc_symmetry":
                if confs["fc_symmetry"].lower() == ".false.":
                    self._set_parameter("fc_symmetry", False)
                elif confs["fc_symmetry"].lower() == ".true.":
                    self._set_parameter("fc_symmetry", True)

            if conf_key == "fc_decimals":
                self._set_parameter("fc_decimals", confs["fc_decimals"])

            if conf_key == "frequency_scale_factor":
                self._set_parameter(
                    "frequency_scale_factor", float(confs["frequency_scale_factor"])
                )

            if conf_key == "frequency_conversion_factor":
                val = float(confs["frequency_conversion_factor"])
                self._set_parameter("frequency_conversion_factor", val)

            if conf_key == "fpitch":
                val = float(confs["fpitch"])
                self._set_parameter("fpitch", val)

            # Group velocity finite difference
            if conf_key == "gv_delta_q":
                self._set_parameter("gv_delta_q", float(confs["gv_delta_q"]))

            # Compression option for writing int hdf5
            if conf_key == "hdf5_compression":
                hdf5_compression = confs["hdf5_compression"]
                try:
                    compression = int(hdf5_compression)
                except ValueError:  # str
                    compression = hdf5_compression
                    if compression.lower() == "none":
                        compression = None
                except TypeError:  # None (this will not happen)
                    compression = hdf5_compression
                self._set_parameter("hdf5_compression", compression)

            if conf_key == "magmom":
                self._set_parameter(
                    "magmom", [float(x) for x in confs["magmom"].split()]
                )

            if conf_key == "mass":
                self._set_parameter("mass", [float(x) for x in confs["mass"].split()])

            if conf_key in ["mesh_numbers", "mp", "mesh"]:
                vals = [x for x in confs[conf_key].split()]
                if len(vals) == 1:
                    self._set_parameter("mesh_numbers", float(vals[0]))
                elif len(vals) < 3:
                    self.setting_error("Mesh numbers are incorrectly set.")
                elif len(vals) == 3:
                    self._set_parameter("mesh_numbers", [int(x) for x in vals])
                elif len(vals) == 9:
                    mesh_array = []
                    for row in np.reshape([int(x) for x in vals], (3, 3)):
                        mesh_array.append(row.tolist())
                    self._set_parameter("mesh_numbers", mesh_array)
                else:
                    self.setting_error(f"{conf_key.upper()} is incorrectly set.")

            if conf_key == "mesh_symmetry":
                if confs["mesh_symmetry"].lower() == ".false.":
                    self._set_parameter("is_mesh_symmetry", False)
                elif confs["mesh_symmetry"].lower() == ".true.":
                    self._set_parameter("is_mesh_symmetry", True)

            if conf_key == "mlp_params":
                self._set_parameter("mlp_params", confs["mlp_params"])

            if conf_key == "nac":
                if confs["nac"].lower() == ".false.":
                    self._set_parameter("is_nac", False)
                elif confs["nac"].lower() == ".true.":
                    self._set_parameter("is_nac", True)

            if conf_key == "nac_method":
                self._set_parameter("nac_method", confs["nac_method"].lower())

            if conf_key == "num_frequency_points":
                val = int(confs["num_frequency_points"])
                self._set_parameter("num_frequency_points", val)

            if conf_key == "pm":
                if confs["pm"].lower() == ".false.":
                    self._set_parameter("pm", False)
                elif confs["pm"].lower() == ".true.":
                    self._set_parameter("pm", True)

            if conf_key in ("primitive_axis", "primitive_axes"):
                if confs[conf_key].strip().lower() == "auto":
                    self._set_parameter("primitive_axes", "auto")
                elif confs[conf_key].strip().upper() in ("P", "F", "I", "A", "C", "R"):
                    self._set_parameter(
                        "primitive_axes", confs[conf_key].strip().upper()
                    )
                elif not len(confs[conf_key].split()) == 9:
                    self.setting_error(
                        "Number of elements in %s has to be 9." % conf_key.upper()
                    )
                else:
                    p_axis = []
                    for x in confs[conf_key].split():
                        p_axis.append(fracval(x))
                    p_axis = np.array(p_axis).reshape(3, 3)
                    if np.linalg.det(p_axis) < 1e-8:
                        self.setting_error(
                            "%s has to have positive determinant." % conf_key.upper()
                        )
                    self._set_parameter("primitive_axes", p_axis)

            if conf_key == "q_direction":
                q_direction = [fracval(x) for x in confs["q_direction"].split()]
                if len(q_direction) < 3:
                    self.setting_error(
                        "Number of elements of q_direction is less than 3"
                    )
                else:
                    self._set_parameter("nac_q_direction", q_direction)

            if conf_key == "qpoints":
                if confs["qpoints"].lower() == ".true.":
                    self._set_parameter("read_qpoints", True)
                elif confs["qpoints"].lower() == ".false.":
                    self._set_parameter("read_qpoints", False)
                else:
                    vals = [fracval(x) for x in confs["qpoints"].split()]
                    if len(vals) == 0 or len(vals) % 3 != 0:
                        self.setting_error("Q-points are incorrectly set.")
                    else:
                        self._set_parameter("qpoints", list(np.reshape(vals, (-1, 3))))

            # Number of supercells with random displacements
            if conf_key == "random_displacements":
                rd = confs["random_displacements"]
                if rd.lower() == "auto":
                    self._set_parameter("random_displacements", "auto")
                else:
                    try:
                        self._set_parameter("random_displacements", int(rd))
                    except ValueError:
                        self.setting_error(f"{conf_key.upper()} is incorrectly set.")

            if conf_key == "random_seed":
                self._set_parameter("random_seed", int(confs["random_seed"]))

            if conf_key == "rd_number_estimation_factor":
                try:
                    factor = float(confs["rd_number_estimation_factor"])
                except ValueError:
                    self.setting_error("RD_NUMBER_ESTIMATION_FACTOR is not a number.")
                self._set_parameter("rd_number_estimation_factor", factor)

            if conf_key == "read_qpoints":
                if confs["read_qpoints"].lower() == ".false.":
                    self._set_parameter("read_qpoints", False)
                elif confs["read_qpoints"].lower() == ".true.":
                    self._set_parameter("read_qpoints", True)

            if conf_key == "relax_atomic_positions":
                if confs["relax_atomic_positions"].lower() == ".true.":
                    self._set_parameter("relax_atomic_positions", True)
                elif confs["relax_atomic_positions"].lower() == ".false.":
                    self._set_parameter("relax_atomic_positions", False)

            # Select yaml summary contents
            if conf_key == "save_params":
                if confs["save_params"].lower() == ".true.":
                    self._set_parameter("save_params", True)
                elif confs["save_params"].lower() == ".false.":
                    self._set_parameter("save_params", False)

            if conf_key == "sigma":
                vals = [float(x) for x in str(confs["sigma"]).split()]
                if len(vals) == 1:
                    self._set_parameter("sigma", vals[0])
                else:
                    self._set_parameter("sigma", vals)

            if conf_key == "symmetry":
                if confs["symmetry"].lower() == ".false.":
                    self._set_parameter("is_symmetry", False)
                    self._set_parameter("is_mesh_symmetry", False)
                elif confs["symmetry"].lower() == ".true.":
                    self._set_parameter("is_symmetry", True)

            if conf_key == "symmetry_tolerance":
                val = float(confs["symmetry_tolerance"])
                self._set_parameter("symmetry_tolerance", val)

            if conf_key == "tetrahedron":
                if confs["tetrahedron"].lower() == ".false.":
                    self._set_parameter("is_tetrahedron_method", False)
                if confs["tetrahedron"].lower() == ".true.":
                    self._set_parameter("is_tetrahedron_method", True)

            if conf_key == "time_reversal_symmetry":
                if confs["time_reversal_symmetry"].lower() == ".true.":
                    self._set_parameter("is_time_reversal_symmetry", True)
                elif confs["time_reversal_symmetry"].lower() == ".false.":
                    self._set_parameter("is_time_reversal_symmetry", False)

            if conf_key == "tmax":
                val = float(confs["tmax"])
                self._set_parameter("tmax", val)

            if conf_key == "tmin":
                val = float(confs["tmin"])
                self._set_parameter("tmin", val)

            if conf_key == "trigonal":
                if confs["trigonal"].lower() == ".false.":
                    self._set_parameter("is_trigonal_displacement", False)
                elif confs["trigonal"].lower() == ".true.":
                    self._set_parameter("is_trigonal_displacement", True)

            if conf_key == "tstep":
                val = float(confs["tstep"])
                self._set_parameter("tstep", val)

            if conf_key == "use_pypolymlp":
                if confs["use_pypolymlp"].lower() == ".true.":
                    self._set_parameter("use_pypolymlp", True)
                elif confs["use_pypolymlp"].lower() == ".false.":
                    self._set_parameter("use_pypolymlp", False)

    def _set_parameter(self, key, val):
        """Pass to another data structure."""
        self._parameters[key] = val

    def _set_settings(self, settings: Settings):
        """Store parameters in Settings class instance."""
        params = self._parameters

        # Chemical symbols
        if "atom_name" in params:
            settings.chemical_symbols = params["atom_name"]

        # Sets of band indices that are summed
        if "band_indices" in params:
            settings.band_indices = params["band_indices"]

        # Band paths
        if "band_paths" in params:
            settings.band_paths = params["band_paths"]

        # This number includes end points
        if "band_points" in params:
            settings.band_points = params["band_points"]

        # Force calculator
        if "calculator" in params:
            settings.calculator = params["calculator"]

        # Filename of input unit cell
        if "cell_filename" in params:
            settings.cell_filename = params["cell_filename"]

        # Is getting least displacements?
        if "create_displacements" in params:
            settings.create_displacements = params["create_displacements"]

        # Treat statistics classically?
        if "classical" in params:
            settings.classical = params["classical"]

        # Cutoff frequency
        if "cutoff_frequency" in params:
            settings.cutoff_frequency = params["cutoff_frequency"]

        # Diagonal displacement
        if "diag" in params:
            settings.is_diagonal_displacement = params["diag"]

        # Distance of finite displacements introduced
        if "displacement_distance" in params:
            settings.displacement_distance = params["displacement_distance"]

        if "displacement_distance_max" in params:
            settings.displacement_distance_max = params["displacement_distance_max"]

        # Decimals of values of dynamical matrxi
        if "dm_decimals" in params:
            settings.dm_decimals = int(params["dm_decimals"])

        # Force constants calculator
        if "fc_calculator" in params:
            settings.fc_calculator = params["fc_calculator"]

        # Force constants calculator options as str
        if "fc_calculator_options" in params:
            settings.fc_calculator_options = params["fc_calculator_options"]

        # Decimals of values of force constants
        if "fc_decimals" in params:
            settings.fc_decimals = int(params["fc_decimals"])

        # Enforce translational invariance and index permutation symmetry
        # to force constants?
        if "fc_symmetry" in params:
            settings.fc_symmetry = params["fc_symmetry"]

        # Frequency unit conversion factor
        if "frequency_conversion_factor" in params:
            settings.frequency_conversion_factor = params["frequency_conversion_factor"]

        # This scale factor is multiplied to force constants by
        # fc * scale_factor ** 2, therefore only changes
        # frequencies but does not change NAC part.
        if "frequency_scale_factor" in params:
            settings.frequency_scale_factor = params["frequency_scale_factor"]

        # Spectram drawing step
        if "fpitch" in params:
            settings.frequency_pitch = params["fpitch"]

        # Group velocity finite difference
        if "gv_delta_q" in params:
            settings.group_velocity_delta_q = params["gv_delta_q"]

        # Is getting eigenvectors?
        if "is_eigenvectors" in params:
            settings.is_eigenvectors = params["is_eigenvectors"]

        # Is reciprocal mesh symmetry searched?
        if "is_mesh_symmetry" in params:
            settings.is_mesh_symmetry = params["is_mesh_symmetry"]

        # Non analytical term correction?
        if "is_nac" in params:
            settings.is_nac = params["is_nac"]

        # Is crystal symmetry searched?
        if "is_symmetry" in params:
            settings.is_symmetry = params["is_symmetry"]

        # Tetrahedron method
        if "is_tetrahedron_method" in params:
            settings.is_tetrahedron_method = params["is_tetrahedron_method"]

        if "is_time_reversal_symmetry" in params:
            settings.is_time_reversal_symmetry = params["is_time_reversal_symmetry"]

        # Trigonal displacement
        if "is_trigonal_displacement" in params:
            settings.is_trigonal_displacement = params["is_trigonal_displacement"]

        if "is_band_const_interval" in params:
            settings.is_band_const_interval = params["is_band_const_interval"]

        # Compression option for writing int hdf5
        if "hdf5_compression" in params:
            settings.hdf5_compression = params["hdf5_compression"]

        # Magnetic moments
        if "magmom" in params:
            settings.magnetic_moments = params["magmom"]

        # Atomic mass
        if "mass" in params:
            settings.masses = params["mass"]

        # Mesh sampling numbers
        if "mesh_numbers" in params:
            settings.mesh_numbers = params["mesh_numbers"]

        if "mlp_params" in params:
            settings.mlp_params = params["mlp_params"]

        # non analytical term correction method
        if "nac_method" in params:
            settings.nac_method = params["nac_method"]

        # q-direction for non analytical term correction
        if "nac_q_direction" in params:
            settings.nac_q_direction = params["nac_q_direction"]

        # Number of sampling points for spectram drawing
        if "num_frequency_points" in params:
            settings.num_frequency_points = params["num_frequency_points"]

        # Plus minus displacement
        if "pm" in params:
            settings.is_plusminus_displacement = params["pm"]

        # Primitive cell shape
        if "primitive_axes" in params:
            settings.primitive_matrix = params["primitive_axes"]

        # Q-points mode
        if "qpoints" in params:
            settings.qpoints = params["qpoints"]

        # Number of supercells with random displacements
        if "random_displacements" in params:
            settings.random_displacements = params["random_displacements"]

        if "random_seed" in params:
            settings.random_seed = params["random_seed"]

        # Random displacements number estimation factor
        if "rd_number_estimation_factor" in params:
            settings.rd_number_estimation_factor = params["rd_number_estimation_factor"]

        if "read_qpoints" in params:
            settings.read_qpoints = params["read_qpoints"]

        if "relax_atomic_positions" in params:
            settings.relax_atomic_positions = params["relax_atomic_positions"]

        # Smearing width
        if "sigma" in params:
            settings.sigma = params["sigma"]

        # Symmetry tolerance
        if "symmetry_tolerance" in params:
            settings.symmetry_tolerance = params["symmetry_tolerance"]

        # Supercell size
        if "supercell_matrix" in params:
            settings.supercell_matrix = params["supercell_matrix"]

        # Temperatures or temerature range
        if "tmax" in params:
            settings.max_temperature = params["tmax"]
        if "tmin" in params:
            settings.min_temperature = params["tmin"]
        if "tstep" in params:
            settings.temperature_step = params["tstep"]

        # Select yaml summary contents
        if "save_params" in params:
            settings.save_params = params["save_params"]

        # Machine learning potential
        if "use_pypolymlp" in params:
            settings.use_pypolymlp = params["use_pypolymlp"]


#
# For phonopy
#
class PhonopySettings(Settings):
    """Phonopy settings container.

    Basic part is stored in Settings and extended part is stored in this class.

    This works almost like a dictionary.
    Method names without 'set_' and 'get_' and keys of self._v have to be same.

    """

    def __init__(self, load_phonopy_yaml: bool = False):
        """Init method."""
        super().__init__(load_phonopy_yaml=load_phonopy_yaml)
        self.anime_band_index: int | None = None
        self.anime_amplitude: float | None = None
        self.anime_division: int | None = None
        self.anime_qpoint: list | None = None
        self.anime_shift: Sequence[float] | None = None
        self.anime_type = "v_sim"
        self.band_format = "yaml"
        self.band_labels = None
        self.create_force_sets = None
        self.create_force_sets_zero = None
        self.create_force_constants = None
        self.cutoff_radius = None
        self.dos = None
        self.fc_spg_symmetry = False
        self.fits_Debye_model = False
        self.max_frequency = None
        self.min_frequency = None
        self.irreps_q_point: list | None = None
        self.irreps_tolerance: float | None = None
        self.is_band_connection = False
        self.is_dos_mode = False
        self.is_full_fc = False
        self.is_group_velocity = False
        self.is_gamma_center = False
        self.is_hdf5 = False
        self.is_legacy_plot = False
        self.is_little_cogroup = False
        self.is_moment = False
        self.is_thermal_displacements = False
        self.is_thermal_displacement_matrices = False
        self.is_thermal_distances = False
        self.is_thermal_properties = False
        self.is_projected_thermal_properties = False
        self.include_force_constants = False
        self.include_force_sets = False
        self.include_nac_params = True
        self.include_displacements = False
        self.lapack_solver = False
        self.mesh_shift = None
        self.mesh_format = "yaml"
        self.modulation: dict | None = None
        self.moment_order = None
        self.pdos_indices: list | None = None
        self.pretend_real = False
        self.projection_direction = None
        self.qpoints_format = "yaml"
        self.random_displacement_temperature = None
        if load_phonopy_yaml:
            self.read_force_constants = True
        else:
            self.read_force_constants = False
        self.readfc_format = "text"
        self.run_mode: str | None = None
        self.show_irreps = False
        self.sscha_iterations = None
        self.store_dense_svecs = True
        self.thermal_atom_pairs = None
        self.thermal_displacement_matrix_temperature = None
        self.write_dynamical_matrices = False
        self.write_mesh = True
        self.write_force_constants = False
        self.writefc_format = "text"
        self.xyz_projection = False


class PhonopyConfParser(ConfParser):
    """Phonopy conf parser.

    Attributes
    ----------
    settings : PhonopySettings
        Phonopy settings container.
    confs : dict
        Dictionary of settings read from conf file or command options.

    """

    def __init__(
        self,
        filename: str | os.PathLike | None = None,
        args: argparse.Namespace | None = None,
        load_phonopy_yaml: bool = False,
    ):
        """Init method."""
        super().__init__()
        if filename is not None:
            self._read_file(filename)
        if args is not None:
            self._read_options(args)
        self._parse_conf()
        self.settings = PhonopySettings(load_phonopy_yaml=load_phonopy_yaml)
        self._set_settings(self.settings)

    def _read_options(self, args: argparse.Namespace):
        super()._read_options(args)  # store data in self._confs
        arg_list = vars(args)
        if "band_format" in arg_list:
            if args.band_format:
                self._confs["band_format"] = args.band_format

        if "band_labels" in arg_list:
            if args.band_labels is not None:
                self._confs["band_labels"] = " ".join(args.band_labels)

        if "is_gamma_center" in arg_list:
            if args.is_gamma_center:
                self._confs["gamma_center"] = ".true."

        if "create_force_sets" in arg_list:
            if args.create_force_sets:
                self._confs["create_force_sets"] = args.create_force_sets

        if "create_force_sets_zero" in arg_list:
            if args.create_force_sets_zero:
                fc_sets_zero = args.create_force_sets_zero
                self._confs["create_force_sets_zero"] = fc_sets_zero

        if "create_force_constants" in arg_list:
            if args.create_force_constants is not None:
                fc_filename = args.create_force_constants
                self._confs["create_force_constants"] = fc_filename

        if "is_dos_mode" in arg_list:
            if args.is_dos_mode:
                self._confs["dos"] = ".true."

        if "pdos" in arg_list:
            if args.pdos is not None:
                self._confs["pdos"] = " ".join(args.pdos)

        if "xyz_projection" in arg_list:
            if args.xyz_projection:
                self._confs["xyz_projection"] = ".true."

        if "fc_spg_symmetry" in arg_list:
            if args.fc_spg_symmetry:
                self._confs["fc_spg_symmetry"] = ".true."

        if "is_full_fc" in arg_list:
            if args.is_full_fc:
                self._confs["full_force_constants"] = ".true."

        if "fits_debye_model" in arg_list:
            if args.fits_debye_model:
                self._confs["debye_model"] = ".true."

        if "fmax" in arg_list:
            if args.fmax is not None:
                self._confs["fmax"] = args.fmax

        if "fmin" in arg_list:
            if args.fmin is not None:
                self._confs["fmin"] = args.fmin

        if "is_thermal_properties" in arg_list:
            if args.is_thermal_properties:
                self._confs["tprop"] = ".true."

        if "pretend_real" in arg_list:
            if args.pretend_real:
                self._confs["pretend_real"] = ".true."

        if "is_projected_thermal_properties" in arg_list:
            if args.is_projected_thermal_properties:
                self._confs["ptprop"] = ".true."

        if "is_thermal_displacements" in arg_list:
            if args.is_thermal_displacements:
                self._confs["tdisp"] = ".true."

        if "is_thermal_displacement_matrices" in arg_list:
            if args.is_thermal_displacement_matrices:
                self._confs["tdispmat"] = ".true."

        if "thermal_displacement_matrices_cif" in arg_list:
            opt_tdm_cif = args.thermal_displacement_matrices_cif
            if opt_tdm_cif:
                self._confs["tdispmat_cif"] = opt_tdm_cif

        if "projection_direction" in arg_list:
            opt_proj_dir = args.projection_direction
            if opt_proj_dir is not None:
                self._confs["projection_direction"] = " ".join(opt_proj_dir)

        if "read_force_constants" in arg_list:
            if args.read_force_constants:
                self._confs["read_force_constants"] = ".true."
            elif args.read_force_constants is False:
                self._confs["read_force_constants"] = ".false."

        if "write_force_constants" in arg_list:
            if args.write_force_constants:
                self._confs["write_force_constants"] = ".true."

        if "readfc_format" in arg_list:
            if args.readfc_format:
                self._confs["readfc_format"] = args.readfc_format

        if "writefc_format" in arg_list:
            if args.writefc_format:
                self._confs["writefc_format"] = args.writefc_format

        if "fc_format" in arg_list:
            if args.fc_format:
                self._confs["fc_format"] = args.fc_format

        if "is_hdf5" in arg_list:
            if args.is_hdf5:
                self._confs["hdf5"] = ".true."

        if "write_dynamical_matrices" in arg_list:
            if args.write_dynamical_matrices:
                self._confs["writedm"] = ".true."

        if "write_mesh" in arg_list:
            if args.write_mesh is False:
                self._confs["write_mesh"] = ".false."

        if "mesh_format" in arg_list:
            if args.mesh_format:
                self._confs["mesh_format"] = args.mesh_format

        if "qpoints_format" in arg_list:
            if args.qpoints_format:
                self._confs["qpoints_format"] = args.qpoints_format

        if "irreps_qpoint" in arg_list:
            if args.irreps_qpoint is not None:
                self._confs["irreps"] = " ".join(args.irreps_qpoint)

        if "show_irreps" in arg_list:
            if args.show_irreps:
                self._confs["show_irreps"] = ".true."

        if "is_little_cogroup" in arg_list:
            if args.is_little_cogroup:
                self._confs["little_cogroup"] = ".true."

        if "is_legacy_plot" in arg_list:
            if args.is_legacy_plot:
                self._confs["legacy_plot"] = ".true."

        if "is_band_connection" in arg_list:
            if args.is_band_connection:
                self._confs["band_connection"] = ".true."

        if "cutoff_radius" in arg_list:
            if args.cutoff_radius:
                self._confs["cutoff_radius"] = args.cutoff_radius

        if "modulation" in arg_list:
            if args.modulation:
                self._confs["modulation"] = " ".join(args.modulation)

        if "anime" in arg_list:
            if args.anime:
                self._confs["anime"] = " ".join(args.anime)

        if "is_group_velocity" in arg_list:
            if args.is_group_velocity:
                self._confs["group_velocity"] = ".true."

        if "is_moment" in arg_list:
            if args.is_moment:
                self._confs["moment"] = ".true."

        if "moment_order" in arg_list:
            if args.moment_order:
                self._confs["moment_order"] = args.moment_order

        if "rd_temperature" in arg_list:
            if args.rd_temperature is not None:
                self._confs["random_displacement_temperature"] = args.rd_temperature

        if "temperature" in arg_list:
            if args.temperature is not None:
                print(
                    "*****************************************************************"
                )
                print(
                    "--temperature option is deprecated. Use --rd-temperature instead."
                )
                print(
                    "*****************************************************************"
                )
                print()
                self._confs["random_displacement_temperature"] = args.temperature

        if "include_fc" in arg_list:
            if args.include_fc:
                self._confs["include_fc"] = ".true."

        if "include_fs" in arg_list:
            if args.include_fs:
                self._confs["include_fs"] = ".true."

        if "include_nac_params" in arg_list:
            if args.include_nac_params:
                self._confs["include_nac_params"] = ".true."
            elif args.include_nac_params is False:
                self._confs["include_nac_params"] = ".false."

        if "include_disp" in arg_list:
            if args.include_disp:
                self._confs["include_disp"] = ".true."

        if "include_all" in arg_list:
            if args.include_all:
                self._confs["include_all"] = ".true."

        if "store_dense_svecs" in arg_list:
            if args.store_dense_svecs:
                self._confs["store_dense_svecs"] = ".true."
            else:
                self._confs["store_dense_svecs"] = ".false."

        if "lapack_solver" in arg_list:
            if args.lapack_solver:
                self._confs["lapack_solver"] = ".true."

        if "is_check_symmetry" in arg_list:
            if args.is_check_symmetry:
                if "dim" not in self._confs:
                    # Dummy 'dim' setting not to exit by no-dim check.
                    self._confs["dim"] = "1 1 1"

        if "sscha_iterations" in arg_list:
            if args.sscha_iterations:
                self._confs["sscha_iterations"] = args.sscha_iterations

    def _parse_conf(self):
        super()._parse_conf()
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == "band_format":
                self._set_parameter("band_format", confs["band_format"].lower())

            if conf_key == "band_labels":
                labels = confs["band_labels"].split()
                self._set_parameter("band_labels", labels)

            if conf_key == "band_connection":
                if confs["band_connection"].lower() == ".true.":
                    self._set_parameter("band_connection", True)
                elif confs["band_connection"].lower() == ".false.":
                    self._set_parameter("band_connection", False)

            if conf_key == "legacy_plot":
                if confs["legacy_plot"].lower() == ".true.":
                    self._set_parameter("legacy_plot", True)
                elif confs["legacy_plot"].lower() == ".false.":
                    self._set_parameter("legacy_plot", False)

            if conf_key == "create_force_sets":
                if isinstance(confs["create_force_sets"], str):
                    fnames = confs["create_force_sets"].split()
                else:
                    fnames = confs["create_force_sets"]
                self._set_parameter("create_force_sets", fnames)

            if conf_key == "create_force_sets_zero":
                if isinstance(confs["create_force_sets_zero"], str):
                    fnames = confs["create_force_sets_zero"].split()
                else:
                    fnames = confs["create_force_sets_zero"]

                self._set_parameter("create_force_sets_zero", fnames)

            if conf_key == "create_force_constants":
                self._set_parameter(
                    "create_force_constants", confs["create_force_constants"]
                )

            if conf_key == "force_constants":
                self._set_parameter("force_constants", confs["force_constants"].lower())

            if conf_key == "read_force_constants":
                if confs["read_force_constants"].lower() == ".true.":
                    self._set_parameter("read_force_constants", True)
                elif confs["read_force_constants"].lower() == ".false.":
                    self._set_parameter("read_force_constants", False)

            if conf_key == "write_force_constants":
                if confs["write_force_constants"].lower() == ".true.":
                    self._set_parameter("write_force_constants", True)
                elif confs["write_force_constants"].lower() == ".false.":
                    self._set_parameter("write_force_constants", False)

            if conf_key == "full_force_constants":
                if confs["full_force_constants"].lower() == ".true.":
                    self._set_parameter("is_full_fc", True)
                elif confs["full_force_constants"].lower() == ".false.":
                    self._set_parameter("is_full_fc", False)

            if conf_key == "cutoff_radius":
                val = float(confs["cutoff_radius"])
                self._set_parameter("cutoff_radius", val)

            if conf_key == "writedm":
                if confs["writedm"].lower() == ".true.":
                    self._set_parameter("write_dynamical_matrices", True)
                elif confs["writedm"].lower() == ".false.":
                    self._set_parameter("write_dynamical_matrices", False)

            if conf_key == "write_mesh":
                if confs["write_mesh"].lower() == ".true.":
                    self._set_parameter("write_mesh", True)
                elif confs["write_mesh"].lower() == ".false.":
                    self._set_parameter("write_mesh", False)

            if conf_key == "hdf5":
                if confs["hdf5"].lower() == ".true.":
                    self._set_parameter("hdf5", True)
                elif confs["hdf5"].lower() == ".false.":
                    self._set_parameter("hdf5", False)

            if conf_key == "mp_shift":
                vals = [fracval(x) for x in confs["mp_shift"].split()]
                if len(vals) < 3:
                    self.setting_error("MP_SHIFT is incorrectly set.")
                self._set_parameter("mp_shift", vals[:3])

            if conf_key == "mesh_format":
                self._set_parameter("mesh_format", confs["mesh_format"].lower())

            if conf_key == "qpoints_format":
                self._set_parameter("qpoints_format", confs["qpoints_format"].lower())

            if conf_key == "gamma_center":
                if confs["gamma_center"].lower() == ".true.":
                    self._set_parameter("is_gamma_center", True)
                elif confs["gamma_center"].lower() == ".false.":
                    self._set_parameter("is_gamma_center", False)

            if conf_key == "fc_spg_symmetry":
                if confs["fc_spg_symmetry"].lower() == ".true.":
                    self._set_parameter("fc_spg_symmetry", True)
                elif confs["fc_spg_symmetry"].lower() == ".false.":
                    self._set_parameter("fc_spg_symmetry", False)

            if conf_key == "readfc_format":
                self._set_parameter("readfc_format", confs["readfc_format"].lower())

            if conf_key == "writefc_format":
                self._set_parameter("writefc_format", confs["writefc_format"].lower())

            if conf_key == "fc_format":
                self._set_parameter("readfc_format", confs["fc_format"].lower())
                self._set_parameter("writefc_format", confs["fc_format"].lower())

            # Animation
            if conf_key == "anime":
                vals = []
                data = confs["anime"].split()
                if len(data) < 3:
                    self.setting_error("ANIME is incorrectly set.")
                else:
                    self._set_parameter("anime", data)

            if conf_key == "anime_type":
                anime_type = confs["anime_type"].lower()
                if anime_type in ("arc", "v_sim", "poscar", "xyz", "jmol"):
                    self._set_parameter("anime_type", anime_type)
                else:
                    self.setting_error(
                        "%s is not available for ANIME_TYPE tag." % confs["anime_type"]
                    )

            # Modulation
            if conf_key == "modulation":
                self._parse_conf_modulation(confs["modulation"])

            # Character table
            if conf_key == "irreps":
                vals = [fracval(x) for x in confs["irreps"].split()]
                if len(vals) == 3 or len(vals) == 4:
                    self._set_parameter("irreps_qpoint", vals)
                else:
                    self.setting_error("IRREPS is incorrectly set.")

            if conf_key == "show_irreps":
                if confs["show_irreps"].lower() == ".true.":
                    self._set_parameter("show_irreps", True)
                elif confs["show_irreps"].lower() == ".false.":
                    self._set_parameter("show_irreps", False)

            if conf_key == "little_cogroup":
                if confs["little_cogroup"].lower() == ".true.":
                    self._set_parameter("little_cogroup", True)
                elif confs["little_cogroup"].lower() == ".false.":
                    self._set_parameter("little_cogroup", False)

            # DOS
            if conf_key == "pdos":
                if confs["pdos"].strip().lower() == "auto":
                    self._set_parameter("pdos", "auto")
                else:
                    vals = []
                    for index_set in confs["pdos"].split(","):
                        vals.append([int(x) - 1 for x in index_set.split()])
                    self._set_parameter("pdos", vals)

            if conf_key == "xyz_projection":
                if confs["xyz_projection"].lower() == ".true.":
                    self._set_parameter("xyz_projection", True)
                elif confs["xyz_projection"].lower() == ".false.":
                    self._set_parameter("xyz_projection", False)

            if conf_key == "dos":
                if confs["dos"].lower() == ".true.":
                    self._set_parameter("dos", True)
                elif confs["dos"].lower() == ".false.":
                    self._set_parameter("dos", False)

            if conf_key == "debye_model":
                if confs["debye_model"].lower() == ".true.":
                    self._set_parameter("fits_debye_model", True)
                elif confs["debye_model"].lower() == ".false.":
                    self._set_parameter("fits_debye_model", False)

            if conf_key == "dos_range":
                vals = [float(x) for x in confs["dos_range"].split()]
                self._set_parameter("dos_range", vals)

            if conf_key == "fmax":
                self._set_parameter("fmax", float(confs["fmax"]))

            if conf_key == "fmin":
                self._set_parameter("fmin", float(confs["fmin"]))

            # Thermal properties
            if conf_key == "tprop":
                if confs["tprop"].lower() == ".true.":
                    self._set_parameter("tprop", True)
                if confs["tprop"].lower() == ".false.":
                    self._set_parameter("tprop", False)

            # Projected thermal properties
            if conf_key == "ptprop":
                if confs["ptprop"].lower() == ".true.":
                    self._set_parameter("ptprop", True)
                elif confs["ptprop"].lower() == ".false.":
                    self._set_parameter("ptprop", False)

            # Use imaginary frequency as real for thermal property calculation
            if conf_key == "pretend_real":
                if confs["pretend_real"].lower() == ".true.":
                    self._set_parameter("pretend_real", True)
                elif confs["pretend_real"].lower() == ".false.":
                    self._set_parameter("pretend_real", False)

            # Thermal displacement
            if conf_key == "tdisp":
                if confs["tdisp"].lower() == ".true.":
                    self._set_parameter("tdisp", True)
                elif confs["tdisp"].lower() == ".false.":
                    self._set_parameter("tdisp", False)

            # Thermal displacement matrices
            if conf_key == "tdispmat":
                if confs["tdispmat"].lower() == ".true.":
                    self._set_parameter("tdispmat", True)
                elif confs["tdispmat"].lower() == ".false.":
                    self._set_parameter("tdispmat", False)

            # Write thermal displacement matrices to cif file,
            # for which the temperature to execute is stored.
            if conf_key == "tdispmat_cif":
                self._set_parameter("tdispmat_cif", float(confs["tdispmat_cif"]))

            # Thermal distance
            if conf_key == "tdistance":
                atom_pairs = []
                for atoms in confs["tdistance"].split(","):
                    pair = [int(x) - 1 for x in atoms.split()]
                    if len(pair) == 2:
                        atom_pairs.append(pair)
                    else:
                        self.setting_error("TDISTANCE is incorrectly specified.")
                if len(atom_pairs) > 0:
                    self._set_parameter("tdistance", atom_pairs)

            # Projection direction used for thermal displacements and PDOS
            if conf_key == "projection_direction":
                vals = [float(x) for x in confs["projection_direction"].split()]
                if len(vals) < 3:
                    self.setting_error(
                        "PROJECTION_DIRECTION (--pd) is incorrectly specified."
                    )
                else:
                    self._set_parameter("projection_direction", vals)

            # Group velocity
            if conf_key == "group_velocity":
                if confs["group_velocity"].lower() == ".true.":
                    self._set_parameter("is_group_velocity", True)
                elif confs["group_velocity"].lower() == ".false.":
                    self._set_parameter("is_group_velocity", False)

            # Moment of phonon states distribution
            if conf_key == "moment":
                if confs["moment"].lower() == ".true.":
                    self._set_parameter("moment", True)
                elif confs["moment"].lower() == ".false.":
                    self._set_parameter("moment", False)

            if conf_key == "moment_order":
                self._set_parameter("moment_order", int(confs["moment_order"]))

            if conf_key == "random_displacement_temperature":
                val = confs["random_displacement_temperature"]
                self._set_parameter("random_displacement_temperature", float(val))

            # Use Lapack solver via Lapacke
            if conf_key == "lapack_solver":
                if confs["lapack_solver"].lower() == ".true.":
                    self._set_parameter("lapack_solver", True)
                elif confs["lapack_solver"].lower() == ".false.":
                    self._set_parameter("lapack_solver", False)

            if conf_key == "include_fc":
                if confs["include_fc"].lower() == ".true.":
                    self._set_parameter("include_fc", True)
                elif confs["include_fc"].lower() == ".false.":
                    self._set_parameter("include_fc", False)

            if conf_key == "include_fs":
                if confs["include_fs"].lower() == ".true.":
                    self._set_parameter("include_fs", True)
                elif confs["include_fs"].lower() == ".false.":
                    self._set_parameter("include_fs", False)

            if conf_key in ("include_born", "include_nac_params"):
                if confs[conf_key].lower() == ".true.":
                    self._set_parameter("include_nac_params", True)
                elif confs[conf_key].lower() == ".false.":
                    self._set_parameter("include_nac_params", False)

            if conf_key == "include_disp":
                if confs["include_disp"].lower() == ".true.":
                    self._set_parameter("include_disp", True)
                elif confs["include_disp"].lower() == ".false.":
                    self._set_parameter("include_disp", False)

            if conf_key == "include_all":
                if confs["include_all"].lower() == ".true.":
                    self._set_parameter("include_all", True)
                elif confs["include_all"].lower() == ".false.":
                    self._set_parameter("include_all", False)

            # Pair shortest vectors in supercell are stored in dense format.
            if conf_key == "store_dense_svecs":
                if confs["store_dense_svecs"].lower() == ".true.":
                    self._set_parameter("store_dense_svecs", True)

            # SSCHA
            if conf_key == "sscha_iterations":
                val = int(confs["sscha_iterations"])
                self._set_parameter("sscha_iterations", val)

    def _parse_conf_modulation(self, conf_modulation):
        modulation = {}
        modulation["dimension"] = [1, 1, 1]
        modulation["order"] = None
        mod_list = conf_modulation.split(",")
        header = mod_list[0].split()
        if len(header) > 2 and len(mod_list) > 1:
            if len(header) > 8:
                dimension = [int(x) for x in header[:9]]
                modulation["dimension"] = dimension
                if len(header) > 11:
                    delta_q = [float(x) for x in header[9:12]]
                    modulation["delta_q"] = delta_q
                if len(header) == 13:
                    modulation["order"] = int(header[12])
            else:
                dimension = [int(x) for x in header[:3]]
                modulation["dimension"] = dimension
                if len(header) > 3:
                    delta_q = [float(x) for x in header[3:6]]
                    modulation["delta_q"] = delta_q
                if len(header) == 7:
                    modulation["order"] = int(header[6])

            vals = []
            for phonon_mode in mod_list[1:]:
                mode_conf = [x for x in phonon_mode.split()]
                if len(mode_conf) < 4 or len(mode_conf) > 6:
                    self.setting_error("MODULATION tag is wrongly set.")
                    break
                else:
                    q = [fracval(x) for x in mode_conf[:3]]

                if len(mode_conf) == 4:
                    vals.append([q, int(mode_conf[3]) - 1, 1.0, 0])
                elif len(mode_conf) == 5:
                    vals.append([q, int(mode_conf[3]) - 1, float(mode_conf[4]), 0])
                else:
                    vals.append(
                        [
                            q,
                            int(mode_conf[3]) - 1,
                            float(mode_conf[4]),
                            float(mode_conf[5]),
                        ]
                    )

            modulation["modulations"] = vals
            self._set_parameter("modulation", modulation)
        else:
            self.setting_error("MODULATION tag is wrongly set.")

    def _set_settings(self, settings: PhonopySettings):
        super()._set_settings(settings)
        params = self._parameters

        # Create FORCE_SETS
        if "create_force_sets" in params:
            settings.create_force_sets = params["create_force_sets"]

        if "create_force_sets_zero" in params:
            settings.create_force_sets_zero = params["create_force_sets_zero"]

        if "create_force_constants" in params:
            settings.create_force_constants = params["create_force_constants"]

        # Is force constants written or read?
        if "force_constants" in params:
            if params["force_constants"] == "write":
                settings.write_force_constants = True
            elif params["force_constants"] == "read":
                settings.read_force_constants = True

        if "read_force_constants" in params:
            settings.read_force_constants = params["read_force_constants"]

        if "write_force_constants" in params:
            settings.write_force_constants = params["write_force_constants"]

        if "is_full_fc" in params:
            settings.is_full_fc = params["is_full_fc"]

        # Enforce space group symmetyr to force constants?
        if "fc_spg_symmetry" in params:
            settings.fc_spg_symmetry = params["fc_spg_symmetry"]

        if "readfc_format" in params:
            settings.readfc_format = params["readfc_format"]

        if "writefc_format" in params:
            settings.writefc_format = params["writefc_format"]

        # Use hdf5?
        if "hdf5" in params:
            settings.is_hdf5 = params["hdf5"]

        # Cutoff radius of force constants
        if "cutoff_radius" in params:
            settings.cutoff_radius = params["cutoff_radius"]

        # Mesh
        if "mesh_numbers" in params:
            settings.run_mode = "mesh"
            settings.mesh_numbers = params["mesh_numbers"]
        if "mp_shift" in params:
            settings.mesh_shift = params["mp_shift"]
        if "is_mesh_symmetry" in params:
            settings.is_mesh_symmetry = params["is_mesh_symmetry"]
        if "is_gamma_center" in params:
            settings.is_gamma_center = params["is_gamma_center"]
        if "mesh_format" in params:
            settings.mesh_format = params["mesh_format"]

        # band mode
        if "band_paths" in params:
            settings.run_mode = "band"
        if "band_format" in params:
            settings.band_format = params["band_format"]
        if "band_labels" in params:
            settings.band_labels = params["band_labels"]
        if "band_connection" in params:
            settings.is_band_connection = params["band_connection"]
        if "legacy_plot" in params:
            settings.is_legacy_plot = params["legacy_plot"]

        # Q-points mode
        if "qpoints" in params or "read_qpoints" in params:
            settings.run_mode = "qpoints"
            if "qpoints_format" in params:
                settings.qpoints_format = params["qpoints_format"]

        # Whether write out dynamical matrices or not
        if "write_dynamical_matrices" in params:
            settings.write_dynamical_matrices = params["write_dynamical_matrices"]

        # Whether write out mesh.yaml or mesh.hdf5
        if "write_mesh" in params:
            settings.write_mesh = params["write_mesh"]

        # Anime mode
        if "anime_type" in params:
            settings.anime_type = params["anime_type"]

        if "anime" in params:
            settings.run_mode = "anime"
            anime_type = settings.anime_type
            if anime_type == "v_sim":
                qpoints = [fracval(x) for x in params["anime"][0:3]]
                settings.anime_qpoint = qpoints
                if len(params["anime"]) > 3:
                    settings.anime_amplitude = float(params["anime"][3])
            else:
                settings.anime_band_index = int(params["anime"][0])
                settings.anime_amplitude = float(params["anime"][1])
                settings.anime_division = int(params["anime"][2])
            if len(params["anime"]) == 6:
                settings.anime_shift = [fracval(x) for x in params["anime"][3:6]]

        # Modulation mode
        if "modulation" in params:
            settings.run_mode = "modulation"
            settings.modulation = params["modulation"]

        # Character table mode
        if "irreps_qpoint" in params:
            settings.run_mode = "irreps"
            settings.irreps_q_point = params["irreps_qpoint"][:3]
            if len(params["irreps_qpoint"]) == 4:
                settings.irreps_tolerance = float(params["irreps_qpoint"][3])
        if "show_irreps" in params:
            settings.show_irreps = params["show_irreps"]
        if "little_cogroup" in params:
            settings.is_little_cogroup = params["little_cogroup"]

        # DOS
        if "dos_range" in params:
            fmin = params["dos_range"][0]
            fmax = params["dos_range"][1]
            fpitch = params["dos_range"][2]
            settings.min_frequency = fmin
            settings.max_frequency = fmax
            settings.frequency_pitch = fpitch
        if "dos" in params:
            settings.is_dos_mode = params["dos"]

        if "fits_debye_model" in params:
            settings.fits_Debye_model = params["fits_debye_model"]

        if "fmax" in params:
            settings.max_frequency = params["fmax"]

        if "fmin" in params:
            settings.min_frequency = params["fmin"]

        # Project PDOS x, y, z directions in Cartesian coordinates
        if "xyz_projection" in params:
            settings.xyz_projection = params["xyz_projection"]
            if "pdos" not in params and settings.pdos_indices is None:
                self._set_parameter("pdos", [])

        if "pdos" in params:
            settings.pdos_indices = params["pdos"]
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False

        if "projection_direction" in params and not settings.xyz_projection:
            settings.projection_direction = params["projection_direction"]
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False

        # Thermal properties
        if "tprop" in params:
            settings.is_thermal_properties = params["tprop"]
            # Exclusive conditions
            settings.is_thermal_displacements = False
            settings.is_thermal_displacement_matrices = False
            settings.is_thermal_distances = False

        # Projected thermal properties
        if "ptprop" in params and params["ptprop"]:
            settings.is_thermal_properties = True
            settings.is_projected_thermal_properties = True
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False
            # Exclusive conditions
            settings.is_thermal_displacements = False
            settings.is_thermal_displacement_matrices = False
            settings.is_thermal_distances = False

        # Use imaginary frequency as real for thermal property calculation
        if "pretend_real" in params:
            settings.pretend_real = params["pretend_real"]

        # Thermal displacements
        if "tdisp" in params and params["tdisp"]:
            settings.is_thermal_displacements = True
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False
            # Exclusive conditions
            settings.is_thermal_properties = False
            settings.is_thermal_displacement_matrices = False
            settings.is_thermal_distances = True

        # Thermal displacement matrices
        if "tdispmat" in params and params["tdispmat"] or "tdispmat_cif" in params:
            settings.is_thermal_displacement_matrices = True
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False
            # Exclusive conditions
            settings.is_thermal_properties = False
            settings.is_thermal_displacements = False
            settings.is_thermal_distances = False

            # Temperature used to calculate thermal displacement matrix
            # to write aniso_U to cif
            if "tdispmat_cif" in params:
                settings.thermal_displacement_matrix_temperature = params[
                    "tdispmat_cif"
                ]

        # Thermal distances
        if "tdistance" in params:
            settings.is_thermal_distances = True
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False
            settings.thermal_atom_pairs = params["tdistance"]
            # Exclusive conditions
            settings.is_thermal_properties = False
            settings.is_thermal_displacements = False
            settings.is_thermal_displacement_matrices = False

        # Group velocity
        if "is_group_velocity" in params:
            settings.is_group_velocity = params["is_group_velocity"]

        # Moment mode
        if "moment" in params:
            settings.is_moment = params["moment"]
            settings.is_eigenvectors = True
            settings.is_mesh_symmetry = False
            if "moment_order" in params:
                settings.moment_order = params["moment_order"]

        if "random_displacement_temperature" in params:
            settings.random_displacement_temperature = params[
                "random_displacement_temperature"
            ]

        # Use Lapack solver via Lapacke
        if "lapack_solver" in params:
            settings.lapack_solver = params["lapack_solver"]

        if "include_fc" in params:
            settings.include_force_constants = params["include_fc"]

        if "include_fs" in params:
            settings.include_force_sets = params["include_fs"]

        if "include_nac_params" in params:
            settings.include_nac_params = params["include_nac_params"]

        if "include_disp" in params:
            settings.include_displacements = params["include_disp"]
        if settings.random_displacements is not None or settings.create_displacements:
            settings.include_displacements = True

        if "include_all" in params:
            settings.include_force_constants = True
            settings.include_force_sets = True
            settings.include_nac_params = True
            settings.include_displacements = True

        # Pair shortest vectors in supercell are stored in dense format.
        if "store_dense_svecs" in params:
            settings.store_dense_svecs = params["store_dense_svecs"]

        # ***********************************************************
        # This has to come last in this method to overwrite run_mode.
        # ***********************************************************
        if "pdos" in params and params["pdos"] == "auto":
            if "band_paths" in params:
                settings.run_mode = "band_mesh"
            else:
                settings.run_mode = "mesh"

        if "mesh_numbers" in params and "band_paths" in params:
            settings.run_mode = "band_mesh"

        # SSCHA
        if "sscha_iterations" in params:
            settings.sscha_iterations = params["sscha_iterations"]
