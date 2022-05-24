"""phonopy.yaml reader and writer."""
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

from typing import TYPE_CHECKING

import numpy as np
import yaml

from phonopy.structure.cells import Primitive

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

if TYPE_CHECKING:
    from phonopy import Phonopy

from phonopy.file_IO import get_io_module_to_decompress
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict


class PhonopyYaml:
    """PhonopyYaml is a container of phonopy setting.

    This contains the writer (__str__) and reader (read) of phonopy.yaml type
    file.

    Methods
    -------
    __str__
        Return string of phonopy.yaml.
    get_yaml_lines
        Return a list of string lines of phonopy.yaml.
    read
        Read specific properties written in phonopy.yaml.
    set_phonon_info
        Copy specific properties in Phonopy instance to self.

    Attributes
    ----------
    configuration : dict
        Phonopy setting tags or options (e.g., {"DIM": "2 2 2", ...})
    calculator : str
        Force calculator.
    physical_units : dict
        Set of physical units used in this phonon calculation.
    unitcell : PhonopyAtoms
        Unit cell.
    primitive : PhonopyAtoms
        Primitive cell. The instance of Primitive class is necessary has to
        be created from the instance of Supercell class with
        np.dot(np.linalg.inv(supercell_matrix), primitive_matrix).
    supercell : PhonopyAtoms
        Supercell. The instance of Supercell class is necessary has to be
        created from unitcell with supercel_matrix.
    dataset
    supercell_matrix
    primitive_matrix
    nac_params
    force_constants
    symmetry
    s2p_map
    u2p_map
    frequency_unit_conversion_factor
    version
    yaml_filename
    settings
    command_name
    default_filenames

    """

    command_name = "phonopy"
    default_filenames = ("phonopy_params.yaml", "phonopy_disp.yaml", "phonopy.yaml")
    default_settings = {
        "force_sets": True,
        "displacements": True,
        "force_constants": False,
        "born_effective_charge": True,
        "dielectric_constant": True,
    }

    def __init__(
        self, configuration=None, calculator=None, physical_units=None, settings=None
    ):
        """Init method.

        Parameters
        ----------
        configuration : dict
            Key-value pairs of phonopy calculation settings.
        calculator : str
            Force calculator name.
        physical_units : dict
            Physical units used for the calculation.
        settings : dict
            This controls amount of information in yaml output. See Phonopy.save().

        """
        self.configuration = configuration
        self.calculator = calculator
        self.physical_units = physical_units

        self.unitcell = None
        self.primitive = None
        self.supercell = None
        self.dataset = None
        self.supercell_matrix = None
        self.primitive_matrix = None
        self.nac_params = None
        self.force_constants = None

        self.symmetry = None  # symmetry of supercell
        self.s2p_map = None
        self.u2p_map = None
        self.frequency_unit_conversion_factor = None
        self.version = None
        self.yaml_filename = None

        self.settings = self.default_settings.copy()
        if type(settings) is dict:
            self.settings.update(settings)

        self._yaml = None

    def __str__(self):
        """Return string text of yaml output."""
        return "\n".join(self.get_yaml_lines())

    def read(self, filename):
        """Read phonopy.yaml like file."""
        self.yaml_filename = filename
        self._load(filename)

    @property
    def yaml_data(self):
        """Raw yaml data as dict."""
        return self._yaml

    @yaml_data.setter
    def yaml_data(self, yaml_data):
        self._yaml = yaml_data

    def parse(self):
        """Parse raw yaml data."""
        self._parse_command_header()
        self._parse_transformation_matrices()
        self._parse_all_cells()
        self._parse_force_constants()
        self._parse_dataset()
        self._parse_nac_params()
        self._parse_calculator()

    def set_phonon_info(self, phonopy: "Phonopy"):
        """Collect data from Phonopy instance."""
        self.unitcell = phonopy.unitcell
        self.primitive = phonopy.primitive
        self.supercell = phonopy.supercell
        self.version = phonopy.version
        self.supercell_matrix = phonopy.supercell_matrix
        self.symmetry = phonopy.symmetry
        self.primitive_matrix = phonopy.primitive_matrix
        s2p_map = self.primitive.s2p_map
        u2s_map = self.supercell.u2s_map
        u2u_map = self.supercell.u2u_map
        s2u_map = self.supercell.s2u_map
        self.u2p_map = [u2u_map[i] for i in (s2u_map[s2p_map])[u2s_map]]
        self.nac_params = phonopy.nac_params
        self.frequency_unit_conversion_factor = phonopy.unit_conversion_factor
        self.calculator = phonopy.calculator
        self.force_constants = phonopy.force_constants
        self.dataset = phonopy.dataset

    def get_yaml_lines(self):
        """Return yaml string lines as a list."""
        lines = self._header_yaml_lines()
        lines += self._physical_units_yaml_lines()
        lines += self._symmetry_yaml_lines()
        lines += self._cell_info_yaml_lines()
        lines += self._nac_yaml_lines()
        lines += self._dataset_yaml_lines()
        lines += self._force_constants_yaml_lines()
        return lines

    def _header_yaml_lines(self):
        lines = []
        lines.append("%s:" % self.command_name)
        if self.version is None:
            from phonopy.version import __version__

            lines.append("  version: %s" % __version__)
        else:
            lines.append("  version: %s" % self.version)
        if self.calculator:
            lines.append("  calculator: %s" % self.calculator)
        if self.frequency_unit_conversion_factor:
            lines.append(
                "  frequency_unit_conversion_factor: %f"
                % self.frequency_unit_conversion_factor
            )
        if self.symmetry:
            lines.append("  symmetry_tolerance: %.5e" % self.symmetry.tolerance)
        if self.nac_params:
            lines.append("  nac_unit_conversion_factor: %f" % self.nac_params["factor"])
        if self.configuration is not None:
            lines.append("  configuration:")
            for key in self.configuration:
                val = self.configuration[key]
                if type(val) is str:
                    val = val.replace("\\", "\\\\")
                lines.append('    %s: "%s"' % (key, val))
        lines.append("")
        return lines

    def _physical_units_yaml_lines(self):
        lines = []
        lines.append("physical_unit:")
        lines.append('  atomic_mass: "AMU"')
        units = self.physical_units
        if units is not None:
            if units["length_unit"] is not None:
                lines.append('  length: "%s"' % units["length_unit"])
            if (
                self.command_name == "phonopy"
                and units["force_constants_unit"] is not None
            ):
                lines.append('  force_constants: "%s"' % units["force_constants_unit"])
        lines.append("")
        return lines

    def _symmetry_yaml_lines(self):
        lines = []
        if self.symmetry is not None and self.symmetry.dataset is not None:
            lines.append("space_group:")
            lines.append('  type: "%s"' % self.symmetry.dataset["international"])
            lines.append("  number: %d" % self.symmetry.dataset["number"])
            hall_symbol = self.symmetry.dataset["hall"]
            if '"' in hall_symbol:
                hall_symbol = hall_symbol.replace('"', '\\"')
            lines.append('  Hall_symbol: "%s"' % hall_symbol)
            lines.append("")
        return lines

    def _cell_info_yaml_lines(self):
        lines = self._primitive_matrix_yaml_lines(
            self.primitive_matrix, "primitive_matrix"
        )
        lines += self._supercell_matrix_yaml_lines(
            self.supercell_matrix, "supercell_matrix"
        )
        lines += self._primitive_yaml_lines(self.primitive, "primitive_cell")
        lines += self._unitcell_yaml_lines()
        lines += self._supercell_yaml_lines()
        return lines

    def _primitive_matrix_yaml_lines(self, matrix, name):
        lines = []
        if matrix is not None:
            lines.append("%s:" % name)
            for v in matrix:
                lines.append("- [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            lines.append("")
        return lines

    def _supercell_matrix_yaml_lines(self, matrix, name):
        lines = []
        if matrix is not None:
            lines.append("%s:" % name)
            for v in matrix:
                lines.append("- [ %3d, %3d, %3d ]" % tuple(v))
            lines.append("")
        return lines

    def _primitive_yaml_lines(self, primitive: Primitive, name):
        lines = []
        if primitive is not None:
            lines += self._cell_yaml_lines(self.primitive, name, None)
            lines.append("  reciprocal_lattice: # without 2pi")
            rec_lat = np.linalg.inv(primitive.cell)
            for v, a in zip(rec_lat.T, ("a*", "b*", "c*")):
                lines.append(
                    "  - [ %21.15f, %21.15f, %21.15f ] # %s" % (v[0], v[1], v[2], a)
                )
            lines.append("")
        return lines

    def _unitcell_yaml_lines(self):
        lines = []
        if self.unitcell is not None:
            lines += self._cell_yaml_lines(self.unitcell, "unit_cell", self.u2p_map)
            lines.append("")
        return lines

    def _supercell_yaml_lines(self):
        lines = []
        if self.supercell is not None:
            s2p_map = getattr(self.primitive, "s2p_map", None)
            lines += self._cell_yaml_lines(self.supercell, "supercell", s2p_map)
            lines.append("")
        return lines

    def _cell_yaml_lines(self, cell: PhonopyAtoms, name, map_to_primitive):
        lines = []
        lines.append("%s:" % name)
        count = 0
        for line in cell.get_yaml_lines():
            lines.append("  " + line)
            if map_to_primitive is not None and "mass" in line:
                lines.append("    reduced_to: %d" % (map_to_primitive[count] + 1))
                count += 1
        return lines

    def _nac_yaml_lines(self):
        if self.primitive is None:
            return []
        else:
            return self._nac_yaml_lines_given_symbols(self.primitive.symbols)

    def _nac_yaml_lines_given_symbols(self, symbols):
        lines = []
        if self.nac_params is not None:
            if self.settings["born_effective_charge"]:
                lines.append("born_effective_charge:")
                for i, z in enumerate(self.nac_params["born"]):
                    text = "- # %d" % (i + 1)
                    if symbols:
                        text += " (%s)" % symbols[i]
                    lines.append(text)
                    for v in z:
                        lines.append("  - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
                lines.append("")

            if self.settings["dielectric_constant"]:
                lines.append("dielectric_constant:")
                for v in self.nac_params["dielectric"]:
                    lines.append("  - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
                lines.append("")
        return lines

    def _dataset_yaml_lines(self):
        lines = []
        if self.settings["force_sets"] or self.settings["displacements"]:
            disp_yaml_lines = self._displacements_yaml_lines(
                with_forces=self.settings["force_sets"]
            )
            lines += disp_yaml_lines
        return lines

    def _displacements_yaml_lines(self, with_forces=False):
        return self._displacements_yaml_lines_2types(
            self.dataset, with_forces=with_forces
        )

    def _displacements_yaml_lines_2types(
        self, dataset, with_forces=False, key="displacements"
    ):
        """Choose yaml writer depending on the dataset type.

        See type1 and type2 at Phonopy.dataset.

        """
        if dataset is not None:
            if "first_atoms" in dataset:
                return self._displacements_yaml_lines_type1(
                    dataset, with_forces=with_forces, key=key
                )
            elif "displacements" in dataset:
                return self._displacements_yaml_lines_type2(
                    dataset, with_forces=with_forces, key=key
                )
        return []

    def _displacements_yaml_lines_type1(
        self, dataset, with_forces=False, key="displacements"
    ):
        """Return type1 dataset in yaml.

        See data structure at Phonopy.dataset.

        """
        lines = [
            "%s:" % key,
        ]
        for d in dataset["first_atoms"]:
            lines.append("- atom: %4d" % (d["number"] + 1))
            lines.append("  displacement:")
            lines.append("    [ %20.16f,%20.16f,%20.16f ]" % tuple(d["displacement"]))
            if with_forces and "forces" in d:
                lines.append("  forces:")
                for f in d["forces"]:
                    lines.append("  - [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        lines.append("")
        return lines

    def _displacements_yaml_lines_type2(
        self, dataset, with_forces=False, key="displacements"
    ):
        """Return type2 dataset in yaml.

        See data structure at Phonopy.dataset.

        """
        if "random_seed" in dataset:
            lines = ["random_seed: %d" % dataset["random_seed"], "displacements:"]
        else:
            lines = [
                "%s:" % key,
            ]
        for i, dset in enumerate(dataset["displacements"]):
            lines.append("- # %4d" % (i + 1))
            for j, d in enumerate(dset):
                lines.append("  - displacement: # %d" % (j + 1))
                lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(d))
                if with_forces and "forces" in dataset:
                    f = dataset["forces"][i][j]
                    lines.append("    force:")
                    lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        lines.append("")
        return lines

    def _force_constants_yaml_lines(self):
        lines = []
        if self.settings["force_constants"] and self.force_constants is not None:
            shape = self.force_constants.shape[:2]
            lines = [
                "force_constants:",
            ]
            if shape[0] == shape[1]:
                lines.append('  format: "full"')
            else:
                lines.append('  format: "compact"')
            lines.append("  shape: [ %d, %d ]" % shape)
            lines.append("  elements:")
            for (i, j) in list(np.ndindex(shape)):
                lines.append("  - # (%d, %d)" % (i + 1, j + 1))
                for v in self.force_constants[i, j]:
                    lines.append("    - [ %21.15f, %21.15f, %21.15f ]" % tuple(v))
        return lines

    def _load(self, filename):
        self._yaml = load_yaml(filename)
        if type(self._yaml) is str:
            msg = "Could not open %s's yaml file." % self.command_name
            raise TypeError(msg)
        self.parse()

    def _parse_command_header(self):
        if self.command_name in self._yaml:
            header = self._yaml[self.command_name]
            self.version = header["version"]

    def _parse_transformation_matrices(self):
        if "supercell_matrix" in self._yaml:
            self.supercell_matrix = np.array(
                self._yaml["supercell_matrix"], dtype="intc", order="C"
            )
        if "primitive_matrix" in self._yaml:
            self.primitive_matrix = np.array(
                self._yaml["primitive_matrix"], dtype="double", order="C"
            )

    def _parse_all_cells(self):
        if "unit_cell" in self._yaml:
            self.unitcell = self._parse_cell(self._yaml["unit_cell"])
        if "primitive_cell" in self._yaml:
            self.primitive = self._parse_cell(self._yaml["primitive_cell"])
        if "supercell" in self._yaml:
            self.supercell = self._parse_cell(self._yaml["supercell"])
        if self.unitcell is None:
            if "lattice" in self._yaml and (
                "points" in self._yaml or "atoms" in self._yaml
            ):
                self.unitcell = parse_cell_dict(self._yaml)

    def _parse_cell(self, cell_dict):
        return parse_cell_dict(cell_dict)

    def _parse_force_constants(self):
        if "force_constants" in self._yaml:
            shape = tuple(self._yaml["force_constants"]["shape"]) + (3, 3)
            fc = np.reshape(self._yaml["force_constants"]["elements"], shape)
            self.force_constants = np.array(fc, dtype="double", order="C")

    def _parse_dataset(self):
        self.dataset = self._get_dataset(self.supercell)

    def _get_dataset(self, supercell, key="displacements"):
        dataset = None
        if key in self._yaml:
            if supercell is not None:
                natom = len(supercell)
            else:
                natom = None
            disp = self._yaml[key][0]
            if type(disp) is dict:  # type1
                dataset = self._parse_force_sets_type1(natom=natom, key=key)
            elif type(disp) is list:  # type2
                if "displacement" in disp[0]:
                    dataset = self._parse_force_sets_type2(key=key)
        return dataset

    def _parse_force_sets_type1(self, natom=None, key="displacements"):
        with_forces = False
        if "forces" in self._yaml[key][0]:
            with_forces = True
            dataset = {"natom": len(self._yaml[key][0]["forces"])}
        elif natom is not None:
            dataset = {"natom": natom}
        elif "natom" in self._yaml:
            dataset = {"natom": self._yaml["natom"]}
        else:
            raise RuntimeError("Number of atoms in supercell could not be found.")

        first_atoms = []
        for d in self._yaml[key]:
            data = {
                "number": d["atom"] - 1,
                "displacement": np.array(d["displacement"], dtype="double"),
            }
            if with_forces:
                data["forces"] = np.array(d["forces"], dtype="double", order="C")
            first_atoms.append(data)
        dataset["first_atoms"] = first_atoms

        return dataset

    def _parse_force_sets_type2(self, key="displacements"):
        nsets = len(self._yaml[key])
        natom = len(self._yaml[key][0])
        if "force" in self._yaml[key][0][0]:
            with_forces = True
            forces = np.zeros((nsets, natom, 3), dtype="double", order="C")
        else:
            with_forces = False
        displacements = np.zeros((nsets, natom, 3), dtype="double", order="C")
        for i, dfset in enumerate(self._yaml[key]):
            for j, df in enumerate(dfset):
                if with_forces:
                    forces[i, j] = df["force"]
                displacements[i, j] = df["displacement"]
        if with_forces:
            return {"forces": forces, "displacements": displacements}
        else:
            return {"displacements": displacements}

    def _parse_nac_params(self):
        nac_params = {}
        if "born_effective_charge" in self._yaml:
            nac_params["born"] = np.array(
                self._yaml["born_effective_charge"], dtype="double", order="C"
            )
        if "dielectric_constant" in self._yaml:
            nac_params["dielectric"] = np.array(
                self._yaml["dielectric_constant"], dtype="double", order="C"
            )
        if (
            self.command_name in self._yaml
            and "nac_unit_conversion_factor" in self._yaml[self.command_name]
        ):
            nac_params["factor"] = self._yaml[self.command_name][
                "nac_unit_conversion_factor"
            ]
        if "born" in nac_params and "dielectric" in nac_params:
            self.nac_params = nac_params

    def _parse_calculator(self):
        if (
            self.command_name in self._yaml
            and "calculator" in self._yaml[self.command_name]
        ):
            self.calculator = self._yaml[self.command_name]["calculator"]


def read_cell_yaml(filename, cell_type="unitcell"):
    """Read crystal structure from a phonopy.yaml or PhonopyAtoms.__str__ like file.

    phonopy.yaml like file can contain several different cells, e.g., unit cell,
    primitive cell, or supercell. In this case, the default preference order of
    the returned cell is unit cell > primitive cell > supercell. ``cell_type``
    is used to specify to choose one of them.

    When output of PhonopyAtoms.__str__ is given (like below), this file is
    parsed and its cell is returned.

    lattice:
    - [     0.000000000000000,     2.845150738087836,     2.845150738087836 ] # a
    - [     2.845150738087836,     0.000000000000000,     2.845150738087836 ] # b
    - [     2.845150738087836,     2.845150738087836,     0.000000000000000 ] # c
    points:
    - symbol: Na # 1
      coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
      mass: 22.989769
    - symbol: Cl # 2
      coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
      mass: 35.453000

    """
    ph_yaml = PhonopyYaml()
    ph_yaml.read(filename)
    if ph_yaml.unitcell and cell_type == "unitcell":
        return ph_yaml.unitcell
    elif ph_yaml.primitive and cell_type == "primitive":
        return ph_yaml.primitive
    elif ph_yaml.supercell and cell_type == "supercell":
        return ph_yaml.supercell
    else:
        return None


def load_yaml(filename):
    """Load yaml file.

    lzma and gzip comppressed files can be loaded.

    """
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename) as f:
        yaml_data = yaml.load(f, Loader=Loader)

    return yaml_data
