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

from __future__ import annotations

import dataclasses
import io
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import yaml

from phonopy.structure.cells import Primitive, Supercell
from phonopy.structure.symmetry import Symmetry

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

if TYPE_CHECKING:
    from phonopy import Phonopy

from phonopy.file_IO import get_io_module_to_decompress
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict
from phonopy.structure.dataset import forces_in_dataset


@dataclasses.dataclass
class PhonopyYamlData:
    """PhonopyYaml data structure."""

    configuration: Optional[dict] = None
    calculator: Optional[str] = None
    physical_units: Optional[dict] = None
    unitcell: Optional[PhonopyAtoms] = None
    primitive: Optional[Union[Primitive, PhonopyAtoms]] = None
    supercell: Optional[Union[Supercell, PhonopyAtoms]] = None
    dataset: Optional[dict] = None
    supercell_matrix: Optional[np.ndarray] = None
    primitive_matrix: Optional[np.ndarray] = None
    nac_params: Optional[dict] = None
    force_constants: Optional[np.ndarray] = None
    symmetry: Optional[Symmetry] = None  # symmetry of supercell
    frequency_unit_conversion_factor: Optional[float] = None
    version: Optional[str] = None
    command_name: str = "phonopy"


def phonopy_yaml_property_factory(name):
    """Property factor for PhonopyYaml class."""

    def getter(instance):
        return instance._data.__dict__[name]

    def setter(instance, value):
        instance._data.__dict__[name] = value

    return property(getter, setter)


class PhonopyYamlLoaderBase(ABC):
    """Base class of PhonopyYaml loader."""

    def __init__(
        self, yaml_data, configuration=None, calculator=None, physical_units=None
    ):
        """Init method.

        Parameters
        ----------
        yaml_data : dict
            Data in dict, which is normally obtained parsing ``phonopy.yaml``
            like file using pyyaml.
        configuration : dict, optional
            Phonopy calculation configuration. Default is None.
        calculator : str, optional
            Force calculator. Default is None.
        physical_units : dict, optional
            Physical units used for the phonopy calculation.

        """
        self._yaml = yaml_data
        self._data = PhonopyYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )

    @property
    def data(self):
        """Return PhonopyYamlData instance."""
        return self._data

    def parse(self):
        """Parse raw yaml data."""
        self._parse_command_header()
        self._parse_physical_units()
        self._parse_transformation_matrices()
        self._parse_all_cells()
        self._parse_force_constants()
        self._parse_dataset()
        self._parse_nac()
        self._parse_calculator()
        return self

    def _parse_command_header(self):
        if self._data.command_name in self._yaml:
            header = self._yaml[self._data.command_name]
            self._data.version = header["version"]

    def _parse_physical_units(self):
        if "physical_unit" in self._yaml:
            self._data.physical_units = {}
            for key, val in self._yaml["physical_unit"].items():
                if key == "atomic_mass":
                    continue
                if key in ["length", "force", "force_constants"]:
                    self._data.physical_units[key + "_unit"] = val
                else:
                    self._data.physical_units[key] = val

    def _parse_transformation_matrices(self):
        if "supercell_matrix" in self._yaml:
            self._data.supercell_matrix = np.array(
                self._yaml["supercell_matrix"], dtype="intc", order="C"
            )
        if "primitive_matrix" in self._yaml:
            self._data.primitive_matrix = np.array(
                self._yaml["primitive_matrix"], dtype="double", order="C"
            )

    def _parse_all_cells(self):
        if "unit_cell" in self._yaml:
            self._data.unitcell = self._parse_cell(self._yaml["unit_cell"])
        if "primitive_cell" in self._yaml:
            self._data.primitive = self._parse_cell(self._yaml["primitive_cell"])
        if "supercell" in self._yaml:
            self._data.supercell = self._parse_cell(self._yaml["supercell"])
        if self._data.unitcell is None:
            if "lattice" in self._yaml and (
                "points" in self._yaml or "atoms" in self._yaml
            ):
                self._data.unitcell = parse_cell_dict(self._yaml)

    def _parse_cell(self, cell_dict):
        return parse_cell_dict(cell_dict)

    def _parse_force_constants(self):
        if "force_constants" in self._yaml:
            shape = tuple(self._yaml["force_constants"]["shape"]) + (3, 3)
            fc = np.reshape(self._yaml["force_constants"]["elements"], shape)
            self._data.force_constants = np.array(fc, dtype="double", order="C")

    @abstractmethod
    def _parse_dataset(self):
        pass

    def _get_dataset(
        self, supercell: PhonopyAtoms, key_prefix: str = ""
    ) -> Optional[dict]:
        dataset = None
        if f"{key_prefix}displacements" in self._yaml:
            if supercell is not None:
                natom = len(supercell)
            else:
                natom = None

            disp = self._yaml[f"{key_prefix}displacements"][0]
            if isinstance(disp, dict):  # type1
                dataset = self._parse_force_sets_type1(
                    natom=natom, key_prefix=key_prefix
                )
            elif isinstance(disp, list):  # type2
                if "displacement" in disp[0]:
                    dataset = self._parse_force_sets_type2_v223(key_prefix=key_prefix)
        elif f"{key_prefix}dataset" in self._yaml:
            dataset = self._parse_force_sets_type2(key_prefix=key_prefix)
        return dataset

    def _parse_force_sets_type1(
        self, natom: Optional[int] = None, key_prefix: str = ""
    ) -> dict:
        key = f"{key_prefix}displacements"
        if "forces" in self._yaml[key][0]:
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
            if "forces" in d:
                data["forces"] = np.array(d["forces"], dtype="double", order="C")
            if "supercell_energy" in d:
                data["supercell_energy"] = d["supercell_energy"]
            first_atoms.append(data)
        dataset["first_atoms"] = first_atoms

        return dataset

    def _parse_force_sets_type2(self, key_prefix: str = "") -> dict:
        """Parse displacements, forces and energies in type2 format.

        This is the format >= v2.24.0 as follows:

        dataset:
          displacements:
            ...
          forces:
            ...
          supercell_energies:
            ...

        """
        key = f"{key_prefix}dataset"
        dataset = {}
        if "displacements" in self._yaml[key]:
            dataset["displacements"] = np.array(
                self._yaml[key]["displacements"], dtype="double", order="C"
            )
        if "forces" in self._yaml[key]:
            dataset["forces"] = np.array(
                self._yaml[key]["forces"], dtype="double", order="C"
            )
        if "supercell_energies" in self._yaml[key]:
            dataset["supercell_energies"] = np.array(
                self._yaml[key]["supercell_energies"], dtype="double"
            )
        return dataset

    def _parse_force_sets_type2_v223(self, key_prefix: str = "") -> dict:
        """Parse displacements, forces and energies in type2 legacy format.

        This is the format < v2.24 as follows:

        displacements:
        - # 1
          - displacement: # 1
              [  -0.0201336051051884,  0.0137526506253330,  0.0174786311319239 ]
              force:
              [   0.0551556600000000, -0.0257346500000000, -0.0282983400000000 ]
          ...
        ...
        supercell_energies:
        - -216.84472784 # 1
        ...

        """
        key = f"{key_prefix}displacements"
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
            dataset = {"forces": forces, "displacements": displacements}
        else:
            dataset = {"displacements": displacements}

        if "supercell_energies" in self._yaml:
            dataset["supercell_energies"] = np.array(
                self._yaml[f"{key_prefix}supercell_energies"], dtype="double"
            )

        return dataset

    def _parse_nac(self):
        """Parse NAC parameters.

        Older than v2.18, keys below "nac" are put on top level.

        nac:
          born_effective_charge:
            ...
          dielectric_constant:
            ...
          unit_conversion_factor:
            ...
          method:
            ...

        """
        nac_params = self._parse_nac_params(self._yaml)  # older than v2.18
        if "nac" in self._yaml:
            nac_params = self._parse_nac_params(self._yaml["nac"])
        if "born" in nac_params and "dielectric" in nac_params:
            self._data.nac_params = nac_params

    def _parse_nac_params(self, nac_yaml):
        nac_params = {}
        if "born_effective_charge" in nac_yaml:
            nac_params["born"] = np.array(
                nac_yaml["born_effective_charge"], dtype="double", order="C"
            )
        if "dielectric_constant" in nac_yaml:
            nac_params["dielectric"] = np.array(
                nac_yaml["dielectric_constant"], dtype="double", order="C"
            )
        if (  # older than v2.18
            self._data.command_name in nac_yaml
            and "nac_unit_conversion_factor" in nac_yaml[self._data.command_name]
        ):
            nac_params["factor"] = nac_yaml[self._data.command_name][
                "nac_unit_conversion_factor"
            ]
        if "unit_conversion_factor" in nac_yaml:
            nac_params["factor"] = nac_yaml["unit_conversion_factor"]
        if "nac" in self._yaml and "method" in nac_yaml:
            nac_params["method"] = nac_yaml["method"].lower()
        return nac_params

    def _parse_calculator(self):
        if (
            self._data.command_name in self._yaml
            and "calculator" in self._yaml[self._data.command_name]
        ):
            self._data.calculator = self._yaml[self._data.command_name]["calculator"]


class PhonopyYamlLoader(PhonopyYamlLoaderBase):
    """PhonopyYaml loader."""

    def _parse_dataset(self):
        self._data.dataset = self._get_dataset(self._data.supercell)


class PhonopyYamlDumperBase(ABC):
    """Base class of PhonopyYaml dumper."""

    _default_dumper_settings = {
        "force_sets": True,
        "displacements": True,
        "force_constants": False,
        "born_effective_charge": True,
        "dielectric_constant": True,
    }

    def __init__(self, data: PhonopyYamlData, dumper_settings=None):
        """Init method."""
        self._data = data
        self._init_dumper_settings(dumper_settings)

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
        lines.append("%s:" % self._data.command_name)
        if self._data.version is None:
            from phonopy.version import __version__

            version = __version__
        else:
            version = self._data.version
        lines.append(f'  version: "{version}"')
        if self._data.calculator:
            lines.append("  calculator: %s" % self._data.calculator)
        if self._data.frequency_unit_conversion_factor:
            lines.append(
                "  frequency_unit_conversion_factor: %f"
                % self._data.frequency_unit_conversion_factor
            )
        if self._data.symmetry:
            lines.append("  symmetry_tolerance: %.5e" % self._data.symmetry.tolerance)
        if self._data.configuration is not None:
            lines.append("  configuration:")
            for key in self._data.configuration:
                val = self._data.configuration[key]
                if isinstance(val, str):
                    val = val.replace("\\", "\\\\")
                lines.append('    %s: "%s"' % (key, val))
        lines.append("")
        return lines

    def _physical_units_yaml_lines(self):
        lines = []
        units = self._data.physical_units
        if units is not None:
            lines.append("physical_unit:")
            lines.append('  atomic_mass: "AMU"')
            length_unit = units.get("length_unit", None)
            if length_unit is not None:
                lines.append(f'  length: "{length_unit}"')
            force_unit = units.get("force_unit", None)
            if force_unit is not None and forces_in_dataset(self._data.dataset):
                lines.append(f'  force: "{force_unit}"')
            if self._data.command_name == "phonopy":
                fc_unit = units.get("force_constants_unit", None)
                if fc_unit is not None and self._data.force_constants is not None:
                    lines.append(f'  force_constants: "{fc_unit}"')
            lines.append("")
        return lines

    def _symmetry_yaml_lines(self):
        lines = []
        if self._data.symmetry is None:
            return lines

        dataset = self._data.symmetry.dataset
        if dataset is not None:
            try:
                uni_number = dataset.uni_number
                lines.append("magnetic_space_group:")
                lines.append(f"  uni_number: {uni_number}")
                lines.append(f"  msg_type: {dataset.msg_type}")
                lines.append("")
            except AttributeError:
                lines.append("space_group:")
                spg_type = dataset.international
                lines.append(f'  type: "{spg_type}"')
                lines.append(f"  number: {dataset.number}")
                hall_symbol = dataset.hall
                if '"' in hall_symbol:
                    hall_symbol = hall_symbol.replace('"', '\\"')
                lines.append(f'  Hall_symbol: "{hall_symbol}"')
                lines.append("")
        return lines

    def _cell_info_yaml_lines(self):
        lines = self._primitive_matrix_yaml_lines(
            self._data.primitive_matrix, "primitive_matrix"
        )
        lines += self._supercell_matrix_yaml_lines(
            self._data.supercell_matrix, "supercell_matrix"
        )
        lines += self._primitive_yaml_lines(self._data.primitive, "primitive_cell")
        lines += self._unitcell_yaml_lines()
        lines += self._supercell_yaml_lines()
        return lines

    def _primitive_matrix_yaml_lines(self, matrix, name):
        lines = []
        if matrix is not None:
            lines.append(f"{name}:")
            for v in matrix:
                lines.append("- [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            lines.append("")
        return lines

    def _supercell_matrix_yaml_lines(self, matrix, name):
        lines = []
        if matrix is not None:
            lines.append(f"{name}:")
            for v in matrix:
                lines.append("- [ %3d, %3d, %3d ]" % tuple(v))
            lines.append("")
        return lines

    def _primitive_yaml_lines(self, primitive: Primitive, name):
        lines = []
        if primitive is not None:
            lines += self._cell_yaml_lines(self._data.primitive, name, None)
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
        if self._data.unitcell is not None:
            if isinstance(self._data.primitive, Primitive) and isinstance(
                self._data.supercell, Supercell
            ):
                s2p_map = self._data.primitive.s2p_map
                u2s_map = self._data.supercell.u2s_map
                u2u_map = self._data.supercell.u2u_map
                s2u_map = self._data.supercell.s2u_map
                u2p_map = [u2u_map[i] for i in (s2u_map[s2p_map])[u2s_map]]
            else:
                u2p_map = None
            lines += self._cell_yaml_lines(self._data.unitcell, "unit_cell", u2p_map)
            lines.append("")
        return lines

    def _supercell_yaml_lines(self):
        lines = []
        if self._data.supercell is not None:
            s2p_map = getattr(self._data.primitive, "s2p_map", None)
            lines += self._cell_yaml_lines(self._data.supercell, "supercell", s2p_map)
            lines.append("")
        return lines

    def _cell_yaml_lines(
        self, cell: PhonopyAtoms, name, map_to_primitive: Optional[list] = None
    ):
        lines = []
        lines.append("%s:" % name)
        count = 0
        for line in cell.get_yaml_lines():
            lines.append("  " + line)
            if map_to_primitive is not None and "mass" in line:
                lines.append("    reduced_to: %d" % (map_to_primitive[count] + 1))
                count += 1
        return lines

    @abstractmethod
    def _nac_yaml_lines(self):
        pass

    def _nac_yaml_lines_given_symbols(self, symbols=None):
        lines = []
        if self._data.nac_params is not None:
            if self._dumper_settings["born_effective_charge"]:
                lines.append("  born_effective_charge:")
                for i, z in enumerate(self._data.nac_params["born"]):
                    text = "  - # %d" % (i + 1)
                    if symbols:
                        text += " (%s)" % symbols[i]
                    lines.append(text)
                    for v in z:
                        lines.append("    - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            if self._dumper_settings["dielectric_constant"]:
                lines.append("  dielectric_constant:")
                for v in self._data.nac_params["dielectric"]:
                    lines.append("    - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            if lines:
                if "method" in self._data.nac_params:
                    if self._data.nac_params["method"].lower() == "gonze":
                        lines.append('  method: "Gonze"')
                    if self._data.nac_params["method"].lower() == "wang":
                        lines.append('  method: "Wang"')
                if "factor" in self._data.nac_params:
                    factor = self._data.nac_params["factor"]
                    # if factor is not None:
                    lines.append("  unit_conversion_factor: %f" % factor)
                lines.insert(0, "nac:")
                lines.append("")
        return lines

    def _dataset_yaml_lines(self):
        lines = []
        if (
            self._dumper_settings["force_sets"]
            or self._dumper_settings["displacements"]
        ):
            disp_yaml_lines = self._displacements_yaml_lines(
                with_forces=self._dumper_settings["force_sets"]
            )
            lines += disp_yaml_lines
        return lines

    @abstractmethod
    def _displacements_yaml_lines(self, with_forces=False) -> list:
        pass

    def _displacements_yaml_lines_2types(
        self,
        dataset: dict,
        with_forces: bool = False,
        key_prefix: str = "",
        v223_mode: bool = False,
    ) -> list:
        """Choose yaml writer depending on the dataset type.

        See type1 and type2 at Phonopy.dataset.

        Parameters
        ----------
        v223_mode : bool
            When True, old dataset yaml format is generated. Default is False.

        """
        if dataset is not None:
            if "first_atoms" in dataset:
                return self._displacements_yaml_lines_type1(
                    dataset, with_forces=with_forces, key_prefix=key_prefix
                )
            elif "displacements" in dataset:
                if v223_mode:
                    return self._displacements_yaml_lines_type2_v223(
                        dataset, with_forces=with_forces, key_prefix=key_prefix
                    )
                else:
                    return self._displacements_yaml_lines_type2(
                        dataset, with_forces=with_forces, key_prefix=key_prefix
                    )
        return []

    @abstractmethod
    def _displacements_yaml_lines_type1(
        self, dataset: dict, with_forces: bool = False, key_prefix: str = ""
    ) -> list:
        pass

    def _displacements_yaml_lines_type2(
        self, dataset: dict, with_forces: bool = False, key_prefix: str = ""
    ) -> list:
        """Return type2 dataset in yaml.

        This is the format >= v2.24.

        See data structure at Phonopy.dataset.

        """
        lines = [f"{key_prefix}dataset:"]
        if "random_seed" in dataset:
            lines.append("  random_seed: {:d}".format(dataset["random_seed"]))
        for key in ("displacements", "forces"):
            if key not in dataset:
                continue
            if key == "forces" and not with_forces:
                continue
            lines.append(f"  {key}:")
            for i, dset in enumerate(dataset[key]):
                lines.append(f"  - # {i + 1}")
                for _, d in enumerate(dset):
                    lines.append("    - [ %21.16f, %21.16f, %21.16f ]" % tuple(d))

        if "supercell_energies" in dataset:
            lines.append("  supercell_energies:")
            for i, energy in enumerate(dataset["supercell_energies"]):
                lines.append(f"  - {energy:.16f} # {i + 1}")
            lines.append("")

        return lines

    def _displacements_yaml_lines_type2_v223(
        self, dataset: dict, with_forces: bool = False, key_prefix: str = ""
    ) -> list:
        """Return type2 dataset in old stype yaml.

        This is the format < v2.24.

        See data structure at Phonopy.dataset.

        """
        if "random_seed" in dataset:
            lines = [
                "random_seed: %d" % dataset["random_seed"],
                f"{key_prefix}displacements:",
            ]
        else:
            lines = [
                f"{key_prefix}displacements:",
            ]
        for i, dset in enumerate(dataset["displacements"]):
            lines.append(f"- # {i + 1}")
            for j, d in enumerate(dset):
                lines.append(f"  - displacement: # {j + 1}")
                lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(d))
                if with_forces and "forces" in dataset:
                    f = dataset["forces"][i][j]
                    lines.append("    force:")
                    lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        lines.append("")

        if "supercell_energies" in dataset:
            lines.append(f"{key_prefix}supercell_energies:")
            for i, energy in enumerate(dataset["supercell_energies"]):
                lines.append(f"- {energy:.8f} # {i + 1}")
            lines.append("")

        return lines

    def _force_constants_yaml_lines(self):
        lines = []
        if (
            self._dumper_settings["force_constants"]
            and self._data.force_constants is not None
        ):
            shape = self._data.force_constants.shape[:2]
            lines = [
                "force_constants:",
            ]
            if shape[0] == shape[1]:
                lines.append('  format: "full"')
            else:
                lines.append('  format: "compact"')
            lines.append("  shape: [ %d, %d ]" % shape)
            lines.append("  elements:")
            for i, j in list(np.ndindex(shape)):
                lines.append("  - # (%d, %d)" % (i + 1, j + 1))
                for v in self._data.force_constants[i, j]:
                    lines.append("    - [ %21.15f, %21.15f, %21.15f ]" % tuple(v))
        return lines

    def _init_dumper_settings(self, dumper_settings):
        self._dumper_settings = self._default_dumper_settings.copy()
        if isinstance(dumper_settings, dict):
            self._dumper_settings.update(dumper_settings)


class PhonopyYamlDumper(PhonopyYamlDumperBase):
    """Base class of PhonopyYaml dumper."""

    def _displacements_yaml_lines(self, with_forces=False) -> list:
        return self._displacements_yaml_lines_2types(
            self._data.dataset, with_forces=with_forces
        )

    def _nac_yaml_lines(self):
        if self._data.primitive is None:
            symbols = []
        else:
            symbols = self._data.primitive.symbols
        return self._nac_yaml_lines_given_symbols(symbols)

    def _displacements_yaml_lines_type1(
        self, dataset: dict, with_forces: bool = False, key_prefix: str = ""
    ) -> list:
        """Return type1 dataset in yaml.

        See data structure at Phonopy.dataset.

        """
        return _displacements_yaml_lines_type1(
            dataset, with_forces=with_forces, key_prefix=key_prefix
        )


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
    primitive : Primitive
        Primitive cell. The instance of Primitive class is necessary has to
        be created from the instance of Supercell class with
        np.dot(np.linalg.inv(supercell_matrix), primitive_matrix).
    supercell : Supercell
        Supercell. The instance of Supercell class is necessary has to be
        created from unitcell with supercel_matrix.
    dataset
    supercell_matrix
    primitive_matrix
    nac_params
    force_constants
    symmetry
    frequency_unit_conversion_factor
    version
    settings
    command_name
    default_filenames

    """

    default_filenames = ("phonopy_disp.yaml", "phonopy.yaml")
    command_name = "phonopy"

    configuration = phonopy_yaml_property_factory("configuration")
    calculator = phonopy_yaml_property_factory("calculator")
    physical_units = phonopy_yaml_property_factory("physical_units")
    unitcell = phonopy_yaml_property_factory("unitcell")
    primitive = phonopy_yaml_property_factory("primitive")
    supercell = phonopy_yaml_property_factory("supercell")
    dataset = phonopy_yaml_property_factory("dataset")
    supercell_matrix = phonopy_yaml_property_factory("supercell_matrix")
    primitive_matrix = phonopy_yaml_property_factory("primitive_matrix")
    nac_params = phonopy_yaml_property_factory("nac_params")
    force_constants = phonopy_yaml_property_factory("force_constants")
    symmetry = phonopy_yaml_property_factory("symmetry")
    frequency_unit_conversion_factor = phonopy_yaml_property_factory(
        "frequency_unit_conversion_factor"
    )
    version = phonopy_yaml_property_factory("version")

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
        self._data = PhonopyYamlData(
            configuration=configuration,
            calculator=calculator,
            physical_units=physical_units,
        )
        self._dumper_settings = settings

    def __str__(self):
        """Return string text of yaml output."""
        phyml_dumper = PhonopyYamlDumper(
            self._data, dumper_settings=self._dumper_settings
        )
        return "\n".join(phyml_dumper.get_yaml_lines())

    def read(self, filename: Union[str, bytes, os.PathLike, io.IOBase]):
        """Read PhonopyYaml file."""
        self._data = read_phonopy_yaml(
            filename,
            configuration=self._data.configuration,
            calculator=self._data.calculator,
            physical_units=self._data.physical_units,
        )
        return self

    def set_phonon_info(self, phonopy: "Phonopy") -> PhonopyYaml:
        """Collect data from Phonopy instance."""
        self._data.unitcell = phonopy.unitcell
        self._data.primitive = phonopy.primitive
        self._data.supercell = phonopy.supercell
        self._data.version = phonopy.version
        self._data.supercell_matrix = phonopy.supercell_matrix
        self._data.symmetry = phonopy.symmetry
        self._data.primitive_matrix = phonopy.primitive_matrix
        self._data.nac_params = phonopy.nac_params
        self._data.frequency_unit_conversion_factor = phonopy.unit_conversion_factor
        self._data.calculator = phonopy.calculator
        self._data.force_constants = phonopy.force_constants
        self._data.dataset = phonopy.dataset
        return self


def read_phonopy_yaml(
    filename: Union[str, bytes, os.PathLike, io.IOBase],
    configuration=None,
    calculator=None,
    physical_units=None,
) -> PhonopyYamlData:
    """Read phonopy.yaml like file."""
    yaml_data = load_yaml(filename)
    if isinstance(yaml_data, str):
        if isinstance(filename, io.IOBase):
            msg = "Could not load stream properly."
        else:
            msg = f'Could not load "{filename}" properly.'
        raise TypeError(msg)

    return load_phonopy_yaml(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )


def load_phonopy_yaml(
    yaml_data, configuration=None, calculator=None, physical_units=None
) -> PhonopyYamlData:
    """Return PhonopyYamlData instance loading yaml data.

    Parameters
    ----------
    yaml_data : dict

    """
    phyml_loader = PhonopyYamlLoader(
        yaml_data,
        configuration=configuration,
        calculator=calculator,
        physical_units=physical_units,
    )
    phyml_loader.parse()
    return phyml_loader.data


def read_cell_yaml(
    filename: Union[str, bytes, os.PathLike, io.IOBase], cell_type: str = "unitcell"
) -> PhonopyAtoms:
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

    Parameters
    ----------
    filename : str, bytes, os.PathLike, io.IOBase
        File name or file stream.
    cell_type : str
        "unitcell", "primitive", or "supercell". Default is "unitcell".

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
        if isinstance(filename, io.IOBase):
            raise RuntimeError("Crystal structure data was not found in stream.")
        else:
            raise RuntimeError(f'Crystal structure data was not found in "{filename}".')


def load_yaml(fp: Union[str, bytes, os.PathLike, io.IOBase]):
    """Load yaml file.

    Parameters
    ----------
    fp : str, bytes, os.PathLike or io.IOBase
        Filename, file path, or file stream.

    lzma and gzip comppressed non-stream files can be loaded.

    """
    if isinstance(fp, io.IOBase):
        yaml_data = yaml.load(fp, Loader=Loader)
    else:
        myio = get_io_module_to_decompress(fp)
        with myio.open(fp) as f:
            yaml_data = yaml.load(f, Loader=Loader)

    return yaml_data


def _displacements_yaml_lines_type1(
    dataset: dict, with_forces: bool = False, key_prefix: str = ""
) -> list:
    """Return type1 dataset in yaml.

    See data structure at Phonopy.dataset.

    """
    lines = [
        f"{key_prefix}displacements:",
    ]
    for d in dataset["first_atoms"]:
        lines.append("- atom: %4d" % (d["number"] + 1))
        lines.append("  displacement:")
        lines.append("    [ %20.16f,%20.16f,%20.16f ]" % tuple(d["displacement"]))
        if with_forces and "forces" in d:
            lines.append("  forces:")
            for f in d["forces"]:
                lines.append("  - [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        if "supercell_energy" in d:
            lines.append(
                "  supercell_energy: {energy:.8f}".format(energy=d["supercell_energy"])
            )
    lines.append("")
    return lines
