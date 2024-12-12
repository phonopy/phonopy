"""Phonopy class."""

# Copyright (C) 2015 Atsushi Togo
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

import copy
import lzma
import os
import sys
import textwrap
import warnings
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np

from phonopy.exception import ForcesetsNotFoundError
from phonopy.harmonic.displacement import (
    directions_to_displacement_dataset,
    get_least_displacements,
    get_random_displacements_dataset,
)
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    DynamicalMatrixGL,
    get_dynamical_matrix,
)
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.harmonic.force_constants import (
    cutoff_force_constants,
    set_tensor_symmetry_PJ,
    show_drift_force_constants,
    symmetrize_compact_force_constants,
    symmetrize_force_constants,
)
from phonopy.interface.calculator import get_default_physical_units
from phonopy.interface.fc_calculator import get_fc2
from phonopy.interface.mlp import PhonopyMLP
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.pypolymlp import PypolymlpParams
from phonopy.phonon.animation import write_animation
from phonopy.phonon.band_structure import BandStructure, get_band_qpoints_by_seekpath
from phonopy.phonon.dos import ProjectedDos, TotalDos, get_dos_frequency_range
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.irreps import IrReps
from phonopy.phonon.mesh import IterMesh, Mesh
from phonopy.phonon.modulation import Modulation
from phonopy.phonon.moment import PhononMoment
from phonopy.phonon.qpoints import QpointsPhonon
from phonopy.phonon.random_displacements import RandomDisplacements
from phonopy.phonon.thermal_displacement import (
    ThermalDisplacementMatrices,
    ThermalDisplacements,
)
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.spectrum.dynamic_structure_factor import DynamicStructureFactor
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    Primitive,
    Supercell,
    get_primitive,
    get_primitive_matrix,
    get_supercell,
    guess_primitive_matrix,
    isclose,
    shape_supercell_matrix,
)
from phonopy.structure.dataset import forces_in_dataset
from phonopy.structure.grid_points import length2mesh
from phonopy.structure.symmetry import Symmetry, symmetrize_borns_and_epsilon
from phonopy.units import VaspToTHz
from phonopy.version import __version__


class Phonopy:
    """Phonopy main API given as a class.

    Attributes
    ----------
    version : str
    unitcell : PhonopyAtoms
    primitive : Primitive
    supercell : Supercell
    symmetry : Symmetry
        Symmetry of supercell.
    primitive_symmetry : Symmetry
        Symmetry of primitive cell.
    supercell_matrix : ndarray
        shape=(3, 3), dtype='intc', order='C'.
    primitive_matrix : ndarray
        shape=(3, 3), dtype='double', order='C'.
    unit_conversion_factor : float
        Phonon frequency unit conversion factor.
    calculator : str
    dataset : dict
    displacements : ndarray or list of list (getter) and array-like (setter).
    forces : ndarray (getter) or array_like (setter).
    force_constants : ndarray (getter) and array_like (setter).
    nac_params : dict (Deprecated)
    supercells_with_displacements : list of PhonopyAtoms.
    dynamical_matrix : DynamicalMatrix

    qpoints : QpointsPhonon
    band_structure : BandStructure
    mesh : Mesh or IterMesh
    thermal_properties : ThermalProperties
    thermal_displacements : ThermalDisplacements
    thermal_displacement_matrix : ThermalDisplacementMatrices
    random_displacements : RandomDisplacements
    dynamic_structure_factor : DynamicStructureFactor.
    irreps : IrReps
    moment : PhononMoment
    total_dos : TotalDos

    """

    def __init__(
        self,
        unitcell,
        supercell_matrix: Optional[Union[Sequence, np.ndarray]] = None,
        primitive_matrix: Optional[Union[str, Sequence, np.ndarray]] = None,
        nac_params: Optional[dict] = None,
        factor: float = VaspToTHz,
        frequency_scale_factor: Optional[float] = None,
        dynamical_matrix_decimals: Optional[int] = None,
        force_constants_decimals: Optional[int] = None,
        group_velocity_delta_q: Optional[float] = None,
        symprec: float = 1e-5,
        is_symmetry: bool = True,
        store_dense_svecs: bool = True,
        use_SNF_supercell: bool = False,
        calculator: Optional[str] = None,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        unitcell : PhonopyAtoms
            Input unit cell.
        supercell_matrix : array_like, optional
            Transformation matrix to supercell cell from unit cell. shape=(3,
            3), dtype=int.
        primitive_matrix : str or array_like, optional
            Transformation matrix to primitive cell from unit cell. shape=(3,
            3), dtype=float.
        nac_params : None
            Deprecated.
        factor : float, optional
            Phonon frequency unit conversion factor.
        group_velocity_delta_q : float, optional
            Delta-q distance to calculate group velocity.
        symprec : float, optional
            Symmetry search precision. Default is 1e-5.
        is_symmetry : bool, optional
            Whether to search symmetry of supercell. Default is True.
        use_SNF_supercell : bool, optional
            Supercell is built by SNF algorithm when True. Default is False. SNF
            algorithm is faster than the original one, but the order of atoms in
            the supercell can be different from the original one. So the
            backward compatibility to the old data (e.g., force constants) may
            not be preserved.
        calculator : str, optional
            Calculator name such as 'vasp', 'qe', etc. Default is None.
        log_level : int, optional
            Log level. Default is 0.
        store_dense_svecs : bool, optional
            Deprected. Dataset of shortest vectors between atoms in primitive
            cell and supercell is stored in the dense format when this is True.
            Default is True. In phonopy v3 or later version, False will not be
            supported.
        frequency_scale_factor : None
            Deprecated.
        dynamical_matrix_decimals : None
            Deprecated.
        force_constants_decimals : None
            Deprecated.

        """
        if not store_dense_svecs:
            warnings.warn(
                (
                    "store_dense_svecs=False is deprecated and will not be supported "
                    "in Phonopy v3 and later versions."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        if int(self.version.split(".")[0]) > 2:
            self._store_dense_svecs = True
        else:
            self._store_dense_svecs = store_dense_svecs

        if nac_params is not None:
            warnings.warn(
                (
                    "Phonopy class instanciation with nac_params is deprecated. "
                    "Use Phonopy.nac_params attribute instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        self._nac_params = nac_params

        if frequency_scale_factor is not None:
            warnings.warn(
                (
                    "Phonopy class instanciation with frequency_scale_factor is "
                    "deprecated."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        self._frequency_scale_factor = frequency_scale_factor

        if dynamical_matrix_decimals is not None:
            warnings.warn(
                (
                    "Phonopy class instanciation with dynamical_matrix_decimals is "
                    "deprecated."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        self._dynamical_matrix_decimals = dynamical_matrix_decimals

        if force_constants_decimals is not None:
            warnings.warn(
                (
                    "Phonopy class instanciation with force_constants_decimals is "
                    "deprecated."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        self._force_constants_decimals = force_constants_decimals

        self._symprec = symprec
        self._factor = factor
        self._is_symmetry = is_symmetry
        self._calculator = calculator

        self._use_SNF_supercell = use_SNF_supercell
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = unitcell.copy()
        self._supercell_matrix = self._shape_supercell_matrix(supercell_matrix)
        if isinstance(primitive_matrix, str):
            self._primitive_matrix = self._set_primitive_matrix(primitive_matrix)
        elif primitive_matrix is not None:
            self._primitive_matrix = np.array(
                primitive_matrix, dtype="double", order="c"
            )
        else:
            self._primitive_matrix = None
        self._supercell = None
        self._primitive = None
        self._build_supercell()
        self._build_primitive_cell()

        # Set supercell and primitive symmetry
        self._symmetry: Optional[Symmetry] = None
        self._primitive_symmetry: Optional[Symmetry] = None
        self._search_symmetry()
        self._search_primitive_symmetry()

        # displacements
        self._dataset = None
        self._supercells_with_displacements = None

        # set_force_constants or set_forces
        self._force_constants = None

        # set_dynamical_matrix
        self._dynamical_matrix = None

        # MLP
        self._mlp = None
        self._mlp_dataset = None

        # set_band_structure
        self._band_structure = None

        # set_mesh
        self._mesh = None

        # set_tetrahedron_method
        self._tetrahedron_method = None

        # set_thermal_properties
        self._thermal_properties = None

        # set_thermal_displacements
        self._thermal_displacements = None

        # set_thermal_displacement_matrices
        self._thermal_displacement_matrices = None

        # set_dynamic_structure_factor
        self._dynamic_structure_factor = None

        # set_partial_DOS
        self._pdos = None

        # set_total_DOS
        self._total_dos = None

        # set_modulation
        self._modulation = None

        # set_character_table
        self._irreps = None

        # set_group_velocity
        self._group_velocity = None
        self._gv_delta_q = group_velocity_delta_q

    @property
    def version(self) -> str:
        """Return phonopy release version number.

        str
            Phonopy release version number

        """
        return __version__

    def get_version(self):
        """Return phonopy release version number."""
        warnings.warn(
            "Phonopy.get_version() is deprecated." "Use Phonopy.version attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.version

    @property
    def primitive(self) -> Primitive:
        """Return primitive cell.

        Primitive
            Primitive cell.

        """
        return self._primitive

    def get_primitive(self):
        """Return primitive cell."""
        warnings.warn(
            "Phonopy.get_primitive() is deprecated." "Use Phonopy.primitive attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.primitive

    @property
    def unitcell(self) -> PhonopyAtoms:
        """Return input unit cell.

        PhonopyAtoms
            Input unit cell.

        """
        return self._unitcell

    def get_unitcell(self):
        """Return input unit cell."""
        warnings.warn(
            "Phonopy.get_unitcell() is deprecated." "Use Phonopy.unitcell attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.unitcell

    def set_unitcell(self, unitcell):
        """Set input unit cell."""
        warnings.warn(
            "Phonopy.set_unitcell() is deprecated."
            "No way to set unit cell will be provided.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._unitcell = unitcell
        self._build_supercell()
        self._build_primitive_cell()
        self._search_symmetry()
        self._search_primitive_symmetry()
        self._dataset = None

    @property
    def supercell(self) -> Supercell:
        """Return supercell.

        Supercell
            Supercell.

        """
        return self._supercell

    def get_supercell(self):
        """Return supercell."""
        warnings.warn(
            "Phonopy.get_supercell() is deprecated." "Use Phonopy.supercell attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.supercell

    @property
    def symmetry(self) -> Symmetry:
        """Return symmetry of supercell.

        Symmetry
            Symmetry of supercell.

        """
        return self._symmetry

    def get_symmetry(self):
        """Return symmetry of supercell."""
        warnings.warn(
            "Phonopy.get_symmetry() is deprecated." "Use Phonopy.symmetry attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.symmetry

    @property
    def primitive_symmetry(self) -> Symmetry:
        """Return symmetry of primitive cell.

        Symmetry
            Symmetry of primitive cell.

        """
        return self._primitive_symmetry

    def get_primitive_symmetry(self):
        """Return symmetry of primitive cell."""
        warnings.warn(
            "Phonopy.get_primitive_symmetry() is deprecated."
            "Use Phonopy.primitive_symmetry attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.primitive_symmetry

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return transformation matrix to supercell cell from unit cell.

        ndarray
            Supercell matrix with respect to unit cell.
            shape=(3, 3), dtype='intc', order='C'.

        """
        return self._supercell_matrix

    def get_supercell_matrix(self):
        """Return transformation matrix to supercell cell from unit cell."""
        warnings.warn(
            "Phonopy.get_supercell_matrix() is deprecated."
            "Use Phonopy.supercell_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.supercell_matrix

    @property
    def primitive_matrix(self) -> np.ndarray:
        """Return transformation matrix to primitive cell from unit cell.

        ndarray
            Primitive matrix with respect to unit cell.
            shape=(3, 3), dtype='double', order='C'.

        """
        return self._primitive_matrix

    def get_primitive_matrix(self):
        """Return transformation matrix to primitive cell from unit cell."""
        warnings.warn(
            "Phonopy.get_primitive_matrix() is deprecated."
            "Use Phonopy.primitive_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.primitive_matrix

    @property
    def unit_conversion_factor(self) -> float:
        """Return phonon frequency unit conversion factor.

        float
            Phonon frequency unit conversion factor. This factor
            converts sqrt(<force>/<distance>/<AMU>)/2pi/1e12 to the
            other favorite phonon frequency unit. Normally this factor
            is recommended to be that converts to THz (ordinary
            frequency) to calculate a variety of phonon properties
            that assumes that input phonon frequencies have THz unit.

        """
        return self._factor

    def get_unit_conversion_factor(self):
        """Return phonon frequency unit conversion factor."""
        warnings.warn(
            "Phonopy.get_unit_conversion_factor() is deprecated."
            "Use Phonopy.unit_conversion_factor attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.unit_conversion_factor

    @property
    def calculator(self) -> str:
        """Return calculator name.

        str
            Calculator name such as 'vasp', 'qe', etc.

        """
        return self._calculator

    @property
    def dataset(self) -> dict:
        """Return displacement-force dataset.

        Dataset containing information of displacements in supercells.
        This optionally contains energies and forces of respective supercells.

        dataset : dict
            The format can be either one of two types

            Type 1. One atomic displacement in each supercell:
                {'natom': number of atoms in supercell,
                 'first_atoms': [
                   {'number': atom index of displaced atom,
                    'displacement': displacement in Cartesian coordinates,
                    'forces': forces on atoms in supercell,
                    'supercell_energy': energy of supercell},
                   {...}, ...]}
            Elements of the list accessed by 'first_atoms' corresponds to each
            displaced supercell. Each displaced supercell contains only one
            displacement. dict['first_atoms']['forces'] gives atomic forces in
            each displaced supercell.

            Type 2. All atomic displacements in each supercell:
                {'displacements': ndarray, dtype='double', order='C',
                                  shape=(supercells, natom, 3)
                 'forces': ndarray, dtype='double', order='C',
                                  shape=(supercells, natom, 3),
                 'supercell_energies': ndarray, dtype='double'}

            To set in type 2, displacements and forces can be given by numpy
            array with different shape but that can be reshaped to
            (supercells, natom, 3).

        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if dataset is None:
            self._dataset = None
        elif "first_atoms" in dataset:
            self._dataset = copy.deepcopy(dataset)
        elif "displacements" in dataset:
            self._dataset = {}
            self.displacements = dataset["displacements"]
            if "forces" in dataset:
                self.forces = dataset["forces"]
            if "supercell_energies" in dataset:
                self.supercell_energies = dataset["supercell_energies"]
        else:
            raise RuntimeError("Data format of dataset is wrong.")

        self._supercells_with_displacements = None

    @property
    def mlp_dataset(self) -> Optional[dict]:
        """Return displacement-force dataset.

        The supercell matrix is equal to that of usual displacement-force
        dataset. Only type 2 format is supported. "displacements",
        "forces", and "supercell_energies" should be contained.

        """
        return self._mlp_dataset

    @mlp_dataset.setter
    def mlp_dataset(self, mlp_dataset: dict):
        if isinstance(mlp_dataset, dict):
            if "displacements" not in mlp_dataset:
                raise RuntimeError("Displacements have to be given.")
            if "forces" not in mlp_dataset:
                raise RuntimeError("Forces have to be given.")
            if "supercell_energy" in mlp_dataset:
                raise RuntimeError("Supercell energies have to be given.")
            if len(mlp_dataset["displacements"]) != len(mlp_dataset["forces"]):
                raise RuntimeError("Length of displacements and forces are different.")
            if len(mlp_dataset["displacements"]) != len(
                mlp_dataset["supercell_energies"]
            ):
                raise RuntimeError(
                    "Length of displacements and supercell_energies are different."
                )
            self._mlp_dataset = mlp_dataset
        elif mlp_dataset is None:
            self._mlp_dataset = None
        else:
            raise TypeError("mlp_dataset has to be a dictionary or None.")

    @property
    def mlp(self) -> Optional[PhonopyMLP]:
        """Setter and getter of PhonopyMLP dataclass."""
        return self._mlp

    @mlp.setter
    def mlp(self, mlp) -> Optional[PhonopyMLP]:
        self._mlp = mlp

    @property
    def displacement_dataset(self):
        """Return dataset to store displacements and forces."""
        warnings.warn(
            "Phonopy.displacement_dataset attribute is deprecated."
            "Use Phonopy.dataset attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dataset

    def get_displacement_dataset(self):
        """Return dataset to store displacements and forces."""
        warnings.warn(
            "Phonopy.get_displacement_dataset() is deprecated."
            "Use Phonopy.dataset attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dataset

    def set_displacement_dataset(self, displacement_dataset):
        """Set displacements."""
        warnings.warn(
            "Phonopy.set_displacement_dataset() is deprecated."
            "Use Phonopy.dataset attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.dataset = displacement_dataset

    @property
    def displacements(self) -> Union[np.ndarray, list]:
        """Getter and setter of displacements in supercells.

        There are two types of displacement dataset. See the docstring
        of dataset about types 1 and 2 for the displacement dataset formats.
        Displacements set returned depends on either type-1 or type-2 as
        follows:

        Type-1, List of list
            The internal list has 4 elements such as [32, 0.01, 0.0, 0.0]].
            The first element is the supercell atom index starting with 0.
            The remaining three elements give the displacement in Cartesian
            coordinates.
        Type-2, array_like
            Displacements of all atoms of all supercells in Cartesian
            coordinates.
            shape=(supercells, natom, 3)
            dtype='double'


        For setter, only type-2 dataset format is allowed.

        displacements : array_like
            Atomic displacements of all atoms of all supercells.
            Only all displacements in each supercell case (type-2) is
            supported.
            shape=(supercells, natom, 3), dtype='double', order='C'

        """
        disps = []
        if "first_atoms" in self._dataset:
            for disp in self._dataset["first_atoms"]:
                x = disp["displacement"]
                disps.append([disp["number"], x[0], x[1], x[2]])
        elif "displacements" in self._dataset:
            disps = self._dataset["displacements"]

        return disps

    @displacements.setter
    def displacements(self, displacements):
        disp = np.array(displacements, dtype="double", order="C")
        if disp.ndim != 3 or disp.shape[1:] != (len(self._supercell), 3):
            raise RuntimeError("Array shape of displacements is incorrect.")
        if self._dataset is None:
            self._dataset = {}
        if "first_atoms" in self._dataset:
            raise RuntimeError(
                "Setting displacements to type-1 dataset is not supported."
            )

        self._dataset["displacements"] = disp
        self._supercells_with_displacements = None

    def get_displacements(self):
        """Return displacements in supercells."""
        warnings.warn(
            "Phonopy.get_displacements() is deprecated."
            "Use Phonopy.displacements attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.displacements

    @property
    def force_constants(self) -> np.ndarray:
        """Getter and setter of supercell force constants.

        Force constants matrix.

        ndarray to get
            There are two shapes:
            full:
                shape=(atoms in supercell, atoms in supercell, 3, 3)
            compact:
                shape=(atoms in primitive cell, atoms in supercell, 3, 3)
            dtype='double', order='C'

        array_like to set
            If this is given in own condiguous ndarray with order='C' and
            dtype='double', internal copy of data is avoided. Therefore
            some computational resources are saved.
            shape=(atoms in supercell, atoms in supercell, 3, 3),
            dtype='double'

        """
        return self._force_constants

    @force_constants.setter
    def force_constants(self, force_constants):
        if type(force_constants) is np.ndarray:
            fc_shape = force_constants.shape
            if fc_shape[0] != fc_shape[1]:
                if len(self._primitive) != fc_shape[0]:
                    msg = (
                        "Force constants shape disagrees with crystal "
                        "structure setting. This may be due to "
                        "PRIMITIVE_AXIS."
                    )
                    raise RuntimeError(msg)

        self._force_constants = force_constants
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def get_force_constants(self):
        """Return supercell force constants."""
        warnings.warn(
            "Phonopy.get_force_constants() is deprecated."
            "Use Phonopy.force_constants attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.force_constants

    def set_force_constants(self, force_constants, show_drift=True):
        """Set force constants."""
        warnings.warn(
            "Phonopy.set_force_constants() is deprecated."
            "Use Phonopy.force_constants attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.force_constants = force_constants
        if show_drift and self._log_level:
            show_drift_force_constants(self._force_constants, primitive=self._primitive)

    def set_force_constants_zero_with_radius(self, cutoff_radius):
        """Set zero to force constants within cutoff radius."""
        cutoff_force_constants(
            self._force_constants,
            self._supercell,
            self._primitive,
            cutoff_radius,
            symprec=self._symprec,
        )
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    @property
    def supercell_energies(self) -> np.ndarray:
        """Return energies of supercells.

        Returns
        -------
        ndarray
            shape=(len(supercells),)

        """
        return self._get_forces_energies(target="supercell_energies")

    @supercell_energies.setter
    def supercell_energies(self, set_of_energies):
        self._set_forces_energies(set_of_energies, target="supercell_energies")

    @property
    def forces(self) -> np.ndarray:
        """Return forces of supercells.

        ndarray to get and array_like to set
            A set of atomic forces in displaced supercells. The order of
            displaced supercells has to match with that in displacement
            dataset.
            shape=(supercells with displacements, atoms in supercell, 3)
            dtype='double', order='C'

            [[[f_1x, f_1y, f_1z], [f_2x, f_2y, f_2z], ...], # first supercell
             [[f_1x, f_1y, f_1z], [f_2x, f_2y, f_2z], ...], # second supercell
             ...
            ]

        """
        return self._get_forces_energies(target="forces")

    @forces.setter
    def forces(self, sets_of_forces):
        self._set_forces_energies(sets_of_forces, target="forces")

    def set_forces(self, sets_of_forces):
        """Set forces of supercells."""
        warnings.warn(
            "Phonopy.set_forces() is deprecated." "Use Phonopy.forces attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.forces = sets_of_forces

    @property
    def dynamical_matrix(self) -> DynamicalMatrix:
        """Return DynamicalMatrix instance.

        This is not dynamical matrices but the instance of DynamicalMatrix
        class.

        """
        return self._dynamical_matrix

    def get_dynamical_matrix(self):
        """Return DynamicalMatrix instance."""
        warnings.warn(
            "Phonopy.get_dynamical_matrix() is deprecated."
            "Use Phonopy.dynamical_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dynamical_matrix

    @property
    def nac_params(self) -> dict:
        """Getter and setter of parameters for non-analytical term correction.

        dict
            Parameters used for non-analytical term correction
            'born': ndarray
                Born effective charges.
                shape=(primitive cell atoms, 3, 3), dtype='double', order='C'
            'dielectric': ndarray
                Dielectric constant tensor.
                shape=(3, 3), dtype='double', order='C'
            'factor': float, optional
                Unit conversion factor.
            'method': str, optional
                Method to calculate NAC.

        """
        return self._nac_params

    @nac_params.setter
    def nac_params(self, nac_params):
        self._nac_params = nac_params
        if self._force_constants is not None:
            self._set_dynamical_matrix()

    def get_nac_params(self):
        """Return parameters for non-analytical term correction."""
        warnings.warn(
            "Phonopy.get_nac_params() is deprecated."
            "Use Phonopy.nac_params attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.nac_params

    def set_nac_params(self, nac_params):
        """Set parameters for non-analytical term correction."""
        warnings.warn(
            "Phonopy.set_nac_params() is deprecated."
            "Use Phonopy.nac_params attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.nac_params = nac_params

    @property
    def supercells_with_displacements(self) -> Optional[list[PhonopyAtoms]]:
        """Return supercells with displacements.

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phonopy.generate_displacements.

        """
        if self._dataset is None:
            return None
        else:
            if self._supercells_with_displacements is None:
                self._build_supercells_with_displacements()
            return self._supercells_with_displacements

    def get_supercells_with_displacements(self):
        """Return supercells with displacements."""
        warnings.warn(
            "Phonopy.get_supercells_with_displacements() is deprecated."
            "Use Phonopy.supercells_with_displacements attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.supercells_with_displacements

    @property
    def mesh_numbers(self) -> np.ndarray:
        """Return sampling mesh numbers in reciprocal space."""
        if self._mesh is None:
            return None
        else:
            return self._mesh.mesh_numbers

    @property
    def qpoints(self) -> QpointsPhonon:
        """Return QpointsPhonon instance."""
        return self._qpoints

    @property
    def band_structure(self) -> BandStructure:
        """Return BandStructure instance."""
        return self._band_structure

    @property
    def group_velocity(self) -> GroupVelocity:
        """Return GroupVelocity instance."""
        return self._group_velocity

    @property
    def mesh(self) -> Union[Mesh, IterMesh]:
        """Return Mesh or IterMesh instance."""
        return self._mesh

    @property
    def random_displacements(self) -> RandomDisplacements:
        """Return RandomDisplacements instance."""
        return self._random_displacements

    @property
    def dynamic_structure_factor(self) -> DynamicStructureFactor:
        """Return DynamicStructureFactor instance."""
        return self._dynamic_structure_factor

    @property
    def thermal_properties(self) -> ThermalProperties:
        """Return ThermalProperties instance."""
        return self._thermal_properties

    @property
    def thermal_displacements(self) -> ThermalDisplacements:
        """Return ThermalDisplacements instance."""
        return self._thermal_displacements

    @property
    def thermal_displacement_matrices(self) -> ThermalDisplacementMatrices:
        """Return ThermalDisplacementMatrices instance."""
        return self._thermal_displacement_matrices

    @property
    def irreps(self) -> IrReps:
        """Return IrReps instance."""
        return self._irreps

    @property
    def moment(self) -> PhononMoment:
        """Return PhononMoment instance."""
        return self._moment

    @property
    def total_dos(self) -> TotalDos:
        """Return TotalDos instance."""
        return self._total_dos

    @property
    def partial_dos(self):
        """Return PartialDos instance."""
        warnings.warn(
            "Phonopy.partial_dos is deprecated." "Use Phonopy.projected_dos.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.projected_dos

    @property
    def projected_dos(self) -> ProjectedDos:
        """Return ProjectedDOS instance."""
        return self._pdos

    @property
    def masses(self) -> np.ndarray:
        """Getter and setter of masses of primitive cell atoms."""
        return self._primitive.masses

    @masses.setter
    def masses(self, masses):
        p_masses = np.array(masses)
        self._primitive.set_masses(p_masses)
        p2p_map = self._primitive.p2p_map
        s_masses = p_masses[[p2p_map[x] for x in self._primitive.s2p_map]]
        self._supercell.set_masses(s_masses)
        u2s_map = self._supercell.u2s_map
        u_masses = s_masses[u2s_map]
        self._unitcell.set_masses(u_masses)
        if self._force_constants is not None:
            self._set_dynamical_matrix()

    def set_masses(self, masses):
        """Set masses of primitive cell atoms."""
        self.masses = masses

    def generate_displacements(
        self,
        distance: Optional[float] = None,
        is_plusminus: Union[str, bool] = "auto",
        is_diagonal: bool = True,
        is_trigonal: bool = False,
        number_of_snapshots: Optional[int] = None,
        random_seed: Optional[int] = None,
        temperature: Optional[float] = None,
        cutoff_frequency: Optional[float] = None,
        max_distance: Optional[float] = None,
    ) -> None:
        """Generate displacement dataset.

        There are two modes, finite difference method with systematic
        displacements and fitting approach between arbitrary displacements and
        their forces. The default approach is the finite difference method that
        is built-in phonopy. The fitting approach requires external force
        constant calculator.

        The random displacement supercells are created by setting positive
        integer values 'number_of_snapshots' keyword argument. Unless this is
        specified, systematic displacements are created for the finite
        difference method as the default behaviour.

        Parameters
        ----------
        distance : float, optional
            Displacement distance. Unit is the same as that used for crystal
            structure. Default is 0.01. For random direction and random distance
            displacements generation, this value is also used as `min_distance`,
            is used to replace generated random distances smaller than this
            value by this value.
        is_plusminus : 'auto', True, or False, optional
            For each atom, displacement of one direction (False), both
            direction, i.e., one directiona and its opposite direction (True),
            and both direction if symmetry requires ('auto'). Default is 'auto'.
        is_diagonal : bool, optional
            Displacements are made only along basis vectors (False) and can be
            made not being along basis vectors if the number of displacements
            can be reduced by symmetry (True). Default is True.
        is_trigonal : bool, optional
            Existing only testing purpose. Default is False.
        number_of_snapshots : int or None, optional
            Number of snapshots of supercells with random displacements. Random
            displacements are generated displacing all atoms in random
            directions with a fixed displacement distance specified by
            'distance' parameter, i.e., all atoms in supercell are displaced
            with the same displacement distance in direct space. Default is
            None.
        random_seed : int or None, optional
            Random seed for random displacements generation. Default is None.
        temperature : float or None, optional
            With given temperature, random displacements at temperature is
            generated by sampling probability distribution from canonical
            ensemble of harmonic oscillators (harmonic phonons). Default is
            None.
        cutoff_frequency : float or None, optional
            In random displacements generation from canonical ensemble of
            harmonic phonons, phonon occupation number is used to determine the
            deviation of the distribution function. To avoid too large
            deviation, this value is used to exclude the phonon modes whose
            absolute frequency are smaller than this value. Default is None.
        max_distance : float or None, optional
            In random displacements generation from canonical ensemble of
            harmonic phonons, displacements larger than max distance are
            renormalized to the max distance, i.e., a displacement d is shorten
            by d -> d / |d| * max_distance if |d| > max_distance. In random
            direction and distance displacements generation, this value is
            specified. In random direction and random distance displacements
            generation, this value is used as `max_distance`.

        """
        if number_of_snapshots is not None and number_of_snapshots > 0:
            if random_seed is not None and random_seed >= 0 and random_seed < 2**32:
                _random_seed = random_seed
                displacement_dataset = {"random_seed": _random_seed}
            else:
                _random_seed = None
                displacement_dataset = {}
            if temperature is None:
                if distance is None:
                    _distance = 0.01
                else:
                    _distance = distance
                d = get_random_displacements_dataset(
                    number_of_snapshots,
                    len(self._supercell),
                    _distance,
                    random_seed=_random_seed,
                    is_plusminus=(is_plusminus is True),
                    max_distance=max_distance,
                )
                displacement_dataset["displacements"] = d
            else:
                self.init_random_displacements(
                    cutoff_frequency=cutoff_frequency, max_distance=max_distance
                )
                d = self.get_random_displacements_at_temperature(
                    temperature,
                    number_of_snapshots,
                    is_plusminus=(is_plusminus is True),
                    random_seed=_random_seed,
                )
                displacement_dataset["displacements"] = d
        else:
            if distance is None:
                _distance = 0.01
            else:
                _distance = distance
            displacement_directions = get_least_displacements(
                self._symmetry,
                is_plusminus=is_plusminus,
                is_diagonal=is_diagonal,
                is_trigonal=is_trigonal,
                log_level=self._log_level,
            )
            displacement_dataset = directions_to_displacement_dataset(
                displacement_directions, _distance, self._supercell
            )
        self.dataset = displacement_dataset

    def produce_force_constants(
        self,
        forces: Optional[Sequence] = None,
        calculate_full_force_constants: bool = True,
        fc_calculator: Optional[str] = None,
        fc_calculator_options: Optional[str] = None,
        show_drift: bool = True,
        fc_calculator_log_level: Optional[int] = None,
    ) -> None:
        """Compute supercell force constants from forces-displacements dataset.

        Supercell force constants are computed from forces and displacements.
        As the default behaviour, those stored in dataset are used. But
        with setting ``forces``, this set of forces and the set of
        displacements stored in the dataset are used for the computation.

        Parameters
        ----------
        forces : array_like, optional
            See docstring of Phonopy.forces. Default is None.
        calculate_full_force_constants : Bool, optional
            With setting True, full force constants matrix is stored.
            With setting False, compact force constants matrix is stored.
            For more detail, see docstring of Phonopy.force_constants.
            Default is True.
        fc_calculator : str, optional
        fc_calculator_options : str, optional
            External force constants calculator is used. Currently,
            'alm' is supported. See more detail at the docstring of
            phonopy.interface.fc_calculator.get_fc2. Default is None.
        show_drift : Bool, optional
            With setting
        fc_calculator_log_level : int, optional
            Log level for force constants calculator.

        """
        if forces is not None:
            self.forces = forces

        if fc_calculator_log_level is None:
            fc_log_level = self._log_level
        else:
            fc_log_level = fc_calculator_log_level

        # A primitive check if 'forces' key is in displacement_dataset.
        if "first_atoms" in self._dataset:
            for disp in self._dataset["first_atoms"]:
                if "forces" not in disp:
                    raise ForcesetsNotFoundError("Force sets are not yet set.")
        elif "forces" not in self._dataset:
            raise ForcesetsNotFoundError("Force sets are not yet set.")

        self._run_force_constants_from_forces(
            is_compact_fc=not calculate_full_force_constants,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            decimals=self._force_constants_decimals,
            log_level=fc_log_level,
        )

        if show_drift and self._log_level:
            show_drift_force_constants(self._force_constants, primitive=self._primitive)

        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def symmetrize_force_constants(self, level=1, show_drift=True) -> None:
        """Symmetrize force constants.

        This applies translational and permutation symmetries successfully,
        but not simultaneously.

        Parameters
        ----------
        level : int, optional
            Application of translational and permulation symmetries is
            repeated by this number. Default is 1.
        show_drift : bool, optioanl
            Drift forces are displayed when True. Default is True.

        """
        if self._force_constants is None:
            raise RuntimeError("Force constants have not been produced yet.")

        if self._force_constants.shape[0] == self._force_constants.shape[1]:
            symmetrize_force_constants(self._force_constants, level=level)
        else:
            symmetrize_compact_force_constants(
                self._force_constants, self._primitive, level=level
            )
        if show_drift and self._log_level:
            sys.stdout.write("Max drift after symmetrization by translation: ")
            show_drift_force_constants(
                self._force_constants, primitive=self._primitive, values_only=True
            )

        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def symmetrize_force_constants_by_space_group(self, show_drift=True) -> None:
        """Symmetrize force constants using space group operations.

        Space group operations except for pure translations are applied
        to force constants.

        Parameters
        ----------
        show_drift : bool, optioanl
            Drift forces are displayed when True. Default is True.

        """
        set_tensor_symmetry_PJ(
            self._force_constants,
            self._supercell.cell.T,
            self._supercell.scaled_positions,
            self._symmetry,
        )

        if show_drift and self._log_level:
            sys.stdout.write("Max drift after symmetrization by space group: ")
            show_drift_force_constants(
                self._force_constants, primitive=self._primitive, values_only=True
            )

        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def develop_mlp(
        self,
        params: Optional[Union[PypolymlpParams, dict, str]] = None,
        test_size: float = 0.1,
        log_level: Optional[int] = None,
    ):
        """Develop machine learning potential.

        Parameters
        ----------
        params : PypolymlpParams or dict, optional
            Parameters for developing MLP. Default is None. When dict is given,
            PypolymlpParams instance is created from the dict.
        test_size : float, optional
            Training and test data are splitted by this ratio. test_size=0.1
            means the first 90% of the data is used for training and the rest
            is used for test. Default is 0.1.

        """
        if self._mlp_dataset is None:
            raise RuntimeError("MLP dataset is not set.")

        if log_level is None:
            self._mlp = PhonopyMLP(log_level=self._log_level)
        else:
            self._mlp = PhonopyMLP(log_level=log_level)
        self._mlp.develop(
            self._mlp_dataset,
            self._supercell,
            params=params,
            test_size=test_size,
        )

    def save_mlp(self, filename: Optional[str] = None):
        """Save machine learning potential."""
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        self._mlp.save(filename=filename)

    def load_mlp(self, filename: Optional[Union[str, bytes, os.PathLike]] = None):
        """Load machine learning potential."""
        self._mlp = PhonopyMLP(log_level=self._log_level)
        self._mlp.load(filename=filename)

    def evaluate_mlp(self):
        """Evaluate machine learning potential.

        This method calculates the supercell energies and forces from the MLP
        for the displacements in self._dataset of type 2. The results are stored
        in self._dataset.

        The displacements may be generated by the produce_force_constants method
        with number_of_snapshots > 0. With MLP, a small distance parameter, such
        as 0.001, can be numerically stable for the computation of force
        constants.

        """
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        if self.supercells_with_displacements is None:
            raise RuntimeError("Displacements are not set. Run generate_displacements.")

        energies, forces, _ = self._mlp.evaluate(self.supercells_with_displacements)
        self.supercell_energies = energies
        self.forces = forces

    #####################
    # Phonon properties #
    #####################

    # Single q-point
    def get_dynamical_matrix_at_q(self, q) -> np.ndarray:
        """Calculate dynamical matrix at a given q-point.

        Parameters
        ----------
        q: array_like
            A q-vector.
            shape=(3,), dtype='double'

        Returns
        -------
        dynamical_matrix: ndarray
            Dynamical matrix.
            shape=(bands, bands)
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
            order='C'

        """
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        return self._dynamical_matrix.dynamical_matrix

    def get_frequencies(self, q) -> np.ndarray:
        """Calculate phonon frequencies at a given q-point.

        Parameters
        ----------
        q: array_like
            A q-vector.
            shape=(3,), dtype='double'

        Returns
        -------
        frequencies: ndarray
            Phonon frequencies. Imaginary frequenies are represented by
            negative real numbers.
            shape=(bands, ), dtype='double'

        """
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        frequencies = []
        for eig in np.linalg.eigvalsh(dm).real:
            if eig < 0:
                frequencies.append(-np.sqrt(-eig))
            else:
                frequencies.append(np.sqrt(eig))

        return np.array(frequencies) * self._factor

    def get_frequencies_with_eigenvectors(self, q) -> np.ndarray:
        """Calculate phonon frequencies and eigenvectors at a given q-point.

        Parameters
        ----------
        q: array_like
            A q-vector.
            shape=(3,)

        Returns
        -------
        (frequencies, eigenvectors)

        frequencies: ndarray
            Phonon frequencies. Imaginary frequenies are represented by
            negative real numbers.
            shape=(bands, ), dtype='double', order='C'
        eigenvectors: ndarray
            Phonon eigenvectors.
            shape=(bands, bands)
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
            order='C'

        """
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        frequencies = []
        eigvals, eigenvectors = np.linalg.eigh(dm)
        frequencies = []
        for eig in eigvals:
            if eig < 0:
                frequencies.append(-np.sqrt(-eig))
            else:
                frequencies.append(np.sqrt(eig))

        return np.array(frequencies) * self._factor, eigenvectors

    # Band structure
    def run_band_structure(
        self,
        paths,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_band_connection=False,
        path_connections=None,
        labels=None,
        is_legacy_plot=False,
    ) -> None:
        """Run phonon band structure calculation.

        Parameters
        ----------
        paths : List of array_like
            Sets of qpoints that can be passed to phonopy.set_band_structure().
            Numbers of qpoints can be different.
            shape of each array_like : (qpoints, 3)
        with_eigenvectors : bool, optional
            Flag whether eigenvectors are calculated or not. Default is False.
        with_group_velocities : bool, optional
            Flag whether group velocities are calculated or not. Default is
            False.
        is_band_connection : bool, optional
            Flag whether each band is connected or not. This is achieved by
            comparing similarity of eigenvectors of neghboring poins. Sometimes
            this fails. Default is False.
        path_connections : List of bool, optional
            This is only used in graphical plot of band structure and gives
            whether each path is connected to the next path or not,
            i.e., if False, there is a jump of q-points. Number of elements is
            the same at that of paths. Default is None.
        labels : List of str, optional
            This is only used in graphical plot of band structure and gives
            labels of end points of each path. The number of labels is equal
            to (2 - np.array(path_connections)).sum().
        is_legacy_plot: bool, optional
            This makes the old style band structure plot. Default is False.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        self._band_structure = BandStructure(
            paths,
            self._dynamical_matrix,
            with_eigenvectors=with_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=group_velocity,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=is_legacy_plot,
            factor=self._factor,
        )

    def set_band_structure(
        self,
        bands,
        is_eigenvectors=False,
        is_band_connection=False,
        path_connections=None,
        labels=None,
        is_legacy_plot=False,
    ):
        """Calculate phonon band structure."""
        warnings.warn(
            "Phonopy.set_band_structure() is deprecated. "
            "Use Phonopy.run_band_structure().",
            DeprecationWarning,
            stacklevel=2,
        )

        if self._group_velocity is None:
            with_group_velocities = False
        else:
            with_group_velocities = True
        self.run_band_structure(
            bands,
            with_eigenvectors=is_eigenvectors,
            with_group_velocities=with_group_velocities,
            is_band_connection=is_band_connection,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=is_legacy_plot,
        )

    def get_band_structure_dict(self) -> dict:
        """Return calculated band structures.

        Returns
        -------
        dict
            keys: qpoints, distances, frequencies, eigenvectors, and
                  group_velocities
            Each dict value is a list containing properties on number of paths.
            The number of q-points on one path can be different from that on
            the other path. Each set of properties on a path is ndarray and is
            explained as below:

            qpoints[i]: ndarray
                q-points in reduced coordinates of reciprocal space without
                2pi.
                shape=(q-points, 3), dtype='double'
            distances[i]: ndarray
                Distances in reciprocal space along paths.
                shape=(q-points,), dtype='double'
            frequencies[i]: ndarray
                Phonon frequencies. Imaginary frequenies are represented by
                negative real numbers.
                shape=(q-points, bands), dtype='double'
            eigenvectors[i]: ndarray
                Phonon eigenvectors. None if eigenvectors are not stored.
                shape=(q-points, bands, bands)
                dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
                order='C'
            group_velocities[i]: ndarray
                Phonon group velocities. None if group velocities are not
                calculated.
                shape=(q-points, bands, 3), dtype='double'

        """
        if self._band_structure is None:
            msg = "Phonopy.run_band_structure() has to be done."
            raise RuntimeError(msg)

        retdict = {
            "qpoints": self._band_structure.qpoints,
            "distances": self._band_structure.distances,
            "frequencies": self._band_structure.frequencies,
            "eigenvectors": self._band_structure.eigenvectors,
            "group_velocities": self._band_structure.group_velocities,
        }

        return retdict

    def get_band_structure(self):
        """Return calculated band structures.

        Returns
        -------
        (q-points, distances, frequencies, eigenvectors)

        Each tuple element is a list containing properties on number of paths.
        The number of q-points on one path can be different from that on the
        other path. Each set of properties on a path is ndarray and is
        explained as below:

        q-points[i]: ndarray
            q-points in reduced coordinates of reciprocal space without 2pi.
            shape=(q-points, 3), dtype='double'
        distances[i]: ndarray
            Distances in reciprocal space along paths.
            shape=(q-points,), dtype='double'
        frequencies[i]: ndarray
            Phonon frequencies. Imaginary frequenies are represented by
            negative real numbers.
            shape=(q-points, bands), dtype='double'
        eigenvectors[i]: ndarray
            Phonon eigenvectors. None if eigenvectors are not stored.
            shape=(q-points, bands, bands)
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
            order='C'
        group_velocities[i]: ndarray
            Phonon group velocities. None if group velocities are not
            calculated.
            shape=(q-points, bands, 3), dtype='double'

        """
        warnings.warn(
            "Phonopy.get_band_structure() is deprecated. "
            "Use Phonopy.get_band_structure_dict().",
            DeprecationWarning,
            stacklevel=2,
        )

        if self._band_structure is None:
            msg = "run_band_structure has to be done."
            raise RuntimeError(msg)

        retvals = (
            self._band_structure.qpoints,
            self._band_structure.distances,
            self._band_structure.frequencies,
            self._band_structure.eigenvectors,
        )
        return retvals

    def auto_band_structure(
        self,
        npoints=101,
        with_eigenvectors=False,
        with_group_velocities=False,
        plot=False,
        write_yaml=False,
        filename="band.yaml",
    ):
        """Conveniently calculate and draw band structure.

        Parameters
        ----------
        See docstring of ``Phonopy.run_band_structure`` for the parameters of
        ``with_eigenvectors`` (default is False) and ``with_group_velocities``
        (default is False).

        npoints : int, optional
            Number of q-points in each segment of band struture paths.
            The number includes end points. Default is 101.
        plot : Bool, optional
            With setting True, band structure is plotted using matplotlib and
            the matplotlib module (plt) is returned. To watch the result,
            usually ``show()`` has to be called. Default is False.
        write_yaml : Bool
            With setting True, ``band.yaml`` like file is written out. The
            file name can be specified with the ``filename`` parameter.
            Default is False.
        filename : str, optional
            File name used to write ``band.yaml`` like file. Default is
            ``band.yaml``.

        """
        bands, labels, path_connections = get_band_qpoints_by_seekpath(
            self._primitive, npoints, is_const_interval=True
        )
        self.run_band_structure(
            bands,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=False,
        )
        if write_yaml:
            self.write_yaml_band_structure(filename=filename)
        if plot:
            return self.plot_band_structure()

    def plot_band_structure(self):
        """Plot calculated band structure.

        Returns
        -------
        matplotlib.pyplot.

        """
        import matplotlib.pyplot as plt

        if self._band_structure is None:
            msg = "run_band_structure has to be done."
            raise RuntimeError(msg)

        if self._band_structure.is_legacy_plot:
            fig, axs = plt.subplots(1, 1)
        else:
            from mpl_toolkits.axes_grid1 import ImageGrid

            n = len([x for x in self._band_structure.path_connections if not x])
            fig = plt.figure()
            axs = ImageGrid(
                fig,
                111,  # similar to subplot(111)
                nrows_ncols=(1, n),
                axes_pad=0.11,
                label_mode="L",
            )
        self._band_structure.plot(axs)
        return plt

    def write_hdf5_band_structure(self, comment=None, filename="band.hdf5") -> None:
        """Write band structure in hdf5 format.

        Parameters
        ----------
        comment : dict, optional
            Items are stored in hdf5 file in the way of key-value pair.
        filename : str, optional
            Default is ``band.hdf5``.

        """
        self._band_structure.write_hdf5(comment=comment, filename=filename)

    def write_yaml_band_structure(
        self, comment=None, filename=None, compression=None
    ) -> None:
        """Write band structure in yaml.

        Parameters
        ----------
        comment : dict
            Data structure dumped in YAML and the dumped YAML text is put
            at the beggining of the file.
        filename : str
            Default filename is 'band.yaml' when compression=None.
            With compression, an extention of filename is added such as
            'band.yaml.xz'.
        compression : None, 'gzip', or 'lzma'
            None gives usual text file. 'gzip and 'lzma' compresse yaml
            text in respective compression methods.

        """
        self._band_structure.write_yaml(
            comment=comment, filename=filename, compression=compression
        )

    def init_mesh(
        self,
        mesh=100.0,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_gamma_center=False,
        use_iter_mesh=False,
    ) -> None:
        """Initialize mesh sampling phonon calculation without starting to run.

        Phonon calculation starts explicitly with calling Mesh.run() or
        implicitly with accessing getters of Mesh instance, e.g.,
        Mesh.frequencies.

        Parameters
        ----------
        mesh: array_like or float, optional
            Mesh numbers along a, b, c axes when array_like object is given.
            dtype='intc', shape=(3,)
            When float value is given, uniform mesh is generated following
            VASP convention by
                N = max(1, nint(l * |a|^*))
            where 'nint' is the function to return the nearest integer. In this
            case, it is forced to set is_gamma_center=True.
            Default value is 100.0.
        shift: array_like, optional
            Mesh shifts along a*, b*, c* axes with respect to neighboring grid
            points from the original mesh (Monkhorst-Pack or Gamma center).
            0.5 gives half grid shift. Normally 0 or 0.5 is given.
            Otherwise q-points symmetry search is not performed.
            Default is None (no additional shift).
            dtype='double', shape=(3, )
        is_time_reversal: bool, optional
            Time reversal symmetry is considered in symmetry search. By this,
            inversion symmetry is always included. Default is True.
        is_mesh_symmetry: bool, optional
            Wheather symmetry search is done or not. Default is True
        with_eigenvectors: bool, optional
            Eigenvectors are stored by setting True. Default False.
        with_group_velocities : bool, optional
            Group velocities are calculated by setting True. Default is
            False.
        is_gamma_center: bool, default False
            Uniform mesh grids are generated centring at Gamma point but not
            the Monkhorst-Pack scheme. When type(mesh) is float, this parameter
            setting is ignored and it is forced to set is_gamma_center=True.
        use_iter_mesh: bool
            Use IterMesh instead of Mesh class not to store phonon properties
            in its instance to save memory consumption. This is used with
            ThermalDisplacements and ThermalDisplacementMatrices.
            Default is False.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        _mesh = np.array(mesh)
        mesh_nums = None
        if _mesh.shape:
            if _mesh.shape == (3,):
                mesh_nums = mesh
                _is_gamma_center = is_gamma_center
        else:
            if self._primitive_symmetry is not None:
                rots = self._primitive_symmetry.pointgroup_operations
                mesh_nums = length2mesh(mesh, self._primitive.cell, rotations=rots)
            else:
                mesh_nums = length2mesh(mesh, self._primitive.cell)
            _is_gamma_center = True
        if mesh_nums is None:
            msg = "mesh has inappropriate type."
            raise TypeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        if use_iter_mesh:
            self._mesh = IterMesh(
                self._dynamical_matrix,
                mesh_nums,
                shift=shift,
                is_time_reversal=is_time_reversal,
                is_mesh_symmetry=is_mesh_symmetry,
                with_eigenvectors=with_eigenvectors,
                is_gamma_center=is_gamma_center,
                rotations=self._primitive_symmetry.pointgroup_operations,
                factor=self._factor,
            )
        else:
            self._mesh = Mesh(
                self._dynamical_matrix,
                mesh_nums,
                shift=shift,
                is_time_reversal=is_time_reversal,
                is_mesh_symmetry=is_mesh_symmetry,
                with_eigenvectors=with_eigenvectors,
                is_gamma_center=_is_gamma_center,
                group_velocity=group_velocity,
                rotations=self._primitive_symmetry.pointgroup_operations,
                factor=self._factor,
            )

    def run_mesh(
        self,
        mesh=100.0,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_gamma_center=False,
    ) -> None:
        """Run mesh sampling phonon calculation.

        See the parameter details in Phonopy.init_mesh.

        """
        self.init_mesh(
            mesh=mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities,
            is_gamma_center=is_gamma_center,
        )
        self._mesh.run()

    def set_mesh(
        self,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        is_eigenvectors=False,
        is_gamma_center=False,
        run_immediately=True,
    ):
        """Run or initialize phonon calculations on sampling mesh grids.

        Parameters
        ----------
        mesh: array_like
            Mesh numbers along a, b, c axes.
            dtype='intc'
            shape=(3,)
        shift: array_like, optional, default None (no shift)
            Mesh shifts along a*, b*, c* axes with respect to neighboring grid
            points from the original mesh (Monkhorst-Pack or Gamma center).
            0.5 gives half grid shift. Normally 0 or 0.5 is given.
            Otherwise q-points symmetry search is not performed.
            dtype='double'
            shape=(3, )
        is_time_reversal: bool, optional, default True
            Time reversal symmetry is considered in symmetry search. By this,
            inversion symmetry is always included.
        is_mesh_symmetry: bool, optional, default True
            Wheather symmetry search is done or not.
        is_eigenvectors: bool, optional, default False
            Eigenvectors are stored by setting True.
        is_gamma_center: bool, default False
            Uniform mesh grids are generated centring at Gamma point but not
            the Monkhorst-Pack scheme.
        run_immediately: bool, default True
            With True, phonon calculations are performed immediately, which is
            usual usage.

        """
        warnings.warn(
            "Phonopy.set_mesh is deprecated. " "Use Phonopy.run_mesh.",
            DeprecationWarning,
            stacklevel=2,
        )

        if self._group_velocity is None:
            with_group_velocities = False
        else:
            with_group_velocities = True
        if run_immediately:
            self.run_mesh(
                mesh,
                shift=shift,
                is_time_reversal=is_time_reversal,
                is_mesh_symmetry=is_mesh_symmetry,
                with_eigenvectors=is_eigenvectors,
                with_group_velocities=with_group_velocities,
                is_gamma_center=is_gamma_center,
            )
        else:
            self.init_mesh(
                mesh,
                shift=shift,
                is_time_reversal=is_time_reversal,
                is_mesh_symmetry=is_mesh_symmetry,
                with_eigenvectors=is_eigenvectors,
                with_group_velocities=with_group_velocities,
                is_gamma_center=is_gamma_center,
            )

    def get_mesh_dict(self) -> dict:
        """Return phonon properties calculated by mesh sampling.

        Returns
        -------
        dict
            keys: qpoints, weights, frequencies, eigenvectors, and
                  group_velocities

            Each value for the corresponding key is explained as below.

            qpoints: ndarray
                q-points in reduced coordinates of reciprocal lattice
                dtype='double'
                shape=(ir-grid points, 3)
            weights: ndarray
                Geometric q-point weights. Its sum is the number of grid
                points.
                dtype='intc'
                shape=(ir-grid points,)
            frequencies: ndarray
                Phonon frequencies at ir-grid points. Imaginary frequenies are
                represented by negative real numbers.
                dtype='double'
                shape=(ir-grid points, bands)
            eigenvectors: ndarray
                Phonon eigenvectors at ir-grid points. See the data structure
                at np.linalg.eigh.
                shape=(ir-grid points, bands, bands)
                dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
                order='C'
            group_velocities: ndarray
                Phonon group velocities at ir-grid points.
                dtype='double'
                shape=(ir-grid points, bands, 3)

        """
        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)

        retdict = {
            "qpoints": self._mesh.qpoints,
            "weights": self._mesh.weights,
            "frequencies": self._mesh.frequencies,
            "eigenvectors": self._mesh.eigenvectors,
            "group_velocities": self._mesh.group_velocities,
        }

        return retdict

    def get_mesh(self):
        """Return phonon properties calculated by mesh sampling."""
        warnings.warn(
            "Phonopy.get_mesh() is deprecated. " "Use Phonopy.get_mesh_dict().",
            DeprecationWarning,
            stacklevel=2,
        )

        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)

        mesh_dict = self.get_mesh_dict()

        return (
            mesh_dict["qpoints"],
            mesh_dict["weights"],
            mesh_dict["frequencies"],
            mesh_dict["eigenvectors"],
        )

    def get_mesh_grid_info(self):
        """Return grid point information of mesh sampling."""
        warnings.warn(
            "Phonopy.get_mesh_grid_info() is deprecated. "
            "Use attributes of phonon.mesh instance.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)

        return (
            self._mesh.grid_address,
            self._mesh.ir_grid_points,
            self._mesh.grid_mapping_table,
        )

    def write_hdf5_mesh(self) -> None:
        """Write mesh calculation results in hdf5 format."""
        self._mesh.write_hdf5()

    def write_yaml_mesh(self) -> None:
        """Write mesh calculation results in yaml format."""
        self._mesh.write_yaml()

    # Sampling mesh:
    # Solving dynamical matrices at q-points one-by-one as an iterator
    def set_iter_mesh(
        self,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        is_eigenvectors=False,
        is_gamma_center=False,
    ):
        """Create an IterMesh instance.

        See set_mesh method.

        """
        warnings.warn(
            "Phonopy.set_iter_mesh() is deprecated. "
            "Use Phonopy.run_mesh() with use_iter_mesh=True.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.run_mesh(
            mesh=mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            with_eigenvectors=is_eigenvectors,
            is_gamma_center=is_gamma_center,
            use_iter_mesh=True,
        )

    # Plot band structure and DOS (PDOS) together
    def plot_band_structure_and_dos(self, pdos_indices=None):
        """Plot band structure and DOS."""
        import matplotlib.pyplot as plt

        if self._total_dos is None and pdos_indices is None:
            msg = "run_total_dos has to be done."
            raise RuntimeError(msg)
        if self._pdos is None and pdos_indices is not None:
            msg = "run_projected_dos has to be done."
            raise RuntimeError(msg)
        if self._band_structure is None:
            msg = "run_band_structure has to be done."
            raise RuntimeError(msg)

        if self._band_structure.is_legacy_plot:
            import matplotlib.gridspec as gridspec

            # plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax2 = plt.subplot(gs[0, 1])
            if pdos_indices is None:
                self._total_dos.plot(ax2, ylabel="", draw_grid=False, flip_xy=True)
            else:
                self._pdos.plot(
                    ax2, indices=pdos_indices, ylabel="", draw_grid=False, flip_xy=True
                )
            ax2.set_xlim((0, None))
            plt.setp(ax2.get_yticklabels(), visible=False)

            ax1 = plt.subplot(gs[0, 0], sharey=ax2)
            self._band_structure.plot(ax1)

            plt.subplots_adjust(wspace=0.03)
            plt.tight_layout()
        else:
            from mpl_toolkits.axes_grid1 import ImageGrid

            n = len([x for x in self._band_structure.path_connections if not x]) + 1
            fig = plt.figure()
            axs = ImageGrid(
                fig,
                111,  # similar to subplot(111)
                nrows_ncols=(1, n),
                axes_pad=0.11,
                label_mode="L",
            )
            self._band_structure.plot(axs[:-1])

            if pdos_indices is None:
                self._total_dos.plot(
                    axs[-1], xlabel="", ylabel="", draw_grid=False, flip_xy=True
                )
            else:
                self._pdos.plot(
                    axs[-1],
                    indices=pdos_indices,
                    xlabel="",
                    ylabel="",
                    draw_grid=False,
                    flip_xy=True,
                )
            xlim = axs[-1].get_xlim()
            ylim = axs[-1].get_ylim()
            aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
            axs[-1].set_aspect(aspect)
            axs[-1].axhline(y=0, linestyle=":", linewidth=0.5, color="b")
            axs[-1].set_xlim((0, None))

        return plt

    # Sampling at q-points
    def run_qpoints(
        self,
        q_points,
        with_eigenvectors=False,
        with_group_velocities=False,
        with_dynamical_matrices=False,
        nac_q_direction=None,
    ) -> None:
        """Run phonon calculation at specified q-points.

        Parameters
        ----------
        q_points: array_like or float, optional
            q-points in reduced coordinates.
            dtype='double', shape=(q-points, 3)
        with_eigenvectors: bool, optional
            Eigenvectors are stored by setting True. Default False.
        with_group_velocities : bool, optional
            Group velocities are calculated by setting True. Default is False.
        with_dynamical_matrices : bool, optional
            Calculated dynamical matrices are stored by setting True.
            Default is False.
        nac_q_direction : array_like
            q-point direction from Gamma-point in fractional coordinates of
            reciprocal basis vectors. Only the direction is used, i.e.,
            (q_direction / |q_direction|) is computed and used. This parameter
            is activated only at q=(0, 0, 0).
            shape=(3,), dtype='double'

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        self._qpoints = QpointsPhonon(
            np.reshape(q_points, (-1, 3)),
            self._dynamical_matrix,
            nac_q_direction=nac_q_direction,
            with_eigenvectors=with_eigenvectors,
            group_velocity=group_velocity,
            with_dynamical_matrices=with_dynamical_matrices,
            factor=self._factor,
        )

    def set_qpoints_phonon(
        self,
        q_points,
        nac_q_direction=None,
        is_eigenvectors=False,
        write_dynamical_matrices=False,
    ):
        """Run phonon calculation at specified q-points."""
        warnings.warn(
            "Phonopy.set_qpoints_phonon() is deprecated. " "Use Phonopy.run_qpoints().",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._group_velocity is None:
            with_group_velocities = False
        else:
            with_group_velocities = True
        self.run_qpoints(
            q_points,
            with_eigenvectors=is_eigenvectors,
            with_group_velocities=with_group_velocities,
            with_dynamical_matrices=write_dynamical_matrices,
            nac_q_direction=nac_q_direction,
        )

    def get_qpoints_dict(self) -> dict:
        """Return calculated phonon properties at q-points.

        Returns
        -------
        dict
            keys: frequencies, eigenvectors, and dynamical_matrices

            frequencies : ndarray
                Phonon frequencies. Imaginary frequenies are represented by
                negative real numbers.
                shape=(qpoints, bands), dtype='double'
            eigenvectors : ndarray
                Phonon eigenvectors. None if eigenvectors are not stored.
                shape=(qpoints, bands, bands)
                dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
                order='C'
            group_velocities : ndarray
                Phonon group velocities. None if group velocities are not
                calculated.
                shape=(qpoints, bands, 3), dtype='double'
            dynamical_matrices : ndarray
                Dynamical matrices at q-points.
                shape=(qpoints, bands, bands), dtype='double'

        """
        if self._qpoints is None:
            msg = "Phonopy.run_qpoints() has to be done."
            raise RuntimeError(msg)

        return {
            "frequencies": self._qpoints.frequencies,
            "eigenvectors": self._qpoints.eigenvectors,
            "group_velocities": self._qpoints.group_velocities,
            "dynamical_matrices": self._qpoints.dynamical_matrices,
        }

    def get_qpoints_phonon(self):
        """Return phonon properties calculated at q-points."""
        warnings.warn(
            "Phonopy.get_qpoints_phonon() is deprecated. "
            "Use Phonopy.run_get_qpoints_dict().",
            DeprecationWarning,
            stacklevel=2,
        )
        qpt = self.get_qpoints_dict()
        return (qpt["frequencies"], qpt["eigenvectors"])

    def write_hdf5_qpoints_phonon(self) -> None:
        """Write phonon properties calculated at q-points in hdf5 format."""
        self._qpoints.write_hdf5()

    def write_yaml_qpoints_phonon(self) -> None:
        """Write phonon properties calculated at q-points in yaml format."""
        self._qpoints.write_yaml()

    # DOS
    def run_total_dos(
        self,
        sigma=None,
        freq_min=None,
        freq_max=None,
        freq_pitch=None,
        use_tetrahedron_method=True,
    ) -> None:
        """Run total DOS calculation.

        Parameters
        ----------
        sigma : float, optional
            Smearing width for smearing method. Default is None
        freq_min, freq_max, freq_pitch : float, optional
            Minimum and maximum frequencies in which range DOS is computed
            with the specified interval (freq_pitch).
            Defaults are None and they are automatically determined.
        use_tetrahedron_method : float, optional
            Use tetrahedron method when this is True. When sigma is set,
            smearing method is used.

        """
        if self._mesh is None:
            msg = "run_mesh has to be done before DOS calculation."
            raise RuntimeError(msg)

        total_dos = TotalDos(
            self._mesh, sigma=sigma, use_tetrahedron_method=use_tetrahedron_method
        )
        total_dos.set_draw_area(freq_min, freq_max, freq_pitch)
        total_dos.run()
        self._total_dos = total_dos

    def set_total_DOS(
        self,
        sigma=None,
        freq_min=None,
        freq_max=None,
        freq_pitch=None,
        tetrahedron_method=False,
    ):
        """Run total DOS calculation."""
        warnings.warn(
            "Phonopy.set_total_DOS() is deprecated. " "Use Phonopy.run_total_DOS()",
            DeprecationWarning,
            stacklevel=2,
        )

        self.run_total_dos(
            sigma=sigma,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_pitch=freq_pitch,
            use_tetrahedron_method=tetrahedron_method,
        )

    def auto_total_dos(
        self,
        mesh=100.0,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        is_gamma_center=False,
        plot=False,
        xlabel=None,
        ylabel=None,
        with_tight_frequency_range=False,
        write_dat=False,
        filename="total_dos.dat",
    ) -> None:
        """Conveniently calculate and draw total DOS."""
        self.run_mesh(
            mesh=mesh,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            is_gamma_center=is_gamma_center,
        )
        self.run_total_dos()
        if write_dat:
            self.write_total_dos(filename=filename)
        if plot:
            return self.plot_total_dos(
                xlabel=xlabel,
                ylabel=ylabel,
                with_tight_frequency_range=with_tight_frequency_range,
            )

    def get_total_dos_dict(self) -> dict:
        """Return total DOS.

        Returns
        -------
        A dictionary with keys of 'frequency_points' and 'total_dos'.
        Each value of corresponding key is as follows:

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        total_dos:
            shape=(frequency_sampling_points, ), dtype='double'

        """
        return {
            "frequency_points": self._total_dos.frequency_points,
            "total_dos": self._total_dos.dos,
        }

    def get_total_DOS(self):
        """Return total DOS.

        Returns
        -------
        A tuple with (frequency_points, total_dos).

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        total_dos:
            shape=(frequency_sampling_points, ), dtype='double'

        """
        warnings.warn(
            "Phonopy.get_total_DOS() is deprecated. "
            "Use Phonopy.get_total_dos_dict().",
            DeprecationWarning,
            stacklevel=2,
        )

        dos = self.get_total_dos_dict()

        return dos["frequency_points"], dos["total_dos"]

    def set_Debye_frequency(self, freq_max_fit=None) -> None:
        """Calculate Debye frequency on top of total DOS."""
        self._total_dos.set_Debye_frequency(
            len(self._primitive), freq_max_fit=freq_max_fit
        )

    def get_Debye_frequency(self) -> float:
        """Return Debye frequency."""
        return self._total_dos.get_Debye_frequency()

    def plot_total_DOS(self):
        """Plot total DOS."""
        warnings.warn(
            "Phonopy.plot_total_DOS() is deprecated. "
            "Use Phonopy.plot_total_dos() (lowercase on DOS).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_total_dos()

    def plot_total_dos(
        self, xlabel=None, ylabel=None, with_tight_frequency_range=False
    ):
        """Plot total DOS.

        xlabel : str, optional
            x-label of plot. Default is None, which puts a default x-label.
        ylabel : str, optional
            y-label of plot. Default is None, which puts a default y-label.
        with_tight_frequency_range : bool, optional
            Plot with tight frequency range. Default is False.

        """
        if self._total_dos is None:
            msg = "run_total_dos has to be done before plotting " "total DOS."
            raise RuntimeError(msg)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._total_dos.plot(ax, xlabel=xlabel, ylabel=ylabel, draw_grid=False)
        if with_tight_frequency_range:
            fmin, fmax = get_dos_frequency_range(
                self._pdos.frequency_points, self._total_dos.dos
            )
            ax.set_xlim(fmin, fmax)
        ax.set_ylim((0, None))

        return plt

    def write_total_DOS(self, filename="total_dos.dat"):
        """Write total DOS to text file."""
        warnings.warn(
            "Phonopy.write_total_DOS() is deprecated. "
            "Use Phonopy.write_total_dos() (lowercase on DOS).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_total_dos(filename=filename)

    def write_total_dos(self, filename="total_dos.dat") -> None:
        """Write total DOS to text file."""
        self._total_dos.write(filename=filename)

    # PDOS
    def run_projected_dos(
        self,
        sigma=None,
        freq_min=None,
        freq_max=None,
        freq_pitch=None,
        use_tetrahedron_method=True,
        direction=None,
        xyz_projection=False,
    ) -> None:
        """Run projected DOS calculation.

        Parameters
        ----------
        sigma : float, optional
            Smearing width for smearing method. Default is None
        freq_min, freq_max, freq_pitch : float, optional
            Minimum and maximum frequencies in which range DOS is computed
            with the specified interval (freq_pitch).
            Defaults are None and they are automatically determined.
        use_tetrahedron_method : float, optional
            Use tetrahedron method when this is True. When sigma is set,
            smearing method is used.
        direction : array_like, optional
            Specific projection direction. This is specified three values
            along basis vectors or the primitive cell. Default is None,
            i.e., no projection.
        xyz_projection : bool, optional
            This determines whether projected along Cartesian directions or
            not. Default is False, i.e., no projection.

        """
        self._pdos = None

        if self._mesh is None:
            msg = "run_mesh has to be done before PDOS calculation."
            raise RuntimeError(msg)

        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be called with with_eigenvectors=True."
            raise RuntimeError(msg)

        if np.prod(self._mesh.mesh_numbers) != len(self._mesh.ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        if direction is not None:
            direction_cart = np.dot(direction, self._primitive.cell)
        else:
            direction_cart = None
        self._pdos = ProjectedDos(
            self._mesh,
            sigma=sigma,
            use_tetrahedron_method=use_tetrahedron_method,
            direction=direction_cart,
            xyz_projection=xyz_projection,
        )
        self._pdos.set_draw_area(freq_min, freq_max, freq_pitch)
        self._pdos.run()

    def set_partial_DOS(
        self,
        sigma=None,
        freq_min=None,
        freq_max=None,
        freq_pitch=None,
        tetrahedron_method=False,
        direction=None,
        xyz_projection=False,
    ):
        """Run projected DOS calculation."""
        warnings.warn(
            "Phonopy.set_partial_DOS() is deprecated. "
            "Use Phonopy.run_projected_dos()",
            DeprecationWarning,
            stacklevel=2,
        )

        self.run_projected_dos(
            sigma=sigma,
            freq_min=freq_min,
            freq_max=freq_max,
            freq_pitch=freq_pitch,
            use_tetrahedron_method=tetrahedron_method,
            direction=direction,
            xyz_projection=xyz_projection,
        )

    def auto_projected_dos(
        self,
        mesh=100.0,
        is_time_reversal=True,
        is_gamma_center=False,
        plot=False,
        pdos_indices=None,
        legend=None,
        legend_prop=None,
        legend_frameon=True,
        xlabel=None,
        ylabel=None,
        with_tight_frequency_range=False,
        write_dat=False,
        filename="projected_dos.dat",
    ) -> None:
        """Conveniently calculate and draw projected DOS.

        Parameters
        ----------
        See docstring of ``Phonopy.init_mesh`` for the parameters of ``mesh``
        (default is 100.0), ``is_time_reversal`` (default is True), and
        ``is_gamma_center`` (default is False). See docstring of
        ``Phonopy.plot_projected_dos`` for the parameters ``pdos_indices``,
        ``legend``, ``xlabel``, ``ylabel``, ``with_tight_frequency_range``.

        plot : Bool, optional
            With setting True, PDOS is plotted using matplotlib and the
            matplotlib module (plt) is returned. To watch the result, usually
            ``show()`` has to be called. Default is False.
        write_dat : Bool
            With setting True, ``projected_dos.dat`` like file is written out.
            The  file name can be specified with the ``filename`` parameter.
            Default is False.
        filename : str, optional
            File name used to write ``projected_dos.dat`` like file. Default is
            ``projected_dos.dat``.

        """
        self.run_mesh(
            mesh=mesh,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=False,
            with_eigenvectors=True,
            is_gamma_center=is_gamma_center,
        )
        self.run_projected_dos()
        if write_dat:
            self.write_projected_dos(filename=filename)
        if plot:
            return self.plot_projected_dos(
                pdos_indices=pdos_indices,
                legend=legend,
                legend_prop=legend_prop,
                legend_frameon=legend_frameon,
                xlabel=xlabel,
                ylabel=ylabel,
                with_tight_frequency_range=with_tight_frequency_range,
            )

    def get_projected_dos_dict(self) -> dict:
        """Return projected DOS.

        Projection is done to atoms and may be also done along directions
        depending on the parameters at run_projected_dos.

        Returns
        -------
        A dictionary with keys of 'frequency_points' and 'projected_dos'.
        Each value of corresponding key is as follows:

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        projected_dos:
            shape=(projections, frequency_sampling_points), dtype='double'

        """
        return {
            "frequency_points": self._pdos.frequency_points,
            "projected_dos": self._pdos.projected_dos,
        }

    def get_partial_DOS(self):
        """Return projected DOS.

        Projection is done to atoms and may be also done along directions
        depending on the parameters at run_partial_dos.

        Returns
        -------
        A tuple with (frequency_points, partial_dos).

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        partial_dos:
            shape=(projections, frequency_sampling_points), dtype='double'

        """
        warnings.warn(
            "Phonopy.get_partial_DOS() is deprecated. "
            "Use Phonopy.get_projected_dos_dict().",
            DeprecationWarning,
            stacklevel=2,
        )

        pdos = self.get_projected_dos_dict()

        return pdos["frequency_points"], pdos["projected_dos"]

    def plot_partial_DOS(self, pdos_indices=None, legend=None):
        """Plot projected DOS."""
        warnings.warn(
            "Phonopy.plot_partial_DOS() is deprecated. "
            "Use Phonopy.plot_projected_dos() (lowercase on DOS).",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.plot_projected_dos(pdos_indices=pdos_indices, legend=legend)

    def plot_projected_dos(
        self,
        pdos_indices=None,
        legend=None,
        legend_prop=None,
        legend_frameon=True,
        xlabel=None,
        ylabel=None,
        with_tight_frequency_range=False,
    ):
        """Plot projected DOS.

        Parameters
        ----------
        pdos_indices : list of list, optional
            Sets of indices of atoms whose projected DOS are summed over.
            The indices start with 0. An example is as follwos:
                pdos_indices=[[0, 1], [2, 3, 4, 5]]
            Default is None, which means
                pdos_indices=[[i] for i in range(natom)]
        legend : list of instances such as str or int, optional
            The str(instance) are shown in legend.
            It has to be len(pdos_indices)==len(legend). Default is None.
            When None, legend is not shown.
        legend_prop : dict, optional
            Legend properties of matplotlib. Default is None.
        legend_frameon : bool, optional
            Legend with frame or not. Default is True.
        xlabel : str, optional
            x-label of plot. Default is None, which puts a default x-label.
        ylabel : str, optional
            y-label of plot. Default is None, which puts a default y-label.
        with_tight_frequency_range : bool, optional
            Plot with tight frequency range. Default is False.

        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._pdos.plot(
            ax,
            indices=pdos_indices,
            legend=legend,
            legend_prop=legend_prop,
            legend_frameon=legend_frameon,
            xlabel=xlabel,
            ylabel=ylabel,
            draw_grid=False,
        )

        if with_tight_frequency_range:
            fmin, fmax = get_dos_frequency_range(
                self._pdos.frequency_points, self._pdos.projected_dos.sum(axis=0)
            )
            ax.set_xlim(fmin, fmax)
        ax.set_ylim((0, None))

        return plt

    def write_partial_DOS(self, filename="partial_dos.dat"):
        """Write projected DOS to text file."""
        warnings.warn(
            "Phonopy.write_partial_DOS() is deprecated. "
            "Use Phonopy.write_projected_dos() (lowercase on DOS).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write_projected_dos(filename=filename)

    def write_projected_dos(self, filename="projected_dos.dat") -> None:
        """Write projected DOS to text file."""
        self._pdos.write(filename=filename)

    # Thermal property
    def run_thermal_properties(
        self,
        t_min=0,
        t_max=1000,
        t_step=10,
        temperatures=None,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
        classical=False,
    ) -> None:
        """Run calculation of thermal properties at constant volume.

        In phonopy, imaginary frequencies are represented as negative real
        value. Under this situation, `cutoff_frequency` is used to ignore
        phonon modes that have frequencies less than `cutoff_frequency`.

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default values are 0, 1000, and 10.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.
        cutoff_frequency : float, optional
            Ignore phonon modes whose frequencies are smaller than this value.
            Default is None, which gives cutoff frequency as zero.
        pretend_real : bool, optional
            Use absolute value of phonon frequency when True. Default is False.
        band_indices : array_like, optional
            Band indices starting with 0. Normally the numbers correspond to
            phonon bands in ascending order of phonon frequencies. Thermal
            properties are calculated only including specified bands.
            Note that use of this results in unphysical values, and it is not
            recommended to use this feature. Default is None.
        is_projection : bool, optional
            When True, fractions of squeared eigenvector elements are
            multiplied to mode thermal property quantities at respective phonon
            modes. Note that use of this results in unphysical values, and it
            is not recommended to use this feature. Default is False.
        classical : bool optional
            If True use classical statistics.
            If False use quantum statistics.

        """
        if self._mesh is None:
            msg = "run_mesh has to be done before run_thermal_properties."
            raise RuntimeError(msg)

        tp = ThermalProperties(
            self._mesh,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            is_projection=is_projection,
            classical=classical,
        )
        if temperatures is None:
            tp.set_temperature_range(t_step=t_step, t_max=t_max, t_min=t_min)
        else:
            tp.temperatures = temperatures
        tp.run()  # lang='C' if not classical else 'py')
        self._thermal_properties = tp

    def set_thermal_properties(
        self,
        t_step=10,
        t_max=1000,
        t_min=0,
        temperatures=None,
        is_projection=False,
        band_indices=None,
        cutoff_frequency=None,
        pretend_real=False,
        classical=False,
    ):
        """Run calculation of thermal properties at constant volume."""
        warnings.warn(
            "Phonopy.set_thermal_properties() is deprecated. "
            "Use Phonopy.run_thermal_properties()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.run_thermal_properties(
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            temperatures=temperatures,
            is_projection=is_projection,
            band_indices=band_indices,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            classical=classical,
        )

    def get_thermal_properties_dict(self) -> dict:
        """Return thermal properties.

        Returns
        -------
        A dictionary of thermal properties with keys of 'temperatures',
        'free_energy', 'entropy', and 'heat_capacity'.
        Each value of corresponding key is as follows:

        temperatures: ndarray
            shape=(temperatures, ), dtype='double'
        free_energy : ndarray
            shape=(temperatures, ), dtype='double'
        entropy : ndarray
            shape=(temperatures, ), dtype='double'
        heat_capacity : ndarray
            shape=(temperatures, ), dtype='double'

        """
        keys = ("temperatures", "free_energy", "entropy", "heat_capacity")
        return dict(zip(keys, self._thermal_properties.thermal_properties))

    def get_thermal_properties(self):
        """Return thermal properties.

        Returns
        -------
        (temperatures, free energy, entropy, heat capacity)

        """
        warnings.warn(
            "Phonopy.get_thermal_properties() is deprecated. "
            "Use Phonopy.get_thermal_properties_dict().",
            DeprecationWarning,
            stacklevel=2,
        )

        tp = self.get_thermal_properties_dict()
        return (
            tp["temperatures"],
            tp["free_energy"],
            tp["entropy"],
            tp["heat_capacity"],
        )

    def plot_thermal_properties(
        self,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        with_grid: bool = True,
        divide_by_Z: bool = False,
        legend_style: Optional[str] = "normal",
    ):
        """Plot thermal properties.

        Parameters
        ----------
        xlabel : str, optional
            Label used for x-axis.
        ylabel : str, optional
            Label used for y-axis.
        with_grid : bool, optional
            With grid or not. Default is True.
        divide_by_Z : bool, optional
            Divide thermal properties by number of formula units of primitive
            cell. Default is False.
        legend_style : str, optional
            "normal", "compact", None. None will not show legend.

        """
        import matplotlib.pyplot as plt

        if self._thermal_properties is None:
            msg = "run_thermal_properties has to be done."
            raise RuntimeError(msg)

        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._thermal_properties.plot(
            ax,
            xlabel=xlabel,
            ylabel=ylabel,
            with_grid=with_grid,
            divide_by_Z=divide_by_Z,
            legend_style=legend_style,
        )

        temps = self._thermal_properties.temperatures
        ax.set_xlim((0, temps[-1]))

        return plt

    def write_yaml_thermal_properties(self, filename="thermal_properties.yaml") -> None:
        """Write thermal properties in yaml format."""
        self._thermal_properties.write_yaml(filename=filename)

    # Thermal displacement
    def run_thermal_displacements(
        self,
        t_min=0,
        t_max=1000,
        t_step=10,
        temperatures=None,
        direction=None,
        freq_min=None,
        freq_max=None,
    ) -> None:
        """Run thermal displacements calculation.

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default valuues are 0, 1000, and 10.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.
        direction : array_like, optional
            Projection direction in reduced coordinates. Default is None,
            i.e., no projection.
            dtype=float, shape=(3,)
        freq_min, freq_max : float, optional
            Phonon frequencies larger than freq_min and smaller than
            freq_max are included. Default is None, i.e., all phonons.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)
        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)
        mesh_nums = self._mesh.mesh_numbers
        ir_grid_points = self._mesh.ir_grid_points
        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be done with with_eigenvectors=True."
            raise RuntimeError(msg)
        if np.prod(mesh_nums) != len(ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        if direction is not None:
            projection_direction = np.dot(direction, self._primitive.cell)
            td = ThermalDisplacements(
                self._mesh,
                projection_direction=projection_direction,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        else:
            td = ThermalDisplacements(self._mesh, freq_min=freq_min, freq_max=freq_max)

        if temperatures is None:
            td.set_temperature_range(t_min, t_max, t_step)
        else:
            td.temperatures = temperatures
        td.run()

        self._thermal_displacements = td

    def set_thermal_displacements(
        self,
        t_step=10,
        t_max=1000,
        t_min=0,
        temperatures=None,
        direction=None,
        freq_min=None,
        freq_max=None,
    ):
        """Run thermal displacements calculation."""
        warnings.warn(
            "Phonopy.set_thermal_displacements() is deprecated. "
            "Use Phonopy.run_thermal_displacements()",
            DeprecationWarning,
            stacklevel=2,
        )
        self.run_thermal_displacements(
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            direction=direction,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def get_thermal_displacements_dict(self) -> dict:
        """Return thermal displacements."""
        if self._thermal_displacements is None:
            msg = "run_thermal_displacements has to be done."
            raise RuntimeError(msg)

        td = self._thermal_displacements
        return {
            "temperatures": td.temperatures,
            "thermal_displacements": td.thermal_displacements,
        }

    def get_thermal_displacements(self):
        """Return thermal displacements."""
        warnings.warn(
            "Phonopy.get_thermal_displacements() is deprecated. "
            "Use Phonopy.get_thermal_displacements_dict()",
            DeprecationWarning,
            stacklevel=2,
        )
        td = self.get_thermal_displacements_dict()
        return (td["temperatures"], td["thermal_displacements"])

    def plot_thermal_displacements(self, is_legend=False):
        """Plot thermal displacements."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

        self._thermal_displacements.plot(plt, is_legend=is_legend)

        temps, _ = self._thermal_displacements.get_thermal_displacements()
        ax.set_xlim((0, temps[-1]))

        return plt

    def write_yaml_thermal_displacements(self) -> None:
        """Write thermal displacements in yaml format."""
        self._thermal_displacements.write_yaml()

    # Thermal displacement matrix
    def run_thermal_displacement_matrices(
        self,
        t_min=0,
        t_max=1000,
        t_step=10,
        temperatures=None,
        freq_min=None,
        freq_max=None,
    ) -> None:
        """Run thermal displacement matrices calculation.

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default valuues are 0, 1000, and 10.
        freq_min, freq_max : float, optional
            Phonon frequencies larger than freq_min and smaller than
            freq_max are included. Default is None, i.e., all phonons.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.
            Default is None.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)
        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)
        mesh_nums = self._mesh.mesh_numbers
        ir_grid_points = self._mesh.ir_grid_points
        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be done with with_eigenvectors=True."
            raise RuntimeError(msg)
        if np.prod(mesh_nums) != len(ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        tdm = ThermalDisplacementMatrices(
            self._mesh,
            freq_min=freq_min,
            freq_max=freq_max,
            lattice=self._primitive.cell.T,
        )

        if temperatures is None:
            tdm.set_temperature_range(t_min, t_max, t_step)
        else:
            tdm.temperatures = temperatures
        tdm.run()

        self._thermal_displacement_matrices = tdm

    def set_thermal_displacement_matrices(
        self, t_step=10, t_max=1000, t_min=0, freq_min=None, freq_max=None, t_cif=None
    ):
        """Run thermal displacement matrices calculation."""
        warnings.warn(
            "Phonopy.set_thermal_displacement_matrices() is "
            "deprecated. Use Phonopy.run_thermal_displacements()",
            DeprecationWarning,
            stacklevel=2,
        )
        if t_cif is None:
            temperatures = None
        else:
            temperatures = [
                t_cif,
            ]
        self.run_thermal_displacement_matrices(
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            temperatures=temperatures,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def get_thermal_displacement_matrices_dict(self) -> dict:
        """Return thermal displacement matrices."""
        if self._thermal_displacement_matrices is None:
            msg = "run_thermal_displacement_matrices has to be done."
            raise RuntimeError(msg)

        tdm = self._thermal_displacement_matrices
        return {
            "temperatures": tdm.temperatures,
            "thermal_displacement_matrices": tdm.thermal_displacement_matrices,
            "thermal_displacement_matrices_cif": tdm.thermal_displacement_matrices_cif,
        }

    def get_thermal_displacement_matrices(self):
        """Return thermal displacement matrices."""
        warnings.warn(
            "Phonopy.get_thermal_displacement_matrices() is "
            "deprecated. Use "
            "Phonopy.get_thermal_displacement_matrices_dict()",
            DeprecationWarning,
            stacklevel=2,
        )
        tdm = self.get_thermal_displacement_matrices_dict()
        return (tdm["temperatures"], tdm["thermal_displacement_matrices"])

    def write_yaml_thermal_displacement_matrices(self) -> None:
        """Write thermal displacement matrices in yaml format."""
        self._thermal_displacement_matrices.write_yaml()

    def write_thermal_displacement_matrix_to_cif(self, temperature_index) -> None:
        """Write thermal displacement matrices at a termperature in cif."""
        self._thermal_displacement_matrices.write_cif(
            self._primitive, temperature_index
        )

    def write_animation(
        self,
        q_point=None,
        anime_type="v_sim",
        band_index=None,
        amplitude=None,
        num_div=None,
        shift=None,
        filename=None,
    ) -> str:
        """Write atomic modulations in animation format.

        Returns
        -------
        str
            Output filename.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if anime_type in ("arc", "xyz", "jmol", "poscar"):
            if band_index is None or amplitude is None or num_div is None:
                msg = "Parameters are not correctly set for animation."
                raise RuntimeError(msg)

        return write_animation(
            self._dynamical_matrix,
            q_point=q_point,
            anime_type=anime_type,
            band_index=band_index,
            amplitude=amplitude,
            num_div=num_div,
            shift=shift,
            factor=self._factor,
            filename=filename,
        )

    # Atomic modulation of normal mode
    def set_modulations(
        self,
        dimension,
        phonon_modes,
        delta_q=None,
        derivative_order=None,
        nac_q_direction=None,
    ):
        """Generate atomic displacements of phonon modes."""
        warnings.warn(
            "Phonopy.set_modulation() is deprecated. " "Use Phonopy.run_modulation().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.run_modulations(
            dimension,
            phonon_modes,
            delta_q=delta_q,
            derivative_order=derivative_order,
            nac_q_direction=nac_q_direction,
        )

    def run_modulations(
        self,
        dimension,
        phonon_modes,
        delta_q=None,
        derivative_order=None,
        nac_q_direction=None,
    ) -> None:
        """Generate atomic displacements of phonon modes.

        The design of this feature is not very satisfactory, and thus API.
        Therefore it should be reconsidered someday in the fugure.

        Parameters
        ----------
        dimension : array_like
            Supercell dimension with respect to the primitive cell.
            dtype='intc', shape=(3, ), (3, 3), (9, )
        phonon_modes : list of phonon mode settings
            Each element of the outer list gives one phonon mode information:

                [q-point, band index (int), amplitude (float), phase (float)]

            In each list of the phonon mode information, the first element is
            a list that represents q-point in reduced coordinates. The second,
            third, and fourth elements show the band index starting with 0,
            amplitude, and phase factor, respectively.
        nac_q_direction : array_like
            q-point direction from Gamma-point in fractional coordinates of
            reciprocal basis vectors. Only the direction is used, i.e.,
            (q_direction / |q_direction|) is computed and used. This parameter
            is activated only at q=(0, 0, 0).
            shape=(3,), dtype='double'

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._modulation = Modulation(
            self._dynamical_matrix,
            dimension,
            phonon_modes,
            delta_q=delta_q,
            derivative_order=derivative_order,
            nac_q_direction=nac_q_direction,
            factor=self._factor,
        )
        self._modulation.run()

    def get_modulated_supercells(self) -> list[PhonopyAtoms]:
        """Return cells with atom modulations.

        list of PhonopyAtoms
            Modulated structures.

        """
        return self._modulation.get_modulated_supercells()

    def get_modulations_and_supercell(self) -> tuple[np.ndarray, PhonopyAtoms]:
        """Return atomic modulations and perfect supercell.

        (modulations, supercell)

        modulations: Atomic modulations of supercell in Cartesian coordinates
        supercell: Supercell as an PhonopyAtoms instance.

        """
        return self._modulation.get_modulations_and_supercell()

    def write_modulations(self, calculator=None, optional_structure_info=None) -> None:
        """Write modulated structures to MPOSCAR's."""
        self._modulation.write(
            interface_mode=calculator,
            optional_structure_info=optional_structure_info,
        )

    def write_yaml_modulations(self) -> None:
        """Write atomic modulations in yaml format."""
        self._modulation.write_yaml()

    # Irreducible representation
    def set_irreps(
        self,
        q,
        is_little_cogroup=False,
        nac_q_direction=None,
        degeneracy_tolerance=1e-4,
    ) -> None:
        """Identify ir-reps of phonon modes.

        The design of this API is not very satisfactory and is expceted
        to be redesined in the next major versions once the use case
        of the API for ir-reps feature becomes clearer.

        nac_q_direction : array_like
            q-point direction from Gamma-point in fractional coordinates of
            reciprocal basis vectors. Only the direction is used, i.e.,
            (q_direction / |q_direction|) is computed and used. This parameter
            is activated only at q=(0, 0, 0).
            shape=(3,), dtype='double'

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._irreps = IrReps(
            self._dynamical_matrix,
            q,
            is_little_cogroup=is_little_cogroup,
            nac_q_direction=nac_q_direction,
            factor=self._factor,
            symprec=self._symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=self._log_level,
        )

        return self._irreps.run()

    def get_irreps(self):
        """Return Ir-reps."""
        warnings.warn(
            "Phonopy.get_irreps() is deprecated. " "Use Phonopy.irreps attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._irreps

    def show_irreps(self, show_irreps=False) -> None:
        """Show Ir-reps."""
        self._irreps.show(show_irreps=show_irreps)

    def write_yaml_irreps(self, show_irreps=False) -> None:
        """Write Ir-reps in yaml format."""
        self._irreps.write_yaml(show_irreps=show_irreps)

    # Group velocity
    def set_group_velocity(self, q_length=None):
        """Prepare group velocity calculation."""
        warnings.warn(
            "Phonopy.set_group_velocity() is deprecated. "
            "No need to call this. gv_delta_q "
            "(q_length) is set at Phonopy.__init__().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._gv_delta_q = q_length
        self._set_group_velocity()

    def get_group_velocity(self):
        """Return group velocities."""
        warnings.warn(
            "Phonopy.get_group_velocities_on_bands is deprecated. "
            "Use Phonopy.[mode].group_velocities attribute or "
            "Phonopy.get_[mode]_dict()[group_velocities], where "
            "[mode] is band_structure, mesh, or qpoints.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._group_velocity.get_group_velocity()

    def get_group_velocity_at_q(self, q_point) -> np.ndarray:
        """Return group velocity at a q-point."""
        if self._group_velocity is None:
            self._set_group_velocity()
        self._group_velocity.run([q_point])
        return self._group_velocity.group_velocities[0]

    def get_group_velocities_on_bands(self):
        """Return group velocities calculated on band structure."""
        warnings.warn(
            "Phonopy.get_group_velocities_on_bands is deprecated. "
            "Use Phonopy.get_band_structure_dict()['group_velocities'].",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._band_structure.group_velocities

    # Moment
    def run_moment(
        self, order=1, is_projection=False, freq_min=None, freq_max=None
    ) -> None:
        """Run moment calculation."""
        if self._mesh is None:
            msg = "run_mesh has to be done before run_moment."
            raise RuntimeError(msg)
        else:
            if is_projection:
                if self._mesh.eigenvectors is None:
                    return RuntimeError(
                        "run_mesh has to be done with with_eigenvectors=True."
                    )
                self._moment = PhononMoment(
                    self._mesh.frequencies,
                    weights=self._mesh.weights,
                    eigenvectors=self._mesh.eigenvectors,
                )
            else:
                self._moment = PhononMoment(
                    self._mesh.get_frequencies(), weights=self._mesh.get_weights()
                )
            if freq_min is not None or freq_max is not None:
                self._moment.set_frequency_range(freq_min=freq_min, freq_max=freq_max)
            self._moment.run(order=order)

    def set_moment(self, order=1, is_projection=False, freq_min=None, freq_max=None):
        """Run moment calculation."""
        warnings.warn(
            "Phonopy.set_moment() is deprecated. " "Use Phonopy.run_moment().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.run_moment(
            order=order,
            is_projection=is_projection,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def get_moment(self) -> Optional[float]:
        """Return moment."""
        return self._moment.moment

    def init_dynamic_structure_factor(
        self,
        Qpoints,
        T,
        atomic_form_factor_func=None,
        scattering_lengths=None,
        freq_min=None,
        freq_max=None,
    ) -> None:
        """Initialize dynamic structure factor calculation.

        *******************************************************************
         This is still an experimental feature. API can be changed without
         notification.
        *******************************************************************

        Need to call DynamicStructureFactor.run() to start calculation.

        Parameters
        ----------
        Qpoints: array_like
            Q-points in any Brillouin zone.
            dtype='double'
            shape=(qpoints, 3)
        T: float
            Temperature in K.
        atomic_form_factor_func: Function object
            Function that returns atomic form factor (``func`` below):

                f_params = {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                                   0.767888, 0.070139, 0.995612, 14.1226457,
                                   0.968249, 0.217037, 0.045300],
                            'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                                   6.524271, 19.467656, 2.355626, 60.320301,
                                   35.829404, 0.000436, -34.916604],b|

                def get_func_AFF(f_params):
                    def func(symbol, Q):
                        return atomic_form_factor_WK1995(Q, f_params[symbol])
                    return func

        scattering_lengths: dictionary
            Coherent scattering lengths averaged over isotopes and spins.
            Supposed for INS. For example, {'Na': 3.63, 'Cl': 9.5770}.
        freq_min, freq_min: float
            Minimum and maximum phonon frequencies to determine whether
            phonons are included in the calculation.

        """
        if self._mesh is None:
            msg = (
                "run_mesh has to be done before initializing dynamic"
                "structure factor."
            )
            raise RuntimeError(msg)

        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be called with with_eigenvectors=True."
            raise RuntimeError(msg)

        if np.prod(self._mesh.mesh_numbers) != len(self._mesh.ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        self._dynamic_structure_factor = DynamicStructureFactor(
            self._mesh,
            Qpoints,
            T,
            atomic_form_factor_func=atomic_form_factor_func,
            scattering_lengths=scattering_lengths,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def run_dynamic_structure_factor(
        self,
        Qpoints,
        T,
        atomic_form_factor_func=None,
        scattering_lengths=None,
        freq_min=None,
        freq_max=None,
    ) -> None:
        """Run dynamic structure factor calculation.

        See the detail of parameters at
        Phonopy.init_dynamic_structure_factor().

        """
        self.init_dynamic_structure_factor(
            Qpoints,
            T,
            atomic_form_factor_func=atomic_form_factor_func,
            scattering_lengths=scattering_lengths,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        self._dynamic_structure_factor.run()

    def set_dynamic_structure_factor(
        self,
        Qpoints,
        T,
        atomic_form_factor_func=None,
        scattering_lengths=None,
        freq_min=None,
        freq_max=None,
        run_immediately=True,
    ):
        """Run dynamic structure factor calculation."""
        warnings.warn(
            "Phonopy.set_dynamic_structure_factor() is deprecated. "
            "Use Phonopy.run_dynamic_structure_factor()",
            DeprecationWarning,
            stacklevel=2,
        )
        if run_immediately:
            self.run_dynamic_structure_factor(
                Qpoints,
                T,
                atomic_form_factor_func=atomic_form_factor_func,
                scattering_lengths=scattering_lengths,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        else:
            self.init_dynamic_structure_factor(
                Qpoints,
                T,
                atomic_form_factor_func=atomic_form_factor_func,
                scattering_lengths=scattering_lengths,
                freq_min=freq_min,
                freq_max=freq_max,
            )

    def get_dynamic_structure_factor(self) -> tuple[np.ndarray, np.ndarray]:
        """Return dynamic structure factors."""
        return (
            self._dynamic_structure_factor.qpoints,
            self._dynamic_structure_factor.dynamic_structure_factors,
        )

    def init_random_displacements(
        self,
        dist_func: Optional[str] = None,
        cutoff_frequency: Optional[float] = None,
        max_distance: Optional[float] = None,
    ) -> None:
        """Initialize random displacements at finite temperature.

        dist_func : str or None, optional
            Harmonic oscillator distribution function either by 'quantum'
            or 'classical'. Default is None, corresponding to 'quantum'.
            Default is None.
        cutoff_frequency : float or None
            Phonon frequency in THz below that phonons are ignored
            to generate random displacements. Default is None.
        max_distance : float or None, optional
            In random displacements generation from canonical ensemble of
            harmonic phonons, displacements larger than max distance are
            renormalized to the max distance, i.e., a disptalcement d is shorten
            by d -> d / |d| * max_distance if |d| > max_distance.

        """
        import phonopy._phonopy as phonoc

        self._random_displacements = RandomDisplacements(
            self._supercell,
            self._primitive,
            self._force_constants,
            dist_func=dist_func,
            cutoff_frequency=cutoff_frequency,
            max_distance=max_distance,
            factor=self._factor,
            use_openmp=phonoc.use_openmp(),
        )

    def get_random_displacements_at_temperature(
        self,
        temperature: float,
        number_of_snapshots: int,
        is_plusminus: bool = False,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate random displacements from phonon structure.

        Some more details are written at generate_displacements.

        temperature : float
            Temperature.
        number_of_snapshots : int
            Number of snapshots with random displacements created.
        random_seed : 32bit unsigned int or None, optional
            Random seed. Default is None

        """
        if self._random_displacements is None:
            raise RuntimeError(
                "Phonopy.init_random_displacements has to be called "
                "before calling this method."
            )
        self._random_displacements.run(
            temperature,
            number_of_snapshots=number_of_snapshots,
            random_seed=random_seed,
        )
        units = get_default_physical_units(self._calculator)
        d = np.array(
            self._random_displacements.u / units["distance_to_A"],
            dtype="double",
            order="C",
        )
        if is_plusminus is True:
            d = np.array(
                np.concatenate((d, -d), axis=0),
                dtype="double",
                order="C",
            )
        return d

    def save(
        self,
        filename="phonopy_params.yaml",
        settings=None,
        hdf5_settings=None,
        compression: Union[str, bool] = False,
    ) -> str:
        """Save phonopy parameters into file.

        Parameters
        ----------
        filename: str, optional
            File name. Default is "phonopy_params.yaml"
        settings: dict, optional
            It is described which parameters are written out. Only the settings
            expected to be updated from the following default settings are
            needed to be set in the dictionary.  The possible parameters and
            their default settings are:
                {'force_sets': True,
                 'displacements': True, 'force_constants': False,
                 'born_effective_charge': True, 'dielectric_constant': True}
            This default settings are updated by {'force_constants': True} when
            dataset is None and force_constants is not None unless
            {'force_constants': False} is specified.
        hdf5_settings: dict, optional (To be implemented)
            Force constants and force_sets are stored in hdf5 file when they are
            activated in the dict. The dict has the following keys. The default
            filename is the filename of yaml file where '.yaml' is replaced by
            '.hdf5'.
                'filename' : str 'force_constants': bool (default=False)
                'force_sets': bool (default=False)
        compression : bool or str
            If True, phonopy_params.yaml like file is compressed by xz. When
            compression=='xz', the file is compressed by xz. Default is False.

        Returns
        -------
        str :
            File name of the saved phonopy_params.yaml like file. If it is
            compressed,

        """
        if hdf5_settings is not None:
            msg = "hdf5_settings parameter has not yet been implemented."
            raise NotImplementedError(msg)

        if settings is None:
            _settings = {}
        else:
            _settings = settings.copy()
        if _settings.get("force_constants") is False:
            pass
        elif not forces_in_dataset(self.dataset) and self.force_constants is not None:
            _settings.update({"force_constants": True})
        phpy_yaml = PhonopyYaml(settings=_settings)
        phpy_yaml.set_phonon_info(self)

        if compression == "xz" or compression is True:
            out_filename = f"{filename}.xz"
            with lzma.open(f"{out_filename}", "wt") as w:
                w.write(str(phpy_yaml))
        else:
            with open(filename, "w") as w:
                out_filename = filename
                w.write(str(phpy_yaml))

        return out_filename

    def ph2ph(self, supercell_matrix, with_nac=False) -> Phonopy:
        """Transform force constants in Phonopy class instance to other shape.

        Fourier interpolation of force constants is performed. This Phonopy
        class instance has to have force constants in it. Returned init
        parameters of this Phonopy class instance are copied to returned Phonopy
        class instance.

        For example, if self._supercell_matrix is [2, 2, 2] and given
        supercell_matrix is [4, 4, 4], the former force constants are Fourier
        interpolated sampling at the commensurate points of the supercell of
        the latter and new Phonopy class instance with the Fourier
        interpolated force constants is returned.

        Parameters
        ----------
        supercell_matrix : array_like
            This specifies array shape of the force constants.
        with_nac : bool, optional
            Non-analytical term correction (NAC) is used under the Fourier
            interpolation, i.e., dynamical matricies at commensurate points
            are computed with NAC, then they are Fourier transform back to
            force constants of supercell_matrix. NAC parameters are not
            copied to returned Phonopy class instance.

        Returns
        -------
        ph : Phonopy
            Phonopy class instance with init parameters of this Phonopy class
            instance and transformed force constants of `supercell_matrix`.

        """
        if self._force_constants is None:
            raise RuntimeError("Force constants are not prepared.")

        import phonopy._phonopy as phonoc

        fc_shape = self._force_constants.shape
        ph_copy = self._copy()
        ph_copy.force_constants = self._force_constants

        if with_nac and self._nac_params is not None:
            ph_copy.nac_params = self._nac_params

        ph = self._copy(supercell_matrix)
        assert isclose(ph.primitive, ph_copy.primitive)
        d2f = DynmatToForceConstants(
            ph.primitive,
            ph.supercell,
            is_full_fc=(fc_shape[0] == fc_shape[1]),
            use_openmp=phonoc.use_openmp(),
        )
        ph_copy.run_qpoints(d2f.commensurate_points, with_dynamical_matrices=True)
        ph_dict = ph_copy.get_qpoints_dict()
        d2f.dynamical_matrices = ph_dict["dynamical_matrices"]
        d2f.run()
        ph.force_constants = d2f.force_constants

        return ph

    def copy(self, log_level=None) -> Phonopy:
        """Copy this Phonopy class instance with init parameters.

        Note
        ----
        Phonopy class instance with the initial parameters is returned, but
        internal variables such as force constants, NAC params, MLP parameters, etc,
        are not stored.

        Returns
        -------
        ph : Phonopy
            Copied phonopy class instace.

        """
        return self._copy(log_level=log_level)

    ###################
    # private methods #
    ###################
    def _copy(self, supercell_matrix=None, log_level=None) -> Phonopy:
        """Copy this Phonopy class instance with init parameters.

        Parameters
        ----------
        supercell_matrix : array_like or None, optional
            Supercell matrix can be specified. None gives the same supercell
            matrix as this Phonopy class instance.

        Returns
        -------
        ph : Phonopy
            Copied phonopy class instace.

        """
        if supercell_matrix is None:
            smat = self._supercell_matrix
        else:
            smat = supercell_matrix
        if log_level is not None:
            _log_level = log_level
        else:
            _log_level = self._log_level
        return Phonopy(
            self._unitcell,
            supercell_matrix=smat,
            primitive_matrix=self._primitive_matrix,
            factor=self._factor,
            frequency_scale_factor=self._frequency_scale_factor,
            dynamical_matrix_decimals=self._dynamical_matrix_decimals,
            force_constants_decimals=self._force_constants_decimals,
            group_velocity_delta_q=self._gv_delta_q,
            symprec=self._symprec,
            is_symmetry=self._is_symmetry,
            store_dense_svecs=self._store_dense_svecs,
            use_SNF_supercell=self._use_SNF_supercell,
            calculator=self._calculator,
            log_level=_log_level,
        )

    def _run_force_constants_from_forces(
        self,
        is_compact_fc: bool = False,
        fc_calculator: Optional[str] = None,
        fc_calculator_options: Optional[str] = None,
        decimals: Optional[int] = None,
        log_level: int = 0,
    ) -> None:
        if self._dataset is not None:
            self._force_constants = get_fc2(
                self._supercell,
                self._dataset,
                primitive=self._primitive,
                fc_calculator=fc_calculator,
                fc_calculator_options=fc_calculator_options,
                is_compact_fc=is_compact_fc,
                symmetry=self._symmetry,
                log_level=log_level,
            )
            if decimals:
                self._force_constants = self._force_constants.round(decimals=decimals)

    def _set_dynamical_matrix(self) -> None:
        import phonopy._phonopy as phonoc

        self._dynamical_matrix = None

        if self._is_symmetry and self._nac_params is not None:
            if len(self._nac_params["born"]) != len(self._primitive):
                raise ValueError(
                    "Numbers of atoms in primitive cell and Born effective charges "
                    "are different."
                )
            borns, epsilon = symmetrize_borns_and_epsilon(
                self._nac_params["born"],
                self._nac_params["dielectric"],
                self._primitive,
                symprec=self._symprec,
            )
            nac_params = self._nac_params.copy()
            nac_params.update({"born": borns, "dielectric": epsilon})
        else:
            nac_params = self._nac_params

        if self._supercell is None or self._primitive is None:
            raise RuntimeError("Supercell or primitive is not created.")
        if self._force_constants is None:
            raise RuntimeError("Force constants are not prepared.")
        if self._primitive.masses is None:
            raise RuntimeError("Atomic masses are not correctly set.")
        self._dynamical_matrix = get_dynamical_matrix(
            self._force_constants,
            self._supercell,
            self._primitive,
            nac_params,
            self._frequency_scale_factor,
            self._dynamical_matrix_decimals,
            log_level=self._log_level,
            use_openmp=phonoc.use_openmp(),
        )
        # DynamialMatrix instance transforms force constants in correct
        # type of numpy array.
        self._force_constants = self._dynamical_matrix.force_constants

        if self._group_velocity is not None:
            self._set_group_velocity()

    def _set_group_velocity(self) -> None:
        if self._dynamical_matrix is None:
            raise RuntimeError("Dynamical matrix has not yet built.")

        if (
            isinstance(self._dynamical_matrix, DynamicalMatrixGL)
            and self._gv_delta_q is None
        ):
            if self._log_level:
                msg = "Group velocity calculation:\n"
                text = (
                    "Analytical derivative of dynamical matrix is not "
                    "implemented for NAC by Gonze et al. Instead "
                    "numerical derivative of it is used with dq=%.1e "
                    "for group velocity calculation." % GroupVelocity.Default_q_length
                )
                msg += textwrap.fill(
                    text, initial_indent="  ", subsequent_indent="  ", width=70
                )
                print(msg)

        self._group_velocity = GroupVelocity(
            self._dynamical_matrix,
            q_length=self._gv_delta_q,
            symmetry=self._primitive_symmetry,
            frequency_factor_to_THz=self._factor,
        )

    def _search_symmetry(self) -> None:
        self._symmetry = Symmetry(
            self._supercell,
            self._symprec,
            self._is_symmetry,
            s2p_map=self._primitive.s2p_map,
        )

    def _search_primitive_symmetry(self) -> None:
        self._primitive_symmetry = Symmetry(
            self._primitive, self._symprec, self._is_symmetry
        )

        if len(self._symmetry.pointgroup_operations) != len(
            self._primitive_symmetry.pointgroup_operations
        ):
            print(
                "Warning: Point group symmetries of supercell and primitive"
                "cell are different."
            )

    def _build_supercell(self) -> None:
        self._supercell = get_supercell(
            self._unitcell,
            self._supercell_matrix,
            is_old_style=(not self._use_SNF_supercell),
            symprec=self._symprec,
        )

    def _build_supercells_with_displacements(self) -> None:
        all_positions = []
        if "first_atoms" in self._dataset:  # type-1
            for disp in self._dataset["first_atoms"]:
                positions = self._supercell.positions
                positions[disp["number"]] += disp["displacement"]
                all_positions.append(positions)
        elif "displacements" in self._dataset:
            for disp in self._dataset["displacements"]:
                all_positions.append(self._supercell.positions + disp)
        else:
            raise RuntimeError("displacement_dataset is not set.")

        supercells = []
        for positions in all_positions:
            supercells.append(
                PhonopyAtoms(
                    symbols=self._supercell.symbols,
                    masses=self._supercell.masses,
                    magnetic_moments=self._supercell.magnetic_moments,
                    positions=positions,
                    cell=self._supercell.cell,
                )
            )
        self._supercells_with_displacements = supercells

    def _build_primitive_cell(self) -> None:
        """Create primitive cell.

        primitive_matrix:
          Relative axes of primitive cell to the input unit cell.
          Relative axes to the supercell is calculated by:
             supercell_matrix^-1 * primitive_matrix
          Therefore primitive cell lattice is finally calculated by:
             (supercell_lattice * (supercell_matrix)^-1 * primitive_matrix)^T

        """
        inv_supercell_matrix = np.linalg.inv(self._supercell_matrix)
        if self._primitive_matrix is None:
            trans_mat = inv_supercell_matrix
        else:
            trans_mat = np.dot(inv_supercell_matrix, self._primitive_matrix)

        try:
            self._primitive = get_primitive(
                self._supercell,
                trans_mat,
                self._symprec,
                store_dense_svecs=self._store_dense_svecs,
            )
        except ValueError as exc:
            msg = (
                "Creating primitive cell is failed. "
                "PRIMITIVE_AXIS may be incorrectly specified."
            )
            raise RuntimeError(msg) from exc

    def _set_primitive_matrix(
        self, primitive_matrix
    ) -> Optional[Union[str, np.ndarray]]:
        pmat = get_primitive_matrix(primitive_matrix, symprec=self._symprec)
        if isinstance(pmat, str) and pmat == "auto":
            return guess_primitive_matrix(self._unitcell, symprec=self._symprec)
        else:
            return pmat

    def _shape_supercell_matrix(self, smat) -> np.ndarray:
        return shape_supercell_matrix(smat)

    def _get_forces_energies(
        self, target: Literal["forces", "supercell_energies"]
    ) -> Optional[list]:
        """Return forces and supercell energies.

        Return None if tagert data is not found.

        """
        if self._dataset is None:
            return None
        if target in self._dataset:  # type-2
            return self._dataset[target]
        if "first_atoms" in self._dataset:  # type-1
            values = []
            for disp in self._dataset["first_atoms"]:
                if target == "forces":
                    if target in disp:
                        values.append(disp[target])
                elif target == "supercell_energies":
                    if "supercell_energy" in disp:
                        values.append(disp["supercell_energy"])
            if values:
                return np.array(values, dtype="double", order="C")
        return None

    def _set_forces_energies(
        self, values, target: Literal["forces", "supercell_energies"]
    ):
        if "first_atoms" in self._dataset:  # type-1
            for disp, v in zip(self._dataset["first_atoms"], values):
                if target == "forces":
                    disp[target] = np.array(v, dtype="double", order="C")
                elif target == "supercell_energies":
                    disp["supercell_energy"] = float(v)
        elif "displacements" in self._dataset:  # type-2
            _values = np.array(values, dtype="double", order="C")
            natom = len(self._supercell)
            ndisps = len(self._dataset["displacements"])
            if target == "forces" and (
                _values.ndim != 3 or _values.shape != (ndisps, natom, 3)
            ):
                raise RuntimeError(f"Array shape of input {target} is incorrect.")
            elif target == "supercell_energies":
                if _values.ndim != 1 or _values.shape != (ndisps,):
                    raise RuntimeError(f"Array shape of input {target} is incorrect.")
            self._dataset[target] = _values
        else:
            raise RuntimeError("Set of displacements is not available.")
