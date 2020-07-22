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

import sys
import warnings
import textwrap
import numpy as np
from phonopy.version import __version__
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry, symmetrize_borns_and_epsilon
from phonopy.structure.grid_points import length2mesh
from phonopy.structure.cells import (
    get_supercell, get_primitive, guess_primitive_matrix,
    shape_supercell_matrix)
from phonopy.structure.dataset import (
    get_displacements_and_forces, forces_in_dataset)
from phonopy.harmonic.displacement import (
    get_least_displacements, directions_to_displacement_dataset,
    get_random_displacements_dataset)
from phonopy.harmonic.force_constants import (
    symmetrize_force_constants, symmetrize_compact_force_constants,
    show_drift_force_constants, cutoff_force_constants,
    set_tensor_symmetry_PJ)
from phonopy.harmonic.force_constants import get_fc2 as get_phonopy_fc2
from phonopy.interface.calculator import get_default_physical_units
from phonopy.interface.fc_calculator import get_fc2
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.phonon.band_structure import (
    BandStructure, get_band_qpoints_by_seekpath)
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.phonon.mesh import Mesh, IterMesh
from phonopy.units import VaspToTHz
from phonopy.phonon.dos import TotalDos, PartialDos
from phonopy.phonon.thermal_displacement import (
    ThermalDisplacements, ThermalDisplacementMatrices)
from phonopy.phonon.random_displacements import RandomDisplacements
from phonopy.phonon.animation import Animation
from phonopy.phonon.modulation import Modulation
from phonopy.phonon.qpoints import QpointsPhonon
from phonopy.phonon.irreps import IrReps
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.moment import PhononMoment
from phonopy.spectrum.dynamic_structure_factor import DynamicStructureFactor

# Uncomment below to watch DeprecationWarning,
# warnings.simplefilter("always")


class Phonopy(object):
    """Phonopy class"""

    def __init__(self,
                 unitcell,
                 supercell_matrix=None,
                 primitive_matrix=None,
                 nac_params=None,
                 factor=VaspToTHz,
                 frequency_scale_factor=None,
                 dynamical_matrix_decimals=None,
                 force_constants_decimals=None,
                 group_velocity_delta_q=None,
                 symprec=1e-5,
                 is_symmetry=True,
                 calculator=None,
                 use_lapack_solver=False,
                 log_level=0):
        self._symprec = symprec
        self._factor = factor
        self._frequency_scale_factor = frequency_scale_factor
        self._is_symmetry = is_symmetry
        self._calculator = calculator
        self._use_lapack_solver = use_lapack_solver
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = PhonopyAtoms(atoms=unitcell)
        self._supercell_matrix = self._shape_supercell_matrix(supercell_matrix)
        if type(primitive_matrix) is str and primitive_matrix == 'auto':
            self._primitive_matrix = self._guess_primitive_matrix()
        elif primitive_matrix is not None:
            self._primitive_matrix = np.array(primitive_matrix,
                                              dtype='double', order='c')
        else:
            self._primitive_matrix = None
        self._supercell = None
        self._primitive = None
        self._build_supercell()
        self._build_primitive_cell()

        # Set supercell and primitive symmetry
        self._symmetry = None
        self._primitive_symmetry = None
        self._search_symmetry()
        self._search_primitive_symmetry()

        # displacements
        self._displacement_dataset = {'natom':
                                      self._supercell.get_number_of_atoms()}
        self._supercells_with_displacements = None

        # set_force_constants or set_forces
        self._force_constants = None
        self._force_constants_decimals = force_constants_decimals

        # set_dynamical_matrix
        self._dynamical_matrix = None
        self._nac_params = nac_params
        self._dynamical_matrix_decimals = dynamical_matrix_decimals

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
    def version(self):
        """Phonopy release version number

        str
            Phonopy release version number

        """

        return __version__

    def get_version(self):
        return self.version

    @property
    def primitive(self):
        """Primitive cell

        Primitive
            Primitive cell.

        """

        return self._primitive

    def get_primitive(self):
        return self.primitive

    @property
    def unitcell(self):
        """Unit cell

        PhonopyAtoms
            Unit cell.

        """

        return self._unitcell

    def get_unitcell(self):
        return self.unitcell

    @property
    def supercell(self):
        """Supercell

        Supercell
            Supercell.

        """

        return self._supercell

    def get_supercell(self):
        return self.supercell

    @property
    def symmetry(self):
        """Symmetry of supercell

        Symmetry
            Symmetry of supercell.

        """

        return self._symmetry

    def get_symmetry(self):
        return self.symmetry

    @property
    def primitive_symmetry(self):
        """Symmetry of primitive cell

        Symmetry
            Symmetry of primitive cell.

        """

        return self._primitive_symmetry

    def get_primitive_symmetry(self):
        return self.primitive_symmetry

    @property
    def supercell_matrix(self):
        """Transformation matrix to supercell cell from unit cell

        ndarray
            Supercell matrix with respect to unit cell.
            shape=(3, 3), dtype='intc', order='C'

        """

        return self._supercell_matrix

    def get_supercell_matrix(self):
        return self.supercell_matrix

    @property
    def primitive_matrix(self):
        """Transformation matrix to primitive cell from unit cell

        ndarray
            Primitive matrix with respect to unit cell.
            shape=(3, 3), dtype='double', order='C'

        """

        return self._primitive_matrix

    def get_primitive_matrix(self):
        return self.primitive_matrix

    @property
    def unit_conversion_factor(self):
        """Phonon frequency unit conversion factor.

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
        return self.unit_conversion_factor

    @property
    def calculator(self):
        """Calculator name

        str
            Calculator name such as 'vasp', 'qe', etc.

        """
        return self._calculator

    @property
    def dataset(self):
        """Dataset to store displacements and forces

        Dataset containing information of displacements in supercells.
        This optionally contains forces of respective supercells.

        dataset : dict
            The format can be either one of two types

            Type 1. One atomic displacement in each supercell:
                {'natom': number of atoms in supercell,
                 'first_atoms': [
                   {'number': atom index of displaced atom,
                    'displacement': displacement in Cartesian coordinates,
                    'forces': forces on atoms in supercell},
                   {...}, ...]}
            Elements of the list accessed by 'first_atoms' corresponds to each
            displaced supercell. Each displaced supercell contains only one
            displacement. dict['first_atoms']['forces'] gives atomic forces in
            each displaced supercell.

            Type 2. All atomic displacements in each supercell:
                {'displacements': ndarray, dtype='double', order='C',
                                  shape=(supercells, natom, 3)
                 'forces': ndarray, dtype='double',, order='C',
                                  shape=(supercells, natom, 3)}

            To set in type 2, displacements and forces can be given by numpy
            array with different shape but that can be reshaped to
            (supercells, natom, 3).

        """

        return self._displacement_dataset

    @property
    def displacement_dataset(self):
        warnings.warn("Phonopy.displacement_dataset is deprecated."
                      "Use Phonopy.dataset.",
                      DeprecationWarning)
        return self.dataset

    def get_displacement_dataset(self):
        return self.dataset

    @property
    def displacements(self):
        """Displacements in supercells

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


        To set displacements set, only type-2 datast case is allowed.

        displacemens : array_like
            Atomic displacements of all atoms of all supercells.
            Only all displacements in each supercell case (type-2) is
            supported.
            shape=(supercells, natom, 3), dtype='double', order='C'

        """

        disps = []
        if 'first_atoms' in self._displacement_dataset:
            for disp in self._displacement_dataset['first_atoms']:
                x = disp['displacement']
                disps.append([disp['number'], x[0], x[1], x[2]])
        elif 'displacements' in self._displacement_dataset:
            disps = self._displacement_dataset['displacements']

        return disps

    def get_displacements(self):
        return self.displacements

    @displacements.setter
    def displacements(self, displacements):
        disp = np.array(displacements, dtype='double', order='C')
        if (disp.ndim != 3 or
            disp.shape[1:] != (self._supercell.get_number_of_atoms(), 3)):
            raise RuntimeError("Array shape of displacements is incorrect.")

        if 'first_atoms' in self._displacement_dataset:
            raise RuntimeError("This displacement format is not supported.")

        self._displacement_dataset['displacements'] = disp

    @property
    def force_constants(self):
        """Supercell force constants

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

    def get_force_constants(self):
        return self.force_constants

    @property
    def forces(self):
        """Set of forces of supercells

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

        if 'forces' in self._displacement_dataset:
            return self._displacement_dataset['forces']
        elif 'first_atoms' in self._displacement_dataset:
            forces = []
            for disp in self._displacement_dataset['first_atoms']:
                if 'forces' in disp:
                    forces.append(disp['forces'])
            if forces:
                return np.array(forces, dtype='double', order='C')
            else:
                None
        else:
            return None

    @property
    def dynamical_matrix(self):
        """DynamicalMatrix instance

        This is not dynamical matrices but the instance of DynamicalMatrix
        class.

        """

        return self._dynamical_matrix

    def get_dynamical_matrix(self):
        return self.dynamical_matrix

    @property
    def nac_params(self):
        """Parameters for non-analytical term correction

        dict
            Parameters used for non-analytical term correction
            'born': ndarray
                Born effective charges
                shape=(primitive cell atoms, 3, 3), dtype='double', order='C'
            'factor': float
                Unit conversion factor
            'dielectric': ndarray
                Dielectric constant tensor
                shape=(3, 3), dtype='double', order='C'

        """
        return self._nac_params

    def get_nac_params(self):
        return self.nac_params

    @property
    def supercells_with_displacements(self):
        """Supercells with displacements

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phonopy.generate_displacements.

        """

        if self._displacement_dataset is None:
            return None
        else:
            if self._supercells_with_displacements is None:
                self._build_supercells_with_displacements()
            return self._supercells_with_displacements

    def get_supercells_with_displacements(self):
        return self.supercells_with_displacements

    @property
    def mesh_numbers(self):
        """Sampling mesh numbers in reciprocal space"""
        if self._mesh is None:
            return None
        else:
            return self._mesh.mesh_numbers

    @property
    def qpoints(self):
        return self._qpoints

    @property
    def band_structure(self):
        return self._band_structure

    @property
    def mesh(self):
        return self._mesh

    @property
    def random_displacements(self):
        return self._random_displacements

    @property
    def dynamic_structure_factor(self):
        return self._dynamic_structure_factor

    @property
    def thermal_properties(self):
        return self._thermal_properties

    @property
    def thermal_displacements(self):
        return self._thermal_displacements

    @property
    def thermal_displacement_matrices(self):
        return self._thermal_displacement_matrices

    @property
    def irreps(self):
        return self._irreps

    @property
    def moment(self):
        return self._moment

    @property
    def total_dos(self):
        return self._total_dos

    @property
    def partial_dos(self):
        warnings.warn("Phonopy.partial_dos is deprecated."
                      "Use Phonopy.projected_dos.",
                      DeprecationWarning)
        return self.projected_dos

    @property
    def projected_dos(self):
        return self._pdos

    def set_unitcell(self, unitcell):
        warnings.warn("Phonopy.set_unitcell is deprecated.",
                      DeprecationWarning)
        self._unitcell = unitcell
        self._build_supercell()
        self._build_primitive_cell()
        self._search_symmetry()
        self._search_primitive_symmetry()
        self._displacement_dataset = None

    @property
    def masses(self):
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
        self.masses = masses

    @nac_params.setter
    def nac_params(self, nac_params):
        self._nac_params = nac_params
        if self._force_constants is not None:
            self._set_dynamical_matrix()

    def set_nac_params(self, nac_params):
        self.nac_params = nac_params

    @dataset.setter
    def dataset(self, dataset):
        if dataset is None:
            self._displacement_dataset = None
        elif 'first_atoms' in dataset:
            self._displacement_dataset = dataset
        elif 'displacements' in dataset:
            self._displacement_dataset = {}
            self.displacements = dataset['displacements']
            if 'forces' in dataset:
                self.forces = dataset['forces']
        else:
            raise RuntimeError("Data format of dataset is wrong.")

        self._supercells_with_displacements = None

    def set_displacement_dataset(self, displacement_dataset):
        self.dataset = displacement_dataset

    @forces.setter
    def forces(self, sets_of_forces):
        if 'first_atoms' in self._displacement_dataset:
            for disp, forces in zip(self._displacement_dataset['first_atoms'],
                                    sets_of_forces):
                disp['forces'] = forces
        elif 'displacements' in self._displacement_dataset:
            forces = np.array(sets_of_forces, dtype='double', order='C')
            natom = self._supercell.get_number_of_atoms()
            if forces.ndim != 3 or forces.shape[1:] != (natom, 3):
                raise RuntimeError("Array shape of input forces is incorrect.")

            self._displacement_dataset['forces'] = forces

    def set_forces(self, sets_of_forces):
        self.forces = sets_of_forces

    @force_constants.setter
    def force_constants(self, force_constants):
        if type(force_constants) is np.ndarray:
            fc_shape = force_constants.shape
            if fc_shape[0] != fc_shape[1]:
                if self._primitive.get_number_of_atoms() != fc_shape[0]:
                    msg = ("Force constants shape disagrees with crystal "
                           "structure setting. This may be due to "
                           "PRIMITIVE_AXIS.")
                    raise RuntimeError(msg)

        self._force_constants = force_constants
        if self._primitive.get_masses() is not None:
            self._set_dynamical_matrix()

    def set_force_constants(self, force_constants, show_drift=True):
        self.force_constants = force_constants
        if show_drift and self._log_level:
            show_drift_force_constants(self._force_constants,
                                       primitive=self._primitive)

    def set_force_constants_zero_with_radius(self, cutoff_radius):
        cutoff_force_constants(self._force_constants,
                               self._supercell,
                               self._primitive,
                               cutoff_radius,
                               symprec=self._symprec)
        if self._primitive.get_masses() is not None:
            self._set_dynamical_matrix()

    def generate_displacements(self,
                               distance=0.01,
                               is_plusminus='auto',
                               is_diagonal=True,
                               is_trigonal=False,
                               number_of_snapshots=None,
                               random_seed=None,
                               temperature=None,
                               cutoff_frequency=None):
        """Generate displacement dataset

        There are two modes, finite difference method with systematic
        displacements and fitting approach between arbitrary displacements and
        their forces. The default approach is the finite difference method that
        is built-in phonopy. The fitting approach requires external force
        constant calculator.

        The random displacement supercells are created by setting positive
        integer values 'number_of_snapshots' keyword argument. Unless
        this is specified, systematic displacements are created for the finite
        difference method as the default behaviour.

        Parameters
        ----------
        distance : float, optional
            Displacement distance. Unit is the same as that used for crystal
            structure. Default is 0.01.
        is_plusminus : 'auto', True, or False, optional
            For each atom, displacement of one direction (False), both
            direction, i.e., one directiona and its opposite direction
            (True), and both direction if symmetry requires ('auto').
            Default is 'auto'.
        is_diagonal : bool, optional
            Displacements are made only along basis vectors (False) and
            can be made not being along basis vectors if the number of
            displacements can be reduced by symmetry (True). Default is
            True.
        is_trigonal : bool, optional
            Existing only testing purpose.
        number_of_snapshots : int or None, optional
            Number of snapshots of supercells with random displacements.
            Random displacements are generated displacing all atoms in
            random directions with a fixed displacement distance specified
            by 'distance' parameter, i.e., all atoms in supercell are
            displaced with the same displacement distance in direct space.
        random_seed : 32bit unsigned int or None, optional
            Random seed for random displacements generation.
        temperature : float
            With given temperature, random displacements at temperature is
            generated by sampling probability distribution from canonical
            ensemble of harmonic oscillators (harmonic phonons).
        cutoff_frequency : float
            In random displacements generation from canonical ensemble
            of harmonic phonons, phonon occupation number is used to
            determine the deviation of the distribution function.
            To avoid too large deviation, this value is used to exclude
            the phonon modes whose absolute frequency are smaller than
            this value.

        """

        if (np.issubdtype(type(number_of_snapshots), np.integer) and
            number_of_snapshots > 0):
            if temperature is None:
                displacement_dataset = get_random_displacements_dataset(
                    number_of_snapshots,
                    distance,
                    self._supercell.get_number_of_atoms(),
                    random_seed=random_seed)
            else:
                self.run_random_displacements(
                    temperature,
                    number_of_snapshots=number_of_snapshots,
                    random_seed=random_seed,
                    cutoff_frequency=cutoff_frequency)
                units = get_default_physical_units(self._calculator)
                d = np.array(
                    self._random_displacements.u / units['distance_to_A'],
                    dtype='double', order='C')
                displacement_dataset = {'displacements': d}
        else:
            displacement_directions = get_least_displacements(
                self._symmetry,
                is_plusminus=is_plusminus,
                is_diagonal=is_diagonal,
                is_trigonal=is_trigonal,
                log_level=self._log_level)
            displacement_dataset = directions_to_displacement_dataset(
                displacement_directions,
                distance,
                self._supercell)
        self.dataset = displacement_dataset

    def produce_force_constants(self,
                                forces=None,
                                calculate_full_force_constants=True,
                                fc_calculator=None,
                                fc_calculator_options=None,
                                show_drift=True):
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

        """

        if forces is not None:
            self.forces = forces

        # A primitive check if 'forces' key is in displacement_dataset.
        if 'first_atoms' in self._displacement_dataset:
            for disp in self._displacement_dataset['first_atoms']:
                if 'forces' not in disp:
                    raise RuntimeError("Forces are not yet set.")
        elif 'forces' not in self._displacement_dataset:
            raise RuntimeError("Forces are not yet set.")

        if calculate_full_force_constants:
            self._run_force_constants_from_forces(
                fc_calculator=fc_calculator,
                fc_calculator_options=fc_calculator_options,
                decimals=self._force_constants_decimals)
        else:
            p2s_map = self._primitive.get_primitive_to_supercell_map()
            self._run_force_constants_from_forces(
                distributed_atom_list=p2s_map,
                fc_calculator=fc_calculator,
                fc_calculator_options=fc_calculator_options,
                decimals=self._force_constants_decimals)

        if show_drift and self._log_level:
            show_drift_force_constants(self._force_constants,
                                       primitive=self._primitive)

        if self._primitive.get_masses() is not None:
            self._set_dynamical_matrix()

    def symmetrize_force_constants(self, level=1, show_drift=True):
        if self._force_constants is None:
            raise RuntimeError("Force constants have not been produced yet.")

        if self._force_constants.shape[0] == self._force_constants.shape[1]:
            symmetrize_force_constants(self._force_constants, level=level)
        else:
            symmetrize_compact_force_constants(self._force_constants,
                                               self._primitive,
                                               level=level)
        if show_drift and self._log_level:
            sys.stdout.write("Max drift after symmetrization by translation: ")
            show_drift_force_constants(self._force_constants,
                                       primitive=self._primitive,
                                       values_only=True)

        if self._primitive.get_masses() is not None:
            self._set_dynamical_matrix()

    def symmetrize_force_constants_by_space_group(self, show_drift=True):
        set_tensor_symmetry_PJ(self._force_constants,
                               self._supercell.cell.T,
                               self._supercell.scaled_positions,
                               self._symmetry)

        if show_drift and self._log_level:
            sys.stdout.write("Max drift after symmetrization by space group: ")
            show_drift_force_constants(self._force_constants,
                                       primitive=self._primitive,
                                       values_only=True)

        if self._primitive.get_masses() is not None:
            self._set_dynamical_matrix()

    #####################
    # Phonon properties #
    #####################

    # Single q-point
    def get_dynamical_matrix_at_q(self, q):
        """Calculate dynamical matrix at a given q-point

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
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        return self._dynamical_matrix.get_dynamical_matrix()

    def get_frequencies(self, q):
        """Calculate phonon frequencies at a given q-point

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
            msg = ("Dynamical matrix has not yet built.")
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

    def get_frequencies_with_eigenvectors(self, q):
        """Calculate phonon frequencies and eigenvectors at a given q-point

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
            msg = ("Dynamical matrix has not yet built.")
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
    def run_band_structure(self,
                           paths,
                           with_eigenvectors=False,
                           with_group_velocities=False,
                           is_band_connection=False,
                           path_connections=None,
                           labels=None,
                           is_legacy_plot=False):
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
            msg = ("Dynamical matrix has not yet built.")
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
            factor=self._factor)

    def set_band_structure(self,
                           bands,
                           is_eigenvectors=False,
                           is_band_connection=False,
                           path_connections=None,
                           labels=None,
                           is_legacy_plot=False):
        warnings.warn("Phonopy.set_band_structure is deprecated. "
                      "Use Phonopy.run_band_structure.", DeprecationWarning)

        if self._group_velocity is None:
            with_group_velocities = False
        else:
            with_group_velocities = True
        self.run_band_structure(bands,
                                with_eigenvectors=is_eigenvectors,
                                with_group_velocities=with_group_velocities,
                                is_band_connection=is_band_connection,
                                path_connections=path_connections,
                                labels=labels,
                                is_legacy_plot=is_legacy_plot)

    def get_band_structure_dict(self):
        """Returns calculated band structures

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
            msg = ("run_band_structure has to be done.")
            raise RuntimeError(msg)

        retdict = {'qpoints': self._band_structure.qpoints,
                   'distances': self._band_structure.distances,
                   'frequencies': self._band_structure.frequencies,
                   'eigenvectors': self._band_structure.eigenvectors,
                   'group_velocities': self._band_structure.group_velocities}

        return retdict

    def get_band_structure(self):
        """Returns calculated band structures

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

        warnings.warn("Phonopy.get_band_structure is deprecated. "
                      "Use Phonopy.get_band_structure_dict.",
                      DeprecationWarning)

        if self._band_structure is None:
            msg = ("run_band_structure has to be done.")
            raise RuntimeError(msg)

        retvals = (self._band_structure.qpoints,
                   self._band_structure.distances,
                   self._band_structure.frequencies,
                   self._band_structure.eigenvectors)
        return retvals

    def auto_band_structure(self,
                            npoints=101,
                            with_eigenvectors=False,
                            with_group_velocities=False,
                            plot=False,
                            write_yaml=False,
                            filename="band.yaml"):
        """Convenient method to calculate/draw band structure

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
            self._primitive, npoints, is_const_interval=True)
        self.run_band_structure(bands,
                                with_eigenvectors=with_eigenvectors,
                                with_group_velocities=with_group_velocities,
                                path_connections=path_connections,
                                labels=labels,
                                is_legacy_plot=False)
        if write_yaml:
            self.write_yaml_band_structure(filename=filename)
        if plot:
            return self.plot_band_structure()

    def plot_band_structure(self):
        import matplotlib.pyplot as plt

        if self._band_structure.labels:
            from matplotlib import rc
            rc('text', usetex=True)

        if self._band_structure.is_legacy_plot:
            fig, axs = plt.subplots(1, 1)
        else:
            from mpl_toolkits.axes_grid1 import ImageGrid
            n = len([x for x in self._band_structure.path_connections
                     if not x])
            fig = plt.figure()
            axs = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(1, n),
                            axes_pad=0.11,
                            label_mode="L")
        self._band_structure.plot(axs)
        return plt

    def write_hdf5_band_structure(self,
                                  comment=None,
                                  filename="band.hdf5"):
        self._band_structure.write_hdf5(comment=comment, filename=filename)

    def write_yaml_band_structure(self,
                                  comment=None,
                                  filename=None,
                                  compression=None):
        """Write band structure in yaml

        Parameters
        ----------
        comment : str
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
        self._band_structure.write_yaml(comment=comment,
                                        filename=filename,
                                        compression=compression)

    def init_mesh(self,
                  mesh=100.0,
                  shift=None,
                  is_time_reversal=True,
                  is_mesh_symmetry=True,
                  with_eigenvectors=False,
                  with_group_velocities=False,
                  is_gamma_center=False,
                  use_iter_mesh=False):
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
                rots = self._primitive_symmetry.get_pointgroup_operations()
                mesh_nums = length2mesh(mesh,
                                        self._primitive.get_cell(),
                                        rotations=rots)
            else:
                mesh_nums = length2mesh(mesh, self._primitive.get_cell())
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
                rotations=self._primitive_symmetry.get_pointgroup_operations(),
                factor=self._factor)
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
                rotations=self._primitive_symmetry.get_pointgroup_operations(),
                factor=self._factor,
                use_lapack_solver=self._use_lapack_solver)

    def run_mesh(self,
                 mesh=100.0,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 with_eigenvectors=False,
                 with_group_velocities=False,
                 is_gamma_center=False):
        """Run mesh sampling phonon calculation.

        See the parameter details in Phonopy.init_mesh.

        """

        self.init_mesh(mesh=mesh,
                       shift=shift,
                       is_time_reversal=is_time_reversal,
                       is_mesh_symmetry=is_mesh_symmetry,
                       with_eigenvectors=with_eigenvectors,
                       with_group_velocities=with_group_velocities,
                       is_gamma_center=is_gamma_center)
        self._mesh.run()

    def set_mesh(self,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 run_immediately=True):
        """Phonon calculations on sampling mesh grids

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

        warnings.warn("Phonopy.set_mesh is deprecated. "
                      "Use Phonopy.run_mesh.", DeprecationWarning)

        if self._group_velocity is None:
            with_group_velocities = False
        else:
            with_group_velocities = True
        if run_immediately:
            self.run_mesh(mesh,
                          shift=shift,
                          is_time_reversal=is_time_reversal,
                          is_mesh_symmetry=is_mesh_symmetry,
                          with_eigenvectors=is_eigenvectors,
                          with_group_velocities=with_group_velocities,
                          is_gamma_center=is_gamma_center)
        else:
            self.init_mesh(mesh,
                           shift=shift,
                           is_time_reversal=is_time_reversal,
                           is_mesh_symmetry=is_mesh_symmetry,
                           with_eigenvectors=is_eigenvectors,
                           with_group_velocities=with_group_velocities,
                           is_gamma_center=is_gamma_center)

    def get_mesh_dict(self):
        """Returns calculated mesh sampling phonons

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
            msg = ("run_mesh has to be done.")
            raise RuntimeError(msg)

        retdict = {'qpoints': self._mesh.qpoints,
                   'weights': self._mesh.weights,
                   'frequencies': self._mesh.frequencies,
                   'eigenvectors': self._mesh.eigenvectors,
                   'group_velocities': self._mesh.group_velocities}

        return retdict

    def get_mesh(self):
        warnings.warn("Phonopy.get_mesh is deprecated. "
                      "Use Phonopy.get_mesh_dict.",
                      DeprecationWarning)

        if self._mesh is None:
            msg = ("run_mesh has to be done.")
            raise RuntimeError(msg)

        return (self._mesh.qpoints,
                self._mesh.weights,
                self._mesh.frequencies,
                self._mesh.eigenvectors)

    def get_mesh_grid_info(self):
        warnings.warn("Phonopy.get_mesh_grid_info is deprecated. "
                      "Use attributes of phonon.mesh instance.",
                      DeprecationWarning)
        if self._mesh is None:
            msg = ("run_mesh has to be done.")
            raise RuntimeError(msg)

        return (self._mesh.grid_address,
                self._mesh.ir_grid_points,
                self._mesh.grid_mapping_table)

    def write_hdf5_mesh(self):
        self._mesh.write_hdf5()

    def write_yaml_mesh(self):
        self._mesh.write_yaml()

    # Sampling mesh:
    # Solving dynamical matrices at q-points one-by-one as an iterator
    def set_iter_mesh(self,
                      mesh,
                      shift=None,
                      is_time_reversal=True,
                      is_mesh_symmetry=True,
                      is_eigenvectors=False,
                      is_gamma_center=False):
        """Create an IterMesh instancer

        Attributes
        ----------
        See set_mesh method.

        """

        warnings.warn("Phonopy.set_iter_mesh is deprecated. "
                      "Use Phonopy.run_mesh with use_iter_mesh=True.",
                      DeprecationWarning)

        self.run_mesh(mesh=mesh,
                      shift=shift,
                      is_time_reversal=is_time_reversal,
                      is_mesh_symmetry=is_mesh_symmetry,
                      with_eigenvectors=is_eigenvectors,
                      is_gamma_center=is_gamma_center,
                      use_iter_mesh=True)

    # Plot band structure and DOS (PDOS) together
    def plot_band_structure_and_dos(self, pdos_indices=None):
        import matplotlib.pyplot as plt
        if self._band_structure.labels:
            from matplotlib import rc
            rc('text', usetex=True)

        if self._band_structure.is_legacy_plot:
            import matplotlib.gridspec as gridspec
            # plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax2 = plt.subplot(gs[0, 1])
            if pdos_indices is None:
                self._total_dos.plot(ax2,
                                     ylabel="",
                                     draw_grid=False,
                                     flip_xy=True)
            else:
                self._pdos.plot(ax2,
                                indices=pdos_indices,
                                ylabel="",
                                draw_grid=False,
                                flip_xy=True)
            ax2.set_xlim((0, None))
            plt.setp(ax2.get_yticklabels(), visible=False)

            ax1 = plt.subplot(gs[0, 0], sharey=ax2)
            self._band_structure.plot(ax1)

            plt.subplots_adjust(wspace=0.03)
            plt.tight_layout()
        else:
            from mpl_toolkits.axes_grid1 import ImageGrid
            n = len([x for x in self._band_structure.path_connections
                     if not x]) + 1
            fig = plt.figure()
            axs = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(1, n),
                            axes_pad=0.11,
                            label_mode="L")
            self._band_structure.plot(axs[:-1])

            if pdos_indices is None:
                self._total_dos.plot(axs[-1],
                                     xlabel="",
                                     ylabel="",
                                     draw_grid=False,
                                     flip_xy=True)
            else:
                self._pdos.plot(axs[-1],
                                indices=pdos_indices,
                                xlabel="",
                                ylabel="",
                                draw_grid=False,
                                flip_xy=True)
            xlim = axs[-1].get_xlim()
            ylim = axs[-1].get_ylim()
            aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
            axs[-1].set_aspect(aspect)
            axs[-1].axhline(y=0, linestyle=':', linewidth=0.5, color='b')
            axs[-1].set_xlim((0, None))

        return plt

    # Sampling at q-points
    def run_qpoints(self,
                    q_points,
                    with_eigenvectors=False,
                    with_group_velocities=False,
                    with_dynamical_matrices=False,
                    nac_q_direction=None):
        """Phonon calculations on q-points.

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
            q=(0,0,0) is replaced by q=epsilon * nac_q_direction where epsilon
            is infinitsimal for non-analytical term correction. This is used,
            e.g., to observe LO-TO splitting.

        """

        if self._dynamical_matrix is None:
            msg = ("Dynamical matrix has not yet built.")
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
            factor=self._factor)

    def set_qpoints_phonon(self,
                           q_points,
                           nac_q_direction=None,
                           is_eigenvectors=False,
                           write_dynamical_matrices=False):
        warnings.warn("Phonopy.set_qpoints_phonon is deprecated. "
                      "Use Phonopy.run_qpoints.", DeprecationWarning)
        if self._group_velocity is None:
            with_group_velocities = False
        else:
            with_group_velocities = True
        self.run_qpoints(
            q_points,
            with_eigenvectors=is_eigenvectors,
            with_group_velocities=with_group_velocities,
            with_dynamical_matrices=write_dynamical_matrices,
            nac_q_direction=nac_q_direction)

    def get_qpoints_dict(self):
        """Returns calculated phonons at q-points

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
            msg = ("run_qpoints has to be done.")
            raise RuntimeError(msg)

        return {'frequencies': self._qpoints.frequencies,
                'eigenvectors': self._qpoints.eigenvectors,
                'group_velocities': self._qpoints.group_velocities,
                'dynamical_matrices': self._qpoints.dynamical_matrices}

    def get_qpoints_phonon(self):
        warnings.warn("Phonopy.get_qpoints_phonon is deprecated. "
                      "Use Phonopy.run_get_qpoints_dict.", DeprecationWarning)
        qpt = self.get_qpoints_dict()
        return (qpt['frequencies'], qpt['eigenvectors'])

    def write_hdf5_qpoints_phonon(self):
        self._qpoints.write_hdf5()

    def write_yaml_qpoints_phonon(self):
        self._qpoints.write_yaml()

    # DOS
    def run_total_dos(self,
                      sigma=None,
                      freq_min=None,
                      freq_max=None,
                      freq_pitch=None,
                      use_tetrahedron_method=True):
        """Calculate total DOS from phonons on sampling mesh.

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

        total_dos = TotalDos(self._mesh,
                             sigma=sigma,
                             use_tetrahedron_method=use_tetrahedron_method)
        total_dos.set_draw_area(freq_min, freq_max, freq_pitch)
        total_dos.run()
        self._total_dos = total_dos

    def set_total_DOS(self,
                      sigma=None,
                      freq_min=None,
                      freq_max=None,
                      freq_pitch=None,
                      tetrahedron_method=False):
        warnings.warn("Phonopy.set_total_DOS is deprecated. "
                      "Use Phonopy.run_total_DOS", DeprecationWarning)

        self.run_total_dos(sigma=sigma,
                           freq_min=freq_min,
                           freq_max=freq_max,
                           freq_pitch=freq_pitch,
                           use_tetrahedron_method=tetrahedron_method)

    def auto_total_dos(self,
                       mesh=100.0,
                       is_time_reversal=True,
                       is_mesh_symmetry=True,
                       is_gamma_center=False,
                       plot=False,
                       write_dat=False,
                       filename="total_dos.dat"):
        self.run_mesh(mesh=mesh,
                      is_time_reversal=is_time_reversal,
                      is_mesh_symmetry=is_mesh_symmetry,
                      is_gamma_center=is_gamma_center)
        self.run_total_dos()
        if write_dat:
            self.write_total_dos(filename=filename)
        if plot:
            return self.plot_total_dos()

    def get_total_dos_dict(self):
        """Return frequencies and total DOS as a dictionary.

        Returns
        -------
        A dictionary with keys of 'frequency_points' and 'total_dos'.
        Each value of corresponding key is as follows:

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        total_dos:
            shape=(frequency_sampling_points, ), dtype='double'

        """

        return {'frequency_points': self._total_dos.frequency_points,
                'total_dos': self._total_dos.dos}

    def get_total_DOS(self):
        """Return frequency points and total DOS as a tuple.

        Returns
        -------
        A tuple with (frequency_points, total_dos).

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        total_dos:
            shape=(frequency_sampling_points, ), dtype='double'

        """

        warnings.warn("Phonopy.get_total_DOS is deprecated. "
                      "Use Phonopy.get_total_dos_dict.", DeprecationWarning)

        dos = self.get_total_dos_dict()

        return dos['frequency_points'], dos['total_dos']

    def set_Debye_frequency(self, freq_max_fit=None):
        self._total_dos.set_Debye_frequency(
            self._primitive.get_number_of_atoms(),
            freq_max_fit=freq_max_fit)

    def get_Debye_frequency(self):
        return self._total_dos.get_Debye_frequency()

    def plot_total_DOS(self):
        warnings.warn("Phonopy.plot_total_DOS is deprecated. "
                      "Use Phonopy.plot_total_dos (lowercase on DOS).",
                      DeprecationWarning)
        return self.plot_total_dos()

    def plot_total_dos(self):
        if self._total_dos is None:
            msg = ("run_total_dos has to be done before plotting "
                   "total DOS.")
            raise RuntimeError(msg)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        self._total_dos.plot(ax, draw_grid=False)
        ax.set_ylim((0, None))

        return plt

    def write_total_DOS(self, filename="total_dos.dat"):
        warnings.warn("Phonopy.write_total_DOS is deprecated. "
                      "Use Phonopy.write_total_dos (lowercase on DOS).",
                      DeprecationWarning)
        self.write_total_dos(filename=filename)

    def write_total_dos(self, filename="total_dos.dat"):
        self._total_dos.write(filename=filename)

    # PDOS
    def run_projected_dos(self,
                          sigma=None,
                          freq_min=None,
                          freq_max=None,
                          freq_pitch=None,
                          use_tetrahedron_method=True,
                          direction=None,
                          xyz_projection=False):
        """Calculate projected DOS from phonons on sampling mesh.

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
            direction_cart = np.dot(direction, self._primitive.get_cell())
        else:
            direction_cart = None
        self._pdos = PartialDos(self._mesh,
                                sigma=sigma,
                                use_tetrahedron_method=use_tetrahedron_method,
                                direction=direction_cart,
                                xyz_projection=xyz_projection)
        self._pdos.set_draw_area(freq_min, freq_max, freq_pitch)
        self._pdos.run()

    def set_partial_DOS(self,
                        sigma=None,
                        freq_min=None,
                        freq_max=None,
                        freq_pitch=None,
                        tetrahedron_method=False,
                        direction=None,
                        xyz_projection=False):
        warnings.warn("Phonopy.set_partial_DOS is deprecated. "
                      "Use Phonopy.run_projected_dos", DeprecationWarning)

        self.run_projected_dos(sigma=sigma,
                               freq_min=freq_min,
                               freq_max=freq_max,
                               freq_pitch=freq_pitch,
                               use_tetrahedron_method=tetrahedron_method,
                               direction=direction,
                               xyz_projection=xyz_projection)

    def auto_projected_dos(self,
                           mesh=100.0,
                           is_time_reversal=True,
                           is_gamma_center=False,
                           plot=False,
                           pdos_indices=None,
                           legend=None,
                           write_dat=False,
                           filename="projected_dos.dat"):
        """Convenient method to calculate/draw projected density of states

        Parameters
        ----------

        See docstring of ``Phonopy.init_mesh`` for the parameters of ``mesh``
        (default is 100.0), ``is_time_reversal`` (default is True),
        and ``is_gamma_center`` (default is False).
        See docstring of ``Phonopy.plot_projected_dos`` for the parameters
        ``pdos_indices`` and ``legend``.

        plot : Bool, optional
            With setting True, PDOS is plotted using matplotlib and
            the matplotlib module (plt) is returned. To watch the result,
            usually ``show()`` has to be called. Default is False.
        write_dat : Bool
            With setting True, ``projected_dos.dat`` like file is written out.
            The  file name can be specified with the ``filename`` parameter.
            Default is False.
        filename : str, optional
            File name used to write ``projected_dos.dat`` like file. Default
            is ``projected_dos.dat``.

        """

        self.run_mesh(mesh=mesh,
                      is_time_reversal=is_time_reversal,
                      is_mesh_symmetry=False,
                      with_eigenvectors=True,
                      is_gamma_center=is_gamma_center)
        self.run_projected_dos()
        if write_dat:
            self.write_projected_dos(filename=filename)
        if plot:
            return self.plot_projected_dos(pdos_indices=pdos_indices,
                                           legend=legend)

    def get_projected_dos_dict(self):
        """Return frequency points and projected DOS as a tuple.

        Projection is done to atoms and may be also done along directions
        depending on the parameters at run_projected_dos.

        Returns
        -------
        A dictionary with keys of 'frequency_points' and 'projected_dos'.
        Each value of corresponding key is as follows:

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        partial_dos:
            shape=(frequency_sampling_points, projections), dtype='double'

        """
        return {'frequency_points': self._pdos.frequency_points,
                'projected_dos': self._pdos.partial_dos}

    def get_partial_DOS(self):
        """Return frequency points and partial DOS as a tuple.

        Projection is done to atoms and may be also done along directions
        depending on the parameters at run_partial_dos.

        Returns
        -------
        A tuple with (frequency_points, partial_dos).

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        partial_dos:
            shape=(frequency_sampling_points, projections), dtype='double'

        """
        warnings.warn("Phonopy.get_partial_DOS is deprecated. "
                      "Use Phonopy.get_projected_dos_dict.",
                      DeprecationWarning)

        pdos = self.get_projected_dos_dict()

        return pdos['frequency_points'], pdos['projected_dos']

    def plot_partial_DOS(self, pdos_indices=None, legend=None):
        warnings.warn("Phonopy.plot_partial_DOS is deprecated. "
                      "Use Phonopy.plot_projected_dos (lowercase on DOS).",
                      DeprecationWarning)

        return self.plot_projected_dos(pdos_indices=pdos_indices,
                                       legend=legend)

    def plot_projected_dos(self, pdos_indices=None, legend=None):
        """Plot projected DOS

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

        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._pdos.plot(ax,
                        indices=pdos_indices,
                        legend=legend,
                        draw_grid=False)

        ax.set_ylim((0, None))

        return plt

    def write_partial_DOS(self, filename="partial_dos.dat"):
        warnings.warn("Phonopy.write_partial_DOS is deprecated. "
                      "Use Phonopy.write_projected_dos (lowercase on DOS).",
                      DeprecationWarning)
        self.write_projected_dos(filename=filename)

    def write_projected_dos(self, filename="projected_dos.dat"):
        self._pdos.write(filename=filename)

    # Thermal property
    def run_thermal_properties(self,
                               t_min=0,
                               t_max=1000,
                               t_step=10,
                               temperatures=None,
                               is_projection=False,
                               band_indices=None,
                               cutoff_frequency=None,
                               pretend_real=False):
        """Calculate thermal properties at constant volume

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default values are 0, 1000, and 10.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.

        """
        if self._mesh is None:
            msg = ("run_mesh has to be done before"
                   "run_thermal_properties.")
            raise RuntimeError(msg)

        tp = ThermalProperties(self._mesh,
                               is_projection=is_projection,
                               band_indices=band_indices,
                               cutoff_frequency=cutoff_frequency,
                               pretend_real=pretend_real)
        if temperatures is None:
            tp.set_temperature_range(t_step=t_step,
                                     t_max=t_max,
                                     t_min=t_min)
        else:
            tp.set_temperatures(temperatures)
        tp.run()
        self._thermal_properties = tp

    def set_thermal_properties(self,
                               t_step=10,
                               t_max=1000,
                               t_min=0,
                               temperatures=None,
                               is_projection=False,
                               band_indices=None,
                               cutoff_frequency=None,
                               pretend_real=False):
        warnings.warn("Phonopy.set_thermal_properties is deprecated. "
                      "Use Phonopy.run_thermal_properties",
                      DeprecationWarning)
        self.run_thermal_properties(t_step=t_step,
                                    t_max=t_max,
                                    t_min=t_min,
                                    temperatures=temperatures,
                                    is_projection=is_projection,
                                    band_indices=band_indices,
                                    cutoff_frequency=cutoff_frequency,
                                    pretend_real=pretend_real)

    def get_thermal_properties_dict(self):
        """Return thermal properties by a dictionary

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

        keys = ('temperatures', 'free_energy', 'entropy', 'heat_capacity')
        return dict(zip(keys, self._thermal_properties.thermal_properties))

    def get_thermal_properties(self):
        """Return thermal properties

        Returns
        -------
        (temperatures, free energy, entropy, heat capacity)

        """
        warnings.warn("Phonopy.get_thermal_properties is deprecated. "
                      "Use Phonopy.get_thermal_properties_dict.",
                      DeprecationWarning)

        tp = self.get_thermal_properties_dict()
        return (tp['temperatures'],
                tp['free_energy'],
                tp['entropy'],
                tp['heat_capacity'])

    def plot_thermal_properties(self):
        import matplotlib.pyplot as plt
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._thermal_properties.plot(plt)

        temps = self._thermal_properties.temperatures
        ax.set_xlim((0, temps[-1]))

        return plt

    def write_yaml_thermal_properties(self,
                                      filename='thermal_properties.yaml'):
        self._thermal_properties.write_yaml(filename=filename)

    # Thermal displacement
    def run_thermal_displacements(self,
                                  t_min=0,
                                  t_max=1000,
                                  t_step=10,
                                  temperatures=None,
                                  direction=None,
                                  freq_min=None,
                                  freq_max=None):
        """Prepare thermal displacements calculation

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
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)
        if self._mesh is None:
            msg = ("run_mesh has to be done.")
            raise RuntimeError(msg)
        mesh_nums = self._mesh.mesh_numbers
        ir_grid_points = self._mesh.ir_grid_points
        if not self._mesh.with_eigenvectors:
            msg = ("run_mesh has to be done with with_eigenvectors=True.")
            raise RuntimeError(msg)
        if np.prod(mesh_nums) != len(ir_grid_points):
            msg = ("run_mesh has to be done with is_mesh_symmetry=False.")
            raise RuntimeError(msg)

        if direction is not None:
            projection_direction = np.dot(direction,
                                          self._primitive.get_cell())
            td = ThermalDisplacements(
                self._mesh,
                projection_direction=projection_direction,
                freq_min=freq_min,
                freq_max=freq_max)
        else:
            td = ThermalDisplacements(self._mesh,
                                      freq_min=freq_min,
                                      freq_max=freq_max)

        if temperatures is None:
            td.set_temperature_range(t_min, t_max, t_step)
        else:
            td.set_temperatures(temperatures)
        td.run()

        self._thermal_displacements = td

    def set_thermal_displacements(self,
                                  t_step=10,
                                  t_max=1000,
                                  t_min=0,
                                  temperatures=None,
                                  direction=None,
                                  freq_min=None,
                                  freq_max=None):
        warnings.warn("Phonopy.set_thermal_displacements is deprecated. "
                      "Use Phonopy.run_thermal_displacements",
                      DeprecationWarning)
        self.run_thermal_displacements(t_min=t_min,
                                       t_max=t_max,
                                       t_step=t_step,
                                       temperatures=temperatures,
                                       direction=direction,
                                       freq_min=freq_min,
                                       freq_max=freq_max)

    def get_thermal_displacements_dict(self):
        if self._thermal_displacements is None:
            msg = ("run_thermal_displacements has to be done.")
            raise RuntimeError(msg)

        td = self._thermal_displacements
        return {'temperatures': td.temperatures,
                'thermal_displacements': td.thermal_displacements}

    def get_thermal_displacements(self):
        warnings.warn("Phonopy.get_thermal_displacements is deprecated. "
                      "Use Phonopy.get_thermal_displacements_dict",
                      DeprecationWarning)
        td = self.get_thermal_displacements_dict()
        return (td['temperatures'], td['thermal_displacements'])

    def plot_thermal_displacements(self, is_legend=False):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        self._thermal_displacements.plot(plt, is_legend=is_legend)

        temps, _ = self._thermal_displacements.get_thermal_displacements()
        ax.set_xlim((0, temps[-1]))

        return plt

    def write_yaml_thermal_displacements(self):
        self._thermal_displacements.write_yaml()

    # Thermal displacement matrix
    def run_thermal_displacement_matrices(self,
                                          t_min=0,
                                          t_max=1000,
                                          t_step=10,
                                          temperatures=None,
                                          freq_min=None,
                                          freq_max=None):
        """Prepare thermal displacement matrices

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
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)
        if self._mesh is None:
            msg = ("run_mesh has to be done.")
            raise RuntimeError(msg)
        mesh_nums = self._mesh.mesh_numbers
        ir_grid_points = self._mesh.ir_grid_points
        if not self._mesh.with_eigenvectors:
            msg = ("run_mesh has to be done with with_eigenvectors=True.")
            raise RuntimeError(msg)
        if np.prod(mesh_nums) != len(ir_grid_points):
            msg = ("run_mesh has to be done with is_mesh_symmetry=False.")
            raise RuntimeError(msg)

        tdm = ThermalDisplacementMatrices(
            self._mesh,
            freq_min=freq_min,
            freq_max=freq_max,
            lattice=self._primitive.get_cell().T)

        if temperatures is None:
            tdm.set_temperature_range(t_min, t_max, t_step)
        else:
            tdm.set_temperatures(temperatures)
        tdm.run()

        self._thermal_displacement_matrices = tdm

    def set_thermal_displacement_matrices(self,
                                          t_step=10,
                                          t_max=1000,
                                          t_min=0,
                                          freq_min=None,
                                          freq_max=None,
                                          t_cif=None):
        warnings.warn("Phonopy.set_thermal_displacement_matrices is "
                      "deprecated. Use Phonopy.run_thermal_displacements",
                      DeprecationWarning)
        if t_cif is None:
            temperatures = None
        else:
            temperatures = [t_cif, ]
        self.run_thermal_displacement_matrices(t_min=t_min,
                                               t_max=t_max,
                                               t_step=t_step,
                                               temperatures=temperatures,
                                               freq_min=freq_min,
                                               freq_max=freq_max)

    def get_thermal_displacement_matrices_dict(self):
        if self._thermal_displacement_matrices is None:
            msg = ("run_thermal_displacement_matrices has to be done.")
            raise RuntimeError(msg)

        tdm = self._thermal_displacement_matrices
        return {'temperatures': tdm.temperatures,
                'thermal_displacement_matrices':
                tdm.thermal_displacement_matrices,
                'thermal_displacement_matrices_cif':
                tdm.thermal_displacement_matrices_cif}

    def get_thermal_displacement_matrices(self):
        warnings.warn("Phonopy.get_thermal_displacement_matrices is "
                      "deprecated. Use "
                      "Phonopy.get_thermal_displacement_matrices_dict",
                      DeprecationWarning)
        tdm = self.get_thermal_displacement_matrices_dict()
        return (tdm['temperatures'],
                tdm['thermal_displacement_matrices'])

    def write_yaml_thermal_displacement_matrices(self):
        self._thermal_displacement_matrices.write_yaml()

    def write_thermal_displacement_matrix_to_cif(self, temperature_index):
        self._thermal_displacement_matrices.write_cif(self._primitive,
                                                      temperature_index)

    # Normal mode animation
    def write_animation(self,
                        q_point=None,
                        anime_type='v_sim',
                        band_index=None,
                        amplitude=None,
                        num_div=None,
                        shift=None,
                        filename=None):
        if self._dynamical_matrix is None:
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)

        if q_point is None:
            animation = Animation([0, 0, 0],
                                  self._dynamical_matrix,
                                  shift=shift)
        else:
            animation = Animation(q_point,
                                  self._dynamical_matrix,
                                  shift=shift)
        if anime_type == 'v_sim':
            if amplitude:
                amplitude_ = amplitude
            else:
                amplitude_ = 1.0

            if filename:
                animation.write_v_sim(amplitude=amplitude_,
                                      factor=self._factor,
                                      filename=filename)
            else:
                animation.write_v_sim(amplitude=amplitude_,
                                      factor=self._factor)
        if anime_type in ('arc', 'xyz', 'jmol', 'poscar'):
            if band_index is None or amplitude is None or num_div is None:
                msg = ("Parameters are not correctly set for animation.")
                raise RuntimeError(msg)

            if anime_type == 'arc' or anime_type is None:
                if filename:
                    animation.write_arc(band_index,
                                        amplitude,
                                        num_div,
                                        filename=filename)
                else:
                    animation.write_arc(band_index,
                                        amplitude,
                                        num_div)

            if anime_type == 'xyz':
                if filename:
                    animation.write_xyz(band_index,
                                        amplitude,
                                        num_div,
                                        self._factor,
                                        filename=filename)
                else:
                    animation.write_xyz(band_index,
                                        amplitude,
                                        num_div,
                                        self._factor)

            if anime_type == 'jmol':
                if filename:
                    animation.write_xyz_jmol(amplitude=amplitude,
                                             factor=self._factor,
                                             filename=filename)
                else:
                    animation.write_xyz_jmol(amplitude=amplitude,
                                             factor=self._factor)

            if anime_type == 'poscar':
                if filename:
                    animation.write_POSCAR(band_index,
                                           amplitude,
                                           num_div,
                                           filename=filename)
                else:
                    animation.write_POSCAR(band_index,
                                           amplitude,
                                           num_div)

    # Atomic modulation of normal mode
    def set_modulations(self,
                        dimension,
                        phonon_modes,
                        delta_q=None,
                        derivative_order=None,
                        nac_q_direction=None):
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

        """
        if self._dynamical_matrix is None:
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)

        self._modulation = Modulation(self._dynamical_matrix,
                                      dimension,
                                      phonon_modes,
                                      delta_q=delta_q,
                                      derivative_order=derivative_order,
                                      nac_q_direction=nac_q_direction,
                                      factor=self._factor)
        self._modulation.run()

    def get_modulated_supercells(self):
        """Returns cells with modulations as PhonopyAtoms instances"""
        return self._modulation.get_modulated_supercells()

    def get_modulations_and_supercell(self):
        """Return modulations and supercell

        (modulations, supercell)

        modulations: Atomic modulations of supercell in Cartesian coordinates
        supercell: Supercell as an PhonopyAtoms instance.

        """
        return self._modulation.get_modulations_and_supercell()

    def write_modulations(self):
        """Create MPOSCAR's"""
        self._modulation.write()

    def write_yaml_modulations(self):
        self._modulation.write_yaml()

    # Irreducible representation
    def set_irreps(self,
                   q,
                   is_little_cogroup=False,
                   nac_q_direction=None,
                   degeneracy_tolerance=1e-4):
        """Identify ir-reps of phonon modes.

        The design of this API is not very satisfactory and is expceted
        to be redesined in the next major versions once the use case
        of the API for ir-reps feature becomes clearer.

        """

        if self._dynamical_matrix is None:
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)

        self._irreps = IrReps(
            self._dynamical_matrix,
            q,
            is_little_cogroup=is_little_cogroup,
            nac_q_direction=nac_q_direction,
            factor=self._factor,
            symprec=self._symprec,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=self._log_level)

        return self._irreps.run()

    def get_irreps(self):
        return self._irreps

    def show_irreps(self, show_irreps=False):
        self._irreps.show(show_irreps=show_irreps)

    def write_yaml_irreps(self, show_irreps=False):
        self._irreps.write_yaml(show_irreps=show_irreps)

    # Group velocity
    def set_group_velocity(self, q_length=None):
        warnings.warn("Phonopy.set_group_velocity is deprecated. "
                      "No need to call this. gv_delta_q "
                      "(q_length) is set at Phonopy.__init__().",
                      DeprecationWarning)
        self._gv_delta_q = q_length
        self._set_group_velocity()

    def get_group_velocity(self):
        warnings.warn("Phonopy.get_group_velocities_on_bands is deprecated. "
                      "Use Phonopy.[mode].group_velocities attribute or "
                      "Phonopy.get_[mode]_dict()[group_velocities], where "
                      "[mode] is band_structure, mesh, or qpoints.",
                      DeprecationWarning)
        return self._group_velocity.get_group_velocity()

    def get_group_velocity_at_q(self, q_point):
        if self._group_velocity is None:
            self._set_group_velocity()
        self._group_velocity.run([q_point])
        return self._group_velocity.group_velocities[0]

    def get_group_velocities_on_bands(self):
        warnings.warn(
            "Phonopy.get_group_velocities_on_bands is deprecated. "
            "Use Phonopy.get_band_structure_dict()['group_velocities'].",
            DeprecationWarning)
        return self._band_structure.group_velocities

    # Moment
    def run_moment(self,
                   order=1,
                   is_projection=False,
                   freq_min=None,
                   freq_max=None):
        if self._mesh is None:
            msg = ("run_mesh has to be done before run_moment.")
            raise RuntimeError(msg)
        else:
            if is_projection:
                if self._mesh.eigenvectors is None:
                    return RuntimeError(
                        "run_mesh has to be done with with_eigenvectors=True.")
                self._moment = PhononMoment(
                    self._mesh.frequencies,
                    weights=self._mesh.weights,
                    eigenvectors=self._mesh.eigenvectors)
            else:
                self._moment = PhononMoment(
                    self._mesh.get_frequencies(),
                    weights=self._mesh.get_weights())
            if freq_min is not None or freq_max is not None:
                self._moment.set_frequency_range(freq_min=freq_min,
                                                 freq_max=freq_max)
            self._moment.run(order=order)

    def set_moment(self,
                   order=1,
                   is_projection=False,
                   freq_min=None,
                   freq_max=None):
        warnings.warn("Phonopy.set_moment is deprecated. "
                      "Use Phonopy.run_moment.", DeprecationWarning)
        self.run_moment(order=order,
                        is_projection=is_projection,
                        freq_min=freq_min,
                        freq_max=freq_max)

    def get_moment(self):
        return self._moment.moment

    def init_dynamic_structure_factor(self,
                                      Qpoints,
                                      T,
                                      atomic_form_factor_func=None,
                                      scattering_lengths=None,
                                      freq_min=None,
                                      freq_max=None):
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
            msg = ("run_mesh has to be done before initializing dynamic"
                   "structure factor.")
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
            freq_max=freq_max)

    def run_dynamic_structure_factor(self,
                                     Qpoints,
                                     T,
                                     atomic_form_factor_func=None,
                                     scattering_lengths=None,
                                     freq_min=None,
                                     freq_max=None):
        """Run dynamic structure factor calculation

        See the detail of parameters at
        Phonopy.init_dynamic_structure_factor().

        """

        self.init_dynamic_structure_factor(
            Qpoints,
            T,
            atomic_form_factor_func=atomic_form_factor_func,
            scattering_lengths=scattering_lengths,
            freq_min=freq_min,
            freq_max=freq_max)
        self._dynamic_structure_factor.run()

    def set_dynamic_structure_factor(self,
                                     Qpoints,
                                     T,
                                     atomic_form_factor_func=None,
                                     scattering_lengths=None,
                                     freq_min=None,
                                     freq_max=None,
                                     run_immediately=True):
        if run_immediately:
            self.run_dynamic_structure_factor(
                Qpoints,
                T,
                atomic_form_factor_func=atomic_form_factor_func,
                scattering_lengths=scattering_lengths,
                freq_min=freq_min,
                freq_max=freq_max)
        else:
            self.init_dynamic_structure_factor(
                Qpoints,
                T,
                atomic_form_factor_func=atomic_form_factor_func,
                scattering_lengths=scattering_lengths,
                freq_min=freq_min,
                freq_max=freq_max)

    def get_dynamic_structure_factor(self):
        return (self._dynamic_structure_factor.qpoints,
                self._dynamic_structure_factor.dynamic_structure_factors)

    def run_random_displacements(self,
                                 temperature,
                                 number_of_snapshots=1,
                                 random_seed=None,
                                 dist_func=None,
                                 cutoff_frequency=None):
        """Generate random displacements from phonon structure

        Some more details are written at generate_displacements.

        temperature : float
            Temperature.
        number_of_snapshots : int
            Number of snapshots with random displacements created.
        random_seed : 32bit unsigned int
            Random seed.
        dist_func : str, None
            Harmonic oscillator distribution function either by 'quantum'
            or 'classical'. Default is None, corresponding to 'quantum'.
        cutoff_frequency : float
            Phonon frequency in THz below that phonons are ignored
            to generate random displacements.

        """

        self._random_displacements = RandomDisplacements(
            self._supercell,
            self._primitive,
            self._force_constants,
            dist_func=dist_func,
            cutoff_frequency=cutoff_frequency,
            factor=self._factor)
        self._random_displacements.run(
            temperature,
            number_of_snapshots=number_of_snapshots,
            random_seed=random_seed)

    def save(self,
             filename="phonopy_params.yaml",
             settings=None):
        """Save phonopy parameters into file.

        Parameters
        ----------
        filename: str, optional
            File name. Default is "phonopy_params.yaml"
        settings: dict, optional

            It is described which parameters are written out. Only the
            settings expected to be updated from the following default
            settings are needed to be set in the dictionary.  The
            possible parameters and their default settings are:
                {'force_sets': True,
                 'displacements': True,
                 'force_constants': False,
                 'born_effective_charge': True,
                 'dielectric_constant': True}
            This default settings are updated by {'force_constants': True}
            when dataset is None and force_constants is not None.

        """

        phpy_yaml = PhonopyYaml(settings=settings)
        if (not forces_in_dataset(self.dataset) and
            self.force_constants is not None):
            phpy_yaml.settings.update({'force_sets': False,
                                       'displacements': False,
                                       'force_constants': True})
        phpy_yaml.set_phonon_info(self)
        with open(filename, 'w') as w:
            w.write(str(phpy_yaml))

    ###################
    # private methods #
    ###################
    def _run_force_constants_from_forces(self,
                                         distributed_atom_list=None,
                                         fc_calculator=None,
                                         fc_calculator_options=None,
                                         decimals=None):
        if self._displacement_dataset is not None:
            if fc_calculator is not None:
                disps, forces = get_displacements_and_forces(
                    self._displacement_dataset)
                self._force_constants = get_fc2(
                    self._supercell,
                    self._primitive,
                    disps,
                    forces,
                    fc_calculator=fc_calculator,
                    fc_calculator_options=fc_calculator_options,
                    atom_list=distributed_atom_list,
                    log_level=self._log_level,
                    symprec=self._symprec)
            else:
                if 'displacements' in self._displacement_dataset:
                    lines = [
                        "Type-II dataset for displacements and forces was "
                        "given. fc_calculator",
                        "(external force constants calculator) is required "
                        "to produce force constants."]
                    raise RuntimeError("\n".join(lines))
                self._force_constants = get_phonopy_fc2(
                    self._supercell,
                    self._symmetry,
                    self._displacement_dataset,
                    atom_list=distributed_atom_list,
                    decimals=decimals)

    def _set_dynamical_matrix(self):
        self._dynamical_matrix = None

        if self._is_symmetry and self._nac_params is not None:
            borns, epsilon = symmetrize_borns_and_epsilon(
                self._nac_params['born'],
                self._nac_params['dielectric'],
                self._primitive,
                symprec=self._symprec)
            nac_params = self._nac_params.copy()
            nac_params.update({'born': borns, 'dielectric': epsilon})
        else:
            nac_params = self._nac_params

        if self._supercell is None or self._primitive is None:
            raise RuntimeError("Supercell or primitive is not created.")
        if self._force_constants is None:
            raise RuntimeError("Force constants are not prepared.")
        if self._primitive.get_masses() is None:
            raise RuntimeError("Atomic masses are not correctly set.")
        self._dynamical_matrix = get_dynamical_matrix(
            self._force_constants,
            self._supercell,
            self._primitive,
            nac_params,
            self._frequency_scale_factor,
            self._dynamical_matrix_decimals,
            symprec=self._symprec,
            log_level=self._log_level)
        # DynamialMatrix instance transforms force constants in correct
        # type of numpy array.
        self._force_constants = self._dynamical_matrix.force_constants

        if self._group_velocity is not None:
            self._set_group_velocity()

    def _set_group_velocity(self):
        if self._dynamical_matrix is None:
            raise RuntimeError("Dynamical matrix has not yet built.")

        if (self._dynamical_matrix.is_nac() and
            self._dynamical_matrix.get_nac_method() == 'gonze' and
            self._gv_delta_q is None):
            self._gv_delta_q = 1e-5
            if self._log_level:
                msg = "Group velocity calculation:\n"
                text = ("Analytical derivative of dynamical matrix is not "
                        "implemented for NAC by Gonze et al. Instead "
                        "numerical derivative of it is used with dq=1e-5 "
                        "for group velocity calculation.")
                msg += textwrap.fill(text,
                                     initial_indent="  ",
                                     subsequent_indent="  ",
                                     width=70)
                print(msg)

        self._group_velocity = GroupVelocity(
            self._dynamical_matrix,
            q_length=self._gv_delta_q,
            symmetry=self._primitive_symmetry,
            frequency_factor_to_THz=self._factor)

    def _search_symmetry(self):
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)

    def _search_primitive_symmetry(self):
        self._primitive_symmetry = Symmetry(self._primitive,
                                            self._symprec,
                                            self._is_symmetry)

        if (len(self._symmetry.get_pointgroup_operations()) !=
            len(self._primitive_symmetry.get_pointgroup_operations())):
            print("Warning: Point group symmetries of supercell and primitive"
                  "cell are different.")

    def _build_supercell(self):
        self._supercell = get_supercell(self._unitcell,
                                        self._supercell_matrix,
                                        self._symprec)

    def _build_supercells_with_displacements(self):
        all_positions = []
        if 'first_atoms' in self._displacement_dataset:  # type-1
            for disp in self._displacement_dataset['first_atoms']:
                positions = self._supercell.get_positions()
                positions[disp['number']] += disp['displacement']
                all_positions.append(positions)
        elif 'displacements' in self._displacement_dataset:
            for disp in self._displacement_dataset['displacements']:
                all_positions.append(self._supercell.positions + disp)
        else:
            raise RuntimeError("displacement_dataset is not set.")

        supercells = []
        for positions in all_positions:
            supercells.append(PhonopyAtoms(
                    numbers=self._supercell.get_atomic_numbers(),
                    masses=self._supercell.get_masses(),
                    magmoms=self._supercell.get_magnetic_moments(),
                    positions=positions,
                    cell=self._supercell.get_cell(),
                    pbc=True))
        self._supercells_with_displacements = supercells

    def _build_primitive_cell(self):
        """
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
                self._supercell, trans_mat, self._symprec)
        except ValueError:
            msg = ("Creating primitive cell is failed. "
                   "PRIMITIVE_AXIS may be incorrectly specified.")
            raise RuntimeError(msg)

    def _guess_primitive_matrix(self):
        return guess_primitive_matrix(self._unitcell, symprec=self._symprec)

    def _shape_supercell_matrix(self, smat):
        return shape_supercell_matrix(smat)
