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
import numpy as np
from phonopy.version import __version__
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import get_supercell, get_primitive
from phonopy.harmonic.displacement import (get_least_displacements,
                                           direction_to_displacement)
from phonopy.harmonic.force_constants import (get_fc2,
                                              symmetrize_force_constants,
                                              rotational_invariance,
                                              cutoff_force_constants,
                                              set_tensor_symmetry)
from phonopy.harmonic.dynamical_matrix import (DynamicalMatrix,
                                               DynamicalMatrixNAC)
from phonopy.phonon.band_structure import BandStructure
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.phonon.mesh import Mesh
from phonopy.units import VaspToTHz
from phonopy.phonon.dos import TotalDos, PartialDos
from phonopy.phonon.thermal_displacement import (ThermalDisplacements,
                                                 ThermalDistances,
                                                 ThermalDisplacementMatrices)
from phonopy.phonon.animation import Animation
from phonopy.phonon.modulation import Modulation
from phonopy.phonon.qpoints_mode import QpointsPhonon
from phonopy.phonon.irreps import IrReps
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh
from phonopy.phonon.moment import PhononMoment

class Phonopy(object):
    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 primitive_matrix=None,
                 nac_params=None,
                 distance=None,
                 factor=VaspToTHz,
                 is_auto_displacements=None,
                 dynamical_matrix_decimals=None,
                 force_constants_decimals=None,
                 symprec=1e-5,
                 is_symmetry=True,
                 use_lapack_solver=False,
                 log_level=0):

        if is_auto_displacements is not None:
            print("Warning: \'is_auto_displacements\' argument is obsolete.")
            if is_auto_displacements is False:
                print("Sets of displacements are not created as default.")
            else:
                print("Use \'generate_displacements\' method explicitly to "
                      "create sets of displacements.")

        if distance is not None:
            print("Warning: \'distance\' keyword argument is obsolete at "
                  "Phonopy instantiation.")
            print("Specify \'distance\' keyword argument when calling "
                  "\'generate_displacements\'")
            print("method (See the Phonopy API document).")

        self._symprec = symprec
        self._factor = factor
        self._is_symmetry = is_symmetry
        self._use_lapack_solver = use_lapack_solver
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = Atoms(atoms=unitcell)
        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        self._supercell = None
        self._primitive = None
        self._build_supercell()
        self._build_primitive_cell()

        # Set supercell and primitive symmetry
        self._symmetry = None
        self._primitive_symmetry = None
        self._search_symmetry()
        self._search_primitive_symmetry()

        # set_displacements (used only in preprocess)
        self._displacement_dataset = None
        self._displacements = None
        self._displacement_directions = None
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

    def get_version(self):
        return __version__

    def get_primitive(self):
        return self._primitive
    primitive = property(get_primitive)

    def get_unitcell(self):
        return self._unitcell
    unitcell = property(get_unitcell)

    def get_supercell(self):
        return self._supercell
    supercell = property(get_supercell)

    def get_symmetry(self):
        """return symmetry of supercell"""
        return self._symmetry
    symmetry = property(get_symmetry)

    def get_primitive_symmetry(self):
        """return symmetry of primitive cell"""
        return self._primitive_symmetry

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def get_primitive_matrix(self):
        return self._primitive_matrix

    def get_unit_conversion_factor(self):
        return self._factor
    unit_conversion_factor = property(get_unit_conversion_factor)

    def get_displacement_dataset(self):
        return self._displacement_dataset

    def get_displacements(self):
        return self._displacements
    displacements = property(get_displacements)

    def get_displacement_directions(self):
        return self._displacement_directions
    displacement_directions = property(get_displacement_directions)

    def get_supercells_with_displacements(self):
        if self._displacement_dataset is None:
            return None
        else:
            self._build_supercells_with_displacements()
            return self._supercells_with_displacements

    def get_force_constants(self):
        return self._force_constants
    force_constants = property(get_force_constants)

    def get_rotational_condition_of_fc(self):
        return rotational_invariance(self._force_constants,
                                     self._supercell,
                                     self._primitive,
                                     self._symprec)

    def get_nac_params(self):
        return self._nac_params

    def get_dynamical_matrix(self):
        return self._dynamical_matrix
    dynamical_matrix = property(get_dynamical_matrix)

    def set_unitcell(self, unitcell):
        self._unitcell = unitcell
        self._build_supercell()
        self._build_primitive_cell()
        self._search_symmetry()
        self._search_primitive_symmetry()
        self._displacement_dataset = None

    def set_masses(self, masses):
        p_masses = np.array(masses)
        self._primitive.set_masses(p_masses)
        p2p_map = self._primitive.get_primitive_to_primitive_map()
        s_masses = p_masses[[p2p_map[x] for x in
                             self._primitive.get_supercell_to_primitive_map()]]
        self._supercell.set_masses(s_masses)
        u2s_map = self._supercell.get_unitcell_to_supercell_map()
        u_masses = s_masses[u2s_map]
        self._unitcell.set_masses(u_masses)
        self._set_dynamical_matrix()

    def set_nac_params(self, nac_params=None):
        self._nac_params = nac_params
        self._set_dynamical_matrix()

    def set_displacement_dataset(self, displacement_dataset):
        """
        displacement_dataset:
           {'natom': number_of_atoms_in_supercell,
            'first_atoms': [
              {'number': atom index of displaced atom,
               'displacement': displacement in Cartesian coordinates,
               'direction': displacement direction with respect to axes
               'forces': forces on atoms in supercell},
              {...}, ...]}
        """
        self._displacement_dataset = displacement_dataset

        self._displacements = []
        self._displacement_directions = []
        for disp in self._displacement_dataset['first_atoms']:
            x = disp['displacement']
            self._displacements.append([disp['number'], x[0], x[1], x[2]])
            if 'direction' in disp:
                y = disp['direction']
                self._displacement_directions.append(
                    [disp['number'], y[0], y[1], y[2]])
        if not self._displacement_directions:
            self._displacement_directions = None

    def set_forces(self, sets_of_forces):
        """
        sets_of_forces:
           [[[f_1x, f_1y, f_1z], [f_2x, f_2y, f_2z], ...], # first supercell
             [[f_1x, f_1y, f_1z], [f_2x, f_2y, f_2z], ...], # second supercell
             ...                                                  ]
        """
        for disp, forces in zip(
                self._displacement_dataset['first_atoms'], sets_of_forces):
            disp['forces'] = forces

    def set_force_constants(self, force_constants):
        self._force_constants = force_constants
        self._set_dynamical_matrix()

    def set_force_constants_zero_with_radius(self, cutoff_radius):
        cutoff_force_constants(self._force_constants,
                               self._supercell,
                               cutoff_radius,
                               symprec=self._symprec)
        self._set_dynamical_matrix()

    def set_dynamical_matrix(self):
        self._set_dynamical_matrix()

    def generate_displacements(self,
                               distance=0.01,
                               is_plusminus='auto',
                               is_diagonal=True,
                               is_trigonal=False):
        """Generate displacements automatically

        displacemsts: List of displacements in Cartesian coordinates.
           [[0, 0.01, 0.00, 0.00], ...]
        where each set of elements is defined by:
           First value:      Atom index in supercell starting with 0
           Second to fourth: Displacement in Cartesian coordinates

        displacement_directions:
          List of directions with respect to axes. This gives only the
          symmetrically non equivalent directions. The format is like:
             [[0, 1, 0, 0],
              [7, 1, 0, 1], ...]
          where each list is defined by:
             First value:      Atom index in supercell starting with 0
             Second to fourth: If the direction is displaced or not ( 1, 0, or -1 )
                               with respect to the axes.

        """
        displacement_directions = get_least_displacements(
            self._symmetry,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal,
            is_trigonal=is_trigonal,
            log_level=self._log_level)
        displacement_dataset = direction_to_displacement(
            displacement_directions,
            distance,
            self._supercell)
        self.set_displacement_dataset(displacement_dataset)

    def produce_force_constants(self,
                                forces=None,
                                calculate_full_force_constants=True,
                                computation_algorithm="svd"):
        if forces is not None:
            self.set_forces(forces)

        # A primitive check if 'forces' key is in displacement_dataset.
        for disp in self._displacement_dataset['first_atoms']:
            if 'forces' not in disp:
                return False

        if calculate_full_force_constants:
            self._run_force_constants_from_forces(
                decimals=self._force_constants_decimals,
                computation_algorithm=computation_algorithm)
        else:
            p2s_map = self._primitive.get_primitive_to_supercell_map()
            self._run_force_constants_from_forces(
                distributed_atom_list=p2s_map,
                decimals=self._force_constants_decimals,
                computation_algorithm=computation_algorithm)

        self._set_dynamical_matrix()

        return True

    def symmetrize_force_constants(self, iteration=3):
        symmetrize_force_constants(self._force_constants, iteration)
        self._set_dynamical_matrix()

    def symmetrize_force_constants_by_space_group(self):
        from phonopy.harmonic.force_constants import (set_tensor_symmetry,
                                                      set_tensor_symmetry_PJ)
        set_tensor_symmetry_PJ(self._force_constants,
                               self._supercell.get_cell().T,
                               self._supercell.get_scaled_positions(),
                               self._symmetry)

        self._set_dynamical_matrix()

    #####################
    # Phonon properties #
    #####################

    # Single q-point
    def get_dynamical_matrix_at_q(self, q):
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            return None

        self._dynamical_matrix.set_dynamical_matrix(q)
        return self._dynamical_matrix.get_dynamical_matrix()


    def get_frequencies(self, q):
        """
        Calculate phonon frequencies at q

        q: q-vector in reduced coordinates of primitive cell
        """
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            return None

        self._dynamical_matrix.set_dynamical_matrix(q)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        frequencies = []
        for eig in np.linalg.eigvalsh(dm).real:
            if eig < 0:
                frequencies.append(-np.sqrt(-eig))
            else:
                frequencies.append(np.sqrt(eig))

        return np.array(frequencies) * self._factor

    def get_frequencies_with_eigenvectors(self, q):
        """
        Calculate phonon frequencies and eigenvectors at q

        q: q-vector in reduced coordinates of primitive cell
        """
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            return None

        self._dynamical_matrix.set_dynamical_matrix(q)
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
    def set_band_structure(self,
                           bands,
                           is_eigenvectors=False,
                           is_band_connection=False):
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._band_structure = None
            return False

        self._band_structure = BandStructure(
            bands,
            self._dynamical_matrix,
            is_eigenvectors=is_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=self._group_velocity,
            factor=self._factor)
        return True

    def get_band_structure(self):
        band = self._band_structure
        return (band.get_qpoints(),
                band.get_distances(),
                band.get_frequencies(),
                band.get_eigenvectors())

    def plot_band_structure(self, labels=None):
        import matplotlib.pyplot as plt
        if labels:
            from matplotlib import rc
            rc('text', usetex=True)

        self._band_structure.plot(plt, labels=labels)
        return plt

    def write_yaml_band_structure(self,
                                  labels=None,
                                  comment=None,
                                  filename="band.yaml"):
        self._band_structure.write_yaml(labels=labels,
                                        comment=comment,
                                        filename=filename)

    # Sampling mesh
    def set_mesh(self,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False):
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._mesh = None
            return False

        self._mesh = Mesh(
            self._dynamical_matrix,
            mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            is_eigenvectors=is_eigenvectors,
            is_gamma_center=is_gamma_center,
            group_velocity=self._group_velocity,
            rotations=self._primitive_symmetry.get_pointgroup_operations(),
            factor=self._factor,
            use_lapack_solver=self._use_lapack_solver)
        return True

    def get_mesh(self):
        if self._mesh is None:
            return None
        else:
            return (self._mesh.get_qpoints(),
                    self._mesh.get_weights(),
                    self._mesh.get_frequencies(),
                    self._mesh.get_eigenvectors())
    
    def get_mesh_grid_info(self):
        if self._mesh is None:
            return None
        else:
            return (self._mesh.get_grid_address(),
                    self._mesh.get_ir_grid_points(),
                    self._mesh.get_grid_mapping_table())

    def write_hdf5_mesh(self):
        self._mesh.write_hdf5()

    def write_yaml_mesh(self):
        self._mesh.write_yaml()

    # Plot band structure and DOS (PDOS) together
    def plot_band_structure_and_dos(self, pdos_indices=None, labels=None):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        if labels:
            from matplotlib import rc
            rc('text', usetex=True)

        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0, 0])
        self._band_structure.plot(plt, labels=labels)
        ax2 = plt.subplot(gs[0, 1], sharey=ax1)
        plt.subplots_adjust(wspace=0.03)
        plt.setp(ax2.get_yticklabels(), visible=False)

        if pdos_indices is None:
            self._total_dos.plot(plt,
                                 ylabel="",
                                 draw_grid=False,
                                 flip_xy=True)
        else:
            self._pdos.plot(plt,
                            indices=pdos_indices,
                            ylabel="",
                            draw_grid=False,
                            flip_xy=True)

        return plt

    # DOS
    def set_total_DOS(self,
                      sigma=None,
                      freq_min=None,
                      freq_max=None,
                      freq_pitch=None,
                      tetrahedron_method=False):

        if self._mesh is None:
            print("Warning: \'set_mesh\' has to finish correctly "
                  "before DOS calculation.")
            self._total_dos = None
            return False

        total_dos = TotalDos(self._mesh,
                             sigma=sigma,
                             tetrahedron_method=tetrahedron_method)
        total_dos.set_draw_area(freq_min, freq_max, freq_pitch)
        total_dos.run()
        self._total_dos = total_dos
        return True

    def get_total_DOS(self):
        """
        Retern frequencies and total dos.
        The first element is freqs and the second is total dos.

        frequencies: [freq1, freq2, ...]
        total_dos: [dos1, dos2, ...]
        """
        return self._total_dos.get_dos()

    def set_Debye_frequency(self, freq_max_fit=None):
        self._total_dos.set_Debye_frequency(
            self._primitive.get_number_of_atoms(),
            freq_max_fit=freq_max_fit)

    def get_Debye_frequency(self):
        return self._total_dos.get_Debye_frequency()

    def plot_total_DOS(self):
        import matplotlib.pyplot as plt
        self._total_dos.plot(plt)
        return plt

    def write_total_DOS(self):
        self._total_dos.write()

    # PDOS
    def set_partial_DOS(self,
                        sigma=None,
                        freq_min=None,
                        freq_max=None,
                        freq_pitch=None,
                        tetrahedron_method=False,
                        direction=None,
                        xyz_projection=False):
        self._pdos = None

        if self._mesh is None:
            print("Warning: \'set_mesh\' has to be called before "
                  "PDOS calculation.")
            return False

        if self._mesh.get_eigenvectors() is None:
            print("Warning: Eigenvectors have to be calculated.")
            return False

        num_grid = np.prod(self._mesh.get_mesh_numbers())
        if num_grid != len(self._mesh.get_ir_grid_points()):
            print("Warning: \'set_mesh\' has to be called with "
                  "is_mesh_symmetry=False.")
            return False

        if direction is not None:
            direction_cart = np.dot(direction, self._primitive.get_cell())
        else:
            direction_cart = None
        self._pdos = PartialDos(self._mesh,
                                sigma=sigma,
                                tetrahedron_method=tetrahedron_method,
                                direction=direction_cart,
                                xyz_projection=xyz_projection)
        self._pdos.set_draw_area(freq_min, freq_max, freq_pitch)
        self._pdos.run()
        return True

    def get_partial_DOS(self):
        """
        Retern frequencies and partial_dos.
        The first element is freqs and the second is partial_dos.

        frequencies: [freq1, freq2, ...]
        partial_dos:
          [[atom1-freq1, atom1-freq2, ...],
           [atom2-freq1, atom2-freq2, ...],
           ...]
        """
        return self._pdos.get_partial_dos()

    def plot_partial_DOS(self, pdos_indices=None, legend=None):
        import matplotlib.pyplot as plt
        self._pdos.plot(plt,
                        indices=pdos_indices,
                        legend=legend)
        return plt

    def write_partial_DOS(self):
        self._pdos.write()

    # Thermal property
    def set_thermal_properties(self,
                               t_step=10,
                               t_max=1000,
                               t_min=0,
                               temperatures=None,
                               is_projection=False,
                               band_indices=None,
                               cutoff_frequency=None):
        if self._mesh is None:
            print("Warning: set_mesh has to be done before "
                  "set_thermal_properties")
            return False
        else:
            tp = ThermalProperties(self._mesh.get_frequencies(),
                                   weights=self._mesh.get_weights(),
                                   eigenvectors=self._mesh.get_eigenvectors(),
                                   is_projection=is_projection,
                                   band_indices=band_indices,
                                   cutoff_frequency=cutoff_frequency)
            if temperatures is None:
                tp.set_temperature_range(t_step=t_step,
                                         t_max=t_max,
                                         t_min=t_min)
            else:
                tp.set_temperatures(temperatures)
            tp.run()
            self._thermal_properties = tp

    def get_thermal_properties(self):
        temps, fe, entropy, cv = \
            self._thermal_properties.get_thermal_properties()
        return temps, fe, entropy, cv

    def plot_thermal_properties(self):
        import matplotlib.pyplot as plt
        self._thermal_properties.plot(plt)
        return plt

    def write_yaml_thermal_properties(self, filename='thermal_properties.yaml'):
        self._thermal_properties.write_yaml(filename=filename)

    # Thermal displacement
    def set_thermal_displacements(self,
                                  t_step=10,
                                  t_max=1000,
                                  t_min=0,
                                  temperatures=None,
                                  direction=None,
                                  cutoff_frequency=None):
        """
        cutoff_frequency:
          phonon modes that have frequencies below cutoff_frequency
          are ignored.

        direction:
          Projection direction in reduced coordinates
        """
        self._thermal_displacements = None

        if self._mesh is None:
            print("Warning: \'set_mesh\' has to finish correctly "
                  "before \'set_thermal_displacements\'.")
            return False

        eigvecs = self._mesh.get_eigenvectors()
        frequencies = self._mesh.get_frequencies()
        mesh_nums = self._mesh.get_mesh_numbers()

        if self._mesh.get_eigenvectors() is None:
            print("Warning: Eigenvectors have to be calculated.")
            return False

        if np.prod(mesh_nums) != len(eigvecs):
            print("Warning: Sampling mesh must not be symmetrized.")
            return False

        td = ThermalDisplacements(frequencies,
                                  eigvecs,
                                  self._primitive.get_masses(),
                                  cutoff_frequency=cutoff_frequency)
        if temperatures is None:
            td.set_temperature_range(t_min, t_max, t_step)
        else:
            td.set_temperatures(temperatures)
        if direction is not None:
            td.project_eigenvectors(direction, self._primitive.get_cell())
        td.run()

        self._thermal_displacements = td
        return True

    def get_thermal_displacements(self):
        if self._thermal_displacements is not None:
            return self._thermal_displacements.get_thermal_displacements()

    def plot_thermal_displacements(self, is_legend=False):
        import matplotlib.pyplot as plt
        self._thermal_displacements.plot(plt, is_legend=is_legend)
        return plt

    def write_yaml_thermal_displacements(self):
        self._thermal_displacements.write_yaml()

    # Thermal displacement matrix
    def set_thermal_displacement_matrices(self,
                                          t_step=10,
                                          t_max=1000,
                                          t_min=0,
                                          cutoff_frequency=None,
                                          t_cif=None):
        """
        cutoff_frequency:
          phonon modes that have frequencies below cutoff_frequency
          are ignored.

        direction:
          Projection direction in reduced coordinates
        """
        self._thermal_displacement_matrices = None

        if self._mesh is None:
            print("Warning: \'set_mesh\' has to finish correctly "
                  "before \'set_thermal_displacement_matrices\'.")
            return False

        eigvecs = self._mesh.get_eigenvectors()
        frequencies = self._mesh.get_frequencies()
        mesh_nums = self._mesh.get_mesh_numbers()

        if self._mesh.get_eigenvectors() is None:
            print("Warning: Eigenvectors have to be calculated.")
            return False

        if np.prod(mesh_nums) != len(eigvecs):
            print("Warning: Sampling mesh must not be symmetrized.")
            return False

        tdm = ThermalDisplacementMatrices(frequencies,
                                          eigvecs,
                                          self._primitive.get_masses(),
                                          cutoff_frequency=cutoff_frequency,
                                          lattice=self._primitive.get_cell().T)
        if t_cif is None:
            tdm.set_temperature_range(t_min, t_max, t_step)
        else:
            tdm.set_temperatures([t_cif])
        tdm.run()

        self._thermal_displacement_matrices = tdm
        return True

    def get_thermal_displacement_matrices(self):
        tdm = self._thermal_displacement_matrices
        if tdm is not None:
            return tdm.get_thermal_displacement_matrices()

    def write_yaml_thermal_displacement_matrices(self):
        self._thermal_displacement_matrices.write_yaml()

    def write_thermal_displacement_matrix_to_cif(self,
                                                 temperature_index):
        self._thermal_displacement_matrices.write_cif(self._primitive,
                                                      temperature_index)

    # Mean square distance between a pair of atoms
    def set_thermal_distances(self,
                              atom_pairs,
                              t_step=10,
                              t_max=1000,
                              t_min=0,
                              cutoff_frequency=None):
        """
        atom_pairs: List of list
          Mean square distances are calculated for the atom_pairs
          e.g. [[1, 2], [1, 4]]

        cutoff_frequency:
          phonon modes that have frequencies below cutoff_frequency
          are ignored.
        """

        td = ThermalDistances(self._mesh.get_frequencies(),
                              self._mesh.get_eigenvectors(),
                              self._supercell,
                              self._primitive,
                              self._mesh.get_qpoints(),
                              cutoff_frequency=cutoff_frequency)
        td.set_temperature_range(t_min, t_max, t_step)
        td.run(atom_pairs)

        self._thermal_distances = td

    def write_yaml_thermal_distances(self):
        self._thermal_distances.write_yaml()

    # Sampling at q-points
    def set_qpoints_phonon(self,
                           q_points,
                           nac_q_direction=None,
                           is_eigenvectors=False,
                           write_dynamical_matrices=False):
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._qpoints_phonon = None
            return False

        self._qpoints_phonon = QpointsPhonon(
            np.reshape(q_points, (-1, 3)),
            self._dynamical_matrix,
            nac_q_direction=nac_q_direction,
            is_eigenvectors=is_eigenvectors,
            group_velocity=self._group_velocity,
            write_dynamical_matrices=write_dynamical_matrices,
            factor=self._factor)
        return True

    def get_qpoints_phonon(self):
        return (self._qpoints_phonon.get_frequencies(),
                self._qpoints_phonon.get_eigenvectors())

    def write_hdf5_qpoints_phonon(self):
        self._qpoints_phonon.write_hdf5()

    def write_yaml_qpoints_phonon(self):
        self._qpoints_phonon.write_yaml()

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
            print("Warning: Dynamical matrix has not yet built.")
            return False

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
        if (anime_type == 'arc' or
            anime_type == 'xyz' or
            anime_type == 'jmol' or
            anime_type == 'poscar'):
            if band_index is None or amplitude is None or num_div is None:
                print("Warning: Parameters are not correctly set for "
                      "animation.")
                return False

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

        return True

    # Atomic modulation of normal mode
    def set_modulations(self,
                        dimension,
                        phonon_modes,
                        delta_q=None,
                        derivative_order=None,
                        nac_q_direction=None):
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._modulation = None
            return False

        self._modulation = Modulation(self._dynamical_matrix,
                                      dimension,
                                      phonon_modes,
                                      delta_q=delta_q,
                                      derivative_order=derivative_order,
                                      nac_q_direction=nac_q_direction,
                                      factor=self._factor)
        self._modulation.run()
        return True

    def get_modulated_supercells(self):
        """Returns cells with modulations as Atoms objects"""
        return self._modulation.get_modulated_supercells()

    def get_modulations_and_supercell(self):
        """Return modulations and supercell

        (modulations, supercell)

        modulations: Atomic modulations of supercell in Cartesian coordinates
        supercell: Supercell as an Atoms object.

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
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._irreps = None
            return None

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
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._group_velocity = None
            return False

        self._group_velocity = GroupVelocity(
            self._dynamical_matrix,
            q_length=q_length,
            symmetry=self._primitive_symmetry,
            frequency_factor_to_THz=self._factor)
        return True

    def get_group_velocity(self):
        return self._group_velocity.get_group_velocity()

    def get_group_velocity_at_q(self, q_point):
        if self._group_velocity is None:
            self.set_group_velocity()
        self._group_velocity.set_q_points([q_point])
        return self._group_velocity.get_group_velocity()[0]

    # Moment
    def set_moment(self,
                   order=1,
                   is_projection=False,
                   freq_min=None,
                   freq_max=None):
        if self._mesh is None:
            print("Warning: set_mesh has to be done before set_moment")
            return False
        else:
            if is_projection:
                if self._mesh.get_eigenvectors() is None:
                    print("Warning: Eigenvectors have to be calculated.")
                    return False
                moment = PhononMoment(
                    self._mesh.get_frequencies(),
                    weights=self._mesh.get_weights(),
                    eigenvectors=self._mesh.get_eigenvectors())
            else:
                moment = PhononMoment(
                    self._mesh.get_frequencies(),
                    weights=self._mesh.get_weights())
            if freq_min is not None or freq_max is not None:
                moment.set_frequency_range(freq_min=freq_min,
                                           freq_max=freq_max)
            moment.run(order=order)
            self._moment = moment.get_moment()
            return True

    def get_moment(self):
        return self._moment

    #################
    # Local methods #
    #################
    def _run_force_constants_from_forces(self,
                                         distributed_atom_list=None,
                                         decimals=None,
                                         computation_algorithm="svd"):
        if self._displacement_dataset is not None:
            self._force_constants = get_fc2(
                self._supercell,
                self._symmetry,
                self._displacement_dataset,
                atom_list=distributed_atom_list,
                decimals=decimals,
                computation_algorithm=computation_algorithm)

    def _set_dynamical_matrix(self):
        self._dynamical_matrix = None

        if (self._supercell is None or self._primitive is None):
            print("Bug: Supercell or primitive is not created.")
            return False
        elif self._force_constants is None:
            print("Warning: Force constants are not prepared.")
            return False
        elif self._primitive.get_masses() is None:
            print("Warning: Atomic masses are not correctly set.")
            return False
        else:
            if self._nac_params is None:
                self._dynamical_matrix = DynamicalMatrix(
                    self._supercell,
                    self._primitive,
                    self._force_constants,
                    decimals=self._dynamical_matrix_decimals,
                    symprec=self._symprec)
            else:
                self._dynamical_matrix = DynamicalMatrixNAC(
                    self._supercell,
                    self._primitive,
                    self._force_constants,
                    nac_params=self._nac_params,
                    decimals=self._dynamical_matrix_decimals,
                    symprec=self._symprec)
            return True

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
        supercells = []
        for disp in self._displacement_dataset['first_atoms']:
            positions = self._supercell.get_positions()
            positions[disp['number']] += disp['displacement']
            supercells.append(Atoms(
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
        self._primitive = get_primitive(
            self._supercell, trans_mat, self._symprec)
        num_satom = self._supercell.get_number_of_atoms()
        num_patom = self._primitive.get_number_of_atoms()
        if abs(num_satom * np.linalg.det(trans_mat) - num_patom) < 0.1:
            return True
        else:
            return False
