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

import sys
import numpy as np
from phonopy.structure.atoms import Atoms
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import get_supercell, get_primitive
from phonopy.harmonic.displacement import get_least_displacements, \
     direction_to_displacement
from phonopy.harmonic.force_constants import get_fc2, \
     symmetrize_force_constants, rotational_invariance, \
     cutoff_force_constants, set_tensor_symmetry
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import BandStructure
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.phonon.mesh import Mesh
from phonopy.units import VaspToTHz
from phonopy.phonon.dos import TotalDos, PartialDos
from phonopy.phonon.thermal_displacement import ThermalDisplacements, \
     ThermalDistances, ThermalDisplacementMatrices
from phonopy.phonon.animation import Animation
from phonopy.phonon.modulation import Modulation
from phonopy.phonon.qpoints_mode import QpointsPhonon
from phonopy.phonon.irreps import IrReps
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh

class Phonopy:
    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 primitive_matrix=None,
                 nac_params=None,
                 distance=0.01,
                 factor=VaspToTHz,
                 is_auto_displacements=True,
                 dynamical_matrix_decimals=None,
                 force_constants_decimals=None,
                 symprec=1e-5,
                 is_symmetry=True,
                 log_level=0):
        self._symprec = symprec
        self._factor = factor
        self._is_symmetry = is_symmetry
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = unitcell
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
        if is_auto_displacements:
            self.generate_displacements(distance=distance)

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

    def set_post_process(self,
                         primitive_matrix=None,
                         sets_of_forces=None,
                         displacement_dataset=None,
                         force_constants=None,
                         is_nac=None):
        print 
        print ("********************************** Warning"
               "**********************************")
        print "set_post_process will be obsolete."
        print ("  produce_force_constants is used instead of set_post_process"
               " for producing")
        print ("  force constants from forces.")
        if primitive_matrix is not None:
            print ("  primitive_matrix has to be given at Phonopy::__init__"
                   " object creation.")
        print ("******************************************"
               "**********************************")
        print 

        if primitive_matrix is not None:
            self._primitive_matrix = primitive_matrix
            self._build_primitive_cell()
            self._search_primitive_symmetry()
        
        if sets_of_forces is not None:
            self.set_forces(sets_of_forces)
        elif displacement_dataset is not None:
            self._displacement_dataset = displacement_dataset
        elif force_constants is not None:
            self.set_force_constants(force_constants)
            
        if self._displacement_dataset is not None:
            self.produce_force_constants()

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
        
    def get_primitive(self):
        return self._primitive
    primitive = property(get_primitive)

    def get_unitcell(self):
        return self._unitcell
    unitcell = property(get_unitcell)

    def get_supercell(self):
        return self._supercell
    supercell = property(get_supercell)

    def set_supercell(self, supercell):
        self._supercell = supercell

    def get_symmetry(self):
        """return symmetry of supercell"""
        return self._symmetry
    symmetry = property(get_symmetry)

    def get_primitive_symmetry(self):
        """return symmetry of primitive cell"""
        return self._primitive_symmetry

    def get_unit_conversion_factor(self):
        return self._factor
    unit_conversion_factor = property(get_unit_conversion_factor)

    def produce_force_constants(self,
                                forces=None,
                                calculate_full_force_constants=True,
                                computation_algorithm="svd"):
        if forces is not None:
            self.set_forces(forces)
        
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

    def set_nac_params(self, nac_params=None, method=None):
        if method is not None:
            print "set_nac_params:"
            print "  Keyword argument of \"method\" is not more supported."
        self._nac_params = nac_params
        
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

    def set_displacements(self, displacements):
        print 
        print ("********************************** Warning"
               "**********************************")
        print "set_displacements is obsolete. Do nothing."
        print ("******************************************"
               "**********************************")
        print 

    def get_displacements(self):
        return self._displacements
    displacements = property(get_displacements)

    def get_displacement_directions(self):
        return self._displacement_directions
    displacement_directions = property(get_displacement_directions)

    def get_displacement_dataset(self):
        return self._displacement_dataset

    def get_supercells_with_displacements(self):
        if self._displacement_dataset is None:
            return None
        else:
            self._build_supercells_with_displacements()
            return self._supercells_with_displacements

    def get_dynamical_matrix(self):
        return self._dynamical_matrix
    dynamical_matrix = property(get_dynamical_matrix)

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

    def set_force_constants_zero_with_radius(self, cutoff_radius):
        cutoff_force_constants(self._force_constants,
                               self._supercell,
                               cutoff_radius,
                               symprec=self._symprec)

    def set_force_constants(self, force_constants):
        self._force_constants = force_constants

    def set_force_sets(self, force_sets):
        print 
        print ("********************************** Warning"
               "**********************************")
        print "set_force_sets will be obsolete."
        print ("   The method name is changed to set_displacement_dataset.")
        print ("******************************************"
               "**********************************")
        print
        self.set_displacement_dataset(force_sets)

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
        
    def symmetrize_force_constants(self, iteration=3):
        symmetrize_force_constants(self._force_constants, iteration)

    def symmetrize_force_constants_by_space_group(self):
        rotations = self._symmetry.get_symmetry_operations()['rotations']
        translations = self._symmetry.get_symmetry_operations()['translations']
        set_tensor_symmetry(self._force_constants,
                            self._supercell.get_cell().T,
                            self._supercell.get_scaled_positions(),
                            rotations,
                            translations,
                            self._symprec)
        
    def get_force_constants(self):
        return self._force_constants
    force_constants = property(get_force_constants)

    def get_rotational_condition_of_fc(self):
        return rotational_invariance(self._force_constants,
                                     self._supercell,
                                     self._primitive,
                                     self._symprec)

    def set_dynamical_matrix(self):
        self._set_dynamical_matrix()
        
    def get_dynamical_matrix_at_q(self, q):
        self._set_dynamical_matrix()
        self._dynamical_matrix.set_dynamical_matrix(q)
        return self._dynamical_matrix.get_dynamical_matrix()

    def get_frequencies(self, q):
        """
        Calculate phonon frequencies at q
        
        q: q-vector in reduced coordinates of primitive cell
        """
        self._set_dynamical_matrix()
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

    def set_band_structure(self,
                           bands,
                           is_eigenvectors=False,
                           is_band_connection=False):
        self._set_dynamical_matrix()
        self._band_structure = BandStructure(
            bands,
            self._dynamical_matrix,
            is_eigenvectors=is_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=self._group_velocity,
            factor=self._factor)

    def get_band_structure(self):
        band = self._band_structure
        return (band.get_qpoints(),
                band.get_distances(),                
                band.get_frequencies(),
                band.get_eigenvectors())

    def plot_band_structure(self, symbols=None):
        return self._band_structure.plot_band(symbols)

    def write_yaml_band_structure(self):
        self._band_structure.write_yaml()

    def set_mesh(self,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False):
        self._set_dynamical_matrix()
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
            factor=self._factor)

    def get_mesh(self):
        return (self._mesh.get_qpoints(),
                self._mesh.get_weights(),
                self._mesh.get_frequencies(),
                self._mesh.get_eigenvectors())

    def write_yaml_mesh(self):
        self._mesh.write_yaml()

    def set_thermal_properties(self,
                               t_step=10,
                               t_max=1000,
                               t_min=0,
                               is_projection=False,
                               band_indices=None,
                               cutoff_frequency=None):
        if self._mesh is None:
            print "set_mesh has to be done before set_thermal_properties"
            return False
        else:
            tp = ThermalProperties(self._mesh.get_frequencies(),
                                   weights=self._mesh.get_weights(),
                                   eigenvectors=self._mesh.get_eigenvectors(),
                                   is_projection=is_projection,
                                   band_indices=band_indices,
                                   cutoff_frequency=cutoff_frequency)
            tp.set_thermal_properties(t_step=t_step,
                                      t_max=t_max,
                                      t_min=t_min)
            self._thermal_properties = tp

    def get_thermal_properties(self):
        temps, fe, entropy, cv = \
            self._thermal_properties.get_thermal_properties()
        return temps, fe, entropy, cv

    def plot_thermal_properties(self):
        return self._thermal_properties.plot_thermal_properties()

    def write_yaml_thermal_properties(self, filename='thermal_properties.yaml'):
        self._thermal_properties.write_yaml(filename=filename)

    def set_partial_DOS(self,
                        sigma=None,
                        freq_min=None,
                        freq_max=None,
                        freq_pitch=None,
                        tetrahedron_method=False,
                        direction=None):
        if self._mesh is None:
            print "set_mesh has to be done before set_thermal_properties"
            sys.exit(1)
        if self._mesh.get_eigenvectors() is None:
            print "Eigenvectors have to be calculated."
            sys.exit(1)
        if direction is not None:
            direction_cart = np.dot(direction, self._primitive.get_cell())
        else:
            direction_cart = None
        pdos = PartialDos(self._mesh,
                          sigma=sigma,
                          tetrahedron_method=tetrahedron_method,
                          direction=direction_cart)
        pdos.set_draw_area(freq_min, freq_max, freq_pitch)
        pdos.run()
        self._pdos = pdos

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
        return self._pdos.plot_pdos(indices=pdos_indices,
                                    legend=legend)

    def write_partial_DOS(self):
        self._pdos.write()

    def set_total_DOS(self,
                      sigma=None,
                      freq_min=None,
                      freq_max=None,
                      freq_pitch=None,
                      tetrahedron_method=False):

        if self._mesh is None:
            print "set_mesh has to be done before set_thermal_properties"
            sys.exit(1)

        total_dos = TotalDos(self._mesh,
                             sigma=sigma,
                             tetrahedron_method=tetrahedron_method)
        total_dos.set_draw_area(freq_min, freq_max, freq_pitch)
        total_dos.run()
        self._total_dos = total_dos

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
        return self._total_dos.plot_dos()

    def write_total_DOS(self):
        self._total_dos.write()

    def set_thermal_displacements(self,
                                  t_step=10,
                                  t_max=1000,
                                  t_min=0,
                                  direction=None,
                                  cutoff_frequency=None):
        """
        cutoff_frequency:
          phonon modes that have frequencies below cutoff_frequency
          are ignored.

        direction:
          Projection direction in reduced coordinates
        """
        if self._mesh is None:
            print "set_mesh has to be done before set_thermal_properties"
            sys.exit(1)

        eigvecs = self._mesh.get_eigenvectors()
        frequencies = self._mesh.get_frequencies()
        mesh_nums = self._mesh.get_mesh_numbers() 

        if self._mesh.get_eigenvectors() is None:
            print "Eigenvectors have to be calculated."
            sys.exit(1)
            
        if np.prod(mesh_nums) != len(eigvecs):
            print "Sampling mesh must not be symmetrized."
            sys.exit(1)

        td = ThermalDisplacements(frequencies,
                                  eigvecs,
                                  self._primitive.get_masses(),
                                  cutoff_frequency=cutoff_frequency)
        td.set_temperature_range(t_min, t_max, t_step)
        if direction is not None:
            td.project_eigenvectors(direction, self._primitive.get_cell())
        td.run()
        
        self._thermal_displacements = td

    def get_thermal_displacements(self):
        if self._thermal_displacements is not None:
            return self._thermal_displacements.get_thermal_displacements()
        
    def plot_thermal_displacements(self, is_legend=False):
        return self._thermal_displacements.plot(is_legend)

    def write_yaml_thermal_displacements(self):
        self._thermal_displacements.write_yaml()

    def set_thermal_displacement_matrices(self,
                                           t_step=10,
                                           t_max=1000,
                                           t_min=0,
                                           cutoff_frequency=None):
        """
        cutoff_frequency:
          phonon modes that have frequencies below cutoff_frequency
          are ignored.

        direction:
          Projection direction in reduced coordinates
        """
        if self._mesh is None:
            print "set_mesh has to be done before set_thermal_properties"
            sys.exit(1)

        eigvecs = self._mesh.get_eigenvectors()
        frequencies = self._mesh.get_frequencies()
        mesh_nums = self._mesh.get_mesh_numbers() 

        if self._mesh.get_eigenvectors() is None:
            print "Eigenvectors have to be calculated."
            sys.exit(1)
            
        if np.prod(mesh_nums) != len(eigvecs):
            print "Sampling mesh must not be symmetrized."
            sys.exit(1)

        tdm = ThermalDisplacementMatrices(frequencies,
                                           eigvecs,
                                           self._primitive.get_masses(),
                                           cutoff_frequency=cutoff_frequency)
        tdm.set_temperature_range(t_min, t_max, t_step)
        tdm.run()
        
        self._thermal_displacement_matrices = tdm

    def get_thermal_displacement_matrices(self):
        if self._thermal_displacement_matrices is not None:
            return self._thermal_displacement_matrices.get_thermal_displacement_matrices()
        
    def write_yaml_thermal_displacement_matrices(self):
        self._thermal_displacement_matrices.write_yaml()
        
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

    def set_qpoints_phonon(self,
                           q_points,
                           nac_q_direction=None,
                           is_eigenvectors=False,
                           write_dynamical_matrices=False,
                           factor=VaspToTHz):
        self._set_dynamical_matrix()
        self._qpoints_phonon = QpointsPhonon(
            q_points,
            self._dynamical_matrix,
            nac_q_direction=nac_q_direction,
            is_eigenvectors=is_eigenvectors,
            group_velocity=self._group_velocity,
            write_dynamical_matrices=write_dynamical_matrices,
            factor=self._factor)
        
    def get_qpoints_phonon(self):
        return (self._qpoints_phonon.get_frequencies(),
                self._qpoints_phonon.get_eigenvectors())
    
    def write_yaml_qpoints_phonon(self):
        self._qpoints_phonon.write_yaml()

    def write_animation(self,
                        q_point=None,
                        anime_type='v_sim',
                        band_index=None,
                        amplitude=None,
                        num_div=None,
                        shift=None,
                        filename=None):
        self._set_dynamical_matrix()
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
                print "Parameters are not correctly set for animation."
                sys.exit(1)

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
                    
    def set_modulations(self,
                        dimension,
                        phonon_modes,
                        delta_q=None,
                        derivative_order=None,
                        nac_q_direction=None):
        self._set_dynamical_matrix()
        self._modulation = Modulation(self._dynamical_matrix,
                                      dimension,
                                      phonon_modes,
                                      delta_q=delta_q,
                                      derivative_order=derivative_order,
                                      nac_q_direction=nac_q_direction,
                                      factor=self._factor)
        self._modulation.run()
                    
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

    def set_irreps(self,
                   q,
                   is_little_cogroup=False,
                   nac_q_direction=None,
                   degeneracy_tolerance=1e-4):
        self._set_dynamical_matrix()
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

    def set_group_velocity(self, q_length=None):
        self._set_dynamical_matrix()
        self._group_velocity = GroupVelocity(
            self._dynamical_matrix,
            q_length=q_length,
            symmetry=self._primitive_symmetry,
            frequency_factor_to_THz=self._factor)

    def get_group_velocity(self):
        return self._group_velocity.get_group_velocity()
        
    def get_group_velocity_at_q(self, q_point):
        if self._group_velocity is None:
            self.set_group_velocity()
        self._group_velocity.set_q_points([q_point])
        return self._group_velocity.get_group_velocity()[0]

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
            print ("Warning: point group symmetries of supercell and primitive"
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



from phonopy.gruneisen.mesh import Mesh as GruneisenMesh
from phonopy.gruneisen.band_structure import BandStructure as GruneisenBand

class PhonopyGruneisen:
    def __init__(self,
                 phonon,
                 phonon_plus,
                 phonon_minus):
        self._phonon = phonon
        self._phonon_plus = phonon_plus
        self._phonon_minus = phonon_minus

        self._mesh = None
        self._band_structure = None
        
    def set_mesh(self,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_gamma_center=False,
                 is_mesh_symmetry=True):
        self._mesh = GruneisenMesh(self._phonon,
                                   self._phonon_plus,
                                   self._phonon_minus,
                                   mesh,
                                   shift=shift,
                                   is_time_reversal=is_time_reversal,
                                   is_gamma_center=is_gamma_center,
                                   is_mesh_symmetry=is_mesh_symmetry)

    def write_yaml_mesh(self):
        self._mesh.write_yaml()

    def plot_mesh(self,
                  cutoff_frequency=None,
                  color_scheme=None,
                  marker='o',
                  markersize=None):
        return self._mesh.plot(cutoff_frequency=cutoff_frequency,
                               color_scheme=color_scheme,
                               marker=marker,
                               markersize=markersize)

    def set_band_structure(self,
                           paths,
                           num_points):
        self._band_structure = GruneisenBand(self._phonon,
                                             self._phonon_plus,
                                             self._phonon_minus,
                                             paths,
                                             num_points)

    def write_yaml_band_structure(self):
        self._band_structure.write_yaml()

    def plot_band_structure(self,
                            epsilon=1e-4,
                            color_scheme=None):
        return self._band_structure.plot(epsilon=epsilon,
                                         color_scheme=color_scheme)
        


from phonopy.qha import *
from phonopy.units import EvTokJmol, EVAngstromToGPa

class PhonopyQHA:
    def __init__(self,
                 volumes,
                 electronic_energies,
                 eos='vinet',
                 temperatures=None,
                 free_energy=None,
                 cv=None,
                 entropy=None,
                 t_max=None,
                 verbose=False):
        """
        The following two have the same number of elements
          volumes: Unit cell volumes (V) in Angstrom^3
          electronic_energies: Electronic energies (U) in eV

        The following four have the same number of elements
          temperatures: Temperatures ascending order (T) in K
          cv: Heat capacity at constant volume in J/K/mol
          entropy: Entropy at constant volume (S) J/K/mol
          free_energy: Helmholtz free energy (F) kJ/mol

        eos: Equation of state used for fitting F vs V
             'vinet', 'murnaghan' or 'birch_murnaghan'
        tmax: Maximum temperature to be calculated. This has to be not
              greater than the temperature of the third element from the
              end of 'temperatre' elements. If max_t=None, the temperature
              of the third element from the end is used.
        """
        self._bulk_modulus = BulkModulus(volumes,
                                         electronic_energies,
                                         eos=eos)

        if temperatures is not None:
            self._qha = QHA(volumes,
                            electronic_energies,
                            temperatures,
                            cv,
                            entropy,
                            free_energy,
                            eos=eos,
                            t_max=t_max)
            self._qha.run(verbose=verbose)

    def get_bulk_modulus(self):
        return self._bulk_modulus.get_bulk_modulus()
            
    def get_bulk_modulus_parameters(self):
        """Returns bulk modulus
        (lowest energy,
         bulk modulus,
         b_prime,
         equilibrium volume)
        """
        return self._bulk_modulus.get_parameters()
    
    def plot_bulk_modulus(self):
        return self._bulk_modulus.plot()

    def plot_qha(self, thin_number=10, volume_temp_exp=None):
        return self._qha.plot(thin_number=thin_number,
                              volume_temp_exp=volume_temp_exp)

    def get_helmholtz_volume(self):
        """Returns free_energies
        
        free_energies: Free energies calculated at temperatures and volumes
                       [temperatures][volumes]
        """
        return self._qha.get_helmholtz_volume()

    def plot_helmholtz_volume(self, thin_number=10):
        return self._qha.plot_helmholtz_volume(thin_number=thin_number)

    def plot_pdf_helmholtz_volume(self,
                                  thin_number=10,
                                  filename='helmholtz-volume.pdf'):
        self._qha.plot_pdf_helmholtz_volume(thin_number=thin_number,
                                            filename=filename)

    def write_helmholtz_volume(self, filename='helmholtz-volume.dat'):
        self._qha.write_helmholtz_volume(filename=filename)

    def get_volume_temperature(self):
        """Returns volumes at temperatures"""
        return self._qha.get_volume_temperature()

    def plot_volume_temperature(self, exp_data=None):
        return self._qha.plot_volume_temperature(exp_data=exp_data)

    def plot_pdf_volume_temperature(self,
                                    exp_data=None,
                                    filename='volume-temperature.pdf'):
        self._qha.plot_pdf_volume_temperature(exp_data=exp_data,
                                              filename=filename)

    def write_volume_temperature(self, filename='volume-temperature.dat'):
        self._qha.write_volume_temperature(filename=filename)

    def get_thermal_expansion(self):
        """Returns thermal expansion coefficients at temperatures"""
        return self._qha.get_thermal_expansion()

    def plot_thermal_expansion(self):
        return self._qha.plot_thermal_expansion()

    def plot_pdf_thermal_expansion(self,
                                   filename='thermal_expansion.pdf'):
        self._qha.plot_pdf_thermal_expansion(filename=filename)

    def write_thermal_expansion(self,
                                filename='thermal_expansion.dat'):
        self._qha.write_thermal_expansion(filename=filename)

    def get_volume_expansion(self):
        """Return volume expansions at temperatures"""
        return self._qha.get_volume_expansion()

    def plot_volume_expansion(self, exp_data=None, symbol='o'):
        return self._qha.plot_volume_expansion(exp_data=exp_data,
                                               symbol=symbol)

    def plot_pdf_volume_expansion(self,
                                  exp_data=None,
                                  symbol='o',
                                  filename='volume_expansion.pdf'):
        self._qha.plot_pdf_volume_expansion(exp_data=exp_data,
                                            symbol=symbol,
                                            filename=filename)

    def write_volume_expansion(self, filename='volume_expansion.dat'):
        self._qha.write_volume_expansion(filename=filename)

    def get_gibbs_temperature(self):
        """Returns Gibbs free energies at temperatures"""
        return self._qha.get_gibbs_temperature()

    def plot_gibbs_temperature(self):
        return self._qha.plot_gibbs_temperature()

    def plot_pdf_gibbs_temperature(self, filename='gibbs-temperature.pdf'):
        self._qha.plot_pdf_gibbs_temperature(filename=filename)

    def write_gibbs_temperature(self, filename='gibbs-temperature.dat'):
        self._qha.write_gibbs_temperature(filename=filename)

    def get_bulk_modulus_temperature(self):
        """Returns bulk modulus at temperatures"""
        return self._qha.get_bulk_modulus_temperature()

    def plot_bulk_modulus_temperature(self):
        return self._qha.plot_bulk_modulus_temperature()

    def plot_pdf_bulk_modulus_temperature(self,
                                          filename='bulk_modulus-temperature.pdf'):
        self._qha.plot_pdf_bulk_modulus_temperature(filename=filename)

    def write_bulk_modulus_temperature(self,
                                       filename='bulk_modulus-temperature.dat'):
        self._qha.write_bulk_modulus_temperature(filename=filename)

    def get_heat_capacity_P_numerical(self):
        """Returns heat capacities at constant pressure at temperatures
        
        These values are calculated by -T*d^2G/dT^2.
        """
        return self._qha.get_heat_capacity_P_numerical()

    def plot_heat_capacity_P_numerical(self, exp_data=None):
        return self._qha.plot_heat_capacity_P_numerical(exp_data=exp_data)

    def plot_pdf_heat_capacity_P_numerical(self,
                                           exp_data=None,
                                           filename='Cp-temperature.pdf'):
        self._qha.plot_pdf_heat_capacity_P_numerical(exp_data=exp_data,
                                                     filename=filename)

    def write_heat_capacity_P_numerical(self, filename='Cp-temperature.dat'):
        self._qha.write_heat_capacity_P_numerical(filename=filename)

    def get_heat_capacity_P_polyfit(self):
        """Returns heat capacities at constant pressure at temperatures

        These values are calculated from the values obtained by polynomial
        fittings of Cv and S.
        """
        return self._qha.get_heat_capacity_P_polyfit()

    def plot_heat_capacity_P_polyfit(self, exp_data=None):
        return self._qha.plot_heat_capacity_P_polyfit(exp_data=exp_data)

    def plot_pdf_heat_capacity_P_polyfit(self,
                                         exp_data=None,
                                         filename='Cp-temperature_polyfit.pdf'):
        self._qha.plot_pdf_heat_capacity_P_polyfit(exp_data=exp_data,
                                                   filename=filename)

    def write_heat_capacity_P_polyfit(self,
                                      filename='Cp-temperature_polyfit.dat',
                                      filename_ev='entropy-volume.dat',
                                      filename_cvv='Cv-volume.dat',
                                      filename_dsdvt='dsdv-temperature.dat'):
        self._qha.write_heat_capacity_P_polyfit(filename=filename,
                                                filename_ev=filename_ev,
                                                filename_cvv=filename_cvv,
                                                filename_dsdvt=filename_dsdvt)

    def get_gruneisen_temperature(self):
        """Returns Gruneisen parameters at temperatures"""
        return self._qha.get_gruneisen_temperature()

    def plot_gruneisen_temperature(self):
        return self._qha.plot_gruneisen_temperature()

    def plot_pdf_gruneisen_temperature(self,
                                       filename='gruneisen-temperature.pdf'):
        self._qha.plot_pdf_gruneisen_temperature(filename=filename)

    def write_gruneisen_temperature(self, filename='gruneisen-temperature.dat'):
        self._qha.write_gruneisen_temperature(filename=filename)
