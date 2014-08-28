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
import StringIO
import numpy as np
import phonopy.interface.vasp as vasp
import phonopy.interface.wien2k as wien2k
from phonopy.structure.symmetry import Symmetry
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.structure.atoms import Atoms
from phonopy.cui.settings import fracval

# Constant
Damping_Factor = 0.25

# Utility to read file with ignoring blank lines
def get_line_ignore_blank(f):
    line = f.readline().strip()
    if line == '':
        line = get_line_ignore_blank(f)
    return line

# Parse FORCE_SETS
def parse_FORCE_SETS(is_translational_invariance=False, filename="FORCE_SETS"):
    f = open(filename, 'r')
    return get_set_of_forces(f, is_translational_invariance)

def parse_FORCE_SETS_from_strings(strings, is_translational_invariance=False):
    return get_set_of_forces(StringIO.StringIO(strings),
                             is_translational_invariance)

def get_set_of_forces(f, is_translational_invariance):
    set_of_forces = []
    num_atom = int(get_line_ignore_blank(f))
    num_displacements = int(get_line_ignore_blank(f))

    for i in range(num_displacements):
        line = get_line_ignore_blank(f)
        atom_number = int(line)
        line = get_line_ignore_blank(f).split()
        displacement = np.array([float(x) for x in line])
        forces_tmp = []
        for j in range(num_atom):
            line = get_line_ignore_blank(f).split()
            forces_tmp.append(np.array([float(x) for x in line]))
        forces_tmp = np.array(forces_tmp, dtype='double')

        if is_translational_invariance:
            forces_tmp -= np.sum(forces_tmp, axis=0) / len(forces_tmp)

        forces = {'number': atom_number - 1,
                  'displacement': displacement,
                  'forces': forces_tmp}
        set_of_forces.append(forces)

    dataset = {'natom': num_atom,
               'first_atoms': set_of_forces}
    
    return dataset

# Parse QPOINTS
def parse_QPOINTS(filename="QPOINTS"):
    f = open(filename, 'r')
    num_qpoints = int(f.readline().strip())
    qpoints = []
    for i in range(num_qpoints):
        qpoints.append([fracval(x) for x in f.readline().strip().split()])
    return np.array(qpoints)

# Write FORCE_SETS for VASP
def write_FORCE_SETS_wien2k(forces_filenames,
                            displacements,
                            supercell,
                            filename='FORCE_SETS',
                            is_zero_point=False,
                            is_distribute=True,
                            symprec=1e-5):

    natom = supercell.get_number_of_atoms()
    lattice = supercell.get_cell()

    for wien2k_filename, disp in zip(forces_filenames,
                                     displacements['first_atoms']):
        # Parse wien2k case.scf file
        wien2k_forces = wien2k.get_forces_wien2k(wien2k_filename, lattice)
        if is_distribute:
            force_set = wien2k.distribute_forces(
                supercell,
                [disp['number'], disp['displacement']],
                wien2k_forces,
                wien2k_filename,
                symprec)
            if not force_set:
                return False
        else:
            if not (natom == len(wien2k_forces)):
                print "%s contains only forces of %d atoms" % (
                    wien2k_filename, len(wien2k_forces))
                return False
            else:
                force_set = wien2k_forces
        
        drift_force = np.sum(force_set, axis=0) / len(force_set)
        print "Drift force of %s" % wien2k_filename
        print "%12.8f %12.8f %12.8f" % tuple(drift_force)
        print "This drift force was subtracted from forces."
        print

        disp['forces'] = np.array(force_set) - drift_force
                
    write_FORCE_SETS(displacements, filename=filename)
    return True


def iterparse(fname, tag=None):
    try:
        from lxml import etree
        return etree.iterparse(fname, tag=tag)
    except ImportError:
        import xml.etree.cElementTree as etree
        def _iter(fname, t):
            for event, elem in etree.iterparse(fname):
                if t is None or elem.tag == t:
                    yield event, elem
        return _iter(fname, tag)


def write_FORCE_SETS_vasp(forces_filenames,
                          displacements,
                          filename='FORCE_SETS',
                          is_zero_point=False,
                          verbose=True):

    if verbose:
        print "counter (file index):",
        
    num_atom = displacements['natom']
    count = 0
    are_files_correct = True
        
    if is_zero_point:
        force_files = forces_filenames[1:]
        if vasp.is_version528(forces_filenames[0]):
            zero_forces = vasp.get_forces_vasprun_xml(iterparse(
                vasp.VasprunWrapper(forces_filenames[0]), tag='varray'))
        else:
            zero_forces = vasp.get_forces_vasprun_xml(
                iterparse(forces_filenames[0], tag='varray'))

        if verbose:
            print "%d" % (count + 1),
        count += 1
            
        if not check_forces(zero_forces, num_atom, forces_filenames[0]):
            are_files_correct = False
    else:
        force_files = forces_filenames
        zero_forces = None

    for i, disp in enumerate(displacements['first_atoms']):
        if vasp.is_version528(force_files[i]):
            disp['forces'] = vasp.get_forces_vasprun_xml(iterparse(
                vasp.VasprunWrapper(force_files[i]), tag='varray'))
        else:
            disp['forces'] = vasp.get_forces_vasprun_xml(
                iterparse(force_files[i], tag='varray'))

        if verbose:
            print "%d" % (count + 1),
        count += 1
        
        if not check_forces(disp['forces'], num_atom, force_files[i]):
            are_files_correct = False

    if verbose:
        print
        
    write_FORCE_SETS(displacements,
                     filename=filename,
                     zero_forces=zero_forces)

    return are_files_correct

def check_forces(forces, num_atom, filename):
    if len(forces) != num_atom:
        print " \"%s\" does not contain necessary information." % filename,
        return False
    else:
        return True

def write_FORCE_SETS(dataset, filename='FORCE_SETS', zero_forces=None):
    num_atom = dataset['natom']
    displacements = dataset['first_atoms']
    forces = [x['forces'] for x in dataset['first_atoms']]
    
    # Write FORCE_SETS
    fp = open(filename, 'w')
    fp.write("%-5d\n" % num_atom)
    fp.write("%-5d\n" % len(displacements))
    for count, disp in enumerate(displacements):
        fp.write("\n%-5d\n" % (disp['number'] + 1))
        fp.write("%20.16f %20.16f %20.16f\n" % (tuple(disp['displacement'])))

        # Subtract residual forces
        if zero_forces is not None:
            forces[count] -= zero_forces

        for f in forces[count]:
            fp.write("%15.10f %15.10f %15.10f\n" % (tuple(f)))

def mycmp(x, y):
    return cmp(x[0], y[0])

def parse_disp_yaml(filename="disp.yaml", return_cell=False):
    try:
        import yaml
    except ImportError:
        print "You need to install python-yaml."
        exit(1)
        
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    dataset = yaml.load(open(filename), Loader=Loader)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    new_first_atoms = []
    for first_atoms in dataset['displacements']:
        first_atoms['atom'] -= 1
        atom1 = first_atoms['atom']
        disp1 = first_atoms['displacement']
        if 'direction' in first_atoms:
            direct1 = first_atoms['direction']
            new_first_atoms.append({'number': atom1,
                                    'displacement': disp1,
                                    'direction':direct1})
        else:
            new_first_atoms.append({'number': atom1, 'displacement': disp1})
    new_dataset['first_atoms'] = new_first_atoms
    
    if return_cell:
        lattice = dataset['lattice']
        positions = [x['position'] for x in dataset['atoms']]
        symbols = [x['symbol'] for x in dataset['atoms']]
        cell = Atoms(cell=lattice,
                     scaled_positions=positions,
                     symbols=symbols,
                     pbc=True)
        return new_dataset, cell
    else:
        return new_dataset

def parse_DISP(filename='DISP'):
    disp = open(filename)
    displacements = []
    for line in disp:
        if line.strip() != '':
            a = line.split()
            displacements.append(
                [int(a[0])-1, float(a[1]), float(a[2]), float(a[3])])
    return displacements

# Write disp.yaml
def write_disp_yaml(displacements, supercell, directions=None,
                    filename='disp.yaml'):
    file = open(filename, 'w')
    file.write("natom: %4d\n" % supercell.get_number_of_atoms())
    file.write("displacements:\n")
    for i, disp in enumerate(displacements):
        file.write("- atom: %4d\n" % (disp[0] + 1))
        if directions is not None:
            file.write("  direction:\n")
            file.write("    [ %20.16f,%20.16f,%20.16f ]\n" % tuple(directions[i][1:4]))
        file.write("  displacement:\n")
        file.write("    [ %20.16f,%20.16f,%20.16f ]\n" % tuple(disp[1:4]))
            
    file.write("lattice:\n")
    for axis in supercell.get_cell():
        file.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    file.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        file.write("- symbol: %-2s # %d\n" % (s, i+1))
        file.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % \
                       (v[0], v[1], v[2]))
    file.close()

# Write FORCE_CONSTANTS
def write_FORCE_CONSTANTS(force_constants, filename='FORCE_CONSTANTS'):
    w = open(filename, 'w')
    fc_shape = force_constants.shape
    w.write("%4d\n" % (fc_shape[0]))
    for i in range(fc_shape[0]):
        for j in range(fc_shape[1]):
            w.write("%4d%4d\n" % (i+1, j+1))
            for vec in force_constants[i][j]:
                w.write(("%22.15f"*3 + "\n") % tuple(vec))
    w.close()

def write_force_constants_to_hdf5(force_constants,
                                  filename='force_constants.hdf5'):
    import h5py
    w = h5py.File(filename, 'w')
    w.create_dataset('force_constants', data=force_constants)
    w.close()

# Read FORCE_CONSTANTS
def parse_FORCE_CONSTANTS(filename):
    fcfile = open(filename)
    num = int((fcfile.readline().strip().split())[0])
    force_constants = np.zeros((num, num, 3, 3), dtype=float)
    for i in range(num):
        for j in range(num):
            fcfile.readline()
            tensor = []
            for k in range(3):
                tensor.append([float(x) for x in fcfile.readline().strip().split()])
            force_constants[i, j] = np.array(tensor)

    return force_constants


def read_force_constant_vasprun_xml(filename):
    if vasp.is_version528(filename):
        vasprun = iterparse(vasp.VasprunWrapper(filename))
    else:
        vasprun = iterparse(filename)
    return vasp.get_force_constants_vasprun_xml(vasprun)

def read_force_constant_OUTCAR(filename):
    return vasp.get_force_constants_OUTCAR(filename)

def read_force_constants_hdf5(filename="force_constants.hdf5"):
    import h5py
    f = h5py.File(filename)
    return f[f.keys()[0]][:]

# Read BORN
def parse_BORN(primitive, symprec=1e-5, is_symmetry=True, filename="BORN"):
    f = open(filename, 'r')
    symmetry = Symmetry(primitive, symprec=symprec, is_symmetry=is_symmetry)
    return get_born_parameters(f, primitive, symmetry)

def parse_BORN_from_strings(strings, primitive, symprec=1e-5, is_symmetry=True):
    f = StringIO.StringIO(strings)
    symmetry = Symmetry(primitive, symprec=symprec, is_symmetry=is_symmetry)
    return get_born_parameters(f, primitive, symmetry)

def get_born_parameters(f, primitive, symmetry):
    # Read unit conversion factor, damping factor, ...
    factors = [float(x) for x in f.readline().split()]
    if len(factors) < 1:
        print "BORN file format of line 1 is incorrect"
        return False
    if len(factors) < 2:
        factors = factors[0]

    # Read dielectric constant
    line = f.readline().split()
    if not len(line) == 9:
        print "BORN file format of line 2 is incorrect"
        return False
    dielectric = np.reshape([float(x) for x in line], (3, 3))
    
    # Read Born effective charge
    independent_atoms = symmetry.get_independent_atoms()
    born = np.zeros((primitive.get_number_of_atoms(), 3, 3), dtype=float)

    for i in independent_atoms:
        line = f.readline().split()
        if len(line) == 0:
            print "Number of lines for Born effect charge is not enough."
            return False
        if not len(line) == 9:
            print "BORN file format of line %d is incorrect" % (i + 3)
            return False
        born[i] = np.reshape([float(x) for x in line], (3, 3))

    # Expand Born effective charges to all atoms in the primitive cell
    rotations = symmetry.get_symmetry_operations()['rotations']
    map_operations = symmetry.get_map_operations()
    map_atoms = symmetry.get_map_atoms()

    for i in range(primitive.get_number_of_atoms()):
        # R_cart = L R L^-1
        rot_cartesian = similarity_transformation(
            primitive.get_cell().transpose(), rotations[map_operations[i]])
        # R_cart^T B R_cart^-T (inverse rotation is required to transform)
        born[i] = similarity_transformation(rot_cartesian.transpose(),
                                            born[map_atoms[i]])

    non_anal = {'born': born,
                'factor': factors,
                'dielectric': dielectric }

    return non_anal

