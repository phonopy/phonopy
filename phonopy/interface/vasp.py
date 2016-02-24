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
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
from phonopy.structure.atoms import Atoms, symbol_map, atom_data
from phonopy.structure.cells import get_primitive, get_supercell
from phonopy.structure.symmetry import Symmetry
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.file_IO import write_FORCE_SETS, write_force_constants_to_hdf5, write_FORCE_CONSTANTS

def parse_set_of_forces(num_atoms,
                        forces_filenames,
                        is_zero_point=False,
                        verbose=True):
    if verbose:
        sys.stdout.write("counter (file index): ")
        
    count = 0
    is_parsed = True
    force_sets = []
        
    if is_zero_point:
        force_files = forces_filenames[1:]
        if _is_version528(forces_filenames[0]):
            zero_forces = _get_forces_vasprun_xml(_iterparse(
                VasprunWrapper(forces_filenames[0]), tag='varray'))
        else:
            zero_forces = _get_forces_vasprun_xml(
                _iterparse(forces_filenames[0], tag='varray'))

        if verbose:
            sys.stdout.write("%d " % (count + 1))
        count += 1
            
        if not _check_forces(zero_forces, num_atoms, forces_filenames[0]):
            is_parsed = False
    else:
        force_files = forces_filenames

    for filename in force_files:
        if _is_version528(filename):
            force_sets.append(_get_forces_vasprun_xml(_iterparse(
                VasprunWrapper(filename), tag='varray')))
        else:
            force_sets.append(_get_forces_vasprun_xml(
                _iterparse(filename, tag='varray')))

        if is_zero_point:
            force_sets[-1] -= zero_forces

        if verbose:
            sys.stdout.write("%d " % (count + 1))
        count += 1
        
        if not _check_forces(force_sets[-1], num_atoms, filename):
            is_parsed = False

    if verbose:
        print('')

    if is_parsed:
        return force_sets
    else:
        return []

def _check_forces(forces, num_atom, filename):
    if len(forces) != num_atom:
        sys.stdout.write(" \"%s\" does not contain necessary information. " %
                         filename)
        return False
    else:
        return True

def create_FORCE_CONSTANTS(filename, options, log_level):
    fc_and_atom_types = read_force_constant_vasprun_xml(filename)
    if not fc_and_atom_types:
        print('')
        print("\'%s\' dones not contain necessary information." % filename)
        return 1

    force_constants, atom_types = fc_and_atom_types
    if options.is_hdf5:
        try:
            import h5py
        except ImportError:
            print('')
            print("You need to install python-h5py.")
            return 1
    
        write_force_constants_to_hdf5(force_constants)
        if log_level > 0:
            print("force_constants.hdf5 has been created from vasprun.xml.")
    else:
        write_FORCE_CONSTANTS(force_constants)
        if log_level > 0:
            print("FORCE_CONSTANTS has been created from vasprun.xml.")

    if log_level > 0:
        print("Atom types:", atom_types)
    return 0
        
#
# read VASP POSCAR
#
def read_vasp(filename, symbols=None):
    lines = open(filename).readlines()
    return _get_atoms_from_poscar(lines, symbols)

def read_vasp_from_strings(strings, symbols=None):
    return _get_atoms_from_poscar(
        StringIO.StringIO(strings).readlines(), symbols)

def _get_atoms_from_poscar(lines, symbols):
    line1 = [x for x in lines[0].split()]
    if _is_exist_symbols(line1):
        symbols = line1

    scale = float(lines[1])

    cell = []
    for i in range(2, 5):
        cell.append([float(x) for x in lines[i].split()[:3]])
    cell = np.array(cell) * scale

    try:
        num_atoms = np.array([int(x) for x in lines[5].split()])
        line_at = 6
    except ValueError:
        symbols = [x for x in lines[5].split()]
        num_atoms = np.array([int(x) for x in lines[6].split()])
        line_at = 7
    
    expaned_symbols = _expand_symbols(num_atoms, symbols)

    if lines[line_at][0].lower() == 's':
        line_at += 1

    is_scaled = True
    if (lines[line_at][0].lower() == 'c' or
        lines[line_at][0].lower() == 'k'):
        is_scaled = False

    line_at += 1

    positions = []
    for i in range(line_at, line_at + num_atoms.sum()):
        positions.append([float(x) for x in lines[i].split()[:3]])

    if is_scaled:
        atoms = Atoms(symbols=expaned_symbols,
                      cell=cell,
                      scaled_positions=positions)
    else:
        atoms = Atoms(symbols=expaned_symbols,
                      cell=cell,
                      positions=positions)
        
    return atoms
                   
def _is_exist_symbols(symbols):
    for s in symbols:
        if not (s in symbol_map):
            return False
    return True

def _expand_symbols(num_atoms, symbols=None):
    expanded_symbols = []
    is_symbols = True
    if symbols is None:
        is_symbols = False
    else:
        if len(symbols) != len(num_atoms):
            is_symbols = False
        else:
            for s in symbols:
                if not s in symbol_map:
                    is_symbols = False
                    break
    
    if is_symbols:
        for s, num in zip(symbols, num_atoms):
            expanded_symbols += [s] * num
    else:
        for i, num in enumerate(num_atoms):
            expanded_symbols += [atom_data[i+1][1]] * num

    return expanded_symbols

#
# write vasp POSCAR
#
def write_vasp(filename, atoms, direct=True):
    lines = _get_vasp_structure(atoms, direct=direct)
    f = open(filename, 'w')
    f.write(lines)

def write_supercells_with_displacements(supercell,
                                        cells_with_displacements):
    write_vasp("SPOSCAR", supercell, direct=True)
    for i, cell in enumerate(cells_with_displacements):
        write_vasp('POSCAR-%03d' % (i + 1), cell, direct=True)

    _write_magnetic_moments(supercell)

def _write_magnetic_moments(cell):
    magmoms = cell.get_magnetic_moments() 
    if magmoms is not  None:
        w = open("MAGMOM", 'w')
        (num_atoms,
         symbols,
         scaled_positions,
         sort_list) = sort_positions_by_symbols(cell.get_chemical_symbols(),
                                                cell.get_scaled_positions())
        w.write(" MAGMOM = ")
        for i in sort_list:
            w.write("%f " % magmoms[i])
        w.write("\n")
        w.close()
                
def get_scaled_positions_lines(scaled_positions):
    lines = ""
    for i, vec in enumerate(scaled_positions):
        for x in (vec - np.rint(vec)):
            if float('%20.16f' % x) < 0.0:
                lines += "%20.16f" % (x + 1.0)
            else:
                lines += "%20.16f" % (x)
        if i < len(scaled_positions) - 1:
            lines += "\n"

    return lines

def sort_positions_by_symbols(symbols, positions):
    reduced_symbols = _get_reduced_symbols(symbols)
    sorted_positions = []
    sort_list = []
    num_atoms = np.zeros(len(reduced_symbols), dtype=int)
    for i, rs in enumerate(reduced_symbols):
        for j, (s, p) in enumerate(zip(symbols, positions)):
            if rs == s:
                sorted_positions.append(p)
                sort_list.append(j)
                num_atoms[i] += 1
    return num_atoms, reduced_symbols, np.array(sorted_positions), sort_list

def _get_vasp_structure(atoms, direct=True):
    (num_atoms,
     symbols,
     scaled_positions,
     sort_list) = sort_positions_by_symbols(atoms.get_chemical_symbols(),
                                            atoms.get_scaled_positions())
    lines = ""     
    for s in symbols:
        lines += "%s " % s
    lines += "\n"
    lines += "   1.0\n"
    for a in atoms.get_cell():
        lines += "  %21.16f %21.16f %21.16f\n" % tuple(a)
    lines += ("%4d " * len(num_atoms)) % tuple(num_atoms)
    lines += "\n"
    lines += "Direct\n"
    lines += get_scaled_positions_lines(scaled_positions)

    return lines
    
def _get_reduced_symbols(symbols):
    reduced_symbols = []
    for s in symbols:
        if not (s in reduced_symbols):
            reduced_symbols.append(s)
    return reduced_symbols

#
# Non-analytical term
#
def get_born_OUTCAR(poscar_filename="POSCAR",
                    outcar_filename="OUTCAR",
                    primitive_axis=np.eye(3),
                    supercell_matrix=np.eye(3, dtype='intc'),
                    is_symmetry=True,
                    symmetrize_tensors=False,
                    symprec=1e-5):
    ucell = read_vasp(poscar_filename)
    outcar = open(outcar_filename)

    borns, epsilon = _read_born_and_epsilon(outcar)
    num_atom = len(borns)
    assert num_atom == ucell.get_number_of_atoms()
    
    if symmetrize_tensors:
        lattice = ucell.get_cell().T
        positions = ucell.get_scaled_positions()
        u_sym = Symmetry(ucell, is_symmetry=is_symmetry, symprec=symprec)
        point_sym = [similarity_transformation(lattice, r)
                     for r in u_sym.get_pointgroup_operations()]
        epsilon = _symmetrize_tensor(epsilon, point_sym)
        borns = _symmetrize_borns(borns, u_sym, lattice, positions, symprec)
        
    inv_smat = np.linalg.inv(supercell_matrix)
    scell = get_supercell(ucell,
                          supercell_matrix,
                          symprec=symprec)
    pcell = get_primitive(scell,
                          np.dot(inv_smat, primitive_axis),
                          symprec=symprec)
    p2s = np.array(pcell.get_primitive_to_supercell_map(), dtype='intc')
    p_sym = Symmetry(pcell, is_symmetry=is_symmetry, symprec=symprec)
    s_indep_atoms = p2s[p_sym.get_independent_atoms()]
    u2u = scell.get_unitcell_to_unitcell_map()
    u_indep_atoms = [u2u[x] for x in s_indep_atoms]
    reduced_borns = borns[u_indep_atoms].copy()
    
    return reduced_borns, epsilon

def _read_born_and_epsilon(outcar):
    borns = []
    while True:
        line = outcar.readline()
        if not line:
            break
    
        if "NIONS" in line:
            num_atom = int(line.split()[11])
    
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
            epsilon = []
            outcar.readline()
            epsilon.append([float(x) for x in outcar.readline().split()])
            epsilon.append([float(x) for x in outcar.readline().split()])
            epsilon.append([float(x) for x in outcar.readline().split()])
    
        if "BORN" in line:
            outcar.readline()
            line = outcar.readline()
            if "ion" in line:
                for i in range(num_atom):
                    born = []
                    born.append([float(x)
                                 for x in outcar.readline().split()][1:])
                    born.append([float(x)
                                 for x in outcar.readline().split()][1:])
                    born.append([float(x)
                                 for x in outcar.readline().split()][1:])
                    outcar.readline()
                    borns.append(born)

    borns = np.array(borns, dtype='double')
    epsilon = np.array(epsilon, dtype='double')

    return borns, epsilon

def _symmetrize_borns(borns, u_sym, lattice, positions, symprec):
    borns_orig = borns.copy()
    for i, Z in enumerate(borns):
        site_sym = [similarity_transformation(lattice, r)
                    for r in u_sym.get_site_symmetry(i)]
        Z = _symmetrize_tensor(Z, site_sym)

    rotations = u_sym.get_symmetry_operations()['rotations']
    translations = u_sym.get_symmetry_operations()['translations']
    map_atoms = u_sym.get_map_atoms()
    borns_copy = np.zeros_like(borns)
    for i in range(len(borns)):
        count = 0
        for r, t in zip(rotations, translations):
            count += 1
            diff = np.dot(positions, r.T) + t - positions[i]
            diff -= np.rint(diff)
            j = np.nonzero((np.abs(diff) < symprec).all(axis=1))[0][0]
            r_cart = similarity_transformation(lattice, r)
            borns_copy[i] += similarity_transformation(r_cart, borns[j])
        borns_copy[i] /= count

    borns = borns_copy
    sum_born = borns.sum(axis=0) / len(borns)
    borns -= sum_born

    if (np.abs(borns_orig - borns) > 0.1).any():
        sys.stderr.write(
            "Born effective charge symmetrization might go wrong.\n")

    return borns
    
def _symmetrize_tensor(tensor, symmetry_operations):
    sum_tensor = np.zeros_like(tensor)
    for sym in symmetry_operations:
        sum_tensor += similarity_transformation(sym, tensor)
    return sum_tensor / len(symmetry_operations)

#
# vasprun.xml handling
#
class VasprunWrapper(object):
    """VasprunWrapper class
    This is used to avoid VASP 5.2.8 vasprun.xml defect at PRECFOCK,
    xml parser stops with error.
    """
    def __init__(self, filename):
        self.f = open(filename)

    def read(self, size=None):
        element = self.f.next()
        if element.find("PRECFOCK") == -1:
            return element
        else:
            return "<i type=\"string\" name=\"PRECFOCK\"></i>"

def read_atomtypes_vasprun_xml(filename):
    vasprun = _parse_vasprun_xml(filename)
    for event, element in vasprun:
        atomtypes = _get_atomtypes_from_vasprun_xml(element)
        if atomtypes:
            return atomtypes
    return None

def read_force_constant_vasprun_xml(filename):
    vasprun = _parse_vasprun_xml(filename)
    return get_force_constants_vasprun_xml(vasprun)

def get_force_constants_vasprun_xml(vasprun):
    fc_tmp = None
    num_atom = 0
    for event, element in vasprun:
        if num_atom == 0:
            atomtypes = _get_atomtypes_from_vasprun_xml(element)
            if atomtypes:
                num_atoms, elements, elem_masses = atomtypes[:3]
                num_atom = np.sum(num_atoms)
                masses = []
                for n, m in zip(num_atoms, elem_masses):
                    masses += [m] * n

        # Get Hessian matrix (normalized by masses)
        if element.tag == 'varray':
            if element.attrib['name'] == 'hessian':
                fc_tmp = []
                for v in element.findall('./v'):
                    fc_tmp.append([float(x) for x in v.text.strip().split()])

    if fc_tmp is None:
        return False
    else:
        fc_tmp = np.array(fc_tmp)
        if fc_tmp.shape != (num_atom * 3, num_atom * 3):
            return False
        # num_atom = fc_tmp.shape[0] / 3
        force_constants = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
    
        for i in range(num_atom):
            for j in range(num_atom):
                force_constants[i, j] = fc_tmp[i*3:(i+1)*3, j*3:(j+1)*3]
    
        # Inverse normalization by atomic weights
        for i in range(num_atom):
            for j in range(num_atom):
                force_constants[i, j] *= -np.sqrt(masses[i] * masses[j])

        return force_constants, elements
    
def get_forces_from_vasprun_xmls(vaspruns, num_atom, index_shift=0):
    forces = []
    for i, vasprun in enumerate(vaspruns):
        sys.stderr.write("%d " % (i + 1 + index_shift))

        if _is_version528(vasprun):
            force_set = _get_forces_vasprun_xml(
                _iterparse(VasprunWrapper(vasprun), tag='varray'))
        else:
            force_set = _get_forces_vasprun_xml(
                _iterparse(vasprun, tag='varray'))
        if force_set.shape[0] == num_atom:
            forces.append(force_set)
        else:
            print("\nNumber of forces in vasprun.xml #%d is wrong." % (i + 1))
            sys.exit(1)
            
    sys.stderr.wrong("\n")
    return np.array(forces)
    
def get_force_constants_from_vasprun_xmls(vasprun_filenames):
    force_constants_set = []
    for i, filename in enumerate(vasprun_filenames):
        sys.stderr.write("%d: %s\n" % (i + 1, filename))
        force_constants_set.append(
            read_force_constant_vasprun_xml(filename)[0])
    sys.stderr.write("\n")
    return force_constants_set

def _parse_vasprun_xml(filename):
    if _is_version528(filename):
        return _iterparse(VasprunWrapper(filename))
    else:
        return _iterparse(filename)

def _iterparse(fname, tag=None):
    try:
        import xml.etree.cElementTree as etree
        for event, elem in etree.iterparse(fname):
            if tag is None or elem.tag == tag:
                yield event, elem
    except ImportError:
        print("Python 2.5 or later is needed.")
        print("For creating FORCE_SETS file with Python 2.4, you can use "
              "phonopy 1.8.5.1 with python-lxml .")
        sys.exit(1)        

def _is_version528(filename):
    for line in open(filename):
        if '\"version\"' in line:
            if '5.2.8' in line:
                return True
            else:
                return False

def _get_forces_vasprun_xml(vasprun):
    """
    vasprun = etree.iterparse(filename, tag='varray')
    """
    forces = []
    for event, element in vasprun:
        if element.attrib['name'] == 'forces':
            for v in element:
                forces.append([float(x) for x in v.text.split()])
    return np.array(forces)

def _get_atomtypes_from_vasprun_xml(element):
    atom_types = []
    masses = []
    valences = []
    num_atoms = []
    
    if element.tag == 'array':
        if 'name' in element.attrib:
            if element.attrib['name'] == 'atomtypes':
                for rc in element.findall('./set/rc'):
                    atom_info = [x.text for x in rc.findall('./c')]
                    num_atoms.append(int(atom_info[0]))
                    atom_types.append(atom_info[1].strip())
                    masses.append(float(atom_info[2]))
                    valences.append(float(atom_info[3]))
                return num_atoms, atom_types, masses, valences

    return None

#
# OUTCAR handling (obsolete)
#        
def read_force_constant_OUTCAR(filename):
    return get_force_constants_OUTCAR(filename)
    
def get_force_constants_OUTCAR(filename):
    file = open(filename)
    while 1:
        line = file.readline()
        if line == '':
            print("Force constants could not be found.")
            return 0

        if line[:19] == " SECOND DERIVATIVES":
            break

    file.readline()
    num_atom = int(((file.readline().split())[-1].strip())[:-1])

    fc_tmp = []
    for i in range(num_atom * 3):
        fc_tmp.append([float(x) for x in (file.readline().split())[1:]])

    fc_tmp = np.array(fc_tmp)

    force_constants = np.zeros((num_atom, num_atom, 3, 3), dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            force_constants[i, j] = -fc_tmp[i*3:(i+1)*3, j*3:(j+1)*3]

    return force_constants

if __name__ == '__main__':
    import sys
    atoms = read_vasp(sys.argv[1])
    write_vasp('%s-new' % sys.argv[1], atoms)

