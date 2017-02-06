# Copyright (C) 2016 Antti J. Karttunen (antti.j.karttunen@iki.fi)
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

from phonopy.file_IO import iter_collect_forces
from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.structure.atoms import symbol_map
from phonopy.structure.cells import get_cell_parameters, get_angles
from phonopy.units import Hartree, Bohr
from phonopy.structure.symmetry import Symmetry

def parse_set_of_forces(num_atoms,
                        forces_filenames,
                        verbose=True):
    hook = 'ATOM                     X                   Y                   Z'
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        crystal_forces = iter_collect_forces(filename,
                                           num_atoms,
                                           hook,
                                           [2, 3, 4])
        if check_forces(crystal_forces, num_atoms, filename, verbose=verbose):
            is_parsed = True
            drift_force = get_drift_forces(crystal_forces,
                                           filename=filename,
                                           verbose=verbose)
            # Convert forces Hartree / Bohr ->  eV / Angstrom
            # This avoids confusion with the units. CRYSTAL uses Angstroms for
            # coordinates, but Hartree / Bohr for forces. This would lead in mixed
            # units hartree / (Angstrom * Bohr) for force constants, requiring
            # additional tweaks for unit conversions in other parts of the code
            force_sets.append(np.multiply(np.array(crystal_forces) - drift_force, Hartree / Bohr))
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []

def read_crystal(filename):
    f_crystal = open(filename)
    crystal_in = CrystalIn(f_crystal.readlines())
    f_crystal.close()
    tags = crystal_in.get_tags()
     
     
    cell = Atoms(cell=tags['lattice_vectors'], 
                 symbols=tags['atomic_species'],
                 scaled_positions=tags['coordinates'])
    
    magmoms = tags['magnetic_moments']
    if magmoms is not None:
        # Print out symmetry information for magnetic cases
        # Original code from structure/symmetry.py
        symmetry = Symmetry(cell, symprec=1e-5)
        print("CRYSTAL-interface: Magnetic structure, number of operations without spin: %d" %
              len(symmetry.get_symmetry_operations()['rotations']))
        print("CRYSTAL-interface: Spacegroup without spin: %s" % symmetry.get_international_table())

        cell.set_magnetic_moments(magmoms)
        symmetry = Symmetry(cell, symprec=1e-5)
        print("CRYSTAL-interface: Magnetic structure, number of operations with spin: %d" %
              len(symmetry.get_symmetry_operations()['rotations']))
        print("")
    
    return cell, tags['conv_numbers']

def write_crystal(filename, cell, conv_numbers, template_file="TEMPLATE", write_symmetry=False):
    # Write geometry in EXTERNAL file (fort.34)
    f_ext = open(filename + '.ext', 'w')
    f_ext.write(get_crystal_structure(cell, conv_numbers, write_symmetry))
    f_ext.close()

    # Create input file (filename.d12)
    lines = "Created by Phonopy CRYSTAL interface\n"
    lines += "EXTERNAL\n"
    lines += "ENDGEOM\n"
    # If template_file exists, insert it at this point
    try:
        f_template = open(template_file)
        lines += f_template.read()
        f_template.close()
    except IOError:
        lines += "***** Insert basis sets and parameters here *****\n"
    lines += "TOLDEE\n"
    lines += "10\n"
    # For magnetic structures, create ATOMSPIN entry
    # Only spins != 0 are written
    magmoms = cell.get_magnetic_moments()
    if magmoms is not None:
        atomspins = ""
        N_spins = 0
        for i in range(0, len(magmoms)):
            if magmoms[i] != 0:
                N_spins += 1
                atomspins += ("%d %d " % (i + 1, magmoms[i]))
        lines += "ATOMSPIN\n"
        lines += ("%d\n" % N_spins)
        lines += atomspins + "\n"
    lines += "GRADCAL\n"
    lines += "END\n"
    
    # Write the input file
    f_inputfile = open(filename + '.d12', 'w')
    f_inputfile.writelines(lines)
    f_inputfile.close()

def write_supercells_with_displacements(supercell,
                                        cells_with_displacements,
                                        conv_numbers,
                                        supercell_matrix,
                                        pre_filename="supercell",
                                        width=3,
                                        template_file="TEMPLATE"):

    # From wien2k.py: derive conventional atomic numbers 
    # for the supercell from the original unit cell
    v = supercell_matrix
    det = (  v[0,0] * v[1,1] * v[2,2]
           + v[0,1] * v[1,2] * v[2,0]
           + v[0,2] * v[1,0] * v[2,1]
           - v[0,0] * v[1,2] * v[2,1]
           - v[0,1] * v[1,0] * v[2,2]
           - v[0,2] * v[1,1] * v[2,0])
    
    convnum_super = []
    for i in conv_numbers:
        for j in range(abs(det)):
            convnum_super.append(i)

    # Currently, symmetry is not used by default
    # It can be turned on by creating a file called CRY_SYM
    try:
        f = open('CRY_SYM')
        use_symmetry = True
        f.close()
    except IOError:
        use_symmetry = False

    if use_symmetry:
        print("CRYSTAL-interface: WARNING: Symmetry enabled in EXTERNAL files.\n" +
              "  Check the supercells very carefully, some spacegroups do not work (e.g. R-3m)\n" +
              "  Non-displaced supercell is always written without symmetry")

    write_crystal("supercell", supercell, convnum_super, template_file, write_symmetry=False)
    for i, cell in enumerate(cells_with_displacements):
        if cell is not None:
            filename = "{pre_filename}-{0:0{width}}".format(
                       i + 1,
                       pre_filename=pre_filename,
                       width=width)
            write_crystal(filename, cell, convnum_super, template_file, write_symmetry=use_symmetry)

def get_crystal_structure(cell, conv_numbers, write_symmetry=False):
    lattice = cell.get_cell()
    positions = cell.get_positions()
    numbers = cell.get_atomic_numbers()

    # Create and EXTERNAL file (fort.34)
    # Dimensionality, centring, crystal type
    lines = "3 1 1\n"
    # Cartesian components of the lattice vectors
    for lattvec in lattice:
      lines += ("%12.8f" * 3 + "\n") % tuple(lattvec)

    # Symmetry operators
    if write_symmetry:
        symmetry = Symmetry(cell, symprec=1e-5)
        rotations = symmetry.get_symmetry_operations()['rotations']
        translations = symmetry.get_symmetry_operations()['translations']
        N_symmops = 0
        symmlines = ""
        for i in range(0, len(rotations)):
            N_symmops += 1
            for j in range(0, 3):
                symmlines += ("  %5.2f" * 3 + "\n") % tuple(rotations[i][j])
            symmlines += ("  %5.2f" * 3 + "\n") % tuple(translations[i])

        lines += ("%d\n" % N_symmops)
        lines += symmlines

    else:
        lines += "1\n"
        lines += "  1.00  0.00  0.00\n"
        lines += "  0.00  1.00  0.00\n"
        lines += "  0.00  0.00  1.00\n"
        lines += "  0.00  0.00  0.00\n"

    # Number of atoms in the unit cell (asymmetric unit)
    lines += ("%d\n") % len(positions)
    # Conventional atomic number and cartesian coordinates of the atoms
    for i, pos in zip(conv_numbers, positions):
        lines += ("  %d " + "%16.12f"*3 + "\n") % (i,  pos[0], pos[1], pos[2])

    return lines
    
class CrystalIn:
    def __init__(self, lines):
        # conv_numbers = CRYSTAL conventional atomic number mapping: 'Ge' -> 32 or 'Ge' -> 232
        self._tags = {'lattice_vectors':  None,
                      'atomic_species':   None,
                      'coordinates':      None,
                      'magnetic_moments': None,
                      'conv_numbers':     None}

        self._values = None
        self._collect(lines)

    def get_tags(self):
        return self._tags

    def _collect(self, lines):
        # Reads a CRYSTAL output file (lattice vectors, conventional atomic numbers,
        # fractional atomic positions, magnetic moments)
        # NB. For optimization outputs, the final geometry in the file is read 
        #
        # If dielectric tensor and effective Born charges are present, writes a BORN file
        # (Gamma-point frequency calculation with INTENS - recommended choice!)
        #
        # If ATOMSPIN keyword is present, magnetic moments are read from it
        magmoms = []
        atomspins = []
        numspins = 0
        epsilon = []
        zeff = []
        l = 0
        while l < len(lines):
            line = lines[l]
            if 'PRIMITIVE CELL - CENTRING CODE' in line:
                aspecies = []
                coords = []
                convnum = []
                is_asym = []
                l += 4
                # ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    6
                N_asym_atoms = int(lines[l].split()[5])
                N_atoms = int(lines[l].split()[12])
                l += 3
                # 1 T  22 TI    4.721218104494E-21  3.307446203077E-21  1.413771901417E-21
                for atom in range(0, N_atoms):
                    atomdata = lines[l].split()
                    is_asym.append(atomdata[1])
                    aspecies.append(atomdata[3].capitalize())
                    coords.append([float(x) for x in atomdata[4:7]])
                    convnum.append(int(atomdata[2]))
                    l += 1
            elif 'DIRECT LATTICE VECTORS CARTESIAN COMPONENTS' in line:
                lattvecs = []
                l += 2
                #          X                    Y                    Z
                for lattvec in range(1, 4):
                    lattvecs.append([float(x) for x in lines[l].split()])
                    l += 1
            elif 'ATOMSPIN' in line:
                # Read ATOMSPIN, and save the magnetic moments for later parsing
                # (not all necessary information is available at this point)
                # All spins must be entered on one line!
                # ATOMSPIN
                # 8
                # 1 1 2 1 3 -1 4 -1 5 1 6 1 7 -1 8 -1
                l += 1
                numspins = int(lines[l])
                l += 1
                atomspins = [int(x) for x in lines[l].split()]
                l += 1
            elif 'COMPONENT    ALPHA(REAL, IMAGINARY)         EPSILON       CHI(1)' in line:
                # XX      296.9851        0.0000        7.9433        6.9433
                # XX, XY, XZ, YY, YZ, ZZ are given (6 rows). YX = XY, ZX = XZ, ZY = YZ 
                l += 1 
                epsilon.append(float(lines[l].split()[3])) # XX
                l += 1
                epsilon.append(float(lines[l].split()[3])) # XY
                l += 1
                epsilon.append(float(lines[l].split()[3])) # XZ
                epsilon.append(epsilon[1])                 # YX
                l += 1
                epsilon.append(float(lines[l].split()[3])) # YY
                l += 1
                epsilon.append(float(lines[l].split()[3])) # YZ
                epsilon.append(epsilon[2])                 # ZX
                epsilon.append(epsilon[5])                 # ZY
                l += 1
                epsilon.append(float(lines[l].split()[3])) # ZZ
            elif 'ATOMIC BORN CHARGE TENSOR' in line:
                l += 6
                for atom in range(0, N_atoms):
                    if is_asym[atom] == 'T':
                        zeffatom = []
                        for i in range(0, 3):
                            # 1     8.3860E-01  3.4861E-01  3.4861E-01
                            zeffatom += [float(x) for x in lines[l].split()[1:4]]
                            l += 1
                        zeff.append(zeffatom)
                        l += 4
                    else:
                        l += 7

            l += 1 # while l < len(lines)

        if len(lattvecs) == 3 and len(aspecies) > 0 and len(aspecies) == len(coords) and len(aspecies) == len(convnum):
            self._tags['lattice_vectors'] = lattvecs
            self._tags['atomic_species'] = aspecies
            self._tags['coordinates'] = coords
            self._tags['conv_numbers'] = convnum
        else:
            print("CRYSTAL-interface: Error parsing CRYSTAL output file")

        # Create BORN file if epsilon and zeff are available
        if len(epsilon) == 9 and len(zeff) == N_asym_atoms: 
            # Conversion factor
            bornlines = ("default\n")
            # Dielectric tensor
            bornlines += ("%6.4f " * 9 + "\n") % tuple(epsilon)
            # Effective charges
            for atom in range(0, N_asym_atoms):
                bornlines += ("%6.4f " * 9 + "\n") % tuple(zeff[atom])

            # Write file
            bornf = open("BORN", "w")
            bornf.writelines(bornlines)
            bornf.close()
            print("CRYSTAL-interface: Dielectric tensor and effective charges written in BORN file")

        # Set magnetic moments
        if numspins > 0:
            # Initialize all moments to zero
            magmoms = [0] * N_atoms
            if numspins * 2 == len(atomspins):
                for i in range(0, numspins):
                    atomnum = atomspins[i*2] - 1
                    magmom = atomspins[i*2 + 1]
                    magmoms[atomnum] = magmom

                self._tags['magnetic_moments'] = magmoms
                print("CRYSTAL-interface: Following magnetic moments have been read from ATOMSPIN entry:")
                print(magmoms)
            else:
                print("CRYSTAL-interface: Invalid ATOMSPIN entry, magnetic moments have not been set")
        else: 
            print("") 

            
if __name__ == '__main__':
    import sys
    from phonopy.structure.symmetry import Symmetry
    cell, conv_numbers = read_crystal(sys.argv[1])
    symmetry = Symmetry(cell)
    print("# %s" % symmetry.get_international_table())
    print(get_crystal_structure(cell, conv_numbers))
