import sys
import os
import numpy as np
import h5py
from phonopy.structure.atoms import Atoms
from phonopy.interface import vasp
from phonopy.file_IO import write_FORCE_SETS_vasp, read_force_constant_vasprun_xml, parse_FORCE_SETS

###########
#
# File I/O
#
###########

def write_disp_fc3_yaml(dataset, supercell, filename='disp_fc3.yaml'):
    w = open(filename, 'w')
    w.write("natom: %d\n" %  dataset['natom'])

    num_first = len(dataset['first_atoms'])
    w.write("num_first_displacements: %d\n" %  num_first)
    if 'cutoff_distance' in dataset:
        w.write("cutoff_distance: %f\n" %  dataset['cutoff_distance'])

    num_second = 0
    num_disp_files = 0
    for d1 in dataset['first_atoms']:
        num_disp_files += 1
        num_second += len(d1['second_atoms'])
        for d2 in d1['second_atoms']:
            if 'included' in d2:
                if d2['included']:
                    num_disp_files += 1
            else:
                num_disp_files += 1
                
    w.write("num_second_displacements: %d\n" %  num_second)
    w.write("num_displacements_created: %d\n" %  num_disp_files)

    w.write("first_atoms:\n")
    count1 = 1
    count2 = num_first + 1
    for disp1 in dataset['first_atoms']:
        disp_cart1 = disp1['displacement']
        w.write("- number: %5d\n" % (disp1['number'] + 1))
        w.write("  displacement:\n")
        w.write("    [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1))
        w.write("  second_atoms:\n")
        count1 += 1

        included = None
        distance = 0.0
        atom2 = -1
        displacements = []
        for disp2 in disp1['second_atoms']:
            if atom2 != disp2['number']:
                atom2 = disp2['number']
                if 'included' in disp2:
                    included = disp2['included']
                pair_distance = disp2['pair_distance']
                w.write("  - number: %5d\n" % (atom2 + 1))
                w.write("    distance: %f\n" % pair_distance)
                if included is not None:
                    if included:
                        w.write("    included: %s\n" % "true")
                    else:
                        w.write("    included: %s\n" % "false")
                w.write("    displacements:\n")
                
            disp_cart2 = disp2['displacement']
            w.write("    - [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                    (disp_cart2[0], disp_cart2[1], disp_cart2[2], count2))
            count2 += 1

    w.write("lattice:\n")
    for axis in supercell.get_cell():
        w.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    w.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        w.write("- symbol: %-2s # %d\n" % (s, i+1))
        w.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % tuple(v))
    w.close()

    return num_first + num_second, num_disp_files

def write_supercells_with_displacements_from_direction_dataset(
        supercell,
        double_displacements,
        distance=0.03,
        cutoff_distance=None,
        filename='disp_fc3.yaml'):
    """
    This is the disp_fc3.yaml writer previously used (version < 0.8.2).
    """
    # YAML
    w = open(filename, 'w')
    w.write("natom: %d\n" %  supercell.get_number_of_atoms())

    num_first = len(double_displacements)
    w.write("num_first_displacements: %d\n" %  num_first)
    if cutoff_distance is not None:
        w.write("cutoff_distance: %f\n" %  cutoff_distance)

    num_second = 0
    num_disp_files = 0
    for d1 in double_displacements:
        num_disp_files += 1
        for d2 in d1['second_atoms']:
            num_second += len(d2['directions'])
            included = (d2['distance'] < cutoff_distance or
                        cutoff_distance is None)
            if included:
                num_disp_files += len(d2['directions'])
                
    w.write("num_second_displacements: %d\n" %  num_second)
    w.write("num_displacements_created: %d\n" %  num_disp_files)

    w.write("first_atoms:\n")
    lattice = supercell.get_cell()
    count1 = 1
    count2 = num_first + 1
    for disp1 in double_displacements:
        disp_cart1 = np.dot(disp1['direction'], lattice)
        disp_cart1 = disp_cart1 / np.linalg.norm(disp_cart1) * distance
        positions = supercell.get_positions()
        positions[disp1['number']] += disp_cart1
        atoms = Atoms(numbers=supercell.get_atomic_numbers(),
                      masses=supercell.get_masses(),
                      positions=positions,
                      cell=lattice,
                      pbc=True)
        vasp.write_vasp('POSCAR-%05d' % count1, atoms, direct=True)

        # YAML
        w.write("- number: %5d\n" % (disp1['number'] + 1))
        w.write("  displacement:\n")
        w.write("    [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1))
        w.write("  second_atoms:\n")
        count1 += 1

        for disp2 in disp1['second_atoms']:
            # YAML
            w.write("  - number: %5d\n" % (disp2['number'] + 1))
            w.write("    distance: %f\n" % disp2['distance'])

            included = (disp2['distance'] < cutoff_distance or
                        cutoff_distance is None)
            if cutoff_distance is not None:
                if included:
                    w.write("    included: %s\n" % "true")
                else:
                    w.write("    included: %s\n" % "false")

            w.write("    displacements:\n")
            
            for direction in disp2['directions']:
                disp_cart2 = np.dot(direction, lattice)
                disp_cart2 = disp_cart2 / np.linalg.norm(disp_cart2) * distance
                positions = supercell.get_positions()
                positions[disp1['number']] += disp_cart1
                positions[disp2['number']] += disp_cart2
                atoms = Atoms(numbers=supercell.get_atomic_numbers(),
                               masses=supercell.get_masses(),
                               positions=positions,
                               cell=lattice,
                               pbc=True)
                if included:
                    vasp.write_vasp('POSCAR-%05d' % count2, atoms, direct=True)

                # YAML
                w.write("    - [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                           (disp_cart2[0], disp_cart2[1], disp_cart2[2],
                            count2))
                count2 += 1

    w.write("lattice:\n")
    for axis in supercell.get_cell():
        w.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    w.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        w.write("- symbol: %-2s # %d\n" % (s, i+1))
        w.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % \
                       (v[0], v[1], v[2]))
    w.close()

    return num_first + num_second, num_disp_files

def write_supercells_with_three_displacements(supercell,
                                              triple_displacements,
                                              amplitude=None,
                                              filename_fc3='disp_fc3.yaml',
                                              filename_fc4='disp_fc4.yaml'):
    if amplitude==None:
        distance = 0.01
    else:
        distance = amplitude
    
    # YAML
    w3 = open(filename_fc3, 'w')
    w4 = open(filename_fc4, 'w')
    w3.write("natom: %d\n" %  supercell.get_number_of_atoms())
    w4.write("natom: %d\n" %  supercell.get_number_of_atoms())

    num_first = len(triple_displacements)
    w3.write("num_first_displacements: %d\n" % num_first)
    w4.write("num_first_displacements: %d\n" % num_first)
    num_second = 0
    for d1 in triple_displacements:
        num_second += len(d1['second_atoms'])
    w3.write("num_second_displacements: %d\n" %  num_second)
    w4.write("num_second_displacements: %d\n" %  num_second)
    num_third = 0
    for d1 in triple_displacements:
        for d2 in d1['second_atoms']:
            for d3 in d2['third_atoms']:
                num_third += len(d3['directions'])
    w4.write("num_third_displacements: %d\n" %  num_third)

    w3.write("first_atoms:\n")
    w4.write("first_atoms:\n")
    lattice = supercell.get_cell()
    count1 = 1
    count2 = num_first + 1
    count3 = num_second + num_first + 1
    for disp1 in triple_displacements:
        disp_cart1 = np.dot(disp1['direction'], lattice)
        disp_cart1 = disp_cart1 / np.linalg.norm(disp_cart1) * distance
        positions = supercell.get_positions()
        positions[disp1['number']] += disp_cart1
        atoms = Atoms(numbers=supercell.get_atomic_numbers(),
                      masses=supercell.get_masses(),
                      positions=positions,
                      cell=lattice,
                      pbc=True)
        vasp.write_vasp('POSCAR-%05d' % count1, atoms, direct=True)

        # YAML
        w3.write("- number: %5d\n" % (disp1['number'] + 1))
        w3.write("  displacement:\n")
        w3.write("    [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                   (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1))
        w3.write("  second_atoms:\n")
        w4.write("- number: %5d\n" % (disp1['number'] + 1))
        w4.write("  displacement:\n")
        w4.write("    [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                   (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1))
        w4.write("  second_atoms:\n")
        count1 += 1
        second_atom_num = -1
        for disp2 in disp1['second_atoms']:
            disp_cart2 = np.dot(disp2['direction'], lattice)
            disp_cart2 = disp_cart2 / np.linalg.norm(disp_cart2) * distance
            positions = supercell.get_positions()
            positions[disp1['number']] += disp_cart1
            positions[disp2['number']] += disp_cart2
            atoms = Atoms(numbers=supercell.get_atomic_numbers(),
                          masses=supercell.get_masses(),
                          positions=positions,
                          cell=lattice,
                          pbc=True)
            vasp.write_vasp('POSCAR-%05d' % count2, atoms, direct=True)

            # YAML
            if second_atom_num != disp2['number']:
                w3.write("  - number: %5d\n" % (disp2['number'] + 1))
                w3.write("    displacements:\n")
                second_atom_num = disp2['number']
                
            w3.write("    - [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                     (disp_cart2[0], disp_cart2[1], disp_cart2[2], count2))
            w4.write("  - number: %5d\n" % (disp2['number'] + 1))
            w4.write("    displacement:\n")
            w4.write("      [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                    (disp_cart2[0], disp_cart2[1], disp_cart2[2], count2))
            w4.write("    third_atoms:\n")

            count2 += 1

            for disp3 in disp2['third_atoms']:
                w4.write("    - number: %5d\n" % (disp3['number'] + 1))
                w4.write("      displacements:\n")
                for direction in disp3['directions']:
                    disp_cart3 = np.dot(direction, lattice)
                    disp_cart3 = (disp_cart3 / np.linalg.norm(disp_cart3) *
                                  distance)
                    positions = supercell.get_positions()
                    positions[disp1['number']] += disp_cart1
                    positions[disp2['number']] += disp_cart2
                    positions[disp3['number']] += disp_cart3
                    atoms = Atoms(numbers=supercell.get_atomic_numbers(),
                                  masses=supercell.get_masses(),
                                  positions=positions,
                                  cell=lattice,
                                  pbc=True)
                    vasp.write_vasp('POSCAR-%05d' % count3, atoms, direct=True)
    
                    # YAML
                    w4.write("      - [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                            (disp_cart3[0], disp_cart3[1], disp_cart3[2],
                             count3))
                    count3 += 1

    w3.write("lattice:\n")
    w4.write("lattice:\n")
    for axis in supercell.get_cell():
        w3.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
        w4.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    w3.write("atoms:\n")
    w4.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        w3.write("- symbol: %-2s # %d\n" % (s, i+1))
        w3.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % \
                       (v[0], v[1], v[2]))
        w4.write("- symbol: %-2s # %d\n" % (s, i+1))
        w4.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % \
                       (v[0], v[1], v[2]))

    w3.close()
    w4.close()

def write_FORCES_FC3_vasp(vaspruns,
                          disp_dataset,
                          filename='FORCES_FC3'):
    natom = disp_dataset['natom']
    forces_fc3 = get_forces_from_vasprun_xmls(vaspruns, natom)
    write_FORCES_FC3(disp_dataset, forces_fc3, filename=filename)
    
def write_FORCES_FC3(disp_dataset,
                     forces_fc3,
                     filename='FORCES_FC3'):
    natom = disp_dataset['natom']
    num_disp1 = len(disp_dataset['first_atoms'])
    w3 = open(filename, 'w')

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        w3.write("# File: %-5d\n" % (i + 1))
        w3.write("# %-5d " % (disp1['number'] + 1))
        w3.write("%20.16f %20.16f %20.16f\n" %
                         tuple(disp1['displacement']))
        for forces in forces_fc3[i]:
            w3.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
        
    count = num_disp1
    file_count = num_disp1
    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        atom1 = disp1['number']
        for disp2 in disp1['second_atoms']:
            atom2 = disp2['number']
            w3.write("# File: %-5d\n" % (count + 1))
            w3.write("# %-5d " % (atom1 + 1))
            w3.write("%20.16f %20.16f %20.16f\n" % tuple(disp1['displacement']))
            w3.write("# %-5d " % (atom2 + 1))
            w3.write("%20.16f %20.16f %20.16f\n" % tuple(disp2['displacement']))

            # For supercell calculation reduction
            included = True
            if 'included' in disp2:
                included = disp2['included']
            if included:
                for forces in forces_fc3[file_count]:
                    w3.write("%15.10f %15.10f %15.10f\n" % tuple(forces))
                file_count += 1
            else:
                # for forces in forces_fc3[i]:
                #     w3.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
                for j in range(natom):
                    w3.write("%15.10f %15.10f %15.10f\n" % (0, 0, 0))
            count += 1
    w3.close()
            
def write_FORCES_THIRD(vaspruns,
                       disp_dataset,
                       forces_third='FORCES_THIRD',
                       forces_second='FORCES_SECOND'):
    natom = disp_dataset['natom']
    num_disp1 = len(disp_dataset['first_atoms'])
    set_of_forces = get_forces_from_vasprun_xmls(vaspruns, natom)
    w3 = open(forces_third, 'w')
    w2 = open(forces_second, 'w')

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        w2.write("# File: %-5d\n" % (i + 1))
        w2.write("# %-5d " % (disp1['number'] + 1))
        w2.write("%20.16f %20.16f %20.16f\n" %
                         tuple(disp1['displacement']))
        for forces in set_of_forces[i]:
            w2.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
        
    count = num_disp1
    file_count = num_disp1
    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        atom1 = disp1['number']
        for disp2 in disp1['second_atoms']:
            atom2 = disp2['number']
            w3.write("# File: %-5d\n" % (count + 1))
            w3.write("# %-5d " % (atom1 + 1))
            w3.write("%20.16f %20.16f %20.16f\n" % tuple(disp1['displacement']))
            w3.write("# %-5d " % (atom2 + 1))
            w3.write("%20.16f %20.16f %20.16f\n" % tuple(disp2['displacement']))

            # For supercell calculation reduction
            if disp2['included']:
                for forces in set_of_forces[file_count]:
                    w3.write("%15.10f %15.10f %15.10f\n" % tuple(forces))
                file_count += 1
            else:
                # for forces in set_of_forces[i]:
                #     w3.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
                for j in range(natom):
                    w3.write("%15.10f %15.10f %15.10f\n" % (0, 0, 0))
            count += 1
            

def write_FORCES_FOURTH(vaspruns,
                        disp_dataset,
                        forces_fourth='FORCES_FOURTH',
                        forces_third='FORCES_THIRD',
                        forces_second='FORCES_SECOND'):

    count = 0
    for disp1 in disp_dataset['first_atoms']:
        count += 1
        for disp2 in disp1['second_atoms']:
            count += 1
    write_FORCES_THIRD(vaspruns[:count],
                       disp_dataset,
                       forces_third=forces_third,
                       forces_second=forces_second)
    natom = disp_dataset['natom']
    set_of_forces = get_forces_from_vasprun_xmls(vaspruns[count:],
                                                 natom,
                                                 index_shift=count)
    count_begin = count
    w4 = open(forces_fourth, 'w')
    for disp1 in disp_dataset['first_atoms']:
        atom1 = disp1['number']
        for disp2 in disp1['second_atoms']:
            atom2 = disp2['number']
            for disp3 in disp2['third_atoms']:
                atom3 = disp3['number']
                d = disp3['displacement']
                w4.write("# File: %-5d\n" % (count + 1))
                w4.write("# %-5d " % (atom1 + 1))
                w4.write("%20.16f %20.16f %20.16f\n" %
                         tuple(disp1['displacement']))
                w4.write("# %-5d " % (atom2 + 1))
                w4.write("%20.16f %20.16f %20.16f\n" %
                         tuple(disp2['displacement']))
                w4.write("# %-5d " % (atom3 + 1))
                w4.write("%20.16f %20.16f %20.16f\n" % tuple(d))
                for forces in set_of_forces[count - count_begin]:
                    w4.write("%15.10f %15.10f %15.10f\n" % tuple(forces))
                count += 1
                
def write_DELTA_FC2_SETS(vaspruns,
                         disp_dataset,
                         dfc2_file='DELTA_FC2_SETS'):
    fc2_set = get_force_constants_from_vasprun_xmls(vaspruns)
    perfect_fc2 = fc2_set.pop(0)
    write_fc2_to_hdf5(perfect_fc2)
    delta_fc2s = [fc2 - perfect_fc2 for fc2 in fc2_set]
    write_DELTA_FC2_SETS_from_delta_fc2s(delta_fc2s,
                                         disp_dataset,
                                         dfc2_file=dfc2_file)

def write_DELTA_FC2_SETS_from_delta_fc2s(delta_fc2s,
                                         disp_dataset,
                                         dfc2_file='DELTA_FC2_SETS'):
    w = open(dfc2_file, 'w')
    for i, (dfc2, first_disp) in enumerate(zip(delta_fc2s,
                                               disp_dataset['first_atoms'])):
        w.write("# File: %d\n" % (i + 1))
        w.write("# %-5d " % (first_disp['number'] + 1))
        w.write("%20.16f %20.16f %20.16f\n" %
                tuple(first_disp['displacement']))
        for j in range(dfc2.shape[0]):
            for k in range(dfc2.shape[1]):
                w.write("# %d - %d\n" % (j + 1, k + 1))
                for vec in dfc2[j, k]:
                    w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
    w.close()

# From 0000 to the end of the numbers
def write_DELTA_FC2_FOURTH_SETS(vaspruns,
                                disp_dataset,
                                dfc2_file='DELTA_FC2_FOURTH_SETS'):
    """Write displaced fc2 for fc4 from vasprun.xml's"""
    
    fc2_set = get_force_constants_from_vasprun_xmls(vaspruns)
    fc2s_first = []
    count = 0
    w = open(dfc2_file, 'w')

    # fc2
    perfect_fc2 = fc2_set.pop(0)
    write_fc2_to_hdf5(perfect_fc2)

    # fc3
    for i, first_disp in enumerate(disp_dataset['first_atoms']):
        count += 1
        fc2s_first.append(fc2_set.pop(0))
    delta_fc2s = [fc2 - perfect_fc2 for fc2 in fc2s_first]
    write_DELTA_FC2_SETS_from_delta_fc2s(delta_fc2s, disp_dataset)
        
    # fc4
    for i, first_disp in enumerate(disp_dataset['first_atoms']):
        for second_disp in first_disp['second_atoms']:
            disp = second_disp['displacement']
            count += 1
            dfc2 = fc2_set.pop(0) - fc2s_first[i]
            w.write("# File: %d\n" % count)
            w.write("# %-5d" % (first_disp['number'] + 1))
            w.write("%20.16f %20.16f %20.16f\n" %
                    tuple(first_disp['displacement']))
            w.write("# %-5d" % (second_disp['number'] + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp))
        
            for j in range(dfc2.shape[0]):
                for k in range(dfc2.shape[1]):
                    w.write("# %d - %d\n" % (j + 1, k + 1))
                    for vec in dfc2[j, k]:
                        w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))

def write_fc3_yaml(force_constants_third,
                   filename='fc3.yaml',
                   is_symmetrize=False):
    w = open(filename, 'w')
    num_atom = force_constants_third.shape[0]
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                if is_symmetrize:
                    tensor3 = symmetrize_fc3(force_constants_third, i, j, k)
                else:
                    tensor3 = force_constants_third[i, j, k]
                w.write("- index: [ %d - %d - %d ] # (%f)\n" \
                               % (i+1, j+1, k+1, tensor3.sum()))
                w.write("  tensor:\n")
                for tensor2 in tensor3:
                    w.write("  -\n")
                    for vec in tensor2:
                        w.write("    - [ %13.8f, %13.8f, %13.8f ]\n" % tuple(vec))
                w.write("\n")

def write_fc3_dat(force_constants_third, filename='fc3.dat'):
    w = open(filename, 'w')
    for i in range(force_constants_third.shape[0]):
        for j in range(force_constants_third.shape[1]):
            for k in range(force_constants_third.shape[2]):
                tensor3 = force_constants_third[i, j, k]
                w.write(" %d - %d - %d  (%f)\n" % (i+1, j+1, k+1, np.abs(tensor3).sum()))
                for tensor2 in tensor3:
                    for vec in tensor2:
                        w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
                    w.write("\n")

def write_fc4_dat(fc4, filename='fc4.dat'):
    w = open(filename, 'w')
    for (i, j, k, l) in list(np.ndindex(fc4.shape[:4])):
        tensor4 = fc4[i, j, k, l]
        w.write(" %d - %d - %d - %d (%f)\n" % (i+1, j+1, k+1, l+1, np.abs(tensor4).sum()))
        for tensor3 in tensor4:
            for tensor2 in tensor3:
                for vec in tensor2:
                    w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
                w.write("\n")
            w.write("\n")
        w.write("\n")

def write_fc4_to_hdf5(force_constants_fourth, filename='fc4.hdf5'):
    w = h5py.File(filename, 'w')
    w.create_dataset('fc4', data=force_constants_fourth)
    w.close()

def read_fc4_from_hdf5(filename='fc4.hdf5'):
    f = h5py.File(filename, 'r')
    fc4 = f['fc4'][:]
    f.close()
    return fc4
    
def write_fc3_to_hdf5(force_constants_third, filename='fc3.hdf5'):
    w = h5py.File(filename, 'w')
    w.create_dataset('fc3', data=force_constants_third)
    w.close()

def read_fc3_from_hdf5(filename='fc3.hdf5'):
    f = h5py.File(filename, 'r')
    fc3 = f['fc3'][:]
    f.close()
    return fc3
    
def write_fc2_dat(force_constants, filename='fc2.dat'):
    w = open(filename, 'w')
    for i, fcs in enumerate(force_constants):
        for j, fcb in enumerate(fcs):
            w.write(" %d - %d\n" % (i+1, j+1))
            for vec in fcb:
                w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
            w.write("\n")

def write_fc2_to_hdf5(force_constants, filename='fc2.hdf5'):
    w = h5py.File(filename, 'w')
    w.create_dataset('fc2', data=force_constants)
    w.close()

def read_fc2_from_hdf5(filename='fc2.hdf5'):
    f = h5py.File(filename, 'r')
    if 'fc2' in f.keys():
        fc2 = f['fc2'][:]
    elif 'force_constants' in f.keys():
        fc2 = f['force_constants'][:]
    else:
        fc2 = None
    f.close()
    return fc2

def write_triplets(triplets,
                   weights,
                   mesh,
                   grid_address,
                   grid_point=None,
                   filename=None):
    triplets_filename = "triplets"
    suffix = "-m%d%d%d" % tuple(mesh)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if filename is not None:
        suffix += "." + filename
    suffix += ".dat"
    triplets_filename += suffix
    w = open(triplets_filename, 'w')
    for weight, g3 in zip(weights, triplets):
        w.write("%4d    " % weight)
        for q3 in grid_address[g3]:
            w.write("%4d %4d %4d    " % tuple(q3))
        w.write("\n")
    w.close()

def write_grid_address(grid_address, mesh, filename=None):
    grid_address_filename = "grid_address"
    suffix = "-m%d%d%d" % tuple(mesh)
    if filename is not None:
        suffix += "." + filename
    suffix += ".dat"
    grid_address_filename += suffix
    
    w = open(grid_address_filename, 'w')
    w.write("# Grid addresses for %dx%dx%d mesh\n" % tuple(mesh))
    w.write("#%9s    %8s %8s %8s     %8s %8s %8s\n" %
            ("index", "a", "b", "c",
             ("a%%%d" % mesh[0]), ("b%%%d" % mesh[1]), ("c%%%d" % mesh[2])))
    for i, bz_q in enumerate(grid_address):
        if i == np.prod(mesh):
            w.write("#" + "-" * 78 + "\n")
        q = bz_q % mesh
        w.write("%10d    %8d %8d %8d     " % (i, bz_q[0], bz_q[1], bz_q[2]))
        w.write("%8d %8d %8d\n" % tuple(q))

    return grid_address_filename

def write_freq_shifts_to_hdf5(freq_shifts, filename='freq_shifts.hdf5'):
    w = h5py.File(filename, 'w')
    w.create_dataset('shift', data=freq_shifts)
    w.close()

def write_damping_functions(gp,
                            band_indices,
                            mesh,
                            frequencies,
                            gammas,
                            sigma=None,
                            temperature=None,
                            filename=None,
                            is_nosym=False):

    gammas_filename = "gammas"
    gammas_filename += "-m%d%d%d-g%d-" % (mesh[0],
                                          mesh[1],
                                          mesh[2],
                                          gp)
    if sigma is not None:
        gammas_filename += ("s%f" % sigma).rstrip('0').rstrip('\.') + "-"

    if temperature is not None:
        gammas_filename += ("t%f" % temperature).rstrip('0').rstrip('\.') + "-"

    for i in band_indices:
        gammas_filename += "b%d" % (i + 1)

    if not filename == None:
        gammas_filename += ".%s" % filename
    elif is_nosym:
        gammas_filename += ".nosym"
    gammas_filename += ".dat"

    w = open(gammas_filename, 'w')
    for freq, g in zip(frequencies, gammas):
        w.write("%15.7f %20.15e\n" % (freq, g))
    w.close()

def write_joint_dos(gp,
                    mesh,
                    frequencies,
                    jdos,
                    sigma=None,
                    filename=None,
                    is_nosym=False):

    jdos_filename = "jdos-m%d%d%d-g%d" % (mesh[0], mesh[1], mesh[2], gp)
    if sigma is not None:
        jdos_filename += ("-s%f" % sigma).rstrip('0').rstrip('\.')
    if is_nosym:
        jdos_filename += ".nosym"
    if filename is not None:
        jdos_filename += ".%s" % filename
    jdos_filename += ".dat" 
        
    w = open(jdos_filename, 'w')
    for omega, val in zip(frequencies, jdos):
        w.write("%15.7f %20.15e\n" % (omega, val))
    w.close()

def write_linewidth(gp,
                    band_indices,
                    temperatures,
                    gamma,
                    mesh,
                    sigma=None,
                    is_nosym=False,
                    filename=None):

    lw_filename = "linewidth"
    lw_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if sigma is not None:
        lw_filename += ("s%f" % sigma).rstrip('0') + "-"

    for i in band_indices:
        lw_filename += "b%d" % (i + 1)

    if not filename == None:
        lw_filename += ".%s" % filename
    elif is_nosym:
        lw_filename += ".nosym"
    lw_filename += ".dat"

    w = open(lw_filename, 'w')
    for v, t in zip(gamma.sum(axis=1) * 2 / gamma.shape[1], temperatures):
        w.write("%15.7f %20.15e\n" % (t, v))
    w.close()

def write_frequency_shift(gp,
                          band_indices,
                          temperatures,
                          delta,
                          mesh,
                          epsilon=None,
                          is_nosym=False,
                          filename=None):

    fst_filename = "frequency_shift"
    fst_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if epsilon is not None:
        fst_filename += ("s%f" % epsilon).rstrip('0') + "-"
    for i in band_indices:
        fst_filename += "b%d" % (i + 1)
    if not filename == None:
        fst_filename += ".%s" % filename
    elif is_nosym:
        fst_filename += ".nosym"
    fst_filename += ".dat"

    w = open(fst_filename, 'w')
    for v, t in zip(delta.sum(axis=1) / delta.shape[1], temperatures):
        w.write("%15.7f %20.15e\n" % (t, v))
    w.close()
    
def write_kappa(kappa,
                temperatures,
                mesh,
                mesh_divisors=None,
                grid_point=None,
                sigma=None,
                filename=None):
    kappa_filename = "kappa"
    suffix = "-m%d%d%d" % tuple(mesh)
    if mesh_divisors is not None:
        if (np.array(mesh_divisors, dtype=int) != 1).any():
            suffix += "-d%d%d%d" % tuple(mesh_divisors)
    sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if sigma is not None:
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename
    suffix += ".dat"
    kappa_filename += suffix
    print "Kappa",
    if grid_point is not None:
        print "at grid adress %d" % grid_point,
    if sigma is not None:
        if grid_point is not None:
            print "and",
        else:
            print "at",
        print "sigma %s" % sigma_str,
    print "were written into",
    if grid_point is not None:
        print ""
    print "\"%s\"" % kappa_filename
    w = open(kappa_filename, 'w')
    w.write("# temp   kappa\n")
    for t, k in zip(temperatures, kappa):
        w.write("%6.1f %.5f\n" % (t, k))
    w.close()

def write_kappa_to_hdf5(gamma,
                        temperature,
                        mesh,
                        frequency=None,
                        group_velocity=None,
                        heat_capacity=None,
                        kappa=None,
                        qpoint=None,
                        weight=None,
                        mesh_divisors=None,
                        grid_point=None,
                        sigma=None,
                        filename=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    if mesh_divisors is not None:
        if (np.array(mesh_divisors, dtype=int) != 1).any():
            suffix += "-d%d%d%d" % tuple(mesh_divisors)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if sigma is not None:
        sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename
    w = h5py.File("kappa" + suffix + ".hdf5", 'w')
    w.create_dataset('gamma', data=gamma)
    w.create_dataset('frequency', data=frequency)
    w.create_dataset('temperature', data=temperature)
    w.create_dataset('group_velocity', data=group_velocity)
    if heat_capacity is not None:
        w.create_dataset('heat_capacity', data=heat_capacity)
    if kappa is not None:
        w.create_dataset('kappa', data=kappa)
    if qpoint is not None:
        w.create_dataset('qpoint', data=qpoint)
    if weight is not None:
        w.create_dataset('weight', data=weight)
    w.close()

    print "Values to calculate kappa",
    if grid_point is not None:
        print "at grid adress %d" % grid_point,
    if sigma is not None:
        if grid_point is not None:
            print "and",
        else:
            print "at",
        print "sigma %s" % sigma_str
    print "were written into"
    print "\"%s\"" % ("kappa" + suffix + ".hdf5")
    print

def read_gamma_from_hdf5(mesh,
                         mesh_divisors=None,
                         grid_point=None,
                         sigma=None,
                         filename=None,
                         verbose=True):
    suffix = "-m%d%d%d" % tuple(mesh)
    if mesh_divisors is not None:
        if (mesh_divisors != 1).any():
            suffix += "-d%d%d%d" % tuple(mesh_divisors)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if sigma is not None:
        sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename

    if not os.path.exists("kappa" + suffix + ".hdf5"):
        return False
        
    f = h5py.File("kappa" + suffix + ".hdf5", 'r')
    gammas = f['gamma'][:]
    f.close()
    
    if verbose:
        print "Gamma",
        if grid_point is not None:
            print "at grid adress %d" % grid_point,
        if sigma is not None:
            if grid_point is not None:
                print "and",
            else:
                print "at",
            print "sigma %s" % sigma_str,
        print "were read from",
        if grid_point is not None:
            print ""
        print "%s" % ("kappa" + suffix + ".hdf5")
    
    return gammas

def write_amplitude_to_hdf5(amplitude,
                            mesh,
                            grid_point,
                            triplet=None,
                            weight=None,
                            frequency=None,
                            eigenvector=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    suffix += ("-g%d" % grid_point)
    w = h5py.File("amplitude" + suffix + ".hdf5", 'w')
    w.create_dataset('amplitude', data=amplitude)
    if triplet is not None:
        w.create_dataset('triplet', data=triplet)
    if weight is not None:
        w.create_dataset('weight', data=weight)
    if frequency is not None:
        w.create_dataset('frequency', data=frequency)
    if eigenvector is not None:
        w.create_dataset('eigenvector', data=eigenvector)
    w.close()

def read_amplitude_from_hdf5(amplitudes_at_q,
                             mesh,
                             grid_point):
    suffix = "-m%d%d%d" % tuple(mesh)
    suffix += ("-g%d" % grid_point)
    f = h5py.File("amplitude" + suffix + ".hdf5", 'r')
    amplitudes_at_q[:] = f['amplitudes'][:]
        
def write_decay_channels(decay_channels,
                         amplitudes_at_q,
                         frequencies_at_q,
                         triplets_at_q,
                         weights_at_q,
                         grid_address,
                         mesh,
                         band_indices,
                         frequencies,
                         grid_point,
                         filename = None,
                         is_nosym = False):

    if filename == None:
        decay_filename = "decay"
    else:
        decay_filename = "decay%s" % filename
    decay_filename += "-m%d%d%d-" % tuple(mesh)
    decay_filename += "g%d-" % grid_point
    for i in band_indices:
        decay_filename += "b%d" % (i+1)
    if not filename == None:
        decay_filename += ".%s.dat" % filename
    elif is_nosym:
        decay_filename += ".nosym.dat"
    else:
        decay_filename += ".dat"
    w = open(decay_filename, 'w')

    w.write("%10d                            "
            "# Number of triplets\n" % len(triplets_at_q))
    w.write("%10d                            "
            "# Degeneracy\n" % len(band_indices))
    
    for i, j in enumerate(band_indices):
        w.write("%10d %20.10e       # band  freq \n" % (j + 1, frequencies[i]))
    w.write("\n")

    decay_rate_triplets = []
    decay_channels_sum = np.array(
        [d.sum() * weight for d, weight in zip(decay_channels, weights_at_q)]).sum()

    w.write("# %5s %5s %-15s\n" % ("band'", "band''", "decay sum in BZ"))
    decay_rate_bands = []
    pure_sum = 0.0
    for i in range(amplitudes_at_q.shape[2]):
        for j in range(amplitudes_at_q.shape[2]):
            decay_bands_sum = np.dot(decay_channels[:,i,j], weights_at_q)
            decay_rate_bands.append(
                [decay_bands_sum / decay_channels_sum, i, j])
            pure_sum += decay_bands_sum / decay_channels_sum
            w.write("%5d %5d %17.7e %10.5f %%\n" %
                       (i + 1, j + 1, decay_bands_sum,
                        decay_bands_sum * 100 / decay_channels_sum))

    w.write("# Sum       %17.7e %10.5f %%\n\n" %
               (decay_channels_sum, pure_sum*100))

    for i, (d, a, f, tp, weight) in enumerate(zip(decay_channels,
                                                  amplitudes_at_q,
                                                  frequencies_at_q,
                                                  triplets_at_q,
                                                  weights_at_q)):
        sum_d = d.sum()
        decay_rate_triplets.append([sum_d / decay_channels_sum, i])

        w.write("# Triplet %d (%f%%)\n" %
                (i+1, decay_rate_triplets[i][0] * 100))
        w.write(" %4d                                 # weight\n" % weight)
        q0 = grid_address[tp[0]]
        q1 = grid_address[tp[1]]
        q2 = grid_address[tp[2]]
        w.write(" %4d / %-4d %4d / %-4d %4d / %-4d  # q\n" %
                (q0[0], mesh[0], q0[1], mesh[1], q0[2], mesh[2]))
        w.write(" %4d / %-4d %4d / %-4d %4d / %-4d  # q'\n" %
                (q1[0], mesh[0], q1[1], mesh[1], q1[2], mesh[2]))
        w.write(" %4d / %-4d %4d / %-4d %4d / %-4d  # q''\n" %
                (q2[0], mesh[0], q2[1], mesh[1], q2[2], mesh[2]))
        w.write("# %5s %5s    %-15s %-15s %-15s %-5s\n" %
                ("band'", "band''", "freq'", "freq''", "decay", "phi"))

        decay_rate_bands = []
        for j in range(amplitudes_at_q.shape[2]):
            for k in range(amplitudes_at_q.shape[2]):
                decay_rate_bands.append([d[j,k] / sum_d, j, k])

                w.write("%5d %5d %15.7e %15.7e %15.7e %15.7e\n" %
                        (j + 1, k + 1, f[1, j], f[2, k],
                         d[j, k], a[:, j, k].sum() / a.shape[0]))

        if len(decay_rate_bands) > 9:
            w.write("# Top 10 in bands\n")
            decay_rate_bands.sort(mycmp)
            for dr, i, j in decay_rate_bands[:10]:
                w.write("%5d %5d %15.7f%%\n" % (i + 1, j + 1, dr * 100))
        
        w.write("\n")

    if len(decay_rate_triplets) > 9:
        w.write("# Top 10 in triplets\n")
        decay_rate_triplets.sort(mycmp)
        for dr, i in decay_rate_triplets[:10]:
            w.write("%5d %15.7f%%\n" % (i + 1, dr * 100))

    return decay_filename

def mycmp(a, b):
    return cmp(b[0], a[0])

def write_ir_grid_points(mesh,
                         mesh_divs,
                         grid_points,
                         coarse_grid_weights,
                         grid_address,
                         primitive_lattice):
    w = open("ir_grid_points.yaml", 'w')
    w.write("mesh: [ %d, %d, %d ]\n" % tuple(mesh))
    if mesh_divs is not None:
        w.write("mesh_divisors: [ %d, %d, %d ]\n" % tuple(mesh_divs))
    w.write("reciprocal_lattice:\n")
    for vec, axis in zip(primitive_lattice.T, ('a*', 'b*', 'c*')):
        w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" % (tuple(vec) + (axis,)))
    w.write("num_reduced_ir_grid_points: %d\n" % len(grid_points))
    w.write("ir_grid_points:  # [address, weight]\n")

    for g, weight in zip(grid_points, coarse_grid_weights):
        w.write("- grid_point: %d\n" % g)
        w.write("  weight: %d\n" % weight)
        w.write("  grid_address: [ %12d, %12d, %12d ]\n" %
                tuple(grid_address[g]))
        w.write("  q-point:      [ %12.7f, %12.7f, %12.7f ]\n" %
                tuple(grid_address[g].astype('double') / mesh))

def get_forces_from_vasprun_xmls(vaspruns, num_atom, index_shift=0):
    try:
        from lxml import etree
    except ImportError:
        print "You need to install python-lxml."
        sys.exit(1)

    forces = []
    for i, vasprun in enumerate(vaspruns):
        print >> sys.stderr, "%d" % (i + 1 + index_shift),
        force_set = vasp.get_forces_vasprun_xml(
            etree.iterparse(vasp.VasprunWrapper(vasprun), tag='varray'))
        if force_set.shape[0] == num_atom:
            forces.append(force_set)
        else:
            print "\nNumber of forces in vasprun.xml #%d is wrong." % (i + 1)
            sys.exit(1)
            
    print >> sys.stderr
    return np.array(forces)

def get_force_constants_from_vasprun_xmls(vasprun_filenames):
    force_constants_set = []
    for i, filename in enumerate(vasprun_filenames):
        print >> sys.stderr, "%d: %s\n" % (i + 1, filename),
        force_constants_set.append(
            read_force_constant_vasprun_xml(filename)[0])
    print >> sys.stderr
    return force_constants_set

def parse_yaml(file_yaml):
    import yaml
    try:
        from yaml import CLoader as Loader
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    string = open(file_yaml).read()
    data = yaml.load(string, Loader=Loader)
    return data

def parse_force_lines(forcefile, num_atom):
    forces = []
    for line in forcefile:
        if line.strip() == '':
            continue
        if line.strip()[0] == '#':
            continue
        forces.append([float(x) for x in line.strip().split()])
        if len(forces) == num_atom:
            break

    if not len(forces) == num_atom:
        return None
    else:
        return np.array(forces)

def parse_force_constants_lines(fcthird_file, num_atom):
    fc2 = []
    for line in fcthird_file:
        if line.strip() == '':
            continue
        if line.strip()[0] == '#':
            continue
        fc2.append([float(x) for x in line.strip().split()])
        if len(fc2) == num_atom ** 2 * 3:
            break

    if not len(fc2) == num_atom ** 2 * 3:
        return None
    else:
        return np.array(fc2).reshape(num_atom, num_atom, 3, 3)

def parse_disp_yaml_to_disp_dataset(filename="disp.yaml"):
    dataset = parse_yaml(filename)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    new_first_atoms = []
    for first_atoms in dataset['displacements']:
        first_atoms['atom'] -= 1
        atom1 = first_atoms['atom']
        disp1 = first_atoms['displacement']
        new_first_atoms.append({'number': atom1, 'displacement': disp1})
    new_dataset['first_atoms'] = new_first_atoms
    return new_dataset
                       
def parse_disp_fc2_yaml(filename="disp_fc2.yaml"):
    dataset = parse_yaml(filename)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    new_first_atoms = []
    for first_atoms in dataset['first_atoms']:
        first_atoms['number'] -= 1
        atom1 = first_atoms['number']
        disp1 = first_atoms['displacement']
        new_first_atoms.append({'number': atom1, 'displacement': disp1})
    new_dataset['first_atoms'] = new_first_atoms

    return new_dataset
        
def parse_disp_fc3_yaml(filename="disp_fc3.yaml"):
    dataset = parse_yaml(filename)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    if 'cutoff_distance' in dataset:
        new_dataset['cutoff_distance'] = dataset['cutoff_distance']
    new_first_atoms = []
    for first_atoms in dataset['first_atoms']:
        first_atoms['number'] -= 1
        atom1 = first_atoms['number']
        disp1 = first_atoms['displacement']
        new_second_atoms = []
        for second_atom in first_atoms['second_atoms']:
            second_atom['number'] -= 1
            atom2 = second_atom['number']
            if 'included' in second_atom:
                included = second_atom['included']
            else:
                included = True
            for disp2 in second_atom['displacements']:
                new_second_atoms.append({'number': atom2,
                                         'displacement': disp2,
                                         'included': included})
        new_first_atoms.append({'number': atom1,
                                'displacement': disp1,
                                'second_atoms': new_second_atoms})
    new_dataset['first_atoms'] = new_first_atoms

    return new_dataset
    
def parse_disp_fc4_yaml(filename="disp_fc4.yaml"):
    dataset = parse_yaml(filename)
    natom = dataset['natom']
    new_dataset = {}
    new_dataset['natom'] = natom
    new_first_atoms = []
    for first_atoms in dataset['first_atoms']:
        first_atoms['number'] -= 1
        atom1 = first_atoms['number']
        disp1 = first_atoms['displacement']
        new_second_atoms = []
        for second_atoms in first_atoms['second_atoms']:
            second_atoms['number'] -= 1
            atom2 = second_atoms['number']
            disp2 = second_atoms['displacement']
            new_third_atoms = []
            for third_atoms in second_atoms['third_atoms']:
                third_atoms['number'] -= 1
                atom3 = third_atoms['number']
                for disp3 in third_atoms['displacements']:
                    new_third_atoms.append(
                        {'number': atom3, 'displacement': disp3})
            new_second_atoms.append(
                {'number': atom2,
                 'displacement': disp2,
                 'third_atoms': new_third_atoms})
        new_first_atoms.append(
            {'number': atom1,
             'displacement': disp1,
             'second_atoms': new_second_atoms})
    new_dataset['first_atoms'] = new_first_atoms

    return new_dataset
    
def parse_DELTA_FORCES(disp_dataset,
                       filethird='FORCES_THIRD',
                       filesecond='FORCES_SECOND'):
    forces_third = open(filethird, 'r')
    forces_second = open(filesecond, 'r')
    num_atom = disp_dataset['natom']

    for disp1 in disp_dataset['first_atoms']:
        second_forces = parse_force_lines(forces_second, num_atom)
        for disp2 in disp1['second_atoms']:
            third_forces = parse_force_lines(forces_third, num_atom)
            disp2['delta_forces'] = third_forces - second_forces

def parse_DELTA_FORCES_FOURTH(disp_dataset,
                              file4='FORCES_FOURTH',
                              file3='FORCES_THIRD',
                              file2='FORCES_SECOND'):
    f4 = open(file4, 'r')
    f3 = open(file3, 'r')
    f2 = open(file2, 'r')
    num_atom = disp_dataset['natom']

    for disp1 in disp_dataset['first_atoms']:
        second_forces = parse_force_lines(f2, num_atom)
        disp1['forces'] = second_forces
        for disp2 in disp1['second_atoms']:
            third_forces = parse_force_lines(f3, num_atom)
            disp2['delta_forces'] = third_forces - second_forces
            for disp3 in disp2['third_atoms']:
                fourth_forces = parse_force_lines(f4, num_atom)
                disp3['delta_forces'] = fourth_forces - third_forces

def parse_FORCES_FOURTH(disp_dataset,
                        file4='FORCES_FOURTH',
                        file3='FORCES_THIRD',
                        file2='FORCES_SECOND'):
    f4 = open(file4, 'r')
    f3 = open(file3, 'r')
    f2 = open(file2, 'r')
    num_atom = disp_dataset['natom']

    for disp1 in disp_dataset['first_atoms']:
        second_forces = parse_force_lines(f2, num_atom)
        disp1['forces'] = second_forces
        for disp2 in disp1['second_atoms']:
            third_forces = parse_force_lines(f3, num_atom)
            disp2['forces'] = third_forces
            for disp3 in disp2['third_atoms']:
                fourth_forces = parse_force_lines(f4, num_atom)
                disp3['forces'] = fourth_forces

def parse_FORCES_THIRD(disp_dataset,
                       file3='FORCES_THIRD',
                       file2='FORCES_SECOND'):
    f3 = open(file3, 'r')
    f2 = open(file2, 'r')
    num_atom = disp_dataset['natom']

    for disp1 in disp_dataset['first_atoms']:
        second_forces = parse_force_lines(f2, num_atom)
        disp1['forces'] = second_forces
        for disp2 in disp1['second_atoms']:
            third_forces = parse_force_lines(f3, num_atom)
            disp2['forces'] = third_forces

def parse_FORCES_SECOND(disp_dataset, filename="FORCES_SECOND"):
    f2 = open(filename, 'r')
    num_atom = disp_dataset['natom']
    for disp1 in disp_dataset['first_atoms']:
        second_forces = parse_force_lines(f2, num_atom)
        disp1['forces'] = second_forces

def parse_FORCES_FC2(disp_dataset, filename="FORCES_FC2"):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    f2 = open(filename, 'r')
    forces_fc2 = [parse_force_lines(f2, num_atom) for i in range(num_disp)]
    f2.close()
    return forces_fc2

def parse_FORCES_FC3(disp_dataset, filename="FORCES_FC3"):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    for disp1 in disp_dataset['first_atoms']:
        num_disp += len(disp1['second_atoms'])
    f3 = open(filename, 'r')
    forces_fc3 = [parse_force_lines(f3, num_atom) for i in range(num_disp)]
    f3.close()
    return forces_fc3

def parse_FORCE_SETS_with_disp_dataset(disp_dataset, filename="FORCE_SETS"):
    f2 = open(filename, 'r')
    num_atom = disp_dataset['natom']
    force_sets = parse_FORCE_SETS(num_atom)
    for disp1, force_set in zip(disp_dataset['first_atoms'], force_sets):
        disp1['forces'] = force_set.get_forces()

def parse_DELTA_FC2_SETS(disp_dataset,
                         filename='DELTA_FC2_SETS'):
    fc2_file = open(filename, 'r')
    delta_fc2s = []
    num_atom = disp_dataset['natom']
    for first_disp in disp_dataset['first_atoms']:
        first_disp['delta_fc2'] = parse_force_constants_lines(fc2_file, num_atom)

def parse_DELTA_FC2_FOURTH_SETS(disp_dataset,
                                filename='DELTA_FC2_FOURTH_SETS'):
    fc2_file = open(filename, 'r')
    delta_fc2s = []
    num_atom = disp_dataset['natom']
    for first_disp in disp_dataset['first_atoms']:
        for second_disp in first_disp['second_atoms']:
            second_disp['delta_fc2'] = parse_force_constants_lines(fc2_file,
                                                                   num_atom)

def parse_QPOINTS3(filename='QPOINTS3'):
    f = open(filename)
    num = int(f.readline().strip())
    count = 0
    qpoints3 = []
    for line in f:
        line_array = [float(x) for x in line.strip().split()]

        if len(line_array) < 9:
            print "QPOINTS3 format is invalid."
            raise ValueError
        else:
            qpoints3.append(line_array[0:9])

        count += 1
        if count == num:
            break

    return np.array(qpoints3)

def parse_fc3(num_atom, filename='fc3.dat'):
    f = open(filename)
    fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                f.readline()
                for l in range(3):
                    fc3[i, j, k, l] = [
                        [float(x) for x in f.readline().split()],
                        [float(x) for x in f.readline().split()],
                        [float(x) for x in f.readline().split()]]
                    f.readline()
    return fc3

def parse_fc2(num_atom, filename='fc2.dat'):
    f = open(filename)
    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            f.readline()
            fc2[i, j] = [[float(x) for x in f.readline().split()],
                         [float(x) for x in f.readline().split()],
                         [float(x) for x in f.readline().split()]]
            f.readline()

    return fc2

def parse_triplets(filename):
    f = open(filename)
    triplets = []
    weights = []
    for line in f:
        if line.strip()[0] == "#":
            continue

        line_array = [int(x) for x in line.split()]
        triplets.append(line_array[:3])
        weights.append(line_array[3])

    return np.array(triplets), np.array(weights)
    
def parse_grid_address(filename):
    f = open(filename, 'r')
    grid_address = []
    for line in f:
        if line.strip()[0] == "#":
            continue

        line_array = [int(x) for x in line.split()]
        grid_address.append(line_array[1:4])

    return np.array(grid_address)

if __name__ == '__main__':
    import numpy as np
    import sys
    from anharmonic.file_IO import parse_fc3, parse_fc2
    from optparse import OptionParser

    parser = OptionParser()
    parser.set_defaults(num_atom = None,
                        symprec = 1e-3)
    parser.add_option("-n", dest="num_atom", type="int",
                      help="number of atoms")
    parser.add_option("-s", dest="symprec", type="float",
                      help="torrelance")
    (options, args) = parser.parse_args()
    
    num_atom = options.num_atom
    
    fc2_1 = parse_fc2(num_atom, filename=args[0])
    fc2_2 = parse_fc2(num_atom, filename=args[1])
    
    fc3_1 = parse_fc3(num_atom, filename=args[2])
    fc3_2 = parse_fc3(num_atom, filename=args[3])
    
    print "fc2",
    fc2_count = 0
    for i in range(num_atom):
        for j in range(num_atom):
            if (abs(fc2_1[i, j] - fc2_2[i, j]) > options.symprec).any():
                print i + 1,j + 1
                print fc2_1[i, j]
                print fc2_2[i, j]
                fc2_count += 1
    if fc2_count == 0:
        print "OK"
    else:
        print fc2_count

    print "fc3",
    fc3_count = 0
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                if (abs(fc3_1[i, j, k] - fc3_2[i, j, k]) > options.symprec).any():
                    print i + 1, j + 1, k + 1
                    print fc3_1[i, j, k]
                    print fc3_2[i, j, k]
                    print
                    fc3_count += 1
    if fc3_count == 0:
        print "OK"
    else:
        print fc3_count
