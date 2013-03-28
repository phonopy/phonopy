import sys
import numpy as np
import h5py
from phonopy.structure.atoms import Atoms
from phonopy.interface import vasp
from phonopy.hphonopy.file_IO import write_FORCE_SETS_vasp, read_force_constant_vasprun_xml

###########
#
# File I/O
#
###########

#
# Output
#
def write_supercells_with_displacements(supercell,
                                        double_displacements,
                                        amplitude=None):
    if amplitude==None:
        distance = 0.01
    else:
        distance = amplitude
    
    # YAML
    file = open('disp.yaml', 'w')
    file.write("natom: %d\n" %  supercell.get_number_of_atoms())
    file.write("num_first_displacements: %d\n" %  len(double_displacements))
    num_second = 0
    for d1 in double_displacements:
        for d2 in d1['second_atoms']:
            num_second += len(d2['directions'])
    file.write("num_second_displacements: %d\n" %  num_second)
    file.write("first_atoms:\n")
    lattice = supercell.get_cell()
    lattice_inv = np.linalg.inv(lattice)
    count1 = 1
    count2 = len(double_displacements) + 1
    for disp1 in double_displacements:
        disp_cartesian3 = np.dot(disp1['direction'], lattice)
        disp_cartesian3 = disp_cartesian3 / np.linalg.norm(disp_cartesian3) * distance
        positions = supercell.get_positions()
        positions[disp1['number']] += disp_cartesian3
        atoms = Atoms(numbers = supercell.get_atomic_numbers(),
                      masses = supercell.get_masses(),
                      positions = positions,
                      cell = lattice,
                      pbc = True)
        vasp.write_vasp('POSCAR-%04d' % count1, atoms, direct=True)

        # YAML
        file.write("- number: %5d\n" % (disp1['number'] + 1))
        file.write("  displacement:\n")
        file.write("    [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                   (disp_cartesian3[0], disp_cartesian3[1], disp_cartesian3[2], count1))
        file.write("  second_atoms:\n")
        count1 += 1

        for disp2 in disp1['second_atoms']:
            # YAML
            file.write("  - number: %5d\n" % (disp2['number'] + 1))
            file.write("    displacements:\n")
            
            for direction in disp2['directions']:
                disp_cartesian2 = np.dot(direction, lattice)
                disp_cartesian2 = disp_cartesian2 / np.linalg.norm(disp_cartesian2) * distance
                positions = supercell.get_positions()
                positions[disp1['number']] += disp_cartesian3
                positions[disp2['number']] += disp_cartesian2
                atoms = Atoms(numbers = supercell.get_atomic_numbers(),
                               masses = supercell.get_masses(),
                               positions = positions,
                               cell = lattice,
                               pbc = True)
                vasp.write_vasp('POSCAR-%04d' % count2, atoms, direct=True)

                # YAML
                file.write("    - [ %20.16f,%20.16f,%20.16f ] # %d \n" %
                           (disp_cartesian2[0], disp_cartesian2[1], disp_cartesian2[2], count2))
                count2 += 1

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

def write_FORCES_THIRD(vaspruns,
                       displacements,
                       forces_third='FORCES_THIRD',
                       forces_second='FORCES_SECOND',
                       force_sets='FORCE_SETS'):
    natom = displacements['natom']
    num_disp1 = len(displacements['first_atoms'])
    disp_datasets = []
    set_of_forces = get_forces_from_vasprun_xmls(vaspruns, natom)
    filethird = open(forces_third, 'w')
    filesecond = open(forces_second, 'w')

    for i, disp1 in enumerate(displacements['first_atoms']):
        filesecond.write("# File: %-5d\n" % (i + 1))
        filesecond.write("# %-5d " % (disp1['number'] + 1))
        filesecond.write("%20.16f %20.16f %20.16f\n" %
                         tuple(disp1['displacement']))
        for f in set_of_forces[i]:
            filesecond.write("%15.10f %15.10f %15.10f\n" % (tuple(f)))
        
        disp_datasets.append([disp1['number'], disp1['displacement']])

    count = num_disp1
    for i, disp1 in enumerate(displacements['first_atoms']):
        for disp2 in disp1['second_atoms']:
            for d in disp2['displacements']:
                filethird.write("# File: %-5d\n" % (count + 1))
                filethird.write("# %-5d " % (disp1['number'] + 1))
                filethird.write("%20.16f %20.16f %20.16f\n" %
                                 tuple(disp1['displacement']))
                filethird.write("# %-5d " % (disp2['number'] + 1))
                filethird.write("%20.16f %20.16f %20.16f\n" % tuple(d))

                for f in set_of_forces[count]:
                    filethird.write("%15.10f %15.10f %15.10f\n" % (tuple(f)))
                count += 1

    write_FORCE_SETS_vasp(vaspruns[:num_disp1],
                          disp_datasets,
                          natom,
                          force_sets)

def write_DELTA_FC2_SETS(vaspruns,
                         displacements,
                         fc2_file='fc2.dat',
                         dfc2_file='DELTA_FC2_SETS'):
    fc2_set = get_force_constants_from_vasprun_xmls(vaspruns)
    w = open(dfc2_file, 'w')
    perfect_fc2 = fc2_set.pop(0)
    write_fc2_dat(perfect_fc2, fc2_file)
    for i, (dfc2, first_disp) in enumerate(
        zip(fc2_set, displacements['first_atoms'])):
        w.write("# File: %d\n" % (i + 1))
        w.write("# %-5d " % (first_disp['number'] + 1))
        w.write("%20.16f %20.16f %20.16f\n" %
                tuple(first_disp['displacement']))
        for j in range(dfc2.shape[0]):
            for k in range(dfc2.shape[1]):
                w.write("# %d - %d\n" % (j + 1, k + 1))
                for vec in (dfc2 - perfect_fc2)[j, k]:
                    w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))

def write_FC2_FOURTH_SETS(vaspruns,
                          displacements,
                          fc2fc4_file='FC2_FOURTH_SETS'):
    """Write displaced fc2 for fc4 from vasprun.xml's"""
    
    fc2_set = get_force_constants_from_vasprun_xmls(vaspruns)
    w = open(fc2fc4_file, 'w')
    count = 0
    for first_disp in displacements['first_atoms']:
        count += 1
        dfc2 = fc2_set.pop(0)

        w.write("# File: %d\n" % count)
        w.write("# %-5d " % (first_disp['number'] + 1))
        w.write("%20.16f %20.16f %20.16f\n" %
                tuple(first_disp['displacement']))

        for i in range(dfc2.shape[0]):
            for j in range(dfc2.shape[1]):
                w.write("# %d - %d\n" % (i + 1, j + 1))
                for vec in dfc2[i, j]:
                    w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))

    for first_disp in displacements['first_atoms']:
        for second_disp in first_disp['second_atoms']:
            for disp in second_disp['displacements']:
                count += 1
                dfc2 = fc2_set.pop(0)
                
                w.write("# File: %d\n" % count)
                w.write("# %-5d" % (first_disp['number'] + 1))
                w.write("%20.16f %20.16f %20.16f\n" %
                    tuple(first_disp['displacement']))
                w.write("# %-5d" % (second_disp['number'] + 1))
                w.write("%20.16f %20.16f %20.16f\n" % tuple(disp))
            
                for i in range(dfc2.shape[0]):
                    for j in range(dfc2.shape[1]):
                        w.write("# %d - %d\n" % (i + 1, j + 1))
                        for vec in dfc2[i, j]:
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

def write_fc2_dat(force_constants, filename='fc2.dat'):
    w = open(filename, 'w')
    for i, fcs in enumerate(force_constants):
        for j, fcb in enumerate(fcs):
            w.write(" %d - %d\n" % (i+1, j+1))
            for vec in fcb:
                w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
            w.write("\n")

def write_triplets(triplets, weights, mesh, filename='triplets.dat'):
    w = open(filename, 'w')
    w.write("# Triplets for %dx%dx%d mesh (address_1, address_2, address_3, weight)\n" % tuple(mesh))
    for weight, q3 in zip(weights, triplets):
        w.write("%10d %10d %10d %10d\n" % (q3[0], q3[1], q3[2], weight))

def write_grid_address(grid_address, mesh, filename='grid_points.dat'):
    w = open(filename, 'w')
    w.write("# Grid points for %dx %dx%d mesh"
            "(address, grid_a, grid_b, grid_c)\n" % tuple(mesh))
    for i, q in enumerate(grid_address):
        w.write("%10d %10d %10d %10d\n" % (i, q[0], q[1], q[2]))

def write_amplitudes(amplitudes, frequencies, triplet, weight, mesh, filename):
    w = open(filename, 'w')
    w.write("# mesh [ %d %d %d ]\n" % tuple(mesh))
    w.write("# weight %d\n" % weight)
    w.write("# triplet [ %d %d %d ]\n" % tuple(triplet))
    w.write("#%6s %6s %6s %20s %20s %20s %20s\n" %
            ("band1", "band2", "band3", "omega1", "omega2", "omega3", "amplitude"))

    num_band = amplitudes.shape[0]
    for j1 in range(num_band):
        if triplet[0] == 0 and (j1 < 3) :
            continue
        f1 = frequencies[0][j1]
        for j2 in range(num_band):
            if triplet[1] == 0 and j2 < 3:
                continue
            f2 = frequencies[1][j2]
            for j3 in range(num_band):
                if triplet[2] == 0 and j3 < 3:
                    continue
                f3 = frequencies[2][j3]
                w.write("% 6d %6d %6d %20.15f %20.15f %20.15f %20.15e\n" % 
                           (j1+1, j2+1, j3+1, f1, f2, f3, amplitudes[j1, j2, j3]))
            w.write("\n")
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
        gammas_filename += "b%d" % i

    if not filename == None:
        gammas_filename += ".%s" % filename
    elif is_nosym:
        gammas_filename += ".nosym"
    gammas_filename += ".dat"

    w = open(gammas_filename, 'w')
    for freq, g in zip(frequencies, gammas):
        w.write("%15.7f %20.15e\n" % (freq, g))
    w.close()

def write_jointDOS(gp,
                   mesh,
                   frequencies,
                   jdos,
                   filename=None,
                   is_nosym=False):
    if filename==None:
        if is_nosym:
            jdos_filename = "jdos-m%d%d%d-g%d.nosym.dat" % (mesh[0],
                                                            mesh[1],
                                                            mesh[2],
                                                            gp)
        else:
            jdos_filename = "jdos-m%d%d%d-g%d.dat" % (mesh[0],
                                                      mesh[1],
                                                      mesh[2],
                                                      gp)
    else:
        jdos_filename = "jdos-m%d%d%d-g%d.%s.dat" % (mesh[0],
                                                     mesh[1],
                                                     mesh[2],
                                                     gp,
                                                     filename)
        
    w = open(jdos_filename, 'w')
    for omega, val in zip(frequencies, jdos):
        w.write("%15.7f %20.15e\n" % (omega, val))
    w.close()

def write_fwhm(gp,
               band_indices,
               temps,
               fwhms,
               mesh,
               sigma=None,
               is_nosym=False,
               filename=None):

    fwhm_filename = "fwhm"

    fwhm_filename += "-m%d%d%d-g%d-" % (mesh[0],
                                        mesh[1],
                                        mesh[2],
                                        gp)
    if sigma is not None:
        fwhm_filename += ("s%f" % sigma).rstrip('0') + "-"

    for i in band_indices:
        fwhm_filename += "b%d" % i

    if not filename == None:
        fwhm_filename += ".%s" % filename
    elif is_nosym:
        fwhm_filename += ".nosym"
    fwhm_filename += ".dat"

    w = open(fwhm_filename, 'w')
    for v, t in zip(fwhms, temps):
        w.write("%15.7f %20.15e\n" % (t, v))
    w.close()
    
def write_kappa(kappa,
                temperatures,
                mesh,
                mesh_divisors=None,
                gamma=None,
                grid_point=None,
                sigma=None,
                filename=None):
    kappa_filename = "kappa"
    suffix = "-m%d%d%d" % tuple(mesh)
    if (mesh_divisors != 1).any():
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
    if gamma is not None:
        print "and gamma",
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
    print "%s" % kappa_filename
    w = open(kappa_filename, 'w')
    if gamma is not None:
        w.write("# temp   kappa          gamma(band1) ...\n")
        for t, k, g in zip(temperatures, kappa, gamma):
            w.write("%6.1f %15.7e" % (t, k))
            for g_band in g:
                w.write(" %15.7e" % g_band)
            w.write("\n")
        w.close()
    else:
        w.write("# temp   kappa\n")
        for t, k in zip(temperatures, kappa):
            w.write("%6.1f %.5f\n" % (t, k))
        w.close()

def write_gamma_to_hdf5(gammas,
                        kappas,
                        temperatures,
                        frequencies,
                        mesh,
                        mesh_divisors=None,
                        grid_point=None,
                        sigma=None,
                        filename=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    if (mesh_divisors != 1).any():
        suffix += "-d%d%d%d" % tuple(mesh_divisors)
    sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if sigma is not None:
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename
    print "Gamma",
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
    print "%s" % ("gamma" + suffix + ".hdf5"),
    w = h5py.File("gamma" + suffix + ".hdf5", 'w')
    w.create_dataset('gammas', data=gammas)
    w.create_dataset('kappas', data=kappas)
    w.create_dataset('frequencies', data=frequencies)
    w.create_dataset('temperatures', data=temperatures)
    w.close()





        
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

# 
# Input
#
def get_forces_from_vasprun_xmls(vaspruns, num_atom):
    try:
        from lxml import etree
    except ImportError:
        print "You need to install python-lxml."
        sys.exit(1)

    forces = []
    for i, vasprun in enumerate(vaspruns):
        print >> sys.stderr, "%d" % (i + 1),
        force_set = vasp.get_forces_vasprun_xml(
            etree.iterparse(vasp.VasprunWrapper(vasprun), tag='varray'))
        if force_set.shape[0] == num_atom:
            forces.append(force_set)
        else:
            print "\nNumber of forces in vasprun.xml #%d is wrong." % (i+1)
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
                       
def parse_disp_yaml(filename="disp.yaml"):
    dataset = parse_yaml(filename)
    for first_atoms in dataset['first_atoms']:
        first_atoms['number'] -= 1
        for second_atoms in first_atoms['second_atoms']:
            second_atoms['number'] -= 1

    return dataset
    
def parse_DELTA_FORCES(displacements,
                       filethird='FORCES_THIRD',
                       filesecond='FORCES_SECOND'):

    forces_third = open(filethird, 'r')
    forces_second = open(filesecond, 'r')
    sets_of_delta_forces = []
    num_atom = displacements['natom']

    for i, disp1 in enumerate(displacements['first_atoms']):
        second_forces = parse_force_lines(forces_second, num_atom)
        for disp2 in disp1['second_atoms']:
            disp2['delta_forces'] = []            
            for j in range(len(disp2['displacements'])):
                third_forces = parse_force_lines(forces_third, num_atom)
                disp2['delta_forces'].append(third_forces - second_forces)


def parse_FORCES_THIRD(displacements,
                       filename='FORCES_THIRD'):
    forcefile = open(filename, 'r')
    num_atom = displacements['natom']
    num_disp = 0
    for disp1 in displacements['first_atoms']:
        for disp2 in disp1['second_atoms']:
            num_disp += len(disp2['displacements'])
    sets_of_forces = []
    for i in range(num_disp):
        sets_of_forces.append(parse_force_lines(forcefile, num_atom))
    
    return np.array(sets_of_forces)

def parse_DELTA_FC2_SETS(displacements,
                         filename='DELTA_FC2_SETS'):
    fc2_file = open(filename, 'r')
    delta_fc2s = []
    num_atom = displacements['natom']
    for first_disp in displacements['first_atoms']:
        first_disp['delta_fc2'] = parse_force_constants_lines(fc2_file,
                                                              num_atom)

def parse_FC2_FOURTH_SETS(displacements,
                          filename='FC2_FOURTH_SETS'):
    fc2_file = open(filename, 'r')
    delta_fc2s = []
    num_atom = displacements['natom']
    for first_disp in displacements['first_atoms']:
        first_disp['fc2'] = parse_force_constants_lines(fc2_file, num_atom)
    for first_disp in displacements['first_atoms']:
        for second_disp in first_disp['second_atoms']:
            second_disp['delta_fc2'] = []
            for disp in second_disp['displacements']:
                second_disp['delta_fc2'].append(
                    parse_force_constants_lines(fc2_file, num_atom) -
                    first_disp['fc2'])

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

def parse_kappa(filename):
    f = open(filename)
    temps = []
    kappa = []
    for line in f:
        if line.strip()[0] is "#":
            continue
        x = line.split()
        temps.append(float(x[0]))
        kappa.append(float(x[1]))
    return np.array(temps), np.array(kappa)

def parse_gamma(filename):
    f = open(filename)
    temps = []
    gamma = []
    for line in f:
        if line.strip()[0] is "#":
            continue
        x = line.split()
        temps.append(float(x[0]))
        gamma.append([float(g) for g in x[2:]])
    return np.array(temps), np.array(gamma)


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
