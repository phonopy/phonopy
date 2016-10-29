import os
import numpy as np
import h5py
from phonopy.interface.vasp import get_forces_from_vasprun_xmls

def write_cell_yaml(w, supercell):
    w.write("lattice:\n")
    for axis in supercell.get_cell():
        w.write("- [ %20.15f,%20.15f,%20.15f ]\n" % tuple(axis))
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_scaled_positions()
    w.write("atoms:\n")
    for i, (s, v) in enumerate(zip(symbols, positions)):
        w.write("- symbol: %-2s # %d\n" % (s, i+1))
        w.write("  position: [ %18.14f,%18.14f,%18.14f ]\n" % tuple(v))

def write_disp_fc4_yaml(dataset, supercell, filename='disp_fc4.yaml'):
    w = open(filename, 'w')
    w.write("natom: %d\n" %  dataset['natom'])

    num_first = len(dataset['first_atoms'])
    w.write("num_first_displacements: %d\n" %  num_first)

    num_second = 0
    for d1 in dataset['first_atoms']:
        num_second += len(d1['second_atoms'])
    w.write("num_second_displacements: %d\n" %  num_second)

    num_third = 0
    for d1 in dataset['first_atoms']:
        for d2 in d1['second_atoms']:
            num_third += len(d2['third_atoms'])
    w.write("num_third_displacements: %d\n" %  num_third)

    w.write("first_atoms:\n")
    count1 = 1
    count2 = num_first + 1
    count3 = num_first + num_second + 1
    for disp1 in dataset['first_atoms']:
        disp_cart1 = disp1['displacement']
        w.write("- number: %5d\n" % (disp1['number'] + 1))
        w.write("  displacement:\n")
        w.write("    [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                (disp_cart1[0], disp_cart1[1], disp_cart1[2], count1))
        w.write("  second_atoms:\n")
        count1 += 1
        for disp2 in disp1['second_atoms']:
            w.write("  - number: %5d\n" % (disp2['number'] + 1))
            w.write("    displacement:\n")
            disp_cart2 = disp2['displacement']
            w.write("      [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                    (disp_cart2[0], disp_cart2[1], disp_cart2[2], count2))
            w.write("    third_atoms:\n")
            count2 += 1
            atom3 = -1
            for disp3 in disp2['third_atoms']:
                if atom3 != disp3['number']:
                    atom3 = disp3['number']
                    w.write("    - number: %5d\n" % (atom3 + 1))
                    w.write("      displacements:\n")
                disp_cart3 = disp3['displacement']
                w.write("      - [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                        (disp_cart3[0], disp_cart3[1], disp_cart3[2], count3))
                count3 += 1

    write_cell_yaml(w, supercell)

    w.close()

    return num_first + num_second + num_third

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

    write_cell_yaml(w, supercell)

    w.close()

    return num_first + num_second, num_disp_files

def write_disp_fc2_yaml(dataset, supercell, filename='disp_fc2.yaml'):
    w = open(filename, 'w')
    w.write("natom: %d\n" %  dataset['natom'])

    num_first = len(dataset['first_atoms'])
    w.write("num_first_displacements: %d\n" %  num_first)
    w.write("first_atoms:\n")
    for i, disp1 in enumerate(dataset['first_atoms']):
        disp_cart1 = disp1['displacement']
        w.write("- number: %5d\n" % (disp1['number'] + 1))
        w.write("  displacement:\n")
        w.write("    [%20.16f,%20.16f,%20.16f ] # %05d\n" %
                (disp_cart1[0], disp_cart1[1], disp_cart1[2], i + 1))

    write_cell_yaml(w, supercell)

    w.close()

    return num_first

def write_FORCES_FC4_vasp(vaspruns,
                          disp_dataset,
                          filename='FORCES_FC4'):
    natom = disp_dataset['natom']
    forces = get_forces_from_vasprun_xmls(vaspruns, natom)
    w = open(filename, 'w')
    write_FORCES_FC4(disp_dataset, forces, fp=w)
    w.close()

def write_FORCES_FC3_vasp(vaspruns,
                          disp_dataset,
                          filename='FORCES_FC3'):
    natom = disp_dataset['natom']
    forces = get_forces_from_vasprun_xmls(vaspruns, natom)
    w = open(filename, 'w')
    write_FORCES_FC3(disp_dataset, forces, fp=w)
    w.close()

def write_FORCES_FC2_vasp(vaspruns,
                          disp_dataset,
                          filename='FORCES_FC2'):
    natom = disp_dataset['natom']
    forces_fc2 = get_forces_from_vasprun_xmls(vaspruns, natom)
    w = open(filename, 'w')
    write_FORCES_FC2(disp_dataset, forces_fc2=forces_fc2, fp=w)
    w.close()

def write_FORCES_FC2(disp_dataset,
                     forces_fc2=None,
                     fp=None,
                     filename="FORCES_FC2"):
    if fp is None:
        w = open(filename, 'w')
    else:
        w = fp

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        w.write("# File: %-5d\n" % (i + 1))
        w.write("# %-5d " % (disp1['number'] + 1))
        w.write("%20.16f %20.16f %20.16f\n" % tuple(disp1['displacement']))
        if forces_fc2 is None:
            force_set = disp1['forces']
        else:
            force_set = forces_fc2[i]
        for forces in force_set:
            w.write("%15.10f %15.10f %15.10f\n" % tuple(forces))

def write_FORCES_FC3(disp_dataset, forces_fc3, fp=None, filename="FORCES_FC3"):
    if fp is None:
        w = open(filename, 'w')
    else:
        w = fp

    natom = disp_dataset['natom']
    num_disp1 = len(disp_dataset['first_atoms'])
    count = num_disp1
    file_count = num_disp1

    write_FORCES_FC2(disp_dataset, forces_fc2=forces_fc3, fp=w)

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        atom1 = disp1['number']
        for disp2 in disp1['second_atoms']:
            atom2 = disp2['number']
            w.write("# File: %-5d\n" % (count + 1))
            w.write("# %-5d " % (atom1 + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp1['displacement']))
            w.write("# %-5d " % (atom2 + 1))
            w.write("%20.16f %20.16f %20.16f\n" % tuple(disp2['displacement']))

            # For supercell calculation reduction
            included = True
            if 'included' in disp2:
                included = disp2['included']
            if included:
                for forces in forces_fc3[file_count]:
                    w.write("%15.10f %15.10f %15.10f\n" % tuple(forces))
                file_count += 1
            else:
                # for forces in forces_fc3[i]:
                #     w.write("%15.10f %15.10f %15.10f\n" % (tuple(forces)))
                for j in range(natom):
                    w.write("%15.10f %15.10f %15.10f\n" % (0, 0, 0))
            count += 1

def write_FORCES_FC4(disp_dataset, forces_fc4, fp=None, filename="FORCES_FC4"):
    if fp is None:
        w = open(filename, 'w')
    else:
        w = fp

    natom = disp_dataset['natom']
    num_disp1 = len(disp_dataset['first_atoms'])
    num_disp2 = 0
    for disp1 in disp_dataset['first_atoms']:
        num_disp2 += len(disp1['second_atoms'])
    count = num_disp1 + num_disp2

    write_FORCES_FC3(disp_dataset, forces_fc3=forces_fc4, fp=w)

    for i, disp1 in enumerate(disp_dataset['first_atoms']):
        atom1 = disp1['number']
        for disp2 in disp1['second_atoms']:
            atom2 = disp2['number']
            for disp3 in disp2['third_atoms']:
                atom3 = disp3['number']
                w.write("# File: %-5d\n" % (count + 1))
                w.write("# %-5d " % (atom1 + 1))
                w.write("%20.16f %20.16f %20.16f\n" %
                        tuple(disp1['displacement']))
                w.write("# %-5d " % (atom2 + 1))
                w.write("%20.16f %20.16f %20.16f\n" %
                        tuple(disp2['displacement']))
                w.write("# %-5d " % (atom3 + 1))
                w.write("%20.16f %20.16f %20.16f\n" %
                        tuple(disp3['displacement']))
                for forces in forces_fc4[count]:
                    w.write("%15.10f %15.10f %15.10f\n" % tuple(forces))
                count += 1

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
                w.write(" %d - %d - %d  (%f)\n" % (i + 1, j + 1, k + 1,
                                                   np.abs(tensor3).sum()))
                for tensor2 in tensor3:
                    for vec in tensor2:
                        w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
                    w.write("\n")

def write_fc4_dat(fc4, filename='fc4.dat'):
    w = open(filename, 'w')
    for (i, j, k, l) in list(np.ndindex(fc4.shape[:4])):
        tensor4 = fc4[i, j, k, l]
        w.write(" %d - %d - %d - %d (%f)\n" % (i + 1, j + 1, k + 1, l + 1,
                                               np.abs(tensor4).sum()))
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
    with h5py.File(filename, 'w') as w:
        w.create_dataset('fc3', data=force_constants_third)

def read_fc3_from_hdf5(filename='fc3.hdf5'):
    with h5py.File(filename, 'r') as f:
        fc3 = f['fc3'][:]
        return fc3
    return None

def write_fc2_dat(force_constants, filename='fc2.dat'):
    w = open(filename, 'w')
    for i, fcs in enumerate(force_constants):
        for j, fcb in enumerate(fcs):
            w.write(" %d - %d\n" % (i+1, j+1))
            for vec in fcb:
                w.write("%20.14f %20.14f %20.14f\n" % tuple(vec))
            w.write("\n")

def write_fc2_to_hdf5(force_constants, filename='fc2.hdf5'):
    with h5py.File(filename, 'w') as w:
        w.create_dataset('fc2', data=force_constants)

def read_fc2_from_hdf5(filename='fc2.hdf5'):
    with h5py.File(filename, 'r') as f:
        if 'fc2' in f.keys():
            fc2 = f['fc2'][:]
        elif 'force_constants' in f.keys():
            fc2 = f['force_constants'][:]
        else:
            fc2 = None
        return fc2
    return None

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

def write_grid_address_to_hdf5(grid_address,
                               mesh,
                               grid_mapping_table,
                               filename=None):
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "grid_address" + suffix + ".hdf5"
    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('mesh', data=mesh)
        w.create_dataset('grid_address', data=grid_address)
        w.create_dataset('grid_mapping_table', data=grid_mapping_table)
        return full_filename
    return None

def write_freq_shifts_to_hdf5(freq_shifts, filename='freq_shifts.hdf5'):
    with h5py.File(filename, 'w') as w:
        w.create_dataset('shift', data=freq_shifts)

def write_imag_self_energy_at_grid_point(gp,
                                         band_indices,
                                         mesh,
                                         frequencies,
                                         gammas,
                                         sigma=None,
                                         temperature=None,
                                         scattering_event_class=None,
                                         filename=None,
                                         is_mesh_symmetry=True):

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

    if scattering_event_class is not None:
        gammas_filename += "-c%d" % scattering_event_class

    if not filename == None:
        gammas_filename += ".%s" % filename
    elif not is_mesh_symmetry:
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
                    temperatures=None,
                    filename=None,
                    is_mesh_symmetry=True):
    if temperatures is None:
        _write_joint_dos_at_t(gp,
                              mesh,
                              frequencies,
                              jdos,
                              sigma=sigma,
                              temperature=None,
                              filename=filename,
                              is_mesh_symmetry=is_mesh_symmetry)
    else:
        for jdos_at_t, t in zip(jdos, temperatures):
            _write_joint_dos_at_t(gp,
                                  mesh,
                                  frequencies,
                                  jdos_at_t,
                                  sigma=sigma,
                                  temperature=t,
                                  filename=filename,
                                  is_mesh_symmetry=is_mesh_symmetry)

def _write_joint_dos_at_t(gp,
                          mesh,
                          frequencies,
                          jdos,
                          sigma=None,
                          temperature=None,
                          filename=None,
                          is_mesh_symmetry=True):
    jdos_filename = "jdos-m%d%d%d-g%d" % (mesh[0], mesh[1], mesh[2], gp)
    if sigma is not None:
        jdos_filename += ("-s%f" % sigma).rstrip('0').rstrip('\.')
    if temperature is not None:
        jdos_filename += ("-t%f" % temperature).rstrip('0').rstrip('\.')
    if not is_mesh_symmetry:
        jdos_filename += ".nosym"
    if filename is not None:
        jdos_filename += ".%s" % filename
    jdos_filename += ".dat"

    w = open(jdos_filename, 'w')
    for omega, vals in zip(frequencies, jdos):
        w.write("%15.7f" % omega)
        w.write((" %20.15e" * len(vals)) % tuple(vals))
        w.write("\n")
    w.close()

def write_linewidth_at_grid_point(gp,
                                  band_indices,
                                  temperatures,
                                  gamma,
                                  mesh,
                                  sigma=None,
                                  filename=None,
                                  is_mesh_symmetry=True):

    lw_filename = "linewidth"
    lw_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if sigma is not None:
        lw_filename += ("s%f" % sigma).rstrip('0') + "-"

    for i in band_indices:
        lw_filename += "b%d" % (i + 1)

    if not filename == None:
        lw_filename += ".%s" % filename
    elif not is_mesh_symmetry:
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
                          filename=None,
                          is_mesh_symmetry=True):

    fst_filename = "frequency_shift"
    fst_filename += "-m%d%d%d-g%d-" % (mesh[0], mesh[1], mesh[2], gp)
    if epsilon is not None:
        if epsilon > 1e-5:
            fst_filename += ("s%f" % epsilon).rstrip('0') + "-"
        else:
            fst_filename += ("s%.3e" % epsilon) + "-"
    for i in band_indices:
        fst_filename += "b%d" % (i + 1)
    if not filename == None:
        fst_filename += ".%s" % filename
    elif not is_mesh_symmetry:
        fst_filename += ".nosym"
    fst_filename += ".dat"

    w = open(fst_filename, 'w')
    for v, t in zip(delta.sum(axis=1) / delta.shape[1], temperatures):
        w.write("%15.7f %20.15e\n" % (t, v))
    w.close()

def write_collision_to_hdf5(temperature,
                            mesh,
                            gamma=None,
                            gamma_isotope=None,
                            collision_matrix=None,
                            grid_point=None,
                            sigma=None,
                            filename=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if sigma is not None:
        sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename
    with h5py.File("collision" + suffix + ".hdf5", 'w') as w:
        w.create_dataset('temperature', data=temperature)
        if gamma is not None:
            w.create_dataset('gamma', data=gamma)
        if gamma_isotope is not None:
            w.create_dataset('gamma_isotope', data=gamma_isotope)
        if collision_matrix is not None:
            w.create_dataset('collision_matrix', data=collision_matrix)

        text = "Collisions "
        if grid_point is not None:
            text += "at grid adress %d " % grid_point
        if sigma is not None:
            if grid_point is not None:
                text += "and "
            else:
                text += "at "
            text += "sigma %s " % sigma_str
        text += "were written into \n"
        text += "\"%s\"" % ("collision" + suffix + ".hdf5")
        print(text)
        print('')

def write_full_collision_matrix(collision_matrix, filename='fcm.hdf5'):
    with h5py.File(filename, 'w') as w:
        w.create_dataset('collision_matrix', data=collision_matrix)

def write_kappa_to_hdf5(temperature,
                        mesh,
                        frequency=None,
                        group_velocity=None,
                        gv_by_gv=None,
                        heat_capacity=None,
                        kappa=None,
                        mode_kappa=None,
                        gamma=None,
                        gamma_isotope=None,
                        averaged_pp_interaction=None,
                        qpoint=None,
                        weight=None,
                        mesh_divisors=None,
                        grid_point=None,
                        band_index=None,
                        sigma=None,
                        kappa_unit_conversion=None,
                        filename=None,
                        verbose=True):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  mesh_divisors=mesh_divisors,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  filename=filename)
    with h5py.File("kappa" + suffix + ".hdf5", 'w') as w:
        w.create_dataset('temperature', data=temperature)
        w.create_dataset('mesh', data=mesh)
        if frequency is not None:
            w.create_dataset('frequency', data=frequency)
        if group_velocity is not None:
            w.create_dataset('group_velocity', data=group_velocity)
        if gv_by_gv is not None:
            w.create_dataset('gv_by_gv', data=gv_by_gv)
        if heat_capacity is not None:
            w.create_dataset('heat_capacity', data=heat_capacity)
        if kappa is not None:
            w.create_dataset('kappa', data=kappa)
        if mode_kappa is not None:
            w.create_dataset('mode_kappa', data=mode_kappa)
        if gamma is not None:
            w.create_dataset('gamma', data=gamma)
        if gamma_isotope is not None:
            w.create_dataset('gamma_isotope', data=gamma_isotope)
        if averaged_pp_interaction is not None:
            w.create_dataset('ave_pp', data=averaged_pp_interaction)
        if qpoint is not None:
            w.create_dataset('qpoint', data=qpoint)
        if weight is not None:
            w.create_dataset('weight', data=weight)
        if kappa_unit_conversion is not None:
            w.create_dataset('kappa_unit_conversion',
                             data=kappa_unit_conversion)

        if verbose:
            text = ""
            if kappa is not None:
                text += "Thermal conductivity and related properties "
            else:
                text += "Thermal conductivity related properties "
            if grid_point is not None:
                text += "at gp-%d " % grid_point
                if band_index is not None:
                    text += "and band_index-%d\n" % (band_index + 1)
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s\n" % sigma
                text += "were written into "
            else:
                text += "were written into "
                if band_index is None:
                    text += "\n"
            text += "\"%s\"" % ("kappa" + suffix + ".hdf5")
            print(text)

def write_collision_eigenvalues_to_hdf5(temperatures,
                                        mesh,
                                        collision_eigenvalues,
                                        sigma=None,
                                        filename=None,
                                        verbose=True):
    suffix = _get_filename_suffix(mesh,
                                  sigma=sigma,
                                  filename=filename)
    with h5py.File("coleigs" + suffix + ".hdf5", 'w') as w:
        w.create_dataset('temperature', data=temperatures)
        w.create_dataset('collision_eigenvalues', data=collision_eigenvalues)
        w.close()

        if verbose:
            text = "Eigenvalues of collision matrix "
            if sigma is not None:
                text += "with sigma %s\n" % sigma
            text += "were written into "
            text += "\"%s\"" % ("coleigs" + suffix + ".hdf5")
            print(text)

def read_gamma_from_hdf5(mesh,
                         mesh_divisors=None,
                         grid_point=None,
                         band_index=None,
                         sigma=None,
                         filename=None,
                         verbose=True):
    if band_index is None:
        band_indices = None
    else:
        band_indices = [band_index]
    suffix = _get_filename_suffix(mesh,
                                  mesh_divisors=mesh_divisors,
                                  grid_point=grid_point,
                                  band_indices=band_indices,
                                  sigma=sigma,
                                  filename=filename)
    if not os.path.exists("kappa" + suffix + ".hdf5"):
        if verbose:
            print("%s not found." % ("kappa" + suffix + ".hdf5"))
            return False

    with h5py.File("kappa" + suffix + ".hdf5", 'r') as f:
        if len(f['gamma'].shape) > 0:
            gamma = f['gamma'][:]
        else:
            gamma = f['gamma'][()]
        if 'gamma_isotope' in f.keys():
            if len(f['gamma_isotope'].shape) > 0:
                gamma_isotope = f['gamma_isotope'][:]
            else:
                gamma_isotope = f['gamma_isotope'][()]
        else:
            gamma_isotope = None
        if 'ave_pp' in f.keys():
            if len(f['ave_pp'].shape) > 0:
                averaged_pp_interaction = f['ave_pp'][:]
            else:
                averaged_pp_interaction = f['ave_pp'][()]
        else:
            averaged_pp_interaction = None

        if verbose:
            print("Read data from %s." % ("kappa" + suffix + ".hdf5"))

        return gamma, gamma_isotope, averaged_pp_interaction

    return None

def read_collision_from_hdf5(mesh,
                             grid_point=None,
                             sigma=None,
                             filename=None,
                             verbose=True):
    suffix = "-m%d%d%d" % tuple(mesh)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if sigma is not None:
        sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename

    if not os.path.exists("collision" + suffix + ".hdf5"):
        return False

    with h5py.File("collision" + suffix + ".hdf5", 'r') as f:
        gamma = f['gamma'][:]
        collision_matrix = f['collision_matrix'][:]
        temperatures = f['temperature'][:]
        f.close()

        if verbose:
            text = "Collisions "
            if grid_point is not None:
                text += "at grid adress %d " % grid_point
            if sigma is not None:
                if grid_point is not None:
                    text += "and "
                else:
                    text += "at "
                text += "sigma %s " % sigma_str
            text += "were read from "
            if grid_point is not None:
                text += "\n"
            text += "%s" % ("collision" + suffix + ".hdf5")
            print(text)

        return collision_matrix, gamma, temperatures

    return None

def write_amplitude_to_hdf5(amplitude,
                            mesh,
                            grid_point,
                            triplet=None,
                            weight=None,
                            frequency=None,
                            eigenvector=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    suffix += ("-g%d" % grid_point)
    with h5py.File("amplitude" + suffix + ".hdf5", 'w') as w:
        w.create_dataset('amplitude', data=amplitude)
        if triplet is not None:
            w.create_dataset('triplet', data=triplet)
        if weight is not None:
            w.create_dataset('weight', data=weight)
        if frequency is not None:
            w.create_dataset('frequency', data=frequency)
        if eigenvector is not None:
            w.create_dataset('eigenvector', data=eigenvector)

def read_amplitude_from_hdf5(amplitudes_at_q,
                             mesh,
                             grid_point):
    suffix = "-m%d%d%d" % tuple(mesh)
    suffix += ("-g%d" % grid_point)
    with h5py.File("amplitude" + suffix + ".hdf5", 'r') as f:
        amplitudes_at_q[:] = f['amplitudes'][:]
        return amplitudes_at_q
    return None

def write_gamma_detail_to_hdf5(detailed_gamma,
                               temperature,
                               mesh,
                               grid_point,
                               sigma,
                               triplets,
                               weights,
                               frequency_points=None,
                               filename=None):
    suffix = _get_filename_suffix(mesh,
                                  grid_point=grid_point,
                                  sigma=sigma,
                                  filename=filename)
    full_filename = "gamma_detail" + suffix + ".hdf5"

    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('gamma_detail', data=detailed_gamma)
        w.create_dataset('temperature', data=temperature)
        w.create_dataset('mesh', data=mesh)
        w.create_dataset('triplet', data=triplets)
        w.create_dataset('weight', data=weights)
        if frequency_points is not None:
            w.create_dataset('frequency_point', data=frequency_points)
        return full_filename

def write_phonon_to_hdf5(frequency,
                         eigenvector,
                         grid_address,
                         mesh,
                          filename=None):
    suffix = _get_filename_suffix(mesh, filename=filename)
    full_filename = "phonon" + suffix + ".hdf5"

    with h5py.File(full_filename, 'w') as w:
        w.create_dataset('mesh', data=mesh)
        w.create_dataset('grid_address', data=grid_address)
        w.create_dataset('frequency', data=frequency)
        w.create_dataset('eigenvector', data=eigenvector)
        return full_filename

def read_phonon_from_hdf5(mesh,
                          filename=None,
                          verbose=True):
    suffix = _get_filename_suffix(mesh, filename=filename)
    hdf5_filename = "phonon" + suffix + ".hdf5"
    if not os.path.exists(hdf5_filename):
        if verbose:
            print("%s not found." % hdf5_filename)
        return (None, None, None, None, hdf5_filename)

    with h5py.File(hdf5_filename, 'r') as f:
        frequencies = np.array(f['frequency'][:], dtype='double', order='C')
        itemsize = frequencies.itemsize
        eigenvectors = np.array(f['eigenvector'][:],
                                dtype=("c%d" % (itemsize * 2)), order='C')
        mesh_in_file = np.array(f['mesh'][:], dtype='intc')
        grid_address = np.array(f['grid_address'][:], dtype='intc', order='C')
        return (frequencies,
                eigenvectors,
                mesh_in_file,
                grid_address,
                hdf5_filename)

    return (None, None, None, None, hdf5_filename)

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

def parse_disp_fc2_yaml(filename="disp_fc2.yaml"):
    dataset = _parse_yaml(filename)
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
    dataset = _parse_yaml(filename)
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
    dataset = _parse_yaml(filename)
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

    new_dataset['num_first_displacements'] = dataset['num_first_displacements']
    new_dataset['num_second_displacements'] = dataset['num_second_displacements']
    new_dataset['num_third_displacements'] = dataset['num_third_displacements']

    return new_dataset

def parse_FORCES_FC2(disp_dataset, filename="FORCES_FC2"):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    forces_fc2 = []
    with open(filename, 'r') as f2:
        for i in range(num_disp):
            forces = _parse_force_lines(f2, num_atom)
            if forces is None:
                return []
            else:
                forces_fc2.append(forces)
    return forces_fc2

def parse_FORCES_FC3(disp_dataset, filename="FORCES_FC3"):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    for disp1 in disp_dataset['first_atoms']:
        num_disp += len(disp1['second_atoms'])

    forces_fc3 = []
    with open(filename, 'r') as f3:
        for i in range(num_disp):
            forces = _parse_force_lines(f3, num_atom)
            if forces is None:
                return []
            else:
                forces_fc3.append(forces)
    return forces_fc3

def parse_FORCES_FC4(disp_dataset, filename="FORCES_FC4"):
    num_atom = disp_dataset['natom']
    num_disp = len(disp_dataset['first_atoms'])
    for disp1 in disp_dataset['first_atoms']:
        num_disp += len(disp1['second_atoms'])
        for disp2 in disp1['second_atoms']:
            num_disp += len(disp2['third_atoms'])

    assert num_disp == (disp_dataset['num_first_displacements'] +
                        disp_dataset['num_second_displacements'] +
                        disp_dataset['num_third_displacements'])


    f4 = open(filename, 'r')
    forces_fc4 = [_parse_force_lines(f4, num_atom) for i in range(num_disp)]
    f4.close()
    return forces_fc4

def parse_QPOINTS3(filename='QPOINTS3'):
    f = open(filename)
    num = int(f.readline().strip())
    count = 0
    qpoints3 = []
    for line in f:
        line_array = [float(x) for x in line.strip().split()]

        if len(line_array) < 9:
            print("QPOINTS3 format is invalid.")
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

def _get_filename_suffix(mesh,
                         mesh_divisors=None,
                         grid_point=None,
                         band_indices=None,
                         sigma=None,
                         filename=None):
    suffix = "-m%d%d%d" % tuple(mesh)
    if mesh_divisors is not None:
        if (np.array(mesh_divisors, dtype=int) != 1).any():
            suffix += "-d%d%d%d" % tuple(mesh_divisors)
    if grid_point is not None:
        suffix += ("-g%d" % grid_point)
    if band_indices is not None:
        suffix += "-"
        for bi in band_indices:
            suffix += "b%d" % (bi + 1)
    if sigma is not None:
        sigma_str = ("%f" % sigma).rstrip('0').rstrip('\.')
        suffix += "-s" + sigma_str
    if filename is not None:
        suffix += "." + filename

    return suffix

def _parse_yaml(file_yaml):
    import yaml
    try:
        from yaml import CLoader as Loader
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    string = open(file_yaml).read()
    data = yaml.load(string, Loader=Loader)
    return data

def _parse_force_lines(forcefile, num_atom):
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

def _parse_force_constants_lines(fcthird_file, num_atom):
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
