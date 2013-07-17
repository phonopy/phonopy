import numpy as np
from anharmonic.displacement_fc3 import get_bond_symmetry
import phonopy.harmonic.force_constants as fc

def get_restricted_fc2(supercell,
                       displacements,
                       reduced_site_sym,
                       symprec=1e-5):
    """
    displacements = {'number': 3,
                     'displacement': [0.01, 0.00, 0.01]
                     'second_atoms': [{'number': 7,
                                       'displacements': [[]],
                                       'delta_forces': []}]}

    number: Atomic index, starting with 0.
    displacement: displacement of 1st displaced atom in Cartesian.
    displacements: displacements of 2st displaced atom in Cartesian.
    delta_forces: diff. of 2 atomic disp. forces and 1 atomic disp. forces
    Number of elements of 'displacements' and 'delta_forces' are same.
    """
    num_atom = supercell.get_number_of_atoms()
    atom1 = displacements['number']
    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
    atom_list_done = []
    for disps_second in displacements['second_atoms']:
        disps2 = disps_second['displacements']
        atom2 = disps_second['number']
        sets_of_forces = disps_second['delta_forces']
        atom_list_done.append(atom2)
        bond_sym = get_bond_symmetry(
            reduced_site_sym,
            supercell.get_scaled_positions(),
            atom1,
            atom2,
            symprec)

        fc.solve_force_constants(fc2,
                                 atom2,
                                 disps2,
                                 sets_of_forces,
                                 supercell,
                                 bond_sym,
                                 symprec)

    # Shift positions according to set atom1 is at origin
    lattice = supercell.get_cell()
    positions = supercell.get_scaled_positions()
    pos_center = positions[atom1].copy()
    positions -= pos_center
    atom_list = range(num_atom)
    fc.distribute_force_constants(fc2,
                                  atom_list,
                                  atom_list_done,
                                  lattice,
                                  positions,
                                  np.intc(reduced_site_sym).copy(),
                                  np.zeros((len(reduced_site_sym), 3),
                                           dtype='double'),
                                  symprec)
    return fc2
