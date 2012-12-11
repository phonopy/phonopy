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
    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype=float)
    atom_list_done = []
    for disps_second in displacements['second_atoms']:
        disps2 = disps_second['displacements']
        atom2 = disps_second['number']
        forces = disps_second['delta_forces']
        atom_list_done.append(atom2)
        reduced_bond_sym = get_bond_symmetry(
            reduced_site_sym,
            supercell.get_scaled_positions(),
            atom1,
            atom2,
            symprec)
        _get_restricted_force_constant(fc2,
                                       forces,
                                       supercell,
                                       atom2,
                                       disps2,
                                       reduced_bond_sym,
                                       symprec)

    # Shift positions according to set atom1 is at origin
    positions = supercell.get_scaled_positions()
    pos_center = positions[atom1].copy()
    positions -= pos_center

    for i in range(num_atom):
        if not (i in atom_list_done):
            map_atom, map_sym = fc.get_atom_mapping_by_symmetry(
                atom_list_done,
                i,
                reduced_site_sym,
                np.zeros((len(reduced_site_sym), 3), dtype=float),
                positions,
                symprec)
            rot_cartesian = fc.similarity_transformation(
                supercell.get_cell().T, reduced_site_sym[map_sym])
            fc.distribute_fc2_part(fc2,
                                   positions,
                                   i,
                                   map_atom,
                                   rot_cartesian,
                                   reduced_site_sym[map_sym],
                                   np.zeros(3, dtype=float),
                                   symprec)
            
    return fc2

def _get_restricted_force_constant(fc2,
                                   sets_of_forces,
                                   supercell,
                                   disp_atom_number,
                                   displacements, 
                                   site_symmetry,
                                   symprec=1e-5):
    """
    Force constants under a finite displaced first atom for the second
    atom is calculated.

    \Phi_{\alpha\beta}|_\gamma, where
    \gamma: first atom, \beta: second atom, \alpha: all atoms
    """
    
    symmetry_matrices = fc.get_symmetry_matrices(supercell, site_symmetry)

    rot_disps = []
    for disp_cart in displacements:
        rot_disps.append(fc.get_rotated_displacements(disp_cart,
                                                      symmetry_matrices))

    rot_disps = np.array(rot_disps).reshape(-1, 9)
    inv = np.linalg.pinv(rot_disps)

    fc.solve_force_constants_disps(fc2,
                                   supercell,
                                   disp_atom_number,
                                   site_symmetry,
                                   sets_of_forces,
                                   inv,
                                   symprec)
