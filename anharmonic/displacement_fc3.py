import numpy as np
from phonopy.harmonic.displacement import get_least_displacements, \
    directions_axis, get_displacement, is_minus_displacement

def get_third_order_displacements(cell,
                                  symmetry,
                                  is_plusminus='auto',
                                  is_diagonal=False):
    # Atoms 1, 2, and 3 are defined as follows:
    #
    # Atom 1: The first displaced atom. Third order force constant
    #         between Atoms 1, 2, and 3 is calculated.
    # Atom 2: The second displaced atom. Second order force constant
    #         between Atoms 1 and 2 is calculated.
    # Atom 3: Force is mesuared on this atom.

    positions = cell.get_scaled_positions()

    # Least displacements for third order force constants
    #
    # Data structure
    # [{'number': atom1,
    #   'direction': [1,0,1],
    #   'second_atoms': [ {'number': atom2,
    #                      'directions': [[1,0,1],[-1,0,-1],...] },
    #                     {'number': ... } ] },
    #  {'number': atom1, ... } ]

    # Least displacements of first atoms (Atom 1) are searched by
    # using respective site symmetries of the original crystal.
    disps_first = get_least_displacements(symmetry,
                                          is_plusminus=is_plusminus,
                                          is_diagonal=False)

    symprec = symmetry.get_symmetry_tolerance()

    dds = []
    for disp in disps_first:
        atom1 = disp[0]
        disp1 = disp[1:4]
        site_sym = symmetry.get_site_symmetry(atom1)

        dds_atom1 = {'number': atom1,
                     'direction': disp1,
                     'second_atoms': []}

        # Reduced site symmetry at the first atom with respect to 
        # the displacement of the first atoms.
        reduced_site_sym = get_reduced_site_symmetry(site_sym, disp1, symprec)
        # Searching orbits (second atoms) with respect to
        # the first atom and its reduced site symmetry.
        second_atoms = _get_least_orbits(atom1,
                                         cell,
                                         reduced_site_sym,
                                         symmetry.get_symmetry_tolerance())
        
        for atom2 in second_atoms:
            # Bond symmetry between first and second atoms.
            reduced_bond_sym = get_bond_symmetry(
                reduced_site_sym,
                positions, 
                atom1,
                atom2,
                symmetry.get_symmetry_tolerance())

            # Since displacement of first atom breaks translation
            # symmetry, the crystal symmetry is reduced to point
            # symmetry and it is equivalent to the site symmetry
            # on the first atom. Therefore site symmetry on the 
            # second atom with the displacement is equivalent to
            # this bond symmetry.
            if is_diagonal:
                disp_second = get_displacement(reduced_bond_sym)
            else:
                disp_second = get_displacement(reduced_bond_sym,
                                               directions_axis)
            dds_atom2 = {'number': atom2, 'directions': []}
            for disp2 in disp_second:
                dds_atom2['directions'].append(disp2)
                if is_minus_displacement(disp2, reduced_bond_sym):
                    dds_atom2['directions'].append(-disp2)
            dds_atom1['second_atoms'].append(dds_atom2)
        dds.append(dds_atom1)

    return dds

def get_reduced_site_symmetry(site_sym, direction, symprec=1e-5):
    reduced_site_sym = []
    for rot in site_sym:
        if (abs(direction - np.dot(direction, rot.T)) < symprec).all():
            reduced_site_sym.append(rot)
    return np.array(reduced_site_sym)

def get_bond_symmetry(site_symmetry,
                      positions,
                      atom_center,
                      atom_disp,
                      symprec=1e-5):
    """
    Bond symmetry is the symmetry operations that keep the symmetry
    of the cell containing two fixed atoms.
    """
    bond_sym = []
    pos = positions
    for rot in site_symmetry:
        rot_pos = (np.dot(pos[atom_disp] - pos[atom_center], rot.T) +
                   pos[atom_center])
        diff = pos[atom_disp] - rot_pos
        if (abs(diff - diff.round()) < symprec).all():
            bond_sym.append(rot)

    return np.array(bond_sym)

def _get_orbits(atom_index, cell, site_symmetry, symprec=1e-5):
    positions = cell.get_scaled_positions()
    center = positions[atom_index]

    # orbits[num_atoms, num_site_sym]
    orbits = []
    for pos in positions:
        mapping = []

        for rot in site_symmetry:
            rot_pos = np.dot(pos - center, rot.T) + center

            for i, pos in enumerate(positions):
                diff = pos - rot_pos
                diff -= diff.round()
                if (abs(diff) < symprec).all():
                    mapping.append(i)
                    break

        if len(mapping) < len(site_symmetry):
            print "Site symmetry is broken."
            raise ValueError
        else:
            orbits.append(mapping)
        
    return np.array(orbits)

def _get_least_orbits(atom_index, cell, site_symmetry, symprec=1e-5):
    """Find least orbits for a centering atom"""
    orbits = _get_orbits(atom_index, cell, site_symmetry, symprec)
    mapping = range(cell.get_number_of_atoms())

    for i, orb in enumerate(orbits):
        for num in np.unique(orb):
            if mapping[num] > mapping[i]:
                mapping[num] = mapping[i]

    return np.unique(mapping)

