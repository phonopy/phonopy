"""
Spglib interface for ASE
"""

import phonopy._spglib as spg
import numpy as np

def get_symmetry(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return symmetry operations as hash.
    Hash key 'rotations' gives the numpy integer array
    of the rotation matrices for scaled positions
    Hash key 'translations' gives the numpy double array
    of the translation vectors in scaled positions
    """

    # Atomic positions have to be specified by scaled positions for spglib.
    positions = bulk.get_scaled_positions().copy()
    lattice = bulk.get_cell().T.copy()
    numbers = np.intc(bulk.get_atomic_numbers()).copy()
  
    # Get number of symmetry operations and allocate symmetry operations
    # multi = spg.multiplicity(cell, positions, numbers, symprec)
    multi = 48 * bulk.get_number_of_atoms()
    rotation = np.zeros((multi, 3, 3), dtype='intc')
    translation = np.zeros((multi, 3))
  
    # Get symmetry operations
    magmoms = bulk.get_magnetic_moments()
    if magmoms == None:
        num_sym = spg.symmetry(rotation,
                               translation,
                               lattice,
                               positions,
                               numbers,
                               symprec,
                               angle_tolerance)
    else:
        num_sym = spg.symmetry_with_collinear_spin(rotation,
                                                   translation,
                                                   lattice,
                                                   positions,
                                                   numbers,
                                                   magmoms,
                                                   symprec,
                                                   angle_tolerance)
  
    return {'rotations': rotation[:num_sym].copy(),
            'translations': translation[:num_sym].copy()}

def get_symmetry_dataset(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    number: International space group number
    international: International symbol
    hall: Hall symbol
    transformation_matrix:
      Transformation matrix from lattice of input cell to Bravais lattice
      L^bravais = L^original * Tmat
    origin shift: Origin shift in the setting of 'Bravais lattice'
    rotations, translations:
      Rotation matrices and translation vectors
      Space group operations are obtained by
        [(r,t) for r, t in zip(rotations, translations)]
    wyckoffs:
      Wyckoff letters
    """
    positions = bulk.get_scaled_positions().copy()
    lattice = bulk.get_cell().T.copy()
    numbers = np.intc(bulk.get_atomic_numbers()).copy()
    keys = ('number',
            'international',
            'hall',
            'transformation_matrix',
            'origin_shift',
            'rotations',
            'translations',
            'wyckoffs',
            'equivalent_atoms')
    dataset = {}
    for key, data in zip(keys, spg.dataset(lattice,
                                           positions,
                                           numbers,
                                           symprec,
                                           angle_tolerance)):
        dataset[key] = data

    dataset['international'] = dataset['international'].strip()
    dataset['hall'] = dataset['hall'].strip()
    dataset['transformation_matrix'] = np.double(
        dataset['transformation_matrix'])
    dataset['origin_shift'] = np.double(dataset['origin_shift'])
    dataset['rotations'] = np.intc(dataset['rotations'])
    dataset['translations'] = np.double(dataset['translations'])
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    dataset['wyckoffs'] = [letters[x] for x in dataset['wyckoffs']]
    dataset['equivalent_atoms'] = np.intc(dataset['equivalent_atoms'])

    return dataset

def get_spacegroup(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return space group in international table symbol and number
    as a string.
    """
    # Atomic positions have to be specified by scaled positions for spglib.
    return spg.spacegroup(bulk.get_cell().T.copy(),
                          bulk.get_scaled_positions().copy(),
                          np.intc(bulk.get_atomic_numbers()).copy(),
                          symprec,
                          angle_tolerance)

def get_pointgroup(rotations):
    """
    Return point group in international table symbol and number.
    The symbols are mapped to the numbers as follows:
    1   "1    "
    2   "-1   "
    3   "2    "
    4   "m    "
    5   "2/m  "
    6   "222  "
    7   "mm2  "
    8   "mmm  "
    9   "4    "
    10  "-4   "
    11  "4/m  "
    12  "422  "
    13  "4mm  "
    14  "-42m "
    15  "4/mmm"
    16  "3    "
    17  "-3   "
    18  "32   "
    19  "3m   "
    20  "-3m  "
    21  "6    "
    22  "-6   "
    23  "6/m  "
    24  "622  "
    25  "6mm  "
    26  "-62m "
    27  "6/mmm"
    28  "23   "
    29  "m-3  "
    30  "432  "
    31  "-43m "
    32  "m-3m "
    """

    # (symbol, pointgroup_number, transformation_matrix)
    return spg.pointgroup(np.intc(rotations).copy())

def refine_cell(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return refined cell
    """
    # Atomic positions have to be specified by scaled positions for spglib.
    num_atom = bulk.get_number_of_atoms()
    lattice = bulk.get_cell().T.copy()
    pos = np.zeros((num_atom * 4, 3), dtype='double')
    pos[:num_atom] = bulk.get_scaled_positions()

    numbers = np.zeros(num_atom * 4, dtype='intc')
    numbers[:num_atom] = np.intc(bulk.get_atomic_numbers())
    num_atom_bravais = spg.refine_cell(lattice,
                                       pos,
                                       numbers,
                                       num_atom,
                                       symprec,
                                       angle_tolerance)

    return (lattice.T.copy(),
            pos[:num_atom_bravais].copy(),
            numbers[:num_atom_bravais].copy())

def find_primitive(bulk, symprec=1e-5, angle_tolerance=-1.0):
    """
    A primitive cell in the input cell is searched and returned
    as an object of Atoms class.
    If no primitive cell is found, (None, None, None) is returned.
    """

    # Atomic positions have to be specified by scaled positions for spglib.
    positions = bulk.get_scaled_positions().copy()
    lattice = bulk.get_cell().T.copy()
    numbers = np.intc(bulk.get_atomic_numbers()).copy()

    # lattice is transposed with respect to the definition of Atoms class
    num_atom_prim = spg.primitive(lattice,
                                  positions,
                                  numbers,
                                  symprec,
                                  angle_tolerance)
    if num_atom_prim > 0:
        return (lattice.T.copy(),
                positions[:num_atom_prim].copy(),
                numbers[:num_atom_prim].copy())
    else:
        return None, None, None
  
def get_ir_kpoints(kpoint,
                   bulk,
                   is_time_reversal=True,
                   symprec=1e-5):
    """
    Retrun irreducible kpoints
    """
    mapping = np.zeros(kpoint.shape[0], dtype='intc')
    spg.ir_kpoints(mapping,
                   kpoint,
                   bulk.get_cell().T.copy(),
                   bulk.get_scaled_positions().copy(),
                   np.intc(bulk.get_atomic_numbers()).copy(),
                   is_time_reversal * 1,
                   symprec)
    return mapping
  
def get_ir_reciprocal_mesh(mesh,
                           bulk,
                           is_shift=np.zeros(3, dtype='intc'),
                           is_time_reversal=True,
                           symprec=1e-5):
    """
    Return k-points mesh and k-point map to the irreducible k-points
    The symmetry is serched from the input cell.
    is_shift=[0, 0, 0] gives Gamma center mesh.
    """

    mapping = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    spg.ir_reciprocal_mesh(mesh_points,
                           mapping,
                           np.intc(mesh).copy(),
                           np.intc(is_shift).copy(),
                           is_time_reversal * 1,
                           bulk.get_cell().T.copy(),
                           bulk.get_scaled_positions().copy(),
                           np.intc(bulk.get_atomic_numbers()).copy(),
                           symprec)
  
    return mapping, mesh_points
  
def get_stabilized_reciprocal_mesh(mesh,
                                   rotations,
                                   is_shift=np.zeros(3, dtype='intc'),
                                   is_time_reversal=True,
                                   qpoints=np.double([])):
    """
    Return k-point map to the irreducible k-points and k-point grid points .

    The symmetry is searched from the input rotation matrices in real space.
    
    is_shift=[0, 0, 0] gives Gamma center mesh and the values 1 give
    half mesh distance shifts.
    """
    
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    qpoints = np.double(qpoints).copy()
    if qpoints.shape == (3,):
        qpoints = np.double([qpoints])
    spg.stabilized_reciprocal_mesh(mesh_points,
                                   mapping,
                                   np.intc(mesh).copy(),
                                   np.intc(is_shift),
                                   is_time_reversal * 1,
                                   np.intc(rotations).copy(),
                                   np.double(qpoints))
    
    return mapping, mesh_points

def get_triplets_reciprocal_mesh_at_q(fixed_grid_number,
                                      mesh,
                                      rotations,
                                      is_time_reversal=True):

    weights = np.zeros(np.prod(mesh), dtype='intc')
    third_q = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')

    spg.triplets_reciprocal_mesh_at_q(weights,
                                      mesh_points,
                                      third_q,
                                      fixed_grid_number,
                                      np.intc(mesh).copy(),
                                      is_time_reversal * 1,
                                      np.intc(rotations).copy())

    return weights, third_q, mesh_points
        
def get_grid_triplets_at_q(q_grid_point,
                           grid_points,
                           third_q,
                           weights,
                           mesh):
    num_ir_tripltes = (weights > 0).sum()
    triplets = np.zeros((num_ir_tripltes, 3), dtype='intc')
    spg.grid_triplets_at_q(triplets,
                           q_grid_point,
                           grid_points,
                           third_q,
                           weights,
                           np.intc(mesh).copy())
    return triplets
                           
