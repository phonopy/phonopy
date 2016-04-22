# Copyright (C) 2015 Atsushi Togo
# All rights reserved.
#
# This file is part of spglib.
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
# * Neither the name of the spglib project nor the names of its
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

from phonopy import _spglib as spg
import numpy as np

def get_version():
    return tuple(spg.version())

def get_symmetry(cell, use_magmoms=False, symprec=1e-5, angle_tolerance=-1.0):
    """This gives crystal symmetry operations from a crystal structure.

    Args:
        cell: Crystal structrue given either in Atoms object or tuple.
            In the case given by a tuple, it has to follow the form below,
            (Lattice parameters in a 3x3 array (see the detail below),
             Fractional atomic positions in an Nx3 array,
             Integer numbers to distinguish species in a length N array,
             (optional) Collinear magnetic moments in a length N array),
            where N is the number of atoms.
            Lattice parameters are given in the form:
                [[a_x, a_y, a_z],
                 [b_x, b_y, b_z],
                 [c_x, c_y, c_z]]
        use_magmoms:
            bool: If True, collinear magnetic polarizatin is considered.
        symprec:
            float: Symmetry search tolerance in the unit of length.
        angle_tolerance:
            float: Symmetry search tolerance in the unit of angle deg.
                If the value is negative, an internally optimized routine
                is used to judge symmetry.

    Return:
        dictionary: Rotation parts and translation parts.

        'rotations': Gives the numpy 'intc' array of the rotation matrices.
        'translations': Gives the numpy 'double' array of fractional
            translations with respect to a, b, c axes.
    """

    lattice, positions, numbers, magmoms = _expand_cell(
        cell, use_magmoms=use_magmoms)
    multi = 48 * len(positions)
    rotation = np.zeros((multi, 3, 3), dtype='intc')
    translation = np.zeros((multi, 3), dtype='double')

    # Get symmetry operations
    if use_magmoms:
        equivalent_atoms = np.zeros(len(magmoms), dtype='intc')
        num_sym = spg.symmetry_with_collinear_spin(rotation,
                                                   translation,
                                                   equivalent_atoms,
                                                   lattice,
                                                   positions,
                                                   numbers,
                                                   magmoms,
                                                   symprec,
                                                   angle_tolerance)
        return ({'rotations': np.array(rotation[:num_sym],
                                       dtype='intc', order='C'),
                 'translations': np.array(translation[:num_sym],
                                          dtype='double', order='C')},
                equivalent_atoms)
    else:
        num_sym = spg.symmetry(rotation,
                               translation,
                               lattice,
                               positions,
                               numbers,
                               symprec,
                               angle_tolerance)

        return {'rotations': np.array(rotation[:num_sym],
                                      dtype='intc', order='C'),
                'translations': np.array(translation[:num_sym],
                                         dtype='double', order='C')}

def get_symmetry_dataset(cell, symprec=1e-5, angle_tolerance=-1.0):
    """
    number: International space group number
    international: International symbol
    hall: Hall symbol
    transformation_matrix:
      Transformation matrix from input lattice to standardized lattice
      L^original = L^standardized * Tmat
    origin shift: Origin shift from standardized to input origin 
    rotations, translations:
      Rotation matrices and translation vectors
      Space group operations are obtained by
        [(r,t) for r, t in zip(rotations, translations)]
    wyckoffs:
      Wyckoff letters
    std_lattice, std_types, std_positions:
      Standardized unit cell
    pointgroup_number, pointgroup_symbol: Point group number (see get_pointgroup)
    """

    lattice, positions, numbers, _ = _expand_cell(cell)
    keys = ('number',
            'hall_number',
            'international',
            'hall',
            'transformation_matrix',
            'origin_shift',
            'rotations',
            'translations',
            'wyckoffs',
            'equivalent_atoms',
            'std_lattice',
            'std_types',
            'std_positions',
            'pointgroup_number',
            'pointgroup')
    dataset = {}
    for key, data in zip(keys, spg.dataset(lattice,
                                           positions,
                                           numbers,
                                           symprec,
                                           angle_tolerance)):
        dataset[key] = data

    dataset['international'] = dataset['international'].strip()
    dataset['hall'] = dataset['hall'].strip()
    dataset['transformation_matrix'] = np.array(
        dataset['transformation_matrix'], dtype='double', order='C')
    dataset['origin_shift'] = np.array(dataset['origin_shift'], dtype='double')
    dataset['rotations'] = np.array(dataset['rotations'],
                                    dtype='intc', order='C')
    dataset['translations'] = np.array(dataset['translations'],
                                       dtype='double', order='C')
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    dataset['wyckoffs'] = [letters[x] for x in dataset['wyckoffs']]
    dataset['equivalent_atoms'] = np.array(dataset['equivalent_atoms'],
                                           dtype='intc')
    dataset['std_lattice'] = np.array(np.transpose(dataset['std_lattice']),
                                      dtype='double', order='C')
    dataset['std_types'] = np.array(dataset['std_types'], dtype='intc')
    dataset['std_positions'] = np.array(dataset['std_positions'],
                                        dtype='double', order='C')
    dataset['pointgroup'] = dataset['pointgroup'].strip()

    return dataset

def get_spacegroup(cell, symprec=1e-5, angle_tolerance=-1.0, symbol_type=0):
    """
    Return space group in international table symbol and number
    as a string.
    """

    dataset = get_symmetry_dataset(cell,
                                   symprec=symprec,
                                   angle_tolerance=angle_tolerance)
    symbols = spg.spacegroup_type(dataset['hall_number'])

    if symbol_type == 1:
        return "%s (%d)" % (symbols[0], dataset['number'])
    else:
        return "%s (%d)" % (symbols[4], dataset['number'])

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
    return spg.pointgroup(np.array(rotations, dtype='intc', order='C'))

def standardize_cell(cell,
                     to_primitive=0,
                     no_idealize=0,
                     symprec=1e-5,
                     angle_tolerance=-1.0):
    """
    Return standardized cell
    """

    lattice, _positions, _numbers, _ = _expand_cell(cell)

    # Atomic positions have to be specified by scaled positions for spglib.
    num_atom = len(_positions)
    positions = np.zeros((num_atom * 4, 3), dtype='double', order='C')
    positions[:num_atom] = _positions
    numbers = np.zeros(num_atom * 4, dtype='intc')
    numbers[:num_atom] = _numbers
    num_atom_std = spg.standardize_cell(lattice,
                                        positions,
                                        numbers,
                                        num_atom,
                                        to_primitive,
                                        no_idealize,
                                        symprec,
                                        angle_tolerance)

    return (np.array(lattice.T, dtype='double', order='C'),
            np.array(positions[:num_atom_std], dtype='double', order='C'),
            np.array(numbers[:num_atom_std], dtype='intc'))

def refine_cell(cell, symprec=1e-5, angle_tolerance=-1.0):
    """
    Return refined cell
    """

    lattice, _positions, _numbers, _ = _expand_cell(cell)

    # Atomic positions have to be specified by scaled positions for spglib.
    num_atom = len(_positions)
    positions = np.zeros((num_atom * 4, 3), dtype='double', order='C')
    positions[:num_atom] = _positions
    numbers = np.zeros(num_atom * 4, dtype='intc')
    numbers[:num_atom] = _numbers
    num_atom_std = spg.refine_cell(lattice,
                                   positions,
                                   numbers,
                                   num_atom,
                                   symprec,
                                   angle_tolerance)

    return (np.array(lattice.T, dtype='double', order='C'),
            np.array(positions[:num_atom_std], dtype='double', order='C'),
            np.array(numbers[:num_atom_std], dtype='intc'))

def find_primitive(cell, symprec=1e-5, angle_tolerance=-1.0):
    """
    A primitive cell in the input cell is searched and returned
    as an object of Atoms class.
    If no primitive cell is found, (None, None, None) is returned.
    """

    lattice, positions, numbers, _ = _expand_cell(cell)
    num_atom_prim = spg.primitive(lattice,
                                  positions,
                                  numbers,
                                  symprec,
                                  angle_tolerance)
    if num_atom_prim > 0:
        return (np.array(lattice.T, dtype='double', order='C'),
                np.array(positions[:num_atom_prim], dtype='double', order='C'),
                np.array(numbers[:num_atom_prim], dtype='intc'))
    else:
        return None, None, None

def get_symmetry_from_database(hall_number):
    rotations = np.zeros((192, 3, 3), dtype='intc')
    translations = np.zeros((192, 3), dtype='double')
    num_sym = spg.symmetry_from_database(rotations, translations, hall_number)
    if num_sym is None:
        return None
    else:
        return {'rotations':
                np.array(rotations[:num_sym], dtype='intc', order='C'),
                'translations':
                np.array(translations[:num_sym], dtype='double', order='C')}
        
############
# k-points #
############
def get_grid_point_from_address(grid_address, mesh):
    """
    Return grid point index by tranlating grid address
    """

    return spg.grid_point_from_address(np.array(grid_address, dtype='intc'),
                                       np.array(mesh, dtype='intc'))
    

def get_ir_reciprocal_mesh(mesh,
                           cell,
                           is_shift=np.zeros(3, dtype='intc'),
                           is_time_reversal=True,
                           symprec=1e-5):
    """
    Return k-points mesh and k-point map to the irreducible k-points
    The symmetry is serched from the input cell.
    is_shift=[0, 0, 0] gives Gamma center mesh.
    """

    lattice, positions, numbers, _ = _expand_cell(cell)
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    spg.ir_reciprocal_mesh(
        mesh_points,
        mapping,
        np.array(mesh, dtype='intc'),
        np.array(is_shift, dtype='intc'),
        is_time_reversal * 1,
        lattice,
        positions,
        numbers,
        symprec)
  
    return mapping, mesh_points

def get_grid_points_by_rotations(address_orig,
                                 reciprocal_rotations,
                                 mesh,
                                 is_shift=np.zeros(3, dtype='intc')):
    """
    Rotation operations in reciprocal space ``reciprocal_rotations`` are applied
    to a grid point ``grid_point`` and resulting grid points are returned.
    """
    
    rot_grid_points = np.zeros(len(reciprocal_rotations), dtype='intc')
    spg.grid_points_by_rotations(
        rot_grid_points,
        np.array(address_orig, dtype='intc'),
        np.array(reciprocal_rotations, dtype='intc', order='C'),
        np.array(mesh, dtype='intc'),
        np.array(is_shift, dtype='intc'))
    
    return rot_grid_points

def get_BZ_grid_points_by_rotations(address_orig,
                                    reciprocal_rotations,
                                    mesh,
                                    bz_map,
                                    is_shift=np.zeros(3, dtype='intc')):
    """
    Rotation operations in reciprocal space ``reciprocal_rotations`` are applied
    to a grid point ``grid_point`` and resulting grid points are returned.
    """
    
    rot_grid_points = np.zeros(len(reciprocal_rotations), dtype='intc')
    spg.BZ_grid_points_by_rotations(
        rot_grid_points,
        np.array(address_orig, dtype='intc'),
        np.array(reciprocal_rotations, dtype='intc', order='C'),
        np.array(mesh, dtype='intc'),
        np.array(is_shift, dtype='intc'),
        bz_map)
    
    return rot_grid_points
    
def relocate_BZ_grid_address(grid_address,
                             mesh,
                             reciprocal_lattice, # column vectors
                             is_shift=np.zeros(3, dtype='intc')):
    """
    Grid addresses are relocated inside Brillouin zone. 
    Number of ir-grid-points inside Brillouin zone is returned. 
    It is assumed that the following arrays have the shapes of 
      bz_grid_address[prod(mesh + 1)][3] 
      bz_map[prod(mesh * 2)] 
    where grid_address[prod(mesh)][3]. 
    Each element of grid_address is mapped to each element of 
    bz_grid_address with keeping element order. bz_grid_address has 
    larger memory space to represent BZ surface even if some points 
    on a surface are translationally equivalent to the other points 
    on the other surface. Those equivalent points are added successively 
    as grid point numbers to bz_grid_address. Those added grid points 
    are stored after the address of end point of grid_address, i.e. 
                                                                          
    |-----------------array size of bz_grid_address---------------------| 
    |--grid addresses similar to grid_address--|--newly added ones--|xxx| 
                                                                          
    where xxx means the memory space that may not be used. Number of grid 
    points stored in bz_grid_address is returned. 
    bz_map is used to recover grid point index expanded to include BZ 
    surface from grid address. The grid point indices are mapped to 
    (mesh[0] * 2) x (mesh[1] * 2) x (mesh[2] * 2) space (bz_map).
    """
    
    bz_grid_address = np.zeros(
        ((mesh[0] + 1) * (mesh[1] + 1) * (mesh[2] + 1), 3), dtype='intc')
    bz_map = np.zeros(
        (2 * mesh[0]) * (2 * mesh[1]) * (2 * mesh[2]), dtype='intc')
    num_bz_ir = spg.BZ_grid_address(
        bz_grid_address,
        bz_map,
        grid_address,
        np.array(mesh, dtype='intc'),
        np.array(reciprocal_lattice, dtype='double', order='C'),
        np.array(is_shift, dtype='intc'))

    return bz_grid_address[:num_bz_ir], bz_map
  
def get_stabilized_reciprocal_mesh(mesh,
                                   rotations,
                                   is_shift=np.zeros(3, dtype='intc'),
                                   is_time_reversal=True,
                                   qpoints=np.array([], dtype='double')):
    """
    Return k-point map to the irreducible k-points and k-point grid points .

    The symmetry is searched from the input rotation matrices in real space.
    
    is_shift=[0, 0, 0] gives Gamma center mesh and the values 1 give
    half mesh distance shifts.
    """
    
    mapping = np.zeros(np.prod(mesh), dtype='intc')
    mesh_points = np.zeros((np.prod(mesh), 3), dtype='intc')
    qpoints = np.array(qpoints, dtype='double', order='C')
    if qpoints.shape == (3,):
        qpoints = np.array([qpoints], dtype='double', order='C')
    if qpoints.shape == (0,):
        qpoints = np.array([[0, 0, 0]], dtype='double', order='C')

    spg.stabilized_reciprocal_mesh(
        mesh_points,
        mapping,
        np.array(mesh, dtype='intc'),
        np.array(is_shift, dtype='intc'),
        is_time_reversal * 1,
        np.array(rotations, dtype='intc', order='C'),
        qpoints)
    
    return mapping, mesh_points

def niggli_reduce(lattice, eps=1e-5):
    """Run Niggli reduction

    Args:
        lattice: Lattice parameters in the form of
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]
        eps: Tolerance.
    
    Returns:
        if the Niggli reduction succeeded:
            Reduced lattice parameters are given as a numpy 'double' array:
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]
        otherwise returns None.
    """
    niggli_lattice = np.array(np.transpose(lattice), dtype='double', order='C')
    result = spg.niggli_reduce(niggli_lattice, float(eps))
    if result == 0:
        return None
    else:
        return np.array(np.transpose(niggli_lattice), dtype='double', order='C')

def _expand_cell(cell, use_magmoms=False):
    if isinstance(cell, tuple):
        lattice = np.array(np.transpose(cell[0]), dtype='double', order='C')
        positions = np.array(cell[1], dtype='double', order='C')
        numbers = np.array(cell[2], dtype='intc')
        if len(cell) > 3 and use_magmoms:
            magmoms = np.array(cell[3], dtype='double')
        else:
            magmoms = None
    else:
        lattice = np.array(cell.get_cell().T, dtype='double', order='C')
        positions = np.array(cell.get_scaled_positions(),
                             dtype='double', order='C')
        numbers = np.array(cell.get_atomic_numbers(), dtype='intc')
        if use_magmoms:
            magmoms = np.array(cell.get_magnetic_moments(), dtype='double')
        else:
            magmoms = None

    return (lattice, positions, numbers, magmoms)
