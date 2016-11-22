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

class SpglibError(object):
    message = "no error"

spglib_error = SpglibError()

def get_version():
    _set_no_error()
    return tuple(spg.version())

def get_symmetry(cell, symprec=1e-5, angle_tolerance=-1.0):
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
    _set_no_error()

    lattice, positions, numbers, magmoms = _expand_cell(cell)
    if lattice is None:
        return None

    multi = 48 * len(positions)
    rotation = np.zeros((multi, 3, 3), dtype='intc')
    translation = np.zeros((multi, 3), dtype='double')

    # Get symmetry operations
    if magmoms is None:
        dataset = get_symmetry_dataset(cell,
                                       symprec=symprec,
                                       angle_tolerance=angle_tolerance)
        if dataset is None:
            return None
        else:
            return {'rotations': dataset['rotations'],
                    'translations': dataset['translations'],
                    'equivalent_atoms': dataset['equivalent_atoms']}
    else:
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
        _set_error_message()
        if num_sym == 0:
            return None
        else:
            return {'rotations': np.array(rotation[:num_sym],
                                          dtype='intc', order='C'),
                    'translations': np.array(translation[:num_sym],
                                             dtype='double', order='C'),
                    'equivalent_atoms': equivalent_atoms}

def get_symmetry_dataset(cell, symprec=1e-5, angle_tolerance=-1.0):
    """Search symmetry dataset from an input cell.

    Args:
        cell, symprec, angle_tolerance:
            See the docstring of get_symmetry.

    Return:
        A dictionary is returned.

        number:
            int: International space group number
        international:
            str: International symbol
        hall:
            str: Hall symbol
        choice:
            str: Centring, origin, basis vector setting
        transformation_matrix:
            3x3 float matrix:
                Transformation matrix from input lattice to standardized lattice
                L^original = L^standardized * Tmat
        origin shift:
            float vecotr: Origin shift from standardized to input origin
        rotations, translations:
            3x3 int matrix, float vector:
                Rotation matrices and translation vectors. Space group
                operations are obtained by
                [(r,t) for r, t in zip(rotations, translations)]
        wyckoffs:
            List of characters: Wyckoff letters
        std_lattice, std_positions, std_types:
            3x3 float matrix, Nx3 float vectors, list of int:
                Standardized unit cell
        pointgroup:
            str: Pointgroup symbol

        If it fails, None is returned.
    """
    _set_no_error()

    lattice, positions, numbers, _ = _expand_cell(cell)
    if lattice is None:
        return None

    keys = ('number',
            'hall_number',
            'international',
            'hall',
            'choice',
            'transformation_matrix',
            'origin_shift',
            'rotations',
            'translations',
            'wyckoffs',
            'equivalent_atoms',
            'std_lattice',
            'std_types',
            'std_positions',
            # 'pointgroup_number',
            'pointgroup')
    spg_ds = spg.dataset(lattice, positions, numbers, symprec, angle_tolerance)
    if spg_ds is None:
        _set_error_message()
        return None

    dataset = {}
    for key, data in zip(keys, spg_ds):
        dataset[key] = data

    dataset['international'] = dataset['international'].strip()
    dataset['hall'] = dataset['hall'].strip()
    dataset['choice'] = dataset['choice'].strip()
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

    _set_error_message()
    return dataset

def get_spacegroup(cell, symprec=1e-5, angle_tolerance=-1.0, symbol_type=0):
    """Return space group in international table symbol and number as a string.

    If it fails, None is returned.
    """
    _set_no_error()

    dataset = get_symmetry_dataset(cell,
                                   symprec=symprec,
                                   angle_tolerance=angle_tolerance)
    if dataset is None:
        return None

    spg_type = get_spacegroup_type(dataset['hall_number'])
    if symbol_type == 1:
        return "%s (%d)" % (spg_type['schoenflies'], dataset['number'])
    else:
        return "%s (%d)" % (spg_type['international_short'], dataset['number'])

def get_spacegroup_type(hall_number):
    """Translate Hall number to space group type information.

    If it fails, None is returned.
    """
    _set_no_error()

    keys = ('number',
            'international_short',
            'international_full',
            'international',
            'schoenflies',
            'hall_symbol',
            'choice',
            'pointgroup_schoenflies',
            'pointgroup_international',
            'arithmetic_crystal_class_number',
            'arithmetic_crystal_class_symbol')
    spg_type_list = spg.spacegroup_type(hall_number)
    _set_error_message()

    if spg_type_list is not None:
        spg_type = dict(zip(keys, spg_type_list))
        for key in spg_type:
            if key != 'number' and key != 'arithmetic_crystal_class_number':
                spg_type[key] = spg_type[key].strip()
        return spg_type
    else:
        return None

def get_pointgroup(rotations):
    """Return point group in international table symbol and number.

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
    _set_no_error()

    # (symbol, pointgroup_number, transformation_matrix)
    pointgroup = spg.pointgroup(np.array(rotations, dtype='intc', order='C'))
    _set_error_message()
    return pointgroup

def standardize_cell(cell,
                     to_primitive=False,
                     no_idealize=False,
                     symprec=1e-5,
                     angle_tolerance=-1.0):
    """Return standardized cell.

    Args:
        cell, symprec, angle_tolerance:
            See the docstring of get_symmetry.
        to_primitive:
            bool: If True, the standardized primitive cell is created.
        no_idealize:
            bool: If True,  it is disabled to idealize lengths and angles of
                  basis vectors and positions of atoms according to crystal
                  symmetry.
    Return:
        The standardized unit cell or primitive cell is returned by a tuple of
        (lattice, positions, numbers).
        If it fails, None is returned.
    """
    _set_no_error()

    lattice, _positions, _numbers, _ = _expand_cell(cell)
    if lattice is None:
        return None

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
                                        to_primitive * 1,
                                        no_idealize * 1,
                                        symprec,
                                        angle_tolerance)
    _set_error_message()

    if num_atom_std > 0:
        return (np.array(lattice.T, dtype='double', order='C'),
                np.array(positions[:num_atom_std], dtype='double', order='C'),
                np.array(numbers[:num_atom_std], dtype='intc'))
    else:
        return None

def refine_cell(cell, symprec=1e-5, angle_tolerance=-1.0):
    """Return refined cell.

    The standardized unit cell is returned by a tuple of
    (lattice, positions, numbers).
    If it fails, None is returned.
    """
    _set_no_error()

    lattice, _positions, _numbers, _ = _expand_cell(cell)
    if lattice is None:
        return None

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
    _set_error_message()

    if num_atom_std > 0:
        return (np.array(lattice.T, dtype='double', order='C'),
                np.array(positions[:num_atom_std], dtype='double', order='C'),
                np.array(numbers[:num_atom_std], dtype='intc'))
    else:
        return None

def find_primitive(cell, symprec=1e-5, angle_tolerance=-1.0):
    """Primitive cell is searched in the input cell.

    The primitive cell is returned by a tuple of (lattice, positions, numbers).
    If it fails, None is returned.
    """
    _set_no_error()

    lattice, positions, numbers, _ = _expand_cell(cell)
    if lattice is None:
        return None

    num_atom_prim = spg.primitive(lattice,
                                  positions,
                                  numbers,
                                  symprec,
                                  angle_tolerance)
    _set_error_message()

    if num_atom_prim > 0:
        return (np.array(lattice.T, dtype='double', order='C'),
                np.array(positions[:num_atom_prim], dtype='double', order='C'),
                np.array(numbers[:num_atom_prim], dtype='intc'))
    else:
        return None

def get_symmetry_from_database(hall_number):
    """Return symmetry operations corresponding to a Hall symbol.

    The Hall symbol is given by the serial number in between 1 and 530.
    The symmetry operations are given by a dictionary whose keys are
    'rotations' and 'translations'.
    If it fails, None is returned.
    """
    _set_no_error()

    rotations = np.zeros((192, 3, 3), dtype='intc')
    translations = np.zeros((192, 3), dtype='double')
    num_sym = spg.symmetry_from_database(rotations, translations, hall_number)
    _set_error_message()

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
    """Return grid point index by tranlating grid address"""
    _set_no_error()

    return spg.grid_point_from_address(np.array(grid_address, dtype='intc'),
                                       np.array(mesh, dtype='intc'))


def get_ir_reciprocal_mesh(mesh,
                           cell,
                           is_shift=None,
                           is_time_reversal=True,
                           symprec=1e-5):
    """Return k-points mesh and k-point map to the irreducible k-points.

    The symmetry is serched from the input cell.

    Args:
        mesh:
            int array (3,): Uniform sampling mesh numbers
        cell, symprec:
            See the docstring of get_symmetry.
        is_shift:
            int array (3,): [0, 0, 0] gives Gamma center mesh and value 1 gives
                            half mesh shift.
        is_time_reversal:
            bool: Time reversal symmetry is included or not.

    Return:
        mapping_table:
            int array (N,): Grid point mapping table to ir-gird-points
        grid_address:
            int array (N, 3): Address of all grid points
    """
    _set_no_error()

    lattice, positions, numbers, _ = _expand_cell(cell)
    if lattice is None:
        return None

    mapping = np.zeros(np.prod(mesh), dtype='intc')
    grid_address = np.zeros((np.prod(mesh), 3), dtype='intc')
    if is_shift is None:
        is_shift = [0, 0, 0]
    if spg.ir_reciprocal_mesh(
            grid_address,
            mapping,
            np.array(mesh, dtype='intc'),
            np.array(is_shift, dtype='intc'),
            is_time_reversal * 1,
            lattice,
            positions,
            numbers,
            symprec) > 0:
        return mapping, grid_address
    else:
        return None

def get_stabilized_reciprocal_mesh(mesh,
                                   rotations,
                                   is_shift=None,
                                   is_time_reversal=True,
                                   qpoints=None):
    """Return k-point map to the irreducible k-points and k-point grid points .

    The symmetry is searched from the input rotation matrices in real space.

    Args:
        mesh:
            int array (3,): Uniform sampling mesh numbers
        is_shift:
            int array (3,): [0, 0, 0] gives Gamma center mesh and value 1 gives
                            half mesh shift.
        is_time_reversal:
            bool: Time reversal symmetry is included or not.
        qpoints:
            float array (N ,3) or (3,):
                Stabilizer(s) in the fractional coordinates.

    Return:
        mapping_table:
            int array (N,): Grid point mapping table to ir-gird-points
        grid_address:
            int array (N, 3): Address of all grid points
    """
    _set_no_error()

    mapping_table = np.zeros(np.prod(mesh), dtype='intc')
    grid_address = np.zeros((np.prod(mesh), 3), dtype='intc')
    if is_shift is None:
        is_shift = [0, 0, 0]
    if qpoints is None:
        qpoints = np.array([[0, 0, 0]], dtype='double', order='C')
    else:
        qpoints = np.array(qpoints, dtype='double', order='C')
        if qpoints.shape == (3,):
            qpoints = np.array([qpoints], dtype='double', order='C')

    if spg.stabilized_reciprocal_mesh(
            grid_address,
            mapping_table,
            np.array(mesh, dtype='intc'),
            np.array(is_shift, dtype='intc'),
            is_time_reversal * 1,
            np.array(rotations, dtype='intc', order='C'),
            qpoints) > 0:
        return mapping_table, grid_address
    else:
        return None

def get_grid_points_by_rotations(address_orig,
                                 reciprocal_rotations,
                                 mesh,
                                 is_shift=np.zeros(3, dtype='intc')):
    """Rotation operations in reciprocal space ``reciprocal_rotations`` are applied
    to a grid point ``grid_point`` and resulting grid points are returned.
    """
    _set_no_error()

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
    """Rotation operations in reciprocal space ``reciprocal_rotations`` are applied
    to a grid point ``grid_point`` and resulting grid points are returned.
    """
    _set_no_error()

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
    """Grid addresses are relocated inside Brillouin zone.
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
    _set_no_error()

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

def delaunay_reduce(lattice, eps=1e-5):
    """Run Delaunay reduction

    Args:
        lattice: Lattice parameters in the form of
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]
        symprec:
            float: Tolerance to check if volume is close to zero or not and
                   if two basis vectors are orthogonal by the value of dot
                   product being close to zero or not.

    Returns:
        if the Delaunay reduction succeeded:
            Reduced lattice parameters are given as a numpy 'double' array:
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]
        otherwise None is returned.
    """
    _set_no_error()

    delaunay_lattice = np.array(np.transpose(lattice),
                                dtype='double', order='C')
    result = spg.delaunay_reduce(delaunay_lattice, float(eps))
    _set_error_message()

    if result == 0:
        return None
    else:
        return np.array(np.transpose(delaunay_lattice),
                        dtype='double', order='C')

def niggli_reduce(lattice, eps=1e-5):
    """Run Niggli reduction

    Args:
        lattice: Lattice parameters in the form of
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]
        eps:
            float: Tolerance to check if difference of norms of two basis
                   vectors is close to zero or not and if two basis vectors are
                   orthogonal by the value of dot product being close to zero or
                   not. The detail is shown at
                   https://atztogo.github.io/niggli/.

    Returns:
        if the Niggli reduction succeeded:
            Reduced lattice parameters are given as a numpy 'double' array:
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]
        otherwise None is returned.
    """
    _set_no_error()

    niggli_lattice = np.array(np.transpose(lattice), dtype='double', order='C')
    result = spg.niggli_reduce(niggli_lattice, float(eps))
    _set_error_message()

    if result == 0:
        return None
    else:
        return np.array(np.transpose(niggli_lattice), dtype='double', order='C')

def get_error_message():
    return spglib_error.message

def _expand_cell(cell):
    if isinstance(cell, tuple):
        lattice = np.array(np.transpose(cell[0]), dtype='double', order='C')
        positions = np.array(cell[1], dtype='double', order='C')
        numbers = np.array(cell[2], dtype='intc')
        if len(cell) > 3:
            magmoms = np.array(cell[3], dtype='double')
        else:
            magmoms = None
    else:
        lattice = np.array(cell.get_cell().T, dtype='double', order='C')
        positions = np.array(cell.get_scaled_positions(),
                             dtype='double', order='C')
        numbers = np.array(cell.get_atomic_numbers(), dtype='intc')
        _magmoms = cell.get_magnetic_moments()
        if _magmoms is not None:
            magmoms = np.array(_magmoms, dtype='double')
        else:
            magmoms = None

    if _check(lattice, positions, numbers, magmoms):
        return (lattice, positions, numbers, magmoms)
    else:
        return (None, None, None, None)

def _check(lattice, positions, numbers, magmoms):
    if lattice.shape != (3, 3):
        return False
    if positions.ndim != 2:
        return False
    if positions.shape[1] != 3:
        return False
    if numbers.ndim != 1:
        return False
    if len(numbers) != positions.shape[0]:
        return False
    if magmoms is not None:
        if magmoms.ndim != 1:
            return False
        if len(magmoms) != len(numbers):
            return False
    return True

def _set_error_message():
    spglib_error.message = spg.error_message()

def _set_no_error():
    spglib_error.message = "no error"
