# Copyright (C) 2014 Atsushi Togo
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

from phonopy.file_IO import (iter_collect_forces,
                             write_force_constants_to_hdf5,
                             write_FORCE_CONSTANTS)
from phonopy.interface.vasp import (get_scaled_positions_lines, check_forces,
                                    get_drift_forces)
from phonopy.units import Bohr
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.structure.atoms import symbol_map
from phonopy.structure.cells import get_supercell, get_primitive
from phonopy.harmonic.dynmat_to_fc import (
    distribute_force_constants_by_translations)


def parse_set_of_forces(num_atoms,
                        forces_filenames,
                        verbose=True):
    hook = 'Forces acting on atoms'
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        pwscf_forces = iter_collect_forces(filename,
                                           num_atoms,
                                           hook,
                                           [6, 7, 8],
                                           word='force')
        if check_forces(pwscf_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(pwscf_forces,
                                           filename=filename,
                                           verbose=verbose)
            force_sets.append(np.array(pwscf_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_pwscf(filename):
    with open(filename) as f:
        pwscf_in = PwscfIn(f.readlines())
    tags = pwscf_in.get_tags()
    lattice = tags['cell_parameters']
    positions = [pos[1] for pos in tags['atomic_positions']]
    species = [pos[0] for pos in tags['atomic_positions']]
    mass_map = {}
    pp_map = {}
    for vals in tags['atomic_species']:
        mass_map[vals[0]] = vals[1]
        pp_map[vals[0]] = vals[2]
    masses = [mass_map[x] for x in species]
    pp_all_filenames = [pp_map[x] for x in species]

    unique_species = []
    for x in species:
        if x not in unique_species:
            unique_species.append(x)

    numbers = []
    is_unusual = False
    for x in species:
        if x in symbol_map:
            numbers.append(symbol_map[x])
        else:
            numbers.append(-unique_species.index(x))
            is_unusual = True

    if is_unusual:
        positive_numbers = []
        for n in numbers:
            if n > 0:
                if n not in positive_numbers:
                    positive_numbers.append(n)

        available_numbers = range(1, 119)
        for pn in positive_numbers:
            available_numbers.remove(pn)

        for i, n in enumerate(numbers):
            if n < 1:
                numbers[i] = available_numbers[-n]

        cell = Atoms(numbers=numbers,
                     masses=masses,
                     cell=lattice,
                     scaled_positions=positions)
    else:
        cell = Atoms(numbers=numbers,
                     cell=lattice,
                     scaled_positions=positions)

    unique_symbols = []
    pp_filenames = {}
    for i, symbol in enumerate(cell.get_chemical_symbols()):
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
            pp_filenames[symbol] = pp_all_filenames[i]

    return cell, pp_filenames


def write_pwscf(filename, cell, pp_filenames):
    f = open(filename, 'w')
    f.write(get_pwscf_structure(cell, pp_filenames=pp_filenames))


def write_supercells_with_displacements(supercell,
                                        cells_with_displacements,
                                        pp_filenames,
                                        pre_filename="supercell",
                                        width=3):
    write_pwscf("supercell.in", supercell, pp_filenames)
    for i, cell in enumerate(cells_with_displacements):
        if cell is not None:
            filename = "{pre_filename}-{0:0{width}}.in".format(
                i + 1,
                pre_filename=pre_filename,
                width=width)
            write_pwscf(filename,
                        cell,
                        pp_filenames)


def get_pwscf_structure(cell, pp_filenames=None):
    lattice = cell.get_cell()
    positions = cell.get_scaled_positions()
    masses = cell.get_masses()
    chemical_symbols = cell.get_chemical_symbols()
    unique_symbols = []
    atomic_species = []
    for symbol, m in zip(chemical_symbols, masses):
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
            atomic_species.append((symbol, m))

    lines = ""
    lines += ("!    ibrav = 0, nat = %d, ntyp = %d\n" %
              (len(positions), len(unique_symbols)))
    lines += "CELL_PARAMETERS bohr\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "ATOMIC_SPECIES\n"
    for symbol, mass in atomic_species:
        if pp_filenames is None:
            lines += " %2s %10.5f   %s_PP_filename\n" % (symbol, mass, symbol)
        else:
            lines += " %2s %10.5f   %s\n" % (symbol, mass,
                                             pp_filenames[symbol])
    lines += "ATOMIC_POSITIONS crystal\n"
    for i, (symbol, pos_line) in enumerate(zip(
            chemical_symbols,
            get_scaled_positions_lines(positions).split('\n'))):
        lines += (" %2s " % symbol) + pos_line
        if i < len(chemical_symbols) - 1:
            lines += "\n"

    return lines


class PwscfIn(object):
    def __init__(self, lines):
        self._set_methods = {'ibrav':            self._set_ibrav,
                             'celldm(1)':        self._set_celldm1,
                             'nat':              self._set_nat,
                             'ntyp':             self._set_ntyp,
                             'atomic_species':   self._set_atom_types,
                             'atomic_positions': self._set_positions,
                             'cell_parameters':  self._set_lattice}
        self._tags = {'ibrav':            None,
                      'celldm(1)':        None,
                      'nat':              None,
                      'ntyp':             None,
                      'atomic_species':   None,
                      'atomic_positions': None,
                      'cell_parameters':  None}

        self._values = None
        self._collect(lines)

    def get_tags(self):
        return self._tags

    def _collect(self, lines):
        elements = {}
        tag = None
        for line_tmp in lines:
            line = line_tmp.split('!')[0]
            if ('atomic_positions' in line.lower() or
                'cell_parameters' in line.lower()):
                if len(line.split()) == 1:
                    words = [line.lower().strip(), 'alat']
                else:
                    words = line.lower().split()[:2]
            elif 'atomic_species' in line.lower():
                words = line.lower().split()
            else:
                line_replaced = line.replace('=', ' ').replace(',', ' ')
                words = line_replaced.split()
            for val in words:
                if val.lower() in self._set_methods:
                    tag = val.lower()
                    elements[tag] = []
                elif tag is not None:
                    elements[tag].append(val)

        for tag in ['ibrav', 'nat', 'ntyp']:
            if tag not in elements:
                print("%s is not found in the input file." % tag)
                sys.exit(1)

        for tag in elements:
            self._values = elements[tag]
            if tag in ['ibrav', 'nat', 'ntyp', 'celldm(1)']:
                self._set_methods[tag]()

        for tag in elements:
            self._values = elements[tag]
            if tag not in ['ibrav', 'nat', 'ntyp']:
                self._set_methods[tag]()

    def _set_ibrav(self):
        ibrav = int(self._values[0])
        if ibrav != 0:
            print("Only ibrav = 0 is supported.")
            sys.exit(1)

        self._tags['ibrav'] = ibrav

    def _set_celldm1(self):
        self._tags['celldm(1)'] = float(self._values[0])

    def _set_nat(self):
        self._tags['nat'] = int(self._values[0])

    def _set_ntyp(self):
        self._tags['ntyp'] = int(self._values[0])

    def _set_lattice(self):
        """Calculate and set lattice parameters.

        Invoked by CELL_PARAMETERS tag.

        """

        unit = self._values[0]
        if unit == 'alat':
            if not self._tags['celldm(1)']:
                print("celldm(1) has to be specified when using alat.")
                sys.exit(1)
            else:
                factor = self._tags['celldm(1)']  # in Bohr
        elif unit == 'angstrom':
            factor = 1.0 / Bohr
        else:
            factor = 1.0

        if len(self._values[1:]) < 9:
            print("CELL_PARAMETERS is wrongly set.")
            sys.exit(1)

        lattice = np.reshape([float(x) for x in self._values[1:10]], (3, 3))
        self._tags['cell_parameters'] = lattice * factor

    def _set_positions(self):
        unit = self._values[0]
        if unit != 'crystal':
            print("Only ATOMIC_POSITIONS format with "
                  "crystal coordinates is supported.")
            sys.exit(1)

        natom = self._tags['nat']
        pos_vals = self._values[1:]
        if len(pos_vals) < natom * 4:
            print("ATOMIC_POSITIONS is wrongly set.")
            sys.exit(1)

        positions = []
        for i in range(natom):
            positions.append(
                [pos_vals[i * 4],
                 [float(x) for x in pos_vals[i * 4 + 1:i * 4 + 4]]])

        self._tags['atomic_positions'] = positions

    def _set_atom_types(self):
        num_types = self._tags['ntyp']
        if len(self._values) < num_types * 3:
            print("ATOMIC_SPECIES is wrongly set.")
            sys.exit(1)

        species = []

        for i in range(num_types):
            species.append(
                [self._values[i * 3],
                 float(self._values[i * 3 + 1]),
                 self._values[i * 3 + 2]])

        self._tags['atomic_species'] = species


class PH_Q2R(object):
    """Parse QE/q2r output and create supercell force constants array
    that is readable by phonopy. A simple usage is as follows:

    ---------
    #!/usr/bin/env python

    cell, _ = read_pwscf(primcell_filename)
    q2r = PH_Q2R(q2r_filename)
    q2r.run(cell)
    q2r.write_force_constants()
    ---------

    To save memory/storage space of force constants, the shape of
    force constants array is (n_uatom, n_satom, 3, 3), where u_atom is
    the number of atoms in unit cell and n_satom is the number of
    atoms in super cell, i.e., u_atom * prod(dim). When using this
    force constants data from phonopy with primitive cell that is
    differnt from unit cell, force constants have to be regenerated
    for the primitive cell, which is not done in this class.

    Treatment of non-analytical term correction (NAC) is different
    between phonopy and QE. For insulator, QE automatically calculate
    dielectric constant and Born effective charges at PH calculation
    when q-point mesh sampling mode ('ldisp = .true.'). These data are
    written in the Gamma point dynamical matrix file (probably
    numbered as 1 among files). When running q2r.x, these files are
    read including the dielectric constant and Born effective charges,
    and the real space force constants where QE-NAC treatment is done
    are written to the q2r output file. This is not that phonopy
    expects. Therefore the dielectric constant and Born effective
    charges data have to be removed manually from the Gamma point
    dynamical matrix file before running q2r.x. Alternatively Gamma
    point only PH calculation with 'epsil = .false.' can generate the
    dynamical matrix file without the dielectric constant and Born
    effective charges data. So it is possible to replace the Gamma
    point file by this Gamma point only file to run q2r.x for phonopy.

    Attributes
    ----------
    fc : ndarray
        Force constants in either compact or full matrix.
        dtype='double'
        shape=(natom_prim, natom_super, 3, 3) for compact fc or
              (natom_super, natom_super, 3, 3) for full fc
    dimenstion : ndarray
        Supercell dimensions (not matrix)
        dtype='intc'
        shape=(3,)
    epsilon : ndarray
        Dielectric constant tensor
        dtype='double'
        shape=(3, 3)
    born : ndarray
        Born effective charges
        dtype='double'
        shape=(natom_prim, 3, 3)
    primitive : Primitive
        Primitive cell
    supercell : Supercell
        Supercell

    """

    def __init__(self, filename, symprec=1e-5):
        self.fc = None
        self.dimension = None
        self.epsilon = None
        self.borns = None
        self.primitive = None
        self.supercell = None
        self._symprec = symprec
        self._filename = filename

    def run(self, cell, is_full_fc=False, parse_fc=True):
        """Make supercell force constants readable for phonopy

        Note
        ----
        Born effective charges and dielectric constant tensor are read
        from QE output file if they exist. But this means
        dipole-dipole contributions are removed from force constants
        and this force constants matrix is not usable in phonopy.

        Arguments
        ---------
        cell : PhonopyAtoms
            Primitive cell used for QE/PH calculation.
        is_full_fc : Bool, optional, default=False
            Whether to create full or compact force constants.
        parse_fc : Bool, optional, default=True
            Force constants file of QE is not parsed when this is False.
            False may be used when expected to parse only epsilon and born.

        """

        with open(self._filename) as f:
            fc_dct = self._parse_q2r(f)
            self.dimension = fc_dct['dimension']
            self.epsilon = fc_dct['dielectric']
            self.borns = fc_dct['born']
            if parse_fc:
                (self.fc,
                 self.primitive,
                 self.supercell) = self._arrange_supercell_fc(
                     cell, fc_dct['fc'], is_full_fc=is_full_fc)

    def write_force_constants(self, fc_format='hdf5'):
        if self.fc is not None:
            if fc_format == 'hdf5':
                p2s_map = self.primitive.get_primitive_to_supercell_map()
                write_force_constants_to_hdf5(self.fc, p2s_map=p2s_map)
            else:
                write_FORCE_CONSTANTS(self.fc)

    def _parse_q2r(self, f):
        """Parse q2r output file

        The format of q2r output is described at the mailing list below:
        http://www.democritos.it/pipermail/pw_forum/2005-April/002408.html
        http://www.democritos.it/pipermail/pw_forum/2008-September/010099.html
        http://www.democritos.it/pipermail/pw_forum/2009-August/013613.html
        https://www.mail-archive.com/pw_forum@pwscf.org/msg24388.html

        """

        natom, dim, epsilon, borns = self._parse_parameters(f)
        fc_dct = {'fc': self._parse_fc(f, natom, dim),
                  'dimension': dim,
                  'dielectric': epsilon,
                  'born': borns}
        return fc_dct

    def _parse_parameters(self, f):
        line = f.readline()
        ntype, natom, ibrav = (int(x) for x in line.split()[:3])
        if ibrav == 0:
            for i in range(3):
                line = f.readline()
        for i in range(ntype + natom):
            line = f.readline()
        line = f.readline()
        if line.strip() == 'T':
            epsilon, borns = self._parse_born(f, natom)
        else:
            epsilon = None
            borns = None
        line = f.readline()
        dim = np.array([int(x) for x in line.split()], dtype='intc')

        return natom, dim, epsilon, borns

    def _parse_born(self, f, natom):
        epsilon = np.zeros((3, 3), dtype='double', order='C')
        borns = np.zeros((natom, 3, 3), dtype='double', order='C')
        for i in range(3):
            line = f.readline()
            epsilon[i, :] = [float(x) for x in line.split()]
        for i in range(natom):
            line = f.readline()
            for j in range(3):
                line = f.readline()
                borns[i, j, :] = [float(x) for x in line.split()]
        return epsilon, borns

    def _parse_fc(self, f, natom, dim):
        """Parse force constants part

        Physical unit of force cosntants in the file is Ry/au^2.

        """

        ndim = np.prod(dim)
        fc = np.zeros((natom, natom * ndim, 3, 3), dtype='double', order='C')
        for k, l, i, j in np.ndindex((3, 3, natom, natom)):
            line = f.readline()
            for i_dim in range(ndim):
                line = f.readline()
                # fc[i, j * ndim + i_dim, k, l] = float(line.split()[3])
                fc[j, i * ndim + i_dim, l, k] = float(line.split()[3])
        return fc

    def _arrange_supercell_fc(self, cell, q2r_fc, is_full_fc=False):
        dim = self.dimension
        q2r_spos = self._get_q2r_positions(cell)
        scell = get_supercell(cell, np.diag(dim))
        pcell = get_primitive(scell, np.diag(1.0 / dim))

        diff = cell.get_scaled_positions() - pcell.get_scaled_positions()
        diff -= np.rint(diff)
        assert (np.abs(diff) < 1e-8).all()
        assert scell.get_number_of_atoms() == len(q2r_spos)

        site_map = self._get_site_mapping(scell.get_scaled_positions(),
                                          q2r_spos,
                                          scell.get_cell())
        natom = pcell.get_number_of_atoms()
        ndim = np.prod(dim)
        natom_s = natom * ndim

        if is_full_fc:
            fc = np.zeros((natom_s, natom_s, 3, 3), dtype='double', order='C')
            p2s = pcell.get_primitive_to_supercell_map()
            fc[p2s, :] = q2r_fc[:, site_map]
            distribute_force_constants_by_translations(fc,
                                                       pcell,
                                                       scell)
        else:
            fc = np.zeros((natom, natom_s, 3, 3), dtype='double', order='C')
            fc[:, :] = q2r_fc[:, site_map]

        return fc, pcell, scell

    def _get_q2r_positions(self, cell):
        dim = self.dimension
        natom = cell.get_number_of_atoms()
        ndim = np.prod(dim)
        spos = np.zeros((natom * np.prod(dim), 3), dtype='double', order='C')
        trans = [x[::-1] for x in np.ndindex(tuple(dim[::-1]))]
        for i, p in enumerate(cell.get_scaled_positions()):
            spos[i * ndim:(i + 1) * ndim] = (trans + p) / dim
        return spos

    def _get_site_mapping(self, spos, q2r_spos, lattice):
        site_map = []
        for i, p in enumerate(spos):
            diff = q2r_spos - p
            diff -= np.rint(diff)
            distances = np.sqrt(np.sum(np.dot(diff, lattice) ** 2, axis=1))
            indices = np.where(distances < self._symprec)[0]
            assert len(indices) == 1, "%s" % indices
            site_map.append(indices[0])

        assert len(np.unique(site_map)) == len(spos)

        return np.array(site_map)
