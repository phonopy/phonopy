"""PhonopyAtoms class and routines related to atoms."""

# Copyright (C) 2011 Atsushi Togo
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
from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from math import gcd
from typing import Optional, Union

import numpy as np


def Atoms(*args, **kwargs):
    """Atoms class that is same as PhonopyAtoms class.

    This exists backward compatibility.

    """
    warnings.warn(
        "phonopy.atoms.Atoms is deprecated. Please use "
        "PhonopyAtoms instead of Atoms.",
        DeprecationWarning,
        stacklevel=2,
    )
    return PhonopyAtoms(*args, **kwargs)


def split_symbol_and_index(symnum: str):
    """Split symbol and index.

    H --> H, 0
    H2 --> H, 2

    """
    m = re.match(r"([a-zA-Z]+)([0-9]*)", symnum)
    symbol, index = m.groups()
    if symnum != f"{symbol}{index}":
        raise RuntimeError(f"Invalid symbol: {symnum}.")
    if index:
        index = int(index)
        if index < 1:
            raise RuntimeError(
                f"Invalid symbol. Index has to be greater than 0: {symnum}."
            )
    else:
        index = 0
    return symbol, index


class PhonopyAtoms:
    """Class to represent crystal structure.

    Originally this aimed to be compatible ASE Atoms class, but now not.

    Attributes
    ----------
    cell : np.ndarray
        Basis vectors (a, b, c) given in row vectors.
    positions : np.ndarray
        Positions of atoms in Cartesian coordinates. shape=(natom, 3),
        dtype='double', order='C'
    scaled_positions : np.ndarray
        Positions of atoms in fractional (crystallographic) coordinates.
        shape=(natom, 3), dtype='double', order='C'
    symbols : list[str]
        List of chemical symbols of atoms. Chemical symbol + natural number is
        allowed, e.g., "Cl1".
    numbers : np.ndarray
        Atomic numbers. Numbers cannot exceed 118. shape=(natom,), dtype='intc'
    masses : np.ndarray, optional
        Atomic masses. shape=(natom,), dtype='double'
    magnetic_moments : np.ndarray, optional
        shape=(natom,) or (natom, 3), dtype='double', order='C'
    volume : float
        Cell volume.
    Z : int
        Number of formula units in this cell.

    """

    _MOD_DIVISOR = 1000

    def __init__(
        self,
        symbols: Optional[Sequence] = None,
        numbers: Optional[Union[Sequence, np.ndarray]] = None,
        masses: Optional[Union[Sequence, np.ndarray]] = None,
        magnetic_moments: Optional[Union[Sequence, np.ndarray]] = None,
        scaled_positions: Optional[Union[Sequence, np.ndarray]] = None,
        positions: Optional[Union[Sequence, np.ndarray]] = None,
        cell: Optional[Union[Sequence, np.ndarray]] = None,
        atoms: Optional["PhonopyAtoms"] = None,
        magmoms: Optional[Union[Sequence, np.ndarray]] = None,
        pbc: Optional[bool] = None,
    ):  # pbc is dummy argument, and never used.
        """Init method."""
        if magmoms is not None:
            warnings.warn(
                "PhonopyAtoms.__init__ parameter of magmoms is deprecated. "
                "Use magnetic_moments instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if pbc is not None:
            warnings.warn(
                "PhonopyAtoms.__init__ parameter of pbc is deprecated. "
                "It is considered always True.",
                DeprecationWarning,
                stacklevel=2,
            )
        if atoms:
            warnings.warn(
                "PhonopyAtoms.__init__ parameter of atoms is deprecated. "
                "Use PhonopyAtoms.copy() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._set_parameters(
                numbers=atoms.numbers,
                masses=atoms.masses,
                magnetic_moments=atoms.magnetic_moments,
                scaled_positions=atoms.scaled_positions,
                cell=atoms.cell,
            )
        else:
            self._set_parameters(
                symbols=symbols,
                numbers=numbers,
                masses=masses,
                magnetic_moments=magnetic_moments,
                scaled_positions=scaled_positions,
                positions=positions,
                cell=cell,
            )

    def _set_parameters(
        self,
        symbols: Optional[Sequence] = None,
        numbers: Optional[Union[Sequence, np.ndarray]] = None,
        masses: Optional[Union[Sequence, np.ndarray]] = None,
        magnetic_moments: Optional[Union[Sequence, np.ndarray]] = None,
        scaled_positions: Optional[Union[Sequence, np.ndarray]] = None,
        positions: Optional[Union[Sequence, np.ndarray]] = None,
        cell: Optional[Union[Sequence, np.ndarray]] = None,
    ):
        self._cell = None
        self._scaled_positions = None
        self._set_cell_and_positions(
            cell, positions=positions, scaled_positions=scaled_positions
        )

        self._symbols = symbols

        self._numbers_with_shifts = None
        if numbers is not None:
            if (np.array(numbers) > 118).any():  # 118 is the max atomic number.
                raise RuntimeError("Atomic numbers cannot be larger than 118.")
            self._numbers_with_shifts = np.array(numbers, dtype="intc")

        self._masses = None
        self._set_masses(masses)

        # (initial) magnetic moments
        self._magnetic_moments = None
        self._set_magnetic_moments(magnetic_moments)

        # numbers <--> symbols
        if self._numbers_with_shifts is not None:  # number --> symbol
            self._numbers_to_symbols()
        elif self._symbols is not None:  # symbol --> number
            self._symbols_to_numbers()

        # symbol --> mass
        if self._symbols and (self._masses is None):
            if (self._numbers_with_shifts > 118).any():  # 118 is the max atomic number.
                raise RuntimeError(
                    "Masses have to be specified when special symbols are used."
                )
            self._symbols_to_masses()

        self._check()
        self._finalize()

    def __len__(self):
        """Return number of atoms."""
        return len(self.numbers)

    @property
    def cell(self):
        """Setter and getter of basis vectors. For getter, copy is returned."""
        return self._cell.copy()

    @cell.setter
    def cell(self, cell):
        self._set_cell(cell)
        self._check()

    def set_cell(self, cell):
        """Set basis vectors."""
        warnings.warn(
            "PhonopyAtoms.set_cell() is deprecated. Use cell attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cell = cell

    def get_cell(self):
        """Return copy of basis vectors."""
        warnings.warn(
            "PhonopyAtoms.get_cell() is deprecated. Use cell attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cell

    @property
    def positions(self):
        """Setter and getter of positions in Cartesian coordinates."""
        return np.array(
            np.dot(self._scaled_positions, self._cell), dtype="double", order="C"
        )

    @positions.setter
    def positions(self, positions):
        self._set_positions(positions)
        self._check()

    def get_positions(self):
        """Return positions in Cartesian coordinates."""
        warnings.warn(
            "PhonopyAtoms.get_positions() is deprecated. "
            "Use positions attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.positions

    def set_positions(self, positions):
        """Set positions in Cartesian coordinates."""
        warnings.warn(
            "PhonopyAtoms.set_positions() is deprecated. "
            "Use positions attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.positions = positions

    @property
    def scaled_positions(self):
        """Setter and getter of scaled positions. For getter, copy is returned."""
        return self._scaled_positions.copy()

    @scaled_positions.setter
    def scaled_positions(self, scaled_positions):
        self._set_scaled_positions(scaled_positions)
        self._check()

    def get_scaled_positions(self):
        """Return scaled positions."""
        warnings.warn(
            "PhonopyAtoms.get_scaled_positions() is deprecated. "
            "Use scaled_positions attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scaled_positions

    def set_scaled_positions(self, scaled_positions):
        """Set scaled positions."""
        warnings.warn(
            "PhonopyAtoms.set_scaled_positions() is deprecated. "
            "Use scaled_positions attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.scaled_positions = scaled_positions

    @property
    def symbols(self):
        """Setter and getter of chemical symbols."""
        return self._symbols[:]

    @symbols.setter
    def symbols(self, symbols):
        warnings.warn(
            "Setter of PhonopyAtoms.symbols is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._symbols = symbols
        self._check()
        self._symbols_to_numbers()
        self._symbols_to_masses()

    def get_chemical_symbols(self):
        """Return chemical symbols."""
        warnings.warn(
            "PhonopyAtoms.get_chemical_symbols() is deprecated. "
            "Use symbols attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.symbols

    def set_chemical_symbols(self, symbols):
        """Set chemical symbols."""
        warnings.warn(
            "PhonopyAtoms.set_chemical_symbols() is deprecated. "
            "Use symbols attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.symbols = symbols

    @property
    def numbers_with_shifts(self):
        """Getter of atomic numbers + MOD_DIVISOR * index."""
        return self._numbers_with_shifts.copy()

    @property
    def numbers(self):
        """Setter and getter of atomic numbers. For getter, new array is returned."""
        return np.array(
            [n % self._MOD_DIVISOR for n in self._numbers_with_shifts], dtype="intc"
        )

    @numbers.setter
    def numbers(self, numbers):
        if (np.array(numbers) > 118).any():  # 118 is the max atomic number.
            raise RuntimeError("Atomic number is too large.")
        warnings.warn(
            "Setter of PhonopyAtoms.number is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._numbers_with_shifts = numbers
        self._check()
        self._numbers_to_symbols()
        self._symbols_to_masses()

    def get_atomic_numbers(self):
        """Return atomic numbers."""
        warnings.warn(
            "PhonopyAtoms.get_atomic_numbers() is deprecated. "
            "Use numbers attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.numbers

    def set_atomic_numbers(self, numbers):
        """Set atomic numbers."""
        warnings.warn(
            "PhonopyAtoms.set_atomic_numbers() is deprecated. "
            "Use numbers attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.numbers = numbers

    @property
    def masses(self):
        """Setter and getter of atomic masses. For getter copy is returned."""
        if self._masses is None:
            return None
        else:
            return self._masses.copy()

    @masses.setter
    def masses(self, masses):
        self._set_masses(masses)
        self._check()

    def get_masses(self):
        """Return atomic masses."""
        warnings.warn(
            "PhonopyAtoms.get_masses() is deprecated. Use masses attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.masses

    def set_masses(self, masses):
        """Set atomic masses."""
        warnings.warn(
            "PhonopyAtoms.set_masses() is deprecated. Use masses attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.masses = masses

    @property
    def magnetic_moments(self):
        """Setter and getter of magnetic moments. For getter, copy is returned.

        shape=(natom,) or (natom, 3), dtype='double', order='C'

        For setter, the formar can be specified by (natom, 1), which will be
        recognized as (natom,) and the latter can be specified by (natom * 3,),
        which will be converted to (natom, 3).

        """
        if self._magnetic_moments is None:
            return None
        else:
            return self._magnetic_moments.copy()

    @magnetic_moments.setter
    def magnetic_moments(self, magnetic_moments):
        self._set_magnetic_moments(magnetic_moments)
        self._check()
        self._finalize()

    def get_magnetic_moments(self):
        """Return magnetic moments."""
        warnings.warn(
            "PhonopyAtoms.get_magnetic_moments() is deprecated. "
            "Use magnetic_moments attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.magnetic_moments

    def set_magnetic_moments(self, magmoms):
        """Set magnetic moments."""
        warnings.warn(
            "PhonopyAtoms.set_magnetic_moments() is deprecated. "
            "Use magnetic_moments attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.magnetic_moments = magmoms

    @property
    def volume(self):
        """Return cell volume."""
        return np.linalg.det(self._cell)

    def get_volume(self):
        """Return cell volume."""
        warnings.warn(
            "PhonopyAtoms.get_volume() is deprecated. " "Use volume attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.volume

    @property
    def Z(self):
        """Return number of formula units in this cell."""
        count = {}
        for n in self._numbers_with_shifts:
            if n in count:
                count[n] += 1
            else:
                count[n] = 1
        values = list(count.values())
        x = values[0]
        for v in values[1:]:
            x = gcd(x, v)
        return x

    def get_number_of_atoms(self):
        """Return number of atoms."""
        warnings.warn(
            "PhonopyAtoms.get_number_of_atoms() is deprecated. "
            "Use len(PhonopyAtoms).",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self)

    def _set_cell(self, cell):
        _cell = np.array(cell, dtype="double", order="C")
        if _cell.shape == (3, 3):
            self._cell = _cell
        else:
            raise TypeError("Array shape of cell is not 3x3.")

    def _set_positions(self, cart_positions):
        self._scaled_positions = np.array(
            np.dot(cart_positions, np.linalg.inv(self._cell)), dtype="double", order="C"
        )

    def _set_scaled_positions(self, scaled_positions):
        self._scaled_positions = np.array(scaled_positions, dtype="double", order="C")

    def _set_masses(self, masses):
        if masses is None:
            self._masses = None
        else:
            self._masses = np.array(masses, dtype="double")

    def _set_magnetic_moments(self, magmoms):
        if magmoms is None:
            self._magnetic_moments = None
        else:
            self._magnetic_moments = np.array(np.ravel(magmoms), dtype="double")

    def _set_cell_and_positions(self, cell, positions=None, scaled_positions=None):
        self._set_cell(cell)
        if positions is not None:
            self._set_positions(positions)
        elif scaled_positions is not None:
            self._set_scaled_positions(scaled_positions)

    def _numbers_to_symbols(self):
        symbols = []
        for number in self._numbers_with_shifts:
            n = number % self._MOD_DIVISOR
            m = number // self._MOD_DIVISOR
            if m > 0:
                symbols.append(f"{atom_data[n][1]}{m}")
            else:
                symbols.append(f"{atom_data[n][1]}")
        self._symbols = symbols

    def _symbols_to_numbers(self):
        numbers = []
        for symnum in self._symbols:
            symbol, index = split_symbol_and_index(symnum)
            numbers.append(symbol_map[symbol] + self._MOD_DIVISOR * index)

        self._numbers_with_shifts = np.array(numbers, dtype="intc")

    def _symbols_to_masses(self):
        masses = [atom_data[symbol_map[s]][3] for s in self._symbols]
        if None in masses:
            self._masses = None
        else:
            self._masses = np.array(masses, dtype="double")

    def _check(self):
        """Check number of eleemnts in arrays.

        Do not modify the arrays. Modification of array shapes should be done in
        ``self._finalize()``.

        """
        if self._cell is None:
            raise RuntimeError("cell is not set.")
        if self._scaled_positions is None:
            raise RuntimeError("scaled_positions (positions) is not set.")
        if self._numbers_with_shifts is None:
            raise RuntimeError("numbers is not set.")
        if len(self._numbers_with_shifts) != len(self._scaled_positions):
            raise RuntimeError("len(numbers) != len(scaled_positions).")
        if len(self._numbers_with_shifts) != len(self._symbols):
            raise RuntimeError("len(numbers) != len(symbols).")
        if self._masses is not None:
            if len(self._numbers_with_shifts) != len(self._masses):
                raise RuntimeError("len(numbers) != len(masses).")
        if self._magnetic_moments is not None:
            if len(self._magnetic_moments.ravel()) not in (len(self), len(self) * 3):
                raise RuntimeError(
                    "magnetic_moments has to have shape=(natom,) or (natom, 3)."
                )

    def _finalize(self):
        """Modify array shapes to those expeted to be exposed."""
        # When non collinear magnetic moments is given in a flat array.
        if self.magnetic_moments is not None:
            if len(self.magnetic_moments.ravel()) == len(self) * 3:
                self._magnetic_moments = np.reshape(self._magnetic_moments, (-1, 3))

    def copy(self):
        """Return copy of itself."""
        return PhonopyAtoms(
            cell=self._cell,
            scaled_positions=self._scaled_positions,
            masses=self._masses,
            magnetic_moments=self._magnetic_moments,
            symbols=self._symbols,
        )

    def totuple(self, distinguish_symbol_index: bool = False):
        """Return (cell, scaled_position, numbers).

        If magmams is set, (cell, scaled_position, numbers, magmoms) is
        returned.

        Parameters
        ----------
        with_symbol_index : bool
            If True, number is replaced with atomic number + index *
            self.MOD_DIVISOR.

            'H' --> 1
            'H2' --> 1 + self.MOD_DIVISOR * 2

        """
        if distinguish_symbol_index:
            numbers = self._numbers_with_shifts
        else:
            numbers = self.numbers

        if self._magnetic_moments is None:
            return (self._cell, self._scaled_positions, numbers)
        else:
            return (
                self._cell,
                self._scaled_positions,
                numbers,
                self._magnetic_moments,
            )

    def to_tuple(self):
        """Return (cell, scaled_position, numbers).

        If magmams is set, (cell, scaled_position, numbers, magmoms) is returned.

        """
        warnings.warn(
            "PhonopyAtoms.to_tuple() is deprecated. Use totuple() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.totuple()

    def get_yaml_lines(self):
        """Return list of text lines of crystal structure in yaml."""
        lines = ["lattice:"]
        for v, a in zip(self._cell, ("a", "b", "c")):
            lines.append("- [ %21.15f, %21.15f, %21.15f ] # %s" % (v[0], v[1], v[2], a))
        lines.append("points:")
        if self._masses is None:
            masses = [None] * len(self._symbols)
        else:
            masses = self._masses
        if self._magnetic_moments is None:
            magmoms = [None] * len(self._symbols)
        else:
            magmoms = self._magnetic_moments
        for i, (s, num, v, m, mag) in enumerate(
            zip(self._symbols, self.numbers, self._scaled_positions, masses, magmoms)
        ):
            formal_s = atom_data[num][1]
            if s == formal_s:
                lines.append(f"- symbol: {s} # {i + 1}")
            else:
                lines.append(f"- symbol: {formal_s} # {i + 1}")
                lines.append(f"  extended_symbol: {s}")
            lines.append("  coordinates: [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            if m is not None:
                lines.append("  mass: %f" % m)
            if mag is not None:
                if mag.ndim == 0:
                    mag_str = f"{mag:.8f}"
                else:
                    mag_str = f"[{mag[0]:.8f}, {mag[1]:.8f}, {mag[2]:.8f}]"
                lines.append(f"  magnetic_moment: {mag_str}")
        return lines

    def __str__(self):
        """Return text lines of crystal structure in yaml."""
        return "\n".join(self.get_yaml_lines())


def parse_cell_dict(cell_dict: dict) -> Optional[PhonopyAtoms]:
    """Parse cell dict."""
    lattice = None
    if "lattice" in cell_dict:
        lattice = cell_dict["lattice"]
    else:
        return None
    points = []
    symbols = []
    masses = []
    magnetic_moments = []
    if "points" in cell_dict:
        for x in cell_dict["points"]:
            if "coordinates" in x:
                points.append(x["coordinates"])
            if "extended_symbol" in x:  # like Fe1
                symbols.append(x["extended_symbol"])
            elif "symbol" in x:  # like Fe
                symbols.append(x["symbol"])
            if "mass" in x:
                masses.append(x["mass"])
            if "magnetic_moment" in x:
                magnetic_moments.append(x["magnetic_moment"])
    # For version < 1.10.9
    elif "atoms" in cell_dict:
        for x in cell_dict["atoms"]:
            if "coordinates" not in x and "position" in x:
                points.append(x["position"])
            if "symbol" in x:
                symbols.append(x["symbol"])
            if "mass" in x:
                masses.append(x["mass"])

    if not masses:
        masses = None
    if not magnetic_moments:
        magnetic_moments = None

    if points and symbols:
        return PhonopyAtoms(
            symbols=symbols,
            cell=lattice,
            masses=masses,
            scaled_positions=points,
            magnetic_moments=magnetic_moments,
        )
    else:
        return None


# Pure Appl. Chem., Vol. 83, No. 2, pp. 359-396, 2011. is available
# but the following list is from 2006.

# Pure Appl. Chem., Vol. 78, No. 11, pp. 2051-2066, 2006.
# The masses of following elements are obtained from wikipedia:
# Ac: 227
# Np: 237
# Pm: 145
# Tc: 98
atom_data = [
    [0, "X", "X", None],  # 0
    [1, "H", "Hydrogen", 1.00794],  # 1
    [2, "He", "Helium", 4.002602],  # 2
    [3, "Li", "Lithium", 6.941],  # 3
    [4, "Be", "Beryllium", 9.012182],  # 4
    [5, "B", "Boron", 10.811],  # 5
    [6, "C", "Carbon", 12.0107],  # 6
    [7, "N", "Nitrogen", 14.0067],  # 7
    [8, "O", "Oxygen", 15.9994],  # 8
    [9, "F", "Fluorine", 18.9984032],  # 9
    [10, "Ne", "Neon", 20.1797],  # 10
    [11, "Na", "Sodium", 22.98976928],  # 11
    [12, "Mg", "Magnesium", 24.3050],  # 12
    [13, "Al", "Aluminium", 26.9815386],  # 13
    [14, "Si", "Silicon", 28.0855],  # 14
    [15, "P", "Phosphorus", 30.973762],  # 15
    [16, "S", "Sulfur", 32.065],  # 16
    [17, "Cl", "Chlorine", 35.453],  # 17
    [18, "Ar", "Argon", 39.948],  # 18
    [19, "K", "Potassium", 39.0983],  # 19
    [20, "Ca", "Calcium", 40.078],  # 20
    [21, "Sc", "Scandium", 44.955912],  # 21
    [22, "Ti", "Titanium", 47.867],  # 22
    [23, "V", "Vanadium", 50.9415],  # 23
    [24, "Cr", "Chromium", 51.9961],  # 24
    [25, "Mn", "Manganese", 54.938045],  # 25
    [26, "Fe", "Iron", 55.845],  # 26
    [27, "Co", "Cobalt", 58.933195],  # 27
    [28, "Ni", "Nickel", 58.6934],  # 28
    [29, "Cu", "Copper", 63.546],  # 29
    [30, "Zn", "Zinc", 65.38],  # 30
    [31, "Ga", "Gallium", 69.723],  # 31
    [32, "Ge", "Germanium", 72.64],  # 32
    [33, "As", "Arsenic", 74.92160],  # 33
    [34, "Se", "Selenium", 78.96],  # 34
    [35, "Br", "Bromine", 79.904],  # 35
    [36, "Kr", "Krypton", 83.798],  # 36
    [37, "Rb", "Rubidium", 85.4678],  # 37
    [38, "Sr", "Strontium", 87.62],  # 38
    [39, "Y", "Yttrium", 88.90585],  # 39
    [40, "Zr", "Zirconium", 91.224],  # 40
    [41, "Nb", "Niobium", 92.90638],  # 41
    [42, "Mo", "Molybdenum", 95.96],  # 42
    [43, "Tc", "Technetium", 98],  # 43 (mass is from wikipedia)
    [44, "Ru", "Ruthenium", 101.07],  # 44
    [45, "Rh", "Rhodium", 102.90550],  # 45
    [46, "Pd", "Palladium", 106.42],  # 46
    [47, "Ag", "Silver", 107.8682],  # 47
    [48, "Cd", "Cadmium", 112.411],  # 48
    [49, "In", "Indium", 114.818],  # 49
    [50, "Sn", "Tin", 118.710],  # 50
    [51, "Sb", "Antimony", 121.760],  # 51
    [52, "Te", "Tellurium", 127.60],  # 52
    [53, "I", "Iodine", 126.90447],  # 53
    [54, "Xe", "Xenon", 131.293],  # 54
    [55, "Cs", "Caesium", 132.9054519],  # 55
    [56, "Ba", "Barium", 137.327],  # 56
    [57, "La", "Lanthanum", 138.90547],  # 57
    [58, "Ce", "Cerium", 140.116],  # 58
    [59, "Pr", "Praseodymium", 140.90765],  # 59
    [60, "Nd", "Neodymium", 144.242],  # 60
    [61, "Pm", "Promethium", 145],  # 61 (mass is from wikipedia)
    [62, "Sm", "Samarium", 150.36],  # 62
    [63, "Eu", "Europium", 151.964],  # 63
    [64, "Gd", "Gadolinium", 157.25],  # 64
    [65, "Tb", "Terbium", 158.92535],  # 65
    [66, "Dy", "Dysprosium", 162.500],  # 66
    [67, "Ho", "Holmium", 164.93032],  # 67
    [68, "Er", "Erbium", 167.259],  # 68
    [69, "Tm", "Thulium", 168.93421],  # 69
    [70, "Yb", "Ytterbium", 173.054],  # 70
    [71, "Lu", "Lutetium", 174.9668],  # 71
    [72, "Hf", "Hafnium", 178.49],  # 72
    [73, "Ta", "Tantalum", 180.94788],  # 73
    [74, "W", "Tungsten", 183.84],  # 74
    [75, "Re", "Rhenium", 186.207],  # 75
    [76, "Os", "Osmium", 190.23],  # 76
    [77, "Ir", "Iridium", 192.217],  # 77
    [78, "Pt", "Platinum", 195.084],  # 78
    [79, "Au", "Gold", 196.966569],  # 79
    [80, "Hg", "Mercury", 200.59],  # 80
    [81, "Tl", "Thallium", 204.3833],  # 81
    [82, "Pb", "Lead", 207.2],  # 82
    [83, "Bi", "Bismuth", 208.98040],  # 83
    [84, "Po", "Polonium", None],  # 84
    [85, "At", "Astatine", None],  # 85
    [86, "Rn", "Radon", None],  # 86
    [87, "Fr", "Francium", None],  # 87
    [88, "Ra", "Radium", None],  # 88
    [89, "Ac", "Actinium", 227],  # 89 (mass is from wikipedia)
    [90, "Th", "Thorium", 232.03806],  # 90
    [91, "Pa", "Protactinium", 231.03588],  # 91
    [92, "U", "Uranium", 238.02891],  # 92
    [93, "Np", "Neptunium", 237],  # 93 (mass is from wikipedia)
    [94, "Pu", "Plutonium", None],  # 94
    [95, "Am", "Americium", None],  # 95
    [96, "Cm", "Curium", None],  # 96
    [97, "Bk", "Berkelium", None],  # 97
    [98, "Cf", "Californium", None],  # 98
    [99, "Es", "Einsteinium", None],  # 99
    [100, "Fm", "Fermium", None],  # 100
    [101, "Md", "Mendelevium", None],  # 101
    [102, "No", "Nobelium", None],  # 102
    [103, "Lr", "Lawrencium", None],  # 103
    [104, "Rf", "Rutherfordium", None],  # 104
    [105, "Db", "Dubnium", None],  # 105
    [106, "Sg", "Seaborgium", None],  # 106
    [107, "Bh", "Bohrium", None],  # 107
    [108, "Hs", "Hassium", None],  # 108
    [109, "Mt", "Meitnerium", None],  # 109
    [110, "Ds", "Darmstadtium", None],  # 110
    [111, "Rg", "Roentgenium", None],  # 111
    [112, "Cn", "Copernicium", None],  # 112
    [113, "Uut", "Ununtrium", None],  # 113
    [114, "Uuq", "Ununquadium", None],  # 114
    [115, "Uup", "Ununpentium", None],  # 115
    [116, "Uuh", "Ununhexium", None],  # 116
    [117, "Uus", "Ununseptium", None],  # 117
    [118, "Uuo", "Ununoctium", None],  # 118
]

symbol_map = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Uut": 113,
    "Uuq": 114,
    "Uup": 115,
    "Uuh": 116,
    "Uus": 117,
    "Uuo": 118,
}


isotope_data_CIAAW = {
    # https://www.ciaaw.org/molybdenum.htm (accessed at 14th Jun. 2022)
    # Isotope Mo
    # Adam J. Mayer* and Michael E. Wieser, J. Anal. At. Spectrom., 2014, 29, 85
    # DOI: 10.1039/c3ja50164g
    # "The absolute isotopic composition and atomic weight of molybdenum in
    # SRM 3134 using an isotopic double-spike"
    "Mo": [
        [92, 91.906807, 0.14649],
        [94, 93.905084, 0.09187],
        [95, 94.9058374, 0.15873],
        [96, 95.9046748, 0.16673],
        [97, 96.906017, 0.09582],
        [98, 97.905404, 0.24292],
        [100, 99.907468, 0.09744],
    ]
}

# This data are obtained from
# J. R. de Laeter, J. K. Böhlke, P. De Bièvre, H. Hidaka, H. S. Peiser,
# K. J. R. Rosman and P. D. P. Taylor (2003).
# "Atomic weights of the elements. Review 2000 (IUPAC Technical Report)"
isotope_data = {
    "H": [[1, 1.0078250319, 0.999885], [2, 2.0141017779, 0.000115]],
    "He": [[3, 3.0160293094, 0.00000134], [4, 4.0026032497, 0.99999866]],
    "Li": [[6, 6.0151223, 0.0759], [7, 7.0160041, 0.9241]],
    "Be": [[9, 9.0121822, 1.0000]],
    "B": [[10, 10.0129371, 0.199], [11, 11.0093055, 0.801]],
    "C": [[12, 12, 0.9893], [13, 13.003354838, 0.0107]],
    "N": [[14, 14.0030740074, 0.99636], [15, 15.000108973, 0.00364]],
    "O": [
        [16, 15.9949146223, 0.99757],
        [17, 16.99913150, 0.00038],
        [18, 17.9991604, 0.00205],
    ],
    "F": [[19, 18.99840320, 1.0000]],
    "Ne": [
        [20, 19.992440176, 0.9048],
        [21, 20.99384674, 0.0027],
        [22, 21.99138550, 0.0925],
    ],
    "Na": [[23, 22.98976966, 1.0000]],
    "Mg": [
        [24, 23.98504187, 0.7899],
        [25, 24.98583700, 0.1000],
        [26, 25.98259300, 0.1101],
    ],
    "Al": [[27, 26.98153841, 1.0000]],
    "Si": [
        [28, 27.97692649, 0.92223],
        [29, 28.97649468, 0.04685],
        [30, 29.97377018, 0.03092],
    ],
    "P": [[31, 30.97376149, 1.0000]],
    "S": [
        [32, 31.97207073, 0.9499],
        [33, 32.97145854, 0.0075],
        [34, 33.96786687, 0.0425],
        [36, 35.96708088, 0.0001],
    ],
    "Cl": [[35, 34.96885271, 0.7576], [37, 36.96590260, 0.2424]],
    "Ar": [
        [36, 35.96754626, 0.003365],
        [38, 37.9627322, 0.000632],
        [40, 39.962383124, 0.996003],
    ],
    "K": [
        [39, 38.96370, 0.932581],
        [40, 39.96399867, 0.000117],
        [41, 40.96182597, 0.067302],
    ],
    "Ca": [
        [40, 39.9625912, 0.96941],
        [42, 41.9586183, 0.00647],
        [43, 42.9587668, 0.00135],
        [44, 43.9554811, 0.02086],
        [46, 45.9536927, 0.00004],
        [48, 47.952533, 0.00187],
    ],
    "Sc": [[45, 44.9559102, 1.0000]],
    "Ti": [
        [46, 45.9526295, 0.0825],
        [47, 46.9517637, 0.0744],
        [48, 47.9479470, 0.7372],
        [49, 48.9478707, 0.0541],
        [50, 49.9447920, 0.0518],
    ],
    "V": [[50, 49.9471627, 0.00250], [51, 50.9439635, 0.99750]],
    "Cr": [
        [50, 49.9460495, 0.04345],
        [52, 51.9405115, 0.83789],
        [53, 52.9406534, 0.09501],
        [54, 53.9388846, 0.02365],
    ],
    "Mn": [[55, 54.9380493, 1.0000]],
    "Fe": [
        [54, 53.9396147, 0.05845],
        [56, 55.9349418, 0.91754],
        [57, 56.9353983, 0.02119],
        [58, 57.9332801, 0.00282],
    ],
    "Co": [[59, 58.9331999, 1.0000]],
    "Ni": [
        [58, 57.9353477, 0.680769],
        [60, 59.9307903, 0.262231],
        [61, 60.9310601, 0.011399],
        [62, 61.9283484, 0.036345],
        [64, 63.9279692, 0.009256],
    ],
    "Cu": [[63, 62.9296007, 0.6915], [65, 64.9277938, 0.3085]],
    "Zn": [
        [64, 63.9291461, 0.48268],
        [66, 65.9260364, 0.27975],
        [67, 66.9271305, 0.04102],
        [68, 67.9248473, 0.19024],
        [70, 69.925325, 0.00631],
    ],
    "Ga": [[69, 68.925581, 0.60108], [71, 70.9247073, 0.39892]],
    "Ge": [
        [70, 69.9242500, 0.2038],
        [72, 71.9220763, 0.2731],
        [73, 72.9234595, 0.0776],
        [74, 73.9211784, 0.3672],
        [76, 75.921402, 0.0783],
    ],
    "As": [[75, 74.9215966, 1.0000]],
    "Se": [
        [74, 73.9224767, 0.0089],
        [76, 75.9192143, 0.0937],
        [77, 76.9199148, 0.0763],
        [78, 77.9173097, 0.2377],
        [80, 79.9165221, 0.4961],
        [82, 81.9167003, 0.0873],
    ],
    "Br": [[79, 78.9183379, 0.5069], [81, 80.916291, 0.4931]],
    "Kr": [
        [78, 77.920388, 0.00355],
        [80, 79.916379, 0.02286],
        [82, 81.9134850, 0.11593],
        [83, 82.914137, 0.11500],
        [84, 83.911508, 0.56987],
        [86, 85.910615, 0.17279],
    ],
    "Rb": [[85, 84.9117924, 0.7217], [87, 86.9091858, 0.2783]],
    "Sr": [
        [84, 83.913426, 0.0056],
        [86, 85.9092647, 0.0986],
        [87, 86.9088816, 0.0700],
        [88, 87.9056167, 0.8258],
    ],
    "Y": [[89, 88.9058485, 1.0000]],
    "Zr": [
        [90, 89.9047022, 0.5145],
        [91, 90.9056434, 0.1122],
        [92, 91.9050386, 0.1715],
        [94, 93.9063144, 0.1738],
        [96, 95.908275, 0.0280],
    ],
    "Nb": [[93, 92.9063762, 1.0000]],
    "Mo": [
        [92, 91.906810, 0.1477],
        [94, 93.9050867, 0.0923],
        [95, 94.9058406, 0.1590],
        [96, 95.9046780, 0.1668],
        [97, 96.9060201, 0.0956],
        [98, 97.905406, 0.2419],
        [100, 99.907476, 0.0967],
    ],
    "Tc": None,
    "Ru": [
        [96, 95.907604, 0.0554],
        [98, 97.905287, 0.0187],
        [99, 98.9059385, 0.1276],
        [100, 99.9042189, 0.1260],
        [101, 100.9055815, 0.1706],
        [102, 101.9043488, 0.3155],
        [104, 103.905430, 0.1862],
    ],
    "Rh": [[103, 102.905504, 1.0000]],
    "Pd": [
        [102, 101.905607, 0.0102],
        [104, 103.904034, 0.1114],
        [105, 104.905083, 0.2233],
        [106, 105.903484, 0.2733],
        [108, 107.903895, 0.2646],
        [110, 109.905153, 0.1172],
    ],
    "Ag": [[107, 106.905093, 0.51839], [109, 108.904756, 0.48161]],
    "Cd": [
        [106, 105.906458, 0.0125],
        [108, 107.904183, 0.0089],
        [110, 109.903006, 0.1249],
        [111, 110.904182, 0.1280],
        [112, 111.9027577, 0.2413],
        [113, 112.9044014, 0.1222],
        [114, 113.9033586, 0.2873],
        [116, 115.904756, 0.0749],
    ],
    "In": [[113, 112.904062, 0.0429], [115, 114.903879, 0.9571]],
    "Sn": [
        [112, 111.904822, 0.0097],
        [114, 113.902783, 0.0066],
        [115, 114.903347, 0.0034],
        [116, 115.901745, 0.1454],
        [117, 116.902955, 0.0768],
        [118, 117.901608, 0.2422],
        [119, 118.903311, 0.0859],
        [120, 119.9021985, 0.3258],
        [122, 121.9034411, 0.0463],
        [124, 123.9052745, 0.0579],
    ],
    "Sb": [[121, 120.9038222, 0.5721], [123, 122.9042160, 0.4279]],
    "Te": [
        [120, 119.904026, 0.0009],
        [122, 121.9030558, 0.0255],
        [123, 122.9042711, 0.0089],
        [124, 123.9028188, 0.0474],
        [125, 124.9044241, 0.0707],
        [126, 125.9033049, 0.1884],
        [128, 127.9044615, 0.3174],
        [130, 129.9062229, 0.3408],
    ],
    "I": [[127, 126.904468, 1.0000]],
    "Xe": [
        [124, 123.9058954, 0.000952],
        [126, 125.904268, 0.000890],
        [128, 127.9035305, 0.019102],
        [129, 128.9047799, 0.264006],
        [130, 129.9035089, 0.040710],
        [131, 130.9050828, 0.212324],
        [132, 131.9041546, 0.269086],
        [134, 133.9053945, 0.104357],
        [136, 135.907220, 0.088573],
    ],
    "Cs": [[133, 132.905447, 1.0000]],
    "Ba": [
        [130, 129.906311, 0.00106],
        [132, 131.905056, 0.00101],
        [134, 133.904504, 0.02417],
        [135, 134.905684, 0.06592],
        [136, 135.904571, 0.07854],
        [137, 136.905822, 0.11232],
        [138, 137.905242, 0.71698],
    ],
    "La": [[138, 137.907108, 0.00090], [139, 138.906349, 0.99910]],
    "Ce": [
        [136, 135.907140, 0.00185],
        [138, 137.905986, 0.00251],
        [140, 139.905435, 0.88450],
        [142, 141.909241, 0.11114],
    ],
    "Pr": [[141, 140.907648, 1.0000]],
    "Nd": [
        [142, 141.907719, 0.272],
        [143, 142.909810, 0.122],
        [144, 143.910083, 0.238],
        [145, 144.912569, 0.083],
        [146, 145.913113, 0.172],
        [148, 147.916889, 0.057],
        [150, 149.920887, 0.056],
    ],
    "Pm": None,
    "Sm": [
        [144, 143.911996, 0.0307],
        [147, 146.914894, 0.1499],
        [148, 147.914818, 0.1124],
        [149, 148.917180, 0.1382],
        [150, 149.917272, 0.0738],
        [152, 151.919729, 0.2675],
        [154, 153.922206, 0.2275],
    ],
    "Eu": [[151, 150.919846, 0.4781], [153, 152.921227, 0.5219]],
    "Gd": [
        [152, 151.919789, 0.0020],
        [154, 153.920862, 0.0218],
        [155, 154.922619, 0.1480],
        [156, 155.922120, 0.2047],
        [157, 156.923957, 0.1565],
        [158, 157.924101, 0.2484],
        [160, 159.927051, 0.2186],
    ],
    "Tb": [[159, 158.925343, 1.0000]],
    "Dy": [
        [156, 155.924278, 0.00056],
        [158, 157.924405, 0.00095],
        [160, 159.925194, 0.02329],
        [161, 160.926930, 0.18889],
        [162, 161.926795, 0.25475],
        [163, 162.928728, 0.24896],
        [164, 163.929171, 0.28260],
    ],
    "Ho": [[165, 164.930319, 1.0000]],
    "Er": [
        [162, 161.928775, 0.00139],
        [164, 163.929197, 0.01601],
        [166, 165.930290, 0.33503],
        [167, 166.932046, 0.22869],
        [168, 167.932368, 0.26978],
        [170, 169.935461, 0.14910],
    ],
    "Tm": [[169, 168.934211, 1.0000]],
    "Yb": [
        [168, 167.933895, 0.0013],
        [170, 169.934759, 0.0304],
        [171, 170.936323, 0.1428],
        [172, 171.936378, 0.2183],
        [173, 172.938207, 0.1613],
        [174, 173.938858, 0.3183],
        [176, 175.942569, 0.1276],
    ],
    "Lu": [[175, 174.9407682, 0.9741], [176, 175.9426827, 0.0259]],
    "Hf": [
        [174, 173.940042, 0.0016],
        [176, 175.941403, 0.0526],
        [177, 176.9432204, 0.1860],
        [178, 177.9436981, 0.2728],
        [179, 178.9458154, 0.1362],
        [180, 179.9465488, 0.3508],
    ],
    "Ta": [[180, 179.947466, 0.00012], [181, 180.947996, 0.99988]],
    "W": [
        [180, 179.946706, 0.0012],
        [182, 181.948205, 0.2650],
        [183, 182.9502242, 0.1431],
        [184, 183.9509323, 0.3064],
        [186, 185.95436, 0.2843],
    ],
    "Re": [[185, 184.952955, 0.3740], [187, 186.9557505, 0.6260]],
    "Os": [
        [184, 183.952491, 0.0002],
        [186, 185.953838, 0.0159],
        [187, 186.9557476, 0.0196],
        [188, 187.9558357, 0.1324],
        [189, 188.958145, 0.1615],
        [190, 189.958445, 0.2626],
        [192, 191.961479, 0.4078],
    ],
    "Ir": [[191, 190.960591, 0.373], [193, 192.962923, 0.627]],
    "Pt": [
        [190, 189.959930, 0.00014],
        [192, 191.961035, 0.00782],
        [194, 193.962663, 0.32967],
        [195, 194.964774, 0.33832],
        [196, 195.964934, 0.25242],
        [198, 197.967875, 0.07163],
    ],
    "Au": [[197, 196.966551, 1.0000]],
    "Hg": [
        [196, 195.965814, 0.0015],
        [198, 197.966752, 0.0997],
        [199, 198.968262, 0.1687],
        [200, 199.968309, 0.2310],
        [201, 200.970285, 0.1318],
        [202, 201.970625, 0.2986],
        [204, 203.973475, 0.0687],
    ],
    "Tl": [[203, 202.972329, 0.2952], [205, 204.974412, 0.7048]],
    "Pb": [
        [204, 203.973028, 0.014],
        [206, 205.974449, 0.241],
        [207, 206.975880, 0.221],
        [208, 207.976636, 0.524],
    ],
    "Bi": [[209, 208.980384, 1.0000]],
    "Po": None,
    "At": None,
    "Rn": None,
    "Fr": None,
    "Ra": None,
    "Ac": None,
    "Th": [[232, 232.0380495, 1.0000]],
    "Pa": [[231, 231.03588, 1.0000]],
    "U": [
        [234, 234.0409447, 0.000054],
        [235, 235.0439222, 0.007204],
        [238, 238.0507835, 0.992742],
    ],
}
