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

import dataclasses
import re
import warnings
from collections.abc import Sequence
from math import gcd
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from phonopy.structure.atomic_data import get_atomic_data


class _PointEntry(TypedDict, total=False):
    coordinates: list[float]
    symbol: str
    extended_symbol: str
    mass: float
    magnetic_moment: float | list[float]


class _LegacyAtomEntry(TypedDict, total=False):
    """Legacy format (phonopy < v1.10.9)."""

    position: list[float]
    coordinates: list[float]
    symbol: str
    mass: float


class _CellDictBase(TypedDict):
    """Base class holding the required field of CellDict.

    Separated from CellDict because TypedDict does not support mixing required
    and optional fields in a single class (Python < 3.11). Required fields are
    defined here with the default total=True, while optional fields are added
    in the subclass with total=False.

    """

    lattice: list[list[float]]


class CellDict(_CellDictBase, total=False):
    """Dict representation of a crystal cell as parsed from phonopy YAML."""

    points: list[_PointEntry]
    atoms: list[_LegacyAtomEntry]


def Atoms(*args, **kwargs) -> "PhonopyAtoms":
    """Atoms class that is same as PhonopyAtoms class.

    This exists backward compatibility.

    """
    warnings.warn(
        "phonopy.atoms.Atoms is deprecated. Please use PhonopyAtoms instead of Atoms.",
        DeprecationWarning,
        stacklevel=2,
    )
    return PhonopyAtoms(*args, **kwargs)


_SYMBOL_INDEX_RE = re.compile(r"^([A-Z][a-z]{0,2})([0-9]*)$")


def split_symbol_and_index(symnum: str) -> tuple[str, int]:
    """Split a single-element symbol with an optional natural-number suffix.

    Accepted: a single chemical symbol (1 uppercase letter, optionally followed
    by up to 2 lowercase letters — Uuo etc.), optionally followed by digits
    forming a positive integer suffix.

    H --> H, 0
    H2 --> H, 2
    Cl1 --> Cl, 1
    Uuo --> Uuo, 0

    Composite symbols like "GeSn" or invalid forms like "Cl_1" raise
    RuntimeError.

    """
    m = _SYMBOL_INDEX_RE.match(symnum)
    if m is None:
        raise RuntimeError(f"Invalid symbol: {symnum}.")
    symbol, index_str = m.groups()
    if index_str:
        index = int(index_str)
        if index < 1:
            raise RuntimeError(
                f"Invalid symbol. Index has to be greater than 0: {symnum}."
            )
    else:
        index = 0
    return symbol, index


@dataclasses.dataclass(frozen=True)
class _Species:
    """Identity of a chemical species used inside PhonopyAtoms.

    Two atoms share a species iff their (symbol, atomic_number) tuple is
    equal. ``symbol`` carries the suffix ("Cl1") so that "Cl" and "Cl1"
    are distinct species that share the same atomic number.

    """

    symbol: str
    atomic_number: int


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
        Atomic numbers per atom in 1..118. shape=(natom,), dtype='int64'
    species_ids : np.ndarray
        Per-atom unique species index. Atoms with the same (symbol,
        atomic_number) get the same id; "Cl" and "Cl1" get different ids.
        Suitable as the ``types`` argument for spglib when atoms with the
        same atomic number but different symbol suffix must be distinguished.
        shape=(natom,), dtype='int64'
    masses : np.ndarray, optional
        Atomic masses. shape=(natom,), dtype='double'
    magnetic_moments : np.ndarray, optional
        shape=(natom,), (natom*3), (natom, 3), dtype='double', order='C'
    volume : float
        Cell volume.
    Z : int
        Number of formula units in this cell.

    """

    def __init__(
        self,
        symbols: Sequence[str] | None = None,
        numbers: Sequence[int] | NDArray[np.int64] | None = None,
        masses: Sequence[float] | NDArray[np.double] | None = None,
        magnetic_moments: Sequence[float]
        | Sequence[Sequence[float]]
        | NDArray[np.double]
        | None = None,
        scaled_positions: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
        positions: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
        cell: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    ) -> None:  # pbc is dummy argument, and never used.
        """Set crystal structure parameters.

        Exactly one of ``symbols`` or ``numbers`` must be specified. ``numbers``
        accepts atomic numbers in the range 1..118.

        """
        self._cell: NDArray[np.double]
        self._scaled_positions: NDArray[np.double]
        self._species: list[_Species]
        self._species_ids: NDArray[np.int64]
        self._magnetic_moments: NDArray[np.double] | None
        self._masses: NDArray[np.double]

        self._set_cell_and_positions(
            cell, positions=positions, scaled_positions=scaled_positions
        )

        if symbols is None and numbers is None:
            raise RuntimeError("Either symbols or numbers has to be specified.")
        if symbols is not None and numbers is not None:
            raise ValueError("symbols and numbers cannot be specified together.")

        if symbols is not None:
            self._build_species_from_symbols(list(symbols))
        else:
            assert numbers is not None
            self._build_species_from_numbers(np.asarray(numbers, dtype="int64"))

        if masses is not None:
            self._set_masses(masses)
        else:
            self._set_default_masses()

        self._set_magnetic_moments(magnetic_moments)

        self._check()

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._species_ids)

    @property
    def cell(self) -> NDArray[np.double]:
        """Setter and getter of basis vectors. For getter, copy is returned."""
        return self._cell.copy()

    @cell.setter
    def cell(self, cell: Sequence[Sequence[float]] | NDArray[np.double]) -> None:
        self._set_cell(cell)
        self._check()

    @property
    def positions(self) -> NDArray[np.double]:
        """Setter and getter of positions in Cartesian coordinates."""
        return np.array(
            np.dot(self._scaled_positions, self._cell), dtype="double", order="C"
        )

    @positions.setter
    def positions(self, positions: Sequence[Sequence[float]] | NDArray[np.double]):
        self._set_positions(positions)
        self._check()

    @property
    def scaled_positions(self) -> NDArray[np.double]:
        """Setter and getter of scaled positions. For getter, copy is returned."""
        return self._scaled_positions.copy()

    @scaled_positions.setter
    def scaled_positions(
        self, scaled_positions: Sequence[Sequence[float]] | NDArray[np.double]
    ):
        self._set_scaled_positions(scaled_positions)
        self._check()

    @property
    def symbols(self) -> list[str]:
        """Chemical symbols per atom."""
        return [self._species[sid].symbol for sid in self._species_ids]

    @property
    def species_ids(self) -> NDArray[np.int64]:
        """Per-atom unique species index. For getter, copy is returned.

        Atoms with the same (symbol, atomic_number) pair share an id; "Cl"
        and "Cl1" get different ids. Suitable as the ``types`` argument for
        spglib when atoms with the same atomic number but different symbol
        suffix must be distinguished.

        """
        return self._species_ids.copy()

    @property
    def numbers(self) -> NDArray[np.int64]:
        """Atomic numbers per atom in 1..118. For getter, new array is returned."""
        return np.array(
            [self._species[sid].atomic_number for sid in self._species_ids],
            dtype="int64",
        )

    @property
    def masses(self) -> NDArray[np.double]:
        """Setter and getter of atomic masses. For getter copy is returned."""
        return self._masses.copy()

    @masses.setter
    def masses(self, masses: Sequence[float] | NDArray[np.double]):
        self._set_masses(masses)
        self._check()

    @property
    def magnetic_moments(self) -> NDArray[np.double] | None:
        """Setter and getter of magnetic moments. For getter, copy is returned.

        shape=(natom,) or (natom, 3), dtype='double', order='C'

        For setter, the formar can be specified by (natom, 1), which will be
        recognized as (natom,) and the latter can be specified by (natom * 3,),
        which will be converted to (natom, 3).

        """
        if self._magnetic_moments is None:
            return None
        else:
            if len(self._magnetic_moments) == len(self) * 3:
                return np.reshape(self._magnetic_moments, (-1, 3)).copy()
            elif len(self._magnetic_moments) == len(self):
                return self._magnetic_moments.copy()
            else:
                raise RuntimeError(
                    "_magnetic_moments has to have shape=(natom,) or (natom*3)."
                )

    @magnetic_moments.setter
    def magnetic_moments(
        self,
        magnetic_moments: Sequence[float]
        | Sequence[Sequence[float]]
        | NDArray[np.double]
        | None,
    ):
        self._set_magnetic_moments(magnetic_moments)
        self._check()

    @property
    def volume(self) -> float:
        """Return cell volume."""
        return float(np.linalg.det(self._cell))

    @property
    def Z(self) -> int:
        """Return number of formula units in this cell."""
        count: dict[int, int] = {}
        for sid in self._species_ids:
            count[sid] = count.get(sid, 0) + 1
        values = list(count.values())
        x = values[0]
        for v in values[1:]:
            x = gcd(x, v)
        return x

    def _set_cell(self, cell: Sequence[Sequence[float]] | NDArray[np.double]) -> None:
        _cell = np.array(cell, dtype="double", order="C")
        if _cell.shape == (3, 3):
            self._cell = _cell
        else:
            raise TypeError("Array shape of cell is not 3x3.")

    def _set_positions(
        self, cart_positions: Sequence[Sequence[float]] | NDArray[np.double]
    ) -> None:
        self._scaled_positions = np.array(
            np.dot(cart_positions, np.linalg.inv(self._cell)), dtype="double", order="C"
        )

    def _set_scaled_positions(
        self, scaled_positions: Sequence[Sequence[float]] | NDArray[np.double]
    ) -> None:
        self._scaled_positions = np.array(scaled_positions, dtype="double", order="C")

    def _set_masses(self, masses: Sequence[float] | NDArray[np.double]) -> None:
        self._masses = np.array(masses, dtype="double")

    def _set_magnetic_moments(
        self,
        magmoms: Sequence[float]
        | Sequence[Sequence[float]]
        | NDArray[np.double]
        | None,
    ) -> None:
        """Set magnetic moments in 1D array of shape=(natom,) or (natom*3)."""
        if magmoms is None:
            self._magnetic_moments = None
        else:
            if len(np.ravel(magmoms)) not in (len(self) * 3, len(self)):
                raise RuntimeError(
                    "magnetic_moments has to have shape=(natom,) or (natom*3)."
                )
            self._magnetic_moments = np.array(np.ravel(magmoms), dtype="double")

    def _set_cell_and_positions(
        self,
        cell: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
        positions: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
        scaled_positions: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    ) -> None:
        if cell is not None:
            self._set_cell(cell)
        if positions is not None:
            self._set_positions(positions)
        elif scaled_positions is not None:
            self._set_scaled_positions(scaled_positions)

    def _build_species_from_symbols(self, symbols: list[str]) -> None:
        """Parse symbol strings and populate _species and _species_ids."""
        symbol_map = get_atomic_data().symbol_map
        species_table: dict[_Species, int] = {}
        species_list: list[_Species] = []
        ids: list[int] = []
        for symnum in symbols:
            base, _ = split_symbol_and_index(symnum)
            if base not in symbol_map:
                raise RuntimeError(f"Invalid symbol: {symnum}.")
            sp = _Species(symbol=symnum, atomic_number=symbol_map[base])
            sid = species_table.get(sp)
            if sid is None:
                sid = len(species_list)
                species_list.append(sp)
                species_table[sp] = sid
            ids.append(sid)
        self._species = species_list
        self._species_ids = np.array(ids, dtype="int64")

    def _build_species_from_numbers(self, numbers: NDArray[np.int64]) -> None:
        """Populate _species and _species_ids from atomic numbers (1..118)."""
        if numbers.size and ((numbers > 118).any() or (numbers < 1).any()):
            raise ValueError("Atomic numbers must be in 1..118.")
        atom_data = get_atomic_data().atom_data
        species_table: dict[_Species, int] = {}
        species_list: list[_Species] = []
        ids: list[int] = []
        for n in numbers:
            sp = _Species(symbol=atom_data[n][1], atomic_number=int(n))
            sid = species_table.get(sp)
            if sid is None:
                sid = len(species_list)
                species_list.append(sp)
                species_table[sp] = sid
            ids.append(sid)
        self._species = species_list
        self._species_ids = np.array(ids, dtype="int64")

    def _set_default_masses(self) -> None:
        """Set _masses from the periodic table using each atom's species."""
        atom_data = get_atomic_data().atom_data
        masses: list[float] = []
        undefined: set[str] = set()
        for sid in self._species_ids:
            sp = self._species[sid]
            m = atom_data[sp.atomic_number][3]
            if m is None:
                undefined.add(sp.symbol)
            else:
                masses.append(m)
        if undefined:
            raise RuntimeError(
                f"Masses of {undefined} are undefined. "
                "These have to be specified by masses parameter."
            )
        self._masses = np.array(masses, dtype="double")

    def _check(self) -> None:
        """Check number of elements in arrays.

        Do not modify the arrays.

        """
        if self._cell is None:
            raise RuntimeError("cell is not set.")
        if self._scaled_positions is None:
            raise RuntimeError("scaled_positions (positions) is not set.")
        if self._species_ids is None:
            raise RuntimeError("species_ids is not set.")
        natom = len(self._scaled_positions)
        if len(self._species_ids) != natom:
            raise RuntimeError("len(species_ids) != len(scaled_positions).")
        if len(self._masses) != natom:
            raise RuntimeError("len(masses) != len(scaled_positions).")
        if self._species_ids.size and (
            self._species_ids.max() >= len(self._species) or self._species_ids.min() < 0
        ):
            raise RuntimeError("species_ids out of range of species table.")
        if self._magnetic_moments is not None:
            if len(self._magnetic_moments) not in (len(self), len(self) * 3):
                raise RuntimeError(
                    "_magnetic_moments has to have shape=(natom,) or (natom*3)."
                )

    def copy(self) -> PhonopyAtoms:
        """Return copy of itself."""
        return PhonopyAtoms(
            cell=self._cell,
            scaled_positions=self._scaled_positions,
            masses=self._masses,
            magnetic_moments=self._magnetic_moments,
            symbols=self.symbols,
        )

    def totuple(
        self, distinguish_symbol_index: bool = False
    ) -> (
        tuple[NDArray[np.double], NDArray[np.double], NDArray[np.int64]]
        | tuple[
            NDArray[np.double],
            NDArray[np.double],
            NDArray[np.int64],
            NDArray[np.double] | None,
        ]
    ):
        """Return (cell, scaled_position, numbers).

        If magmoms is set, (cell, scaled_position, numbers, magmoms) is
        returned.

        Parameters
        ----------
        distinguish_symbol_index : bool
            If True, the per-atom integer is the species id (atoms with the
            same symbol but different suffix get different ids); suitable as
            the ``types`` argument for spglib when symbol-suffix groupings
            must be preserved. If False (default), the per-atom integer is
            the atomic number.

        """
        if distinguish_symbol_index:
            numbers = self.species_ids
        else:
            numbers = self.numbers

        if self._magnetic_moments is None:
            return (self.cell, self.scaled_positions, numbers)
        else:
            return (
                self.cell,
                self.scaled_positions,
                numbers,
                self.magnetic_moments,
            )

    def get_yaml_lines(self) -> list[str]:
        """Return list of text lines of crystal structure in yaml."""
        _atom_data = get_atomic_data().atom_data
        lines = ["lattice:"]
        for pos, a in zip(self._cell, ("a", "b", "c"), strict=True):
            lines.append(
                "- [ %21.15f, %21.15f, %21.15f ] # %s" % (pos[0], pos[1], pos[2], a)
            )
        lines.append("points:")
        if self.magnetic_moments is None:
            magmoms = [None] * len(self)
        else:
            magmoms = self.magnetic_moments
        for i, (symbol, number, pos, mass, mag) in enumerate(
            zip(
                self.symbols,
                self.numbers,
                self.scaled_positions,
                self.masses,
                magmoms,
                strict=True,
            )
        ):
            formal_s = _atom_data[number][1]
            if symbol == formal_s:
                lines.append(f"- symbol: {symbol} # {i + 1}")
            else:
                lines.append(f"- symbol: {formal_s} # {i + 1}")
                lines.append(f"  extended_symbol: {symbol}")
            lines.append("  coordinates: [ %18.15f, %18.15f, %18.15f ]" % tuple(pos))
            if mass is not None:
                lines.append("  mass: %f" % mass)
            if mag is not None:
                if mag.ndim == 0:
                    mag_str = f"{mag:.8f}"
                else:
                    mag_str = f"[{mag[0]:.8f}, {mag[1]:.8f}, {mag[2]:.8f}]"
                lines.append(f"  magnetic_moment: {mag_str}")
        return lines

    def __str__(self) -> str:
        """Return text lines of crystal structure in yaml."""
        return "\n".join(self.get_yaml_lines())

    def _get_element_counts(self) -> dict[str, int]:
        """Return dict of element counts, with indices stripped from symbols."""
        counts: dict[str, int] = {}
        for symbol in self.symbols:
            base_symbol = symbol.rstrip("0123456789")
            counts[base_symbol] = counts.get(base_symbol, 0) + 1
        return counts

    def _build_formula(self, counts: dict[str, int], divisor: int = 1) -> str:
        """Build formula string from element counts and optional divisor.

        Parameters
        ----------
        counts : dict
            Dictionary mapping element symbols to their counts
        divisor : int, optional
            Number to divide counts by, defaults to 1

        Returns
        -------
        str
            Formula string with elements in alphabetical order

        """
        if not counts:
            return ""

        formula_parts = []
        for element in sorted(counts):
            count = counts[element] // divisor
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")

        return "".join(formula_parts)

    @property
    def formula(self) -> str:
        """Return chemical formula as a string.

        The formula is constructed by sorting elements alphabetically and
        appending numbers for elements that appear more than once.
        E.g., "Si2O4" for two Si and four O atoms.

        """
        counts = self._get_element_counts()
        return self._build_formula(counts)

    @property
    def reduced_formula(self) -> str:
        """Return reduced chemical formula as a string.

        The reduced formula divides all element counts by their GCD.
        E.g., "Fe4O8" becomes "Fe2O4".

        """
        counts = self._get_element_counts()
        if not counts:
            return ""

        # Find GCD of all counts
        numbers = list(counts.values())
        divisor = numbers[0]
        for n in numbers[1:]:
            divisor = gcd(divisor, n)

        return self._build_formula(counts, divisor)

    @property
    def normalized_formula(self) -> str:
        """Return normalized formula as a string.

        The normalized formula scales all element counts so they sum to 1.
        E.g., "Fe2O3" becomes "Fe0.4O0.6".

        """
        counts = self._get_element_counts()
        if not counts:
            return ""

        # Get total count
        total = sum(counts.values())

        # Build normalized formula string
        formula_parts = []
        for element in sorted(counts):
            count = counts[element] / total
            # Always show decimal for normalized formula
            formula_parts.append(f"{element}{count:.3}")

        return "".join(formula_parts)


def parse_cell_dict(cell_dict: CellDict) -> PhonopyAtoms | None:
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
        warnings.warn(
            'Cell dict key "atoms" is deprecated. Use "points" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
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


def __getattr__(name):
    if name in ("symbol_map", "atom_data", "isotope_data"):
        warnings.warn(
            "symbol_map, atom_data, and isotope_data are deprecated. "
            "Use phonopy.atomic_data.get_atomic_data() dataclass instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "atom_data":
            return get_atomic_data().atom_data
        if name == "isotope_data":
            return get_atomic_data().isotope_data
        if name == "symbol_map":
            return get_atomic_data().symbol_map
