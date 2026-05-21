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
    mixture: list[list]  # [[symbol_str, weight_float], ...] for a mixed-species site


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

    Two atoms share a species iff all fields are equal.

    For a normal (single-element) species, ``atomic_number`` is set and
    ``mixture`` is None. ``symbol`` carries the suffix ("Cl1") so that
    "Cl" and "Cl1" are distinct species that share the same atomic number.

    For a mixed-species site (e.g. a VCA virtual crystal site, an alloy
    site, a solid-solution site), ``atomic_number`` is None and ``mixture``
    holds the constituent (symbol, weight) pairs with weights summing to
    1.0. ``symbol`` is the composite label (e.g. "GeSn", or "GeSn1"/"GeSn2"
    if the cell has multiple distinct GeSn mixtures).

    """

    symbol: str
    atomic_number: int | None
    mixture: tuple[tuple[str, float], ...] | None = None

    def __post_init__(self) -> None:
        if (self.atomic_number is None) == (self.mixture is None):
            raise ValueError("Exactly one of atomic_number or mixture must be set.")


def build_species_table_from_mixtures(
    mixtures: Sequence[Sequence[tuple[str, float]]],
    sort_constituents: bool = True,
) -> tuple[list[_Species], NDArray[np.int64]]:
    """Convert per-atom species mixtures into a (species_table, species_ids) pair.

    Each entry of ``mixtures`` is a list of ``(symbol, weight)`` pairs
    describing the constituents of one atomic site. Weights must sum to 1.0.
    A single-component entry of weight 1.0 is canonicalized to a normal
    (single-element) species. The composite symbol of a true mixture is
    derived as the concatenation of constituent symbols (e.g. "GeSn"); when
    several distinct mixtures within the same cell would collide on the same
    composite label, every colliding mixed species receives a 1-based suffix
    (``"GeSn1"``, ``"GeSn2"``, ...) in table order while unique composites
    stay unsuffixed.

    Parameters
    ----------
    mixtures :
        Per-site lists of ``(symbol, weight)`` pairs.
    sort_constituents :
        If True (default), each per-site mixture is reordered alphabetically
        by symbol before deriving species identity and the composite label.
        This treats two physically equivalent sites that differ only in the
        order constituents were listed (e.g. ``[("Ge", 0.5), ("Sn", 0.5)]``
        and ``[("Sn", 0.5), ("Ge", 0.5)]``) as the same species. Pass False
        to preserve the caller's order verbatim.

    The returned pair can be passed to ``PhonopyAtoms(species_table=...,
    species_ids=...)``. Use cases include the Virtual Crystal Approximation
    (see ``apply_site_mixture``) and any other site-disorder model
    expressible as weighted constituent symbols.

    """
    symbol_map = get_atomic_data().symbol_map
    seen: dict[_Species, int] = {}
    species_list: list[_Species] = []
    ids: list[int] = []
    for atom_mixture in mixtures:
        mix = tuple((str(s), float(w)) for s, w in atom_mixture)
        if not mix:
            raise ValueError("mixture entry must be non-empty.")
        if sort_constituents:
            mix = tuple(sorted(mix, key=lambda sw: sw[0]))
        wsum = sum(w for _, w in mix)
        if not np.isclose(wsum, 1.0):
            raise ValueError(f"mixture weights must sum to 1.0, got {wsum} for {mix}.")
        # Each component symbol may carry a suffix like "Cl1"; validate the
        # base part exists in the periodic table.
        for sym, _ in mix:
            base, _ = split_symbol_and_index(sym)
            if base not in symbol_map:
                raise RuntimeError(f"Invalid symbol: {sym}.")
        if len(mix) == 1:
            sym, _ = mix[0]
            base, _ = split_symbol_and_index(sym)
            sp = _Species(symbol=sym, atomic_number=symbol_map[base])
        else:
            composite = "".join(s for s, _ in mix)
            sp = _Species(symbol=composite, atomic_number=None, mixture=mix)
        sid = seen.get(sp)
        if sid is None:
            sid = len(species_list)
            species_list.append(sp)
            seen[sp] = sid
        ids.append(sid)
    species_list = _disambiguate_composite_labels(species_list)
    return species_list, np.array(ids, dtype="int64")


def _disambiguate_composite_labels(
    species_table: list[_Species],
) -> list[_Species]:
    """Append 1-based suffixes to mixed-species symbols that collide.

    Distinct mixtures may produce the same composite label
    (e.g. two GeSn mixtures with different ratios both yield ``"GeSn"``).
    When that happens within a single species table, every colliding mixed
    species is renamed to ``"GeSn1"`` / ``"GeSn2"`` / ... in table order,
    while unique composites stay unsuffixed. Non-mixture species are
    untouched.

    """
    label_counts: dict[str, int] = {}
    for sp in species_table:
        if sp.mixture is not None:
            label_counts[sp.symbol] = label_counts.get(sp.symbol, 0) + 1
    duplicates = {label for label, count in label_counts.items() if count > 1}
    if not duplicates:
        return species_table

    counters: dict[str, int] = {}
    new_table: list[_Species] = []
    for sp in species_table:
        if sp.mixture is not None and sp.symbol in duplicates:
            counters[sp.symbol] = counters.get(sp.symbol, 0) + 1
            new_label = f"{sp.symbol}{counters[sp.symbol]}"
            new_table.append(dataclasses.replace(sp, symbol=new_label))
        else:
            new_table.append(sp)
    return new_table


class PhonopyAtoms:
    """Class to represent crystal structure.

    Originally this class aimed to be compatible with the ASE ``Atoms``
    class, but the APIs have since diverged.

    A cell is described by basis vectors (``cell``), per-atom positions
    (``positions`` / ``scaled_positions``), and per-atom species
    information. Species can be expressed either as plain chemical
    symbols (``symbols``) / atomic numbers (``numbers``) for ordinary
    cells, or as a deduplicated table plus per-atom indices
    (``species_table`` / ``species_ids``) which additionally supports
    mixed-species sites for the Virtual Crystal Approximation. See
    :ref:`phonopy_Atoms` for the tutorial-style overview and per-attribute
    documentation below for details.

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
        species_table: Sequence[_Species] | None = None,
        species_ids: Sequence[int] | NDArray[np.int64] | None = None,
    ) -> None:  # pbc is dummy argument, and never used.
        """Set crystal structure parameters.

        Exactly one of the species inputs (``symbols``, ``numbers``, or
        the ``species_table`` / ``species_ids`` pair) must be supplied.
        Exactly one of the position inputs (``positions`` or
        ``scaled_positions``) must be supplied.

        Parameters
        ----------
        symbols : sequence of str, optional
            Per-atom chemical symbols (``"Si"``, ``"Cl1"``, ...).
            Shorthand for non-VCA cells. Atomic numbers and default
            masses are derived from these.
        numbers : sequence of int, optional
            Per-atom atomic numbers in 1..118. Shorthand alternative to
            ``symbols``.
        species_table : sequence of _Species, optional
            Canonical deduplicated species table. Used together with
            ``species_ids``. Required for cells containing mixed-species
            sites (see
            :func:`build_species_table_from_mixtures`) and by
            cell-manipulation helpers (supercell, primitive, trimmed
            cell) that already know the species identity.
        species_ids : sequence of int, optional
            Per-atom index into ``species_table``. Used together with
            ``species_table``.
        masses : sequence of float, optional
            Per-atom atomic masses. If omitted, masses are derived from
            the species.
        magnetic_moments : array_like, optional
            Per-atom magnetic moments. ``shape=(natom,)`` for scalar
            moments or ``shape=(natom, 3)`` for vector moments.
        cell : array_like
            Basis vectors as row vectors. ``shape=(3, 3)``.
        positions : array_like, optional
            Per-atom positions in Cartesian coordinates.
            ``shape=(natom, 3)``. Mutually exclusive with
            ``scaled_positions``.
        scaled_positions : array_like, optional
            Per-atom positions in fractional (crystallographic)
            coordinates. ``shape=(natom, 3)``. Mutually exclusive with
            ``positions``.

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

        if (species_table is None) != (species_ids is None):
            raise ValueError(
                "species_table and species_ids must be specified together."
            )
        n_specified = sum(x is not None for x in (symbols, numbers, species_table))
        if n_specified == 0:
            raise RuntimeError(
                "One of symbols, numbers, or (species_table, species_ids) "
                "has to be specified."
            )
        if n_specified > 1:
            raise ValueError(
                "symbols, numbers, and species_table are mutually exclusive."
            )

        if symbols is not None:
            self._build_species_from_symbols(list(symbols))
        elif numbers is not None:
            self._build_species_from_numbers(np.asarray(numbers, dtype="int64"))
        else:
            assert species_table is not None and species_ids is not None
            self._species = list(species_table)
            self._species_ids = np.asarray(species_ids, dtype="int64")

        if masses is not None:
            self._set_masses(masses)
        else:
            self._set_default_masses()

        self._set_magnetic_moments(magnetic_moments)

        self._check()

    def __len__(self) -> int:
        """Return the number of atoms in the cell."""
        return len(self._species_ids)

    @property
    def cell(self) -> NDArray[np.double]:
        """Setter and getter of basis vectors.

        Basis vectors (a, b, c) given as row vectors.
        ``shape=(3, 3)``, ``dtype='double'``, ``order='C'``.
        For getter, a copy is returned.

        """
        return self._cell.copy()

    @cell.setter
    def cell(self, cell: Sequence[Sequence[float]] | NDArray[np.double]) -> None:
        self._set_cell(cell)
        self._check()

    @property
    def positions(self) -> NDArray[np.double]:
        """Setter and getter of positions in Cartesian coordinates.

        ``shape=(natom, 3)``, ``dtype='double'``, ``order='C'``.

        """
        return np.array(
            np.dot(self._scaled_positions, self._cell), dtype="double", order="C"
        )

    @positions.setter
    def positions(self, positions: Sequence[Sequence[float]] | NDArray[np.double]):
        self._set_positions(positions)
        self._check()

    @property
    def scaled_positions(self) -> NDArray[np.double]:
        """Setter and getter of scaled positions.

        Positions of atoms in fractional (crystallographic) coordinates.
        ``shape=(natom, 3)``, ``dtype='double'``, ``order='C'``.
        For getter, a copy is returned.

        """
        return self._scaled_positions.copy()

    @scaled_positions.setter
    def scaled_positions(
        self, scaled_positions: Sequence[Sequence[float]] | NDArray[np.double]
    ):
        self._set_scaled_positions(scaled_positions)
        self._check()

    @property
    def symbols(self) -> list[str]:
        """Chemical symbols per atom.

        Chemical symbol with an appended natural number is allowed, e.g.,
        ``"Cl1"``. Mixed-species sites carry a composite label formed by
        concatenating constituent symbols (e.g., ``"GeSn"``).

        """
        return [self._species[sid].symbol for sid in self._species_ids]

    @property
    def species_ids(self) -> NDArray[np.int64]:
        """Per-atom index into ``species_table``.

        Atoms that reference the same ``species_table`` entry share the
        same id; ordinary ``"Cl"`` and labeled ``"Cl1"`` get different
        ids, and two atoms differing only in mixture composition or
        weights also get different ids. Suitable as the ``types``
        argument for spglib when atoms with the same atomic number but
        different symbol suffix must be distinguished.
        ``shape=(natom,)``, ``dtype='int64'``.
        For getter, a copy is returned.

        """
        return self._species_ids.copy()

    @property
    def species_table(self) -> list[_Species]:
        """Deduplicated species table indexed by ``species_ids``.

        Each entry is either an ordinary single-element species (with
        ``atomic_number`` set) or a weighted mixture (``mixture`` set to
        a tuple of ``(symbol, weight)`` pairs summing to 1.0). See
        :func:`phonopy.structure.atoms.build_species_table_from_mixtures`.
        For getter, a shallow list copy is returned; ``_Species`` entries
        are frozen dataclasses so the list elements are themselves
        immutable.

        """
        return list(self._species)

    @property
    def numbers(self) -> NDArray[np.int64]:
        """Atomic numbers per atom in 1..118.

        ``shape=(natom,)``, ``dtype='int64'``.
        For getter, a new array is returned.

        Raises ``RuntimeError`` if the cell contains any mixed-species
        site, since a mixture has no single atomic number. Use
        ``species_ids`` instead when an opaque per-atom species
        discriminator is needed.

        """
        nums = []
        for sid in self._species_ids:
            sp = self._species[sid]
            if sp.atomic_number is None:
                raise RuntimeError(
                    "cell.numbers is undefined for cells containing mixed-species "
                    f"sites (species '{sp.symbol}'). Use cell.species_ids instead."
                )
            nums.append(sp.atomic_number)
        return np.array(nums, dtype="int64")

    @property
    def has_mixtures(self) -> bool:
        """Return True if the cell contains any mixed-species site.

        ``True`` when any species in the cell is a weighted mixture of
        constituents (e.g., a Virtual Crystal Approximation site).

        """
        return any(sp.mixture is not None for sp in self._species)

    @property
    def masses(self) -> NDArray[np.double]:
        """Setter and getter of atomic masses.

        For a mixed-species site, the mass is the weight-averaged sum
        over its constituents.
        ``shape=(natom,)``, ``dtype='double'``.
        For getter, a copy is returned.

        """
        return self._masses.copy()

    @masses.setter
    def masses(self, masses: Sequence[float] | NDArray[np.double]):
        self._set_masses(masses)
        self._check()

    @property
    def magnetic_moments(self) -> NDArray[np.double] | None:
        """Setter and getter of magnetic moments.

        ``shape=(natom,)`` or ``(natom, 3)``, ``dtype='double'``,
        ``order='C'``.

        For the setter, the former can also be specified as
        ``(natom, 1)`` (recognized as ``(natom,)``) and the latter as
        ``(natom * 3,)`` (converted to ``(natom, 3)``).
        For getter, a copy is returned.

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
        """Return the cell volume.

        Computed as the determinant of the basis-vector matrix. The unit
        follows that of the input ``cell`` (e.g., Angstrom^3 for an
        Angstrom-valued ``cell``).

        """
        return float(np.linalg.det(self._cell))

    @property
    def Z(self) -> int:
        """Return the number of formula units in this cell.

        Computed as the GCD of the per-species atom counts. For example,
        a cell with 4 Fe and 8 O atoms returns 4.

        """
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
        """Set _masses from the periodic table using each atom's species.

        For a mixed-species site the mass is the weight-averaged sum of
        constituent atomic masses (sum_j w_j * m_j).

        """
        atom_data = get_atomic_data().atom_data
        symbol_map = get_atomic_data().symbol_map
        masses: list[float] = []
        undefined: set[str] = set()
        for sid in self._species_ids:
            sp = self._species[sid]
            if sp.mixture is not None:
                m_total = 0.0
                bad = False
                for sym, w in sp.mixture:
                    base, _ = split_symbol_and_index(sym)
                    cm = atom_data[symbol_map[base]][3]
                    if cm is None:
                        undefined.add(sym)
                        bad = True
                    else:
                        m_total += w * cm
                if not bad:
                    masses.append(m_total)
            else:
                assert sp.atomic_number is not None
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
        """Return an independent copy of this cell.

        Internal arrays (cell, scaled positions, masses, magnetic
        moments, species ids) are re-allocated by the constructor so
        that mutating one cell does not affect the other. The species
        table entries themselves are frozen dataclasses and are safe to
        share.

        """
        return PhonopyAtoms(
            cell=self._cell,
            scaled_positions=self._scaled_positions,
            masses=self._masses,
            magnetic_moments=self._magnetic_moments,
            species_table=self._species,
            species_ids=self._species_ids,
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
        """Return ``(cell, scaled_positions, numbers)``.

        If ``magnetic_moments`` is set,
        ``(cell, scaled_positions, numbers, magnetic_moments)`` is
        returned instead.

        Parameters
        ----------
        distinguish_symbol_index : bool, optional
            If True, the per-atom integer is the species id (atoms with
            the same symbol but different suffix get different ids);
            suitable as the ``types`` argument for spglib when
            symbol-suffix groupings must be preserved. If False
            (default), the per-atom integer is the atomic number. VCA
            cells always use ``species_ids`` regardless, since a VCA
            mixture has no single atomic number.

        Returns
        -------
        tuple
            ``(cell, scaled_positions, numbers)`` or
            ``(cell, scaled_positions, numbers, magnetic_moments)``.

        """
        if distinguish_symbol_index or self.has_mixtures:
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
        """Return the crystal structure as a list of yaml text lines."""
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
        for i, (sid, pos, mass, mag) in enumerate(
            zip(
                self._species_ids,
                self.scaled_positions,
                self.masses,
                magmoms,
                strict=True,
            )
        ):
            sp = self._species[sid]
            if sp.mixture is not None:
                lines.append(f"- symbol: {sp.symbol} # {i + 1}")
                mix_str = ", ".join(f"[{s}, {w}]" for s, w in sp.mixture)
                lines.append(f"  mixture: [ {mix_str} ]")
            else:
                assert sp.atomic_number is not None
                formal_s = _atom_data[sp.atomic_number][1]
                if sp.symbol == formal_s:
                    lines.append(f"- symbol: {sp.symbol} # {i + 1}")
                else:
                    lines.append(f"- symbol: {formal_s} # {i + 1}")
                    lines.append(f"  extended_symbol: {sp.symbol}")
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
        """Return the crystal structure as yaml text."""
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
    points: list = []
    symbols: list[str] = []
    masses: list[float] = []
    magnetic_moments: list = []
    mixture_per_atom: list[list[tuple[str, float]] | None] = []
    has_any_mixture = False
    if "points" in cell_dict:
        for x in cell_dict["points"]:
            if "coordinates" in x:
                points.append(x["coordinates"])
            if "extended_symbol" in x:  # like Fe1
                symbols.append(x["extended_symbol"])
            elif "symbol" in x:  # like Fe
                symbols.append(x["symbol"])
            if "mixture" in x:
                mix = [(str(s), float(w)) for s, w in x["mixture"]]
                mixture_per_atom.append(mix)
                has_any_mixture = True
            else:
                mixture_per_atom.append(None)
            if "mass" in x:
                masses.append(x["mass"])
            if "magnetic_moment" in x:
                magnetic_moments.append(x["magnetic_moment"])

    masses_arg: list[float] | None = masses if masses else None
    magmoms_arg: list | None = magnetic_moments if magnetic_moments else None

    if not (points and symbols):
        return None

    if has_any_mixture:
        # Cells with any mixed-species site route through
        # build_species_table_from_mixtures; non-mixture atoms become
        # single-component entries of weight 1.0 (canonicalized back to a
        # normal species inside the factory). Composite symbols of true
        # mixtures are re-derived, so suffix-numbered labels (e.g.
        # "GeSn1"/"GeSn2") from the YAML are not preserved -- the species
        # table still distinguishes them by mixture content.
        mixtures: list[list[tuple[str, float]]] = []
        for m, s in zip(mixture_per_atom, symbols, strict=True):
            mixtures.append(m if m is not None else [(s, 1.0)])
        species_table, species_ids = build_species_table_from_mixtures(mixtures)
        return PhonopyAtoms(
            cell=lattice,
            scaled_positions=points,
            masses=masses_arg,
            magnetic_moments=magmoms_arg,
            species_table=species_table,
            species_ids=species_ids,
        )
    return PhonopyAtoms(
        symbols=symbols,
        cell=lattice,
        masses=masses_arg,
        scaled_positions=points,
        magnetic_moments=magmoms_arg,
    )


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
