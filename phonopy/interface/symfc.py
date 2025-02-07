"""Symfc force constants calculator interface."""

# Copyright (C) 2024 Atsushi Togo
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

import math
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry


def run_symfc(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    displacements: np.ndarray,
    forces: np.ndarray,
    orders: Optional[Sequence[int]] = None,
    is_compact_fc: bool = False,
    symmetry: Optional[Symmetry] = None,
    options: Optional[Union[str, dict]] = None,
    log_level: int = 0,
) -> dict[int, np.ndarray]:
    """Calculate force constants using symfc.

    The details of the parameters are found in the SymfcFCSolver class.

    """
    if orders is None:
        _orders = [2]
    else:
        _orders = orders

    options_dict = parse_symfc_options(options)

    symfc_calculator = SymfcFCSolver(
        supercell,
        displacements=displacements,
        forces=forces,
        symmetry=symmetry,
        orders=_orders,
        options=options_dict,
        is_compact_fc=is_compact_fc,
        log_level=log_level,
    )
    assert np.array_equal(symfc_calculator.p2s_map, primitive.p2s_map)

    return symfc_calculator.force_constants


class SymfcFCSolver:
    """Interface to symfc."""

    def __init__(
        self,
        supercell: PhonopyAtoms,
        displacements: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        symmetry: Optional[Symmetry] = None,
        orders: Optional[Sequence[int]] = None,
        options: Optional[dict] = None,
        is_compact_fc: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.
        displacements : np.ndarray, optional
            Displacements. Default is None.
        forces : np.ndarray, optional.
            Forces. Default is None.
        symmetry : Symmetry, optional
            Symmetry of supercell. Default is None.
        orders: Sequence[int], optional
            Orders of force constants. Default is None.
        options : dict, optional
            Options for symfc. Default is None, which gives {}.
        is_compact_fc : bool, optional
            Whether force constants are compact or full. When True, check if
            SymfcFCSolver.p2s_map is equal to primitive.p2s_map. Default is
            False.
        log_level : int, optional
            Log level. Default is 0.

        """
        if options is None:
            self._options = {}
        else:
            self._options = options
        self._supercell = supercell
        self._log_level = log_level
        self._orders = orders
        self._is_compact_fc = is_compact_fc
        self._symfc = None

        self._initialize(
            supercell,
            symmetry=symmetry,
            displacements=displacements,
            forces=forces,
        )
        if self._orders is not None:
            self.run(self._orders)

    @property
    def force_constants(self) -> dict[int, np.ndarray]:
        """Return force constants.

        Returns
        -------
        dict[int, np.ndarray]
            Force constants with order as key.

        """
        if self._orders is None:
            raise RuntimeError("Run SymfcCalculator.run() first.")
        return self._symfc.force_constants

    def run(self, orders: Sequence[int]):
        """Run symfc."""
        if self._log_level:
            print(
                "--------------------------------"
                " Symfc start "
                "-------------------------------"
            )
            print("Symfc is a force constants calculator. See the following paper:")
            print("A. Seko and A. Togo, Phys. Rev. B, 110, 214302 (2024).")
            print("Symfc is developed at https://github.com/symfc/symfc.")
            print(f"Computing {orders} order force constants.", flush=True)
            if self._options:
                print("Parameters:")
                for key, val in self._options.items():
                    print(f"  {key}: {val}", flush=True)

        self._orders = orders
        self._symfc.run(orders=orders, is_compact_fc=self._is_compact_fc)

        if self._log_level == 1:
            print("Increase log-level to watch detailed symfc log.")

        if self._log_level:
            print(
                "---------------------------------"
                " Symfc end "
                "--------------------------------"
            )

    @property
    def p2s_map(self) -> np.ndarray:
        """Return indices of translationally independent atoms."""
        return self._symfc.p2s_map

    @property
    def basis_set(self) -> dict:
        """Return basis set."""
        return self._symfc.basis_set

    def estimate_basis_size(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
    ) -> dict:
        """Estimate basis size."""
        basis_sizes = self._symfc.estimate_basis_size(
            orders=orders, max_order=max_order
        )
        return basis_sizes

    def estimate_numbers_of_supercells(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
    ) -> dict:
        """Estimate numbers of supercells."""
        basis_sizes = self.estimate_basis_size(max_order=max_order, orders=orders)
        n_scells = {}
        for order, basis_size in basis_sizes.items():
            n_scells[order] = math.ceil(basis_size / len(self._symfc.supercell) / 3)
        return n_scells

    def _initialize(
        self,
        supercell: PhonopyAtoms,
        symmetry: Optional[Symmetry] = None,
        displacements: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
    ):
        """Calculate force constants."""
        try:
            from symfc import Symfc
            from symfc.utils.utils import SymfcAtoms
        except ImportError as exc:
            raise ModuleNotFoundError("Symfc python module was not found.") from exc

        symfc_supercell = SymfcAtoms(
            cell=supercell.cell,
            scaled_positions=supercell.scaled_positions,
            numbers=supercell.numbers,
        )
        spacegroup_operations = symmetry.symmetry_operations if symmetry else None
        self._symfc = Symfc(
            symfc_supercell,
            spacegroup_operations=spacegroup_operations,
            cutoff=self._options.get("cutoff"),
            use_mkl=self._options.get("use_mkl", False),
            log_level=self._log_level - 1 if self._log_level else 0,
        )
        if displacements is not None and forces is not None:
            self._symfc.displacements = displacements
            self._symfc.forces = forces

    def compute_basis_set(
        self,
        max_order: Optional[int] = None,
        orders: Optional[list] = None,
    ) -> dict:
        """Run basis set calculations and return basis sets.

        Parameters
        ----------
        max_order : int
            Maximum fc order.
        orders: list
            Orders of force constants.

        Returns
        -------
        dict[FCBasisSetBase]
            Basis sets. Keys are orders.

        """
        self._symfc.compute_basis_set(max_order=max_order, orders=orders)
        return self._symfc.basis_set


def parse_symfc_options(options: Optional[Union[str, dict]]) -> dict:
    """Parse symfc options.

    Parameters
    ----------
    options : Union[str, dict]
        Options for symfc.

    Returns
    -------
    dict

    Note
    ----
    When str, it should be written as follows:

        "cutoff = 10.0"

    """
    if options is None:
        return {}
    if isinstance(options, dict):
        return options
    elif isinstance(options, str):
        options_dict = {}
        for option in options.split(","):
            key_val = [v.strip().lower() for v in option.split("=")]
            if len(key_val) != 2:
                break
            key, val = key_val
            if key == "cutoff":
                try:
                    options_dict[key] = {3: float(val)}
                except ValueError:
                    print("Warning: Cutoff value must be float.")
            if key == "use_mkl":
                if val.strip().lower() == "true":
                    options_dict[key] = True
        return options_dict
    else:
        raise TypeError(f"options must be str or dict, not {type(options)}.")
