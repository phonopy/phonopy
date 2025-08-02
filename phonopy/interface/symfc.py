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
import warnings
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
    full_fc_to_compact_fc,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry


class SymfcFCSolver:
    """Interface to symfc."""

    def __init__(
        self,
        supercell: PhonopyAtoms,
        displacements: NDArray | None = None,
        forces: NDArray | None = None,
        symmetry: Symmetry | None = None,
        orders: Sequence[int] | None = None,
        options: dict | None = None,
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
        self._symmetry = symmetry
        self._log_level = log_level
        self._orders = orders
        self._is_compact_fc = is_compact_fc

        import symfc

        self._symfc: symfc.Symfc

        self._initialize(
            displacements=displacements,
            forces=forces,
        )
        if self._orders is not None:
            self.run(self._orders)

    @property
    def supercell(self) -> PhonopyAtoms:
        """Return supercell."""
        return self._supercell

    @property
    def force_constants(self) -> dict[int, NDArray]:
        """Return force constants.

        Returns
        -------
        dict[int, np.ndarray]
            Force constants with order as key.

        """
        if self._orders is None or not self._symfc:
            raise RuntimeError("Run SymfcFCSolver.run() first.")
        return self._symfc.force_constants

    def run(self, orders: Sequence[int]):
        """Run symfc."""
        if self._log_level:
            print(
                "--------------------------------"
                " Symfc start "
                "-------------------------------"
            )
            self.show_credit()
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
    def version(self) -> str:
        """Return symfc version."""
        import symfc

        return symfc.__version__

    @property
    def p2s_map(self) -> NDArray | None:
        """Return indices of translationally independent atoms."""
        return self._symfc.p2s_map

    @property
    def basis_set(self) -> dict:
        """Return basis set."""
        return self._symfc.basis_set

    def show_credit(self):
        """Show credit."""
        print(f"Symfc version {self.version} (https://github.com/symfc/symfc)")
        print("Citation: A. Seko and A. Togo, Phys. Rev. B, 110, 214302 (2024)")

    def estimate_basis_size(
        self,
        max_order: int | None = None,
        orders: list | None = None,
    ) -> dict:
        """Estimate basis size."""
        basis_sizes = self._symfc.estimate_basis_size(
            orders=orders, max_order=max_order
        )
        return basis_sizes

    def estimate_numbers_of_supercells(
        self,
        max_order: int | None = None,
        orders: list | None = None,
    ) -> dict:
        """Estimate numbers of supercells."""
        basis_sizes = self.estimate_basis_size(max_order=max_order, orders=orders)
        n_scells = {}
        for order, basis_size in basis_sizes.items():
            n_scells[order] = math.ceil(basis_size / len(self._symfc.supercell) / 3)
        return n_scells

    def compute_basis_set(
        self,
        max_order: int | None = None,
        orders: list | None = None,
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

    def get_nonzero_atomic_indices_fc3(self) -> NDArray[np.bool] | None:
        """Get nonzero atomic indices for fc3.

        Returns
        -------
        np.ndarray | None
            3D array of shape (N, N, N) where N is the number of atoms in the
            supercell. Each element is True if the corresponding atomic
            indices are nonzero in the third-order force constants, otherwise
            False. If there is no cutoff for fc3, return None.

        """
        if 3 not in self.basis_set:
            raise ValueError("Run compute_basis_set(orders=[3]) first.")

        fc_cutoff = self.basis_set[3].fc_cutoff
        if fc_cutoff is None:
            return None

        fc3_nonzero_elems = fc_cutoff.nonzero_atomic_indices_fc3()
        N = len(self._supercell)
        assert len(fc3_nonzero_elems) == N**3
        return fc3_nonzero_elems.reshape((N, N, N))

    @property
    def options(self) -> dict | None:
        """Return options."""
        return self._options

    def _initialize(
        self,
        displacements: NDArray | None = None,
        forces: NDArray | None = None,
    ):
        """Calculate force constants."""
        try:
            from symfc import Symfc
            from symfc.utils.utils import SymfcAtoms
        except ImportError as exc:
            raise ModuleNotFoundError("Symfc python module was not found.") from exc

        symfc_supercell = SymfcAtoms(
            cell=self._supercell.cell,
            scaled_positions=self._supercell.scaled_positions,
            numbers=self._supercell.numbers,
        )
        spacegroup_operations = (
            self._symmetry.symmetry_operations if self._symmetry else None
        )

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


def parse_symfc_options(options: Optional[Union[str, dict]], order: int) -> dict:
    """Parse symfc options.

    Parameters
    ----------
    options : Union[str, dict]
        Options for symfc.
    order : int
        Order of force constants.

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
                    options_dict[key] = {order: float(val)}
                except ValueError:
                    print("Warning: Cutoff value must be float.")
            if key == "memsize":
                try:
                    options_dict[key] = {order: float(val)}
                except ValueError:
                    print("Warning: Memsize value must be float.")
            if key == "use_mkl":
                if val.strip().lower() == "true":
                    options_dict[key] = True
        return options_dict
    else:
        raise TypeError(f"options must be str or dict, not {type(options)}.")


def estimate_symfc_memory_usage(
    supercell: PhonopyAtoms,
    symmetry: Symmetry,
    order: int,
    cutoff: float | None = None,
    batch_size: int = 100,
):
    """Estimate memory usage to run symfc for fc with cutoff.

    Total memory usage is memsize + memsize2. These are separated because
    they behave differently with respect to cutoff distance.

    batch_size is hardcoded to 100 because it is so in symfc.

    """
    if order not in [2, 3]:
        raise ValueError("order must be 2 or 3.")

    symfc_solver = SymfcFCSolver(
        supercell, symmetry=symmetry, options={"cutoff": {order: cutoff}}
    )
    basis_size = symfc_solver.estimate_basis_size(orders=[order])[order]
    memsize = basis_size**2 * 3 * 8 / 10**9
    memsize2 = len(supercell) * 3 * batch_size * basis_size * 8 / 10**9
    return memsize, memsize2


def estimate_symfc_cutoff_from_memsize(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    symmetry: Symmetry,
    order: int,
    max_memsize: Optional[float] = None,
    verbose: bool = False,
) -> Optional[float]:
    """Estimate cutoff from max memory size."""
    vecs, _ = primitive.get_smallest_vectors()
    dists = (
        np.unique(np.round(np.linalg.norm(vecs @ primitive.cell, axis=-1), decimals=1))
        + 0.1
    )
    for i, cutoff in enumerate(dists[1:]):
        memsize, memsize2 = estimate_symfc_memory_usage(
            supercell,
            symmetry,
            order,
            cutoff=float(cutoff),
        )
        if max_memsize and memsize + memsize2 > max_memsize:
            return float(dists[i])

        if verbose:
            print(
                f"{cutoff:5.1f}  {memsize + memsize2:6.2f} GB "
                f"({memsize:.2f}+{memsize2:.2f})",
                flush=True,
            )

    return None


def update_symfc_cutoff_by_memsize(
    options: dict,
    supercell: PhonopyAtoms,
    primitive: Primitive,
    symmetry: Symmetry,
    verbose: bool = False,
):
    """Update cutoff in options following max memsize.

    Note
    ----
    This function modifies the options dictionary in place.

    """
    cutoff = options.get("cutoff")
    if "memsize" not in options:
        return

    if cutoff is None:
        cutoff = {}
    for key, val in options["memsize"].items():
        if verbose:
            print(f"Estimate symfc cutoff for fc{key} by memsize of {val} GB")
        if key in (2, 3, 4):
            _cutoff = estimate_symfc_cutoff_from_memsize(
                supercell,
                primitive,
                symmetry,
                key,
                max_memsize=val,
                verbose=verbose,
            )
            if _cutoff is None:
                if verbose:
                    print(
                        "Specified memsize covers "
                        "all supercell force constants elements."
                    )
            else:
                cutoff.update({key: _cutoff})
        else:
            raise ValueError("order must be 2, 3, or 4.")
    if not cutoff:
        cutoff = None
    options["cutoff"] = cutoff
    del options["memsize"]


def symmetrize_by_projector(
    supercell: PhonopyAtoms,
    fc: NDArray,
    order: int,
    primitive: Primitive | None = None,
    options: dict | None = None,
    log_level: int = 0,
    show_credit: bool = False,
) -> NDArray:
    """Symmetrize force constants by projector method.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell.
    fc : np.ndarray
        Force constants to be symmetrized.
    order : int
        Order of force constants.
    primitive : Primitive, optional
        Primitive cell. If provided, it is used to check if the force constants
        are consistent with the primitive cell.
    options : dict, optional
        Options for symfc.
    log_level : int, optional
        Log level for symfc. Default is 0, which means no log.
    show_credit : bool, optional
        Whether to show credit information of symfc. Default is False.

    """
    symfc = SymfcFCSolver(supercell, options=options, log_level=log_level)
    if show_credit and log_level:
        symfc.show_credit()
    symfc.compute_basis_set(orders=[order])
    basis_set = symfc.basis_set[order]
    if fc.shape[0] == fc.shape[1]:
        compmat = basis_set.compression_matrix.tocsc()
    else:
        if primitive is None:
            raise ValueError("Primitive cell must be provided for compact fc.")
        assert symfc.p2s_map is not None
        if (
            len(primitive.p2s_map) != len(symfc.p2s_map)
            or (primitive.p2s_map != symfc.p2s_map).any()
        ):
            warnings.warn(
                "p2s_map of primitive cell does not match with p2s_map of symfc.",
                UserWarning,
                stacklevel=2,
            )
            return _convert_compact_fc(
                primitive,
                fc,
                supercell,
                order,
                log_level=log_level,
            )
        compmat = basis_set.compact_compression_matrix.tocsc()

    fc_sym = fc.ravel() @ compmat
    # fc_sym = fc_sym @ basis_set.blocked_basis_set
    # fc_sym = fc_sym @ basis_set.basis_set.T
    fc_sym = basis_set.blocked_basis_set.transpose_dot(fc_sym.T)
    fc_sym = basis_set.blocked_basis_set.dot(fc_sym.T)
    fc_sym = fc_sym.T @ compmat.T
    if fc.shape[0] != fc.shape[1]:
        n_lp = len(basis_set.translation_permutations)
        fc_sym *= n_lp
    return fc_sym.reshape(fc.shape)


def _convert_compact_fc(
    primitive: Primitive,
    compact_fc: NDArray,
    supercell: PhonopyAtoms,
    order: int,
    log_level: int = 0,
) -> NDArray:
    full_fc = compact_fc_to_full_fc(
        primitive,
        compact_fc,
        log_level=log_level,
    )
    full_fc = symmetrize_by_projector(
        supercell=supercell,
        fc=full_fc,
        order=order,
        primitive=primitive,
        log_level=log_level,
    )
    return full_fc_to_compact_fc(
        primitive,
        full_fc,
        log_level=log_level,
    )
