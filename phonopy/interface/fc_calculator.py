"""Force constants calculator interfaces."""

# Copyright (C) 2019 Atsushi Togo
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

from collections.abc import Sequence
from typing import Literal, Optional, cast

import numpy as np

from phonopy.exception import (
    ForceCalculatorRequiredError,
    ForceConstantsCalculatorNotFoundError,
)
from phonopy.harmonic.force_constants import FDFCSolver
from phonopy.interface.alm import ALMFCSolver
from phonopy.interface.symfc import SymfcFCSolver, parse_symfc_options
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.dataset import get_displacements_and_forces
from phonopy.structure.symmetry import Symmetry

fc_calculator_names = {
    "alm": "ALM",
    "symfc": "symfc",
    "traditional": "phonopy-traditional",
}


def check_and_cast_fc_calculator_name(
    fc_calculator: str | None,
) -> Literal["traditional", "symfc", "alm"] | None:
    """Check if the force constants calculator name is valid."""
    if fc_calculator is None:
        return None
    if fc_calculator not in fc_calculator_names:
        raise ForceConstantsCalculatorNotFoundError(
            f"{fc_calculator} is not a valid force constants calculator."
        )
    else:
        return cast(Literal["traditional", "symfc", "alm"], fc_calculator)


def get_fc_solver(
    supercell: PhonopyAtoms,
    dataset: dict,
    primitive: Optional[Primitive] = None,
    fc_calculator: Optional[Literal["traditional", "symfc", "alm"]] = None,
    fc_calculator_options: Optional[str] = None,
    orders: Optional[Sequence[int]] = None,
    is_compact_fc: bool = False,
    symmetry: Optional[Symmetry] = None,
    log_level: int = 0,
) -> FCSolver:
    """Return force constants solver class instance.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell
    dataset : dict
        Dataset that contains displacements, forces, and optionally energies.
    primitive : Primitive
        Primitive cell. Only needed for the traditional FC calculator.
    fc_calculator : str, optional
        Currently 'traditional' (FD method), 'alm', and 'symfc' are supported.
        Default is None, meaning invoking 'traditional'.
    fc_calculator_options : str, optional
        This is arbitrary string.
    orders : Sequence[int], optional
        Orders of force constants to be calculated. Default is None.
    is_compact_fc : bool, optional
        If True, force constants are returned in the compact form.
    symmetry : Symmetry, optional
        Symmetry of supercell. This is used for the traditional and symfc FC
        solver. Default is None.
    log_level : integer or bool, optional
        Verbosity level. False or 0 means quiet. True or 1 means normal level of
        log to stdout. 2 gives verbose mode.

    Returns
    -------
    FCSolver
        Force constants solver class instance.

    """
    fc_solver_name = fc_calculator if fc_calculator is not None else "traditional"
    fc_solver = FCSolver(
        fc_solver_name,
        supercell,
        symmetry=symmetry,
        dataset=dataset,
        is_compact_fc=is_compact_fc,
        primitive=primitive,
        orders=orders,
        options=fc_calculator_options,
        log_level=log_level,
    )
    return fc_solver


def get_fc2(
    supercell: PhonopyAtoms,
    dataset: dict,
    primitive: Optional[Primitive] = None,
    fc_calculator: Optional[Literal["traditional", "symfc", "alm"]] = None,
    fc_calculator_options: Optional[str] = None,
    is_compact_fc: bool = False,
    symmetry: Optional[Symmetry] = None,
    log_level: int = 0,
):
    """Supercell 2nd order force constants (fc2) are calculated.

    The expected shape of supercell fc2 to be returned is
        (len(atom_list), num_atoms, 3, 3).

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell
    dataset : dict
        Dataset that contains displacements, forces, and optionally energies.
    primitive : Primitive
        Primitive cell. Only needed for the traditional FC calculator.
    fc_calculator : str, optional
        Currently 'traditional' (FD method), 'alm', and 'symfc' are supported.
        Default is None, meaning invoking 'traditional'.
    fc_calculator_options : str, optional
        This is arbitrary string.
    is_compact_fc : bool, optional
        If True, force constants are returned in the compact form.
    symmetry : Symmetry, optional
        Symmetry of supercell. This is used for the traditional and symfc FC
        solver. Default is None.
    log_level : integer or bool, optional
        Verbosity level. False or 0 means quiet. True or 1 means normal level of
        log to stdout. 2 gives verbose mode.

    Returns
    -------
    fc2 : ndarray
        2nd order force constants. shape=(len(atom_list), num_atoms, 3, 3),
        dtype='double', order='C'.

    """
    fc_solver = get_fc_solver(
        supercell,
        dataset,
        primitive=primitive,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
        orders=[2],
        is_compact_fc=is_compact_fc,
        symmetry=symmetry,
        log_level=log_level,
    )
    return fc_solver.force_constants[2]


class FCSolver:
    """Force constants calculator interface."""

    def __init__(
        self,
        fc_solver_name: Literal["traditional", "symfc", "alm"],
        supercell: PhonopyAtoms,
        symmetry: Symmetry | None = None,
        dataset: dict | None = None,
        is_compact_fc: bool = False,
        primitive: Primitive | None = None,
        orders: Sequence[int] | None = None,
        options: str | None = None,
        log_level: int = 0,
    ):
        """Init method.

        Force constants are calculated if necessary data are provided.

        For the symfc FC solver, SymfcFCSolver instance can be returned without
        necessary data for computing force constants. For the other solvers,
        raise an error.

        Parameters
        ----------
        fc_solver_name : Literal["traditional", "symfc", "alm"]
            Force constants calculator name.
        supercell : PhonopyAtoms
            Supercell.
        symmetry : Symmetry, optional
            Symmetry of supercell. Default is None.
        dataset : dict, optional
            Dataset that contains displacements, forces, and optionally
            energies. Default is None.
        is_compact_fc : bool, optional
            If True, force constants are returned in the compact form.
        primitive : Primitive, optional
            Primitive cell. This is needed for the traditional and alm FC
            calculator. Default is None.
        orders : Sequence[int], optional
            Orders of force constants to be calculated. Default is None.
        options : str, optional
            This is arbitrary string that is used for each FC solver. Default is
            None.
        log_level : int, optional
            Log level. Default is 0.

        """
        self._supercell = supercell
        self._symmetry = symmetry
        self._dataset = dataset
        self._is_compact_fc = is_compact_fc
        self._primitive = primitive
        self._orders = orders
        self._options = options
        self._log_level = log_level

        self._fc_solver = self._set_fc_solver(fc_solver_name)

    @property
    def fc_solver(self) -> FDFCSolver | SymfcFCSolver | ALMFCSolver:
        """Return force constants solver class instance."""
        return self._fc_solver

    @property
    def force_constants(self) -> dict[int, np.ndarray]:
        """Return force constants."""
        return self._fc_solver.force_constants

    def _set_fc_solver(
        self, fc_calculator_name: str
    ) -> FDFCSolver | SymfcFCSolver | ALMFCSolver:
        if fc_calculator_name == "traditional":
            return self._set_traditional_solver()
        if fc_calculator_name == "symfc":
            return self._set_symfc_solver()
        if fc_calculator_name == "alm":
            return self._set_alm_solver()
        raise ValueError(f"Unknown fc_calculator_name: {fc_calculator_name}")

    def _set_traditional_solver(
        self, solver_class: Optional[type] = FDFCSolver
    ) -> FDFCSolver:
        if self._primitive is None:
            raise RuntimeError(
                "Primitive cell is required for the traditional FC solver."
            )

        if self._dataset is None:
            raise RuntimeError(
                "Displacement-force dataset is required for the traditional FC solver."
            )

        if "displacements" in self._dataset:
            lines = [
                "Type-II dataset for displacements and forces was provided, ",
                "but the selected force constants calculator cannot process it.",
                "Use another force constants calculator, e.g., symfc, ",
                "to generate force constants.",
            ]
            raise ForceCalculatorRequiredError("\n".join(lines))

        return solver_class(
            self._supercell,
            self._primitive,
            self._symmetry,
            self._dataset,
            is_compact_fc=self._is_compact_fc,
            log_level=self._log_level,
        )

    def _set_symfc_solver(self, order: int = 2) -> SymfcFCSolver:
        options = parse_symfc_options(self._options, order)
        options.pop("memsize", None)
        if self._dataset is None:
            return SymfcFCSolver(
                self._supercell,
                symmetry=self._symmetry,
                options=options,
                is_compact_fc=self._is_compact_fc,
                log_level=self._log_level,
            )
        else:
            displacements, forces = self._get_displacements_and_forces()
            symfc_solver = SymfcFCSolver(
                self._supercell,
                displacements=displacements,
                forces=forces,
                symmetry=self._symmetry,
                options=options,
                is_compact_fc=self._is_compact_fc,
                log_level=self._log_level,
            )
            if self._orders is not None:
                symfc_solver.run(orders=self._orders)
            return symfc_solver

    def _set_alm_solver(self) -> ALMFCSolver:
        if self._primitive is None:
            raise RuntimeError(
                "Primitive cell is required for the traditional FC solver."
            )

        if self._dataset is None:
            raise RuntimeError(
                "Displacement-force dataset is required for ALM FC solver."
            )
        displacements, forces = self._get_displacements_and_forces()

        return ALMFCSolver(
            self._supercell,
            self._primitive,
            displacements,
            forces,
            max(self._orders) - 1,
            is_compact_fc=self._is_compact_fc,
            options=self._options,
            log_level=self._log_level,
        )

    def _get_displacements_and_forces(self):
        """Return displacements and forces for fc2."""
        return get_displacements_and_forces(self._dataset)
