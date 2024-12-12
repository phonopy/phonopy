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
from typing import Literal, Optional, Union

import numpy as np

from phonopy.harmonic.force_constants import FDFCSolver
from phonopy.interface.alm import ALMFCSolver
from phonopy.interface.symfc import SymfcFCSolver
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.dataset import get_displacements_and_forces
from phonopy.structure.symmetry import Symmetry

fc_calculator_names = {
    "alm": "ALM",
    "symfc": "symfc",
    "traditional": "phonopy-traditional",
}


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
    fc_solver_name = fc_calculator if fc_calculator is not None else "traditional"
    fc_solver = FCSolver(
        fc_solver_name,
        supercell,
        symmetry=symmetry,
        dataset=dataset,
        is_compact_fc=is_compact_fc,
        primitive=primitive,
        orders=[2],
        options=fc_calculator_options,
        log_level=log_level,
    )
    return fc_solver.force_constants[2]


class FCSolver:
    """Force constants calculator interface."""

    def __init__(
        self,
        fc_solver_name: Literal["traditional", "symfc", "alm"],
        supercell: PhonopyAtoms,
        symmetry: Optional[Symmetry] = None,
        dataset: Optional[dict] = None,
        is_compact_fc: bool = False,
        primitive: Optional[Primitive] = None,
        orders: Optional[Sequence[int]] = None,
        options: Optional[str] = None,
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
    def fc_solver(self) -> Union[FDFCSolver, SymfcFCSolver, ALMFCSolver]:
        """Return force constants solver class instance."""
        return self._fc_solver

    @property
    def force_constants(self) -> dict[int, np.ndarray]:
        """Return force constants."""
        return self._fc_solver.force_constants

    def _set_fc_solver(self, fc_calculator_name: str):
        if fc_calculator_name not in fc_calculator_names:
            raise ValueError(
                f"fc_calculator_name must be one of {fc_calculator_names.keys()}"
            )

        if fc_calculator_name == "traditional":
            return self._set_traditional_solver()
        if fc_calculator_name == "symfc":
            return self._set_symfc_solver()
        if fc_calculator_name == "alm":
            return self._set_alm_solver()

    def _set_traditional_solver(self):
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
            raise RuntimeError("\n".join(lines))

        if self._is_compact_fc and self._primitive:
            atom_list = self._primitive.p2s_map
        else:
            atom_list = list(range(len(self._supercell)))
        return FDFCSolver(
            self._supercell,
            self._symmetry,
            self._dataset,
            atom_list=atom_list,
            primitive=self._primitive,
        )

    def _set_symfc_solver(self):
        from phonopy.interface.symfc import SymfcFCSolver, parse_symfc_options

        if self._dataset is None:
            return SymfcFCSolver(
                self._supercell,
                symmetry=self._symmetry,
                options=parse_symfc_options(self._options),
                is_compact_fc=self._is_compact_fc,
                log_level=self._log_level,
            )
        else:
            displacements, forces = get_displacements_and_forces(self._dataset)
            symfc_solver = SymfcFCSolver(
                self._supercell,
                displacements,
                forces,
                symmetry=self._symmetry,
                options=parse_symfc_options(self._options),
                is_compact_fc=self._is_compact_fc,
                log_level=self._log_level,
            )
            if self._orders is not None:
                symfc_solver.run(orders=self._orders)
            return symfc_solver

    def _set_alm_solver(self):
        from phonopy.interface.alm import ALMFCSolver

        if self._primitive is None:
            raise RuntimeError(
                "Primitive cell is required for the traditional FC solver."
            )

        if self._dataset is None:
            raise RuntimeError(
                "Displacement-force dataset is required for ALM FC solver."
            )
        displacements, forces = get_displacements_and_forces(self._dataset)

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
