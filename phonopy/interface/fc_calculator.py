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
from typing import Optional, Union

import numpy as np

from phonopy.harmonic.force_constants import FDFCCalculator
from phonopy.interface.symfc import SymfcCalculator, parse_symfc_options
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
    primitive: Primitive,
    dataset: dict,
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[str] = None,
    atom_list: Optional[Union[Sequence[int], np.ndarray]] = None,
    symmetry: Optional[Symmetry] = None,
    symprec: float = 1e-5,
    log_level: int = 0,
):
    """Supercell 2nd order force constants (fc2) are calculated.

    The expected shape of supercell fc2 to be returned is
        (len(atom_list), num_atoms, 3, 3).

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell
    primitive : Primitive
        Primitive cell
    dataset : dict
        Dataset that contains displacements, forces, and optionally energies.
    fc_calculator : str, optional
        Currently only 'alm' is supported. Default is None, meaning invoking
        'alm'.
    fc_calculator_options : str, optional
        This is arbitrary string.
    atom_list : array_like of int or None, optional
        List of supercell atomic indices that represent the first indices of
        force constant matrix. The default is None, which means
        all atoms in supercell represented by np.arange(num_atoms).
        Two shapes of force constant matrix (called 'full' and 'compact') are
        readable in phonopy, see Returns section below.
        full: atom_list == (0, 1, 2, ..., num_atoms -1) or None,
        compact : atom_list == primitive.p2s_map, i.e.,
        [all atoms in primitive cell in the atomic indices of supercell].
    symmetry : Symmetry, optional
        Symmetry of supercell. Default is None.
    symprec : float, optional
        Tolerance used in symmetry analysis. Default is 1e-5.
    log_level : integer or bool, optional
        Verbosity level. False or 0 means quiet. True or 1 means normal level
        of log to stdout. 2 gives verbose mode.

    Returns
    -------
    fc2 : ndarray
        2nd order force constants.
        shape=(len(atom_list), num_atoms, 3, 3), dtype='double', order='C'.

    """
    if fc_calculator is None or fc_calculator == "traditional":
        from phonopy.harmonic.force_constants import get_fc2

        if "displacements" in dataset:
            lines = [
                "Type-II dataset for displacements and forces was provided, ",
                "but the selected force constants calculator cannot process it.",
                "Use another force constants calculator, e.g., symfc, ",
                "to generate force constants.",
            ]
            raise RuntimeError("\n".join(lines))
        return get_fc2(
            supercell,
            symmetry,
            dataset,
            atom_list=atom_list,
            primitive=primitive,
        )

    displacements, forces = get_displacements_and_forces(dataset)
    if fc_calculator == "alm" or fc_calculator is None:
        from phonopy.interface.alm import get_fc2

        return get_fc2(
            supercell,
            primitive,
            displacements,
            forces,
            atom_list=atom_list,
            options=fc_calculator_options,
            log_level=log_level,
        )
    if fc_calculator == "symfc":
        from phonopy.interface.symfc import get_fc2

        return get_fc2(
            supercell,
            primitive,
            displacements,
            forces,
            atom_list=atom_list,
            symmetry=symmetry,
            log_level=log_level,
        )


class FCCalculator:
    """Force constants calculator interface."""

    def __init__(
        self,
        fc_calculator_name: str,
        supercell: PhonopyAtoms,
        symmetry: Symmetry,
        dataset: dict,
        atom_list: Optional[Sequence] = None,
        primitive: Optional[Primitive] = None,
        options: Optional[str] = None,
        log_level: int = 0,
    ):
        self._supercell = supercell
        self._symmetry = symmetry
        self._dataset = dataset
        self._atom_list = atom_list
        self._primitive = primitive
        self._options = options
        self._log_level = log_level

        self._fc_calculator = self._set_fc_calculator(fc_calculator_name)

        if self._primitive is None:
            self._is_compact_fc = False
        else:
            self._is_compact_fc = (
                atom_list is not None and (atom_list == primitive.p2s_map).all()
            )

    @property
    def fc_calculator(self):
        """Return force constants calculator."""
        return self._fc_calculator

    def _set_fc_calculator(self, fc_calculator_name: str):
        if fc_calculator_name not in fc_calculator_names:
            raise ValueError(
                f"fc_calculator_name must be one of {fc_calculator_names.keys()}"
            )

        if fc_calculator_name == "traditional":
            return FDFCCalculator(
                self._supercell,
                self._symmetry,
                self._dataset,
                atom_list=self._atom_list,
                primitive=self._primitive,
            )
        if fc_calculator_name == "symfc":
            displacements, forces = get_displacements_and_forces(self._dataset)
            return SymfcCalculator(
                self._supercell,
                displacements,
                forces,
                symmetry=self._symmetry,
                options=parse_symfc_options(self._options),
                log_level=self._log_level,
            )
