"""Polynomial machine learning potential interface."""

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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from phonopy.structure.atoms import PhonopyAtoms

try:
    from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
except ImportError:
    Pypolymlp = Any


@dataclass
class PypolymlpParams:
    """Parameters for pypolymlp."""

    cutoff: float = 8.0
    model_type: int = 3
    max_p: int = 2
    gtinv_order: int = 3
    gtinv_maxl: Sequence[int] = (8, 8)
    gaussian_params2: Sequence[float, float, int] = (0.0, 7.0, 10)


def develop_polymlp(
    supercell: PhonopyAtoms,
    atom_energies: dict[str, float],
    train_displacements: np.ndarray,
    train_forces: np.ndarray,
    train_energies: np.ndarray,
    test_displacements: np.ndarray,
    test_forces: np.ndarray,
    test_energies: np.ndarray,
    params: Optional[PypolymlpParams] = None,
):
    """Develop polynomial MLPs of pypolymlp."""
    try:
        from pypolymlp.mlp_dev.pypolymlp import Pypolymlp
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
    except ImportError as exc:
        raise ModuleNotFoundError("Pypolymlp python module was not found.") from exc

    if params is None:
        _params = PypolymlpParams()
    else:
        _params = params

    polymlp = Pypolymlp()
    elements_energies = {s: atom_energies for s in supercell.symbols}
    polymlp.set_params(
        elements=list(elements_energies.keys()),
        cutoff=_params.cutoff,
        model_type=_params.model_type,
        max_p=_params.max_p,
        gtinv_order=_params.gtinv_order,
        gtinv_maxl=_params.gtinv_maxl,
        gaussian_params2=_params.gaussian_params2,
        atomic_energy=list(elements_energies.values()),
    )
    polymlp.set_datasets_displacements(
        train_displacements,
        train_forces,
        train_energies,
        test_displacements,
        test_forces,
        test_energies,
        phonopy_cell_to_st_dict(supercell),
    )
    polymlp.run(verbose=True)
    return polymlp


def evalulate_polymlp(
    polymlp: Pypolymlp,  # type: ignore
    supercells_with_displacements: list[PhonopyAtoms],
):
    """Run force calculation using pypolymlp."""
    try:
        from pypolymlp.calculator.properties import Properties
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
    except ImportError as exc:
        raise ModuleNotFoundError("Pypolymlp python module was not found.") from exc

    prop = Properties(params_dict=polymlp.parameters, coeffs=polymlp.coeffs)
    energies, forces, stresses = prop.eval_multiple(
        [phonopy_cell_to_st_dict(scell) for scell in supercells_with_displacements]
    )
    energies = np.array(energies, dtype="double")
    forces = np.array(np.transpose(forces, (0, 2, 1)), dtype="double", order="C")
    stresses = np.array(stresses, dtype="double", order="C")
    return energies, forces, stresses
