"""Tools to manage displacement dataset."""

# Copyright (C) 2020 Atsushi Togo
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

from typing import Optional

import numpy as np


def get_displacements_and_forces(
    disp_dataset: dict,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return displacements and forces of all atoms from displacement dataset.

    This is used to extract displacements and forces from displacement dataset.
    This method is considered more-or-less as a converter when the input is in
    type-1.

    Parameters
    ----------
    disp_dataset : dict
        Displacement dataset either in type-1 or type-2.

    Returns
    -------
    displacements : ndarray
        Displacements of all atoms in all supercells.
        shape=(snapshots, supercell atoms, 3), dtype='double', order='C'
    forces : ndarray or None
        Forces of all atoms in all supercells.
        shape=(snapshots, supercell atoms, 3), dtype='double', order='C'
        None is returned when forces don't exist.

    """
    if "first_atoms" in disp_dataset:
        natom = disp_dataset["natom"]
        disps = np.zeros(
            (len(disp_dataset["first_atoms"]), natom, 3), dtype="double", order="C"
        )
        forces = None
        for i, disp1 in enumerate(disp_dataset["first_atoms"]):
            disps[i, disp1["number"]] = disp1["displacement"]
            if "forces" in disp1:
                if forces is None:
                    forces = np.zeros_like(disps)
                forces[i] = disp1["forces"]
        return disps, forces
    elif "displacements" in disp_dataset:
        if "forces" in disp_dataset:
            forces = disp_dataset["forces"]
        else:
            forces = None
        return disp_dataset["displacements"], forces


def forces_in_dataset(dataset):
    """Check if forces in displacement dataset."""
    if dataset is None:
        return False

    if not isinstance(dataset, dict):
        raise RuntimeError("dataset is wrongly made.")

    if "first_atoms" in dataset:  # type-1
        for d in dataset["first_atoms"]:
            if "forces" not in d:
                return False
        return True

    if "forces" in dataset:  # type-2
        return True

    return False
