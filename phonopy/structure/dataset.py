# SPDX-License-Identifier: BSD-3-Clause
"""Tools to manage displacement dataset."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.displacement import (
    DisplacementDataset,
    Type1DisplacementDataset,
    Type2DisplacementDataset,
)


def get_displacements_and_forces(
    disp_dataset: DisplacementDataset,
) -> tuple[NDArray[np.double], NDArray[np.double] | None]:
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
        d1 = cast(Type1DisplacementDataset, disp_dataset)
        natom = d1["natom"]
        disps = np.zeros((len(d1["first_atoms"]), natom, 3), dtype="double", order="C")
        forces = None
        for i, disp1 in enumerate(d1["first_atoms"]):
            disps[i, disp1["number"]] = disp1["displacement"]
            if "forces" in disp1:
                if forces is None:
                    # For mixed-species (site-mixture) supercells, per-disp
                    # forces carry one row per expanded constituent rather
                    # than one per site, so the force array can have a
                    # different second-axis length than the displacement
                    # array.
                    force_natom = disp1["forces"].shape[0]
                    forces = np.zeros(
                        (len(d1["first_atoms"]), force_natom, 3),
                        dtype="double",
                        order="C",
                    )
                forces[i] = disp1["forces"]
        return disps, forces
    elif "displacements" in disp_dataset:
        d2 = cast(Type2DisplacementDataset, disp_dataset)
        if "forces" in d2:
            forces = d2["forces"]
        else:
            forces = None
        return d2["displacements"], forces
    else:
        raise RuntimeError("Unknown dataset format.")


def forces_in_dataset(dataset: DisplacementDataset | None) -> bool:
    """Check if forces in displacement dataset."""
    if dataset is None:
        return False

    if "first_atoms" in dataset:  # type-1
        d1 = cast(Type1DisplacementDataset, dataset)
        for d in d1["first_atoms"]:
            if "forces" not in d:
                return False
        return True

    if "forces" in dataset:  # type-2
        return True

    return False
