"""Routines to handle displacements in supercells."""

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

from collections.abc import Sequence
from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import determinant
from phonopy.structure.symmetry import Symmetry

directions_axis: NDArray[np.int64] = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64", order="C"
)

directions_diag: NDArray[np.int64] = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ],
    dtype="int64",
    order="C",
)


DisplacementDirection: TypeAlias = Sequence[int] | NDArray[np.int64]


class FirstAtomDisplacement(TypedDict):
    """Displacement information of one displaced atom."""

    number: int
    displacement: NDArray[np.double]


class FirstAtomDisplacementWithForces(FirstAtomDisplacement, total=False):
    """Type-1 displacement entry optionally containing forces and energy."""

    forces: NDArray[np.double]
    supercell_energy: float


class Type1DisplacementDatasetBase(TypedDict):
    """Displacement dataset in the type-1 format."""

    natom: int
    first_atoms: list[FirstAtomDisplacementWithForces]


class Type1DisplacementDataset(Type1DisplacementDatasetBase, total=False):
    """Type-1 displacement dataset with optional accompanying properties.

    cutoff_distance is not (yet) used in phonopy.

    """

    cutoff_distance: float


class Type2DisplacementDatasetBase(TypedDict):
    """Displacement dataset in the type-2 format."""

    displacements: NDArray[np.double]


class Type2DisplacementDataset(Type2DisplacementDatasetBase, total=False):
    """Type-2 displacement dataset with optional accompanying properties.

    cutoff_distance is not (yet) used in phonopy.

    """

    forces: NDArray[np.double]
    supercell_energies: NDArray[np.double]
    random_seed: int
    cutoff_distance: float


DisplacementDataset: TypeAlias = Type1DisplacementDataset | Type2DisplacementDataset


def directions_to_displacement_dataset(
    displacement_directions: Sequence[DisplacementDirection] | NDArray[np.int64],
    distance: float,
    supercell: PhonopyAtoms,
) -> Type1DisplacementDataset:
    """Transform displacement directions to displacements in Cartesian coordinates."""
    lattice = supercell.cell
    first_atoms: list[FirstAtomDisplacementWithForces] = []
    for disp in displacement_directions:
        direction = disp[1:]
        disp_cartesian = np.dot(direction, lattice)
        disp_cartesian *= distance / np.linalg.norm(disp_cartesian)
        first_atoms.append(
            {
                "number": int(disp[0]),
                "displacement": np.array(disp_cartesian, dtype="double"),
            }
        )
    displacement_dataset: Type1DisplacementDataset = {
        "natom": len(supercell),
        "first_atoms": first_atoms,
    }

    return displacement_dataset


def get_least_displacements(
    symmetry: Symmetry,
    is_plusminus: Literal["auto"] | bool = "auto",
    is_diagonal: bool = True,
    is_trigonal: bool = False,
    log_level: int = 0,
) -> list[list[int]]:
    """Return a set of displacements.

    Returns
    -------
    array_like
        List of directions with respect to axes. This gives only the
        symmetrically non equivalent directions. The format is like:
           [[0, 1, 0, 0],
            [7, 1, 0, 1], ...]
        where each list is defined by:
           First value:      Atom index in supercell starting with 0
           Second to fourth: If the direction is displaced or not (1, 0, or -1)
                             with respect to the axes.

    """
    displacements = []
    if is_diagonal:
        directions = directions_diag
    else:
        directions = directions_axis

    if log_level > 2:
        print("Site point symmetry:")

    for atom_num in symmetry.get_independent_atoms():
        site_symmetry = symmetry.get_site_symmetry(atom_num)

        if log_level > 2:
            print("Atom %d" % (atom_num + 1))
            for i, rot in enumerate(site_symmetry):
                print("----%d----" % (i + 1))
                for v in rot:
                    print("%2d %2d %2d" % tuple(v))

        for disp in get_displacement(site_symmetry, directions, is_trigonal, log_level):
            displacements.append([atom_num, disp[0], disp[1], disp[2]])
            if is_plusminus == "auto":
                if is_minus_displacement(disp, site_symmetry):
                    displacements.append([atom_num, -disp[0], -disp[1], -disp[2]])
            elif is_plusminus is True:
                displacements.append([atom_num, -disp[0], -disp[1], -disp[2]])

    return displacements


def get_displacement(
    site_symmetry: NDArray[np.int64],
    directions: NDArray[np.int64] = directions_diag,
    is_trigonal: bool = False,
    log_level: int = 0,
) -> list[NDArray[np.int64]]:
    """Return displacement directions for one atom."""
    # One
    sitesym_num, disp = _get_displacement_one(site_symmetry, directions)
    if disp is not None:
        if log_level > 2:
            print("Site symmetry used to expand a direction %s" % disp[0])
            print(site_symmetry[sitesym_num])
        return disp
    # Two
    sitesym_num, disps = _get_displacement_two(site_symmetry, directions)
    if disps is not None:
        if log_level > 2:
            print(
                "Site symmetry used to expand directions %s %s" % (disps[0], disps[1])
            )
            print(site_symmetry[sitesym_num])

        if is_trigonal:
            disps_new = [disps[0]]
            if _is_trigonal_axis(site_symmetry[sitesym_num]):
                if log_level > 2:
                    print("Trigonal axis is found.")
                disps_new.append(np.dot(disps[0], site_symmetry[sitesym_num].T))
                disps_new.append(np.dot(disps_new[1], site_symmetry[sitesym_num].T))
            disps_new.append(disps[1])
            return disps_new
        else:
            return disps
    # Three
    return [directions[0], directions[1], directions[2]]


def _get_displacement_one(
    site_symmetry: NDArray[np.int64],
    directions: NDArray[np.int64] = directions_diag,
) -> tuple[int, list[NDArray[np.int64]]] | tuple[None, None]:
    """Return one displacement.

    This method tries to find three linearly independent displacements by
    applying site symmetry to an input displacement.

    """
    for direction in directions:
        rot_directions = []
        for r in site_symmetry:
            rot_directions.append(np.dot(direction, r.T))
        num_sitesym = len(site_symmetry)
        for i in range(num_sitesym):
            for j in range(i + 1, num_sitesym):
                det = determinant([direction, rot_directions[i], rot_directions[j]])
                if det != 0:
                    return i, [direction]
    return None, None


def _get_displacement_two(
    site_symmetry: NDArray[np.int64],
    directions: NDArray[np.int64] = directions_diag,
) -> tuple[int, list[NDArray[np.int64]]] | tuple[None, None]:
    """Return one displacement.

    This method tries to find three linearly independent displacements by
    applying site symmetry to two input displacements.

    """
    for direction in directions:
        rot_directions = []
        for r in site_symmetry:
            rot_directions.append(np.dot(direction, r.T))
        num_sitesym = len(site_symmetry)
        for i in range(num_sitesym):
            for second_direction in directions:
                det = determinant([direction, rot_directions[i], second_direction])
                if det != 0:
                    return i, [direction, second_direction]
    return None, None


def is_minus_displacement(
    direction: NDArray[np.int64], site_symmetry: NDArray[np.int64]
) -> bool:
    """Symmetrically check if minus displacement is necessary or not."""
    is_minus = True
    for r in site_symmetry:
        rot_direction = np.dot(direction, r.T)
        if (rot_direction + direction).any():
            continue
        else:
            is_minus = False
            break
    return is_minus


def _is_trigonal_axis(r: NDArray[np.int64]) -> bool:
    """Check three folded rotation.

    True if r^3 = identity.

    """
    r3 = np.dot(np.dot(r, r), r)
    if (r3 == np.eye(3, dtype=int)).all():
        return True
    else:
        return False


def get_random_displacements_dataset(
    number_of_snapshots: int,
    num_atoms: int,
    distance: float,
    random_seed: int | None = None,
    is_plusminus: bool = False,
    max_distance: float | None = None,
    distance_per_atom: bool = False,
) -> NDArray[np.double]:
    """Return supercell displacements in random directions.

    Every atom is displaced in a direction drawn uniformly on the unit sphere.
    The distance is fixed to `distance` unless `max_distance` is given, in
    which case it is random; see `max_distance` for its distribution and
    `distance_per_atom` for whether it is drawn per supercell or per atom.

    Parameters
    ----------
    number_of_snapshots : int
        Number of supercells with random displacements to generate.
    num_atoms : int
        Number of atoms in supercell.
    distance : float
        Displacement distance. Unit is the same as that used for crystal
        structure. With `max_distance` None, this is the distance every atom
        is displaced by. Otherwise it is the lower bound of the random
        distance; see `max_distance`.
    random_seed : int or None, optional
        Random seed for random displacements generation. Default is None.
    is_plusminus : bool, optional
        In addition to sets of usual random displacements for supercell, sets of
        the opposite displacements for supercell are concatenated. Therefore,
        total number of sets of displacements is `2 * number_of_snapshots`.
        Default is False.
    max_distance : float or None, optional
        Upper bound of the random displacement distance. One distance is drawn
        from the uniform distribution over [0, max_distance) and is then raised
        to `distance` when smaller, so the distances are uniform over
        [distance, max_distance) except for the weight
        `distance / max_distance` piled up exactly at `distance`. `distance`
        acts as a floor here, not as a sampling bound -- unless
        `distance_per_atom` is True, where it is the lower sampling bound and
        no weight piles up. When None, the distance is fixed to `distance`.
        Default is None.
    distance_per_atom : bool, optional
        Requires `max_distance`. When False, one distance is drawn per
        supercell and shared by all its atoms, so a supercell is a shell of
        one amplitude in configuration space, and the weight at `distance`
        reserves a share of wholly near-equilibrium supercells. When True, a
        distance is drawn independently for every atom, uniformly over
        [distance, max_distance), so every supercell spans the whole range
        internally. The floor is dropped in that case because per atom it
        would only put a spike of identical displacement magnitudes in every
        supercell rather than reserving near-equilibrium structures. Default
        is False.

    Returns
    -------
    NDArray[np.double]
        Displacements of atoms in supercells.
        shape=(number_of_snapshots, num_atoms, 3), dtype='double', order='C'.
        With `is_plusminus` True, shape[0] is `2 * number_of_snapshots`.

    """
    if distance_per_atom and max_distance is None:
        raise ValueError("distance_per_atom requires max_distance.")

    if np.issubdtype(type(random_seed), np.integer):
        rng = np.random.default_rng(seed=random_seed)
    else:
        rng = np.random.default_rng()

    num_supercells = number_of_snapshots

    if max_distance is None:
        directions = _get_random_directions(num_atoms * num_supercells, rng)
        disps = directions * distance
        supercell_disps = np.array(
            disps.reshape(num_supercells, num_atoms, 3), dtype="double", order="C"
        )
    else:
        if distance > max_distance:
            raise RuntimeError(
                "Random displacements generation failed. max_distance is too small."
            )
        directions = _get_random_directions(num_atoms * num_supercells, rng).reshape(
            num_supercells, num_atoms, 3
        )
        # Shape (num_supercells, 1) draws and broadcasts one distance over the
        # atoms of a supercell; (num_supercells, num_atoms) draws one per atom.
        # Both consume the same random stream as a flat draw of their size.
        shape = (num_supercells, num_atoms if distance_per_atom else 1)
        if distance_per_atom:
            # Sample [distance, max_distance) directly. Flooring instead would
            # pile weight exactly at `distance`, which is meaningful per
            # supercell -- it reserves a share of wholly near-equilibrium
            # supercells -- but per atom only plants a spike of identical
            # displacement magnitudes in every supercell.
            dists = distance + rng.random(shape) * (max_distance - distance)
        else:
            dists = rng.random(shape) * max_distance
            dists[dists < distance] = distance
        supercell_disps = np.array(
            directions * dists[:, :, None], dtype="double", order="C"
        )

    if is_plusminus is True:
        supercell_disps = np.array(
            np.concatenate((supercell_disps, -supercell_disps), axis=0),
            dtype="double",
            order="C",
        )
    return supercell_disps


def _get_random_directions(
    num_atoms: int, rng: np.random.Generator
) -> NDArray[np.double]:
    """Return random directions in sphere with radius 1."""
    xy = rng.standard_normal(size=(3, num_atoms * 2))
    r = np.linalg.norm(xy, axis=0)
    condition = r > 1e-10
    return (xy[:, condition][:, :num_atoms] / r[condition][:num_atoms]).T


def generate_systematic_displacements(
    supercell: PhonopyAtoms,
    symmetry: Symmetry,
    distance: float | None = None,
    is_plusminus: Literal["auto"] | bool = "auto",
    is_diagonal: bool = True,
    is_trigonal: bool = False,
    log_level: int = 0,
) -> Type1DisplacementDataset:
    """Return a dataset of systematic finite-difference displacements.

    One atom is displaced at a time, along the symmetrically inequivalent
    directions found from the site symmetry, which is the displacement
    pattern the built-in finite-difference force-constants calculation
    consumes.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell the displacements are generated for.
    symmetry : Symmetry
        Symmetry of the supercell, used to reduce the displacements to the
        symmetrically inequivalent ones.
    distance : float, optional
        Displacement distance. Unit is the same as that used for the crystal
        structure. None uses 0.01. Default is None.
    is_plusminus : 'auto', True, or False, optional
        For each atom, generate displacements in one direction (False), in
        both directions (True), or in both directions only when symmetry
        requires it ('auto'). Default is 'auto'.
    is_diagonal : bool, optional
        When False, displacements are made only along basis vectors; when
        True, displacements may be off the basis vectors if doing so reduces
        the number of displacements by symmetry. Default is True.
    is_trigonal : bool, optional
        Exists only for testing purposes. Default is False.
    log_level : int, optional
        Verbosity of the site-symmetry report. Default is 0.

    Returns
    -------
    Type1DisplacementDataset
        Dataset in the type-1 format, one displaced atom per entry.

    """
    _distance = 0.01 if distance is None else distance
    displacement_directions = get_least_displacements(
        symmetry,
        is_plusminus=is_plusminus,
        is_diagonal=is_diagonal,
        is_trigonal=is_trigonal,
        log_level=log_level,
    )
    return directions_to_displacement_dataset(
        displacement_directions, _distance, supercell
    )


def generate_random_displacements(
    supercell: PhonopyAtoms,
    number_of_snapshots: int,
    distance: float | None = None,
    is_plusminus: bool = False,
    random_seed: int | None = None,
    max_distance: float | None = None,
    distance_per_atom: bool = False,
) -> Type2DisplacementDataset:
    """Return a dataset of random-direction displacements at a fixed distance.

    All atoms of a supercell are displaced simultaneously in random
    directions, which is the displacement pattern an external force-constants
    calculator (symfc, ALM) or a machine-learning-potential training set
    consumes. The distance is the same for every atom of a supercell unless
    max_distance is given; see `max_distance`.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell the displacements are generated for. Only its number of
        atoms is used.
    number_of_snapshots : int
        Number of supercells with random displacements to generate.
    distance : float, optional
        Displacement distance. Unit is the same as that used for the crystal
        structure. With max_distance given, this is not the distance itself
        but its floor; see `max_distance`. None uses 0.01. Default is None.
    is_plusminus : bool, optional
        When True, the opposite displacements are concatenated, so the number
        of generated supercells is 2 * number_of_snapshots. Default is False.
    random_seed : int or None, optional
        Random seed. It is recorded in the returned dataset when given.
        Default is None.
    max_distance : float or None, optional
        Upper bound of the displacement distance. One distance is drawn per
        supercell from the uniform distribution over [0, max_distance) and is
        then raised to `distance` when smaller, so the distances are uniform
        over [distance, max_distance) except for the weight
        distance / max_distance piled up exactly at `distance`. When None, the
        distance is fixed to `distance`. Default is None.
    distance_per_atom : bool, optional
        Requires `max_distance`. Draw the random distance per atom rather than
        per supercell, uniformly over [distance, max_distance) and without the
        weight at `distance`, so every supercell spans the whole range
        internally. Default is False.

    Returns
    -------
    Type2DisplacementDataset
        Dataset in the type-2 format, carrying the random seed when given.

    """
    _distance = 0.01 if distance is None else distance
    d = get_random_displacements_dataset(
        number_of_snapshots,
        len(supercell),
        _distance,
        random_seed=random_seed,
        is_plusminus=is_plusminus,
        max_distance=max_distance,
        distance_per_atom=distance_per_atom,
    )
    dataset: Type2DisplacementDataset = {"displacements": d}
    if random_seed is not None:
        dataset["random_seed"] = random_seed
    return dataset


def estimate_number_of_snapshots(
    supercell: PhonopyAtoms,
    symmetry: Symmetry,
    max_distance: float | None = None,
    number_estimation_factor: float | None = None,
) -> int:
    """Return an estimate of the number of random-displacement snapshots.

    symfc counts the supercells needed to determine the second-order force
    constants, and that count is multiplied by a safety factor because the
    estimate assumes an ideal, noise-free fit.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell the displacements are generated for.
    symmetry : Symmetry
        Symmetry of the supercell.
    max_distance : float or None, optional
        Upper bound of the random displacement distance, used only to pick
        the default safety factor. Random distances spread the data over a
        range of amplitudes, so more snapshots are needed. Default is None.
    number_estimation_factor : float, optional
        Safety factor multiplying the symfc estimate. None uses 8 when
        max_distance is given and 4 otherwise. Default is None.

    Returns
    -------
    int
        Estimated number of snapshots.

    """
    from phonopy.interface.symfc import SymfcFCSolver

    number_of_snapshots = SymfcFCSolver(
        supercell, symmetry=symmetry
    ).estimate_numbers_of_supercells(orders=[2])[2]
    if number_estimation_factor is None:
        number_of_snapshots *= 8 if max_distance is not None else 4
    else:
        number_of_snapshots = int(number_of_snapshots * number_estimation_factor)
    return number_of_snapshots
