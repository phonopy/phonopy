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

from typing import Optional

import numpy as np

directions_axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

directions_diag = np.array(
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
    ]
)


def directions_to_displacement_dataset(displacement_directions, distance, supercell):
    """Transform displacement directions to displacements in Cartesian coordinates."""
    lattice = supercell.cell
    first_atoms = []
    for disp in displacement_directions:
        direction = disp[1:]
        disp_cartesian = np.dot(direction, lattice)
        disp_cartesian *= distance / np.linalg.norm(disp_cartesian)
        first_atoms.append(
            {"number": int(disp[0]), "displacement": disp_cartesian.tolist()}
        )
    displacement_dataset = {
        "natom": len(supercell),
        "first_atoms": first_atoms,
    }

    return displacement_dataset


def get_least_displacements(
    symmetry,
    is_plusminus="auto",
    is_diagonal=True,
    is_trigonal=False,
    log_level=0,
):
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
    site_symmetry, directions=directions_diag, is_trigonal=False, log_level=0
):
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


def _get_displacement_one(site_symmetry, directions=directions_diag):
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
                det = _determinant(direction, rot_directions[i], rot_directions[j])
                if det != 0:
                    return i, [direction]
    return None, None


def _get_displacement_two(site_symmetry, directions=directions_diag):
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
                det = _determinant(direction, rot_directions[i], second_direction)
                if det != 0:
                    return i, [direction, second_direction]
    return None, None


def is_minus_displacement(direction, site_symmetry):
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


def _is_trigonal_axis(r):
    """Check three folded rotation.

    True if r^3 = identity.

    """
    r3 = np.dot(np.dot(r, r), r)
    if (r3 == np.eye(3, dtype=int)).all():
        return True
    else:
        return False


def _determinant(a, b, c):
    """Return determinant of 3x3 matrix of [a, b, c]."""
    det = (
        a[0] * b[1] * c[2]
        - a[0] * b[2] * c[1]
        + a[1] * b[2] * c[0]
        - a[1] * b[0] * c[2]
        + a[2] * b[0] * c[1]
        - a[2] * b[1] * c[0]
    )
    return det


def get_random_displacements_dataset(
    num_supercells: int,
    num_atoms: int,
    distance: float,
    random_seed: Optional[int] = None,
    is_plusminus: bool = False,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """Return random displacements at constant displacement distance.

    num_supercells : int
        Number of snapshots of supercells with random displacements. Random
        displacements are generated displacing all atoms in random directions
        with a fixed displacement distance specified by 'distance' parameter,
        i.e., all atoms in supercell are displaced with the same displacement
        distance in direct space.
    num_atoms : int
        Number of atoms in supercell.
    distance : float
        Displacement distance. Unit is the same as that used for crystal
        structure. In random direction displacements generation with random
        distance, the displacement distance smaller than this value is set to
        this value.
    random_seed : int or None, optional
        Random seed for random displacements generation. Default is None.
    is_plusminus : True, or False, optional
        In addition to sets of usual random displacements for supercell, sets of
        the opposite displacements for supercell are concatenated. Therefore,
        total number of sets of displacements is `2 * num_supercells`. Default
        is False.
    max_distance : float or None, optional
        In random direction and distance displacements generation, this value is
        specified. In random direction and random distance displacements
        generation, this value is used as `max_distance`.

    """
    if np.issubdtype(type(random_seed), np.integer):
        rng = np.random.default_rng(seed=random_seed)
    else:
        rng = np.random.default_rng()

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
        supercell_disps = np.zeros(
            (num_supercells, num_atoms, 3), dtype="double", order="C"
        )
        directions = _get_random_directions(num_atoms * num_supercells, rng).reshape(
            num_supercells, num_atoms, 3
        )
        dists = rng.random(num_supercells) * max_distance
        dists[dists < distance] = distance
        for i, (dirs, dist) in enumerate(zip(directions, dists)):
            supercell_disps[i] = dirs * dist

    if is_plusminus is True:
        supercell_disps = np.array(
            np.concatenate((supercell_disps, -supercell_disps), axis=0),
            dtype="double",
            order="C",
        )
    return supercell_disps


def _get_random_directions(num_atoms: int, rng: np.random.Generator) -> np.ndarray:
    """Return random directions in sphere with radius 1."""
    xy = rng.standard_normal(size=(3, num_atoms * 2))
    r = np.linalg.norm(xy, axis=0)
    condition = r > 1e-10
    return (xy[:, condition][:, :num_atoms] / r[condition][:num_atoms]).T
