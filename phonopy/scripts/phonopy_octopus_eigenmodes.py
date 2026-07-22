#!/usr/bin/env python

"""Generate Octopus phonon eigenmodes file from phonopy results."""

from __future__ import annotations

import argparse
import math

import numpy as np

from phonopy import Phonopy, load as phonopy_load
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.physical_units import PhysicalUnits, get_physical_units


class OctopusPhononModes:
    """Class to generate the phonon eigenmodes file for Octopus."""

    def __init__(self, phonon: Phonopy):
        self.version = "0.0"
        self.dim_space = 3

        self.phonon = phonon
        primitive = phonon.primitive
        self.num_atoms = len(primitive)
        self.num_atoms_super = len(phonon.supercell)
        self.Np = self.num_atoms_super // self.num_atoms

        # Supercell matrix with respect to the primitive cell (integer), as
        # used by DynmatToForceConstants for the commensurate q-point set.
        smat = np.rint(np.linalg.inv(primitive.primitive_matrix)).astype("int64")
        self.qpoints = get_commensurate_points(smat)
        self.nq = len(self.qpoints)
        self.region_A, self.region_B = classify_qpoints(self.qpoints)

        # Primitive-atom index for each supercell atom.
        p2p = {s: p for p, s in enumerate(primitive.p2s_map)}
        self.s2p_index = [p2p[s] for s in primitive.s2p_map]

        # Positions of the supercell atoms in primitive-cell reduced
        # coordinates; used for the Bloch phases exp(2*pi*i*q.r).
        self.r_prim = phonon.supercell.positions @ np.linalg.inv(primitive.cell)

        self.calculation_done = False

    def _run(self) -> None:
        self.phonon.run_qpoints(
            self.qpoints, with_eigenvectors=True, nac_q_direction=[1, 0, 0]
        )
        self.eigenvectors = self.phonon.qpoints.eigenvectors
        self.frequencies = self.phonon.qpoints.frequencies

        self.num_eigenvectors = self.dim_space * self.num_atoms
        self.num_modes = (
            len(self.region_A) + 2 * len(self.region_B)
        ) * self.num_eigenvectors

        units: PhysicalUnits = get_physical_units()
        self.THz_to_Ha = 1e12 * units.PlanckConstant / units.Hartree
        self.calculation_done = True

    def _extended_eigenvector(self, iq: int, nu: int) -> np.ndarray:
        """Extend eigenvector nu at q-point iq to the supercell.

        Returns the complex array ``e_kappa * exp(2*pi*i*q.r)`` with shape
        (num_atoms_super, 3), following phonopy's supercell extension
        convention (full atomic positions in the phase, cf.
        phonopy.phonon.modulation).
        """
        # phonopy stores band eigenvectors as the columns of the returned
        # (bands, bands) matrix.
        eig = self.eigenvectors[iq][:, nu].reshape(self.num_atoms, 3)
        phases = np.exp(2j * np.pi * (self.r_prim @ self.qpoints[iq]))
        return eig[self.s2p_index] * phases[:, None]

    def write_phonon_file(self, filename: str) -> None:
        """Write the phonon modes file in the Octopus PhononModesFile format.

        Zero-frequency (acoustic) modes are written as well; Octopus filters
        them itself via PhononModesZeroThreshold.

        The per-mode ``alpha`` amplitude factors follow from requiring each
        real supercell mode to obey the same convention as Octopus' molecular
        reference case (normalized eigenvector, alpha = 1): a region-A mode
        (2q = G) has |extended vector|^2 = Np, which the 1/sqrt(Np) prefactor
        in Octopus normalizes exactly, so alpha = 1; the region-B cos/sin
        modes have |extended vector|^2 = Np/2, so alpha = sqrt(2).
        """
        if not self.calculation_done:
            self._run()

        with open(filename, "w") as f:
            f.write(f"Version: {self.version}\n")
            f.write(f"Nmodes: {self.num_modes}\n")
            f.write(f"Np: {self.Np}\n")

            for iq in self.region_A:
                # At region-A q-points (2q = G) each (possibly degenerate)
                # frequency subspace is closed under complex conjugation, so
                # it has a real basis. Extract it from the real and imaginary
                # parts of the extended eigenvectors via an SVD; this also
                # handles the arbitrary global phase of a single eigenvector.
                for group in self._degenerate_groups(iq):
                    ext = np.array(
                        [
                            self._extended_eigenvector(iq, nu).ravel()
                            for nu in group
                        ]
                    )
                    stacked = np.concatenate([ext.real, ext.imag], axis=0)
                    _, s, vt = np.linalg.svd(stacked, full_matrices=False)
                    d = len(group)
                    if s[d - 1] < 1e-6 * s[0] or (
                        len(s) > d and s[d] > 1e-6 * s[0]
                    ):
                        raise RuntimeError(
                            f"Could not construct a real basis for modes "
                            f"{list(group)} at q = {self.qpoints[iq]}; try "
                            "adjusting the degeneracy grouping tolerance."
                        )
                    for k, nu in enumerate(group):
                        ext_eig = (vt[k] * math.sqrt(self.Np)).reshape(
                            self.num_atoms_super, 3
                        )
                        self._write_mode(f, iq, nu, ext_eig, alpha=1.0)

            for iq in self.region_B:
                for nu in range(self.num_eigenvectors):
                    ext_eig = self._extended_eigenvector(iq, nu)
                    self._write_mode(f, iq, nu, ext_eig.real, alpha=math.sqrt(2.0))
                    self._write_mode(f, iq, nu, ext_eig.imag, alpha=math.sqrt(2.0))

    def _degenerate_groups(self, iq: int, tol: float = 1e-4) -> list[list[int]]:
        """Group band indices at q-point iq by (near-)degenerate frequency."""
        freqs = self.frequencies[iq]
        groups: list[list[int]] = [[0]]
        for nu in range(1, self.num_eigenvectors):
            if abs(freqs[nu] - freqs[nu - 1]) < tol:
                groups[-1].append(nu)
            else:
                groups.append([nu])
        return groups

    def _write_mode(self, f, iq: int, nu: int, ext_eig: np.ndarray, alpha: float):
        """Write a single real mode block."""
        f.write(
            "# q (in reduced coords): "
            + " ".join(f"{x:10.6f}" for x in self.qpoints[iq])
            + "\n"
        )
        freq = self.frequencies[iq, nu] * self.THz_to_Ha
        f.write(f"frequency: {freq:.8f} \n")
        for row in ext_eig:
            f.write("".join(f"{x:10.6f}" for x in row) + "\n")
        f.write(f"alpha: {alpha:.10f}\n")


def classify_qpoints(
    q_list: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Classify q-points into regions A and B."""
    region_A: list[int] = []
    region_B: list[int] = []

    search_space = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    for i, q in enumerate(q_list):
        if any(np.linalg.norm(2 * np.array(q) - search_space, axis=1) < 0.01):
            region_A.append(i)
            continue

        test_q = -np.array(q) + search_space
        test = False
        for j in region_B:
            test = test or any(np.linalg.norm(np.array(q_list[j]) - test_q, axis=1) < 0.01)
        if not test:
            region_B.append(i)

    return region_A, region_B


def run() -> None:
    """Run the command-line entrypoint."""
    parser = argparse.ArgumentParser(
        description="Print PhononModes file for Octopus."
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="filename for phonon modes.",
        default="phonon_modes.txt",
    )
    args = parser.parse_args()

    phonon = phonopy_load(
        "phonopy_disp.yaml", force_sets_filename="FORCE_SETS", calculator="octopus"
    )
    octopus_phonon = OctopusPhononModes(phonon)
    octopus_phonon.write_phonon_file(args.filename)


if __name__ == "__main__":
    run()
