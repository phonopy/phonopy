#!/usr/bin/env python

"""Generate Octopus phonon eigenmodes file from phonopy results.

The file format and all conventions follow the interface notes
"Octopus-PhonoPy interface for multi-trajectory runs" (rev. 2):

- effective real supercell modes W_{l kappa; m} built from the phonopy
  (C-type) eigenvectors: V*cos(q.R_l) for q in region A (2q = G),
  2*Re / -2*Im of W^C*exp(2*pi*i*q.r) for q in region B (one member of
  each +-q pair);
- per-mode amplitude factors alpha_m = (2*g_m)^(-1/2), i.e. 1/sqrt(2)
  for region A and 1/2 for region B (g_m = |extended norm|^2 / Np);
- angular frequencies in atomic units (the h/hbar = 2*pi factor is
  contained in the THz -> Hartree conversion via Planck's constant);
- the acoustic Gamma modes are excluded from the file; imaginary or
  unexpected near-zero frequencies abort the generation;
- the file carries the atomic masses so Octopus can verify consistency.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from phonopy import Phonopy, load as phonopy_load
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.phonon.degeneracy import DEFAULT_CUTOFF, degenerate_sets
from phonopy.physical_units import PhysicalUnits, get_physical_units

ALPHA_A = 1.0 / math.sqrt(2.0)  # region A: alpha = (2*g)^{-1/2}, g = 1
ALPHA_B = 0.5  # region B: g = 2

# Modes with |nu| below this (THz) count as zero-frequency (acoustic at
# Gamma); anywhere else they signal an instability and abort.
ZERO_FREQ_TOLERANCE_THZ = 1e-2


class OctopusPhononModes:
    """Class to generate the phonon eigenmodes file for Octopus."""

    def __init__(
        self,
        phonon: Phonopy,
        nac_q_direction: list[float] | None = None,
        degeneracy_tolerance: float = DEFAULT_CUTOFF,
    ):
        self.version = "1.0"
        self.dim_space = 3

        # None follows the phonopy convention: no direction-dependent
        # non-analytic term at Gamma, i.e. the analytic q = 0 frequencies
        # without LO/TO splitting.
        self.nac_q_direction = nac_q_direction
        self.degeneracy_tolerance = degeneracy_tolerance

        self.phonon = phonon
        primitive = phonon.primitive
        self.num_atoms = len(primitive)
        self.num_atoms_super = len(phonon.supercell)
        self.Np = self.num_atoms_super // self.num_atoms

        # Supercell matrix with respect to the primitive cell (integer), as
        # used by DynmatToForceConstants for the commensurate q-point set.
        self.smat = np.rint(np.linalg.inv(primitive.primitive_matrix)).astype("int64")
        self.qpoints = get_commensurate_points(self.smat)
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
            self.qpoints,
            with_eigenvectors=True,
            nac_q_direction=self.nac_q_direction,
        )
        self.eigenvectors = self.phonon.qpoints.eigenvectors
        self.frequencies = self.phonon.qpoints.frequencies

        self.num_eigenvectors = self.dim_space * self.num_atoms
        # The three acoustic Gamma modes are excluded from the file.
        self.num_modes = (
            len(self.region_A) + 2 * len(self.region_B)
        ) * self.num_eigenvectors - self.dim_space

        self._check_frequencies()

        units: PhysicalUnits = get_physical_units()
        # h*nu / Hartree = hbar*omega in Hartree = angular frequency in a.u.
        self.THz_to_Ha = 1e12 * units.PlanckConstant / units.Hartree
        self.calculation_done = True

    def _check_frequencies(self) -> None:
        """Abort on imaginary or unexpected near-zero frequencies.

        Exactly the three acoustic modes at Gamma may (and must) be close to
        zero; those are excluded from the file. Any other near-zero or
        negative (imaginary) frequency signals a dynamical instability, for
        which the harmonic sampling is meaningless.
        """
        for iq in range(self.nq):
            is_gamma = np.allclose(self.qpoints[iq], 0.0)
            freqs = self.frequencies[iq]
            small = np.abs(freqs) < ZERO_FREQ_TOLERANCE_THZ
            imaginary = freqs < -ZERO_FREQ_TOLERANCE_THZ
            if imaginary.any():
                raise RuntimeError(
                    f"Imaginary phonon frequency at q = {self.qpoints[iq]}: "
                    f"{freqs[imaginary]} THz. The structure is dynamically "
                    "unstable; harmonic Wigner sampling is meaningless."
                )
            expected_zeros = self.dim_space if is_gamma else 0
            if small.sum() != expected_zeros:
                raise RuntimeError(
                    f"Expected {expected_zeros} near-zero frequencies at "
                    f"q = {self.qpoints[iq]}, found {small.sum()} "
                    f"(frequencies {freqs[small]} THz, tolerance "
                    f"{ZERO_FREQ_TOLERANCE_THZ} THz)."
                )

    def _extended_eigenvector(self, iq: int, nu: int) -> np.ndarray:
        """Extend eigenvector nu at q-point iq to the supercell.

        Returns the complex array ``W^C_kappa * exp(2*pi*i*q.r_{l kappa})``
        (= W^D extended by Bloch's theorem) with shape
        (num_atoms_super, 3).
        """
        # phonopy stores band eigenvectors as the columns of the returned
        # (bands, bands) matrix.
        eig = self.eigenvectors[iq][:, nu].reshape(self.num_atoms, 3)
        phases = np.exp(2j * np.pi * (self.r_prim @ self.qpoints[iq]))
        return eig[self.s2p_index] * phases[:, None]

    def write_phonon_file(self, filename: str) -> None:
        """Write the phonon modes file (format version 1.0)."""
        if not self.calculation_done:
            self._run()

        with open(filename, "w") as f:
            f.write(f"Version: {self.version}\n")
            f.write(f"Nmodes: {self.num_modes}\n")
            f.write(f"Natoms: {self.num_atoms_super}\n")
            f.write(f"Np: {self.Np}\n")
            f.write("Masses:\n")
            masses = self.phonon.supercell.masses
            for i in range(0, self.num_atoms_super, 8):
                f.write(" ".join(f"{m:.8f}" for m in masses[i : i + 8]) + "\n")
            f.write("# Masses are in AMU, in the atom order of the geometry file.\n")
            f.write(
                "# Supercell matrix wrt the primitive cell (rows): "
                + "; ".join(" ".join(str(x) for x in row) for row in self.smat)
                + "\n"
            )
            f.write(
                "# Commensurate q-points (reduced coords of the primitive "
                "cell):\n"
            )
            for iq in range(self.nq):
                region = (
                    "A" if iq in self.region_A
                    else ("B" if iq in self.region_B else "C")
                )
                f.write(
                    "#   "
                    + " ".join(f"{x:10.6f}" for x in self.qpoints[iq])
                    + f"  region {region}\n"
                )

            for iq in self.region_A:
                is_gamma = np.allclose(self.qpoints[iq], 0.0)
                # At region-A q-points (2q = G) each (possibly degenerate)
                # frequency subspace is closed under complex conjugation, so
                # it has a real basis. Extract it from the real and imaginary
                # parts of the extended eigenvectors via an SVD; this also
                # handles the arbitrary global phase of a single eigenvector.
                for group in self._degenerate_groups(iq):
                    if is_gamma and group[0] < self.dim_space:
                        # Acoustic Gamma modes are excluded from the file.
                        continue
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
                            "adjusting --degeneracy-tolerance (current value "
                            f"{self.degeneracy_tolerance} THz)."
                        )
                    for k, nu in enumerate(group):
                        # |extended vector|^2 = Np (g = 1).
                        ext_eig = (vt[k] * math.sqrt(self.Np)).reshape(
                            self.num_atoms_super, 3
                        )
                        self._write_mode(f, iq, nu, ext_eig, alpha=ALPHA_A)

            for iq in self.region_B:
                for nu in range(self.num_eigenvectors):
                    ext_eig = self._extended_eigenvector(iq, nu)
                    # Re/Im pair; |2*Re|^2 = |2*Im|^2 = 2*Np (g = 2).
                    self._write_mode(f, iq, nu, 2.0 * ext_eig.real, alpha=ALPHA_B)
                    self._write_mode(f, iq, nu, -2.0 * ext_eig.imag, alpha=ALPHA_B)

    def _degenerate_groups(self, iq: int) -> list[list[int]]:
        """Group band indices at q-point iq by (near-)degenerate frequency.

        Uses phonopy's shared transitive grouping (a band joins a group if it
        is within the cutoff of any member), cf. group_velocity and irreps.
        """
        return degenerate_sets(
            self.frequencies[iq], cutoff=self.degeneracy_tolerance
        )

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
        f.write(f"alpha: {alpha:.16f}\n")


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
    parser.add_argument(
        "--nac-q-direction",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "direction of the q -> 0 limit for the non-analytical term "
            "correction at Gamma, in reduced coordinates (same as phonopy's "
            "--nac-q-direction / Q_DIRECTION). Default: no direction, i.e. "
            "the analytic q = 0 frequencies without LO/TO splitting."
        ),
    )
    parser.add_argument(
        "--degeneracy-tolerance",
        type=float,
        default=DEFAULT_CUTOFF,
        help=(
            "frequency cutoff (THz) for grouping degenerate modes before "
            f"constructing their real basis (default: {DEFAULT_CUTOFF})."
        ),
    )
    args = parser.parse_args()

    phonon = phonopy_load(
        "phonopy_disp.yaml", force_sets_filename="FORCE_SETS", calculator="octopus"
    )
    octopus_phonon = OctopusPhononModes(
        phonon,
        nac_q_direction=args.nac_q_direction,
        degeneracy_tolerance=args.degeneracy_tolerance,
    )
    octopus_phonon.write_phonon_file(args.filename)


if __name__ == "__main__":
    run()
