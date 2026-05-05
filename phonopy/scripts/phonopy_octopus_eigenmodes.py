#!/usr/bin/env python

"""Generate Octopus phonon eigenmodes file from phonopy results."""

from __future__ import annotations

import argparse
import cmath
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
        self.num_atoms = len(phonon.unitcell)
        self.num_atoms_super = len(phonon.supercell)
        self.Np = self.num_atoms_super // self.num_atoms

        self.qpoints = get_commensurate_points(phonon.supercell_matrix)
        self.nq = len(self.qpoints)
        self.region_A, self.region_B = classify_qpoints(self.qpoints)
        self.calculation_done = False

    def _run(self) -> None:
        self.phonon.run_qpoints(
            self.qpoints, with_eigenvectors=True, nac_q_direction=[1, 0, 0]
        )
        result = self.phonon.get_qpoints_dict()
        self.eigenvectors = result["eigenvectors"]
        self.frequencies = result["frequencies"]

        self.num_eigenvectors = len(self.eigenvectors[0])
        self.num_modes = (
            len(self.region_A) + 2 * len(self.region_B)
        ) * self.num_eigenvectors

        units: PhysicalUnits = get_physical_units()
        self.THz_to_Ha = 1e12 * units.PlanckConstant / units.Hartree
        self.calculation_done = True

    def write_phonon_file(
        self, filename: str, keep_zero_modes: bool = True, freq_tolerance: float = 1e-7
    ) -> None:
        if not self.calculation_done:
            self._run()

        with open(filename, "w") as f:
            f.write(f"Version: {self.version}\n")

            num_modes = self.num_modes
            if not keep_zero_modes:
                num_modes -= self.dim_space

            f.write(f"Nmodes: {num_modes}\n")
            f.write(f"Np: {self.Np}\n")

            qR_l = np.zeros((self.nq, self.num_atoms_super))
            qR_l_kappa = np.zeros((self.nq, self.num_atoms_super))

            for iatom in range(self.num_atoms_super):
                R_l_kappa = self.phonon.supercell.scaled_positions[iatom]
                R_l = (
                    R_l_kappa
                    - self.phonon.supercell.scaled_positions[
                        self.phonon.supercell.s2u_map[iatom]
                    ]
                )
                for iq in range(self.nq):
                    q_scaled = self.qpoints[iq] @ self.phonon.supercell_matrix
                    qR_l[iq, iatom] = np.dot(q_scaled, R_l)
                    qR_l_kappa[iq, iatom] = np.dot(q_scaled, R_l_kappa)

            for iq in self.region_A:
                for nu in range(self.num_eigenvectors):
                    freq = self.frequencies[iq, nu] * self.THz_to_Ha
                    if freq <= freq_tolerance and not keep_zero_modes:
                        continue
                    eig = self.eigenvectors[iq, nu].reshape(self.num_atoms, 3)
                    f.write(
                        "# q (in reduced coords): "
                        + " ".join(f"{x:10.6f}" for x in self.qpoints[iq])
                        + "\n"
                    )
                    f.write(f"frequency: {freq:.8f} \n")
                    for iatom in range(self.num_atoms_super):
                        ispecies = int(self.phonon.supercell.s2u_map[iatom] / self.Np)
                        ext_eig = abs(eig[ispecies]) * math.cos(qR_l[iq, iatom])
                        f.write("".join(f"{x:10.6f}" for x in ext_eig) + "\n")
                    f.write("alpha: 1.0\n")

            for iq in self.region_B:
                for nu in range(self.num_eigenvectors):
                    freq = self.frequencies[iq, nu] * self.THz_to_Ha
                    eig = self.eigenvectors[iq, nu].reshape(self.num_atoms, 3)

                    f.write(
                        "# q (in reduced coords): "
                        + " ".join(f"{x:10.6f}" for x in self.qpoints[iq])
                        + "\n"
                    )
                    f.write(f"frequency: {freq:.8f} \n")
                    for iatom in range(self.num_atoms_super):
                        ispecies = int(self.phonon.supercell.s2u_map[iatom] / self.Np)
                        ext_eig = (
                            eig[ispecies]
                            * cmath.exp(complex(0, qR_l_kappa[iq, iatom]))
                        ).real
                        f.write("".join(f"{x:10.6f}" for x in ext_eig) + "\n")
                    f.write("alpha: 0.5\n")

                    f.write(
                        "# q (in reduced coords): "
                        + " ".join(f"{x:10.6f}" for x in self.qpoints[iq])
                        + "\n"
                    )
                    f.write(f"frequency: {freq:.8f} \n")
                    for iatom in range(self.num_atoms_super):
                        ispecies = int(self.phonon.supercell.s2u_map[iatom] / self.Np)
                        ext_eig = (
                            eig[ispecies]
                            * cmath.exp(complex(0, qR_l_kappa[iq, iatom]))
                        ).imag
                        f.write("".join(f"{x:10.6f}" for x in ext_eig) + "\n")
                    f.write("alpha: 0.5\n")


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
    octopus_phonon.write_phonon_file(args.filename, keep_zero_modes=True)


if __name__ == "__main__":
    run()
