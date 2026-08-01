#!/usr/bin/env python
"""Calculate SSCHA free energies from saved phonopy_sscha_fc_*.yaml.xz files.

The saved files contain force constants only, so the anharmonic part of the
free energy is re-evaluated here by sampling displacements from the canonical
ensemble of the given force constants and evaluating supercell energies by the
polynomial MLPs in polymlp.yaml. The displacements are sampled from the force
constants in the file itself, so the value obtained is the SSCHA free energy of
those force constants.

``phonopy --sscha`` records the free energy of every iteration in
phonopy_sscha.yaml, and the value of an iteration belongs to the force
constants that iteration sampled, that is, the ones the previous iteration
wrote. Running this script on phonopy_sscha_fc_{n-1}.yaml.xz therefore
reproduces the value recorded for iteration n, to within the statistical
uncertainty, the two being computed from different random samples. The force
constants of the last iteration are the one set with no recorded value, and
obtaining their free energy is what this script is for.

The sampling mesh has to match the one of the run being compared with, since
the harmonic part of the free energy is sampled on it.

Usage:

    % python sscha_free_energy.py phonopy_sscha_fc_{1..10}.yaml.xz -t 300

"""

from __future__ import annotations

import argparse

import phonopy
from phonopy.sscha.core import MLPSSCHA


def calculate_free_energy(
    filename: str,
    mlp_filename: str = "polymlp.yaml",
    temperature: float = 300.0,
    number_of_snapshots: int = 1000,
    random_seed: int | None = None,
    mesh: float = 100.0,
) -> tuple[float, float]:
    """Return SSCHA free energy and its statistical error in eV.

    Both values are given per primitive cell.

    """
    ph = phonopy.load(filename, log_level=0)
    if ph.force_constants is None:
        raise RuntimeError(f'No force constants found in "{filename}".')
    ph.load_mlp(mlp_filename)
    assert ph.mlp is not None

    # Giving force constants makes MLPSSCHA sample from them rather than
    # generate its own, which is what evaluates their free energy.
    sscha = MLPSSCHA(
        ph,
        ph.mlp,
        temperature=temperature,
        number_of_snapshots=number_of_snapshots,
        mesh=mesh,
        random_seed=random_seed,
    )
    sscha.sample_supercells()
    sscha.calculate_free_energy()
    return sscha.free_energy, sscha.free_energy_error


def main() -> None:
    """Calculate SSCHA free energies of the files given in the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filenames", nargs="+", help="phonopy_sscha_fc_*.yaml.xz")
    parser.add_argument(
        "--mlp", default="polymlp.yaml", help="MLP file (default: polymlp.yaml)"
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=300.0, help="Temperature in K"
    )
    parser.add_argument(
        "--rd",
        type=int,
        default=1000,
        help="Number of supercells with random displacements",
    )
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
    parser.add_argument("--mesh", type=float, default=100.0, help="Mesh sampling")
    args = parser.parse_args()

    print(f"# Temperature: {args.temperature} K")
    print(f"# Number of supercells: {args.rd}")
    print("# free energy and statistical error in meV/primitive cell")
    for filename in args.filenames:
        free_energy, error = calculate_free_energy(
            filename,
            mlp_filename=args.mlp,
            temperature=args.temperature,
            number_of_snapshots=args.rd,
            random_seed=args.random_seed,
            mesh=args.mesh,
        )
        print(f"{filename} {free_energy * 1000:.3f} {error * 1000:.3f}", flush=True)


if __name__ == "__main__":
    main()
