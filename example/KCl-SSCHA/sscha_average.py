#!/usr/bin/env python
"""Average the SSCHA free energies recorded in sscha_free_energies.yaml.

The free energy of every iteration is an estimate of the same quantity once the
iterations have reached the fixed point, so averaging them reduces the
statistical error by 1/sqrt(K) in the number K of iterations averaged. Whether
they have reached it is checked by comparing two estimates of the error of the
average:

    reported = (1/K) (sum_k sigma_k^2)^(1/2),
    scatter  = std(F) / sqrt(K),

where sigma_k is the "free_energy_error" of iteration k. The first assumes that
nothing but the sampling of the supercells moves the free energy; the second
measures how much it actually moved and assumes nothing. When the two agree the
iterations are stationary and independent and the average is worth 1/sqrt(K).
When the scatter is the larger, the force constants are still moving and the
leading iterations are a transient, which `--skip` drops.

Usage:

    % python sscha_average.py sscha_free_energies.yaml
    % python sscha_average.py --skip 1 ntrain-*/sscha_free_energies.yaml

"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml


def average_free_energy(filename: str | os.PathLike, skip: int = 0) -> dict:
    """Return the averaged free energy of one sscha_free_energies.yaml and its error.

    The energies keep the unit of the file, which is reported as
    ``free_energy_unit``.

    """
    with open(filename) as f:
        data = yaml.safe_load(f)

    iterations = data["iterations"][skip:]
    if not iterations:
        raise ValueError(f'No iteration is left in "{filename}" after --skip.')

    free_energies = np.array([entry["free_energy"] for entry in iterations])
    errors = np.array([entry["free_energy_error"] for entry in iterations])
    n = len(free_energies)
    reported = float(np.sqrt((errors**2).sum()) / n)
    if n > 1:
        scatter = float(free_energies.std(ddof=1) / np.sqrt(n))
    else:
        scatter = float("nan")

    return {
        "temperature": data["sscha"]["temperature"],
        "n_iterations": n,
        "free_energy": float(free_energies.mean()),
        "error_from_reported": reported,
        "error_from_scatter": scatter,
        "unit": data["free_energy_unit"],
    }


def main() -> None:
    """Average the files given in the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filenames", nargs="+", help="sscha_free_energies.yaml")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of leading iterations to drop as a transient",
    )
    args = parser.parse_args()

    results = [average_free_energy(f, skip=args.skip) for f in args.filenames]
    unit = {result["unit"] for result in results}
    print(f"# free energies in {'/'.join(sorted(unit))} per primitive cell")
    print(
        f"{'T(K)':>8} {'K':>4} {'mean':>12} {'reported':>10} {'scatter':>10} "
        f"{'ratio':>7}  file"
    )
    for filename, result in zip(args.filenames, results, strict=True):
        ratio = result["error_from_scatter"] / result["error_from_reported"]
        print(
            f"{result['temperature']:8.1f} {result['n_iterations']:4d} "
            f"{result['free_energy']:12.4f} {result['error_from_reported']:10.4f} "
            f"{result['error_from_scatter']:10.4f} {ratio:7.2f}  {filename}"
        )


if __name__ == "__main__":
    main()
