"""Train a stress-enabled pypolymlp for the Ti anisotropic-QHA example.

Reads the structure dataset built by ``phonopy-vasp-mlp-dataset`` (energies,
forces and stresses of strained supercells), splits it into train / test
subsets, trains a polynomial MLP using energies + forces + stresses, and
writes ``polymlp.yaml``.

Usage::

    python train_mlp.py [polymlp_dataset.hdf5] --test-ratio 0.1 --seed 0

"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from phonopy.interface.pypolymlp import (
    PypolymlpParams,
    PypolymlpStructureData,
    develop_pypolymlp_from_structures,
    read_pypolymlp_structure_dataset,
    save_pypolymlp,
)


def split_structure_data(
    data: PypolymlpStructureData, n_train: int, seed: int | None = None
) -> tuple[PypolymlpStructureData, PypolymlpStructureData]:
    """Split a structure dataset into train and test subsets.

    Structures are assigned to the subsets by a random permutation of their
    indices, so train and test both span the sampled (a, c) region. Stresses
    are carried along when present.

    Parameters
    ----------
    data : PypolymlpStructureData
        Full dataset.
    n_train : int
        Number of structures placed in the training subset.
    seed : int, optional
        Seed for the random permutation.

    Returns
    -------
    train, test : PypolymlpStructureData

    """
    n = len(data.structures)
    if not 0 < n_train < n:
        raise ValueError(f"n_train must be in (0, {n}), got {n_train}.")

    perm = np.random.default_rng(seed).permutation(n)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])

    def _subset(idx: NDArray[np.int64]) -> PypolymlpStructureData:
        return PypolymlpStructureData(
            structures=[data.structures[i] for i in idx],
            energies=data.energies[idx],
            forces=[data.forces[i] for i in idx],
            stresses=None if data.stresses is None else data.stresses[idx],
        )

    return _subset(train_idx), _subset(test_idx)


def main() -> None:
    """Run the Ti MLP training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="polymlp_dataset.hdf5",
        help="structure dataset from phonopy-vasp-mlp-dataset",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="fraction of structures held out for testing (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the train/test split (default: 0)"
    )
    parser.add_argument(
        "-o", "--output", default="polymlp.yaml", help="output MLP file"
    )
    args = parser.parse_args()

    data = read_pypolymlp_structure_dataset(args.dataset)
    n = len(data.structures)
    n_test = max(1, int(round(n * args.test_ratio)))
    n_train = n - n_test
    train, test = split_structure_data(data, n_train, seed=args.seed)

    has_stress = "yes" if data.stresses is not None else "no"
    print(
        f"Training on {n_train} structures, testing on {n_test} (stress: {has_stress})."
    )

    polymlp = develop_pypolymlp_from_structures(
        train, test, params=PypolymlpParams(), verbose=True
    )
    save_pypolymlp(polymlp, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
