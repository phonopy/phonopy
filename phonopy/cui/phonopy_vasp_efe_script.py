# SPDX-License-Identifier: BSD-3-Clause
"""Calculate electronic free energy from vasprun.xml at temperatures.

Here the free energy is approximately given by:

    energy(sigma->0) - energy(T=0) + energy(T) - entropy(T) * T

"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.interface.vasp import parse_vasprunxml
from phonopy.qha.electron import (
    ElectronicStates,
    get_free_energy_at_T,
    write_electronic_states_hdf5,
)


@dataclasses.dataclass
class PhonopyVaspEfeMockArgs:
    """Mock args of ArgumentParser."""

    scale_factor: float = 1.0
    tmax: float = 1000.0
    tmin: float = 0.0
    tstep: float = 10.0
    quiet: bool = False
    write_electronic_states: bool = False
    filenames: Sequence[os.PathLike | str] | None = None


def get_options() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Phonopy vasp-efe command-line-tool")
    default_vals = PhonopyVaspEfeMockArgs()
    parser.add_argument(
        "--scale-factor",
        dest="scale_factor",
        type=float,
        default=default_vals.scale_factor,
        help="Scaling factor for volume, energy, and free energy (default: 1.0)",
    )
    parser.add_argument(
        "--tmax",
        dest="tmax",
        type=float,
        default=default_vals.tmax,
        help="Maximum calculated temperature",
    )
    parser.add_argument(
        "--tmin",
        dest="tmin",
        type=float,
        default=default_vals.tmin,
        help="Minimum calculated temperature",
    )
    parser.add_argument(
        "--tstep",
        dest="tstep",
        type=float,
        default=default_vals.tstep,
        help="Calculated temperature step",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=default_vals.quiet,
        help="Suppress progress messages printed while reading vasprun.xml files",
    )
    parser.add_argument(
        "--es",
        "--write-electronic-states",
        dest="write_electronic_states",
        action="store_true",
        default=default_vals.write_electronic_states,
        help=(
            'Write eigenvalues etc. in "electronic_states.hdf5" instead of '
            'computing electronic free energies ("fe-v.dat")'
        ),
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames: vasprun.xml's of all volumes in correct order",
    )
    args = parser.parse_args()
    return args


def get_free_energy_lines(temperatures: NDArray, free_energies: NDArray) -> list[str]:
    """Return Free energy lines."""
    lines = []
    n_vol = free_energies.shape[1]
    for t, fe in zip(temperatures, free_energies, strict=True):
        lines.append(("%10.4f " + " %15.8f" * n_vol) % ((t,) + tuple(fe)))
    return lines


def get_ev_lines(
    volumes: NDArray[np.double], energies: NDArray[np.double]
) -> list[str]:
    """Return e-v.dat lines."""
    lines_ev = ["#   cell volume        energy of cell other than phonon"]
    lines_ev += [
        "%20.8f %20.8f" % (v, e) for v, e in zip(volumes, energies, strict=True)
    ]
    return lines_ev


def collect_electronic_states(
    args: argparse.Namespace | PhonopyVaspEfeMockArgs,
    verbose: bool = False,
) -> tuple[NDArray[np.double], NDArray[np.double], list[ElectronicStates]]:
    """Parse vasprun.xml files into volumes, energies, and electronic states.

    Returns (volumes (angstrom^3), energies (eV, sigma->0), electronic
    states) in the file order. When ``verbose`` is True, progress is
    printed to stdout: which file is being read and what is found in it
    (volume, NELECT, the k-point mesh used).

    """
    volumes = []
    energy_sigma0 = []
    states_list = []
    assert args.filenames is not None
    filenames = list(args.filenames)
    if verbose:
        print("Reading %d vasprun.xml file(s)" % len(filenames))
        sys.stdout.flush()
    for i, filename in enumerate(filenames):
        if verbose:
            print("  [%d/%d] reading %s" % (i + 1, len(filenames), filename))
            sys.stdout.flush()
        vxml = parse_vasprunxml(filename)
        # With KPOINTS_OPT the electronic DOS (hence the free energy) is
        # evaluated on the denser kpoints_opt mesh, so prefer it when present.
        if vxml.has_kpoints_opt:
            weights = vxml.k_weights_kpoints_opt
            eigenvalues = vxml.eigenvalues_kpoints_opt[:, :, :, 0]
            k_mesh = vxml.k_mesh_kpoints_opt
            kpoints_label = "KPOINTS_OPT mesh"
        else:
            weights = vxml.k_weights
            eigenvalues = vxml.eigenvalues[:, :, :, 0]
            k_mesh = vxml.k_mesh
            kpoints_label = "SCF mesh"
        n_electrons = vxml.NELECT
        assert n_electrons is not None
        if verbose:
            k_mesh_str = "x".join("%d" % m for m in k_mesh)
            print(
                "        volume = %.4f A^3, NELECT = %g, "
                "electronic states on %s (%s)"
                % (vxml.volume[-1], n_electrons, kpoints_label, k_mesh_str)
            )
            sys.stdout.flush()
        states_list.append(
            ElectronicStates(
                eigenvalues=eigenvalues,
                weights=weights,
                n_electrons=n_electrons,
                spin_degeneracy=vxml.spin_degeneracy,
                fermi_energy=vxml.efermi,
                volume=vxml.volume[-1],
                internal_energy=vxml.energies[-1, 1],
            )
        )
        volumes.append(vxml.volume[-1])
        energy_sigma0.append(vxml.energies[-1, 1])
    if verbose:
        print("Done. %d volume(s) processed." % len(filenames))
        sys.stdout.flush()
    return (
        np.array(volumes, dtype="double"),
        np.array(energy_sigma0, dtype="double"),
        states_list,
    )


def get_fe_ev_lines(
    args: argparse.Namespace | PhonopyVaspEfeMockArgs,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Return Free energy vs volume lines.

    When ``verbose`` is True, progress is printed to stdout.

    """
    if verbose:
        print("T = %g..%g K step %g K" % (args.tmin, args.tmax, args.tstep))
        sys.stdout.flush()
    volumes, energy_sigma0, states_list = collect_electronic_states(
        args, verbose=verbose
    )

    n_vol = len(states_list)
    if verbose:
        print("Computing electronic free energies for %d volume(s)" % n_vol)
        sys.stdout.flush()
    free_energies = []
    temperatures = None
    for i, (energy, electronic_states) in enumerate(
        zip(energy_sigma0, states_list, strict=True)
    ):
        if verbose:
            print(
                "  [%d/%d] volume = %.4f A^3" % (i + 1, n_vol, electronic_states.volume)
            )
            sys.stdout.flush()
        temps, fe = get_free_energy_at_T(
            args.tmin,
            args.tmax,
            args.tstep,
            electronic_states.eigenvalues,
            electronic_states.weights,
            electronic_states.n_electrons,
        )
        free_energies.append(energy - fe[0] + fe)
        if temperatures is None:
            temperatures = temps
        else:
            assert (np.abs(temperatures - temps) < 1e-5).all()
    assert temperatures is not None
    if verbose:
        print("Done. %d volume(s) processed." % n_vol)
        sys.stdout.flush()

    scale_factor = args.scale_factor
    volumes = volumes * scale_factor
    energy_sigma0 = energy_sigma0 * scale_factor
    free_energies_arr = np.array(free_energies) * scale_factor

    lines_fe = []
    lines_fe.append(("# volume:  " + " %15.8f" * len(volumes)) % tuple(volumes))
    lines_fe.append("#    T(K)     Free energies")
    lines_fe += get_free_energy_lines(temperatures, free_energies_arr.T)

    lines_ev = get_ev_lines(volumes, energy_sigma0)

    return lines_fe, lines_ev


def main(**argparse_control: PhonopyVaspEfeMockArgs) -> None:
    """Run phonopy-vasp-efe."""
    args: PhonopyVaspEfeMockArgs | argparse.Namespace
    if argparse_control:
        args = argparse_control["args"]
    else:
        args = get_options()

    verbose = not getattr(args, "quiet", False)

    if getattr(args, "write_electronic_states", False):
        if args.scale_factor != 1.0:
            print("--scale-factor cannot be combined with --write-electronic-states.")
            sys.exit(1)
        volumes, energy_sigma0, states_list = collect_electronic_states(
            args, verbose=verbose
        )
        write_electronic_states_hdf5(states_list)
        print('* Electronic states are written in "electronic_states.hdf5".')

        with open("e-v.dat", "w") as w:
            w.write("\n".join(get_ev_lines(volumes, energy_sigma0)))
            w.write("\n")
            print('* energy (sigma->0) and volumes are written in "e-v.dat".')

        sys.exit(0)

    lines_fe, lines_ev = get_fe_ev_lines(args, verbose=verbose)

    with open("fe-v.dat", "w") as w:
        w.write("\n".join(lines_fe))
        w.write("\n")
        print('* Electronic free energies are written in "fe-v.dat".')

    with open("e-v.dat", "w") as w:
        w.write("\n".join(lines_ev))
        w.write("\n")
        print('* energy (sigma->0) and volumes are written in "e-v.dat".')

    sys.exit(0)
