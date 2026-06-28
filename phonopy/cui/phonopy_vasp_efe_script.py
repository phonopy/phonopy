# Copyright (C) 2018 Atsushi Togo
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
from phonopy.qha.electron import get_free_energy_at_T


@dataclasses.dataclass
class PhonopyVaspEfeMockArgs:
    """Mock args of ArgumentParser."""

    scale_factor: float = 1.0
    tmax: float = 1000.0
    tmin: float = 0.0
    tstep: float = 10.0
    quiet: bool = False
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


def get_fe_ev_lines(
    args: argparse.Namespace | PhonopyVaspEfeMockArgs,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Return Free energy vs volume lines.

    When ``verbose`` is True, progress is printed to stdout: which file is
    being read and what is found in it (volume, NELECT, the k-point mesh
    used for the electronic free energy).

    """
    volumes = []
    energy_sigma0 = []
    free_energies = []
    temperatures = None
    assert args.filenames is not None
    filenames = list(args.filenames)
    if verbose:
        print(
            "Reading %d vasprun.xml file(s), T = %g..%g K step %g K"
            % (len(filenames), args.tmin, args.tmax, args.tstep)
        )
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
        energy = vxml.energies[-1, 1]
        if verbose:
            k_mesh_str = "x".join("%d" % m for m in k_mesh)
            print(
                "        volume = %.4f A^3, NELECT = %g, "
                "free energy on %s (%s)"
                % (vxml.volume[-1], n_electrons, kpoints_label, k_mesh_str)
            )
            sys.stdout.flush()
        temps, fe = get_free_energy_at_T(
            args.tmin, args.tmax, args.tstep, eigenvalues, weights, n_electrons
        )
        volumes.append(vxml.volume[-1])
        energy_sigma0.append(energy)
        free_energies.append(energy - fe[0] + fe)
        if temperatures is None:
            temperatures = temps
        else:
            assert (np.abs(temperatures - temps) < 1e-5).all()
    assert temperatures is not None
    if verbose:
        print("Done. %d volume(s) processed." % len(filenames))
        sys.stdout.flush()

    scale_factor = args.scale_factor
    volumes = np.array(volumes) * scale_factor
    energy_sigma0 = np.array(energy_sigma0) * scale_factor
    free_energies = np.array(free_energies) * scale_factor

    lines_fe = []
    lines_fe.append(("# volume:  " + " %15.8f" * len(volumes)) % tuple(volumes))
    lines_fe.append("#    T(K)     Free energies")
    lines_fe += get_free_energy_lines(temperatures, free_energies.T)

    lines_ev = ["#   cell volume        energy of cell other than phonon"]
    lines_ev += [
        "%20.8f %20.8f" % (v, e) for v, e in zip(volumes, energy_sigma0, strict=True)
    ]

    return lines_fe, lines_ev


def main(**argparse_control: PhonopyVaspEfeMockArgs) -> None:
    """Run phonopy-vasp-efe."""
    args: PhonopyVaspEfeMockArgs | argparse.Namespace
    if argparse_control:
        args = argparse_control["args"]
    else:
        args = get_options()

    verbose = not getattr(args, "quiet", False)
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
