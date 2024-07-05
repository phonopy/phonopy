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

import numpy as np

from phonopy.interface.vasp import parse_vasprunxml
from phonopy.qha.electron import get_free_energy_at_T

"""Calculate electronic free energy from vasprun.xml at temperatures

Here the free energy is approximately given by:

    energy(sigma->0) - energy(T=0) + energy(T) - entropy(T) * T

"""


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy vasp-efe command-line-tool")
    parser.set_defaults(tmax=1000.0, tmin=0.0, tstep=10.0)
    parser.add_argument(
        "--tmax", dest="tmax", type=float, help="Maximum calculated temperature"
    )
    parser.add_argument(
        "--tmin", dest="tmin", type=float, help="Minimum calculated temperature"
    )
    parser.add_argument(
        "--tstep", dest="tstep", type=float, help="Calculated temperature step"
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames: vasprun.xml's of all volumes in correct order",
    )
    args = parser.parse_args()
    return args


def get_free_energy_lines(temperatures, free_energies):
    """Return Free energy lines."""
    lines = []
    n_vol = free_energies.shape[1]
    for t, fe in zip(temperatures, free_energies):
        lines.append(("%10.4f " + " %15.8f" * n_vol) % ((t,) + tuple(fe)))
    return lines


def get_fe_ev_lines(args):
    """Return Free energy vs volume lines."""
    volumes = []
    energy_sigma0 = []
    free_energies = []
    temperatures = None
    for filename in args.filenames:
        vxml = parse_vasprunxml(filename)
        weights = vxml.k_weights
        eigenvalues = vxml.eigenvalues[:, :, :, 0]
        n_electrons = vxml.NELECT
        energy = vxml.energies[-1, 1]
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

    lines_fe = []
    lines_fe.append(("# volume:  " + " %15.8f" * len(volumes)) % tuple(volumes))
    lines_fe.append("#    T(K)     Free energies")
    lines_fe += get_free_energy_lines(temperatures, np.transpose(free_energies))

    lines_ev = ["#   cell volume        energy of cell other than phonon"]
    lines_ev += ["%20.8f %20.8f" % (v, e) for v, e in zip(volumes, energy_sigma0)]

    return lines_fe, lines_ev


def run():
    """Run phonopy-vasp-efe."""
    args = get_options()

    lines_fe, lines_ev = get_fe_ev_lines(args)

    with open("fe-v.dat", "w") as w:
        w.write("\n".join(lines_fe))
        w.write("\n")
        print('* Electronic free energies are written in "fe-v.dat".')

    with open("e-v.dat", "w") as w:
        w.write("\n".join(lines_ev))
        w.write("\n")
        print('* energy (sigma->0) and volumes are written in "e-v.dat".')
