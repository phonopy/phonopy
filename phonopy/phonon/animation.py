"""Convert phonon results to animation formats."""

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

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.interface.vasp import write_vasp
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_angles, get_cell_matrix, get_cell_parameters


def write_animation(
    dynamical_matrix: DynamicalMatrix,
    q_point: Sequence[float] | NDArray[np.double] | None = None,
    anime_type: str | None = "v_sim",
    band_index: int | None = None,
    amplitude: float | None = None,
    num_div: int | None = None,
    shift: NDArray[np.double] | None = None,
    factor: float | None = None,
    filename: str | os.PathLike | None = None,
) -> str | os.PathLike:
    """Write atomic modulations in animation format."""
    animation = Animation(q_point, dynamical_matrix, shift=shift)

    if anime_type == "v_sim":
        if filename:
            fname_out = animation.write_v_sim(
                amplitude=amplitude, factor=factor, filename=filename
            )
        else:
            fname_out = animation.write_v_sim(amplitude=amplitude, factor=factor)

    elif anime_type == "arc" or anime_type is None:
        assert band_index is not None
        if filename:
            fname_out = animation.write_arc(
                band_index, amplitude, num_div, filename=filename
            )
        else:
            fname_out = animation.write_arc(band_index, amplitude, num_div)
    elif anime_type == "xyz":
        assert band_index is not None
        if filename:
            fname_out = animation.write_xyz(
                band_index, amplitude, num_div, factor, filename=filename
            )
        else:
            fname_out = animation.write_xyz(band_index, amplitude, num_div, factor)
    elif anime_type == "jmol":
        if filename:
            fname_out = animation.write_xyz_jmol(
                amplitude=amplitude, factor=factor, filename=filename
            )
        else:
            fname_out = animation.write_xyz_jmol(amplitude=amplitude, factor=factor)
    elif anime_type == "poscar":
        assert band_index is not None
        if filename:
            fname_out = animation.write_POSCAR(
                band_index, amplitude, num_div, filename=filename
            )
        else:
            fname_out = animation.write_POSCAR(band_index, amplitude, num_div)
    else:
        raise RuntimeError("Animation format '%s' was not found." % anime_type)

    return fname_out


class Animation:
    """Class to convert phonon results to animation formats."""

    def __init__(
        self,
        qpoint: Sequence[float] | NDArray[np.double] | None,
        dynamical_matrix: DynamicalMatrix,
        shift: NDArray[np.double] | None = None,
    ) -> None:
        """Init method."""
        if qpoint is None:
            _qpoint: Sequence[float] | NDArray[np.double] = [0, 0, 0]
        else:
            _qpoint = qpoint
        dynamical_matrix.run(_qpoint)
        dynmat = dynamical_matrix.dynamical_matrix
        self._eigenvalues: NDArray[np.double]
        self._eigenvectors: NDArray[np.cdouble]
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(dynmat)  # type: ignore
        self._qpoint = _qpoint
        primitive = dynamical_matrix.primitive
        self._positions: NDArray[np.double] = primitive.scaled_positions
        self._symbols: list[str] = primitive.symbols
        self._masses: NDArray[np.double] = primitive.masses
        self._lattice: NDArray[np.double] = primitive.cell
        if shift is not None:
            self._positions = (self._positions + shift) % 1

    def _set_cell_oriented(self) -> None:
        # Re-oriented lattice xx, yx, yy, zx, zy, zz
        self._angles = get_angles(self._lattice)
        self._cell_params = get_cell_parameters(self._lattice)
        a, b, c = self._cell_params
        alpha, beta, gamma = self._angles
        self._lattice_oriented: NDArray[np.double] = get_cell_matrix(
            a, b, c, alpha, beta, gamma
        )
        self._positions_oriented: NDArray[np.double] = np.array(
            self._get_oriented_displacements(np.dot(self._positions, self._lattice)),
            dtype="double",
        )

    # For the orientation, see get_cell_matrix
    def _get_oriented_displacements(
        self, vec_cartesian: NDArray[np.double] | NDArray[np.cdouble]
    ) -> NDArray[np.double] | NDArray[np.cdouble]:
        return np.dot(
            np.dot(vec_cartesian, np.linalg.inv(self._lattice)), self._lattice_oriented
        )

    def _set_displacements(self, band_index: int) -> None:
        u = []
        for i, e in enumerate(self._eigenvectors[:, band_index]):
            u.append(e / np.sqrt(self._masses[i // 3]))

        self._displacements: NDArray[np.cdouble] = np.array(u).reshape(-1, 3)

    def write_v_sim(
        self,
        amplitude: float | None = None,
        factor: float | None = None,
        filename: str | os.PathLike = "anime.ascii",
    ) -> str | os.PathLike:
        """Write to file in v_sim format."""
        if amplitude is None:
            _amplitude = 1.0
        else:
            _amplitude = amplitude
        if factor is None:
            _factor = get_physical_units().DefaultToTHz
        else:
            _factor = factor
        self._set_cell_oriented()
        lat = self._lattice_oriented
        q = self._qpoint
        text = "# Phonopy generated file for v_sim 3.6\n"
        text += "%15.9f%15.9f%15.9f\n" % (lat[0, 0], lat[1, 0], lat[1, 1])
        text += "%15.9f%15.9f%15.9f\n" % (lat[2, 0], lat[2, 1], lat[2, 2])
        for s, p in zip(self._symbols, self._positions_oriented, strict=True):
            text += "%15.9f%15.9f%15.9f %2s\n" % (p[0], p[1], p[2], s)

        for i, val in enumerate(self._eigenvalues):
            if val > 0:
                omega = np.sqrt(val)
            else:
                omega = -np.sqrt(-val)
            self._set_displacements(i)
            text += "#metaData: qpt=[%f;%f;%f;%f \\\n" % (
                q[0],
                q[1],
                q[2],
                omega * _factor,
            )
            for u in self._get_oriented_displacements(self._displacements) * _amplitude:
                text += "#; %f; %f; %f; %f; %f; %f \\\n" % (
                    u[0].real,
                    u[1].real,
                    u[2].real,
                    u[0].imag,
                    u[1].imag,
                    u[2].imag,
                )
            text += "# ]\n"
        w = open(filename, "w")
        w.write(text)
        w.close()

        return filename

    def write_arc(
        self,
        band_index: int,
        amplitude: float | None = None,
        num_div: int | None = None,
        filename: str | os.PathLike = "anime.arc",
    ) -> str | os.PathLike:
        """Write to file in BIOSYM archive 3 format."""
        _amplitude = 1.0 if amplitude is None else amplitude
        _num_div = 20 if num_div is None else num_div
        self._set_cell_oriented()
        self._set_displacements(band_index - 1)
        displacements = self._get_oriented_displacements(self._displacements)

        a, b, c = self._cell_params
        alpha, beta, gamma = self._angles

        text = ""
        text += "!BIOSYM archive 3\n"
        text += "PBC=ON\n"

        for i in range(_num_div):
            text += "                                                                        0.000000\n"  # noqa E501
            text += "!DATE\n"
            text += "%-4s%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f\n" % (
                "PBC",
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
            )
            positions = (
                self._positions_oriented
                + (displacements * np.exp(2j * np.pi / _num_div * i)).imag * _amplitude
            )
            for j, p in enumerate(positions):
                text += "%-5s%15.9f%15.9f%15.9f CORE" % (
                    self._symbols[j],
                    p[0],
                    p[1],
                    p[2],
                )
                text += "%5s%3s%3s%9.4f%5s\n" % (
                    j + 1,
                    self._symbols[j],
                    self._symbols[j],
                    0.0,
                    j + 1,
                )

            text += "end\n"
            text += "end\n"

        w = open(filename, "w")
        w.write(text)
        w.close()

        return filename

    def write_xyz_jmol(
        self,
        amplitude: float | None = None,
        factor: float | None = None,
        filename: str | os.PathLike = "anime.xyz_jmol",
    ) -> str | os.PathLike:
        """Write to file in jmol xyz format."""
        _amplitude = 10.0 if amplitude is None else amplitude
        if factor is None:
            _factor = get_physical_units().DefaultToTHz
        else:
            _factor = factor
        self._set_cell_oriented()
        text = ""
        for i, val in enumerate(self._eigenvalues):
            if val > 0:
                freq = np.sqrt(val)
            else:
                freq = -np.sqrt(-val)
            self._set_displacements(i)
            displacements = (
                self._get_oriented_displacements(self._displacements) * _amplitude
            )
            text += "%d\n" % len(self._symbols)
            text += "q %s , b %d , f %f " % (str(self._qpoint), i + 1, freq * _factor)
            text += "(generated by Phonopy)\n"
            for s, p, u in zip(
                self._symbols, self._positions_oriented, displacements, strict=True
            ):
                text += "%-3s  %22.15f %22.15f %22.15f  " % (s, p[0], p[1], p[2])
                text += "%15.9f %15.9f %15.9f\n" % (u[0].real, u[1].real, u[2].real)
        w = open(filename, "w")
        w.write(text)
        w.close()

        return filename

    def write_xyz(
        self,
        band_index: int,
        amplitude: float | None = None,
        num_div: int | None = None,
        factor: float | None = None,
        filename: str | os.PathLike = "anime.xyz",
    ) -> str | os.PathLike:
        """Write to file in xyz format."""
        _amplitude = 1.0 if amplitude is None else amplitude
        _num_div = 20 if num_div is None else num_div
        if factor is None:
            _factor = get_physical_units().DefaultToTHz
        else:
            _factor = factor
        self._set_cell_oriented()
        freq = self._eigenvalues[band_index - 1]
        self._set_displacements(band_index - 1)
        displacements = self._get_oriented_displacements(self._displacements)
        text = ""
        for i in range(_num_div):
            text += "%d\n" % len(self._symbols)
            text += "q %s , b %d , f %f , " % (
                str(self._qpoint),
                band_index,
                freq * _factor,
            )
            text += "div %d / %d " % (i, _num_div)
            text += "(generated by Phonopy)\n"
            positions = (
                self._positions_oriented
                + (displacements * np.exp(2j * np.pi / _num_div * i)).imag * _amplitude
            )
            for j, p in enumerate(positions):
                text += "%-3s %22.15f %22.15f %22.15f\n" % (
                    self._symbols[j],
                    p[0],
                    p[1],
                    p[2],
                )
        w = open(filename, "w")
        w.write(text)
        w.close()

        return filename

    def write_POSCAR(
        self,
        band_index: int,
        amplitude: float | None = None,
        num_div: int | None = None,
        filename: str | os.PathLike = "APOSCAR",
    ) -> str | os.PathLike:
        """Write snapshots to files in VASP POSCAR format."""
        _amplitude = 1.0 if amplitude is None else amplitude
        _num_div = 20 if num_div is None else num_div
        self._set_displacements(band_index - 1)
        for i in range(_num_div):
            positions = (
                np.dot(self._positions, self._lattice)
                + (self._displacements * np.exp(2j * np.pi / _num_div * i)).imag
                * _amplitude
            )
            atoms = PhonopyAtoms(
                cell=self._lattice,
                positions=positions,
                masses=self._masses,
                symbols=self._symbols,
            )
            write_vasp((str(filename) + "-%03d") % i, atoms, direct=True)

        return filename
