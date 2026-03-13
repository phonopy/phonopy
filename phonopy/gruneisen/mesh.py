"""Mode Grueneisen parameters calculation on sampling mesh."""

# Copyright (C) 2012 Atsushi Togo
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

import gzip
import lzma
import os
from collections.abc import Sequence
from typing import Any, Literal, TextIO

import numpy as np
from numpy.typing import NDArray

from phonopy.gruneisen.core import GruneisenBase
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.physical_units import get_physical_units
from phonopy.structure.grid_points import get_qpoints


class GruneisenMesh(GruneisenBase):
    """Class to calculate mode Grueneisen parameters on sampling mesh."""

    def __init__(
        self,
        dynmat: DynamicalMatrix,
        dynmat_plus: DynamicalMatrix,
        dynmat_minus: DynamicalMatrix,
        mesh: Sequence[int] | NDArray[np.int64],
        delta_strain: float | None = None,
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_gamma_center: bool = False,
        is_mesh_symmetry: bool = True,
        rotations: Sequence[Sequence[Sequence[int]]] | NDArray[np.int64] | None = None,
        factor: float | None = None,
    ) -> None:
        """Init method."""
        super().__init__(dynmat, dynmat_plus, dynmat_minus, delta_strain=delta_strain)
        self._mesh = np.array(mesh, dtype="int64")
        if factor is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = factor
        self._cell = dynmat.primitive
        self._qpoints, self._weights = get_qpoints(
            self._mesh,
            np.linalg.inv(self._cell.cell),  # type: ignore[arg-type]
            q_mesh_shift=shift,
            is_time_reversal=is_time_reversal,
            is_gamma_center=is_gamma_center,
            rotations=rotations,
            is_mesh_symmetry=is_mesh_symmetry,
        )
        self.set_qpoints(self._qpoints)
        assert self._gruneisen is not None
        assert self._eigenvalues is not None
        self._gamma: NDArray[np.double] = self._gruneisen
        self._frequencies: NDArray[np.double] = (
            np.sqrt(abs(self._eigenvalues)) * np.sign(self._eigenvalues) * self._factor
        )

    def get_gruneisen(self) -> NDArray[np.double]:
        """Return mode Grueneisen parameters."""
        return self._gamma

    def get_mesh_numbers(self) -> NDArray[np.int64]:
        """Return mesh numbers."""
        return self._mesh

    def get_qpoints(self) -> NDArray[np.double]:
        """Return (irreducible) q-points."""
        assert self._qpoints is not None
        return self._qpoints

    def get_weights(self) -> NDArray[np.int64]:
        """Return weights of (irreducible) q-points."""
        return self._weights

    def get_eigenvalues(self) -> NDArray[np.double]:
        """Return eigenvalues of dynamical matrices."""
        assert self._eigenvalues is not None
        return self._eigenvalues

    def get_eigenvectors(self) -> NDArray[np.cdouble]:
        """Return phonon eigenvectors."""
        assert self._eigenvectors is not None
        return self._eigenvectors

    def get_frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies."""
        return self._frequencies

    def write_yaml(
        self,
        comment: Any = None,
        filename: str | os.PathLike | None = None,
        compression: Literal["gzip", "lzma"] | None = None,
    ) -> None:
        """Write results in yaml file."""
        _filename: str | os.PathLike = ""
        if filename is not None:
            _filename = filename

        if compression is None:
            if filename is None:
                _filename = "gruneisen.yaml"
            with open(_filename, "w") as w:
                self._write_yaml(w, comment)
        elif compression == "gzip":
            if filename is None:
                _filename = "gruneisen.yaml.gz"
            with gzip.open(_filename, "wt") as w:
                self._write_yaml(w, comment)
        elif compression == "lzma":
            if filename is None:
                _filename = "gruneisen.yaml.xz"
            with lzma.open(_filename, "wt") as w:
                self._write_yaml(w, comment)

    def _write_yaml(
        self,
        w: TextIO,
        comment: Any,
    ) -> None:
        assert self._qpoints is not None
        assert self._weights is not None
        natom = len(self._cell)
        rec_lattice = np.linalg.inv(self._cell.cell)  # column vectors
        text = []
        text.append("mesh: [ %5d, %5d, %5d ]" % tuple(self._mesh))
        text.append("nqpoint: %d" % len(self._qpoints))
        text.append("reciprocal_lattice:")
        for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*"), strict=True):
            text.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,)))
        text.append("natom:   %-7d" % natom)
        text.append(str(self._cell))
        text.append("")
        text.append("phonon:")
        for q, m, gs, freqs in zip(
            self._qpoints, self._weights, self._gamma, self._frequencies, strict=True
        ):
            text.append("- q-position: [ %10.7f, %10.7f, %10.7f ]" % tuple(q))
            text.append("  multiplicity: %d" % m)
            text.append("  band:")
            for j, (g, freq) in enumerate(zip(gs, freqs, strict=True)):
                text.append("  - # %d" % (j + 1))
                text.append("    gruneisen: %15.10f" % g)
                text.append("    frequency: %15.10f" % freq)
            text.append("")

        self._write_lines(w, text)

    def _write_lines(self, w: TextIO, lines: list[str]) -> None:
        w.write("\n".join(lines))

    def write_hdf5(self, filename: str | os.PathLike = "gruneisen.hdf5") -> None:
        """Write results in hdf5 file."""
        import h5py  # type: ignore[import-untyped]

        w = h5py.File(filename, "w")
        w.create_dataset("mesh", data=self._mesh)
        w.create_dataset("gruneisen", data=self._gamma)
        w.create_dataset("weight", data=self._weights)
        w.create_dataset("frequency", data=self._frequencies)
        w.create_dataset("qpoint", data=self._qpoints)
        w.close()

    def plot(
        self,
        plt: Any,
        cutoff_frequency: float | None = None,
        color_scheme: str | None = None,
        marker: str = "o",
        markersize: float | None = None,
    ) -> None:
        """Return pyplot of calculation results."""
        n = len(self._gamma.T) - 1
        for i, (g, freqs) in enumerate(
            zip(self._gamma.T, self._frequencies.T, strict=True)
        ):
            if cutoff_frequency:
                g = np.extract(freqs > cutoff_frequency, g)
                freqs = np.extract(freqs > cutoff_frequency, freqs)

            if color_scheme == "RB":
                color = (1.0 / n * i, 0.0, 1.0 / n * (n - i))
                if markersize:
                    plt.plot(freqs, g, marker, color=color, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker, color=color)
            elif color_scheme == "RG":
                color = (1.0 / n * i, 1.0 / n * (n - i), 0.0)
                if markersize:
                    plt.plot(freqs, g, marker, color=color, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker, color=color)
            elif color_scheme == "RGB":
                color = (
                    max(2.0 / n * (i - n / 2.0), 0),
                    min(2.0 / n * i, 2.0 / n * (n - i)),
                    max(2.0 / n * (n / 2.0 - i), 0),
                )
                if markersize:
                    plt.plot(freqs, g, marker, color=color, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker, color=color)
            else:
                if markersize:
                    plt.plot(freqs, g, marker, markersize=markersize)
                else:
                    plt.plot(freqs, g, marker)
