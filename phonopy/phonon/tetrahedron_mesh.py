"""Linear tetrahedron method on regular grid."""

# copyright (C) 2013 Atsushi Togo
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

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.tetrahedron_method import TetrahedronMethod


class TetrahedronMesh:
    """Class to perform linear tetrahedron method on regular grid."""

    def __init__(
        self,
        cell: PhonopyAtoms,
        frequencies,  # only at ir-grid-points
        mesh,
        grid_address,
        grid_mapping_table,
        ir_grid_points,
        grid_order=None,
        lang="C",
    ):
        """Linear tetrahedron method on uniform mesh for phonons.

        Parameters
        ----------
        cell : PhonopyAtoms
            Primitive cell used to calculate frequencies
        frequencies: ndarray
            Phonon frequences on ir-grid points
            shape=(num_ir_grid_points, num_band)
            dtype='double'
        mesh : ndarray or list of int
            Mesh numbers for grids
            shape=(3,)
            dtype='long'
        grid_address : ndarray
            Addresses of all grid points given by GridPoints class.
            shape=(prod(mesh), 3)
            dtype='long'
        grid_mapping_table : ndarray
            Mapping of grid points to irreducible grid points given by
            GridPoints class.
            shape=(prod(mesh),)
            dtype='long'
        ir_grid_points : ndarray
            Irreducible gird points given by GridPoints class.
            shape=(len(np.unique(grid_mapping_table)),)
            dtype='long'
        grid_order : list of int, optional
            This controls how grid addresses are stored either C style or
            Fortran style.
        lang : str, 'C' or else, optional
            With 'C', C implementation is used. Otherwise Python implementation
            runs.

        """
        self._cell = cell
        self._frequencies = frequencies
        self._mesh = np.array(mesh, dtype="long")
        self._grid_address = grid_address
        self._grid_mapping_table = grid_mapping_table
        self._lang = lang
        if lang == "C":
            self._grid_order = None
        else:
            if grid_order is None:
                self._grid_order = [1, mesh[0], mesh[0] * mesh[1]]
            else:
                self._grid_order = grid_order
        self._ir_grid_points = ir_grid_points

        self._gp_ir_index = None

        self._tm = None
        self._tetrahedra_frequencies = None
        self._integration_weights = None
        self._relative_grid_address = None

        self._frequency_points = None
        self._value = None

        self._grid_point_count = 0

        self._prepare()

    def __iter__(self):
        """Define iterator over grid points."""
        return self

    def __next__(self):
        """Peform linear tetrahedron method at a grid point."""
        if self._grid_point_count == len(self._ir_grid_points):
            raise StopIteration
        else:
            gp = self._ir_grid_points[self._grid_point_count]
            self._set_tetrahedra_frequencies(gp)
            for ib, frequencies in enumerate(self._tetrahedra_frequencies):
                self._tm.set_tetrahedra_omegas(frequencies)
                self._tm.run(self._frequency_points, value=self._value)
                iw = self._tm.get_integration_weight()
                self._integration_weights[:, ib] = iw
            self._integration_weights /= np.prod(self._mesh)
            self._grid_point_count += 1
            return self._integration_weights

    def get_integration_weights(self):
        """Return integration weights."""
        return self._integration_weights

    def get_frequency_points(self):
        """Return frequency points."""
        return self._frequency_points

    def set(self, value="I", division_number=201, frequency_points=None, lang="C"):
        """Prepare environment to peform linear tetrahedron method."""
        self._grid_point_count = 0
        self._value = value
        if frequency_points is None:
            max_frequency = np.amax(self._frequencies)
            min_frequency = np.amin(self._frequencies)
            self._frequency_points = np.linspace(
                min_frequency, max_frequency, division_number, dtype="double"
            )
        else:
            self._frequency_points = np.array(frequency_points, dtype="double")

        num_band = self._frequencies.shape[1]
        num_freqs = len(self._frequency_points)
        self._integration_weights = np.zeros((num_freqs, num_band), dtype="double")
        reciprocal_lattice = np.linalg.inv(self._cell.cell)
        self._tm = TetrahedronMethod(reciprocal_lattice, mesh=self._mesh, lang=lang)
        self._relative_grid_address = self._tm.tetrahedra

    def _prepare(self):
        ir_gp_indices = {}
        for i, gp in enumerate(self._ir_grid_points):
            ir_gp_indices[gp] = i

        self._gp_ir_index = np.zeros_like(self._grid_mapping_table)
        for i, gp in enumerate(self._grid_mapping_table):
            self._gp_ir_index[i] = ir_gp_indices[gp]

    def _set_tetrahedra_frequencies(self, gp):
        self._tetrahedra_frequencies = get_tetrahedra_frequencies(
            gp,
            self._mesh,
            self._grid_address,
            self._relative_grid_address,
            self._gp_ir_index,
            self._frequencies,
            grid_order=self._grid_order,
            lang=self._lang,
        )


def get_tetrahedra_frequencies(
    gp,
    mesh,
    grid_address,
    relative_grid_address,
    gp_ir_index,
    frequencies,
    grid_order=None,
    lang="C",
):
    """Return frequencies on the relative_grid_addresses.

    Note
    ----
    This implementation is based on GR-grid.

    Parameters
    ----------
    gp : float
        Grid index
    mesh : ndarray
        Mesh numbers. shape=(3, ), dtype='long'
    grid_address : ndarray
        Grid address in integers. shape=(prod(mesh), 3), dtype='long', order='C'
    relative_grid_addresses : ndarray
        Relative grid addresses from the centre (i.e., gp) shape=(24, 4, 3),
        dtype='long', order='C'
    gp_ir_index : ndarray
        Mapping table from grid index in GR-grid to index corresponding to first
        dimension of frequencies. The ir-grid index is
        range(len(ir-grid-points)). shape=(prod(mesh), ), dtype='long'
    frequencies : ndarray
        Phonon frequences on ir-grid points. shape=(ir-grid-points, num_band)
        dtype='double'
    grid_order : list of int, optional
        This controls how grid addresses are stored either C style or Fortran
        style. This is only valid when lang != 'C'.
    lang : str, 'C' or else, optional
        With 'C', C implementation is used. Otherwise Python implementation
        runs.

    Returns
    -------
    ndarray
        Frequencies at tetheredra tertices. shape=(num_bands, 24, 4),
        dtype='double', order='C'

    """
    if lang == "C":
        try:
            import phonopy._phonopy as phonoc  # noqa F401

            return _get_tetrahedra_frequencies_C(
                gp, mesh, grid_address, relative_grid_address, gp_ir_index, frequencies
            )
        except ImportError:
            return _get_tetrahedra_frequencies_Py(
                gp,
                mesh,
                grid_address,
                relative_grid_address,
                gp_ir_index,
                frequencies,
                grid_order,
            )
    else:
        return _get_tetrahedra_frequencies_Py(
            gp,
            mesh,
            grid_address,
            relative_grid_address,
            gp_ir_index,
            frequencies,
            grid_order,
        )


def _get_tetrahedra_frequencies_C(
    gp, mesh, grid_address, relative_grid_address, gp_ir_index, frequencies
):
    import phonopy._phonopy as phonoc

    t_frequencies = np.zeros((1, frequencies.shape[1], 24, 4), dtype="double")
    phonoc.tetrahedra_frequencies(
        t_frequencies,
        np.array([gp], dtype="long"),
        mesh,
        grid_address,
        gp_ir_index,
        relative_grid_address,
        frequencies,
    )
    return np.array(t_frequencies[0], dtype="double", order="C")


def _get_tetrahedra_frequencies_Py(
    gp, mesh, grid_address, relative_grid_address, gp_ir_index, frequencies, grid_order
):
    t_frequencies = np.zeros((frequencies.shape[1], 24, 4), dtype="double")
    for i, t in enumerate(relative_grid_address):
        address = t + grid_address[gp]
        neighbors = np.dot(address % mesh, grid_order)
        t_frequencies[:, i, :] = frequencies[gp_ir_index[neighbors]].T
    return t_frequencies
