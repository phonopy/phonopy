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

import numpy as np
from phonopy.units import AMU, THzToEv, Kb, EV, Hbar, Angstrom
from phonopy.interface.cif import write_cif_P1


class ThermalMotion(object):
    def __init__(self,
                 iter_mesh,
                 freq_min=None,
                 freq_max=None):
        self._iter_mesh = iter_mesh
        if freq_min is None:
            self._fmin = 0
        else:
            self._fmin = freq_min
        if freq_max is None:
            self._fmax = None
        else:
            self._fmax = freq_max

        masses = iter_mesh.dynamical_matrix.primitive.get_masses()
        self._masses = masses * AMU
        self._masses3 = np.array([[m] * 3 for m in masses]).ravel() * AMU
        self._temperatures = None

    def _get_Q2(self, freq, t):  # freq in THz
        return Hbar * EV / Angstrom ** 2 * (
            (self._get_population(freq, t) + 0.5) / (freq * 1e12 * 2 * np.pi))

    def get_temperatures(self):
        return self._temperatures

    def set_temperature_range(self, t_min=None, t_max=None, t_step=None):
        if t_min is None:
            _t_min = 10
        elif t_min < 0:
            _t_min = 0
        else:
            _t_min = t_min

        if t_max is None:
            _t_max = 1000
        elif t_max > _t_min:
            _t_max = t_max
        else:
            _t_max = _t_min

        if t_step is None:
            _t_step = 10
        elif t_step > 0:
            _t_step = t_step
        else:
            _t_step = 10

        self._temperatures = np.arange(_t_min, _t_max + _t_step / 2.0, _t_step,
                                       dtype='double')

    def set_temperatures(self, temperatures):
        t_array = np.array(temperatures)
        condition = np.logical_not(t_array < 0)
        self._temperatures = np.extract(condition, t_array)

    def _get_population(self, freq, t):  # freq in THz
        """Return phonon population number

        Three types of combinations of array inputs are possible.
        - single freq and single t
        - single freq and len(t) > 1
        - len(freq) > 1 and single t

        """
        condition = t > 1.0
        if type(condition) == bool or type(condition) == np.bool_:
            if condition:
                return 1.0 / (np.exp(freq * THzToEv / (Kb * t)) - 1)
            else:
                return 0.0
        else:
            vals = np.zeros(len(t), dtype='double')
            vals[condition] = 1.0 / (
                np.exp(freq * THzToEv / (Kb * t[condition])) - 1)
            return vals


class ThermalDisplacements(ThermalMotion):
    def __init__(self,
                 iter_mesh,
                 projection_direction=None,
                 freq_min=None,
                 freq_max=None):
        """Calculate mean square displacements

        Parameters
        ----------
        iter_mesh:
            Mesh or IterMesh instance. Grid points must not be reduced by
            symmetry, i.e., IterMesh instance has to be create
            ``is_mesh_symmetry=False``.
        projection_direction:
            Eigenvector projection direction in Cartesian
            coordinates. If None, eigenvector is not projected.
        freq_min:
            Minimum phonon frequency to determine wheather include or not.
        freq_max:
            Maximum phonon frequency to determine wheather include or not.

        """

        ThermalMotion.__init__(self,
                               iter_mesh,
                               freq_min=freq_min,
                               freq_max=freq_max)
        if projection_direction is None:
            self._projection_direction = None
        else:
            self._projection_direction = (projection_direction /
                                          np.linalg.norm(projection_direction))
        self._displacements = None

    def get_thermal_displacements(self):
        return (self._temperatures, self._displacements)

    def run(self):
        if self._projection_direction is not None:
            masses = self._masses
        else:
            masses = self._masses3
        temps = self._temperatures
        disps = np.zeros((len(temps), len(masses)), dtype=float)

        for count, (fs, vecs) in enumerate(self._iter_mesh):
            if self._projection_direction is not None:
                p_vecs = np.dot(
                    vecs.T.reshape(-1, 3),
                    self._projection_direction).reshape(-1, len(masses))
                vecs2 = np.abs(p_vecs) ** 2 / masses
            else:
                vecs2 = (abs(vecs) ** 2).T / masses

            valid_indices = fs > self._fmin
            if self._fmax is not None:
                valid_indices *= fs < self._fmax

            if len(temps) == 1:
                Q2 = self._get_Q2(fs[valid_indices], temps[0])
                disps[0] += np.dot(Q2, vecs2[valid_indices])
            else:
                for f, v2 in zip(fs[valid_indices], vecs2[valid_indices]):
                    disps += np.outer(self._get_Q2(f, temps), v2)

        assert np.prod(self._iter_mesh.mesh_numbers) == count + 1
        self._displacements = disps / (count + 1)

    def write_yaml(self):
        natom = len(self._masses)
        f = open('thermal_displacements.yaml', 'w')
        f.write("# Thermal displacements\n")
        f.write("natom: %5d\n" % (natom))
        f.write("freq_min: %f\n" % self._fmin)

        f.write("thermal_displacements:\n")
        for t, u in zip(self._temperatures, self._displacements):
            f.write("- temperature:   %15.7f\n" % t)
            f.write("  displacements:\n")
            for i, elems in enumerate(np.reshape(u, (natom, -1))):
                f.write("  - [ %10.7f" % elems[0])
                for j in range(len(elems) - 1):
                    f.write(", %10.7f" % elems[j + 1])
                f.write(" ] # atom %d\n" % (i + 1))

    def plot(self, pyplot, is_legend=False):
        xyz = ['x', 'y', 'z']
        for i, u in enumerate(self._displacements.transpose()):
            pyplot.plot(self._temperatures, u,
                        label=("%d-%s" % (i//3 + 1, xyz[i % 3])))

        if is_legend:
            pyplot.legend(loc='upper left')

    def _project_eigenvectors(self):
        """Eigenvectors are projected along Cartesian direction"""

        self._p_eigenvectors = []
        for vecs_q in self._eigenvectors:
            p_vecs_q = []
            for vecs in vecs_q.T:
                p_vecs_q.append(np.dot(vecs.reshape(-1, 3),
                                       self._projection_direction))
            self._p_eigenvectors.append(np.transpose(p_vecs_q))
        self._p_eigenvectors = np.array(self._p_eigenvectors)


class ThermalDisplacementMatrices(ThermalMotion):
    def __init__(self,
                 iter_mesh,
                 freq_min=None,
                 freq_max=None,
                 lattice=None):
        """Calculate mean square displacement matrices

        Parameters
        ----------
        iter_mesh:
            Mesh or IterMesh instance. Grid points must not be reduced by
            symmetry, i.e., IterMesh instance has to be create
            ``is_mesh_symmetry=False``.
        freq_min: float
            Minimum phonon frequency to determine wheather include or not.
        freq_max: float
            Maximum phonon frequency to determine wheather include or not.
        lattice: array_like
            Lattice parameters (column vectors) in real space
            dtype='double'
            shape=(3, 3)

        """

        ThermalMotion.__init__(self,
                               iter_mesh,
                               freq_min=freq_min,
                               freq_max=freq_max)
        self._disp_matrices = None
        self._disp_matrices_cif = None

        if lattice is not None:
            A = lattice
            N = np.diag([np.linalg.norm(x) for x in np.linalg.inv(A)])
            self._ANinv = np.linalg.inv(np.dot(A, N))
        else:
            self._ANinv = None

    def get_thermal_displacement_matrices(self):
        return (self._temperatures, self._disp_matrices)

    def run(self, np_overflow=None):
        """

        Parameters
        ----------
        np_overflow: str or None
            Switch of error handling of numpy. 'raise' to see which phonon it
            is.

        """

        np.seterr(over=np_overflow)
        self._get_disp_matrices()
        np.seterr(over=None)

        if self._ANinv is not None:
            self._disp_matrices_cif = np.zeros(self._disp_matrices.shape,
                                               dtype='double')
            for i, matrices in enumerate(self._disp_matrices):
                for j, mat in enumerate(matrices):
                    mat_cif = np.dot(np.dot(self._ANinv, mat.real),
                                     self._ANinv.T)
                    self._disp_matrices_cif[i, j] = mat_cif

        self._get_disp_matrices()

    def _get_disp_matrices(self):
        dtype_complex = "c%d" % (np.dtype('double').itemsize * 2)
        disps = np.zeros((len(self._temperatures), len(self._masses),
                          3, 3), dtype=dtype_complex)
        for count, (freqs, eigvecs) in enumerate(self._iter_mesh):
            valid_indices = freqs > self._fmin
            if self._fmax is not None:
                valid_indices *= freqs < self._fmax
            for i_band, (f, vec) in enumerate(
                    zip(freqs[valid_indices], (eigvecs.T)[valid_indices])):
                c = np.zeros((len(self._masses), 3, 3),
                             dtype=dtype_complex, order='C')
                for i, (v, m) in enumerate(
                        zip(vec.reshape(-1, 3), self._masses)):
                    c[i] = np.outer(v, v.conj()) / m

                # for i, t in enumerate(self._temperatures):
                #     try:
                #         disps[i] += self._get_Q2(f, t) * np.array(c)
                #     except FloatingPointError as e:
                #         # Probably, overflow in exp(freq / (kB * T))
                #         print("%s: T=%.1f freq=%.2f (band #%d)" %
                #               (e, t, f, i_band))
                try:
                    Q2 = self._get_Q2(f, self._temperatures)
                    disps += Q2[:, None, None, None] * c[None, :, :, :]
                except FloatingPointError as e:
                    # Probably, overflow in exp(freq / (kB * T))
                    print("%s: freq=%.2f (band #%d)" % (e, f, i_band))

        assert np.prod(self._iter_mesh.mesh_numbers) == count + 1
        self._disp_matrices = disps / (count + 1)

    def write_cif(self, cell, temperature_index):
        write_cif_P1(cell,
                     U_cif=self._disp_matrices_cif[temperature_index],
                     filename="tdispmat.cif")

    def write_yaml(self):
        natom = len(self._masses)
        lines = []

        with open('thermal_displacement_matrices.yaml', 'w') as w:
            lines.append("# Thermal displacement_matrices")
            lines.append("natom: %5d" % (natom))
            lines.append("freq_min: %f" % self._fmin)
            lines.append("thermal_displacement_matrices:")
            for i, t in enumerate(self._temperatures):
                matrices = self._disp_matrices[i]
                lines.append("- temperature:   %15.7f" % t)
                lines.append("  displacement_matrices:")
                for j, mat in enumerate(matrices):
                    # For checking imaginary part that should be zero
                    # lines.append("  - # atom %d" % (i + 1))
                    # for v in mat:
                    #     lines.append(
                    #         "    [ %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f ]"
                    #         % (tuple(v.real) + tuple(v.imag)))
                    m = mat.real
                    lines.append(
                        ("  - [ " + "%8.5f, " * 5 + "%8.5f ] # atom %d") %
                        (m[0, 0], m[1, 1], m[2, 2],
                         m[1, 2], m[0, 2], m[0, 1], j + 1))
                if self._ANinv is not None:
                    matrices_cif = self._disp_matrices_cif[i]
                    lines.append("  displacement_matrices_cif:")
                    for j, mat_cif in enumerate(matrices_cif):
                        m = mat_cif
                        lines.append(
                            ("  - [ " + "%8.5f, " * 5 + "%8.5f ] # atom %d") %
                            (m[0, 0], m[1, 1], m[2, 2],
                             m[1, 2], m[0, 2], m[0, 1], j + 1))
            w.write("\n".join(lines))


class ThermalDistances(ThermalMotion):
    def __init__(self,
                 frequencies,
                 eigenvectors,
                 supercell,
                 primitive,
                 qpoints,
                 symprec=1e-5,
                 freq_min=None):

        self._primitive = primitive
        self._supercell = supercell
        self._qpoints = qpoints
        self._symprec = symprec
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors

        ThermalMotion.__init__(self,
                               primitive.get_masses(),
                               freq_min=freq_min)

        self._p_eigenvectors = None
        self._distances = None

    def _get_cross(self, v, delta_r, q, atom1, atom2):
        phase = np.exp(2j * np.pi * np.dot(delta_r, q))
        cross_val = v[atom1]*phase*v[atom2].conjugate()
        return -2*(cross_val).real

    def run(self, atom_pairs):
        s2p = self._primitive.get_supercell_to_primitive_map()
        p2p = self._primitive.get_primitive_to_primitive_map()
        dists = np.zeros((len(self._temperatures), len(atom_pairs)),
                         dtype=float)
        for i, (atom1, atom2) in enumerate(atom_pairs):
            patom1 = p2p[s2p[atom1]]
            patom2 = p2p[s2p[atom2]]
            spos = self._supercell.get_scaled_positions()

            # This may be wrong.
            delta_r = get_smallest_vectors(
                self._supercell.get_cell(),
                spos[[atom2]],
                spos[[atom1]],
                symprec=self._symprec)[0]

            # This is no longer implemented.
            self._project_eigenvectors(delta_r, self._primitive.get_cell())

            for freqs, vecs, q in zip(self._frequencies,
                                      self._p_eigenvectors,
                                      self._qpoints):
                c_cross = 1.0 / np.sqrt(self._masses[patom1] *
                                        self._masses[patom2])
                c1 = 1.0 / self._masses[patom1]
                c2 = 1.0 / self._masses[patom2]

                for f, v in zip(freqs, vecs.T):
                    cross_term = self._get_cross(v, delta_r, q, patom1, patom2)
                    v2 = abs(v)**2
                    if f > self._fmin:
                        for j, t in enumerate(self._temperatures):
                            dists[j, i] += self._get_Q2(f, t) * (
                                v2[patom1] * c1 +
                                cross_term * c_cross + v2[patom2] * c2)

        self._atom_pairs = atom_pairs
        self._distances = dists / len(self._frequencies)

    def write_yaml(self):
        natom = len(self._masses)
        f = open('thermal_distances.yaml', 'w')
        f.write("natom: %5d\n" % (natom))
        f.write("freq_min: %f\n" % self._fmin)

        f.write("thermal_distances:\n")
        for t, u in zip(self._temperatures, self._distances):
            f.write("- temperature:   %15.7f\n" % t)
            f.write("  distance:\n")
            for i, (atom1, atom2) in enumerate(self._atom_pairs):
                f.write("  - %10.7f # atom pair %d-%d\n"
                        % (u[i], atom1 + 1, atom2 + 1))
