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

from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.harmonic.dynmat_to_fc import (
    DynmatToForceConstants,
    categorize_commensurate_points,
    get_commensurate_points_in_integers,
)
from phonopy.units import AMU, EV, Angstrom, Hbar, Kb, THz, THzToEv, VaspToTHz


def bose_einstein_dist(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1)


class RandomDisplacements(object):
    """Generate random displacements by Canonical ensenmble.

    Note
    ----
    Phonon frequencies are used to calculate phonon occupation number,
    for which phonon frequencies have to be given in THz. Therefore unit
    conversion factor has to be specified at the initialization.

    Imaginary phonon modes are treated so as to have their absolute phonon
    frequencies |omega| and phonon modes having |omega| < cutoff_frequency
    are ignored.

    Attributes
    ----------
    u : ndarray
        Random atomic displacements generated by canonical distribution of
        harmonic oscillator. The unit of distance is Angstrom.
        shape=(number_of_snapshots, supercell_atoms, 3)
        dtype='double', order='C'
    qpoints : ndarray
        Commensurate q-points corresponding to the supercell matrix but not
        all. Only half of the commensurate q-points that are not on the BZ
        boundary and Gamma-points are only taken, because of the symmetry
        of dynamical matrix: omega_q = omega_-q and e_q = e_-q^*.
    frequencies : ndarray
        Phonon frequencies at commensurate q-points as explained above
        qpoints attribute. Both of getter and setter are implemented.
        The aim of this is to modify random displacements by modifying
        frequencies by users.
        shape=(len(qpoints), num_band), dtype='double', order='C'
        where num_band is 3 * number of atoms in primitive cell.
    force_constants : ndarray
        Force constants calculated from phonon frequencies and eigenvectors
        at commensurate q-points as given above qpoints attribute. By this,
        phonon can be calculated with modified phonon frequencies. To
        calculate force constants, run_d2f has to be executed. For example,

            rd = RandomDisplacements(supercell, primitive, force_constants)
            freqs = rd.frequencies
            ... modify freqs by users
            rd.frequencies = freqs
            rd.run(500)  # To get random displacements
            rd.run_d2f()
            fc = rd.force_constants  # To draw phonons with modified freqs

        shape=(superell_atoms, supercell_atoms, 3, 3)
        dtype='double', order='C'

    """

    def __init__(
        self,
        supercell,
        primitive,
        force_constants,
        dist_func=None,
        cutoff_frequency=None,
        factor=VaspToTHz,
    ):
        """

        Parameters
        ----------
        supercell : Supercell
            Supercell.
        primitive : Primitive
            Primitive cell
        force_constants : array_like
            Force constants matrix. See the details at docstring of
            DynamialMatrix.
        dist_func : str or None
            Harmonic oscillator distribution function either by 'quantum'
            or 'classical'. The starndard deviation of normal distribution
            is determined following the choice. Default is None, corresponding
            to 'quantum'.
        cutoff_frequency : float
            Lowest phonon frequency below which frequency the phonon mode
            is treated specially. See _get_sigma. Default is None, which
            means 0.01.
        factor : float
            Phonon frequency unit conversion factor to THz

        """

        if cutoff_frequency is None or cutoff_frequency < 0:
            self._cutoff_frequency = 0.01
        else:
            self._cutoff_frequency = cutoff_frequency
        self._factor = factor
        self._T = None
        self._u = None

        if dist_func is None or dist_func == "quantum":
            self._dist_func = "quantum"
        elif dist_func == "classical":
            self._dist_func = "classical"
        else:
            raise RuntimeError("Either 'quantum' or 'classical' is required.")

        self._unit_conversion = Hbar * EV / AMU / THz / (2 * np.pi) / Angstrom ** 2
        self._unit_conversion_classical = (
            Kb * EV / AMU / (THz * (2 * np.pi)) ** 2 / Angstrom ** 2
        )

        # Dynamical matrix without NAC because of commensurate points only
        self._dynmat = get_dynamical_matrix(force_constants, supercell, primitive)

        self._setup_sampling_qpoints(supercell.cell, primitive.cell)

        s2p = primitive.s2p_map
        p2p = primitive.p2p_map
        self._s2pp = [p2p[i] for i in s2p]
        # Transformation matrix of scaled supercell positions to primitive
        tmat = np.dot(supercell.cell, np.linalg.inv(primitive.cell))
        self._spos = np.dot(self._dynmat.supercell.scaled_positions, tmat)
        self._ppos = self._dynmat.primitive.scaled_positions
        self._lpos = self._spos - self._ppos[self._s2pp]

        self._eigvals_ii = []
        self._eigvecs_ii = []
        self._phase_ii = []
        self._eigvals_ij = []
        self._eigvecs_ij = []
        self._phase_ij = []
        self._prepare()

        # This is set when running run_d2f.
        # The aim is to produce force constants from modified frequencies.
        self._force_constants = None

        # Displacement correlation matrix (nsatom, nsatom, 3, 3)
        self._uu = None
        self._uu_inv = None

    def run(self, T, number_of_snapshots=1, random_seed=None, randn=None):
        """

        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        number_of_snapshots : int
            Number of snapshots to be generated.
        random_seed : int or None, optional
            Random seed passed to np.random.seed. Default is None. Integer
            number has to be positive.
        randn : tuple
            (randn_ii, randn_ij).
            Used for testing purpose for the fixed random numbers of
            np.random.normal that can depends on system.

        """

        np.random.seed(seed=random_seed)

        N = len(self._comm_points)

        # This randn is used only for testing purpose.
        if randn is None:
            randn_ii = None
            randn_ij = None
        else:
            randn_ii = randn[0]
            randn_ij = randn[1]

        u_ii = self._solve_ii(T, number_of_snapshots, randn=randn_ii)
        if self._ij:
            u_ij = self._solve_ij(T, number_of_snapshots, randn=randn_ij)
        else:
            u_ij = 0

        mass = self._dynmat.supercell.masses.reshape(-1, 1)
        u = np.array((u_ii + u_ij) / np.sqrt(mass * N), dtype="double", order="C")
        self._u = u

    @property
    def u(self):
        return self._u

    @property
    def uu(self):
        return self._uu

    @property
    def uu_inv(self):
        return self._uu_inv

    @property
    def frequencies(self):
        if self._ij:
            eigvals = np.vstack((self._eigvals_ii, self._eigvals_ij))
        else:
            eigvals = self._eigvals_ii
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * self._factor
        return np.array(freqs, dtype="double", order="C")

    @frequencies.setter
    def frequencies(self, freqs):
        eigvals = (freqs / self._factor) ** 2
        if len(eigvals) != len(self._eigvals_ii) + len(self._eigvals_ij):
            raise RuntimeError("Dimension of frequencies is wrong.")

        self._eigvals_ii = eigvals[: len(self._eigvals_ii)]
        self._eigvals_ij = eigvals[len(self._eigvals_ii) :]

    @property
    def qpoints(self):
        N = len(self._comm_points)
        return self._comm_points[self._ii + self._ij] / float(N)

    @property
    def force_constants(self):
        return self._force_constants

    def run_d2f(self):
        qpoints, eigvals, eigvecs = self._collect_eigensolutions()
        d2f = DynmatToForceConstants(self._dynmat.primitive, self._dynmat.supercell)
        d2f.commensurate_points = qpoints
        d2f.create_dynamical_matrices(eigvals, eigvecs)
        d2f.run()
        self._force_constants = d2f.force_constants

    def run_correlation_matrix(self, T):
        qpoints, eigvals, eigvecs = self._collect_eigensolutions()
        d2f = DynmatToForceConstants(self._dynmat.primitive, self._dynmat.supercell)
        masses = self._dynmat.supercell.masses
        d2f.commensurate_points = qpoints
        freqs = np.sqrt(np.abs(eigvals)) * self._factor
        conditions = freqs > self._cutoff_frequency
        a = self._get_sigma(eigvals, T)
        a2 = a ** 2
        _a = np.where(conditions, a, 1)
        a2_inv = np.where(conditions, 1 / _a ** 2, 0)

        d2f.create_dynamical_matrices(a2_inv, eigvecs)
        d2f.run()
        self._uu_inv = np.array(d2f.force_constants, dtype="double", order="C")

        d2f.create_dynamical_matrices(a2, eigvecs)
        d2f.run()
        matrix = d2f.force_constants
        for i, m_i in enumerate(masses):
            for j, m_j in enumerate(masses):
                matrix[i, j] /= m_i * m_j
        self._uu = np.array(matrix, dtype="double", order="C")

    def _collect_eigensolutions(self):
        N = len(self._comm_points)

        qpoints = self._comm_points[self._ii] / float(N)
        eigvals = self._eigvals_ii
        eigvecs = []
        # Transform eigenvectors of D-type to those of C-type
        for q, eigvec in zip(qpoints, self._eigvecs_ii):
            Vd = np.repeat(np.exp(-2j * np.pi * np.dot(self._ppos, q)), 3)
            eigvecs.append((Vd * eigvec.T).T)

        if self._ij:
            eigvals = np.vstack((eigvals, self._eigvals_ij, self._eigvals_ij))
            eigvecs = np.vstack((eigvecs, self._eigvecs_ij, self._eigvecs_ij))
            eigvecs[-len(self._ij) :] = eigvecs[-len(self._ij) :].conj()
            qpoints = self._comm_points[self._ii + self._ij * 2] / float(N)
            qpoints[-len(self._ij) :] = -qpoints[-len(self._ij) :]

        return qpoints, eigvals, eigvecs

    def _prepare(self):
        N = len(self._comm_points)
        for q in self._comm_points[self._ii] / float(N):
            self._dynmat.run(q)
            dm = self._C_to_D(self._dynmat.dynamical_matrix, q)
            self._phase_ii.append(
                np.cos(2 * np.pi * np.dot(self._lpos, q)).reshape(-1, 1)
            )
            eigvals, eigvecs = np.linalg.eigh(dm)
            self._eigvals_ii.append(eigvals)
            self._eigvecs_ii.append(eigvecs)

        if self._ij:
            for q in self._comm_points[self._ij] / float(N):
                self._dynmat.run(q)
                dm = self._dynmat.dynamical_matrix
                eigvals, eigvecs = np.linalg.eigh(dm)
                self._eigvals_ij.append(eigvals.real)
                self._eigvecs_ij.append(eigvecs)
                self._phase_ij.append(
                    np.exp(2j * np.pi * np.dot(self._spos, q)).reshape(-1, 1)
                )

    def _C_to_D(self, dm, q):
        """Transform C-type dynamical matrix to D-type

        Taking real part is valid only when q is at Gamma or on BZ boundary,
        i.e., q=G-q and q in BZ are assumed.

        D(q) = (D(q) + D(G-q)) / 2 -> real matrix.

        """

        V = np.repeat(np.exp(2j * np.pi * np.dot(self._ppos, q)), 3)
        dm = ((V * (V.conj() * dm).T).T).real  # C-type to D-type
        return dm

    def _setup_sampling_qpoints(self, slat, plat):
        smat = np.rint(np.dot(slat, np.linalg.inv(plat)).T).astype(int)
        self._comm_points = get_commensurate_points_in_integers(smat)
        self._ii, self._ij = categorize_commensurate_points(self._comm_points)

    def _solve_ii(self, T, number_of_snapshots, randn=None):
        """

        randn parameter is used for the test.

        """
        natom = len(self._dynmat.supercell)
        u = np.zeros((number_of_snapshots, natom, 3), dtype="double")

        shape = (len(self._eigvals_ii), number_of_snapshots, len(self._eigvals_ii[0]))
        if randn is None:
            _randn = np.random.normal(size=shape)
        else:
            _randn = randn
        sigmas = self._get_sigma(self._eigvals_ii, T)
        for norm_dist, sigma, eigvecs, phase in zip(
            _randn, sigmas, self._eigvecs_ii, self._phase_ii
        ):
            u_red = np.dot(norm_dist * sigma, eigvecs.T).reshape(
                number_of_snapshots, -1, 3
            )[:, self._s2pp, :]
            # u_red.shape = (snapshots, satoms, 3)
            # phase.shape = (satoms,)
            u += u_red * phase

        return u

    def _solve_ij(self, T, number_of_snapshots, randn=None):
        """

        randn parameter is used for the test.

        """
        natom = len(self._dynmat.supercell)
        u = np.zeros((number_of_snapshots, natom, 3), dtype="double")
        shape = (
            len(self._eigvals_ij),
            2,
            number_of_snapshots,
            len(self._eigvals_ij[0]),
        )
        if randn is None:
            _randn = np.random.normal(size=shape)
        else:
            _randn = randn
        sigmas = self._get_sigma(self._eigvals_ij, T)
        for norm_dist, sigma, eigvecs, phase in zip(
            _randn, sigmas, self._eigvecs_ij, self._phase_ij
        ):
            u_red = np.dot(norm_dist * sigma, eigvecs.T).reshape(
                2, number_of_snapshots, -1, 3
            )[:, :, self._s2pp, :]
            # u_red.shape = (2, snapshots, satoms, 3)
            # phase.shape = (satoms,)
            u += (u_red[0] * phase).real
            u -= (u_red[1] * phase).imag

        return u * np.sqrt(2)

    def _get_sigma(self, eigvals, T):
        """Returns sigma in sqrt(AMU).Angstrom unit"""
        freqs = np.sqrt(np.abs(eigvals)) * self._factor
        conditions = freqs > self._cutoff_frequency
        freqs = np.where(conditions, freqs, 1)
        if self._dist_func == "classical":
            sigma = np.sqrt(T * self._unit_conversion_classical) / freqs
        else:
            n = bose_einstein_dist(freqs, T)
            sigma = np.sqrt(self._unit_conversion / freqs * (0.5 + n))
        sigma = np.where(conditions, sigma, 0)
        return sigma
