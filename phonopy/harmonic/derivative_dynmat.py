"""Calculation of derivative of dynamical matrix with respect to q."""

# Copyright (C) 2013 Atsushi Togo
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

import warnings
from typing import Union

import numpy as np

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.structure.cells import sparse_to_dense_svecs


class DerivativeOfDynamicalMatrix:
    """Compute analytical derivative of dynamical matrix.

    This can be used dynamical matrix without NAC or with Wang-NAC.
    Gonze-Lee NAC doesn't support analytical derivative.

    Attributes
    ----------
    d_dynamical_matrix : ndarray

    """

    Q_DIRECTION_TOLERANCE = 1e-5

    def __init__(self, dynamical_matrix: Union[DynamicalMatrix, DynamicalMatrixNAC]):
        """Init method.

        Parameters
        ----------
        dynamical_matrix : DynamicalMatrix
            A DynamicalMatrix instance.

        """
        self._dynmat = dynamical_matrix
        self._force_constants = self._dynmat.force_constants
        self._scell = self._dynmat.supercell
        self._pcell = self._dynmat.primitive

        dtype = "long"
        self._p2s_map = np.array(self._pcell.p2s_map, dtype=dtype)
        self._s2p_map = np.array(self._pcell.s2p_map, dtype=dtype)
        p2p_map = self._pcell.p2p_map
        self._s2pp_map = np.array(
            [p2p_map[self._s2p_map[i]] for i in range(len(self._s2p_map))], dtype=dtype
        )

        svecs, multi = self._pcell.get_smallest_vectors()
        if self._pcell.store_dense_svecs:
            self._svecs = svecs
            self._multi = multi
        else:
            self._svecs, self._multi = sparse_to_dense_svecs(svecs, multi)

        self._ddm = None

        # Derivative order=2 can work only within the following conditions:
        # 1. Second derivative of NAC is not considered.
        # 2. Python implementation
        self._derivative_order = None

    def run(self, q, q_direction=None, lang="C"):
        """Run at q."""
        if self._derivative_order is not None or lang != "C":
            self._run_py(q, q_direction=q_direction)
        else:
            self._run_c(q, q_direction=q_direction)

    def set_derivative_order(self, order):
        """Set order of derivative."""
        if order == 1 or order == 2:
            self._derivative_order = order
        else:
            print("Error: derivative order has to be 1 or 2")

    @property
    def d_dynamical_matrix(self):
        """Return derivative of dynamical matrix.

        Returns
        -------
        ndarray
            Derivative of dynamical matrix with respect to q.
            shape=(3, num_patom * 3, num_patom * 3),
            dtype="c%d" % (np.dtype('double').itemsize * 2)

        """
        return self._ddm

    def get_derivative_of_dynamical_matrix(self):
        """Return derivative of dynamical matrix."""
        warnings.warn(
            "DerivativeOfDynamicalMatrix.get_derivative_of_dynamical_matrix() is "
            "deprecated. Use d_dynamical_matrix attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.d_dynamical_matrix

    def _run_c(self, q, q_direction=None):
        import phonopy._phonopy as phonoc

        num_patom = len(self._p2s_map)
        fc = self._force_constants
        ddm = np.zeros(
            (3, num_patom * 3, num_patom * 3),
            dtype=("c%d" % (np.dtype("double").itemsize * 2)),
        )
        reclat = np.array(np.linalg.inv(self._pcell.cell), dtype="double", order="C")

        is_nac = False
        is_nac_q_zero = True
        born = np.zeros(9)  # dummy value
        dielectric = np.zeros(9)  # dummy value
        nac_factor = 0  # dummy value
        q_dir = np.zeros(3)  # dummy value

        if isinstance(self._dynmat, DynamicalMatrixNAC):
            is_nac = True
            born = self._dynmat.born
            dielectric = self._dynmat.dielectric_constant
            nac_factor = self._dynmat.nac_factor
            if q_direction is None:
                q_norm = np.linalg.norm(reclat @ q)
                if q_norm < self.Q_DIRECTION_TOLERANCE:
                    is_nac = False
                else:
                    is_nac_q_zero = True
            else:
                q_dir = np.array(q_direction, dtype="double")
                is_nac_q_zero = False

        if fc.shape[0] == fc.shape[1]:  # full fc
            phonoc.derivative_dynmat(
                ddm.view(dtype="double"),
                fc,
                np.array(q, dtype="double"),
                np.array(self._pcell.cell.T, dtype="double", order="C"),
                reclat,
                self._svecs,
                self._multi,
                self._pcell.masses,
                self._s2p_map,
                self._p2s_map,
                nac_factor,
                born,
                dielectric,
                q_dir,
                is_nac * 1,
                is_nac_q_zero * 1,
                self._dynmat.use_openmp * 1,
            )
        else:
            phonoc.derivative_dynmat(
                ddm.view(dtype="double"),
                fc,
                np.array(q, dtype="double"),
                np.array(self._pcell.cell.T, dtype="double", order="C"),
                np.array(np.linalg.inv(self._pcell.cell), dtype="double", order="C"),
                self._svecs,
                self._multi,
                self._pcell.masses,
                self._s2pp_map,
                np.arange(len(self._p2s_map), dtype="long"),
                nac_factor,
                born,
                dielectric,
                q_dir,
                is_nac * 1,
                is_nac_q_zero * 1,
                self._dynmat.use_openmp * 1,
            )

        self._ddm = ddm

    def _run_py(self, q, q_direction=None):
        """Run in python.

        This works only for full-FC.

        """
        if isinstance(self._dynmat, DynamicalMatrixNAC):
            if q_direction is None:
                fc_nac = self._nac(q)
                d_nac = self._d_nac(q)
            else:
                fc_nac = self._nac(q_direction)
                d_nac = self._d_nac(q_direction)

        fc = self._force_constants
        assert fc.shape[0] == fc.shape[1]
        vecs = self._svecs
        multiplicity = self._multi
        num_patom = len(self._p2s_map)
        num_satom = len(self._s2p_map)

        if self._derivative_order == 2:
            num_elem = 6
        else:
            num_elem = 3

        itemsize = np.dtype("double").itemsize
        ddm = np.zeros(
            (num_elem, 3 * num_patom, 3 * num_patom), dtype=("c%d" % (itemsize * 2))
        )

        for i, j in list(np.ndindex(num_patom, num_patom)):
            s_i = self._p2s_map[i]
            s_j = self._p2s_map[j]
            mass = np.sqrt(self._pcell.masses[i] * self._pcell.masses[j])
            ddm_local = np.zeros((num_elem, 3, 3), dtype=("c%d" % (itemsize * 2)))

            for k in range(num_satom):
                if s_j != self._s2p_map[k]:
                    continue

                multi = multiplicity[k, i]
                vecs_multi = vecs[multi[1] : multi[1] + multi[0]]
                phase_multi = np.exp(
                    [np.vdot(vec, q) * 2j * np.pi for vec in vecs_multi]
                )
                vecs_multi_cart = np.dot(vecs_multi, self._pcell.cell)
                coef_order1 = 2j * np.pi * vecs_multi_cart
                if self._derivative_order == 2:
                    coef_order2 = [np.outer(co1, co1) for co1 in coef_order1]
                    coef = np.array(
                        [co2.ravel()[[0, 4, 8, 5, 2, 1]] for co2 in coef_order2]
                    )
                else:
                    coef = coef_order1

                if isinstance(self._dynmat, DynamicalMatrixNAC):
                    fc_elem = fc[s_i, k] + fc_nac[i, j]
                else:
                    fc_elem = fc[s_i, k]

                for ll in range(num_elem):
                    ddm_elem = fc_elem * (coef[:, ll] * phase_multi).sum()
                    if (
                        isinstance(self._dynmat, DynamicalMatrixNAC)
                        and not self._derivative_order == 2
                    ):
                        ddm_elem += d_nac[ll, i, j] * phase_multi.sum()

                    ddm_local[ll] += ddm_elem / mass / multi[0]

            ddm[:, (i * 3) : (i * 3 + 3), (j * 3) : (j * 3 + 3)] = ddm_local

        # Impose Hermite condition
        self._ddm = np.array([(ddm[i] + ddm[i].conj().T) / 2 for i in range(num_elem)])

    def _nac(self, q_direction):
        """nac_term = (A1 (x) A2) / B * coef."""
        num_atom = self._pcell.get_number_of_atoms()
        nac_q = np.zeros((num_atom, num_atom, 3, 3), dtype="double")
        if (np.abs(q_direction) < 1e-5).all():
            return nac_q

        rec_lat = np.linalg.inv(self._pcell.get_cell())
        nac_factor = self._dynmat.get_nac_factor()
        Z = self._dynmat.get_born_effective_charges()
        e = self._dynmat.get_dielectric_constant()
        q = np.dot(rec_lat, q_direction)

        B = self._B(e, q)
        for i in range(num_atom):
            A_i = self._A(q, Z, i)
            for j in range(num_atom):
                A_j = self._A(q, Z, j)
                nac_q[i, j] = np.outer(A_i, A_j) / B

        num_satom = self._scell.get_number_of_atoms()
        N = num_satom // num_atom

        return nac_q * nac_factor / N

    def _d_nac(self, q_direction):
        num_atom = self._pcell.get_number_of_atoms()
        d_nac_q = np.zeros((3, num_atom, num_atom, 3, 3), dtype="double")
        if (np.abs(q_direction) < 1e-5).all():
            return d_nac_q

        rec_lat = np.linalg.inv(self._pcell.get_cell())
        nac_factor = self._dynmat.get_nac_factor()
        Z = self._dynmat.get_born_effective_charges()
        e = self._dynmat.get_dielectric_constant()
        q = np.dot(rec_lat, q_direction)

        B = self._B(e, q)
        for xyz in range(3):
            dB = self._dB(e, q, xyz)
            for i in range(num_atom):
                A_i = self._A(q, Z, i)
                dA_i = self._dA(Z, i, xyz)
                for j in range(num_atom):
                    A_j = self._A(q, Z, j)
                    dA_j = self._dA(Z, j, xyz)
                    d_nac_q[xyz, i, j] = (
                        np.outer(dA_i, A_j) + np.outer(A_i, dA_j)
                    ) / B - np.outer(A_i, A_j) * dB / B**2

        num_satom = self._scell.get_number_of_atoms()
        N = num_satom // num_atom
        return d_nac_q * nac_factor / N

    def _A(self, q, Z, atom_num):
        return np.dot(q, Z[atom_num])

    def _B(self, epsilon, q):
        return np.dot(q, np.dot(epsilon, q))

    def _dA(self, Z, atom_num, xyz):
        return Z[atom_num, xyz, :]

    def _dB(self, epsilon, q, xyz):
        e = epsilon
        return np.dot(e[xyz], q) * 2
