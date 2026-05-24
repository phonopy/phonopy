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

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy._lang import log_dispatch
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    DynamicalMatrixGL,
    DynamicalMatrixNAC,
    DynamicalMatrixWang,
)
from phonopy.structure.cells import sparse_to_dense_svecs


class DerivativeOfDynamicalMatrix:
    """Compute analytical derivative of the dynamical matrix.

    Backend selection follows ``DynamicalMatrix``:

    - C / Rust paths cover no-NAC and Wang NAC.  When given a
      ``DynamicalMatrixGL`` instance, the compiled paths silently fall
      through to the Wang-style kernel; use ``force_python=True`` on
      ``run()`` (or invoke from a context that does) for the correct
      Gonze-Lee derivative until a compiled kernel lands.
    - The Python path covers no-NAC, Wang NAC, and Gonze-Lee NAC.  The
      Gonze-Lee branch evaluates the analytic dipole-dipole derivative
      (see ``dDdq/main.tex``) on top of the short-range part.
    - Derivative order 2 is supported only by the Python path and only
      for no-NAC and Wang NAC.

    Attributes
    ----------
    d_dynamical_matrix : ndarray

    """

    Q_DIRECTION_TOLERANCE = 1e-5

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        lang: Literal["C", "Rust"] | None = None,
    ) -> None:
        """Init method.

        Parameters
        ----------
        dynamical_matrix : DynamicalMatrix
            A DynamicalMatrix instance.
        lang : Literal["C", "Rust"], optional
            Backend implementation.  When None (default) the value is
            inherited from ``dynamical_matrix.lang``; pass an explicit
            string to override.

        """
        self._dynmat = dynamical_matrix
        self._lang: Literal["C", "Rust"] = (
            lang if lang is not None else dynamical_matrix.lang
        )
        self._force_constants = self._dynmat.force_constants
        self._scell = self._dynmat.supercell
        self._pcell = self._dynmat.primitive

        dtype = "int64"
        self._p2s_map: NDArray[np.int64] = np.array(self._pcell.p2s_map, dtype=dtype)
        self._s2p_map: NDArray[np.int64] = np.array(self._pcell.s2p_map, dtype=dtype)
        p2p_map = self._pcell.p2p_map
        self._s2pp_map: NDArray[np.int64] = np.array(
            [p2p_map[self._s2p_map[i]] for i in range(len(self._s2p_map))], dtype=dtype
        )

        svecs, multi = self._pcell.get_smallest_vectors()
        self._svecs: NDArray[np.double]
        self._multi: NDArray[np.int64]
        if self._pcell.store_dense_svecs:
            self._svecs = svecs
            self._multi = multi
        else:
            self._svecs, self._multi = sparse_to_dense_svecs(svecs, multi)

        self._ddm: NDArray[np.cdouble] | None = None

        # Derivative order=2 can work only within the following conditions:
        # 1. Second derivative of NAC is not considered.
        # 2. Python implementation
        self._derivative_order: int | None = None

    def run(
        self,
        q: Sequence[float] | NDArray[np.double],
        q_direction: Sequence[float] | NDArray[np.double] | None = None,
        force_python: bool = False,
    ) -> None:
        """Run at q.

        Parameters
        ----------
        q : array_like
            Reduced q-point coordinates.
        q_direction : array_like or None, optional
            Reduced q-direction used to evaluate the non-analytical term at
            Gamma.  Ignored when ``self._dynmat`` has no NAC.
        force_python : bool, optional, default=False
            Force the Python implementation regardless of ``self._lang``.
            Mainly intended for testing and for derivative orders > 1
            (which only the Python path supports).

        """
        _q = np.array(q, dtype="double")
        if q_direction is None:
            _q_direction = None
        else:
            _q_direction = np.array(q_direction, dtype="double")
        if force_python or self._derivative_order is not None:
            self._run_py(_q, q_direction=_q_direction)
        else:
            self._run_compiled(_q, q_direction=_q_direction)

    def set_derivative_order(self, order: int) -> None:
        """Set order of derivative (must be 1 or 2)."""
        if order not in (1, 2):
            raise ValueError(f"derivative order must be 1 or 2 (got {order!r})")
        self._derivative_order = order

    @property
    def d_dynamical_matrix(self) -> NDArray[np.cdouble] | None:
        """Return derivative of dynamical matrix.

        Returns
        -------
        ndarray or None
            Derivative of dynamical matrix with respect to q.
            shape=(3, num_patom * 3, num_patom * 3),
            dtype="c%d" % (np.dtype('double').itemsize * 2)

        """
        return self._ddm

    def _run_compiled(
        self, q: NDArray[np.double], q_direction: NDArray[np.double] | None = None
    ) -> None:
        """Dispatch to the C or Rust backend based on ``self._lang``."""
        if self._lang == "Rust":
            self._run_rust(q, q_direction=q_direction)
        else:
            self._run_c(q, q_direction=q_direction)

    def _run_c(
        self, q: NDArray[np.double], q_direction: NDArray[np.double] | None = None
    ) -> None:
        """Run the derivative through the C kernel."""
        import phonopy._phonopy as phonoc  # type: ignore[import-untyped]

        log_dispatch("C", "DerivativeOfDynamicalMatrix._run_c")

        num_patom = len(self._p2s_map)
        fc = self._force_constants
        ddm = np.zeros(
            (3, num_patom * 3, num_patom * 3),
            dtype=("c%d" % (np.dtype("double").itemsize * 2)),
        )
        reclat = np.array(np.linalg.inv(self._pcell.cell), dtype="double", order="C")

        is_nac = False
        is_nac_q_zero = True
        born: NDArray[np.double] = np.zeros(9, dtype="double")
        dielectric: NDArray[np.double] = np.zeros(9, dtype="double")
        nac_factor: float = 0.0
        q_dir: NDArray[np.double] = np.zeros(3, dtype="double")

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

        # full-FC or compact-FC
        if fc.shape[0] == fc.shape[1]:  # type: ignore
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
                np.arange(len(self._p2s_map), dtype="int64"),
                nac_factor,
                born,
                dielectric,
                q_dir,
                is_nac * 1,
                is_nac_q_zero * 1,
                self._dynmat.use_openmp * 1,
            )

        self._ddm = ddm

    def _run_rust(
        self, q: NDArray[np.double], q_direction: NDArray[np.double] | None = None
    ) -> None:
        """Run the derivative through the Rust kernel (phonors)."""
        import phonors  # type: ignore[import-untyped]

        log_dispatch("Rust", "DerivativeOfDynamicalMatrix._run_rust")

        num_patom = len(self._p2s_map)
        fc = self._force_constants
        ddm = np.zeros(
            (3, num_patom * 3, num_patom * 3),
            dtype=("c%d" % (np.dtype("double").itemsize * 2)),
        )
        reclat = np.array(np.linalg.inv(self._pcell.cell), dtype="double", order="C")
        lattice = np.array(self._pcell.cell.T, dtype="double", order="C")

        born: NDArray[np.double] | None = None
        dielectric: NDArray[np.double] | None = None
        nac_factor = 0.0
        q_dir: NDArray[np.double] | None = None
        is_nac = False
        is_nac_q_zero = True

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

        # Full fc has shape (n_satom, n_satom, 3, 3); compact fc has
        # (n_patom, n_satom, 3, 3).  ``s2_for_inner`` matches the inner
        # supercell-to-primitive comparison in the kernel; ``fc_row``
        # selects the row index into ``fc``.
        if fc.shape[0] == fc.shape[1]:  # type: ignore
            s2_for_inner = self._s2p_map
            fc_row = self._p2s_map
        else:
            s2_for_inner = self._s2pp_map
            fc_row = np.arange(len(self._p2s_map), dtype="int64")

        phonors.derivative_dynmat_at_q(
            ddm,
            fc,
            np.array(q, dtype="double"),
            lattice,
            reclat,
            self._svecs,
            self._multi,
            self._pcell.masses,
            s2_for_inner,
            fc_row,
            born=born if is_nac else None,
            dielectric=dielectric if is_nac else None,
            q_direction=q_dir if (is_nac and not is_nac_q_zero) else None,
            nac_factor=nac_factor,
        )

        self._ddm = ddm

    def _run_py(
        self, q: NDArray[np.double], q_direction: NDArray[np.double] | None = None
    ) -> None:
        """Run in Python.

        Supports no-NAC, Wang NAC, and Gonze-Lee NAC.  Both full and
        compact force constants are supported.  For Gonze-Lee NAC, the
        short-range force constants are used for the real-space loop
        and the analytic dipole-dipole derivative is added on top (see
        ``_gonze_lee_d_recip_dipole_dipole``).

        Derivative order 2 is supported only for no-NAC and Wang NAC.

        """
        fc, fc_nac, d_fc_nac, d_dd = self._prepare_py_inputs(q, q_direction)

        is_full_fc = fc.shape[0] == fc.shape[1]
        vecs = self._svecs
        multiplicity = self._multi
        num_patom = len(self._p2s_map)
        num_satom = len(self._s2p_map)
        num_elem = 6 if self._derivative_order == 2 else 3

        ddm = np.zeros((num_elem, 3 * num_patom, 3 * num_patom), dtype=np.cdouble)

        for i, j in np.ndindex(num_patom, num_patom):
            s_i = self._p2s_map[i]
            s_j = self._p2s_map[j]
            fc_row = s_i if is_full_fc else i
            mass = np.sqrt(self._pcell.masses[i] * self._pcell.masses[j])
            ddm_local = np.zeros((num_elem, 3, 3), dtype=np.cdouble)

            for k in range(num_satom):
                if s_j != self._s2p_map[k]:
                    continue

                multi = multiplicity[k, i]
                vecs_multi = vecs[multi[1] : multi[1] + multi[0]]
                phase_multi = np.exp(2j * np.pi * (vecs_multi @ q))
                vecs_multi_cart = vecs_multi @ self._pcell.cell
                coef = self._derivative_coef(vecs_multi_cart, num_elem)

                fc_elem = fc[fc_row, k]
                if fc_nac is not None:
                    # Wang: add NAC term to FC before differentiating.
                    fc_elem = fc_elem + fc_nac[i, j]

                # Sum over multi vectors at once: gives (num_elem,) scalars.
                scalars = (coef.T @ phase_multi) / (mass * multi[0])
                ddm_local += scalars[:, None, None] * fc_elem

                if d_fc_nac is not None and num_elem == 3:
                    # Wang: add the d/dq of the NAC correction.
                    ddm_local += d_fc_nac[:, i, j] * (
                        phase_multi.sum() / (mass * multi[0])
                    )

            ddm[:, (i * 3) : (i * 3 + 3), (j * 3) : (j * 3 + 3)] = ddm_local

        if d_dd is not None:
            # Gonze-Lee: add the analytic dipole-dipole derivative on top of d D_SR/dq.
            ddm += d_dd

        # Impose Hermite condition.
        self._ddm = (ddm + ddm.conj().transpose(0, 2, 1)) / 2

    def _prepare_py_inputs(
        self,
        q: NDArray[np.double],
        q_direction: NDArray[np.double] | None,
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.cdouble] | None,
    ]:
        """Pick the FC and prepare NAC derivative terms for ``_run_py``.

        Returns ``(fc, fc_nac, d_fc_nac, d_dd)``:

        - ``fc``: force constants to use in the real-space loop
          (short-range FC for Gonze-Lee, original FC otherwise).
        - ``fc_nac``: Wang NAC additive FC contribution at q, or None.
        - ``d_fc_nac``: Wang NAC d/dq contribution, or None.
        - ``d_dd``: Gonze-Lee analytic dipole-dipole derivative
          (already shaped ``(3, 3*num_patom, 3*num_patom)``), or None.

        """
        if isinstance(self._dynmat, DynamicalMatrixGL):
            # Gonze-Lee: short-range FC + analytic Ewald derivative.
            if self._derivative_order == 2:
                raise NotImplementedError(
                    "Derivative order 2 is not implemented for Gonze-Lee NAC."
                )
            dm_gl = self._dynmat
            if dm_gl.short_range_force_constants is None:
                dm_gl.make_Gonze_nac_dataset()
            assert dm_gl.short_range_force_constants is not None
            fc = dm_gl.short_range_force_constants
            d_dd = self._gonze_lee_d_recip_dipole_dipole(q, q_direction)
            return fc, None, None, d_dd
        if isinstance(self._dynmat, DynamicalMatrixWang):
            # Wang: full FC + (A_i (x) A_j)/B NAC term and its derivative.
            q_for_nac = q if q_direction is None else q_direction
            fc_nac = self._wang_nac(q_for_nac)
            d_fc_nac = self._d_wang_nac(q_for_nac)
            return self._force_constants, fc_nac, d_fc_nac, None
        if isinstance(self._dynmat, DynamicalMatrixNAC):
            raise NotImplementedError(
                "Only Wang and Gonze-Lee NAC are implemented in the "
                "Python derivative path."
            )
        return self._force_constants, None, None, None

    @staticmethod
    def _derivative_coef(
        vecs_cart: NDArray[np.double], num_elem: int
    ) -> NDArray[np.cdouble]:
        """Return per-multi-vector derivative coefficients.

        Shape ``(M, num_elem)`` where M is the number of equivalent
        shortest vectors.  For order 1, ``coef[m, l] = 2*pi*i * vec[m, l]``.
        For order 2, the upper-triangular outer-product
        ``(2*pi*i)^2 * vec ⊗ vec`` is packed in the order ``[xx, yy, zz,
        yz, xz, xy]``.

        """
        coef1 = 2j * np.pi * vecs_cart
        if num_elem == 3:
            return coef1
        outer = coef1[:, :, None] * coef1[:, None, :]  # (M, 3, 3)
        return outer.reshape(len(coef1), 9)[:, [0, 4, 8, 5, 2, 1]]

    # --- Gonze-Lee NAC: analytical dD_DD/dq from the Ewald sum. ---
    # See dDdq/main.tex for the derivation; the formula is
    #
    #   d D_DD[k,a; k',a']/d q_mu =
    #     factor * (1/sqrt(m_k m_k')) *
    #     sum_{beta,beta'} Z*[k,beta,a] Z*[k',beta',a'] *
    #     sum_{Q=G+q} T_mu(Q;beta,beta') * exp(-B(Q)/(4 Lambda^2)) *
    #     exp(i G . (R[k] - R[k'])),
    #
    # where
    #   v_mu(Q) = (eps Q)_mu = sum_g eps[mu,g] Q[g],
    #   B(Q)    = Q^T eps Q,
    #   K_mu(Q) = v_mu(Q) * (2/B^2 + 1/(2 Lambda^2 B)),
    #   T_mu(Q;beta,beta') =
    #     (delta_{mu,beta} Q[beta'] + Q[beta] delta_{mu,beta'}) / B
    #     - Q[beta] Q[beta'] K_mu(Q).
    #
    # The phase exp(i G . (R[k] - R[k'])) depends on G only and is not
    # differentiated.  The translational-invariance correction
    # (Eq. (52) of the review) is q-independent and contributes zero.

    def _gonze_lee_d_recip_dipole_dipole(
        self,
        q_red: NDArray[np.double],
        q_direction: NDArray[np.double] | None,
    ) -> NDArray[np.cdouble]:
        """Return analytical d/dq of the reciprocal dipole-dipole term.

        Mass-weighted, summed over G, ready to be added to the
        derivative of the short-range part.

        Parameters
        ----------
        q_red : ndarray
            Reduced q-point coordinates.
        q_direction : ndarray or None
            Reduced q-direction used at q == Gamma for the G=0 term.
            Outside Gamma it is ignored.

        Returns
        -------
        ndarray
            shape=(3, num_patom * 3, num_patom * 3), dtype=cdouble.

        """
        assert isinstance(self._dynmat, DynamicalMatrixGL)
        dm = self._dynmat

        pos = self._pcell.positions  # (num_atom, 3) Cartesian
        num_atom = len(pos)
        rec_lat = np.linalg.inv(self._pcell.cell)
        q_cart = np.array(np.dot(q_red, rec_lat.T), dtype="double")

        G_list: NDArray[np.double] = dm._G_list
        Lambda: float = dm._Lambda
        born: NDArray[np.double] = dm.born  # (num_atom, 3, 3): Z*[k, beta, alpha]
        eps: NDArray[np.double] = dm.dielectric_constant  # (3, 3)
        factor: float = dm.nac_factor  # unit_conversion * 4*pi / V_c

        is_q_gamma = np.linalg.norm(q_cart) < self.Q_DIRECTION_TOLERANCE
        if is_q_gamma and q_direction is not None:
            q_dir_cart = np.array(np.dot(q_direction, rec_lat.T), dtype="double")
            q_dir_cart = q_dir_cart / np.linalg.norm(q_dir_cart)
        else:
            q_dir_cart = None

        d_dd = np.zeros((3, num_atom, 3, num_atom, 3), dtype="cdouble")

        delta = np.eye(3)
        for G in G_list:
            Q = G + q_cart
            if np.linalg.norm(Q) < self.Q_DIRECTION_TOLERANCE:
                # G=0 at q=Gamma: undefined without a direction.  When a
                # q_direction is supplied, evaluate the G=0 term along
                # that direction (LO-TO convention).  Otherwise skip,
                # matching the value-side convention.
                if q_dir_cart is None:
                    continue
                Q = q_dir_cart

            v = eps @ Q  # (3,)
            B = float(Q @ v)
            C = float(np.exp(-B / (4.0 * Lambda**2)))
            K = v * (2.0 / B**2 + 1.0 / (2.0 * Lambda**2 * B))
            # T[mu, beta, beta'] shape=(3, 3, 3)
            T = (
                delta[:, :, None] * Q[None, None, :]
                + Q[None, :, None] * delta[:, None, :]
            ) / B - np.outer(Q, Q)[None, :, :] * K[:, None, None]

            # Contract Born tensors:
            #   ZTZ[mu, k, a, kp, ap] =
            #     sum_{b, c} Born[k, b, a] * T[mu, b, c] * Born[kp, c, ap]
            ZTZ = np.einsum("kba,mbc,lcp->mkalp", born, T, born)

            # Phase exp(2 pi i G . (R[k] - R[kp])) -- G_list follows the
            # phonopy convention of rec_lat = inv(cell) without the 2 pi
            # factor (see dynmat.c:694).
            phase = np.exp(2j * np.pi * (pos @ G))  # (num_atom,)
            phase_pair = np.outer(phase, phase.conj())  # (num_atom, num_atom)

            d_dd += ZTZ * C * phase_pair[None, :, None, :, None]

        # Mass weighting 1 / sqrt(m_k m_kp)
        inv_sqrt_m = 1.0 / np.sqrt(self._pcell.masses)
        mass_pair = np.outer(inv_sqrt_m, inv_sqrt_m)
        d_dd *= mass_pair[None, :, None, :, None]
        d_dd *= factor

        return d_dd.reshape(3, num_atom * 3, num_atom * 3)

    # --- Wang NAC: analytical derivative of the (A_i (x) A_j) / B term. ---
    # The reciprocal-vector simple-pole form used here is specific to
    # Wang's NAC; Gonze-Lee's dipole-dipole derivative is implemented
    # above via the Ewald sum.

    def _wang_nac(self, q_direction: NDArray[np.double]) -> NDArray[np.double]:
        """Return Wang NAC contribution to FC at q.

        Returns
        -------
        ndarray
            shape=(num_patom, num_patom, 3, 3), dtype="double"

        """
        assert isinstance(self._dynmat, DynamicalMatrixNAC)
        num_atom = len(self._pcell)
        nac_q = np.zeros((num_atom, num_atom, 3, 3), dtype="double")
        if (np.abs(q_direction) < 1e-5).all():
            return nac_q

        rec_lat = np.linalg.inv(self._pcell.cell)
        nac_factor = self._dynmat.nac_factor
        Z = self._dynmat.born
        e = self._dynmat.dielectric_constant
        q = np.dot(rec_lat, q_direction)

        B = self._wang_B(e, q)
        for i in range(num_atom):
            A_i = self._wang_A(q, Z, i)
            for j in range(num_atom):
                A_j = self._wang_A(q, Z, j)
                nac_q[i, j] = np.outer(A_i, A_j) / B

        num_satom = len(self._scell)
        N = num_satom // num_atom

        return nac_q * nac_factor / N

    def _d_wang_nac(self, q_direction: NDArray[np.double]) -> NDArray[np.double]:
        """Return d/dq of Wang NAC contribution to FC at q.

        Returns
        -------
        ndarray
            shape=(3, num_patom, num_patom, 3, 3), dtype="double"

        """
        assert isinstance(self._dynmat, DynamicalMatrixNAC)
        num_atom = len(self._pcell)
        d_nac_q = np.zeros((3, num_atom, num_atom, 3, 3), dtype="double")
        if (np.abs(q_direction) < 1e-5).all():
            return d_nac_q

        rec_lat = np.linalg.inv(self._pcell.cell)
        nac_factor = self._dynmat.nac_factor
        Z = self._dynmat.born
        e = self._dynmat.dielectric_constant
        q = np.dot(rec_lat, q_direction)

        B = self._wang_B(e, q)
        for xyz in range(3):
            dB = self._wang_dB(e, q, xyz)
            for i in range(num_atom):
                A_i = self._wang_A(q, Z, i)
                dA_i = self._wang_dA(Z, i, xyz)
                for j in range(num_atom):
                    A_j = self._wang_A(q, Z, j)
                    dA_j = self._wang_dA(Z, j, xyz)
                    d_nac_q[xyz, i, j] = (
                        np.outer(dA_i, A_j) + np.outer(A_i, dA_j)
                    ) / B - np.outer(A_i, A_j) * dB / B**2

        num_satom = len(self._scell)
        N = num_satom // num_atom
        return d_nac_q * nac_factor / N

    def _wang_A(
        self, q: NDArray[np.double], Z: NDArray[np.double], atom_num: int
    ) -> NDArray[np.double]:
        return np.dot(q, Z[atom_num])

    def _wang_B(self, epsilon: NDArray[np.double], q: NDArray[np.double]) -> float:
        return float(np.dot(q, np.dot(epsilon, q)))

    def _wang_dA(
        self, Z: NDArray[np.double], atom_num: int, xyz: int
    ) -> NDArray[np.double]:
        return Z[atom_num, xyz, :]

    def _wang_dB(
        self, epsilon: NDArray[np.double], q: NDArray[np.double], xyz: int
    ) -> NDArray[np.double]:
        return np.dot(epsilon[xyz], q) * 2
