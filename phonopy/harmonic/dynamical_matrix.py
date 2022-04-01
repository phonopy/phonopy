"""Dynamical matrix classes."""

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

import sys
import warnings
from typing import Type, Union

import numpy as np

from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, sparse_to_dense_svecs


class DynamicalMatrix:
    """Dynamical matrix base class.

    When prmitive and supercell lattices are L_p and L_s, respectively,
    frame F is defined by
    L_p = dot(F, L_s), then L_s = dot(F^-1, L_p).
    where lattice matrix is defined by axies a,b,c in Cartesian:
        [ a1 a2 a3 ]
    L = [ b1 b2 b3 ]
        [ c1 c2 c3 ]

    Phase difference in primitive cell unit
    between atoms 1 and 2 in supercell is calculated by, e.g.,
    1j * dot((x_s(2) - x_s(1)), F^-1) * 2pi
    where x_s is reduced atomic coordinate in supercell unit.

    Attributes
    ----------
    primitive: Primitive
        Primitive cell instance. Note that Primitive is inherited from
        PhonopyAtoms.
    supercell: PhonopyAtoms.
        Supercell instance.
    force_constants: ndarray
        Supercell force constants. Full and compact shapes of arrays are
        supported.
        dtype='double'
        shape=(supercell atoms, supercell atoms, 3, 3) for full array
        shape=(primitive atoms, supercell atoms, 3, 3) for compact array
    dynatmical_matrix: ndarray
        Dynamical matrix at specified q.
        dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
        shape=(primitive atoms * 3, primitive atoms * 3)

    """

    # Non analytical term correction
    _nac = False

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        force_constants,
        decimals=None,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms.
            Supercell.
        primitive : Primitive
            Primitive cell.
        force_constants : array_like
            Supercell force constants. Full and compact shapes of arrays are
            supported.
            shape=(supercell atoms, supercell atoms, 3, 3) for full FC.
            shape=(primitive atoms, supercell atoms, 3, 3) for compact FC.
            dtype='double'
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).

        """
        self._scell = supercell
        self._pcell = primitive
        self._decimals = decimals
        self._dynamical_matrix = None
        self._force_constants = None
        self._set_force_constants(force_constants)

        self._dtype_complex = "c%d" % (np.dtype("double").itemsize * 2)

        self._p2s_map = np.array(self._pcell.p2s_map, dtype="int_")
        self._s2p_map = np.array(self._pcell.s2p_map, dtype="int_")
        p2p_map = self._pcell.p2p_map
        self._s2pp_map = np.array(
            [p2p_map[self._s2p_map[i]] for i in range(len(self._s2p_map))], dtype="int_"
        )
        svecs, multi = self._pcell.get_smallest_vectors()
        if self._pcell.store_dense_svecs:
            self._svecs = svecs
            self._multi = multi
        else:
            self._svecs, self._multi = sparse_to_dense_svecs(svecs, multi)

    def is_nac(self):
        """Return bool if NAC is considered or not."""
        return self._nac

    def get_dimension(self):
        """Return number of bands."""
        warnings.warn(
            "DynamicalMatrix.get_dimension() is deprecated.", DeprecationWarning
        )
        return len(self._pcell) * 3

    @property
    def decimals(self):
        """Return number of decimals of dynamical matrix values."""
        return self._decimals

    def get_decimals(self):
        """Return number of decimals of dynamical matrix values."""
        warnings.warn(
            "DynamicalMatrix.get_decimals() is deprecated."
            "Use DynamicalMatrix.decimals attribute.",
            DeprecationWarning,
        )
        return self.decimals

    @property
    def supercell(self):
        """Return supercell."""
        return self._scell

    def get_supercell(self):
        """Return supercell."""
        warnings.warn(
            "DynamicalMatrix.get_supercell() is deprecated."
            "Use DynamicalMatrix.supercell attribute.",
            DeprecationWarning,
        )
        return self.supercell

    @property
    def primitive(self) -> Primitive:
        """Return primitive cell."""
        return self._pcell

    def get_primitive(self):
        """Return primitive cell."""
        warnings.warn(
            "DynamicalMatrix.get_primitive() is deprecated."
            "Use DynamicalMatrix.primitive attribute.",
            DeprecationWarning,
        )
        return self.primitive

    @property
    def force_constants(self):
        """Return supercell force constants."""
        return self._force_constants

    def get_force_constants(self):
        """Return supercell force constants."""
        warnings.warn(
            "DynamicalMatrix.get_force_constants() is deprecated."
            "Use DynamicalMatrix.force_constants attribute.",
            DeprecationWarning,
        )
        return self.force_constants

    @property
    def dynamical_matrix(self):
        """Return dynamcial matrix calculated at q.

        Returns
        -------
        ndarray
            shape=(natom * 3, natom *3)
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)

        """
        dm = self._dynamical_matrix

        if self._dynamical_matrix is None:
            return None

        if self._decimals is None:
            return dm
        else:
            return dm.round(decimals=self._decimals)

    def get_dynamical_matrix(self):
        """Return dynamcial matrix calculated at q."""
        warnings.warn(
            "DynamicalMatrix.get_get_dynamical_matrix() is "
            "deprecated."
            "Use DynamicalMatrix.get_dynamical_matrix attribute.",
            DeprecationWarning,
        )
        return self.dynamical_matrix

    def run(self, q, lang="C"):
        """Run dynamical matrix calculation at a q-point.

        q : array_like
            q-point in fractional coordinates without 2pi.
            shape=(3,), dtype='double'

        """
        self._run(q, lang=lang)

    def set_dynamical_matrix(self, q):
        """Run dynamical matrix calculation at a q-point."""
        warnings.warn(
            "DynamicalMatrix.set_dynamical_matrix() is deprecated."
            "Use DynamicalMatrix.run().",
            DeprecationWarning,
        )
        self.run(q)

    def _run(self, q, lang="C"):
        if lang == "C":
            self._run_c_dynamical_matrix(q)
        else:
            self._run_py_dynamical_matrix(q)

    def _set_force_constants(self, fc):
        if (
            type(fc) is np.ndarray
            and fc.dtype is np.double
            and fc.flags.aligned
            and fc.flags.owndata
            and fc.flags.c_contiguous
        ):  # noqa E129
            self._force_constants = fc
        else:
            self._force_constants = np.array(fc, dtype="double", order="C")

    def _run_c_dynamical_matrix(self, q):
        import phonopy._phonopy as phonoc

        fc = self._force_constants
        mass = self._pcell.masses
        size_prim = len(mass)
        dm = np.zeros((size_prim * 3, size_prim * 3), dtype=self._dtype_complex)

        if fc.shape[0] == fc.shape[1]:  # full-fc
            s2p_map = self._s2p_map
            p2s_map = self._p2s_map
        else:  # compact-fc
            s2p_map = self._s2pp_map
            p2s_map = np.arange(len(self._p2s_map), dtype="int_")

        phonoc.dynamical_matrix(
            dm.view(dtype="double"),
            fc,
            np.array(q, dtype="double"),
            self._svecs,
            self._multi,
            mass,
            s2p_map,
            p2s_map,
        )

        # Data of dm array are stored in memory by the C order of
        # (size_prim * 3, size_prim * 3, 2), where the last 2 means
        # real and imaginary parts. This code assumes this memory
        # order is that expected by numpy. Otherwise, numpy complex array
        # should be created as follows:
        #   dm_double = dm.view(dtype='double').reshape(size_prim * 3,
        #                                               size_prim * 3, 2)
        #   dm = dm_double[:, :, 0] + 1j * dm_double[:, :, 1]
        self._dynamical_matrix = dm

    def _run_py_dynamical_matrix(self, q):
        """Python implementation of building dynamical matrix.

        This is not used in production.
        This works only with full-fc.

        """
        fc = self._force_constants
        svecs = self._svecs
        multi = self._multi
        num_atom = len(self._pcell)
        dm = np.zeros((3 * num_atom, 3 * num_atom), dtype=self._dtype_complex)
        mass = self._pcell.masses
        if fc.shape[0] == fc.shape[1]:
            is_compact_fc = False
        else:
            is_compact_fc = True

        for i, s_i in enumerate(self._pcell.p2s_map):
            if is_compact_fc:
                fc_elem = fc[i]
            else:
                fc_elem = fc[s_i]
            for j, s_j in enumerate(self._pcell.p2s_map):
                sqrt_mm = np.sqrt(mass[i] * mass[j])
                dm_local = np.zeros((3, 3), dtype=self._dtype_complex)
                # Sum in lattice points
                for k in range(len(self._scell)):
                    if s_j == self._s2p_map[k]:
                        m, adrs = multi[k][i]
                        svecs_at = svecs[adrs : adrs + m]
                        phase = []
                        for ll in range(m):
                            vec = svecs_at[ll]
                            phase.append(np.vdot(vec, q) * 2j * np.pi)
                        phase_factor = np.exp(phase).sum()
                        dm_local += fc_elem[k] * phase_factor / sqrt_mm / m

                dm[(i * 3) : (i * 3 + 3), (j * 3) : (j * 3 + 3)] += dm_local

        # Impose Hermisian condition
        self._dynamical_matrix = (dm + dm.conj().transpose()) / 2


class DynamicalMatrixNAC(DynamicalMatrix):
    """Dynamical matrix with NAC base class."""

    _nac = True

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        force_constants,
        symprec=1e-5,
        decimals=None,
        log_level=0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.
        primitive : Primitive
            Primitive cell.
        force_constants : array_like
            Supercell force constants. Full and compact shapes of arrays are
            supported.
            shape=(supercell atoms, supercell atoms, 3, 3) for full FC.
            shape=(primitive atoms, supercell atoms, 3, 3) for compact FC.
            dtype='double'
        symprec : float, optional, defualt=1e-5
            Symmetri tolerance.
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).
        log_levelc : int, optional, defualt=0
            Log level.

        """
        super().__init__(supercell, primitive, force_constants, decimals=decimals)
        self._symprec = symprec
        self._log_level = log_level

    def run(self, q, q_direction=None):
        """Calculate dynamical matrix at q-point.

        q : array_like
            q-point in fractional coordinates without 2pi.
            shape=(3,), dtype='double'
        q_direction : array_like
            q-point direction from Gamma-point in fractional coordinates of
            reciprocal basis vectors. Only the direction is used, i.e.,
            (q_direction / |q_direction|) is computed and used.
            shape=(3,), dtype='double'

        """
        rec_lat = np.linalg.inv(self._pcell.cell)  # column vectors
        if q_direction is None:
            q_norm = np.linalg.norm(np.dot(q, rec_lat.T))
        else:
            q_norm = np.linalg.norm(np.dot(q_direction, rec_lat.T))

        if q_norm < self._symprec:
            self._run(q)
            return False

        self._compute_dynamical_matrix(q, q_direction)

    @property
    def born(self):
        """Return Born effective charge."""
        return self._born

    def get_born_effective_charges(self):
        """Return Born effective charge."""
        warnings.warn(
            "DynamicalMatrixNAC.get_born_effective_charges() is deprecated."
            "Use DynamicalMatrixNAC.born attribute.",
            DeprecationWarning,
        )
        return self.born

    @property
    def nac_factor(self):
        """Return NAC unit conversion factor."""
        return self._unit_conversion * 4.0 * np.pi / self._pcell.volume

    def get_nac_factor(self):
        """Return NAC unit conversion factor."""
        warnings.warn(
            "DynamicalMatrixNAC.get_nac_factor() is deprecated."
            "Use DynamicalMatrixNAC.nac_factor attribute.",
            DeprecationWarning,
        )
        return self.nac_factor

    @property
    def dielectric_constant(self):
        """Return dielectric constant."""
        return self._dielectric

    def get_dielectric_constant(self):
        """Return dielectric constant."""
        warnings.warn(
            "DynamicalMatrixNAC.get_dielectric_constant() is deprecated."
            "Use DynamicalMatrixNAC.dielectric_constant attribute.",
            DeprecationWarning,
        )
        return self.dielectric_constant

    @property
    def nac_method(self):
        """Return NAC method name."""
        return self._method

    def get_nac_method(self):
        """Return NAC method name."""
        warnings.warn(
            "DynamicalMatrixNAC.get_nac_method() is deprecated."
            "Use DynamicalMatrixNAC.nac_method attribute.",
            DeprecationWarning,
        )
        return self.nac_method

    @property
    def nac_params(self):
        """Return NAC basic parameters."""
        return {"born": self.born, "factor": self.factor, "dielectric": self.dielectric}

    @nac_params.setter
    def nac_params(self, nac_params):
        """Set NAC parameters."""
        self._set_nac_params(nac_params)

    def set_nac_params(self, nac_params):
        """Set NAC parameters."""
        warnings.warn(
            "DynamicalMatrixNAC.set_nac_params() is deprecated."
            "Use DynamicalMatrixNAC.nac_params attribute instead.",
            DeprecationWarning,
        )
        self.nac_params = nac_params

    @property
    def symprec(self):
        """Return symmetry tolerance."""
        return self._symprec

    @property
    def log_level(self):
        """Return log level."""
        return self._log_level

    def _set_nac_params(self, nac_params):
        raise NotImplementedError()

    def _set_basic_nac_params(self, nac_params):
        """Set basic NAC parameters."""
        self._born = np.array(nac_params["born"], dtype="double", order="C")
        self._unit_conversion = nac_params["factor"]
        self._dielectric = np.array(nac_params["dielectric"], dtype="double", order="C")

    def set_dynamical_matrix(self, q, q_direction=None):
        """Run dynamical matrix calculation at q-point."""
        warnings.warn(
            "DynamicalMatrixNAC.set_dynamical_matrix() is deprecated."
            "Use DynamicalMatrixNAC.run().",
            DeprecationWarning,
        )
        self.run(q, q_direction=q_direction)

    def _get_charge_sum(self, num_atom, q, born):
        nac_q = np.zeros((num_atom, num_atom, 3, 3), dtype="double", order="C")
        A = np.dot(q, born)
        for i in range(num_atom):
            for j in range(num_atom):
                nac_q[i, j] = np.outer(A[i], A[j])
        return nac_q

    def _get_constant_factor(self, q, dielectric, volume, unit_conversion):
        return (
            unit_conversion * 4.0 * np.pi / volume / np.dot(q.T, np.dot(dielectric, q))
        )

    def _compute_dynamical_matrix(self, q_red, q_direction):
        raise NotImplementedError()


class DynamicalMatrixGL(DynamicalMatrixNAC):
    """Non analytical term correction (NAC) by Gonze and Lee."""

    _method = "gonze"

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        force_constants,
        nac_params=None,
        num_G_points=None,  # For Gonze NAC
        decimals=None,
        symprec=1e-5,
        log_level=0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : Supercell
            Supercell.
        primitive : Primitive
            Primitive cell.
        force_constants : array_like
            Supercell force constants. Full and compact shapes of arrays are
            supported.
            shape=(supercell atoms, supercell atoms, 3, 3) for full FC.
            shape=(primitive atoms, supercell atoms, 3, 3) for compact FC.
            dtype='double'
        symprec : float, optional, defualt=1e-5
            Symmetri tolerance.
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).
        log_levelc : int, optional, defualt=0
            Log level.

        """
        super().__init__(
            supercell,
            primitive,
            force_constants,
            symprec=symprec,
            decimals=decimals,
            log_level=log_level,
        )

        # For the method by Gonze et al.
        self._Gonze_force_constants = None
        if num_G_points is None:
            self._num_G_points = 300
        else:
            self._num_G_points = num_G_points
        self._G_list = None
        self._G_cutoff = None
        self._Lambda = None  # 4*Lambda**2 is stored.
        self._dd_q0 = None

        if nac_params is not None:
            self.nac_params = nac_params

    @property
    def Gonze_nac_dataset(self):
        """Return Gonze-Lee NAC dataset."""
        return (
            self._Gonze_force_constants,
            self._dd_q0,
            self._G_cutoff,
            self._G_list,
            self._Lambda,
        )

    def get_Gonze_nac_dataset(self):
        """Return Gonze-Lee NAC dataset."""
        warnings.warn(
            "DynamicalMatrixGL.get_Gonze_nac_dataset() is deprecated."
            "Use DynamicalMatrixGL.Gonze_nac_dataset attribute instead.",
            DeprecationWarning,
        )
        return self.Gonze_nac_dataset

    def _set_nac_params(self, nac_params):
        """Set and prepare NAC parameters.

        This is called via DynamicalMatrixNAC.nac_params.

        """
        self._set_basic_nac_params(nac_params)
        if "G_cutoff" in nac_params:
            self._G_cutoff = nac_params["G_cutoff"]
        else:
            self._G_cutoff = (
                3 * self._num_G_points / (4 * np.pi) / self._pcell.volume
            ) ** (1.0 / 3)
        self._G_list = self._get_G_list(self._G_cutoff)
        if "Lambda" in nac_params:
            self._Lambda = nac_params["Lambda"]
        else:
            exp_cutoff = 1e-10
            GeG = self._G_cutoff**2 * np.trace(self._dielectric) / 3
            self._Lambda = np.sqrt(-GeG / 4 / np.log(exp_cutoff))

        # self._H = self._get_H()

    def make_Gonze_nac_dataset(self):
        """Prepare Gonze-Lee force constants.

        Dipole-dipole interaction contribution is subtracted from
        supercell force constants.

        """
        try:
            import phonopy._phonopy as phonoc  # noqa F401

            self._run_c_recip_dipole_dipole_q0()
        except ImportError:
            print(
                "Python version of dipole-dipole calculation is not well "
                "implemented."
            )
            sys.exit(1)

        fc_shape = self._force_constants.shape
        d2f = DynmatToForceConstants(
            self._pcell, self._scell, is_full_fc=(fc_shape[0] == fc_shape[1])
        )
        dynmat = []
        num_q = len(d2f.commensurate_points)
        for i, q_red in enumerate(d2f.commensurate_points):
            if self._log_level > 2:
                print("%d/%d %s" % (i + 1, num_q, q_red))
            self._run(q_red)
            dm_dd = self._get_Gonze_dipole_dipole(q_red, None)
            self._dynamical_matrix -= dm_dd
            dynmat.append(self._dynamical_matrix)
        d2f.dynamical_matrices = dynmat
        d2f.run()

        self._Gonze_force_constants = d2f.force_constants
        self._Gonze_count = 0

    def show_nac_message(self):
        """Show message on Gonze-Lee NAC method."""
        print(
            "Use NAC by Gonze et al. (no real space sum in current " "implementation)"
        )
        print("  PRB 50, 13035(R) (1994), PRB 55, 10355 (1997)")
        print(
            "  G-cutoff distance: %4.2f, Number of G-points: %d, "
            "Lambda: %4.2f" % (self._G_cutoff, len(self._G_list), self._Lambda)
        )

    def show_Gonze_nac_message(self):
        """Show message on Gonze-Lee NAC method."""
        warnings.warn(
            "DynamicalMatrixGL.show_Gonze_nac_message() is deprecated."
            "Use DynamicalMatrixGL.show_nac_message instead.",
            DeprecationWarning,
        )
        self.show_nac_message()

    def _compute_dynamical_matrix(self, q_red, q_direction):
        if self._Gonze_force_constants is None:
            self.make_Gonze_nac_dataset()

        if self._log_level > 2:
            print("%d %s" % (self._Gonze_count + 1, q_red))
        self._Gonze_count += 1
        fc = self._force_constants
        self._force_constants = self._Gonze_force_constants
        self._run(q_red)
        self._force_constants = fc
        dm_dd = self._get_Gonze_dipole_dipole(q_red, q_direction)
        self._dynamical_matrix += dm_dd

    def _get_Gonze_dipole_dipole(self, q_red, q_direction):
        rec_lat = np.linalg.inv(self._pcell.cell)  # column vectors
        q_cart = np.array(np.dot(q_red, rec_lat.T), dtype="double")
        if q_direction is None:
            q_dir_cart = None
        else:
            q_dir_cart = np.array(np.dot(q_direction, rec_lat.T), dtype="double")

        try:
            import phonopy._phonopy as phonoc  # noqa F401

            C_recip = self._get_c_recip_dipole_dipole(q_cart, q_dir_cart)
        except ImportError:
            print(
                "Python version of dipole-dipole calculation is not well "
                "implemented."
            )
            sys.exit(1)

        # Mass weighted
        mass = self._pcell.masses
        num_atom = len(self._pcell)
        for i in range(num_atom):
            for j in range(num_atom):
                C_recip[i, :, j, :] *= 1.0 / np.sqrt(mass[i] * mass[j])

        C_dd = C_recip.reshape(num_atom * 3, num_atom * 3)

        return C_dd

    def _get_c_recip_dipole_dipole(self, q_cart, q_dir_cart):
        """Reciprocal part of Eq.(71) on the right hand side.

        This is subtracted from supercell force constants to create
        short-range force constants. Only once at commensurate points.

        This is added to interpolated short range force constants
        to create full force constants. Called many times.

        """
        import phonopy._phonopy as phonoc

        pos = self._pcell.positions
        num_atom = len(pos)
        volume = self._pcell.volume
        dd = np.zeros((num_atom, 3, num_atom, 3), dtype=self._dtype_complex, order="C")

        phonoc.recip_dipole_dipole(
            dd.view(dtype="double"),
            self._dd_q0.view(dtype="double"),
            self._G_list,
            q_cart,
            q_dir_cart,
            self._born,
            self._dielectric,
            np.array(pos, dtype="double", order="C"),
            self._unit_conversion * 4.0 * np.pi / volume,
            self._Lambda,
            self._symprec,
        )
        return dd

    def _run_c_recip_dipole_dipole_q0(self):
        """Reciprocal part of Eq.(71) second term on the right hand side.

        Computed only once.

        """
        import phonopy._phonopy as phonoc

        pos = self._pcell.positions
        self._dd_q0 = np.zeros((len(pos), 3, 3), dtype=self._dtype_complex, order="C")

        phonoc.recip_dipole_dipole_q0(
            self._dd_q0.view(dtype="double"),
            self._G_list,
            self._born,
            self._dielectric,
            np.array(pos, dtype="double", order="C"),
            self._Lambda,
            self._symprec,
        )

        # Limiting contribution
        # inv_eps = np.linalg.inv(self._dielectric)
        # sqrt_det_eps = np.sqrt(np.linalg.det(self._dielectric))
        # coef = (4.0 / 3 / np.sqrt(np.pi) * inv_eps
        #         / sqrt_det_eps * self._Lambda ** 3)
        # self._dd_q0 -= coef

    def _get_py_dipole_dipole(self, K_list, q, q_dir_cart):
        pos = self._pcell.positions
        num_atom = len(self._pcell)
        volume = self._pcell.volume
        C = np.zeros((num_atom, 3, num_atom, 3), dtype=self._dtype_complex, order="C")

        for q_K in K_list:
            if np.linalg.norm(q_K) < self._symprec:
                if q_dir_cart is None:
                    continue
                else:
                    dq_K = q_dir_cart
            else:
                dq_K = q_K

            Z_mat = self._get_charge_sum(
                num_atom, dq_K, self._born
            ) * self._get_constant_factor(
                dq_K, self._dielectric, volume, self._unit_conversion
            )
            for i in range(num_atom):
                dpos = -pos + pos[i]
                phase_factor = np.exp(2j * np.pi * np.dot(dpos, q_K))
                for j in range(num_atom):
                    C[i, :, j, :] += Z_mat[i, j] * phase_factor[j]

        for q_K in K_list:
            q_G = q_K - q
            if np.linalg.norm(q_G) < self._symprec:
                continue
            Z_mat = self._get_charge_sum(
                num_atom, q_G, self._born
            ) * self._get_constant_factor(
                q_G, self._dielectric, volume, self._unit_conversion
            )
            for i in range(num_atom):
                C_i = np.zeros((3, 3), dtype=self._dtype_complex, order="C")
                dpos = -pos + pos[i]
                phase_factor = np.exp(2j * np.pi * np.dot(dpos, q_G))
                for j in range(num_atom):
                    C_i += Z_mat[i, j] * phase_factor[j]
                C[i, :, i, :] -= C_i

        return C

    def _get_G_list(self, G_cutoff, g_rad=100):
        rec_lat = np.linalg.inv(self._pcell.cell)  # column vectors
        # g_rad must be greater than 0 for broadcasting.
        G_vec_list = self._get_G_vec_list(g_rad, rec_lat)
        G_norm2 = ((G_vec_list) ** 2).sum(axis=1)
        return np.array(G_vec_list[G_norm2 < G_cutoff**2], dtype="double", order="C")

    def _get_G_vec_list(self, g_rad, rec_lat):
        pts = np.arange(-g_rad, g_rad + 1)
        grid = np.meshgrid(pts, pts, pts)
        for i in range(3):
            grid[i] = grid[i].ravel()
        return np.dot(rec_lat, grid).T

    def _get_H(self):
        lat = self._scell.cell
        cart_vecs = np.dot(self._svecs, lat)
        Delta = np.dot(cart_vecs, np.linalg.inv(self._dielectric).T)
        D = np.sqrt(cart_vecs * Delta).sum(axis=3)
        x = self._Lambda * Delta
        y = self._Lambda * D
        y2 = y**2
        y3 = y**3
        exp_y2 = np.exp(-y2)
        eps_inv = np.linalg.inv(self._dielectric)

        try:
            from scipy.special import erfc

            erfc_y = erfc(y)
        except ImportError:
            from math import erfc

            erfc_y = np.zeros_like(y)
            for i in np.ndindex(y.shape):
                erfc_y[i] = erfc(y[i])

        with np.errstate(divide="ignore", invalid="ignore"):
            A = (3 * erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 * (3 / y2 + 2)) / y2
            A[A == np.inf] = 0
            A = np.nan_to_num(A)
            B = erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 / y2
            B[B == np.inf] = 0
            B = np.nan_to_num(B)
        H = np.zeros((3, 3) + y.shape, dtype="double", order="C")
        for i, j in np.ndindex((3, 3)):
            H[i, j] = x[:, :, :, i] * x[:, :, :, j] * A - eps_inv[i, j] * B
        return H


class DynamicalMatrixWang(DynamicalMatrixNAC):
    """Non analytical term correction (NAC) by Wang et al."""

    _method = "wang"

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        force_constants,
        nac_params=None,
        decimals=None,
        symprec=1e-5,
        log_level=0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell.
        primitive : Primitive
            Primitive cell.
        force_constants : array_like
            Supercell force constants. Full and compact shapes of arrays are
            supported.
            shape=(supercell atoms, supercell atoms, 3, 3) for full FC.
            shape=(primitive atoms, supercell atoms, 3, 3) for compact FC.
            dtype='double'
        symprec : float, optional, defualt=1e-5
            Symmetri tolerance.
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).
        log_levelc : int, optional, defualt=0
            Log level.

        """
        super().__init__(
            supercell,
            primitive,
            force_constants,
            symprec=symprec,
            decimals=decimals,
            log_level=log_level,
        )

        self._symprec = symprec
        if nac_params is not None:
            self.nac_params = nac_params

    def show_nac_message(self):
        """Show Wang et al.'s paper reference."""
        if self._log_level:
            print("NAC by Wang et al., J. Phys. Condens. Matter 22, " "202201 (2010)")

    def _set_nac_params(self, nac_params):
        """Set NAC parameters.

        This is called via DynamicalMatrixNAC.nac_params.

        """
        self._set_basic_nac_params(nac_params)

    def _compute_dynamical_matrix(self, q_red, q_direction):
        # Wang method (J. Phys.: Condens. Matter 22 (2010) 202201)
        rec_lat = np.linalg.inv(self._pcell.cell)  # column vectors
        if q_direction is None:
            q = np.dot(q_red, rec_lat.T)
        else:
            q = np.dot(q_direction, rec_lat.T)

        constant = self._get_constant_factor(
            q, self._dielectric, self._pcell.volume, self._unit_conversion
        )
        try:
            import phonopy._phonopy as phonoc  # noqa F401

            self._run_c_Wang_dynamical_matrix(q_red, q, constant)
        except ImportError:
            num_atom = len(self._pcell)
            fc_backup = self._force_constants.copy()
            nac_q = self._get_charge_sum(num_atom, q, self._born) * constant
            self._run_py_Wang_force_constants(self._force_constants, nac_q)
            self._run(q_red)
            self._force_constants = fc_backup

    def _run_c_Wang_dynamical_matrix(self, q_red, q, factor):
        import phonopy._phonopy as phonoc

        fc = self._force_constants
        mass = self._pcell.masses
        size_prim = len(mass)
        dm = np.zeros((size_prim * 3, size_prim * 3), dtype=self._dtype_complex)

        if fc.shape[0] == fc.shape[1]:  # full fc
            phonoc.nac_dynamical_matrix(
                dm.view(dtype="double"),
                fc,
                np.array(q_red, dtype="double"),
                self._svecs,
                self._multi,
                mass,
                self._s2p_map,
                self._p2s_map,
                np.array(q, dtype="double"),
                self._born,
                factor,
            )
        else:
            phonoc.nac_dynamical_matrix(
                dm.view(dtype="double"),
                fc,
                np.array(q_red, dtype="double"),
                self._svecs,
                self._multi,
                mass,
                self._s2pp_map,
                np.arange(len(self._p2s_map), dtype="int_"),
                np.array(q, dtype="double"),
                self._born,
                factor,
            )

        self._dynamical_matrix = dm

    def _run_py_Wang_force_constants(self, fc, nac_q):
        N = len(self._scell) // len(self._pcell)
        for s1 in range(len(self._scell)):
            # This if-statement is the trick.
            # In contructing dynamical matrix in phonopy
            # fc of left indices with s1 == self._s2p_map[ s1 ] are
            # only used.
            if s1 != self._s2p_map[s1]:
                continue
            p1 = self._s2pp_map[s1]
            for s2 in range(len(self._scell)):
                p2 = self._s2pp_map[s2]
                fc[s1, s2] += nac_q[p1, p2] / N


def get_dynamical_matrix(
    fc2,
    supercell: PhonopyAtoms,
    primitive: Primitive,
    nac_params=None,
    frequency_scale_factor=None,
    decimals=None,
    symprec=1e-5,
    log_level=0,
):
    """Return dynamical matrix.

    The instance of a class inherited from DynamicalMatrix will be returned
    depending on paramters.

    """
    if frequency_scale_factor is None:
        _fc2 = fc2
    else:
        _fc2 = fc2 * frequency_scale_factor**2

    if nac_params is None:
        dm = DynamicalMatrix(supercell, primitive, _fc2, decimals=decimals)
    else:
        if "method" not in nac_params:
            method = "gonze"
        else:
            method = nac_params["method"]

        DM_cls: Union[Type[DynamicalMatrixGL], Type[DynamicalMatrixWang]]
        if method == "wang":
            DM_cls = DynamicalMatrixWang
        else:
            DM_cls = DynamicalMatrixGL
        dm = DM_cls(
            supercell,
            primitive,
            _fc2,
            decimals=decimals,
            symprec=symprec,
            log_level=log_level,
        )
        dm.nac_params = nac_params
    return dm
