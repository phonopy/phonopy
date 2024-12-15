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

import itertools
import sys
import warnings
from typing import Optional, Type, Union

import numpy as np

from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.brillouin_zone import BrillouinZone
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
        use_openmp=False,
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
        use_openmp : bool, optional, default=False
            Use OpenMP in calculate dynamical matrix.

        """
        self._scell = supercell
        self._pcell = primitive
        self._decimals = decimals
        self._use_openmp = use_openmp
        self._dynamical_matrix = None
        self._force_constants = None
        self._set_force_constants(force_constants)

        self._dtype_complex = "c%d" % (np.dtype("double").itemsize * 2)

        self._p2s_map = np.array(self._pcell.p2s_map, dtype="long")
        self._s2p_map = np.array(self._pcell.s2p_map, dtype="long")
        p2p_map = self._pcell.p2p_map
        self._s2pp_map = np.array(
            [p2p_map[self._s2p_map[i]] for i in range(len(self._s2p_map))], dtype="long"
        )
        svecs, multi = self._pcell.get_smallest_vectors()
        if self._pcell.store_dense_svecs:
            self._svecs = svecs
            self._multi = multi
        else:
            self._svecs, self._multi = sparse_to_dense_svecs(svecs, multi)

    def is_nac(self) -> bool:
        """Return bool if NAC is considered or not."""
        return self._nac

    def get_dimension(self):
        """Return number of bands."""
        warnings.warn(
            "DynamicalMatrix.get_dimension() is deprecated.",
            DeprecationWarning,
            stacklevel=2,
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
            stacklevel=2,
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
            stacklevel=2,
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
            stacklevel=2,
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
            stacklevel=2,
        )
        return self.force_constants

    @property
    def dynamical_matrix(self) -> Optional[np.ndarray]:
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
            "Use DynamicalMatrix.dynamical_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dynamical_matrix

    @property
    def use_openmp(self) -> bool:
        """Return activate OpenMP or not."""
        return self._use_openmp

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
            stacklevel=2,
        )
        self.run(q)

    def _run(self, q, lang="C"):
        if lang == "C":
            self._run_c_dynamical_matrix(q)
        else:
            self._run_py_dynamical_matrix(q)

    def _set_force_constants(self, fc):
        if (
            isinstance(fc, np.ndarray)
            and fc.dtype == np.dtype("double")
            and fc.flags.aligned
            and fc.flags.owndata
            and fc.flags.c_contiguous
        ):
            self._force_constants = fc
        else:
            self._force_constants = np.array(fc, dtype="double", order="C")

    def _run_c_dynamical_matrix(self, q):
        self._dynamical_matrix = run_dynamical_matrix_solver_c(self, q, is_nac=False)

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
    Q_DIRECTION_TOLERANCE = 1e-5

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        force_constants,
        decimals=None,
        log_level=0,
        use_openmp=False,
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
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).
        log_levelc : int, optional, defualt=0
            Log level.
        use_openmp : bool, optional, default=False
            Use OpenMP in calculate dynamical matrix.

        """
        super().__init__(
            supercell,
            primitive,
            force_constants,
            decimals=decimals,
            use_openmp=use_openmp,
        )
        self._log_level = log_level
        self._rec_lat = np.linalg.inv(self._pcell.cell)  # column vectors

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
        if q_direction is None:
            q_norm = np.linalg.norm(self._rec_lat @ q)
        else:
            q_norm = np.linalg.norm(self._rec_lat @ q_direction)

        if q_norm < self.Q_DIRECTION_TOLERANCE:
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
            stacklevel=2,
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
            stacklevel=2,
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
            stacklevel=2,
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
            stacklevel=2,
        )
        return self.nac_method

    @property
    def nac_params(self):
        """Return NAC basic parameters."""
        return {
            "born": self.born,
            "factor": self.nac_factor,
            "dielectric": self.dielectric_constant,
        }

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
            stacklevel=2,
        )
        self.nac_params = nac_params

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
            stacklevel=2,
        )
        self.run(q, q_direction=q_direction)

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
        with_full_terms=False,
        decimals=None,
        log_level=0,
        use_openmp=False,
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
        with_full_terms : bool, optional
            When False, only reciprocal terms are considered for NAC. False is the
            default and the reasonable choice.
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).
        log_levelc : int, optional, defualt=0
            Log level.
        use_openmp : bool, optional, default=False
            Use OpenMP in calculate dynamical matrix.

        """
        super().__init__(
            supercell,
            primitive,
            force_constants,
            decimals=decimals,
            log_level=log_level,
            use_openmp=use_openmp,
        )

        # For the method by Gonze et al.
        self._Gonze_force_constants = None
        self._with_full_terms = with_full_terms
        if num_G_points is None:
            self._num_G_points = 300
        else:
            self._num_G_points = num_G_points
        self._G_list = None
        self._G_cutoff = None
        self._Lambda = None  # 4*Lambda**2 is stored.
        self._dd_q0 = None
        self._dd_real_q0 = None
        self._dd_limiting = None
        self._H = None
        self._bz = BrillouinZone(self._rec_lat)

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

    @property
    def short_range_force_constants(self):
        """Getter and setter of short-range force constants.

        Initial short range force constants are computed at
        make_Gonze_nac_dataset.

        """
        return self._Gonze_force_constants

    @short_range_force_constants.setter
    def short_range_force_constants(self, short_range_force_constants):
        """Set short-range force constants."""
        self._Gonze_force_constants = short_range_force_constants

    def get_Gonze_nac_dataset(self):
        """Return Gonze-Lee NAC dataset."""
        warnings.warn(
            "DynamicalMatrixGL.get_Gonze_nac_dataset() is deprecated."
            "Use DynamicalMatrixGL.Gonze_nac_dataset attribute instead.",
            DeprecationWarning,
            stacklevel=2,
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

        if self._with_full_terms:
            self._run_limiting_dipole_dipole()
            self._H = self._get_H()
            self._run_real_dipole_dipole_q0()

        fc_shape = self._force_constants.shape
        d2f = DynmatToForceConstants(
            self._pcell,
            self._scell,
            is_full_fc=(fc_shape[0] == fc_shape[1]),
            use_openmp=self._use_openmp,
        )

        # Bring commensurate points into first-BZ because
        # Gonze-dipole-dipole is not, strictly speaking, periodict over G.
        self._bz.run(d2f.commensurate_points)
        comm_points_in_BZ = np.array(
            [pts[0] for pts in self._bz.shortest_qpoints], dtype="double", order="C"
        )
        d2f.commensurate_points = comm_points_in_BZ

        dynmat = []
        num_q = len(d2f.commensurate_points)
        for i, q_red in enumerate(comm_points_in_BZ):
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
            stacklevel=2,
        )
        self.show_nac_message()

    def _compute_dynamical_matrix(self, q_red, q_direction):
        if self._with_full_terms:
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
        else:
            self._dynamical_matrix = run_dynamical_matrix_solver_c(
                self, q_red, q_direction
            )

    def _get_Gonze_dipole_dipole(self, q_red, q_direction):
        num_atom = len(self._pcell)
        q_cart = np.array(np.dot(q_red, self._rec_lat.T), dtype="double")
        if q_direction is None:
            q_dir_cart = None
        else:
            q_dir_cart = np.array(np.dot(q_direction, self._rec_lat.T), dtype="double")

        try:
            import phonopy._phonopy as phonoc  # noqa F401

            C_recip = self._get_c_recip_dipole_dipole(q_cart, q_dir_cart)
        except ImportError:
            print(
                "Python version of dipole-dipole calculation is not well "
                "implemented."
            )
            sys.exit(1)

        if self._with_full_terms:
            for i in range(num_atom):
                C_recip[i, :, i, :] += self._dd_limiting
            C_recip += self._get_real_dipole_dipole(q_red)
            drift = self._dd_q0 + self._dd_limiting * num_atom + self._dd_real_q0
            for i in range(num_atom):
                C_recip[i, :, i, :] -= drift[i]

        # Mass weighted
        mass = self._pcell.masses
        for i in range(num_atom):
            for j in range(num_atom):
                C_recip[i, :, j, :] *= 1.0 / np.sqrt(mass[i] * mass[j])

        C_dd = C_recip.reshape(num_atom * 3, num_atom * 3)

        return C_dd

    def _get_c_recip_dipole_dipole(
        self, q_cart: np.ndarray, q_dir_cart: Optional[np.ndarray]
    ) -> np.ndarray:
        """Reciprocal part of Eq.(71) on the right hand side.

        This is subtracted from supercell force constants to create
        short-range force constants. Only once at commensurate points.

        This is added to interpolated short range force constants
        to create full force constants. Called many times.

        Returns
        -------
        shape=(num_atom, 3, num_atom, 3), dtype=complex

        """
        import phonopy._phonopy as phonoc

        pos = self._pcell.positions
        num_atom = len(pos)
        volume = self._pcell.volume
        dd = np.zeros((num_atom, 3, num_atom, 3), dtype=self._dtype_complex, order="C")

        if self._with_full_terms:
            dd_q0 = np.zeros((len(pos), 3, 3), dtype=self._dtype_complex, order="C")
        else:
            dd_q0 = self._dd_q0

        if q_dir_cart is None:
            is_nac_q_zero = True
            _q_dir_cart = np.zeros(3, dtype="double")
        else:
            is_nac_q_zero = False
            _q_dir_cart = q_dir_cart

        phonoc.recip_dipole_dipole(
            dd.view(dtype="double"),
            dd_q0.view(dtype="double"),
            self._G_list,
            q_cart,
            _q_dir_cart,
            self._born,
            self._dielectric,
            np.array(pos, dtype="double", order="C"),
            is_nac_q_zero * 1,
            self._unit_conversion * 4.0 * np.pi / volume,
            self._Lambda,
            self.Q_DIRECTION_TOLERANCE,
            self._use_openmp * 1,
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
            self.Q_DIRECTION_TOLERANCE,
            self._use_openmp * 1,
        )

    def _get_real_dipole_dipole(self, q_red):
        num_atom = len(self._pcell)
        phase_all = np.exp(2j * np.pi * np.dot(self._svecs, q_red))
        C_real = np.zeros((num_atom, 3, num_atom, 3), dtype=self._dtype_complex)
        vals = (
            -(self._Lambda**3)
            * self._H
            * phase_all
            * np.linalg.det(self._dielectric) ** (-0.5)
        )
        for i_s in range(self._multi.shape[0]):
            for i_p in range(self._multi.shape[1]):
                m = self._multi[i_s, i_p]
                C_real[self._s2pp_map[i_s], :, i_p, :] += (
                    (vals[:, :, m[1]] + vals[:, :, m[1]].conj().T) / 2 / m[0]
                )
        return C_real

    def _run_real_dipole_dipole_q0(self):
        self._dd_real_q0 = np.zeros((len(self._pcell), 3, 3), dtype=self._dtype_complex)
        vals = -(self._Lambda**3) * self._H * np.linalg.det(self._dielectric) ** (-0.5)
        for i_s in range(self._multi.shape[0]):
            for i_p in range(self._multi.shape[1]):
                m = self._multi[i_s, i_p]
                self._dd_real_q0[self._s2pp_map[i_s], :, :] += (
                    (vals[:, :, m[1]] + vals[:, :, m[1]].conj().T) / 2 / m[0]
                )

    def _run_limiting_dipole_dipole(self):
        """Calculate limiting contribution.

        Calculated only once.

        shape=(3, 3)

        """
        inv_eps = np.linalg.inv(self._dielectric)
        sqrt_det_eps = np.sqrt(np.linalg.det(self._dielectric))
        self._dd_limiting = (
            -4.0 / 3 / np.sqrt(np.pi) * inv_eps / sqrt_det_eps * self._Lambda**3
        )

    def _get_G_list(self, G_cutoff: float, g_rad: int = 100) -> np.ndarray:
        """Return list of G vectors at which the values are summed.

        Note
        ----
        g_rad must be greater than 0 for broadcasting.

        """
        _g_rad = self._get_minimum_g_rad(G_cutoff, g_rad)
        G_vec_list = self._get_G_vec_list(_g_rad)
        G_norm2 = ((G_vec_list) ** 2).sum(axis=1)
        return np.array(G_vec_list[G_norm2 < G_cutoff**2], dtype="double", order="C")

    def _get_minimum_g_rad(self, G_cutoff: float, g_rad: int) -> int:
        """Return minimum g_rad."""
        for _g_rad in range(g_rad, 0, -1):
            for a, b, c in itertools.product((-1, 0, 1), repeat=3):
                if (a, b, c) == (0, 0, 0):
                    continue
                norm = np.linalg.norm(self._rec_lat @ [a, b, c]) * _g_rad
                if norm < G_cutoff:
                    return _g_rad + 1
        return g_rad

    def _get_G_vec_list(self, g_rad: int):
        """Return reciprocal lattice point vectors withing g_rad cutoff.

        With g_rad = 2,
        grid.T = [[-2, -2, -2],
                  [-2, -2, -1],
                  [-2, -2,  0],
                  [-2, -2,  1],
                  [-2, -2,  2],
                  [-1, -2, -2],
                  [-1, -2, -1],
                  ...]

        The implmentation using meshgrid may be unstable at numpy 2.0.
        Therefore, another way is used although it can be slower.

        """
        # pts = np.arange(-g_rad, g_rad + 1, dtype="long")
        # grid = np.r_["-1,2,0", np.meshgrid(pts, pts, pts)].reshape(3, -1)
        # return (self._rec_lat @ grid).T
        npts = g_rad * 2 + 1
        grid = np.array(list(np.ndindex((npts, npts, npts)))) - g_rad
        return grid @ self._rec_lat.T

    def _get_H(self):
        lat = self._scell.cell
        cart_vecs = np.dot(self._svecs, lat)
        eps_inv = np.linalg.inv(self._dielectric)
        Delta = np.dot(cart_vecs, eps_inv.T)  # (N, 3)
        D = np.sqrt((cart_vecs * Delta).sum(axis=1))  # (N,)
        x = self._Lambda * Delta
        y = self._Lambda * D
        condition = y < 1e-10
        y[condition] = 1  # dummy to avoid divergence
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

        A = np.where(
            condition,
            0,
            (3 * erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 * (3 / y2 + 2)) / y2,
        )
        B = np.where(condition, 0, erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 / y2)
        H = np.zeros((3, 3) + y.shape, dtype="double", order="C")
        for i, j in np.ndindex((3, 3)):
            H[i, j, :] = x[:, i] * x[:, j] * A - eps_inv[i, j] * B
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
        log_level=0,
        use_openmp=False,
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
        decimals : int, optional, default=None
            Number of decimals. Use like dm.round(decimals).
        log_levelc : int, optional, defualt=0
            Log level.
        use_openmp : bool, optional, default=False
            Use OpenMP in calculate dynamical matrix.

        """
        super().__init__(
            supercell,
            primitive,
            force_constants,
            decimals=decimals,
            log_level=log_level,
            use_openmp=use_openmp,
        )

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
        try:
            import phonopy._phonopy as phonoc  # noqa F401

            self._dynamical_matrix = run_dynamical_matrix_solver_c(
                self, q_red, q_direction
            )
            # self._run_c_Wang_dynamical_matrix(q_red, q_cart, constant)
        except ImportError:
            if q_direction is None:
                q_cart = np.dot(q_red, self._rec_lat.T)
            else:
                q_cart = np.dot(q_direction, self._rec_lat.T)

            constant = self._get_constant_factor(
                q_cart, self._dielectric, self._pcell.volume, self._unit_conversion
            )
            num_atom = len(self._pcell)
            fc_backup = self._force_constants.copy()
            nac_q = self._get_charge_sum(num_atom, q_cart, self._born) * constant
            self._run_py_Wang_force_constants(self._force_constants, nac_q)
            self._run(q_red)
            self._force_constants = fc_backup

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
    log_level=0,
    use_openmp=False,
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
        dm = DynamicalMatrix(
            supercell,
            primitive,
            _fc2,
            decimals=decimals,
            use_openmp=use_openmp,
        )
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
            log_level=log_level,
            use_openmp=use_openmp,
        )
        dm.nac_params = nac_params
    return dm


def run_dynamical_matrix_solver_c(
    dm: Union[DynamicalMatrix, DynamicalMatrixWang, DynamicalMatrixGL],
    qpoints,
    nac_q_direction: Optional[np.ndarray] = None,
    is_nac: Optional[bool] = None,
):
    """Bulid and solve dynamical matrices on grid in C-API.

    If dynamical matrices at many qpoints are calculated, it is recommended not
    to use this function one qpoint by one qpoint to avoid overhead in the
    preparation steps.

    Parameters
    ----------
    dm : DynamicalMatrix
        DynamicalMatrix instance.
    qpoints : array_like,
        q-points in crystallographic coordinates. shape=(n_qpoints, 3),
        dtype='double', order='C'
    nac_q_direction : array_like, optional
        Direction of q from Gamma point given in reduced coordinates. This is
        only activated when q-point->[0,0,0] case. For example, this is used for
        q->[0,0,0] where approaching direction is known, e.g., band structure
        calculation. Default is None.
    is_nac : bool, optional
        True if NAC is considered. Default is None. If None, it is determined
        from dm.is_nac().

    """
    import phonopy._phonopy as phonoc

    if (
        isinstance(qpoints, np.ndarray)
        and qpoints.dtype == np.dtype("double")
        and qpoints.flags.aligned
        and qpoints.flags.owndata
        and qpoints.flags.c_contiguous
    ):
        _qpoints = qpoints
    else:
        _qpoints = np.array(qpoints, dtype="double", order="C")
    qpoints_ndim = _qpoints.ndim
    _qpoints = _qpoints.reshape(-1, 3)

    if is_nac is None:
        _is_nac = dm.is_nac()
    else:
        _is_nac = is_nac

    (
        svecs,
        multi,
        masses,
        rec_lattice,  # column vectors
        positions,  # primitive cell positions
        born,
        nac_factor,
        dielectric,
    ) = _extract_params(dm)

    use_Wang_NAC = False
    if _is_nac:
        if dm.nac_method == "gonze":
            gonze_nac_dataset = dm.Gonze_nac_dataset
            if gonze_nac_dataset[0] is None:
                dm.make_Gonze_nac_dataset()
                gonze_nac_dataset = dm.Gonze_nac_dataset
            (
                gonze_fc,  # fc where the dipole-diple contribution is removed.
                dd_q0,  # second term of dipole-dipole expression.
                G_cutoff,  # Cutoff radius in reciprocal space. This will not be used.
                G_list,  # List of G points where d-d interactions are integrated.
                Lambda,
            ) = gonze_nac_dataset  # Convergence parameter
            fc = gonze_fc
        if dm.nac_method == "wang":
            use_Wang_NAC = True
            dd_q0 = np.zeros((len(positions), 3, 3), dtype="double", order="C")
            G_list = np.zeros((1, 3), dtype="double", order="C")  # dummy value
            Lambda = 0.0
            fc = dm.force_constants
    else:
        dd_q0 = np.zeros((len(positions), 3, 3), dtype="double", order="C")
        G_list = np.zeros((1, 3), dtype="double", order="C")  # dummy value
        Lambda = 0.0
        fc = dm.force_constants

    if nac_q_direction is None:
        is_nac_q_zero = True
        _nac_q_direction = np.zeros(3, dtype="double")  # dummy variable
    else:
        is_nac_q_zero = False
        _nac_q_direction = np.array(nac_q_direction, dtype="double")

    p2s, s2p = _get_fc_elements_mapping(dm, fc)

    dtype_complex = "c%d" % (np.dtype("double").itemsize * 2)
    dynmat = np.zeros(
        (len(qpoints), len(p2s) * 3, len(p2s) * 3), dtype=dtype_complex, order="C"
    )

    phonoc.dynamical_matrices_with_dd_openmp_over_qpoints(
        dynmat.view(dtype="double"),
        _qpoints,
        fc,
        svecs,
        multi,
        positions,
        masses,
        s2p,
        p2s,
        _nac_q_direction,
        born,
        dielectric,
        rec_lattice,
        nac_factor,
        dd_q0,
        G_list,
        Lambda,
        _is_nac * 1,
        is_nac_q_zero * 1,
        use_Wang_NAC * 1,  # use_Wang_NAC
    )

    if qpoints_ndim == 1:
        return dynmat[0]
    else:
        return dynmat


def _extract_params(dm: Union[DynamicalMatrix, DynamicalMatrixNAC]):
    svecs, multi = dm.primitive.get_smallest_vectors()
    if dm.primitive.store_dense_svecs:
        _svecs = svecs
        _multi = multi
    else:
        _svecs, _multi = sparse_to_dense_svecs(svecs, multi)

    masses = dm.primitive.masses
    rec_lattice = np.array(np.linalg.inv(dm.primitive.cell), dtype="double", order="C")
    positions = dm.primitive.positions
    if dm.is_nac():
        born = dm.born
        nac_factor = float(dm.nac_factor)
        dielectric = dm.dielectric_constant
    else:
        born = np.zeros(9)  # dummy variable
        nac_factor = 0.0  # dummy variable
        dielectric = np.zeros(9)  # dummy variable

    return (
        _svecs,
        _multi,
        masses,
        rec_lattice,
        positions,
        born,
        nac_factor,
        dielectric,
    )


def _get_fc_elements_mapping(dm: DynamicalMatrix, fc: np.ndarray):
    p2s_map = dm.primitive.p2s_map
    s2p_map = dm.primitive.s2p_map
    if fc.shape[0] == fc.shape[1]:  # full fc
        return np.array(p2s_map, dtype="long"), np.array(s2p_map, dtype="long")
    else:  # compact fc
        primitive = dm.primitive
        p2p_map = primitive.p2p_map
        s2pp_map = np.array(
            [p2p_map[s2p_map[i]] for i in range(len(s2p_map))], dtype="long"
        )
        return np.arange(len(p2s_map), dtype="long"), s2pp_map
