"""Transform dynamical matrix to force constants."""

# Copyright (C) 2014 Atsushi Togo
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
from typing import TYPE_CHECKING

import numpy as np

from phonopy.harmonic.force_constants import distribute_force_constants_by_translations
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, get_supercell, sparse_to_dense_svecs
from phonopy.structure.snf import SNF3x3

if TYPE_CHECKING:
    from phonopy import Phonopy


def get_commensurate_points(supercell_matrix):  # wrt primitive cell
    """Commensurate q-points are returned.

    Parameters
    ----------
    supercell_matrix : array_like
        Supercell matrix with respect to primitive cell basis vectors.
        shape=(3, 3), dtype=int

    Returns
    -------
    commensurate_points : ndarray
        Commensurate points corresponding to supercell matrix.
        shape=(N, 3), dtype='double', order='C'
        where N = det(supercell_matrix)

    """
    smat = np.array(supercell_matrix, dtype=int)
    rec_primitive = PhonopyAtoms(
        symbols=["H"], scaled_positions=[[0, 0, 0]], cell=np.diag([1, 1, 1])
    )
    rec_supercell = get_supercell(rec_primitive, smat.T)
    q_pos = rec_supercell.scaled_positions
    return np.array(
        np.where(q_pos > 1 - 1e-15, q_pos - 1, q_pos), dtype="double", order="C"
    )


def get_commensurate_points_in_integers(supercell_matrix):
    """Commensurate q-points in integer representation are returned.

    A set of integer representation of lattice points is transformed to
    the equivalent set of lattice points in fractional coordinates with
    respect to supercell basis vectors by
        integer_lattice_points / det(supercell_matrix)

    Parameters
    ----------
    supercell_matrix : array_like
        Supercell matrix with respect to primitive cell basis vectors.
        shape=(3, 3)
        dtype=intc

    Returns
    -------
    lattice_points : ndarray
        Integer representation of lattice points in supercell.
        shape=(N, 3)

    """
    smat = np.array(supercell_matrix, dtype=int)
    snf = SNF3x3(smat.T)
    snf.run()
    D = snf.D.diagonal()
    b, c, a = np.meshgrid(range(D[1]), range(D[2]), range(D[0]))
    lattice_points = np.dot(
        np.c_[
            a.ravel() * D[1] * D[2], b.ravel() * D[0] * D[2], c.ravel() * D[0] * D[1]
        ],
        snf.Q.T,
    )
    lattice_points = np.array(lattice_points % np.prod(D), dtype="intc", order="C")
    return lattice_points


def categorize_commensurate_points(comm_points):
    """Categorize integer commensurate points.

    Points are sorted by either q = -q + G or q != -q + G.

    """
    N = len(comm_points)
    ii = []
    ij = []
    for i, p in enumerate(comm_points):
        for j, _p in enumerate(comm_points):
            if ((p + _p) % N == 0).all():
                if i == j:
                    ii.append(i)
                elif i < j:
                    ij.append(i)
                break

    assert len(ii) + len(ij) * 2 == len(comm_points)

    return ii, ij


def ph2fc(ph_orig: "Phonopy", supercell_matrix, with_nac=True):
    """Transform force constants in Phonopy instance to other shape.

    This function is deprecated. Use ph2ph or Phonopy.ph2ph.

    Parameters
    ----------
    supercell_matrix : array_like
        This specifies array shape of the force constants.
    with_nac : bool, optional
        Use non-analytical term correction if NAC paramerters exist. Default is
        True.

    Returns
    -------
    force_constants : ndarray
        Transformed force constants of ``supercell_matrix``.

    """
    warnings.warn(
        "ph2fc function is deprecated. Use Phonopy.ph2ph instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ph2ph(ph_orig, supercell_matrix, with_nac=with_nac).force_constants


def ph2ph(ph_orig: "Phonopy", supercell_matrix, with_nac=False) -> "Phonopy":
    """Transform force constants in Phonopy instance to other shape.

    Parameters
    ----------
    supercell_matrix : array_like
        This specifies array shape of the force constants.
    with_nac : bool, optional
        Non-analytical term correction (NAC) is used under the Fourier
        interpolation and NAC parameters are copied to Phonopy class
        instance if they exist. Default is False.

    Returns
    -------
    ph : Phonopy
        Phonopy class instance with init parameters of this Phonopy class
        instance and transformed force constants of `supercell_matrix`.

    """
    return ph_orig.ph2ph(supercell_matrix, with_nac=with_nac)


class DynmatToForceConstants:
    """Transforms eigensolutions to force constants.

    This is the inverse transform of force constants to eigensolutions.
    Eigenvalue-eigenvector pairs of dynamical matrices are transformed to
    force constants by this class. Optionally, dynamical matrices
    (not the DynamicalMatrix class instance, but the calculated matrices)
    can be used instead of above pairs.

    The eigensolutions have to be those calculated at commensurate q-points
    of supercell matrix. The commensurate q-points are obtained by the
    attribute ``commensurate_points``. This set of commensurate points can be
    also given by setter of it.

    One supposed usage is::

       d2f = DynmatToForceConstants(primitive, supercell)
       comm_points = d2f.commensurate_points
       ... calculated phonons at comm_points
       d2f.create_dynamical_matrices(eigenvalues=eigenvalues,
                                     eigenvectors=eigenvectors)
       d2f.run()
       fc = d2f.force_constants

    Instead of recreating dynamical matrices from eigensolutions,
    dynamical matrices can be used directly as follows::

       d2f = DynmatToForceConstants(primitive, supercell)
       comm_points = d2f.commensurate_points
       ... calculated phonons at comm_points
       d2f.dynamical_matrices = dynmat
       d2f.run()
       fc = d2f.force_constants

    Attributes
    ----------
    force_constants : ndarray
        Calculated force constants. Only getter. Array shape depends on
        ``is_full_fc``.
            True : shape=(supercell_atoms, supercell_atoms, 3, 3)
            False : shape=(primitive_atoms, supercell_atoms, 3, 3)
        Default is True.
        dtype='double', order='C'.
    commensurate_points : ndarray
        Commensurate q-points of supercell matrix.
        shape=(det(supercell_matrix), 3), dtype='double', order='C'.
    dynamical_matrices : ndarray
        shape=(det(supercell_matrix), num_band, num_band),
        where num_band is 3 x number of atoms in primitive cell.
        dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
        order='C'.

    """

    def __init__(
        self,
        primitive: Primitive,
        supercell: PhonopyAtoms,
        eigenvalues=None,
        eigenvectors=None,
        dynamical_matrices=None,
        commensurate_points=None,
        is_full_fc=True,
        use_openmp=False,
    ):
        """Init method.

        Parameters
        ----------
        primitive : Primitive
            Primitive cell
        supercell : PhonopyAtoms
            Supercell, not necessarily being an instance of Supercell class.
        is_full_fc : bool
            This controls the matrix shape of calculated force constants.
            True and False give the full and compact force cosntants,
            respectively. The default is True. See more details in Attributes
            section of this class.
        use_openmp : bool, optional, default=False
            Use OpenMP in calculate force constants from dynamical matrix.

        """
        self._pcell = primitive
        self._scell = supercell
        supercell_matrix = np.linalg.inv(self._pcell.primitive_matrix)
        supercell_matrix = np.rint(supercell_matrix).astype("intc")
        if commensurate_points is None:
            self._commensurate_points = get_commensurate_points(supercell_matrix)
        else:
            self._commensurate_points = commensurate_points

        svecs, multi = self._pcell.get_smallest_vectors()
        if self._pcell.store_dense_svecs:
            self._svecs = svecs
            self._multi = multi
        else:
            self._svecs, self._multi = sparse_to_dense_svecs(svecs, multi)

        self._use_openmp = use_openmp

        self._dynmat = None
        self._fc = None

        n_s = len(self._scell)
        n_p = len(self._pcell)
        if is_full_fc:
            self._fc_shape = (n_s, n_s, 3, 3)
        else:
            self._fc_shape = (n_p, n_s, 3, 3)

        self._dtype_complex = "c%d" % (np.dtype("double").itemsize * 2)

        if dynamical_matrices is not None or commensurate_points is not None:
            warnings.warn(
                "Instanciation init parameters of dynamical_matrices"
                " and commensurate_points are deprecated. Use "
                "respecitve attributes.",
                DeprecationWarning,
                stacklevel=2,
            )

        if eigenvalues is not None or eigenvectors is not None:
            warnings.warn(
                "Instanciation init parameters of eigenvalues and "
                "eigenvectors are deprecated. Use "
                "create_dynamical_matrices method.",
                DeprecationWarning,
                stacklevel=2,
            )

        if eigenvalues is not None and eigenvectors is not None:
            self.create_dynamical_matrices(
                eigenvalues=eigenvalues, eigenvectors=eigenvectors
            )
        elif dynamical_matrices is not None:
            self.dynamical_matrices = dynamical_matrices

    def run(self, lang="C"):
        """Run."""
        self._fc = np.zeros(self._fc_shape, dtype="double", order="C")
        self._inverse_transformation(lang=lang)

    @property
    def force_constants(self):
        """Return force constants."""
        return self._fc

    def get_force_constants(self):
        """Return force constants."""
        warnings.warn(
            "Use attribute, force_constants.", DeprecationWarning, stacklevel=2
        )
        return self.force_constants

    @property
    def commensurate_points(self):
        """Getter and setter of commensurate points.

        Returns for getter and Parameters for setter
        --------------------------------------------
        commensurate_points : ndarray (array_like for setter)
            Commensurate q-points corresponding to supercell_matrix. The order
            has to be the same as those of eigenvalues and eigenvectors. As
            the default behaviour, commensurate q-points are generated unless
            they are given.
            shape=(det(supercell_matrix), 3), dtype='double', order='C'

        """
        return self._commensurate_points

    @commensurate_points.setter
    def commensurate_points(self, comm_points):
        self._commensurate_points = np.array(comm_points, dtype="double", order="C")

    def get_commensurate_points(self):
        """Commensurate points in supercell with respect to primitive cell."""
        warnings.warn(
            "Use attribute, commensurate_points.", DeprecationWarning, stacklevel=2
        )
        return self.commensurate_points

    @property
    def dynamical_matrices(self):
        """Getter and setter of numerical matrices of dynamical matrices.

        dynamical_matrices : ndarray (array_like for setter)
            Dynamical matrices at commensurate q-points.
            shape=(det(supercell_matrix), num_band, num_band)
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
            order='C'.
            where ``num_band`` is 3 x number of atoms in primitive cell.

        """
        return self._dynmat

    @dynamical_matrices.setter
    def dynamical_matrices(self, dynmat):
        self._dynmat = np.array(dynmat, dtype=self._dtype_complex, order="C")

    def get_dynamical_matrices(self):
        """Return numerical matrices of dynamical matrices."""
        warnings.warn(
            "Use attribute, dynamical_matrix.", DeprecationWarning, stacklevel=2
        )
        return self.dynamical_matrices

    def set_dynamical_matrices(self, dynmat):
        """Set numerical matrices of dynamical matrices."""
        warnings.warn(
            "Use attribute, dynamical_matrix.", DeprecationWarning, stacklevel=2
        )
        self.dynamical_matrices = dynmat

    def create_dynamical_matrices(self, eigenvalues, eigenvectors):
        """Create dynamcial matrices from eigenvalues and eigenvectors pairs.

        Parameters
        ----------
        eigenvalues : ndarray
            Phonon eigenvalues as the solution of dynamical matrices at
            commensurate q-points.
            shape=(det(supercell_matrix), num_band), dtype='double', order='C'
            where ``num_band`` is 3 x number of atoms in primitive cell.
        eigenvectors : ndarray
            Phonon eigenvectors as the solution of dynamical matrices at
            commensurate q-points.
            shape=(det(supercell_matrix), num_band, num_band)
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
            order='C'.
            where ``num_band`` is 3 x number of atoms in primitive cell.

        """
        dm = []
        for eigvals, eigvecs in zip(eigenvalues, eigenvectors):
            dm.append(np.dot(np.dot(eigvecs, np.diag(eigvals)), eigvecs.T.conj()))
        self.dynamical_matrices = dm

    def _inverse_transformation(self, lang="C"):
        if lang == "C":
            self._c_inverse_transformation()
        else:
            self._py_inverse_transformation()

        if self._fc.shape[0] == self._fc.shape[1]:
            distribute_force_constants_by_translations(self._fc, self._pcell)

    def _c_inverse_transformation(self):
        import phonopy._phonopy as phonoc

        s2p = np.array(self._pcell.s2p_map, dtype="long")
        p2p = self._pcell.p2p_map
        s2pp = np.array([p2p[i] for i in s2p], dtype="long")

        if self._fc.shape[0] == self._fc.shape[1]:
            fc_index_map = np.array(self._pcell.p2s_map, dtype="long")
        else:
            fc_index_map = np.arange(self._fc.shape[0], dtype="long")

        phonoc.transform_dynmat_to_fc(
            self._fc,
            self._dynmat.view(dtype="double"),
            self._commensurate_points,
            self._svecs,
            self._multi,
            self._pcell.masses,
            s2pp,
            fc_index_map,
            self._use_openmp * 1,
        )

    def _py_inverse_transformation(self):
        s2p = self._pcell.s2p_map
        p2s = self._pcell.p2s_map
        p2p = self._pcell.p2p_map

        m = self._pcell.masses
        N = len(self._scell) / len(self._pcell)

        for p_i, s_i in enumerate(p2s):
            for s_j, p_j in enumerate([p2p[i] for i in s2p]):
                coef = np.sqrt(m[p_i] * m[p_j]) / N
                fc_elem = self._sum_q(p_i, s_j, p_j) * coef
                if self._fc.shape[0] == self._fc.shape[1]:
                    self._fc[s_i, s_j] = fc_elem
                else:
                    self._fc[p_i, s_j] = fc_elem

    def _sum_q(self, p_i, s_j, p_j):
        """Sum over commensurate q-points for a pair of atoms."""
        multi, adrs = self._multi[s_j, p_i]
        pos = self._svecs[adrs : (adrs + multi)]
        sum_q = np.zeros((3, 3), dtype=self._dtype_complex, order="C")
        phases = -2j * np.pi * np.dot(self._commensurate_points, pos.T)
        phase_factors = np.exp(phases).sum(axis=1) / multi
        for i, coef in enumerate(phase_factors):
            sum_q += (
                self._dynmat[i, (p_i * 3) : (p_i * 3 + 3), (p_j * 3) : (p_j * 3 + 3)]
                * coef
            )
        return sum_q.real
