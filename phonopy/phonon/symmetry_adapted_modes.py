# SPDX-License-Identifier: BSD-3-Clause
"""Symmetry-adapted phonon modes at a q-point.

Degenerate sets are determined from the representation of the little group
of q including time reversal, not from closeness of frequencies.

The dynamical matrix is assumed to be in phonopy's convention whose phase
contains atomic positions (C-type).  Eigenvectors are returned in the same
convention.

The equations and symbols used in the docstrings of this module are given
in doc/symmetry-adapted-modes.md.  In short:

- x_j: position of atom j in crystallographic coordinates.
- q: q-point in crystallographic coordinates (row vector).
- S = {R|t}: space-group operation with integer rotation R and translation
  t in crystallographic coordinates.
- eta = +1 for a unitary operation S and -1 for an antiunitary operation
  S Theta, where Theta is complex conjugation.
- q' = eta q R^-1.  S belongs to the little group of q when q' - q = G is
  an integer vector.
- T(S): 3N x 3N representation matrix acting on C-type eigenvectors at q.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry
from phonopy.utils import similarity_transformation

QPOINT_TOLERANCE = 1e-5
SUBSPACE_TOLERANCE = 1e-8
CHARACTER_TOLERANCE = 1e-5
RANDOM_SEED = 42


@dataclass(frozen=True)
class LittleGroupOperation:
    """Representation matrix T(S) of one little-group operation at q.

    T(S) is the 3N x 3N matrix

        T[3j + a, 3j' + b] = R_cart[a, b] * phases[j'] * delta(j, permutation[j'])

    with T(S) = V(G) Gamma^{C, eta q}(S) in doc/symmetry-adapted-modes.md.  It acts
    on a C-type eigenvector e at q as

        e -> T e        (unitary, eta = +1)
        e -> T conj(e)  (antiunitary, eta = -1)

    and the result is again an eigenvector at q with the same frequency.  The
    dense matrix is not built; the operation is kept as a permutation of
    atoms, one phase per atom and a 3 x 3 rotation.

    Attributes
    ----------
    rotation : ndarray
        Rotation matrix R in crystallographic coordinates.
        shape=(3, 3), dtype=int64
    translation : ndarray
        Translation t in crystallographic coordinates.
        shape=(3,), dtype=double
    rotation_cartesian : ndarray
        Rotation matrix R_cart = L R L^-1 in Cartesian coordinates, where the
        columns of L are the basis vectors.
        shape=(3, 3), dtype=double
    permutation : ndarray
        Atom j' of the primitive cell is sent to atom j = permutation[j'],
        i.e., R x_j' + t = x_j + n with an integer vector n.  Indices are
        those of the primitive cell.  This is the direction of the mapping,
        not its inverse: the eigenvector component of atom j' is moved to
        atom j.
        shape=(num_atom,), dtype=int64
    phases : ndarray
        Phase factor multiplied to the eigenvector component moved from atom
        j' to atom j = permutation[j']:

            phases[j'] = exp(2 pi i ((q' - q) . x_j - q' . t)),

        where q' = eta q R^-1 and G = q' - q is an integer vector.  The term
        G . x_j appears because the phase of phonopy's dynamical matrix
        contains atomic positions (C-type), so the eigenvector labelled
        q + G differs from that labelled q by exp(2 pi i G . x_j) on atom j.
        The same phase is

            phases[j'] = exp(-2 pi i q . (R x_j' + t - eta x_j')),

        where R x_j' + t is the image not brought back into the unit cell.
        shape=(num_atom,), dtype=cdouble
    is_antiunitary : bool
        True for an operation combined with time reversal (eta = -1).

    """

    rotation: NDArray[np.int64]
    translation: NDArray[np.double]
    rotation_cartesian: NDArray[np.double]
    permutation: NDArray[np.int64]
    phases: NDArray[np.cdouble]
    is_antiunitary: bool

    def transform_vectors(self, vecs: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        """Return T v, or T conj(v) for an antiunitary operation.

        For each column v, the component of atom j' is rotated, multiplied
        by phases[j'] and placed at atom permutation[j'].

        Parameters
        ----------
        vecs : ndarray
            Column vectors.
            shape=(num_atom * 3, num_vecs), dtype=cdouble

        Returns
        -------
        ndarray
            Transformed column vectors.
            shape=(num_atom * 3, num_vecs), dtype=cdouble

        """
        num_atom = len(self.permutation)
        v = vecs.conj() if self.is_antiunitary else vecs
        # R_cart applied to the three components of each atom j'
        rotated = self.rotation_cartesian @ v.reshape(num_atom, 3, -1)
        rotated *= self.phases[:, None, None]
        out = np.empty_like(rotated)
        out[self.permutation] = rotated
        return out.reshape(vecs.shape)

    def transform_matrix(self, mat: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        """Return T M T^dagger, or T conj(M) T^dagger for an antiunitary one.

        The 3 x 3 block M[j', k'] of atoms j' and k' becomes the block at
        atoms (permutation[j'], permutation[k']):

            phases[j'] * conj(phases[k']) * R_cart M[j', k'] R_cart^T

        A C-type dynamical matrix at q is left unchanged by this map.

        With T = (P (x) I_3)(I_N (x) R_cart), where P is the N x N permutation
        matrix with the phases, the computation is done in two stages.  First
        every block is rotated, R_cart M[j', k'] R_cart^T, by one matrix
        product from the right and one batched product from the left.  Then
        the phases are multiplied and the blocks are moved by one gather.

        Parameters
        ----------
        mat : ndarray
            Matrix in the basis of atomic displacements.
            shape=(num_atom * 3, num_atom * 3), dtype=cdouble

        Returns
        -------
        ndarray
            Transformed matrix.
            shape=(num_atom * 3, num_atom * 3), dtype=cdouble

        """
        num_atom = len(self.permutation)
        r = self.rotation_cartesian
        m = mat.conj() if self.is_antiunitary else mat
        # M[j', k'] R^T for all blocks, then R (...) for each row atom j'
        right = (m.reshape(-1, 3) @ r.T).reshape(num_atom, 3, -1)
        rotated = (r @ right).reshape(num_atom, 3, num_atom, 3)
        rotated *= (
            self.phases[:, None, None, None] * self.phases.conj()[None, None, :, None]
        )
        # Block (j', k') goes to (permutation[j'], permutation[k'])
        inverse = np.empty_like(self.permutation)
        inverse[self.permutation] = np.arange(num_atom)
        cart = np.arange(3)
        return rotated[np.ix_(inverse, cart, inverse, cart)].reshape(mat.shape)


def get_little_group_operations(
    qpoint: Sequence[float] | NDArray[np.double],
    primitive: PhonopyAtoms,
    primitive_symmetry: Symmetry,
    with_time_reversal: bool = True,
) -> list[LittleGroupOperation]:
    """Return operations that leave q invariant, unitary ones first.

    An operation S = {R|t} of the space group belongs to the little group
    when

        eta q R^-1 - q = G  (an integer vector),

    with eta = +1 for S and eta = -1 for S Theta.  One operation is kept for
    each rotation, as spglib returns them for the primitive cell.  The
    lattice and the positions are symmetrized over the space group before
    the phases are computed, so that the T(S) close as a group to round-off.

    Parameters
    ----------
    qpoint : array_like
        q-point in reduced coordinates of the reciprocal basis vectors.
        shape=(3,)
    primitive : PhonopyAtoms
        Primitive cell.
    primitive_symmetry : Symmetry
        Symmetry of the primitive cell.
    with_time_reversal : bool, optional
        Include antiunitary operations.  Default is True.

    Returns
    -------
    list of LittleGroupOperation
        The unitary operations (is_antiunitary=False) followed by the
        antiunitary ones (is_antiunitary=True).  Within each part, the
        operations are in the order of primitive_symmetry.symmetry_operations.
        The length is |G_q| + |A_q|, where |A_q| is either 0 or |G_q|; it is
        |G_q| when with_time_reversal is False.

    """
    q = np.array(qpoint, dtype="double")
    symprec = primitive_symmetry.tolerance
    rotations = primitive_symmetry.symmetry_operations["rotations"]
    translations = primitive_symmetry.symmetry_operations["translations"]
    lattice = _get_symmetrized_lattice(primitive.cell, rotations)
    positions = _get_symmetrized_positions(
        primitive.scaled_positions, lattice, rotations, translations, symprec
    )
    ops = []
    sources = [(q, False)]
    if with_time_reversal:
        sources.append((-q, True))
    for q_src, is_antiunitary in sources:
        for r, t in zip(rotations, translations, strict=True):
            q_dst = q_src @ np.linalg.inv(r)
            diff = q_dst - q
            if (np.abs(diff - np.rint(diff)) > QPOINT_TOLERANCE).any():
                continue
            permutation = _get_atom_permutation(r, t, positions, lattice, symprec)
            phase = (q_dst - q) @ positions[permutation].T - q_dst @ t
            ops.append(
                LittleGroupOperation(
                    rotation=np.array(r, dtype="int64"),
                    translation=np.array(t, dtype="double"),
                    rotation_cartesian=similarity_transformation(lattice.T, r),
                    permutation=permutation,
                    phases=np.exp(2j * np.pi * phase),
                    is_antiunitary=is_antiunitary,
                )
            )
    return ops


def _get_atom_permutation(
    rotation: NDArray[np.int64],
    translation: NDArray[np.double],
    positions: NDArray[np.double],
    lattice: NDArray[np.double],
    symprec: float,
) -> NDArray[np.int64]:
    """Return j = permutation[j'] with R x_j' + t = x_j + n, n integer.

    Parameters
    ----------
    rotation : ndarray
        shape=(3, 3), dtype=int64
    translation : ndarray
        shape=(3,), dtype=double
    positions : ndarray
        shape=(num_atom, 3), dtype=double
    lattice : ndarray
        Basis vectors in rows.
        shape=(3, 3), dtype=double

    Returns
    -------
    ndarray
        shape=(num_atom,), dtype=int64

    """
    num_atom = len(positions)
    permutation = np.zeros(num_atom, dtype="int64")
    for i, p in enumerate(positions):
        diff = rotation @ p + translation - positions
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff @ lattice, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > symprec:
            raise RuntimeError("Atom mapping by symmetry operation failed.")
        permutation[i] = j
    if len(np.unique(permutation)) != num_atom:
        raise RuntimeError("Atom mapping by symmetry operation is not one-to-one.")
    return permutation


def _get_symmetrized_lattice(
    lattice: NDArray[np.double], rotations: NDArray[np.int64]
) -> NDArray[np.double]:
    """Return basis vectors whose metric tensor is exactly invariant by rotations.

    With the basis vectors in rows (lattice = L^T, L having them in
    columns), the metric tensor g = lattice lattice^T = L^T L, with
    g[i, k] = a_i . a_k, is averaged as

        g_sym = mean over R of R^T g R.

    The basis vectors are then changed through the polar decomposition
    L = Q g^(1/2) with Q = L g^(-1/2) orthogonal: Q is kept and g^(1/2) is
    replaced by g_sym^(1/2),

        L_sym = Q g_sym^(1/2) = L g^(-1/2) g_sym^(1/2),
        lattice_sym = L_sym^T = g_sym^(1/2) g^(-1/2) lattice.

    The metric tensor of L_sym is g_sym.  L_sym is not in general the basis
    closest to L among those with metric tensor g_sym (that one solves the
    orthogonal Procrustes problem), but it differs from L only by the order
    of g_sym - g.  Rotation matrices in Cartesian coordinates then become
    orthogonal to round-off, which the representation needs to close as a
    group.

    Parameters
    ----------
    lattice : ndarray
        Basis vectors in rows.
        shape=(3, 3), dtype=double
    rotations : ndarray
        shape=(num_ops, 3, 3), dtype=int64

    Returns
    -------
    ndarray
        shape=(3, 3), dtype=double

    """
    metric = lattice @ lattice.T
    metric_sym = np.mean([r.T @ metric @ r for r in rotations], axis=0)
    return _sqrtm(metric_sym) @ _sqrtm(metric, inv=True) @ lattice


def _sqrtm(mat: NDArray[np.double], inv: bool = False) -> NDArray[np.double]:
    """Return the symmetric square root of a symmetric positive-definite matrix.

    With the eigenvalue decomposition mat = U diag(lambda) U^T, where U is
    orthogonal, the result is

        S = U diag(sqrt(lambda)) U^T,

    which is symmetric and positive definite and satisfies S^2 = mat.  This
    is the unique such square root.  With inv=True, the inverse

        S^-1 = U diag(1 / sqrt(lambda)) U^T

    is returned instead.  The eigenvalues must be positive; this is not
    checked.

    Parameters
    ----------
    mat : ndarray
        Symmetric positive-definite matrix.
        shape=(n, n), dtype=double
    inv : bool, optional
        Return the inverse of the square root.  Default is False.

    Returns
    -------
    ndarray
        shape=(n, n), dtype=double

    """
    vals, vecs = np.linalg.eigh(mat)
    diag = 1 / np.sqrt(vals) if inv else np.sqrt(vals)
    return (vecs * diag) @ vecs.T


def _get_symmetrized_positions(
    positions: NDArray[np.double],
    lattice: NDArray[np.double],
    rotations: NDArray[np.int64],
    translations: NDArray[np.double],
    symprec: float,
) -> NDArray[np.double]:
    """Return positions averaged over the images by the space group.

    For each operation s and each atom j' with j = permutation_s[j'],

        x_bar_j = mean over s of (R_s x_j' + t_s + n_s),
        n_s = rint(x_j - R_s x_j' - t_s),

    where the integer vector n_s moves each image next to x_j.

    Parameters
    ----------
    positions : ndarray
        shape=(num_atom, 3), dtype=double
    lattice : ndarray
        Basis vectors in rows.
        shape=(3, 3), dtype=double
    rotations : ndarray
        shape=(num_ops, 3, 3), dtype=int64
    translations : ndarray
        shape=(num_ops, 3), dtype=double

    Returns
    -------
    ndarray
        shape=(num_atom, 3), dtype=double

    """
    pos_sum = np.zeros_like(positions)
    for r, t in zip(rotations, translations, strict=True):
        permutation = _get_atom_permutation(r, t, positions, lattice, symprec)
        images = positions @ r.T + t
        images += np.rint(positions[permutation] - images)
        pos_sum[permutation] += images
    return pos_sum / len(rotations)


class SymmetryAdaptedModes:
    """Phonon modes at q decomposed by the little group of q.

    With the group average of a 3N x 3N matrix M,

        <M> = (sum over unitary S of T M T^dagger
               + sum over antiunitary S Theta of T conj(M) T^dagger)
              / (number of operations),

    the steps are:

    1. X = <(Y + Y^dagger) / 2> for a random complex matrix Y with a fixed
       seed, and D_sym = <D>, where D is the C-type dynamical matrix at q.
    2. X is diagonalized.  Eigenvalues closer than SUBSPACE_TOLERANCE
       (relative) are grouped; each group spans a subspace U_k (3N x d_k)
       that is invariant under the operations.
    3. chi_k(S) = Tr(U_k^dagger T(S) U_k) for the unitary S.  Subspaces with
       equal d_k and equal chi_k within CHARACTER_TOLERANCE form one type mu
       of m_mu subspaces of dimension d_mu.
    4. For each type, B_mu = (U_k1, ..., U_k(m_mu)) (3N x m_mu d_mu), and
       B_mu^dagger D_sym B_mu is diagonalized.  Its eigenvalues come in runs
       of d_mu equal values, and each run is one degenerate set.
    5. Within each type, the basis of every set is rotated so that the set
       transforms by the same matrices as the first set of the type.  These
       matrices depend on X and on the basis eigh returns for the first set,
       so they are not a standard form of the irrep.
    6. The sets of all types are sorted by eigenvalue.

    The size of a degenerate set is the subspace dimension d_mu, so no
    frequency tolerance is used.  See doc/symmetry-adapted-modes.md.

    Attributes
    ----------
    frequencies : ndarray
        shape=(num_band,), dtype=double
    eigenvectors : ndarray
        Column i is the eigenvector of band i.
        shape=(num_band, num_band), dtype=cdouble
    degenerate_sets : list[list[int]]
        Band indices of each degenerate set.
    characters : ndarray
        Characters of the unitary operations for each degenerate set.
        shape=(num_sets, num_unitary_operations), dtype=cdouble
    operations : list[LittleGroupOperation]
        Unitary operations first, then antiunitary ones.

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        qpoint: Sequence[float] | NDArray[np.double],
        primitive_symmetry: Symmetry,
        with_time_reversal: bool = True,
        factor: float | None = None,
    ):
        """Init method."""
        self._qpoint = np.array(qpoint, dtype="double")
        self._primitive = dynamical_matrix.primitive
        if factor is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = factor
        self._operations = get_little_group_operations(
            self._qpoint, self._primitive, primitive_symmetry, with_time_reversal
        )
        self._num_unitary = sum(not op.is_antiunitary for op in self._operations)

        dynamical_matrix.run(self._qpoint)
        dm = dynamical_matrix.dynamical_matrix
        assert dm is not None
        self._dynamical_matrix = np.array(dm, dtype="cdouble")

        self._frequencies: NDArray[np.double]
        self._eigenvectors: NDArray[np.cdouble]
        self._degenerate_sets: list[list[int]]
        self._characters: NDArray[np.cdouble]
        self._run()

    @property
    def qpoint(self) -> NDArray[np.double]:
        """Return q-point."""
        return self._qpoint

    @property
    def operations(self) -> list[LittleGroupOperation]:
        """Return little-group operations."""
        return self._operations

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return frequencies."""
        return self._frequencies

    @property
    def eigenvectors(self) -> NDArray[np.cdouble]:
        """Return eigenvectors."""
        return self._eigenvectors

    @property
    def degenerate_sets(self) -> list[list[int]]:
        """Return degenerate sets."""
        return self._degenerate_sets

    @property
    def characters(self) -> NDArray[np.cdouble]:
        """Return characters of unitary operations."""
        return self._characters

    def get_representation_matrices(self, set_index: int) -> NDArray[np.cdouble]:
        """Return representation matrices of unitary operations for a set.

        For the eigenvectors E (3N x dim) of the set, the matrix of a
        unitary operation S is E^dagger T(S) E.  Its trace is the character
        in ``characters``.  Sets of the same type give the same matrices.

        Returns
        -------
        ndarray
            shape=(num_unitary_operations, dim, dim), dtype=cdouble

        """
        vecs = self._eigenvectors[:, self._degenerate_sets[set_index]]
        return np.array(
            [
                vecs.conj().T @ op.transform_vectors(vecs)
                for op in self._operations[: self._num_unitary]
            ],
            dtype="cdouble",
        )

    def _run(self) -> None:
        """Run steps 1-6 of the class docstring.

        Within a type mu, the eigenvalues of B_mu^dagger D_sym B_mu are taken
        in consecutive chunks of d_mu.  The eigenvalue of a set is the mean
        of its chunk, and frequencies are sgn(w2) sqrt(|w2|) * factor.

        """
        # Steps 1 and 2
        subspaces = self._get_irreducible_subspaces()
        dm_sym = self._symmetrize(self._dynamical_matrix)

        # Step 3
        irrep_types = self._group_by_characters(subspaces)

        # Step 4
        eigvals: list[float] = []
        eigvecs: list[NDArray[np.cdouble]] = []
        chars: list[NDArray[np.cdouble]] = []
        # Indices in eigvecs of the sets of each type with two or more sets
        sets_of_types: list[list[int]] = []
        for members, type_chars in irrep_types:
            dim = subspaces[members[0]].shape[1]
            basis = np.hstack([subspaces[k] for k in members])
            vals, vecs = np.linalg.eigh(basis.conj().T @ dm_sym @ basis)
            vecs = basis @ vecs
            if len(members) > 1:
                sets_of_types.append(
                    list(range(len(eigvecs), len(eigvecs) + len(members)))
                )
            for k in range(len(members)):
                chunk = slice(k * dim, (k + 1) * dim)
                eigvals.append(vals[chunk].mean())
                eigvecs.append(vecs[:, chunk])
                chars.append(type_chars)

        # Step 5
        to_align = [i for indices in sets_of_types for i in indices]
        matrices = self._get_representation_matrices([eigvecs[i] for i in to_align])
        gammas = dict(zip(to_align, matrices, strict=True))
        for reference, *others in sets_of_types:
            for i in others:
                eigvecs[i] = self._align(eigvecs[i], gammas[i], gammas[reference])

        # Step 6
        order = np.argsort(eigvals, kind="stable")
        self._eigenvectors = np.hstack([eigvecs[k] for k in order])
        self._characters = np.array([chars[k] for k in order], dtype="cdouble")
        self._degenerate_sets = []
        vals_band = []
        start = 0
        for k in order:
            dim = eigvecs[k].shape[1]
            self._degenerate_sets.append(list(range(start, start + dim)))
            vals_band += [eigvals[k]] * dim
            start += dim
        vals_band_arr = np.array(vals_band, dtype="double")
        self._frequencies = (
            np.sqrt(np.abs(vals_band_arr)) * np.sign(vals_band_arr) * self._factor
        )

    def _get_representation_matrices(
        self, sets: list[NDArray[np.cdouble]]
    ) -> list[NDArray[np.cdouble]]:
        """Return Gamma_E(g) = E^dagger T(g) E^(*) of sets for all operations g.

        E^(*) is conj(E) for antiunitary g.  With the sets side by side in
        E = (E_1, E_2, ...), each operation is applied once, and Gamma_(E_c)(g)
        is E_c^dagger times the columns of T(g) E^(*) that belong to E_c.  Only
        these diagonal blocks of E^dagger T(g) E^(*) are computed.

        Parameters
        ----------
        sets : list of ndarray
            Eigenvectors E_c of degenerate sets.  Each has
            shape=(num_band, dim_c), dtype=cdouble.

        Returns
        -------
        list of ndarray
            Gamma_(E_c)(g) for each set, in the order of ``sets``.  Each has
            shape=(num_operations, dim_c, dim_c), dtype=cdouble.

        """
        if not sets:
            return []
        vecs = np.hstack(sets)
        # Columns bounds[c]:bounds[c + 1] of vecs are E_c
        bounds = np.cumsum([0] + [e.shape[1] for e in sets])
        gammas = [
            np.empty((len(self._operations), e.shape[1], e.shape[1]), dtype="cdouble")
            for e in sets
        ]
        for i, op in enumerate(self._operations):
            rotated = op.transform_vectors(vecs)
            for c, gamma in enumerate(gammas):
                block = slice(bounds[c], bounds[c + 1])
                gamma[i] = vecs[:, block].conj().T @ rotated[:, block]
        return gammas

    def _align(
        self,
        vecs: NDArray[np.cdouble],
        gammas_vecs: NDArray[np.cdouble],
        gammas_ref: NDArray[np.cdouble],
    ) -> NDArray[np.cdouble]:
        """Rotate a degenerate set to transform like a reference set.

        With Gamma_E(g) = E^dagger T(g) E^(*) over all operations g (E^(*) is
        conj(E) for antiunitary g), the matrix

            J = sum over g of Gamma_vecs(g) R Gamma_reference(g)^dagger

        satisfies Gamma_vecs(g) J^(*) = J Gamma_reference(g), and by Schur's
        lemma it is a multiple of a unitary matrix.  R runs over the matrix
        units E_ab, and the J of largest norm is used.  With J = W S V^dagger,
        vecs W V^dagger transforms by Gamma_reference.

        Parameters
        ----------
        vecs : ndarray
            Eigenvectors of the set to rotate.
            shape=(num_band, dim), dtype=cdouble
        gammas_vecs : ndarray
            Gamma_vecs(g) from _get_representation_matrices.
            shape=(num_operations, dim, dim), dtype=cdouble
        gammas_ref : ndarray
            Gamma_reference(g) of the first set of the same type, from
            _get_representation_matrices.
            shape=(num_operations, dim, dim), dtype=cdouble

        Returns
        -------
        ndarray
            shape=(num_band, dim), dtype=cdouble

        """
        # With R = E_ab, J[a, b] = sum over g of outer(Gamma_vecs(g)[:, a],
        # conj(Gamma_reference(g)[:, b]))
        js = np.einsum("gia,gjb->abij", gammas_vecs, gammas_ref.conj())
        dim = vecs.shape[1]
        norms = np.linalg.norm(js.reshape(dim * dim, -1), axis=1)
        a, b = divmod(int(np.argmax(norms)), dim)
        w, _, vh = np.linalg.svd(js[a, b])
        return vecs @ (w @ vh)

    def _symmetrize(self, mat: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        """Return <M>, the average of M over the little group.

        <M> = (sum over operations of T M T^dagger, with M replaced by
        conj(M) for antiunitary ones) / (number of operations), made
        Hermitian by (<M> + <M>^dagger) / 2.

        Parameters
        ----------
        mat : ndarray
            shape=(num_band, num_band), dtype=cdouble

        """
        mat_sym = np.zeros_like(mat)
        for op in self._operations:
            mat_sym += op.transform_matrix(mat)
        mat_sym /= len(self._operations)
        return (mat_sym + mat_sym.conj().T) / 2

    def _get_irreducible_subspaces(self) -> list[NDArray[np.cdouble]]:
        """Return orthonormal bases U_k of irreducible invariant subspaces.

        X = <(Y + Y^dagger) / 2> with Y a complex Gaussian random matrix
        drawn with RANDOM_SEED.  X commutes with every T(S), so for
        X u = lambda u also X (T u) = lambda (T u): each eigenspace of X is
        closed under the operations.  By Schur's lemma a commuting matrix is
        X = sum over irreps mu of X_mu (x) I_(d_mu), so for a generic X each
        eigenspace is one irreducible component of one irrep mu (or of one
        pair of irreps joined by time reversal, since X also commutes with
        the antiunitary operations).  An irrep can appear several times, and
        each appearance is a separate irreducible component.  Two components
        share an eigenspace only when eigenvalues of X coincide, which has
        probability zero for a random Y.  The
        dynamical matrix is not used here because its eigenvalues can
        coincide or nearly coincide for physical reasons.

        Neighbouring eigenvalues of X with lambda_(i+1) - lambda_i <=
        SUBSPACE_TOLERANCE * max(max|lambda|, 1) are put in the same
        subspace.  Within one irreducible component the eigenvalues are equal
        to round-off; between components they are separated by the typical
        spacing of the eigenvalues of a random matrix.

        Each basis has shape=(num_band, dim), dtype=cdouble.

        """
        n = len(self._dynamical_matrix)
        rng = np.random.default_rng(RANDOM_SEED)
        y = np.array(
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)),
            dtype="cdouble",
        )
        x = self._symmetrize((y + y.conj().T) / 2)
        # eigh returns the eigenvalues in ascending order, so degenerate ones
        # are adjacent
        vals, vecs = np.linalg.eigh(x)
        tolerance = SUBSPACE_TOLERANCE * max(np.abs(vals).max(), 1.0)
        # A new subspace starts at column i when vals[i] is apart from vals[i - 1]
        starts = [i for i in range(1, n) if vals[i] - vals[i - 1] > tolerance]
        # Columns bounds[k]:bounds[k + 1] of vecs span the k-th subspace
        bounds = [0] + starts + [n]
        return [vecs[:, bounds[k] : bounds[k + 1]] for k in range(len(bounds) - 1)]

    def _group_by_characters(
        self, subspaces: list[NDArray[np.cdouble]]
    ) -> list[tuple[list[int], NDArray[np.cdouble]]]:
        """Group subspaces carrying the same representation.

        chi_k(S) = Tr(U_k^dagger T(S) U_k) for the unitary operations S.
        Subspaces k and k' belong to the same type when d_k = d_k' and
        max over S of |chi_k(S) - chi_k'(S)| < CHARACTER_TOLERANCE.

        The characters of all subspaces are computed together.  With the bases
        side by side in U = (U_1, U_2, ...), each operation is applied once,
        the diagonal element of U^dagger T(S) U for column i is
        sum over a of conj(U[a, i]) (T(S) U)[a, i], and chi_k(S) is the sum of
        these over the columns of U_k.

        Parameters
        ----------
        subspaces : list of ndarray
            Orthonormal bases U_k of the irreducible invariant subspaces, in
            the order returned by _get_irreducible_subspaces.  Each has
            shape=(num_band, d_k), dtype=cdouble.

        Returns
        -------
        list of tuple (members, characters)
            One tuple per type mu, in the order in which the first subspace of
            each type appears in ``subspaces``.

            members : list of int
                Indices of the subspaces in ``subspaces`` that belong to the
                type, in increasing order.  The first one, members[0], is the
                subspace whose characters define the type.
            characters : ndarray
                chi_(members[0])(S) for the unitary operations S, in the order
                of self._operations.
                shape=(num_unitary_operations,), dtype=cdouble

        """
        unitary_ops = self._operations[: self._num_unitary]
        bases = np.hstack(subspaces)
        # Columns bounds[k]:bounds[k + 1] of bases are U_k
        bounds = np.cumsum([0] + [basis.shape[1] for basis in subspaces])
        # diags[s, i] = (U^dagger T(S_s) U)[i, i] for every column i of U
        diags = np.array(
            [
                np.einsum("ai,ai->i", bases.conj(), op.transform_vectors(bases))
                for op in unitary_ops
            ],
            dtype="cdouble",
        )

        irrep_types: list[tuple[list[int], NDArray[np.cdouble]]] = []
        for k in range(len(subspaces)):
            dim = bounds[k + 1] - bounds[k]
            chars = diags[:, bounds[k] : bounds[k + 1]].sum(axis=1)
            for members, type_chars in irrep_types:
                first = members[0]
                if (
                    bounds[first + 1] - bounds[first] == dim
                    and np.abs(type_chars - chars).max() < CHARACTER_TOLERANCE
                ):
                    members.append(k)
                    break
            else:  # no existing type matches
                irrep_types.append(([k], chars))
        return irrep_types
