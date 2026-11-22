# SPDX-License-Identifier: BSD-3-Clause
"""Create atomic displacements."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    diagonalize_dynamical_matrices,
    get_dynamical_matrices_at_qpoints,
)
from phonopy.interface.calculator import (
    StructureInfo,
    get_default_cell_filename,
    write_crystal_structure,
)
from phonopy.phonon.degeneracy import derivative_along_direction, lift_degeneracy
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell

# Relative tolerance to find elements of the largest magnitude, among which
# the first one is used as the reference of the phase of modulations.
PHASE_REFERENCE_RTOL = 1e-8


class Modulation:
    """Atomic displacements along phonon modes in a supercell.

    The complex displacement of atom j in the supercell for one phonon mode is

        u_j = A / sqrt(N * m_j) * exp(i * phi) * e_j * exp(2 * pi * i * q . r_j),

    where A is the amplitude, phi is the phase, N is the number of atoms in
    the supercell, m_j is the mass of atom j, e_j is the part of the
    eigenvector for atom j, and r_j is the position of atom j in reduced
    coordinates of the primitive cell.  The phase phi is measured from the
    reference element chosen by ``_get_phase_factor``.  Modulated supercells
    are made from the real part of u_j.

    Attributes
    ----------
    supercell : PhonopyAtoms
        Supercell without modulations.
    modulations : ndarray
        Complex displacements u_j in Cartesian coordinates, one set per phonon
        mode.  Zeros before ``run()`` is called.
        ``shape=(modes, supercell atoms, 3)``, ``dtype=np.cdouble``.
    modulated_supercells : list[PhonopyAtoms]
        Supercells modulated by each phonon mode.
    modulated_supercell : PhonopyAtoms
        Supercell modulated by the sum of all phonon modes.
    frequencies : ndarray
        Frequencies of the phonon modes, in THz with the default ``factor``.
        ``shape=(modes,)``, ``dtype=np.double``.
    eigenvectors : ndarray
        Eigenvectors of the phonon modes.
        ``shape=(modes, primitive atoms * 3)``, ``dtype=np.cdouble``.

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        dimension: Sequence[int] | Sequence[Sequence[int]] | NDArray[np.int64],
        phonon_modes: Sequence[tuple[NDArray[np.double], int, float, float]],
        delta_q: NDArray[np.double] | Sequence[float] | None = None,
        nac_q_direction: Sequence[float] | NDArray[np.double] | None = None,
        factor: float | None = None,
        degeneracy_lifting_direction: Sequence[float]
        | NDArray[np.double]
        | None = None,
    ) -> None:
        """Init method.

        Parameters
        ----------
        dynamical_matrix : DynamicalMatrix
            Dynamical matrix of the primitive cell.
        dimension : array_like
            Supercell matrix with respect to the primitive cell.
            ``shape=(3,)``, ``(9,)`` or ``(3, 3)``.
        phonon_modes : sequence
            Each element is ``(q, band_index, amplitude, phase)`` of one phonon
            mode.  ``q`` is in reduced coordinates of the reciprocal lattice,
            ``band_index`` starts from 0 in ascending order of frequency, and
            ``phase`` is in degrees.
        delta_q : array_like, optional
            Deprecated.  Use ``degeneracy_lifting_direction``.  This is the
            same direction in Cartesian coordinates.
        nac_q_direction : array_like, optional
            Direction of q from the Gamma point in reduced coordinates of the
            reciprocal lattice, used for the non-analytical term correction at
            q=(0, 0, 0).  ``shape=(3,)``.
        factor : float, optional
            Factor to convert the square root of eigenvalues of the dynamical
            matrix to frequencies in THz.  Default is the value for the default
            units of phonopy.
        degeneracy_lifting_direction : array_like, optional
            Direction in reduced coordinates of the reciprocal lattice used to
            choose eigenvectors of degenerate bands.  In each degenerate
            subspace, the eigenvectors are rotated to diagonalize the
            derivative of the dynamical matrix along this direction.  Bands
            that this derivative does not split are left as returned by the
            eigensolver.  Only the direction is used.  When None, no rotation
            is made.  ``shape=(3,)``.

        """
        self._dm = dynamical_matrix
        self._primitive = dynamical_matrix.primitive
        self._phonon_modes = phonon_modes
        self._dimension = np.array(dimension, dtype="int64").ravel()
        if delta_q is not None:
            # REMOVE-AT-V5: delta_q is removed.
            if degeneracy_lifting_direction is not None:
                raise ValueError(
                    "delta_q and degeneracy_lifting_direction cannot be given together."
                )
            warnings.warn(
                "delta_q is deprecated. Use degeneracy_lifting_direction, which "
                "is in reduced coordinates of the reciprocal lattice.",
                DeprecationWarning,
                stacklevel=2,
            )
            degeneracy_lifting_direction = self._primitive.cell @ np.array(
                delta_q, dtype="double"
            )
        self._lifting_direction = (
            np.array(degeneracy_lifting_direction, dtype="double")
            if degeneracy_lifting_direction is not None
            else None
        )
        self._nac_q_direction = (
            np.array(nac_q_direction, dtype="double")
            if nac_q_direction is not None
            else None
        )
        self._ddm: DerivativeOfDynamicalMatrix | None = None
        if self._lifting_direction is not None:
            self._ddm = DerivativeOfDynamicalMatrix(
                dynamical_matrix, lang=dynamical_matrix.lang
            )

        if factor is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = factor
        dim = self._get_dimension_3x3()
        self._supercell = get_supercell(self._primitive, dim)
        self._u = np.zeros(
            (len(self._phonon_modes), len(self._supercell), 3),
            dtype=np.cdouble,
            order="C",
        )
        self._eigvals = np.zeros(len(self._phonon_modes), dtype="double")
        self._eigvecs = np.zeros(
            (len(self._phonon_modes), len(self._primitive) * 3), dtype=np.cdouble
        )

    def run(self) -> None:
        """Calculate modulations."""
        if len(self._phonon_modes) == 0:
            return

        qpoints = np.array(
            [ph_mode[0] for ph_mode in self._phonon_modes], dtype="double", order="C"
        )
        dynmats = get_dynamical_matrices_at_qpoints(
            self._dm, qpoints, nac_q_direction=self._nac_q_direction
        )
        eigvals, eigvecs = diagonalize_dynamical_matrices(dynmats)
        assert eigvecs is not None
        if self._lifting_direction is not None:
            eigvecs = self._lift_degeneracy(qpoints, eigvals, eigvecs)

        for i, ph_mode in enumerate(self._phonon_modes):
            q, band_index, amplitude, argument = ph_mode
            eigvec = eigvecs[i][:, band_index]
            self._u[i] = self._get_displacements(eigvec, q, amplitude, argument)
            self._eigvecs[i] = eigvec
            self._eigvals[i] = eigvals[i, band_index]

    @property
    def supercell(self) -> PhonopyAtoms:
        """Return the perfect (unmodulated) supercell."""
        return self._supercell

    @property
    def modulations(self) -> NDArray[np.cdouble]:
        """Return complex atomic displacement fields.

        Displacements in Cartesian coordinates, one field per phonon
        mode. ``shape=(modes, supercell atoms, 3)``,
        ``dtype=np.cdouble``. Zeros before ``run()`` is called.

        """
        return self._u

    @property
    def modulated_supercells(self) -> list[PhonopyAtoms]:
        """Return supercells modulated by respective phonon modes."""
        return [self._get_cell_with_modulation(u) for u in self._u]

    @property
    def modulated_supercell(self) -> PhonopyAtoms:
        """Return the supercell modulated by the sum of all phonon modes."""
        return self._get_cell_with_modulation(np.sum(self._u, axis=0))

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies of the selected phonon modes."""
        return self._eigvals_to_frequencies(self._eigvals)

    @property
    def eigenvectors(self) -> NDArray[np.cdouble]:
        """Return eigenvectors of the selected phonon modes.

        ``shape=(modes, primitive cell atoms * 3)``,
        ``dtype=np.cdouble``.

        """
        return self._eigvecs

    def get_modulated_supercells(self) -> list[PhonopyAtoms]:
        """Return supercells modulated by respective phonon modes."""
        return self.modulated_supercells

    def get_modulations_and_supercell(
        self,
    ) -> tuple[NDArray[np.cdouble], PhonopyAtoms]:
        """Return modulations and perfect supercell."""
        return self._u, self._supercell

    def write(
        self,
        interface_mode: str | None = None,
        optional_structure_info: StructureInfo | None = None,
    ) -> None:
        """Write supercells with modulations.

        With the default cell filename ``POSCAR`` of VASP, the files are
        ``MPOSCAR-001``, ``MPOSCAR-002``, ... for the phonon modes,
        ``MPOSCAR`` for the sum of all phonon modes, and ``MPOSCAR-orig``
        without modulations.

        Parameters
        ----------
        interface_mode : str, optional
            Calculator interface such as 'vasp', 'qe', ...  Default is None,
            which is equivalent to 'vasp'.
        optional_structure_info : StructureInfo, optional
            Calculator-specific information returned by
            ``read_crystal_structure``, such as the pseudopotential filenames
            of QE.  Default is None.

        """
        base_fname = get_default_cell_filename(interface_mode)

        for i, cell in enumerate(self.modulated_supercells):
            write_crystal_structure(
                f"M{base_fname}-{(i + 1):03d}",
                cell,
                interface_mode=interface_mode,
                optional_structure_info=optional_structure_info,
            )

        write_crystal_structure(
            f"M{base_fname}",
            self.modulated_supercell,
            interface_mode=interface_mode,
            optional_structure_info=optional_structure_info,
        )
        no_modulations = np.zeros(self._u.shape[1:], dtype=np.cdouble)
        cell = self._get_cell_with_modulation(no_modulations)
        write_crystal_structure(
            f"M{base_fname}-orig",
            cell,
            interface_mode=interface_mode,
            optional_structure_info=optional_structure_info,
        )

    def write_yaml(self, filename: str | os.PathLike = "modulation.yaml") -> None:
        """Write modulations to file in yaml.

        Parameters
        ----------
        filename : str or os.PathLike, optional
            Default is "modulation.yaml".

        """
        lines = self._get_yaml_lines()
        with open(filename, "w") as w:
            w.write("\n".join(lines) + "\n")

    def _get_cell_with_modulation(
        self, modulation: NDArray[np.cdouble]
    ) -> PhonopyAtoms:
        positions = self._supercell.positions + modulation.real
        scaled_positions = positions @ np.linalg.inv(self._supercell.cell)
        scaled_positions -= np.floor(scaled_positions)
        cell = self._supercell.copy()
        cell.scaled_positions = scaled_positions

        return cell

    def _lift_degeneracy(
        self,
        qpoints: NDArray[np.double],
        eigvals: NDArray[np.double],
        eigvecs: NDArray[np.cdouble],
    ) -> NDArray[np.cdouble]:
        """Rotate degenerate eigenvectors by derivative of D along a direction.

        Parameters
        ----------
        qpoints : ndarray
            ``shape=(num_qpoints, 3)``.
        eigvals : ndarray
            ``shape=(num_qpoints, num_band)``.
        eigvecs : ndarray
            Eigenvectors in columns, ``shape=(num_qpoints, num_band, num_band)``.

        """
        assert self._ddm is not None
        assert self._lifting_direction is not None
        # Reciprocal basis vectors are the columns of inv(lattice).
        direction = np.linalg.inv(self._primitive.cell) @ self._lifting_direction
        norm = np.linalg.norm(direction)

        # Derivatives are computed in a loop separate from the one below,
        # which calls BLAS, because the derivative kernel of phonors runs
        # with rayon threads.
        ddms_along_direction = []
        for q in qpoints:
            self._ddm.run(q)
            ddm_vals = self._ddm.d_dynamical_matrix
            assert ddm_vals is not None
            ddms_along_direction.append(
                derivative_along_direction(ddm_vals, direction) / norm
            )

        freqs = self._eigvals_to_frequencies(eigvals)
        rot_eigvecs = np.empty_like(eigvecs)
        for i, ddm_along_direction in enumerate(ddms_along_direction):
            _, rot_eigvecs[i] = lift_degeneracy(
                freqs[i], eigvecs[i], ddm_along_direction
            )
        return rot_eigvecs

    def _get_dimension_3x3(self) -> NDArray[np.int64]:
        if len(self._dimension) == 3:
            return np.diag(self._dimension)
        if len(self._dimension) == 9:
            return self._dimension.reshape(3, 3)
        raise ValueError(
            f"dimension must have 3 or 9 elements, but has {len(self._dimension)}."
        )

    def _get_displacements(
        self,
        eigvec: NDArray[np.cdouble],
        q: NDArray[np.double],
        amplitude: float,
        argument: float,
    ) -> NDArray[np.cdouble]:
        masses = self._supercell.masses
        assert masses is not None
        u2u_map = self._supercell.u2u_map
        # Index of the atom in the primitive cell for each atom in the supercell.
        primitive_indices = [u2u_map[i] for i in self._supercell.s2u_map]
        # Positions in reduced coordinates of the primitive cell.
        positions = (
            self._supercell.scaled_positions @ self._supercell.supercell_matrix.T
        )
        coefs = np.exp(2j * np.pi * (positions @ q)) / np.sqrt(masses)
        u = eigvec.reshape(-1, 3)[primitive_indices] * coefs[:, None]
        u /= np.sqrt(len(masses))
        phase_factor = self._get_phase_factor(u, argument)
        u *= phase_factor * amplitude

        return u

    def _get_phase_factor(
        self, modulation: NDArray[np.cdouble], argument: float
    ) -> np.cdouble:
        """Return the phase factor that fixes the phase of a modulation.

        Atoms related by lattice translations have displacements of equal
        magnitude.  Among the elements of ``modulation.ravel()`` of the largest
        magnitude, the first one is used as the reference element, instead of
        the one chosen by rounding errors, so that the result does not depend
        on the phase of the eigenvector.

        Parameters
        ----------
        modulation : ndarray
            Complex displacements of the atoms in the supercell before the
            phase is fixed.  The first index is the atom in the supercell, and
            the second index is the Cartesian component.
            ``shape=(supercell atoms, 3)``, ``dtype=np.cdouble``.
        argument : float
            Complex argument of the reference element after the
            multiplication, in degrees.

        Returns
        -------
        np.cdouble
            Phase factor ``exp(i * pi * argument / 180) / (u_ref / |u_ref|)``,
            where ``u_ref`` is the reference element.  The magnitude of the
            phase factor is 1.

        """
        u = modulation.ravel()
        abs_u = np.abs(u)
        index = np.isclose(
            abs_u, abs_u.max(), rtol=PHASE_REFERENCE_RTOL, atol=0
        ).nonzero()[0][0]
        phase_for_zero = u[index] / abs_u[index]
        phase_factor = np.exp(1j * np.pi * argument / 180) / phase_for_zero

        return phase_factor

    def _eigvals_to_frequencies(
        self, eigvals: NDArray[np.double]
    ) -> NDArray[np.double]:
        e = np.array(eigvals, dtype="double")
        return np.sqrt(np.abs(e)) * np.sign(e) * self._factor

    def _get_yaml_lines(self) -> list[str]:
        from phonopy import __version__

        lines = ["phonopy:", f'  version: "{__version__}"']
        lines.append("primitive_cell:")
        lines += self._get_cell_yaml_lines(self._primitive)
        lines.append("supercell:")
        lines.append("  dimension:")
        for v in self._supercell.supercell_matrix:
            lines.append("  - [ %d, %d, %d ]" % tuple(v))
        lines += self._get_cell_yaml_lines(self._supercell)
        inv_lattice = np.linalg.inv(self._supercell.cell.T)

        lines.append("modulations:")
        for u, mode in zip(self._u, self._phonon_modes, strict=True):
            q = mode[0]
            lines.append("- q-position: [ %12.7f, %12.7f, %12.7f ]" % tuple(q))
            lines.append("  band: %d" % (mode[1] + 1))
            lines.append("  amplitude: %f" % mode[2])
            lines.append("  phase: %f" % mode[3])
            lines.append("  displacements:")
            for i, p in enumerate(u):
                for val, axis in zip(p, ("x", "y", "z"), strict=True):
                    lines.append(
                        "  - [ %20.15f, %20.15f ] # %d %s (%f)"
                        % (val.real, val.imag, i + 1, axis, abs(val))
                    )
            lines.append("  fractional_displacements:")
            for i, p in enumerate(np.dot(u, inv_lattice.T)):
                for val, axis in zip(p, ("a", "b", "c"), strict=True):
                    lines.append(
                        "  - [ %20.15f, %20.15f ] # %d %s"
                        % (val.real, val.imag, i + 1, axis)
                    )

        lines.append("phonon:")
        freqs = self._eigvals_to_frequencies(self._eigvals)
        for eigvec, freq, mode in zip(
            self._eigvecs, freqs, self._phonon_modes, strict=True
        ):
            lines.append("- q-position: [ %12.7f, %12.7f, %12.7f ]" % tuple(mode[0]))
            lines.append("  band: %d" % (mode[1] + 1))
            lines.append("  amplitude: %f" % mode[2])
            lines.append("  phase: %f" % mode[3])
            lines.append("  frequency: %15.10f" % freq)
            lines.append("  eigenvector:")
            for j, vec_atom in enumerate(eigvec.reshape(-1, 3)):
                lines.append("  - # atom %d" % (j + 1))
                for val in vec_atom:
                    lines.append(
                        "    - [ %17.14f, %17.14f ] # %f"
                        % (val.real, val.imag, np.angle(val, deg=True))
                    )

        return lines

    def _get_cell_yaml_lines(self, cell: PhonopyAtoms) -> list[str]:
        # Reciprocal basis vectors are the columns of inv(lattice).
        reciprocal_lattice = np.linalg.inv(cell.cell).T
        lines = ["  reciprocal_lattice:"]
        for vec, axis in zip(reciprocal_lattice, ("a*", "b*", "c*"), strict=True):
            lines.append(
                "  - [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,))
            )
        lines += ["  " + line for line in cell.get_yaml_lines()]
        return lines
