"""Phonopy class."""

# Copyright (C) 2015 Atsushi Togo
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

import copy
import lzma
import os
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from phonopy._lang import c_use_openmp, log_dispatch, resolve_lang
from phonopy.exception import ForcesetsNotFoundError
from phonopy.harmonic.displacement import (
    DisplacementDataset,
    Type1DisplacementDataset,
    Type2DisplacementDataset,
    estimate_number_of_snapshots,
    generate_random_displacements,
    generate_systematic_displacements,
)
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    NacParams,
    get_dynamical_matrix,
)
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.harmonic.force_constants import (
    cutoff_force_constants,
    set_tensor_symmetry_PJ,
    show_drift_force_constants,
    symmetrize_compact_force_constants,
    symmetrize_force_constants,
)
from phonopy.interface.calculator import StructureInfo
from phonopy.interface.fc_calculator import get_fc2
from phonopy.interface.mlp import PhonopyMLP
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.pypolymlp import PypolymlpParams
from phonopy.interface.symfc import symmetrize_by_projector
from phonopy.phonon.animation import write_animation
from phonopy.phonon.band_structure import (
    BandStructure,
    BandStructureDict,
    get_band_qpoints_by_seekpath,
)
from phonopy.phonon.dos import (
    ProjectedDos,
    ProjectedDosDict,
    TotalDos,
    TotalDosDict,
)
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.irreps import IrReps
from phonopy.phonon.mesh import IterMesh, IterMeshDict, Mesh, MeshDict
from phonopy.phonon.modulation import Modulation
from phonopy.phonon.moment import PhononMoment
from phonopy.phonon.plot import (
    plot_band_structure,
    plot_band_structure_and_dos,
    plot_projected_dos,
    plot_thermal_displacements,
    plot_thermal_properties,
    plot_total_dos,
)
from phonopy.phonon.qpoints import QpointsDict, QpointsPhonon
from phonopy.phonon.random_displacements import RandomDisplacements
from phonopy.phonon.thermal_displacement import (
    ThermalDisplacementMatrices,
    ThermalDisplacementMatricesDict,
    ThermalDisplacements,
)
from phonopy.phonon.thermal_properties import ThermalProperties, ThermalPropertiesDict
from phonopy.physical_units import get_calculator_physical_units
from phonopy.spectrum.dynamic_structure_factor import DynamicStructureFactor
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    Primitive,
    Supercell,
    get_primitive,
    get_primitive_matrix_with_auto,
    get_supercell,
    isclose,
    shape_supercell_matrix,
    warn_if_primitive_matrix_auto_changed_cell,
)
from phonopy.structure.dataset import forces_in_dataset
from phonopy.structure.mixture import reduce_mixture_forces
from phonopy.structure.symmetry import Symmetry, symmetrize_borns_and_epsilon


class Phonopy:
    """Phonopy main API.

    A ``Phonopy`` instance is created from a unit cell and a supercell
    matrix. It manages displacement generation, force-constant
    construction, and derived phonon quantities such as band structure,
    mesh sampling, DOS, thermal properties, group velocity, irreducible
    representations, modulations, dynamic structure factor, and finite-
    temperature random displacements.

    Most attributes are exposed as ``@property`` accessors documented
    individually below. See :ref:`phonopy_module` for a tutorial-style
    overview of the typical workflow.

    Examples
    --------
    >>> import numpy as np
    >>> from phonopy import Phonopy
    >>> from phonopy.structure.atoms import PhonopyAtoms
    >>> a = 5.404
    >>> unitcell = PhonopyAtoms(
    ...     symbols=["Si"] * 8,
    ...     cell=np.eye(3) * a,
    ...     scaled_positions=[
    ...         [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
    ...         [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
    ...         [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
    ...     ],
    ... )
    >>> phonon = Phonopy(unitcell, supercell_matrix=[2, 2, 2])
    >>> phonon.generate_displacements(distance=0.03)
    >>> # Obtain forces by running an external calculator on
    >>> # phonon.supercells_with_displacements, then:
    >>> # phonon.forces = sets_of_forces
    >>> # phonon.produce_force_constants()
    >>> # phonon.run_mesh([20, 20, 20])
    >>> # phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)

    """

    def __init__(
        self,
        unitcell: PhonopyAtoms,
        supercell_matrix: Sequence[int]
        | Sequence[Sequence[int]]
        | NDArray
        | None = None,
        primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
        | Sequence[Sequence[float]]
        | NDArray
        | None = "auto",
        group_velocity_delta_q: float | None = None,
        symprec: float = 1e-5,
        is_symmetry: bool = True,
        distinguish_symbol_index: bool = False,
        use_SNF_supercell: bool = False,
        hermitianize_dynamical_matrix: bool = True,
        calculator: str | None = None,
        log_level: int = 0,
        lang: Literal["C", "Rust"] = "Rust",
    ):
        """Init method.

        Parameters
        ----------
        unitcell : PhonopyAtoms
            Input unit cell.
        supercell_matrix : array_like, optional
            Transformation matrix to the supercell from the unit cell.
            ``shape=(3, 3)``, ``dtype=int``.
        primitive_matrix : str or array_like, optional
            Transformation matrix to the primitive cell from the unit
            cell. ``shape=(3, 3)``, ``dtype=float``. Default is
            ``"auto"``, which guesses the primitive matrix from crystal
            symmetry. To use the unit cell as the primitive cell
            (identity transformation), pass ``"P"``. ``None`` is treated
            the same as ``"auto"``.
        group_velocity_delta_q : float, optional
            Delta-q distance to calculate group velocity.
        symprec : float, optional
            Symmetry search precision. Default is 1e-5.
        is_symmetry : bool, optional
            Whether to search symmetry of the supercell. Default is
            True.
        distinguish_symbol_index : bool, optional
            When True, atoms whose symbols differ only in the numeric
            suffix ("Cl" vs "Cl1") are treated as distinct species in
            the symmetry search and in the automatic primitive matrix
            determination. By default (False) the suffix is a
            calculator-facing label that does not affect symmetry.
            Default is False.
        use_SNF_supercell : bool, optional
            Build the supercell with the SNF algorithm when True.
            Default is False. The SNF algorithm is faster than the
            original one, but the order of atoms in the supercell can
            be different. Backward compatibility with old data (e.g.,
            force constants) is therefore not guaranteed.
        hermitianize_dynamical_matrix : bool, optional
            Whether to force-Hermitianize the dynamical matrix. Default
            is True, i.e., ``D <- (D + D^H) / 2``.
        calculator : str, optional
            Calculator name such as ``'vasp'``, ``'qe'``, etc. Default
            is None.
        log_level : int, optional
            Log level. Default is 0.
        lang : Literal["C", "Rust"], optional
            Backend implementation for compute-heavy kernels. ``"C"``
            uses the existing C extension; ``"Rust"`` selects the
            experimental phonors backend. Default is ``"Rust"``.

        """
        lang = resolve_lang(lang)
        log_dispatch(lang, "Phonopy.__init__")
        self._symprec = symprec
        self._is_symmetry = is_symmetry
        self._distinguish_symbol_index = distinguish_symbol_index
        self._hermitianize_dynamical_matrix = hermitianize_dynamical_matrix
        self._calculator = calculator
        self._lang: Literal["C", "Rust"] = lang

        self._unit_conversion_factor = get_calculator_physical_units(
            interface_mode=self._calculator
        ).factor
        self._unit_conversion_factor_overridden = False
        self._use_SNF_supercell = use_SNF_supercell
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = unitcell.copy()
        self._supercell_matrix = shape_supercell_matrix(supercell_matrix)
        self._primitive_matrix = get_primitive_matrix_with_auto(
            self._unitcell,
            primitive_matrix,
            symprec=self._symprec,
            distinguish_symbol_index=self._distinguish_symbol_index,
        )
        warn_if_primitive_matrix_auto_changed_cell(
            primitive_matrix, self._primitive_matrix
        )
        self._supercell: Supercell
        self._primitive: Primitive
        self._build_supercell()
        self._build_primitive_cell()

        # Set supercell and primitive symmetry
        self._symmetry: Symmetry
        self._primitive_symmetry: Symmetry
        self._search_symmetry()
        self._search_primitive_symmetry()

        # displacements
        self._dataset: DisplacementDataset | None = None
        self._supercells_with_displacements = None

        # set_force_constants or set_forces
        self._force_constants = None

        # set_dynamical_matrix
        self._dynamical_matrix = None

        # NAC parameters
        self._nac_params: NacParams | None = None

        # MLP
        self._mlp = None
        self._mlp_dataset: Type2DisplacementDataset | None = None

        self._band_structure = None
        self._mesh = None
        self._thermal_properties = None
        self._thermal_displacements = None
        self._thermal_displacement_matrices = None
        self._dynamic_structure_factor = None
        self._pdos = None
        self._total_dos = None
        self._modulation = None
        self._irreps = None
        self._random_displacements: RandomDisplacements | None = None
        self._moment = None
        self._qpoints = None

        self._group_velocity = None
        self._gv_delta_q = group_velocity_delta_q

    @property
    def version(self) -> str:
        """Return phonopy release version number."""
        from phonopy import __version__

        return __version__

    @property
    def primitive(self) -> Primitive:
        """Return primitive cell."""
        return self._primitive

    @property
    def unitcell(self) -> PhonopyAtoms:
        """Return input unit cell."""
        return self._unitcell

    @property
    def supercell(self) -> Supercell:
        """Return supercell."""
        return self._supercell

    @property
    def symmetry(self) -> Symmetry:
        """Return symmetry of the supercell."""
        return self._symmetry

    @property
    def primitive_symmetry(self) -> Symmetry:
        """Return symmetry of the primitive cell."""
        return self._primitive_symmetry

    @property
    def supercell_matrix(self) -> NDArray[np.int64]:
        """Return transformation matrix to supercell from unit cell.

        Supercell matrix with respect to the unit cell.
        ``shape=(3, 3)``, ``dtype='int64'``, ``order='C'``.

        """
        return self._supercell_matrix

    @property
    def primitive_matrix(self) -> NDArray[np.double]:
        """Return transformation matrix to primitive cell from unit cell.

        Primitive matrix with respect to the unit cell.
        ``shape=(3, 3)``, ``dtype='double'``, ``order='C'``.

        """
        return self._primitive_matrix

    @property
    def unit_conversion_factor(self) -> float:
        """Return phonon frequency unit conversion factor.

        This factor converts ``sqrt(<force> / <distance> / <AMU>) / 2pi /
        1e12`` to another preferred phonon frequency unit. It should
        convert to THz (ordinary frequency) when calculating various
        phonon properties that assume input phonon frequencies are in
        THz units. When only frequencies are necessary as output, this
        factor may be used to get results in other units. The default
        frequency conversion factor is the one to THz for displacements
        in Angstroms and forces in eV/Angstrom.

        """
        return self._unit_conversion_factor

    @unit_conversion_factor.setter
    def unit_conversion_factor(self, unit_conversion_factor: float):
        self._unit_conversion_factor = unit_conversion_factor
        self._unit_conversion_factor_overridden = True
        self._invalidate_derived("factor")

    @property
    def calculator(self) -> str | None:
        """Return calculator name such as ``'vasp'``, ``'qe'``, etc."""
        return self._calculator

    @property
    def lang(self) -> Literal["C", "Rust"]:
        """Return the selected backend implementation.

        Literal["C", "Rust"]
            "C" uses the C extension; "Rust" uses the experimental
            phonors backend.

        """
        return self._lang

    @property
    def dataset(self) -> DisplacementDataset | None:
        """Return displacement-force dataset.

        Dataset containing information of displacements in supercells.
        This optionally contains energies and forces of respective
        supercells. The format is either one of two types.

        **Type 1. One atomic displacement in each supercell**::

            {'natom': number of atoms in supercell,
             'first_atoms': [
               {'number': atom index of displaced atom,
                'displacement': displacement in Cartesian coordinates,
                'forces': forces on atoms in supercell,
                'supercell_energy': energy of supercell},
               {...}, ...]}

        Elements of the list accessed by ``'first_atoms'`` correspond to
        each displaced supercell. Each displaced supercell contains only
        one displacement. ``dict['first_atoms']['forces']`` gives atomic
        forces in each displaced supercell.

        **Type 2. All atomic displacements in each supercell**::

            {'displacements': ndarray, dtype='double', order='C',
                              shape=(supercells, natom, 3),
             'forces': ndarray, dtype='double', order='C',
                              shape=(supercells, natom, 3),
             'supercell_energies': ndarray, dtype='double'}

        To set in type 2, displacements and forces can be given by numpy
        arrays with different shape that can be reshaped to
        ``(supercells, natom, 3)``.

        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: DisplacementDataset | None) -> None:
        if dataset is None:
            self._dataset = None
        elif "first_atoms" in dataset:
            self._dataset = copy.deepcopy(dataset)
        elif "displacements" in dataset:
            self._dataset = None
            self.displacements = dataset["displacements"]
            if "forces" in dataset:
                self.forces = dataset["forces"]
            if "supercell_energies" in dataset:
                self.supercell_energies = dataset["supercell_energies"]
        else:
            raise RuntimeError("Data format of dataset is wrong.")

        self._supercells_with_displacements = None
        self._invalidate_derived("dataset_inputs")

    @property
    def mlp_dataset(self) -> Type2DisplacementDataset | None:
        """Return displacement-force dataset used to train an MLP.

        The supercell matrix matches that of the usual
        displacement-force dataset. Only the type-2 format is supported;
        the dict must contain ``"displacements"``, ``"forces"``, and
        ``"supercell_energies"``.

        """
        return self._mlp_dataset

    @mlp_dataset.setter
    def mlp_dataset(self, mlp_dataset: Type2DisplacementDataset | None) -> None:
        if mlp_dataset is None:
            self._mlp_dataset = None
            return
        if "displacements" not in mlp_dataset:
            raise RuntimeError("Displacements have to be given.")
        if "forces" not in mlp_dataset:
            raise RuntimeError("Forces have to be given.")
        if "supercell_energy" in mlp_dataset:
            raise RuntimeError("Supercell energies have to be given.")
        if len(mlp_dataset["displacements"]) != len(mlp_dataset["forces"]):
            raise RuntimeError("Length of displacements and forces are different.")
        if len(mlp_dataset["displacements"]) != len(
            mlp_dataset["supercell_energies"]  # type: ignore[typeddict-item]
        ):
            raise RuntimeError(
                "Length of displacements and supercell_energies are different."
            )
        self._mlp_dataset = mlp_dataset

    @property
    def mlp(self) -> PhonopyMLP | None:
        """Setter and getter of the ``PhonopyMLP`` machine-learning potential."""
        return self._mlp

    @mlp.setter
    def mlp(self, mlp: PhonopyMLP | None) -> None:
        self._mlp = mlp

    @property
    def displacements(self) -> NDArray[np.double] | list:
        """Getter and setter of displacements in supercells.

        There are two types of displacement dataset; see the docstring
        of :attr:`dataset` for the type-1 and type-2 formats. The
        returned displacements depend on the dataset type:

        **Type 1** (list of list)
            Each inner list has 4 elements, e.g.
            ``[32, 0.01, 0.0, 0.0]``. The first element is the
            supercell atom index starting with 0; the remaining three
            elements give the displacement in Cartesian coordinates.

        **Type 2** (ndarray)
            Displacements of all atoms in all supercells in Cartesian
            coordinates.
            ``shape=(supercells, natom, 3)``, ``dtype='double'``.

        The setter accepts only the type-2 format
        (``shape=(supercells, natom, 3)``, ``dtype='double'``,
        ``order='C'``).

        """
        if self._dataset is None:
            raise RuntimeError("Displacement-force dataset is not set.")

        disps = []
        if "first_atoms" in self._dataset:
            for disp in self._dataset["first_atoms"]:
                x = disp["displacement"]
                disps.append([disp["number"], x[0], x[1], x[2]])
        elif "displacements" in self._dataset:
            disps = self._dataset["displacements"]

        return disps

    @displacements.setter
    def displacements(
        self,
        displacements: Sequence[Sequence[Sequence[float]]]
        | Sequence[NDArray[np.double]]
        | NDArray[np.double],
    ) -> None:
        disp = np.array(displacements, dtype="double", order="C")
        if disp.ndim != 3 or disp.shape[1:] != (len(self._supercell), 3):
            raise RuntimeError("Array shape of displacements is incorrect.")
        if self._dataset is not None and "first_atoms" in self._dataset:
            raise RuntimeError(
                "Setting displacements to type-1 dataset is not supported."
            )
        self._dataset = {"displacements": disp}
        self._supercells_with_displacements = None
        self._invalidate_derived("dataset_inputs")

    @property
    def force_constants(self) -> NDArray[np.double] | None:
        """Getter and setter of supercell force constants.

        Force constants matrix.

        **Getter** returns an ``ndarray`` with one of two shapes:

        - full: ``shape=(atoms in supercell, atoms in supercell, 3, 3)``
        - compact: ``shape=(atoms in primitive cell, atoms in supercell, 3, 3)``

        with ``dtype='double'``, ``order='C'``.

        **Setter** accepts any array-like. If given as an own-contiguous
        ndarray with ``order='C'`` and ``dtype='double'``, an internal
        copy of the data is avoided and some computational resources are
        saved. Expected shape is
        ``(atoms in supercell, atoms in supercell, 3, 3)``,
        ``dtype='double'``.

        """
        return self._force_constants

    @force_constants.setter
    def force_constants(self, force_constants: NDArray[np.double] | None) -> None:
        if force_constants is None:
            self._force_constants = None
            self._invalidate_derived("dm_inputs")
            return

        self._force_constants = np.array(force_constants, dtype="double", order="C")
        fc_shape = self._force_constants.shape
        if fc_shape[0] != fc_shape[1]:
            if len(self._primitive) != fc_shape[0]:
                msg = (
                    "Force constants shape disagrees with crystal "
                    "structure setting. This may be due to "
                    "PRIMITIVE_AXIS."
                )
                raise RuntimeError(msg)

        self._invalidate_derived("dm_inputs")
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def set_force_constants_zero_with_radius(self, cutoff_radius: float) -> None:
        """Set zero to force constants within cutoff radius."""
        if self._force_constants is None:
            raise RuntimeError("Force constants are not set.")

        cutoff_force_constants(
            self._force_constants,
            self._supercell,
            self._primitive,
            cutoff_radius,
            symprec=self._symprec,
        )
        self._invalidate_derived("dm_inputs")
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    @property
    def supercell_energies(self) -> NDArray[np.double]:
        """Return energies of supercells.

        Returns
        -------
        ndarray
            ``shape=(len(supercells),)``, ``dtype='double'``.

        """
        return self._get_forces_energies(target="supercell_energies")

    @supercell_energies.setter
    def supercell_energies(
        self, set_of_energies: Sequence[float] | NDArray[np.double]
    ) -> None:
        self._set_forces_energies(set_of_energies, target="supercell_energies")
        self._invalidate_derived("dataset_inputs")

    @property
    def forces(self) -> NDArray[np.double]:
        """Return forces of supercells.

        A set of atomic forces in displaced supercells. The order of
        displaced supercells has to match with that in the displacement
        dataset.

        The getter returns an ``ndarray`` and the setter accepts any
        array-like with::

            shape=(supercells with displacements, atoms in supercell, 3),
            dtype='double', order='C'.

        That is::

            [[[f_1x, f_1y, f_1z], [f_2x, f_2y, f_2z], ...],  # first supercell
             [[f_1x, f_1y, f_1z], [f_2x, f_2y, f_2z], ...],  # second supercell
             ...]

        """
        return self._get_forces_energies(target="forces")

    @forces.setter
    def forces(
        self,
        sets_of_forces: NDArray[np.double]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]],
    ) -> None:
        self._set_forces_energies(sets_of_forces, target="forces")
        self._invalidate_derived("dataset_inputs")

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return the ``DynamicalMatrix`` instance.

        This is the dynamical-matrix builder object, not the matrix
        itself. Call ``dm.run(q)`` and then access ``dm.dynamical_matrix``
        to obtain the matrix at a given q-point.

        """
        return self._dynamical_matrix

    @property
    def nac_params(self) -> NacParams | None:
        """Getter and setter of parameters for non-analytical term correction.

        A ``dict`` (typed as :class:`NacParams`) with the following
        entries:

        ``'born'`` : ndarray
            Born effective charges.
            ``shape=(primitive cell atoms, 3, 3)``, ``dtype='double'``,
            ``order='C'``.
        ``'dielectric'`` : ndarray
            Dielectric constant tensor.
            ``shape=(3, 3)``, ``dtype='double'``, ``order='C'``.
        ``'factor'`` : float, optional
            Unit conversion factor. When omitted, the value for the
            calculator interface is used.
        ``'method'`` : str, optional
            Method to calculate NAC, either ``'gonze'`` (default) or
            ``'wang'``.

        """
        return self._nac_params

    @nac_params.setter
    def nac_params(self, nac_params: NacParams | None) -> None:
        self._nac_params = nac_params
        self._invalidate_derived("dm_inputs")
        if self._force_constants is not None:
            self._set_dynamical_matrix()

    @property
    def supercells_with_displacements(self) -> list[PhonopyAtoms] | None:
        """Return supercells with displacements as a list of ``PhonopyAtoms``.

        Generated by :meth:`generate_displacements`.

        """
        if self._dataset is None:
            return None
        else:
            if self._supercells_with_displacements is None:
                self._build_supercells_with_displacements()
            return self._supercells_with_displacements

    @property
    def mesh_numbers(self) -> NDArray[np.int64] | None:
        """Return sampling mesh numbers in reciprocal space.

        ``shape=(3,)``, ``dtype='int64'``. ``None`` if ``run_mesh`` /
        ``init_mesh`` has not been called.

        """
        if self._mesh is None:
            return None
        else:
            return self._mesh.mesh_numbers

    @property
    def qpoints(self) -> QpointsPhonon | None:
        """Return QpointsPhonon instance."""
        return self._qpoints

    @property
    def band_structure(self) -> BandStructure | None:
        """Return BandStructure instance."""
        return self._band_structure

    @property
    def group_velocity(self) -> GroupVelocity | None:
        """Return GroupVelocity instance."""
        return self._group_velocity

    @property
    def mesh(self) -> Mesh | IterMesh | None:
        """Return Mesh or IterMesh instance."""
        return self._mesh

    @property
    def random_displacements(self) -> RandomDisplacements | None:
        """Return RandomDisplacements instance."""
        return self._random_displacements

    @property
    def dynamic_structure_factor(self) -> DynamicStructureFactor | None:
        """Return DynamicStructureFactor instance."""
        return self._dynamic_structure_factor

    @property
    def thermal_properties(self) -> ThermalProperties | None:
        """Return ThermalProperties instance."""
        return self._thermal_properties

    @property
    def thermal_displacements(self) -> ThermalDisplacements | None:
        """Return ThermalDisplacements instance."""
        return self._thermal_displacements

    @property
    def thermal_displacement_matrices(self) -> ThermalDisplacementMatrices | None:
        """Return ThermalDisplacementMatrices instance."""
        return self._thermal_displacement_matrices

    @property
    def irreps(self) -> IrReps | None:
        """Return IrReps instance."""
        return self._irreps

    @property
    def modulation(self) -> Modulation | None:
        """Return Modulation instance."""
        return self._modulation

    @property
    def moment(self) -> PhononMoment | None:
        """Return PhononMoment instance."""
        return self._moment

    @property
    def total_dos(self) -> TotalDos | None:
        """Return TotalDos instance."""
        return self._total_dos

    @property
    def projected_dos(self) -> ProjectedDos | None:
        """Return ``ProjectedDos`` instance."""
        return self._pdos

    @property
    def masses(self) -> NDArray[np.double]:
        """Getter and setter of masses of primitive cell atoms.

        By setter, masses of supercell and unit cell atoms are also updated.

        """
        return self._primitive.masses

    @masses.setter
    def masses(self, masses: Sequence[float] | NDArray[np.double]) -> None:
        p_masses = np.array(masses)
        self._primitive.masses = p_masses
        p2p_map = self._primitive.p2p_map
        s_masses = p_masses[[p2p_map[x] for x in self._primitive.s2p_map]]
        self._supercell.masses = s_masses
        u2s_map = self._supercell.u2s_map
        u_masses = s_masses[u2s_map]
        self._unitcell.masses = u_masses
        self._invalidate_derived("dm_inputs")
        if self._force_constants is not None:
            self._set_dynamical_matrix()

    def generate_displacements(
        self,
        distance: float | None = None,
        is_plusminus: Literal["auto"] | bool = "auto",
        is_diagonal: bool = True,
        is_trigonal: bool = False,
        number_of_snapshots: int | Literal["auto"] | None = None,
        random_seed: int | None = None,
        temperature: float | None = None,
        cutoff_frequency: float | None = None,
        max_distance: float | None = None,
        distance_per_atom: bool = False,
        number_estimation_factor: float | None = None,
    ) -> None:
        """Generate displacement dataset and store it in Phonopy.dataset.

        This method selects one of three generators and stores its result;
        the generators hold the details of each mode.

        Systematic displacements for the built-in finite-difference method
        are the default. Random displacements, which require an external
        force-constants calculator (symfc, ALM), are selected by giving
        ``number_of_snapshots``, and come in two flavours: random directions
        at a fixed or randomly drawn distance, or displacements sampled from
        the canonical ensemble of harmonic phonons at ``temperature``. The
        latter needs force constants to be set already.

        See Also
        --------
        phonopy.harmonic.displacement.generate_systematic_displacements
        phonopy.harmonic.displacement.generate_random_displacements
        phonopy.harmonic.displacement.estimate_number_of_snapshots
        Phonopy.init_random_displacements
        Phonopy.get_random_displacements_at_temperature

        Parameters
        ----------
        distance : float, optional
            Displacement distance in the unit of the crystal structure.
            Default is 0.01. With ``max_distance`` and without
            ``temperature``, it is the floor of the random distance rather
            than the distance itself.
        is_plusminus : 'auto', True, or False, optional
            For each atom, generate displacements in one direction (False),
            in both directions (True), or in both directions only when
            symmetry requires it (``'auto'``). Default is ``'auto'``.
        is_diagonal : bool, optional
            Systematic displacements only. When False, displace only along
            basis vectors. Default is True.
        is_trigonal : bool, optional
            Exists only for testing purposes. Default is False.
        number_of_snapshots : int, "auto", or None, optional
            Number of supercells with random displacements. When ``"auto"``,
            it is estimated with symfc. None selects systematic
            displacements. Default is None.
        random_seed : int or None, optional
            Random seed, used when in ``[0, 2**32)``. Default is None.
        temperature : float or None, optional
            When given, random displacements are sampled from the canonical
            ensemble of harmonic phonons at this temperature. Default is
            None.
        cutoff_frequency : float or None, optional
            Finite-temperature generation only. Phonon modes whose absolute
            frequencies are below this value are excluded. Default is None.
        max_distance : float or None, optional
            Upper bound of the displacement distance. With ``temperature``,
            a displacement longer than this is renormalized to it; without
            ``temperature``, one distance per supercell is drawn from
            ``[0, max_distance)`` and floored at ``distance``. Default is
            None.
        distance_per_atom : bool, optional
            Requires ``max_distance`` and is incompatible with
            ``temperature``, which already draws per atom. Draw the random
            distance per atom rather than per supercell, uniformly over
            ``[distance, max_distance)`` and without the weight at
            ``distance``. Default is False.
        number_estimation_factor : float, optional
            Safety factor on the symfc estimate used by
            ``number_of_snapshots="auto"``. Default is None.

        """
        if distance_per_atom and max_distance is None:
            raise ValueError("distance_per_atom requires max_distance.")
        if number_of_snapshots is not None and (
            number_of_snapshots == "auto" or number_of_snapshots > 0
        ):
            if number_of_snapshots == "auto":
                _number_of_snapshots = estimate_number_of_snapshots(
                    self._supercell,
                    self._symmetry,
                    max_distance=max_distance,
                    number_estimation_factor=number_estimation_factor,
                )
            else:
                _number_of_snapshots = number_of_snapshots
            if random_seed is not None and 0 <= random_seed < 2**32:
                _random_seed = random_seed
            else:
                _random_seed = None
            if temperature is None:
                self.dataset = generate_random_displacements(
                    self._supercell,
                    _number_of_snapshots,
                    distance=distance,
                    is_plusminus=(is_plusminus is True),
                    random_seed=_random_seed,
                    max_distance=max_distance,
                    distance_per_atom=distance_per_atom,
                )
            else:
                if distance_per_atom:
                    raise ValueError(
                        "distance_per_atom is incompatible with temperature; "
                        "the canonical ensemble already gives each atom its "
                        "own displacement."
                    )
                displacement_dataset, random_displacements = (
                    self._generate_finite_temperature_displacement_dataset(
                        _number_of_snapshots,
                        temperature=temperature,
                        is_plusminus=is_plusminus,
                        random_seed=_random_seed,
                        cutoff_frequency=cutoff_frequency,
                        max_distance=max_distance,
                    )
                )
                self.dataset = displacement_dataset
                # The self.dataset assignment above clears
                # self._random_displacements via _invalidate_derived. Restore
                # the finite-temperature instance so callers can read its
                # q-points, frequencies, and integrated modes.
                self._random_displacements = random_displacements
        else:
            self.dataset = generate_systematic_displacements(
                self._supercell,
                self._symmetry,
                distance=distance,
                is_plusminus=is_plusminus,
                is_diagonal=is_diagonal,
                is_trigonal=is_trigonal,
                log_level=self._log_level,
            )

    def _generate_finite_temperature_displacement_dataset(
        self,
        number_of_snapshots: int,
        temperature: float,
        is_plusminus: Literal["auto"] | bool,
        random_seed: int | None,
        cutoff_frequency: float | None,
        max_distance: float | None,
    ) -> tuple[Type2DisplacementDataset, RandomDisplacements]:
        """Build a type-2 dataset of random displacements at finite temperature.

        This is a helper for :meth:`generate_displacements`; see there for
        the meaning of the parameters.

        Returns
        -------
        tuple[Type2DisplacementDataset, RandomDisplacements]
            The generated type-2 displacement dataset and the
            RandomDisplacements instance used to create it.

        Note
        ----
        This method sets self._random_displacements as a side effect:
        init_random_displacements creates the instance and
        get_random_displacements_at_temperature runs it. The instance
        returned here is that same object.

        IMPORTANT: the caller must put the returned instance back into
        self._random_displacements *after* assigning self.dataset. That
        assignment triggers _invalidate_derived, which resets
        self._random_displacements to None. Without the restore, the
        q-points, frequencies, and integrated modes computed here are
        lost, and Phonopy.random_displacements returns None even though
        generation succeeded.

        """
        self.init_random_displacements(
            cutoff_frequency=cutoff_frequency, max_distance=max_distance
        )
        d = self.get_random_displacements_at_temperature(
            temperature,
            number_of_snapshots,
            is_plusminus=(is_plusminus is True),
            random_seed=random_seed,
        )
        assert self._random_displacements is not None
        dataset: Type2DisplacementDataset = {"displacements": d}
        if random_seed is not None:
            dataset["random_seed"] = random_seed
        return dataset, self._random_displacements

    def produce_force_constants(
        self,
        forces: NDArray[np.double]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]]
        | None = None,  # deprecated, use Phonopy.forces setter instead
        calculate_full_force_constants: bool = True,
        fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
        fc_calculator_options: str | None = None,
        show_drift: bool = True,
        fc_calculator_log_level: int | None = None,
    ) -> None:
        """Compute supercell force constants from forces-displacements dataset.

        Supercell force constants are computed from forces and displacements.
        As the default behaviour, those stored in dataset are used. But
        with setting ``forces``, this set of forces and the set of
        displacements stored in the dataset are used for the computation.

        Parameters
        ----------
        forces : array_like, optional
            Deprecated. Use the :attr:`forces` setter instead. Default
            is None.
        calculate_full_force_constants : bool, optional
            When True, the full force-constants matrix is stored. When
            False, the compact force-constants matrix is stored. See
            the docstring of :attr:`force_constants` for details.
            Default is True.
        fc_calculator : {"traditional", "symfc", "alm", None}, optional
            Force constants calculator backend. ``"traditional"`` uses
            phonopy's built-in least-squares fit. ``"symfc"`` and
            ``"alm"`` delegate to external packages. Default is None
            (use the traditional backend).
        fc_calculator_options : str, optional
            Backend-specific options string passed to the chosen
            ``fc_calculator``. See the docstring of
            :func:`phonopy.interface.fc_calculator.get_fc2`. Default is
            None.
        show_drift : bool, optional
            Display residual translational drift of force constants
            after computation. Default is True.
        fc_calculator_log_level : int, optional
            Log level for the force-constants calculator. Default is
            None (use the Phonopy instance's ``log_level``).

        """
        if forces is not None:
            warnings.warn(
                (
                    "forces parameter of produce_force_constants is deprecated. "
                    "Use Phonopy.forces setter instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            self.forces = forces

        if self._dataset is None:
            raise RuntimeError("Displacement dataset is not set.")

        if fc_calculator_log_level is None:
            fc_log_level = self._log_level
        else:
            fc_log_level = fc_calculator_log_level

        # A primitive check if 'forces' key is in displacement_dataset.
        if "first_atoms" in self._dataset:
            for disp in self._dataset["first_atoms"]:
                if "forces" not in disp:
                    raise ForcesetsNotFoundError("Force sets are not yet set.")
        elif "forces" not in self._dataset:
            raise ForcesetsNotFoundError("Force sets are not yet set.")

        self._run_force_constants_from_forces(
            is_compact_fc=not calculate_full_force_constants,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            log_level=fc_log_level,
        )

        if show_drift and self._log_level:
            assert self._force_constants is not None
            show_drift_force_constants(
                self._force_constants, primitive=self._primitive, lang=self._lang
            )

        self._invalidate_derived("dm_inputs")
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def symmetrize_force_constants(
        self, level: int = 1, show_drift: bool = True, use_symfc_projector: bool = False
    ) -> None:
        """Symmetrize force constants.

        Two schemes are available.

        - Default (``use_symfc_projector=False``): translational and
          permutation symmetries are applied successively, not
          simultaneously. The resulting force constants can break
          space-group symmetry slightly.
        - ``use_symfc_projector=True``: the symfc projector imposes
          space-group, translational, and permutation symmetries
          simultaneously in a single shot.

        Parameters
        ----------
        level : int, optional
            Number of times the successive (translation -> permutation)
            application is repeated. Only used when
            ``use_symfc_projector=False``. Default is 1.
        show_drift : bool, optional
            Display residual drift when True. Default is True.
        use_symfc_projector : bool, optional
            If True, force constants are symmetrized by the symfc
            projector instead of the traditional approach. Default is
            False.

        """
        if self._force_constants is None:
            raise RuntimeError("Force constants have not been produced yet.")

        if use_symfc_projector:
            self._force_constants = symmetrize_by_projector(
                self._supercell,
                self._force_constants,
                2,
                primitive=self._primitive,
                log_level=self._log_level,
            )
        else:
            if self._force_constants.shape[0] == self._force_constants.shape[1]:
                symmetrize_force_constants(
                    self._force_constants, level=level, lang=self._lang
                )
            else:
                symmetrize_compact_force_constants(
                    self._force_constants,
                    self._primitive,
                    level=level,
                    lang=self._lang,
                )

        if show_drift and self._log_level:
            if use_symfc_projector:
                print("Max drift after symmetrization by symfc projector: ", end="")
            else:
                print("Max drift after traditional symmetrization: ", end="")
            show_drift_force_constants(
                self._force_constants,
                primitive=self._primitive,
                values_only=True,
                lang=self._lang,
            )

        self._invalidate_derived("dm_inputs")
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def symmetrize_force_constants_by_space_group(
        self, show_drift: bool = True
    ) -> None:
        """Symmetrize force constants using space group operations.

        Space group operations except for pure translations are applied
        to force constants.

        Parameters
        ----------
        show_drift : bool, optional
            Drift forces are displayed when True. Default is True.

        """
        if self._force_constants is None:
            raise RuntimeError("Force constants have not been produced yet.")

        set_tensor_symmetry_PJ(
            self._force_constants,
            self._supercell.cell.T,
            self._supercell.scaled_positions,
            self._symmetry,
        )

        if show_drift and self._log_level:
            sys.stdout.write("Max drift after symmetrization by space group: ")
            show_drift_force_constants(
                self._force_constants,
                primitive=self._primitive,
                values_only=True,
                lang=self._lang,
            )

        self._invalidate_derived("dm_inputs")
        if self._primitive.masses is not None:
            self._set_dynamical_matrix()

    def develop_mlp(
        self,
        params: PypolymlpParams | dict | str | None = None,
        test_size: float = 0.1,
        log_level: int | None = None,
    ) -> None:
        """Develop machine learning potential.

        Parameters
        ----------
        params : PypolymlpParams or dict, optional
            Parameters for developing MLP. Default is None. When dict is given,
            PypolymlpParams instance is created from the dict.
        test_size : float, optional
            Training and test data are split by this ratio. test_size=0.1
            means the first 90% of the data is used for training and the rest
            is used for test. Default is 0.1.

        """
        if self._mlp_dataset is None:
            raise RuntimeError("MLP dataset is not set.")

        if log_level is None:
            self._mlp = PhonopyMLP(log_level=self._log_level)
        else:
            self._mlp = PhonopyMLP(log_level=log_level)
        self._mlp.develop(
            self._mlp_dataset,
            self._supercell,
            params=params,
            test_size=test_size,
        )

    def save_mlp(self, filename: str | os.PathLike | None = None) -> None:
        """Save machine learning potential."""
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        self._mlp.save(filename=filename)

    def load_mlp(self, filename: str | os.PathLike | None = None) -> None:
        """Load machine learning potential."""
        self._mlp = PhonopyMLP(log_level=self._log_level)
        self._mlp.load(filename=filename)

    def evaluate_mlp(self):
        """Evaluate machine learning potential.

        This method calculates the supercell energies and forces from the MLP
        for the displacements in self._dataset of type 2. The results are stored
        in self._dataset.

        The displacements may be generated by the produce_force_constants method
        with number_of_snapshots > 0. With MLP, a small distance parameter, such
        as 0.01, can be numerically stable for the computation of force
        constants.

        """
        if self._mlp is None:
            raise RuntimeError("MLP is not developed yet.")

        if self.supercells_with_displacements is None:
            raise RuntimeError("Displacements are not set. Run generate_displacements.")

        energies, forces, _ = self._mlp.evaluate(self.supercells_with_displacements)
        self.supercell_energies = energies
        self.forces = forces

    #####################
    # Phonon properties #
    #####################

    # Single q-point
    def get_dynamical_matrix_at_q(
        self, q: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.cdouble]:
        """Calculate dynamical matrix at a given q-point.

        Parameters
        ----------
        q : array_like
            A q-vector. ``shape=(3,)``, ``dtype='double'``.

        Returns
        -------
        ndarray
            Dynamical matrix. ``shape=(bands, bands)``, complex dtype
            (``"c%d" % (np.dtype('double').itemsize * 2)``),
            ``order='C'``.

        .. deprecated::
            Use ``run_qpoints([q], with_dynamical_matrices=True)`` and
            the ``dynamical_matrices`` attribute of the returned
            ``QpointsPhonon`` object instead.

        """
        warnings.warn(
            "get_dynamical_matrix_at_q() is deprecated. Use "
            "run_qpoints([q], with_dynamical_matrices=True); the returned "
            "QpointsPhonon object provides dynamical_matrices[0].",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        assert self._dynamical_matrix.dynamical_matrix is not None
        return self._dynamical_matrix.dynamical_matrix

    def get_frequencies(
        self, q: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Calculate phonon frequencies at a given q-point.

        Parameters
        ----------
        q : array_like
            A q-vector. ``shape=(3,)``, ``dtype='double'``.

        Returns
        -------
        ndarray
            Phonon frequencies. Imaginary frequencies are represented by
            negative real numbers. ``shape=(bands,)``, ``dtype='double'``.

        .. deprecated::
            Use ``run_qpoints([q])`` and the ``frequencies`` attribute
            of the returned ``QpointsPhonon`` object instead.

        """
        warnings.warn(
            "get_frequencies() is deprecated. Use run_qpoints([q]); the "
            "returned QpointsPhonon object provides frequencies[0].",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        dm = self._dynamical_matrix.dynamical_matrix
        frequencies = []
        for eig in np.linalg.eigvalsh(dm).real:  # type: ignore
            if eig < 0:
                frequencies.append(-np.sqrt(-eig))
            else:
                frequencies.append(np.sqrt(eig))

        return (
            np.array(frequencies, dtype="double", order="C")
            * self._unit_conversion_factor
        )

    def get_frequencies_with_eigenvectors(
        self, q: Sequence[float] | NDArray[np.double]
    ) -> tuple[NDArray[np.double], NDArray[np.cdouble]]:
        """Calculate phonon frequencies and eigenvectors at a given q-point.

        Parameters
        ----------
        q : array_like
            A q-vector. ``shape=(3,)``.

        Returns
        -------
        frequencies : ndarray
            Phonon frequencies. Imaginary frequencies are represented by
            negative real numbers. ``shape=(bands,)``, ``dtype='double'``,
            ``order='C'``.
        eigenvectors : ndarray
            Phonon eigenvectors. ``shape=(bands, bands)``, complex
            dtype (``"c%d" % (np.dtype('double').itemsize * 2)``),
            ``order='C'``.

        .. deprecated::
            Use ``run_qpoints([q], with_eigenvectors=True)`` and the
            ``frequencies`` / ``eigenvectors`` attributes of the
            returned ``QpointsPhonon`` object instead.

        """
        warnings.warn(
            "get_frequencies_with_eigenvectors() is deprecated. Use "
            "run_qpoints([q], with_eigenvectors=True); the returned "
            "QpointsPhonon object provides frequencies[0] and "
            "eigenvectors[0].",
            DeprecationWarning,
            stacklevel=2,
        )
        self._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._dynamical_matrix.run(q)
        dm = self._dynamical_matrix.dynamical_matrix
        frequencies = []
        eigvals, eigenvectors = np.linalg.eigh(dm)  # type: ignore
        frequencies = []
        for eig in eigvals:
            if eig < 0:
                frequencies.append(-np.sqrt(-eig))
            else:
                frequencies.append(np.sqrt(eig))

        return np.array(
            frequencies, dtype="double", order="C"
        ) * self._unit_conversion_factor, eigenvectors

    # Band structure
    def run_band_structure(
        self,
        paths: Sequence[NDArray[np.double]] | Sequence[Sequence[float]],
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
        is_band_connection: bool = False,
        path_connections: Sequence[bool] | None = None,
        labels: Sequence[str] | None = None,
        is_legacy_plot: bool = False,
    ) -> BandStructure:
        """Run phonon band structure calculation.

        Parameters
        ----------
        paths : list of array_like
            Sets of q-points defining each band path. The number of
            q-points can differ between paths. Each array has shape
            ``(qpoints, 3)``.
        with_eigenvectors : bool, optional
            Whether eigenvectors are calculated. Default is False.
        with_group_velocities : bool, optional
            Whether group velocities are calculated. Default is False.
        is_band_connection : bool, optional
            Whether to connect bands across neighboring q-points by
            comparing the similarity of their eigenvectors. This
            sometimes fails. Default is False.
        path_connections : list of bool, optional
            Used only when plotting; indicates whether each path is
            connected to the next path (i.e., False means there is a
            jump of q-points between them). The number of elements
            matches that of ``paths``. Default is None.
        labels : list of str, optional
            Used only when plotting; labels of the end points of each
            path. The number of labels equals
            ``(2 - np.array(path_connections)).sum()``.
        is_legacy_plot : bool, optional
            Use the old-style band-structure plot. Default is False.

        Returns
        -------
        BandStructure
            The calculated band structure. The same object is also
            accessible through the ``band_structure`` property.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        self._band_structure = BandStructure(
            paths,
            self._dynamical_matrix,
            with_eigenvectors=with_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=group_velocity,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=is_legacy_plot,
            factor=self._unit_conversion_factor,
        )
        return self._band_structure

    def get_band_structure_dict(self) -> BandStructureDict:
        """Return calculated band structures.

        Returns
        -------
        dict
            Keys are ``qpoints``, ``distances``, ``frequencies``,
            ``eigenvectors``, and ``group_velocities``. Each value is a
            list containing the property along one band path. The number
            of q-points along one path can be different from that of
            other paths. Each per-path entry is an ``ndarray``:

            ``qpoints[i]`` : ndarray
                q-points in reduced coordinates of reciprocal space
                without 2 pi.
                ``shape=(q-points, 3)``, ``dtype='double'``.
            ``distances[i]`` : ndarray
                Distances in reciprocal space along paths.
                ``shape=(q-points,)``, ``dtype='double'``.
            ``frequencies[i]`` : ndarray
                Phonon frequencies. Imaginary frequencies are represented
                by negative real numbers.
                ``shape=(q-points, bands)``, ``dtype='double'``.
            ``eigenvectors[i]`` : ndarray
                Phonon eigenvectors. ``None`` if eigenvectors are not
                stored.
                ``shape=(q-points, bands, bands)``,
                ``dtype=complex`` (``"c%d" % (np.dtype('double').itemsize * 2)``),
                ``order='C'``.
            ``group_velocities[i]`` : ndarray
                Phonon group velocities. ``None`` if group velocities
                are not calculated.
                ``shape=(q-points, bands, 3)``, ``dtype='double'``.

        .. deprecated::
            Use the ``band_structure`` property instead.

        """
        warnings.warn(
            "get_band_structure_dict() is deprecated. Use the band_structure "
            "property to access the BandStructure result object; its "
            "qpoints, distances, frequencies, eigenvectors, and "
            "group_velocities attributes replace the dict keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._band_structure is None:
            msg = "Phonopy.run_band_structure() has to be done."
            raise RuntimeError(msg)

        return BandStructureDict(
            qpoints=self._band_structure.qpoints,
            distances=self._band_structure.distances,
            frequencies=self._band_structure.frequencies,
            eigenvectors=self._band_structure.eigenvectors,
            group_velocities=self._band_structure.group_velocities,
        )

    def auto_band_structure(
        self,
        npoints: int = 101,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
        plot: bool = False,
        write_yaml: bool = False,
        filename: str | os.PathLike = "band.yaml",
    ) -> Any | None:
        """Conveniently calculate and draw band structure.

        See the docstring of :meth:`run_band_structure` for the
        parameters ``with_eigenvectors`` (default False) and
        ``with_group_velocities`` (default False).

        Parameters
        ----------
        npoints : int, optional
            Number of q-points in each segment of the band-structure
            paths. The number includes end points. Default is 101.
        plot : bool, optional
            When True, band structure is plotted using matplotlib and
            the matplotlib module (``plt``) is returned. To watch the
            result, usually ``show()`` has to be called. Default is
            False.
        write_yaml : bool, optional
            When True, a ``band.yaml`` like file is written out. The
            file name can be specified with the ``filename`` parameter.
            Default is False.
        filename : str, optional
            File name used to write the ``band.yaml`` like file. Default
            is ``band.yaml``.

        """
        bands, labels, path_connections = get_band_qpoints_by_seekpath(
            self._primitive, npoints, is_const_interval=True
        )
        self.run_band_structure(
            bands,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities,
            path_connections=path_connections,
            labels=labels,
            is_legacy_plot=False,
        )
        if write_yaml:
            self.write_yaml_band_structure(filename=filename)
        if plot:
            return self.plot_band_structure()

    def plot_band_structure(self) -> Any:
        """Plot calculated band structure.

        Returns
        -------
        matplotlib.pyplot
            The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
            display the figure.

        """
        if self._band_structure is None:
            raise RuntimeError("run_band_structure has to be done.")

        return plot_band_structure(self._band_structure)

    def write_hdf5_band_structure(
        self,
        comment: dict | None = None,
        filename: str | os.PathLike = "band.hdf5",
        compression: Literal["gzip", "lzf"] | int | None = None,
    ) -> None:
        """Write band structure in hdf5 format.

        Parameters
        ----------
        comment : dict, optional
            Items are stored in hdf5 file in the way of key-value pair.
        filename : str, optional
            Default is ``band.hdf5``.

        """
        assert self._band_structure is not None
        self._band_structure.write_hdf5(
            comment=comment, filename=filename, compression=compression
        )

    def write_yaml_band_structure(
        self,
        comment: dict | None = None,
        filename: str | os.PathLike | None = None,
        compression: Literal["gzip", "lzma"] | None = None,
    ) -> None:
        """Write band structure in yaml.

        Parameters
        ----------
        comment : dict
            Data structure dumped in YAML and the dumped YAML text is put
            at the beginning of the file.
        filename : str
            Default filename is 'band.yaml' when compression=None.
            With compression, an extension of filename is added such as
            'band.yaml.xz'.
        compression : None, 'gzip', or 'lzma'
            None gives a plain text file. ``'gzip'`` and ``'lzma'``
            compress the yaml text with the respective compression
            method.

        """
        if self._band_structure is None:
            raise RuntimeError("run_band_structure has to be done.")

        self._band_structure.write_yaml(
            comment=comment, filename=filename, compression=compression
        )

    def init_mesh(
        self,
        mesh: float | Sequence[int] | NDArray[np.int64] = 100.0,
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_mesh_symmetry: bool = True,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
        is_gamma_center: bool = False,
        use_iter_mesh: bool = False,
    ) -> None:
        """Initialize mesh sampling phonon calculation without starting to run.

        Phonon calculation starts explicitly with calling Mesh.run() or
        implicitly with accessing getters of Mesh instance, e.g.,
        Mesh.frequencies.

        Parameters
        ----------
        mesh: array_like or float, optional
            Mesh numbers along a, b, c axes when array_like object is given.
            ``dtype='int64'``, ``shape=(3,)``.
            When a float value is given, a uniform mesh is generated
            following the VASP convention by
            ``N = max(1, nint(l * norm(a*)))``,
            where ``nint`` is the function that returns the nearest
            integer and ``a*`` is each reciprocal basis vector. In this
            case, ``is_gamma_center=True`` is enforced.
            Default value is 100.0.
        shift : array_like, optional
            Mesh shifts along a*, b*, c* axes with respect to neighboring
            grid points from the original mesh (Monkhorst-Pack or Gamma
            center). 0.5 gives a half-grid shift. Normally 0 or 0.5 is
            given; otherwise q-point symmetry search is not performed.
            Default is None (no additional shift).
            ``shape=(3,)``, ``dtype='double'``.
        is_time_reversal : bool, optional
            Whether to include time-reversal symmetry in the symmetry
            search. Default is True.
        is_mesh_symmetry : bool, optional
            Whether mesh symmetry search is performed. Default is True.
        with_eigenvectors : bool, optional
            Store eigenvectors when True. Default is False.
        with_group_velocities : bool, optional
            Calculate group velocities when True. Default is False.
        is_gamma_center : bool, optional
            Generate a uniform mesh centered at Gamma instead of using
            the Monkhorst-Pack scheme. When ``mesh`` is given as a float
            (length measure), this setting is ignored and
            ``is_gamma_center=True`` is enforced. Default is False.
        use_iter_mesh : bool, optional
            Use ``IterMesh`` instead of ``Mesh`` so that phonon
            properties are not stored on the instance, saving memory.
            Used with ``ThermalDisplacements`` and
            ``ThermalDisplacementMatrices``. Default is False.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        # Mesh / IterMesh accept a float (length) or a 3-tuple of ints and
        # handle the float -> mesh-numbers conversion internally.
        if use_iter_mesh:
            self._mesh = IterMesh(
                self._dynamical_matrix,
                mesh,
                shift=shift,
                is_time_reversal=is_time_reversal,
                is_mesh_symmetry=is_mesh_symmetry,
                with_eigenvectors=with_eigenvectors,
                is_gamma_center=is_gamma_center,
                rotations=self._primitive_symmetry.pointgroup_operations,
                primitive_symmetry=self._primitive_symmetry,
                factor=self._unit_conversion_factor,
                lang=self._lang,
            )
        else:
            self._mesh = Mesh(
                self._dynamical_matrix,
                mesh,
                shift=shift,
                is_time_reversal=is_time_reversal,
                is_mesh_symmetry=is_mesh_symmetry,
                with_eigenvectors=with_eigenvectors,
                is_gamma_center=is_gamma_center,
                group_velocity=group_velocity,
                rotations=self._primitive_symmetry.pointgroup_operations,
                primitive_symmetry=self._primitive_symmetry,
                factor=self._unit_conversion_factor,
                lang=self._lang,
            )

    def run_mesh(
        self,
        mesh: float | Sequence[int] | NDArray[np.int64] = 100.0,
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_mesh_symmetry: bool = True,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
        is_gamma_center: bool = False,
    ) -> Mesh:
        """Run mesh sampling phonon calculation.

        See the parameter details in Phonopy.init_mesh.

        Returns
        -------
        Mesh
            The calculated mesh sampling result. The same object is
            also accessible through the ``mesh`` property.

        """
        self.init_mesh(
            mesh=mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities,
            is_gamma_center=is_gamma_center,
        )
        assert isinstance(self._mesh, Mesh)
        self._mesh.run()
        return self._mesh

    def get_mesh_dict(self) -> MeshDict | IterMeshDict:
        """Return phonon properties calculated by mesh sampling.

        Returns
        -------
        dict
            Keys are ``qpoints``, ``weights``, ``frequencies``,
            ``eigenvectors``, and ``group_velocities``.

            ``qpoints`` : ndarray
                q-points in reduced coordinates of the reciprocal
                lattice. ``shape=(ir-grid points, 3)``,
                ``dtype='double'``.
            ``weights`` : ndarray
                Geometric q-point weights. The sum equals the number of
                grid points. ``shape=(ir-grid points,)``,
                ``dtype='int64'``.
            ``frequencies`` : ndarray
                Phonon frequencies at ir-grid points. Imaginary
                frequencies are represented by negative real numbers.
                ``shape=(ir-grid points, bands)``, ``dtype='double'``.
            ``eigenvectors`` : ndarray
                Phonon eigenvectors at ir-grid points. See the data
                structure of ``np.linalg.eigh``.
                ``shape=(ir-grid points, bands, bands)``, complex dtype
                (``"c%d" % (np.dtype('double').itemsize * 2)``),
                ``order='C'``.
            ``group_velocities`` : ndarray
                Phonon group velocities at ir-grid points.
                ``shape=(ir-grid points, bands, 3)``, ``dtype='double'``.

        .. deprecated::
            Use the ``mesh`` property instead.

        """
        warnings.warn(
            "get_mesh_dict() is deprecated. Use the mesh property to access "
            "the Mesh result object; its qpoints, weights, frequencies, "
            "eigenvectors, and group_velocities attributes replace the dict "
            "keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(self._mesh, Mesh):
            return MeshDict(
                qpoints=self._mesh.qpoints,
                weights=self._mesh.weights,
                frequencies=self._mesh.frequencies,
                eigenvectors=self._mesh.eigenvectors,
                group_velocities=self._mesh.group_velocities,
            )
        elif isinstance(self._mesh, IterMesh):
            return IterMeshDict(
                qpoints=self._mesh.qpoints,
                weights=self._mesh.weights,
            )
        else:
            msg = "Mesh is not initialized."
            raise RuntimeError(msg)

    def write_hdf5_mesh(
        self,
        compression: Literal["gzip", "lzf"] | int | None = None,
    ) -> None:
        """Write mesh calculation results in hdf5 format."""
        if not isinstance(self._mesh, Mesh):
            msg = "Mesh is not initialized."
            raise RuntimeError(msg)
        self._mesh.write_hdf5(compression=compression)

    def write_yaml_mesh(self) -> None:
        """Write mesh calculation results in yaml format."""
        if not isinstance(self._mesh, Mesh):
            msg = "Mesh is not initialized."
            raise RuntimeError(msg)
        self._mesh.write_yaml()

    # Plot band structure and DOS (PDOS) together
    def plot_band_structure_and_dos(
        self, pdos_indices: Sequence[Sequence[int]] | None = None
    ) -> Any:
        """Plot band structure and DOS."""
        if self._total_dos is None and pdos_indices is None:
            msg = "run_total_dos has to be done."
            raise RuntimeError(msg)
        if self._pdos is None and pdos_indices is not None:
            msg = "run_projected_dos has to be done."
            raise RuntimeError(msg)
        if self._band_structure is None:
            msg = "run_band_structure has to be done."
            raise RuntimeError(msg)

        return plot_band_structure_and_dos(
            self._band_structure,
            total_dos=self._total_dos,
            projected_dos=self._pdos,
            pdos_indices=pdos_indices,
        )

    # Sampling at q-points
    def run_qpoints(
        self,
        q_points: Sequence[Sequence[float]] | NDArray[np.double],
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False,
        with_dynamical_matrices: bool = False,
        nac_q_direction: Sequence[float] | NDArray[np.double] | None = None,
    ) -> QpointsPhonon:
        """Run phonon calculation at specified q-points.

        Parameters
        ----------
        q_points: array_like
            q-points in reduced coordinates.
            dtype='double', shape=(q-points, 3)
        with_eigenvectors: bool, optional
            Eigenvectors are stored by setting True. Default False.
        with_group_velocities : bool, optional
            Group velocities are calculated by setting True. Default is False.
        with_dynamical_matrices : bool, optional
            Calculated dynamical matrices are stored by setting True.
            Default is False.
        nac_q_direction : array_like, optional
            q-point direction from Gamma-point in fractional coordinates
            of reciprocal basis vectors. Only the direction is used, i.e.,
            ``q_direction / norm(q_direction)`` is computed and used. This
            parameter is activated only at q=(0, 0, 0).
            ``shape=(3,)``, ``dtype='double'``.

        Returns
        -------
        QpointsPhonon
            The calculated phonons at the q-points. The same object is
            also accessible through the ``qpoints`` property.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if with_group_velocities:
            if self._group_velocity is None:
                self._set_group_velocity()
            group_velocity = self._group_velocity
        else:
            group_velocity = None

        self._qpoints = QpointsPhonon(
            np.reshape(q_points, (-1, 3)),
            self._dynamical_matrix,
            nac_q_direction=nac_q_direction,
            with_eigenvectors=with_eigenvectors,
            group_velocity=group_velocity,
            with_dynamical_matrices=with_dynamical_matrices,
            factor=self._unit_conversion_factor,
            lang=self._lang,
        )
        return self._qpoints

    def get_qpoints_dict(self) -> QpointsDict:
        """Return calculated phonon properties at q-points.

        Returns
        -------
        dict
            Keys are ``frequencies``, ``eigenvectors``,
            ``group_velocities``, and ``dynamical_matrices``.

            ``frequencies`` : ndarray
                Phonon frequencies. Imaginary frequencies are represented
                by negative real numbers.
                ``shape=(qpoints, bands)``, ``dtype='double'``.
            ``eigenvectors`` : ndarray or None
                Phonon eigenvectors. ``None`` if eigenvectors are not
                stored. ``shape=(qpoints, bands, bands)``, complex dtype
                (``"c%d" % (np.dtype('double').itemsize * 2)``),
                ``order='C'``.
            ``group_velocities`` : ndarray or None
                Phonon group velocities. ``None`` if group velocities
                are not calculated.
                ``shape=(qpoints, bands, 3)``, ``dtype='double'``.
            ``dynamical_matrices`` : ndarray
                Dynamical matrices at q-points.
                ``shape=(qpoints, bands, bands)``, ``dtype='double'``.

        .. deprecated::
            Use the ``qpoints`` property instead.

        """
        warnings.warn(
            "get_qpoints_dict() is deprecated. Use the qpoints property to "
            "access the Qpoints result object; its frequencies, eigenvectors, "
            "group_velocities, and dynamical_matrices attributes replace the "
            "dict keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._qpoints is None:
            msg = "Phonopy.run_qpoints() has to be done."
            raise RuntimeError(msg)

        return QpointsDict(
            frequencies=self._qpoints.frequencies,
            eigenvectors=self._qpoints.eigenvectors,
            group_velocities=self._qpoints.group_velocities,
            dynamical_matrices=self._qpoints.dynamical_matrices,
        )

    def write_hdf5_qpoints_phonon(
        self,
        compression: Literal["gzip", "lzf"] | int | None = None,
    ) -> None:
        """Write phonon properties calculated at q-points in hdf5 format."""
        if self._qpoints is None:
            msg = "Phonopy.run_qpoints() has to be done."
            raise RuntimeError(msg)
        self._qpoints.write_hdf5(compression=compression)

    def write_yaml_qpoints_phonon(self) -> None:
        """Write phonon properties calculated at q-points in yaml format."""
        if self._qpoints is None:
            msg = "Phonopy.run_qpoints() has to be done."
            raise RuntimeError(msg)
        self._qpoints.write_yaml()

    # DOS
    def run_total_dos(
        self,
        sigma: float | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        freq_pitch: float | None = None,
        use_tetrahedron_method: bool = True,
        smearing_function: Literal["Normal", "Cauchy"] = "Normal",
    ) -> TotalDos:
        """Run total DOS calculation.

        Parameters
        ----------
        sigma : float, optional
            Smearing width for the smearing method. Default is None.
        freq_min, freq_max, freq_pitch : float, optional
            Minimum and maximum frequencies of the frequency range in
            which DOS is computed, and the sampling interval
            (``freq_pitch``). Defaults are None and they are
            automatically determined.
        use_tetrahedron_method : bool, optional
            Use the tetrahedron method when True. When ``sigma`` is
            set, the smearing method is used instead. Default is True.
        smearing_function : {"Normal", "Cauchy"}, optional
            Distribution used by the smearing method. "Normal" is a normal
            distribution and "Cauchy" is a Cauchy (Lorentzian) distribution.
            Default is "Normal".

        Returns
        -------
        TotalDos
            The calculated total DOS. The same object is also
            accessible through the ``total_dos`` property.

        """
        if self._mesh is None:
            msg = "run_mesh has to be done before DOS calculation."
            raise RuntimeError(msg)
        if isinstance(self._mesh, IterMesh):
            msg = "IterMesh is not supported for DOS calculation."
            raise RuntimeError(msg)

        total_dos = TotalDos(
            self._mesh,
            sigma=sigma,
            use_tetrahedron_method=use_tetrahedron_method,
            smearing_function=smearing_function,
            lang=self._lang,
        )
        total_dos.set_draw_area(freq_min, freq_max, freq_pitch)
        total_dos.run()
        self._total_dos = total_dos
        return self._total_dos

    def auto_total_dos(
        self,
        mesh=100.0,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        is_gamma_center=False,
        plot=False,
        xlabel=None,
        ylabel=None,
        with_tight_frequency_range=False,
        write_dat=False,
        filename="total_dos.dat",
    ) -> Any | None:
        """Conveniently calculate and draw total DOS."""
        self.run_mesh(
            mesh=mesh,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            is_gamma_center=is_gamma_center,
        )
        self.run_total_dos()
        if write_dat:
            self.write_total_dos(filename=filename)
        if plot:
            return self.plot_total_dos(
                xlabel=xlabel,
                ylabel=ylabel,
                with_tight_frequency_range=with_tight_frequency_range,
            )

    def get_total_dos_dict(self) -> TotalDosDict:
        """Return total DOS.

        Returns
        -------
        A dictionary with keys of 'frequency_points' and 'total_dos'.
        Each value of corresponding key is as follows:

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        total_dos:
            shape=(frequency_sampling_points, ), dtype='double'

        .. deprecated::
            Use the ``total_dos`` property instead.

        """
        warnings.warn(
            "get_total_dos_dict() is deprecated. Use the total_dos property "
            "to access the TotalDos result object; its frequency_points and "
            "dos attributes replace the dict keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._total_dos is None:
            msg = "run_total_dos has to be done before getting total DOS."
            raise RuntimeError(msg)
        assert self._total_dos.dos is not None
        return TotalDosDict(
            frequency_points=self._total_dos.frequency_points,
            total_dos=self._total_dos.dos,
        )

    def set_Debye_frequency(self, freq_max_fit: float | None = None) -> None:
        """Calculate Debye frequency on top of total DOS.

        .. deprecated::
            After ``run_total_dos()``, call
            ``total_dos.run_debye_frequency(num_atoms)`` instead.

        """
        warnings.warn(
            "set_Debye_frequency() is deprecated. After run_total_dos(), call "
            "total_dos.run_debye_frequency(num_atoms) and read "
            "total_dos.debye_frequency.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._total_dos is None:
            msg = "run_total_dos has to be done before getting total DOS."
            raise RuntimeError(msg)
        self._total_dos.run_debye_frequency(freq_max_fit=freq_max_fit)

    def get_Debye_frequency(self) -> float | None:
        """Return Debye frequency.

        .. deprecated::
            Use ``total_dos.debye_frequency`` instead.

        """
        warnings.warn(
            "get_Debye_frequency() is deprecated. Use the total_dos property "
            "and read its debye_frequency attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._total_dos is None:
            msg = "run_total_dos has to be done before getting total DOS."
            raise RuntimeError(msg)
        return self._total_dos.debye_frequency

    def plot_total_dos(
        self,
        xlabel: str | None = None,
        ylabel: str | None = None,
        with_tight_frequency_range: bool = False,
    ) -> Any:
        """Plot total DOS.

        Parameters
        ----------
        xlabel : str, optional
            x-label of the plot. Default is None, which puts a default
            x-label.
        ylabel : str, optional
            y-label of the plot. Default is None, which puts a default
            y-label.
        with_tight_frequency_range : bool, optional
            Plot with a tight frequency range. Default is False.

        """
        if self._total_dos is None:
            msg = "run_total_dos has to be done before plotting total DOS."
            raise RuntimeError(msg)

        return plot_total_dos(
            self._total_dos,
            xlabel=xlabel,
            ylabel=ylabel,
            with_tight_frequency_range=with_tight_frequency_range,
        )

    def write_total_dos(self, filename: str | os.PathLike = "total_dos.dat") -> None:
        """Write total DOS to text file."""
        if self._total_dos is None:
            msg = "run_total_dos has to be done before writing total DOS."
            raise RuntimeError(msg)
        self._total_dos.write(filename=filename)

    # PDOS
    def run_projected_dos(
        self,
        sigma: float | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
        freq_pitch: float | None = None,
        use_tetrahedron_method: bool = True,
        direction: Sequence[float] | NDArray[np.double] | None = None,
        xyz_projection: bool = False,
        smearing_function: Literal["Normal", "Cauchy"] = "Normal",
    ) -> ProjectedDos:
        """Run projected DOS calculation.

        Parameters
        ----------
        sigma : float, optional
            Smearing width for the smearing method. Default is None.
        freq_min, freq_max, freq_pitch : float, optional
            Minimum and maximum frequencies of the frequency range in
            which DOS is computed, and the sampling interval
            (``freq_pitch``). Defaults are None and they are
            automatically determined.
        use_tetrahedron_method : bool, optional
            Use the tetrahedron method when True. When ``sigma`` is
            set, the smearing method is used instead. Default is True.
        direction : array_like, optional
            Projection direction given as three values along the
            primitive cell basis vectors. Default is None (no
            projection).
        xyz_projection : bool, optional
            Whether to project along Cartesian directions. Default is
            False.
        smearing_function : {"Normal", "Cauchy"}, optional
            Distribution used by the smearing method. "Normal" is a normal
            distribution and "Cauchy" is a Cauchy (Lorentzian) distribution.
            Default is "Normal".

        Returns
        -------
        ProjectedDos
            The calculated projected DOS. The same object is also
            accessible through the ``projected_dos`` property.

        """
        self._pdos = None

        if self._mesh is None:
            msg = "run_mesh has to be done before PDOS calculation."
            raise RuntimeError(msg)

        if isinstance(self._mesh, IterMesh):
            msg = "IterMesh does not support projected DOS calculation."
            raise RuntimeError(msg)

        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be called with with_eigenvectors=True."
            raise RuntimeError(msg)

        if np.prod(self._mesh.mesh_numbers) != len(self._mesh.ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        if direction is not None:
            direction_cart = np.dot(direction, self._primitive.cell)
        else:
            direction_cart = None
        self._pdos = ProjectedDos(
            self._mesh,
            sigma=sigma,
            use_tetrahedron_method=use_tetrahedron_method,
            direction=direction_cart,
            xyz_projection=xyz_projection,
            smearing_function=smearing_function,
            lang=self._lang,
        )
        self._pdos.set_draw_area(freq_min, freq_max, freq_pitch)
        self._pdos.run()
        return self._pdos

    def auto_projected_dos(
        self,
        mesh: float | Sequence[int] | NDArray[np.int64] = 100.0,
        is_time_reversal: bool = True,
        is_gamma_center: bool = False,
        plot: bool = False,
        pdos_indices: Sequence[Sequence[int]] | None = None,
        legend: Sequence[str] | None = None,
        legend_prop: dict | None = None,
        legend_frameon: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        with_tight_frequency_range: bool = False,
        write_dat: bool = False,
        filename: str | os.PathLike = "projected_dos.dat",
    ) -> Any | None:
        """Conveniently calculate and draw projected DOS.

        See the docstring of ``Phonopy.init_mesh`` for the parameters
        ``mesh`` (default 100.0), ``is_time_reversal`` (default True), and
        ``is_gamma_center`` (default False). See the docstring of
        ``Phonopy.plot_projected_dos`` for ``pdos_indices``, ``legend``,
        ``xlabel``, ``ylabel``, and ``with_tight_frequency_range``.

        Parameters
        ----------
        plot : bool, optional
            With setting True, PDOS is plotted using matplotlib and the
            matplotlib module (``plt``) is returned. To watch the result,
            usually ``show()`` has to be called. Default is False.
        write_dat : bool, optional
            With setting True, a ``projected_dos.dat`` like file is
            written out. The file name can be specified with the
            ``filename`` parameter. Default is False.
        filename : str, optional
            File name used to write the ``projected_dos.dat`` like file.
            Default is ``projected_dos.dat``.

        """
        self.run_mesh(
            mesh=mesh,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=False,
            with_eigenvectors=True,
            is_gamma_center=is_gamma_center,
        )
        self.run_projected_dos()
        if write_dat:
            self.write_projected_dos(filename=filename)
        if plot:
            return self.plot_projected_dos(
                pdos_indices=pdos_indices,
                legend=legend,
                legend_prop=legend_prop,
                legend_frameon=legend_frameon,
                xlabel=xlabel,
                ylabel=ylabel,
                with_tight_frequency_range=with_tight_frequency_range,
            )

    def get_projected_dos_dict(self) -> ProjectedDosDict:
        """Return projected DOS.

        Projection is done to atoms and may be also done along directions
        depending on the parameters at run_projected_dos.

        Returns
        -------
        A dictionary with keys of 'frequency_points' and 'projected_dos'.
        Each value of corresponding key is as follows:

        frequency_points: ndarray
            shape=(frequency_sampling_points, ), dtype='double'
        projected_dos:
            shape=(projections, frequency_sampling_points), dtype='double'

        .. deprecated::
            Use the ``projected_dos`` property instead.

        """
        warnings.warn(
            "get_projected_dos_dict() is deprecated. Use the projected_dos "
            "property to access the ProjectedDos result object; its "
            "frequency_points and projected_dos attributes replace the dict "
            "keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._pdos is None:
            msg = "run_projected_dos has to be done before getting projected DOS."
            raise RuntimeError(msg)
        return ProjectedDosDict(
            frequency_points=self._pdos.frequency_points,
            projected_dos=self._pdos.projected_dos,
        )

    def plot_projected_dos(
        self,
        pdos_indices: Sequence[Sequence[int]] | None = None,
        legend: Sequence[str] | None = None,
        legend_prop: dict | None = None,
        legend_frameon: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        with_tight_frequency_range: bool = False,
    ) -> Any:
        """Plot projected DOS.

        Parameters
        ----------
        pdos_indices : list of list, optional
            Sets of indices of atoms whose projected DOS are summed over.
            The indices start with 0. An example is
            ``pdos_indices=[[0, 1], [2, 3, 4, 5]]``. Default is None,
            which means ``pdos_indices=[[i] for i in range(natom)]``.
        legend : list of instances such as str or int, optional
            The str(instance) are shown in legend.
            It has to be len(pdos_indices)==len(legend). Default is None.
            When None, legend is not shown.
        legend_prop : dict, optional
            Legend properties of matplotlib. Default is None.
        legend_frameon : bool, optional
            Legend with frame or not. Default is True.
        xlabel : str, optional
            x-label of plot. Default is None, which puts a default x-label.
        ylabel : str, optional
            y-label of plot. Default is None, which puts a default y-label.
        with_tight_frequency_range : bool, optional
            Plot with tight frequency range. Default is False.

        """
        if self._pdos is None:
            msg = "run_projected_dos has to be done before plotting projected DOS."
            raise RuntimeError(msg)

        return plot_projected_dos(
            self._pdos,
            pdos_indices=pdos_indices,
            legend=legend,
            legend_prop=legend_prop,
            legend_frameon=legend_frameon,
            xlabel=xlabel,
            ylabel=ylabel,
            with_tight_frequency_range=with_tight_frequency_range,
        )

    def write_projected_dos(
        self, filename: str | os.PathLike = "projected_dos.dat"
    ) -> None:
        """Write projected DOS to text file."""
        if self._pdos is None:
            msg = "run_projected_dos has to be done before writing projected DOS."
            raise RuntimeError(msg)
        self._pdos.write(filename=filename)

    # Thermal property
    def run_thermal_properties(
        self,
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        cutoff_frequency: float | None = None,
        pretend_real: bool = False,
        band_indices: Sequence[Sequence[int]] | None = None,
        classical: bool = False,
    ) -> ThermalProperties:
        """Run calculation of thermal properties at constant volume.

        In phonopy, imaginary frequencies are represented as negative real
        value. Under this situation, `cutoff_frequency` is used to ignore
        phonon modes that have frequencies less than `cutoff_frequency`.

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default values are 0, 1000, and 10.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.
        cutoff_frequency : float, optional
            Ignore phonon modes whose frequencies are smaller than this value.
            Default is None, which gives cutoff frequency as zero.
        pretend_real : bool, optional
            Use absolute value of phonon frequency when True. Default is False.
        band_indices : array_like, optional
            Band indices starting with 0. Normally the numbers correspond to
            phonon bands in ascending order of phonon frequencies. Thermal
            properties are calculated only including specified bands.
            Note that use of this results in unphysical values, and it is not
            recommended to use this feature. Default is None.
        classical : bool, optional
            If True, use classical statistics; if False, use quantum
            statistics. Default is False.

        Returns
        -------
        ThermalProperties
            The calculated thermal properties. The same object is also
            accessible through the ``thermal_properties`` property.

        """
        if self._mesh is None:
            msg = "run_mesh has to be done before run_thermal_properties."
            raise RuntimeError(msg)

        if not isinstance(self._mesh, Mesh):
            msg = "IterMesh is not supported for thermal properties."
            raise RuntimeError(msg)

        tp = ThermalProperties(
            self._mesh,
            cutoff_frequency=cutoff_frequency,
            pretend_real=pretend_real,
            band_indices=band_indices,
            classical=classical,
            lang=self._lang,
        )
        if temperatures is None:
            tp.set_temperature_range(t_step=t_step, t_max=t_max, t_min=t_min)
        else:
            tp.temperatures = temperatures
        tp.run()
        self._thermal_properties = tp
        return self._thermal_properties

    def get_thermal_properties_dict(self) -> ThermalPropertiesDict:
        """Return thermal properties.

        Returns
        -------
        A dictionary of thermal properties with keys of 'temperatures',
        'free_energy', 'entropy', and 'heat_capacity'.
        Each value of corresponding key is as follows:

        temperatures : ndarray
            Temperatures in K.
            shape=(temperatures, ), dtype='double'
        free_energy : ndarray
            Helmholtz free energies in kJ/mol.
            shape=(temperatures, ), dtype='double'
        entropy : ndarray
            Entropies in J/K/mol.
            shape=(temperatures, ), dtype='double'
        heat_capacity : ndarray
            Heat capacities in J/K/mol.
            shape=(temperatures, ), dtype='double'

        .. deprecated::
            Use the ``thermal_properties`` property instead.

        """
        warnings.warn(
            "get_thermal_properties_dict() is deprecated. Use the "
            "thermal_properties property to access the ThermalProperties "
            "result object; its temperatures, free_energy, entropy, and "
            "heat_capacity attributes replace the dict keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._thermal_properties is None:
            msg = (
                "run_thermal_properties has to be done before "
                "getting thermal properties."
            )
            raise RuntimeError(msg)

        assert self._thermal_properties.thermal_properties is not None

        tp = self._thermal_properties.thermal_properties
        return ThermalPropertiesDict(
            temperatures=tp[0],
            free_energy=tp[1],
            entropy=tp[2],
            heat_capacity=tp[3],
        )

    def plot_thermal_properties(
        self,
        xlabel: str | None = None,
        ylabel: str | None = None,
        with_grid: bool = True,
        divide_by_Z: bool = False,
        legend_style: str | None = "normal",
    ) -> Any:
        """Plot thermal properties.

        Parameters
        ----------
        xlabel : str, optional
            Label used for x-axis.
        ylabel : str, optional
            Label used for y-axis.
        with_grid : bool, optional
            With grid or not. Default is True.
        divide_by_Z : bool, optional
            Divide thermal properties by number of formula units of primitive
            cell. Default is False.
        legend_style : str, optional
            "normal", "compact", None. None will not show legend.

        """
        if (
            self._thermal_properties is None
            or self._thermal_properties.temperatures is None
        ):
            msg = "run_thermal_properties has to be done."
            raise RuntimeError(msg)

        return plot_thermal_properties(
            self._thermal_properties,
            xlabel=xlabel,
            ylabel=ylabel,
            with_grid=with_grid,
            divide_by_Z=divide_by_Z,
            legend_style=legend_style,
        )

    def write_yaml_thermal_properties(
        self, filename: str | os.PathLike = "thermal_properties.yaml"
    ) -> None:
        """Write thermal properties in yaml format."""
        if self._thermal_properties is None:
            msg = "run_thermal_properties has to be done."
            raise RuntimeError(msg)
        self._thermal_properties.write_yaml(filename=filename)

    # Thermal displacement
    def run_thermal_displacements(
        self,
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        direction: Sequence[float] | NDArray[np.double] | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> ThermalDisplacements:
        """Run thermal displacements calculation.

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default values are 0, 1000, and 10.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.
        direction : array_like, optional
            Projection direction in reduced coordinates.
            ``shape=(3,)``, ``dtype='double'``. Default is None
            (no projection).
        freq_min, freq_max : float, optional
            Only phonon frequencies between ``freq_min`` and
            ``freq_max`` are included. Default is None (all phonons).

        Returns
        -------
        ThermalDisplacements
            The calculated thermal displacements. The same object is
            also accessible through the ``thermal_displacements``
            property.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)
        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)
        mesh_nums = self._mesh.mesh_numbers
        ir_grid_points = self._mesh.ir_grid_points
        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be done with with_eigenvectors=True."
            raise RuntimeError(msg)
        if np.prod(mesh_nums) != len(ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        if direction is not None:
            projection_direction = np.dot(direction, self._primitive.cell)
            td = ThermalDisplacements(
                self._mesh,
                projection_direction=projection_direction,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        else:
            td = ThermalDisplacements(self._mesh, freq_min=freq_min, freq_max=freq_max)

        if temperatures is None:
            td.set_temperature_range(t_min, t_max, t_step)
        else:
            td.temperatures = temperatures
        td.run()

        self._thermal_displacements = td
        return self._thermal_displacements

    def get_thermal_displacements_dict(self) -> dict:
        """Return thermal displacements.

        .. deprecated::
            Use the ``thermal_displacements`` property instead.

        """
        warnings.warn(
            "get_thermal_displacements_dict() is deprecated. Use the "
            "thermal_displacements property to access the ThermalDisplacements"
            " result object; its temperatures and thermal_displacements "
            "attributes replace the dict keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._thermal_displacements is None:
            msg = "run_thermal_displacements has to be done."
            raise RuntimeError(msg)

        td = self._thermal_displacements
        return {
            "temperatures": td.temperatures,
            "thermal_displacements": td.thermal_displacements,
        }

    def plot_thermal_displacements(self, is_legend: bool = False) -> Any:
        """Plot thermal displacements."""
        if self._thermal_displacements is None:
            msg = "run_thermal_displacements has to be done."
            raise RuntimeError(msg)

        return plot_thermal_displacements(
            self._thermal_displacements, is_legend=is_legend
        )

    def write_yaml_thermal_displacements(self) -> None:
        """Write thermal displacements in yaml format."""
        if self._thermal_displacements is None:
            msg = "run_thermal_displacements has to be done."
            raise RuntimeError(msg)
        self._thermal_displacements.write_yaml()

    # Thermal displacement matrix
    def run_thermal_displacement_matrices(
        self,
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> ThermalDisplacementMatrices:
        """Run thermal displacement matrices calculation.

        Parameters
        ----------
        t_min, t_max, t_step : float, optional
            Minimum and maximum temperatures and the interval in this
            temperature range. Default values are 0, 1000, and 10.
        freq_min, freq_max : float, optional
            Phonon frequencies larger than freq_min and smaller than
            freq_max are included. Default is None, i.e., all phonons.
        temperatures : array_like, optional
            Temperature points where thermal properties are calculated.
            When this is set, t_min, t_max, and t_step are ignored.
            Default is None.

        Returns
        -------
        ThermalDisplacementMatrices
            The calculated thermal displacement matrices. The same
            object is also accessible through the
            ``thermal_displacement_matrices`` property.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)
        if self._mesh is None:
            msg = "run_mesh has to be done."
            raise RuntimeError(msg)
        mesh_nums = self._mesh.mesh_numbers
        ir_grid_points = self._mesh.ir_grid_points
        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be done with with_eigenvectors=True."
            raise RuntimeError(msg)
        if np.prod(mesh_nums) != len(ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        tdm = ThermalDisplacementMatrices(
            self._mesh,
            freq_min=freq_min,
            freq_max=freq_max,
            lattice=self._primitive.cell.T,
        )

        if temperatures is None:
            tdm.set_temperature_range(t_min, t_max, t_step)
        else:
            tdm.temperatures = temperatures
        tdm.run()

        self._thermal_displacement_matrices = tdm
        return self._thermal_displacement_matrices

    def get_thermal_displacement_matrices_dict(self) -> ThermalDisplacementMatricesDict:
        """Return thermal displacement matrices.

        .. deprecated::
            Use the ``thermal_displacement_matrices`` property instead.

        """
        warnings.warn(
            "get_thermal_displacement_matrices_dict() is deprecated. Use the "
            "thermal_displacement_matrices property to access the "
            "ThermalDisplacementMatrices result object; its temperatures, "
            "thermal_displacement_matrices, and "
            "thermal_displacement_matrices_cif attributes replace the dict "
            "keys.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._thermal_displacement_matrices is None:
            msg = "run_thermal_displacement_matrices has to be done."
            raise RuntimeError(msg)

        tdm = self._thermal_displacement_matrices
        return ThermalDisplacementMatricesDict(
            temperatures=tdm.temperatures,
            thermal_displacement_matrices=tdm.thermal_displacement_matrices,
            thermal_displacement_matrices_cif=tdm.thermal_displacement_matrices_cif,
        )

    def write_yaml_thermal_displacement_matrices(self) -> None:
        """Write thermal displacement matrices in yaml format."""
        if self._thermal_displacement_matrices is None:
            msg = "run_thermal_displacement_matrices has to be done."
            raise RuntimeError(msg)
        self._thermal_displacement_matrices.write_yaml()

    def write_thermal_displacement_matrix_to_cif(self, temperature_index: int) -> None:
        """Write thermal displacement matrices at a temperature in cif."""
        if self._thermal_displacement_matrices is None:
            msg = "run_thermal_displacement_matrices has to be done."
            raise RuntimeError(msg)
        self._thermal_displacement_matrices.write_cif(
            self._primitive, temperature_index
        )

    def write_animation(
        self,
        q_point: Sequence[float] | NDArray[np.double] | None = None,
        anime_type: str = "v_sim",
        band_index: int | None = None,
        amplitude: float | None = None,
        num_div: int | None = None,
        shift: Sequence[float] | NDArray[np.double] | None = None,
        filename: str | os.PathLike | None = None,
    ) -> str | os.PathLike:
        """Write atomic modulations in animation format.

        Returns
        -------
        str
            Output filename.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        if anime_type in ("arc", "xyz", "jmol", "poscar"):
            if band_index is None or amplitude is None or num_div is None:
                msg = "Parameters are not correctly set for animation."
                raise RuntimeError(msg)

        return write_animation(
            self._dynamical_matrix,
            q_point=q_point,
            anime_type=anime_type,
            band_index=band_index,
            amplitude=amplitude,
            num_div=num_div,
            shift=shift,
            factor=self._unit_conversion_factor,
            filename=filename,
        )

    def run_modulations(
        self,
        dimension: Sequence[int] | NDArray[np.int64],
        phonon_modes: Sequence,
        delta_q: Sequence[float] | NDArray[np.double] | None = None,
        derivative_order: int | None = None,
        nac_q_direction: Sequence[float] | NDArray[np.double] | None = None,
    ) -> Modulation:
        """Generate atomic displacements of phonon modes.

        The design of this feature, and thus its API, is not very
        satisfactory. It should be reconsidered someday in the future.

        Parameters
        ----------
        dimension : array_like
            Supercell dimension with respect to the primitive cell.
            ``shape=(3,)``, ``(3, 3)``, or ``(9,)``,
            ``dtype='int64'``.
        phonon_modes : list of phonon mode settings
            Each element of the outer list specifies one phonon mode::

                [q-point, band index (int), amplitude (float),
                 phase (float)]

            The first element is a list representing the q-point in
            reduced coordinates. The remaining elements are the band
            index (starting with 0), amplitude, and phase factor.
        nac_q_direction : array_like
            q-point direction from Gamma-point in fractional coordinates
            of reciprocal basis vectors. Only the direction is used, i.e.,
            ``q_direction / norm(q_direction)`` is computed and used. This
            parameter is activated only at q=(0, 0, 0).
            ``shape=(3,)``, ``dtype='double'``.

        Returns
        -------
        Modulation
            The calculated modulations. The same object is also
            accessible through the ``modulation`` property.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._modulation = Modulation(
            self._dynamical_matrix,
            dimension,
            phonon_modes,
            delta_q=delta_q,
            derivative_order=derivative_order,
            nac_q_direction=nac_q_direction,
            factor=self._unit_conversion_factor,
        )
        self._modulation.run()
        return self._modulation

    def get_modulated_supercells(self) -> list[PhonopyAtoms]:
        """Return modulated structures as a list of ``PhonopyAtoms``.

        .. deprecated::
            Use the ``modulated_supercells`` attribute of the
            ``Modulation`` object returned by ``run_modulations()`` (or
            of the ``modulation`` property) instead.

        """
        warnings.warn(
            "get_modulated_supercells() is deprecated. Use the "
            "modulated_supercells attribute of the Modulation object "
            "returned by run_modulations() or of the modulation property.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._modulation is None:
            msg = "run_modulations has to be done before getting modulated supercells."
            raise RuntimeError(msg)
        return self._modulation.modulated_supercells

    def get_modulations_and_supercell(self) -> tuple[NDArray[np.cdouble], PhonopyAtoms]:
        """Return atomic modulations and the perfect supercell.

        Returns
        -------
        modulations : ndarray
            Atomic modulations of the supercell in Cartesian
            coordinates.
        supercell : PhonopyAtoms
            The (unmodulated) supercell.

        .. deprecated::
            Use the ``modulations`` and ``supercell`` attributes of the
            ``Modulation`` object returned by ``run_modulations()`` (or
            of the ``modulation`` property) instead.

        """
        warnings.warn(
            "get_modulations_and_supercell() is deprecated. Use the "
            "modulations and supercell attributes of the Modulation object "
            "returned by run_modulations() or of the modulation property.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._modulation is None:
            msg = "run_modulations has to be done before getting modulations."
            raise RuntimeError(msg)
        return self._modulation.modulations, self._modulation.supercell

    def write_modulations(
        self,
        calculator: str | None = None,
        optional_structure_info: StructureInfo | None = None,
    ) -> None:
        """Write modulated structures to MPOSCAR files."""
        if self._modulation is None:
            msg = "run_modulations has to be done before writing modulations."
            raise RuntimeError(msg)
        self._modulation.write(
            interface_mode=calculator,
            optional_structure_info=optional_structure_info,
        )

    def write_yaml_modulations(self) -> None:
        """Write atomic modulations in yaml format."""
        if self._modulation is None:
            msg = "run_modulations has to be done before writing modulations."
            raise RuntimeError(msg)
        self._modulation.write_yaml()

    # Irreducible representation
    def run_irreps(
        self,
        q: Sequence[float] | NDArray[np.double],
        is_little_cogroup: bool = False,
        nac_q_direction: Sequence[float] | NDArray[np.double] | None = None,
        degeneracy_tolerance: float | None = None,
    ) -> IrReps:
        """Identify ir-reps of phonon modes.

        The design of this API is not very satisfactory and is expected
        to be redesigned in the next major versions once the use case
        of the API for the ir-reps feature becomes clearer.

        Parameters
        ----------
        nac_q_direction : array_like
            q-point direction from Gamma-point in fractional coordinates
            of reciprocal basis vectors. Only the direction is used, i.e.,
            ``q_direction / norm(q_direction)`` is computed and used. This
            parameter is activated only at q=(0, 0, 0).
            ``shape=(3,)``, ``dtype='double'``.

        Returns
        -------
        IrReps
            The identified ir-reps. The same object is also accessible
            through the ``irreps`` property.

        """
        if self._dynamical_matrix is None:
            msg = "Dynamical matrix has not yet built."
            raise RuntimeError(msg)

        self._irreps = IrReps(
            self._dynamical_matrix,
            q,
            self._primitive_symmetry,
            is_little_cogroup=is_little_cogroup,
            nac_q_direction=nac_q_direction,
            factor=self._unit_conversion_factor,
            degeneracy_tolerance=degeneracy_tolerance,
            log_level=self._log_level,
        )
        return self._irreps

    def set_irreps(
        self,
        q: Sequence[float] | NDArray[np.double],
        is_little_cogroup: bool = False,
        nac_q_direction: Sequence[float] | NDArray[np.double] | None = None,
        degeneracy_tolerance: float | None = None,
    ) -> IrReps:
        """Identify ir-reps of phonon modes.

        .. deprecated::
            Use :meth:`run_irreps` instead.

        """
        warnings.warn(
            "set_irreps() is deprecated. Use run_irreps() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_irreps(
            q,
            is_little_cogroup=is_little_cogroup,
            nac_q_direction=nac_q_direction,
            degeneracy_tolerance=degeneracy_tolerance,
        )

    def show_irreps(self, show_irreps: bool = False) -> None:
        """Show Ir-reps."""
        if self._irreps is None:
            msg = "run_irreps has to be done before showing Ir-reps."
            raise RuntimeError(msg)
        self._irreps.show(show_irreps=show_irreps)

    def write_yaml_irreps(self, show_irreps: bool = False) -> None:
        """Write Ir-reps in yaml format."""
        if self._irreps is None:
            msg = "run_irreps has to be done before writing Ir-reps."
            raise RuntimeError(msg)
        self._irreps.write_yaml(show_irreps=show_irreps)

    def get_group_velocity_at_q(
        self, q_point: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return group velocity at a q-point.

        .. deprecated::
            Use ``run_qpoints([q], with_group_velocities=True)`` and
            the ``group_velocities`` attribute of the returned
            ``QpointsPhonon`` object instead.

        """
        warnings.warn(
            "get_group_velocity_at_q() is deprecated. Use "
            "run_qpoints([q], with_group_velocities=True); the returned "
            "QpointsPhonon object provides group_velocities[0].",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._group_velocity is None:
            self._set_group_velocity()
        assert self._group_velocity is not None
        self._group_velocity.run([q_point])  # type: ignore
        assert self._group_velocity.group_velocities is not None
        return self._group_velocity.group_velocities[0]

    # Moment
    def run_moment(
        self,
        order: int = 1,
        is_projection: bool = False,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> PhononMoment:
        """Run moment calculation.

        Returns
        -------
        PhononMoment
            The calculated moment. The same object is also accessible
            through the ``moment`` property.

        """
        if self._mesh is None:
            msg = "run_mesh has to be done before run_moment."
            raise RuntimeError(msg)

        if isinstance(self._mesh, IterMesh):
            msg = "IterMesh is not supported for moment calculation."
            raise RuntimeError(msg)

        if is_projection:
            if self._mesh.eigenvectors is None:
                raise RuntimeError(
                    "run_mesh has to be done with with_eigenvectors=True."
                )
            self._moment = PhononMoment(
                self._mesh.frequencies,
                weights=self._mesh.weights,
                eigenvectors=self._mesh.eigenvectors,
            )
        else:
            self._moment = PhononMoment(
                self._mesh.frequencies, weights=self._mesh.weights
            )
        if freq_min is not None or freq_max is not None:
            self._moment.set_frequency_range(freq_min=freq_min, freq_max=freq_max)
        self._moment.run(order=order)
        return self._moment

    def get_moment(self) -> float | NDArray[np.double] | None:
        """Return moment.

        .. deprecated::
            Use the ``moment`` attribute of the ``PhononMoment`` object
            returned by ``run_moment()`` (or of the ``moment``
            property) instead.

        """
        warnings.warn(
            "get_moment() is deprecated. Use the moment attribute of the "
            "PhononMoment object returned by run_moment() or of the moment "
            "property.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._moment is None:
            msg = "run_moment has to be done before getting moment."
            raise RuntimeError(msg)

        return self._moment.moment

    def init_dynamic_structure_factor(
        self,
        Qpoints: Sequence[Sequence[float]] | NDArray[np.double],
        T: float,
        atomic_form_factor_func: Callable | None = None,
        scattering_lengths: dict | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> DynamicStructureFactor:
        """Initialize dynamic structure factor calculation.

        Call ``DynamicStructureFactor.run()`` to start the calculation,
        or iterate over the returned object to compute the structure
        factor q-point by q-point (e.g. for progress reporting).

        Parameters
        ----------
        Qpoints : array_like
            Q-points in any Brillouin zone.
            ``shape=(qpoints, 3)``, ``dtype='double'``.
        T : float
            Temperature in K.
        atomic_form_factor_func : callable
            Function that returns the atomic form factor (``func`` below)::

                f_params = {
                    'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                           0.767888, 0.070139, 0.995612, 14.1226457,
                           0.968249, 0.217037, 0.045300],
                    'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                           6.524271, 19.467656, 2.355626, 60.320301,
                           35.829404, 0.000436, -34.916604],
                }

                def get_func_AFF(f_params):
                    def func(symbol, Q):
                        return atomic_form_factor_WK1995(Q, f_params[symbol])
                    return func

        scattering_lengths : dict
            Coherent scattering lengths averaged over isotopes and spins.
            Supposed for INS. For example, ``{'Na': 3.63, 'Cl': 9.5770}``.
        freq_min, freq_max : float
            Minimum and maximum phonon frequencies to determine whether
            phonons are included in the calculation.

        Returns
        -------
        DynamicStructureFactor
            The initialized (not yet run) calculation. The same object
            is also accessible through the ``dynamic_structure_factor``
            property.

        """
        if self._mesh is None:
            msg = (
                "run_mesh has to be done before initializing dynamic structure factor."
            )
            raise RuntimeError(msg)

        if not self._mesh.with_eigenvectors:
            msg = "run_mesh has to be called with with_eigenvectors=True."
            raise RuntimeError(msg)

        if np.prod(self._mesh.mesh_numbers) != len(self._mesh.ir_grid_points):
            msg = "run_mesh has to be done with is_mesh_symmetry=False."
            raise RuntimeError(msg)

        self._dynamic_structure_factor = DynamicStructureFactor(
            self._mesh,
            Qpoints,
            T,
            atomic_form_factor_func=atomic_form_factor_func,
            scattering_lengths=scattering_lengths,
            freq_min=freq_min,
            freq_max=freq_max,
            factor=self._unit_conversion_factor,
        )
        return self._dynamic_structure_factor

    def run_dynamic_structure_factor(
        self,
        Qpoints: Sequence[Sequence[float]] | NDArray[np.double],
        T: float,
        atomic_form_factor_func: Callable | None = None,
        scattering_lengths: dict | None = None,
        freq_min: float | None = None,
        freq_max: float | None = None,
    ) -> DynamicStructureFactor:
        """Run dynamic structure factor calculation.

        See the detail of parameters at
        Phonopy.init_dynamic_structure_factor().

        Returns
        -------
        DynamicStructureFactor
            The calculated dynamic structure factors. The same object
            is also accessible through the
            ``dynamic_structure_factor`` property.

        """
        self.init_dynamic_structure_factor(
            Qpoints,
            T,
            atomic_form_factor_func=atomic_form_factor_func,
            scattering_lengths=scattering_lengths,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        assert self._dynamic_structure_factor is not None
        self._dynamic_structure_factor.run()
        return self._dynamic_structure_factor

    def get_dynamic_structure_factor(
        self,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Return dynamic structure factors.

        .. deprecated::
            Use the ``qpoints`` and ``dynamic_structure_factors``
            attributes of the ``DynamicStructureFactor`` object
            returned by ``run_dynamic_structure_factor()`` (or of the
            ``dynamic_structure_factor`` property) instead.

        """
        warnings.warn(
            "get_dynamic_structure_factor() is deprecated. Use the qpoints "
            "and dynamic_structure_factors attributes of the "
            "DynamicStructureFactor object returned by "
            "run_dynamic_structure_factor() or of the "
            "dynamic_structure_factor property.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._dynamic_structure_factor is None:
            msg = (
                "run_dynamic_structure_factor has to be done before "
                "getting dynamic structure factor."
            )
            raise RuntimeError(msg)
        return (
            self._dynamic_structure_factor.qpoints,
            self._dynamic_structure_factor.dynamic_structure_factors,
        )

    def init_random_displacements(
        self,
        dist_func: Literal["quantum", "classical"] | None = None,
        cutoff_frequency: float | None = None,
        max_distance: float | None = None,
    ) -> None:
        """Initialize random displacements at finite temperature.

        Parameters
        ----------
        dist_func : str or None, optional
            Harmonic oscillator distribution function: either
            ``'quantum'`` or ``'classical'``. Default is None,
            corresponding to ``'quantum'``.
        cutoff_frequency : float or None
            Phonon frequency in THz below which phonons are ignored when
            generating random displacements. Default is None.
        max_distance : float or None, optional
            In random displacements generation from canonical ensemble of
            harmonic phonons, displacements larger than max distance are
            renormalized to the max distance, i.e., a displacement ``d``
            is shortened by ``d -> d / norm(d) * max_distance`` if
            ``norm(d) > max_distance``.

        """
        if self._force_constants is None:
            msg = "Force constants have not yet been set."
            raise RuntimeError(msg)

        self._random_displacements = RandomDisplacements(
            self._supercell,
            self._primitive,
            self._force_constants,
            dist_func=dist_func,
            cutoff_frequency=cutoff_frequency,
            max_distance=max_distance,
            factor=self._unit_conversion_factor,
            use_openmp=c_use_openmp(),
        )

    def get_random_displacements_at_temperature(
        self,
        temperature: float,
        number_of_snapshots: int,
        is_plusminus: bool = False,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Generate random displacements from phonon structure.

        See :meth:`generate_displacements` for related details.

        Parameters
        ----------
        temperature : float
            Temperature in K.
        number_of_snapshots : int
            Number of snapshots with random displacements to create.
        is_plusminus : bool, optional
            If True, concatenate the displacements with their
            opposites, doubling the number of snapshots. Default is
            False.
        random_seed : 32-bit unsigned int or None, optional
            Random seed. Default is None.

        """
        if self._random_displacements is None:
            raise RuntimeError(
                "Phonopy.init_random_displacements has to be called "
                "before calling this method."
            )
        self._random_displacements.run(
            temperature,
            number_of_snapshots=number_of_snapshots,
            random_seed=random_seed,
        )
        units = get_calculator_physical_units(self._calculator)
        assert self._random_displacements.u is not None
        d = np.array(
            self._random_displacements.u / units.distance_to_A,
            dtype="double",
            order="C",
        )
        if is_plusminus is True:
            d = np.array(
                np.concatenate((d, -d), axis=0),
                dtype="double",
                order="C",
            )
        return d

    def save(
        self,
        filename: str | os.PathLike = "phonopy_params.yaml",
        settings: dict | None = None,
        hdf5_settings: dict | None = None,
        compression: str | bool = False,
    ) -> str:
        """Save phonopy parameters into file.

        Parameters
        ----------
        filename : str, optional
            File name. Default is ``"phonopy_params.yaml"``.
        settings : dict, optional
            Selects which parameters are written out. Only the entries
            to be changed from the defaults need to be set. The
            available parameters and their default settings are::

                {'force_sets': True,
                 'displacements': True,
                 'force_constants': False,
                 'born_effective_charge': True,
                 'dielectric_constant': True}

            The default settings are updated by ``{'force_constants': True}``
            when ``dataset`` is None and ``force_constants`` is not None,
            unless ``{'force_constants': False}`` is specified explicitly.
        hdf5_settings : dict, optional (to be implemented)
            Force constants and force sets are stored in an HDF5 file when
            activated in the dict. The default filename is the filename of
            the yaml file with ``.yaml`` replaced by ``.hdf5``. Keys::

                {'filename': str,
                 'force_constants': bool (default=False),
                 'force_sets': bool (default=False)}

        compression : bool or str
            If True, the ``phonopy_params.yaml`` like file is compressed
            by xz. When ``compression == 'xz'``, the file is compressed
            by xz. Default is False.

        Returns
        -------
        str
            File name of the saved ``phonopy_params.yaml`` like file
            (with an ``.xz`` suffix if compressed).

        """
        if hdf5_settings is not None:
            msg = "hdf5_settings parameter has not yet been implemented."
            raise NotImplementedError(msg)

        if settings is None:
            _settings = {}
        else:
            _settings = settings.copy()
        if _settings.get("force_constants") is False:
            pass
        elif not forces_in_dataset(self.dataset) and self.force_constants is not None:
            _settings.update({"force_constants": True})
        phpy_yaml = self.to_phonopy_yaml(settings=_settings)

        if compression == "xz" or compression is True:
            out_filename = f"{filename}.xz"
            with lzma.open(f"{out_filename}", "wt") as w:
                w.write(str(phpy_yaml))
        else:
            with open(filename, "w") as w:
                out_filename = str(filename)
                w.write(str(phpy_yaml))

        return out_filename

    def ph2ph(
        self,
        supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64],
        with_nac: bool = False,
    ) -> Phonopy:
        """Transform force constants in this Phonopy instance to another shape.

        Force constants are Fourier-interpolated. This Phonopy instance
        must already hold force constants. The init parameters of this
        instance are copied to the returned instance.

        For example, if the current ``supercell_matrix`` is ``[2, 2, 2]``
        and the given ``supercell_matrix`` is ``[4, 4, 4]``, the
        existing force constants are Fourier-interpolated by sampling at
        the commensurate points of the latter supercell, and a new
        Phonopy instance carrying the interpolated force constants is
        returned.

        Parameters
        ----------
        supercell_matrix : array_like
            Specifies the shape of the new force constants.
        with_nac : bool, optional
            Use non-analytical term correction (NAC) under the Fourier
            interpolation: dynamical matrices at commensurate points
            are computed with NAC, then Fourier-transformed back to
            force constants of the requested ``supercell_matrix``. NAC
            parameters are not copied to the returned Phonopy instance.

        Returns
        -------
        Phonopy
            Phonopy instance carrying the init parameters of this
            instance and the transformed force constants for the given
            ``supercell_matrix``.

        """
        if self._force_constants is None:
            raise RuntimeError("Force constants are not prepared.")

        fc_shape = self._force_constants.shape
        ph_copy = self._replicate()
        ph_copy.force_constants = self._force_constants

        if with_nac and self._nac_params is not None:
            ph_copy.nac_params = self._nac_params

        ph = self._replicate(supercell_matrix)
        assert isclose(ph.primitive, ph_copy.primitive)
        d2f = DynmatToForceConstants(
            ph.primitive,
            ph.supercell,
            is_full_fc=(fc_shape[0] == fc_shape[1]),
            use_openmp=c_use_openmp(),
            lang=self._lang,
        )
        ph_copy.run_qpoints(d2f.commensurate_points, with_dynamical_matrices=True)
        assert ph_copy.qpoints is not None
        assert ph_copy.qpoints.dynamical_matrices is not None
        d2f.dynamical_matrices = ph_copy.qpoints.dynamical_matrices
        d2f.run()
        ph.force_constants = d2f.force_constants

        return ph

    def replicate(self, log_level: int | None = None) -> Phonopy:
        """Return a new instance constructed with the same init parameters.

        Notes
        -----
        The returned instance is constructed with the same init
        parameters as this one, but internal state such as force
        constants, NAC parameters, MLP, etc. is **not** carried over.
        Supercell, primitive cell, and symmetry are recomputed.

        Returns
        -------
        Phonopy
            New Phonopy instance.

        """
        return self._replicate(log_level=log_level)

    def copy(self, log_level: int | None = None) -> Phonopy:
        """Return a new instance constructed with the same init parameters.

        Deprecated. Use :meth:`replicate` instead. Despite its name,
        this method does not copy internal state such as force
        constants and NAC parameters.

        """
        warnings.warn(
            "Phonopy.copy is deprecated. Use Phonopy.replicate instead. "
            "Note that neither method carries over internal state such "
            "as force constants and NAC parameters.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.replicate(log_level=log_level)

    def to_phonopy_yaml(
        self, configuration: dict | None = None, settings: dict | None = None
    ) -> PhonopyYaml:
        """Return PhonopyYaml class instance with this data."""
        if self._unit_conversion_factor_overridden:
            phpy_yaml = PhonopyYaml(configuration=configuration, settings=settings)
        else:
            units = get_calculator_physical_units(self.calculator)
            phpy_yaml = PhonopyYaml(
                configuration=configuration, physical_units=units, settings=settings
            )
        set_data_to_phonopy_yaml(phpy_yaml, self)
        return phpy_yaml

    ###################
    # private methods #
    ###################
    def _replicate(
        self,
        supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
        log_level: int | None = None,
    ) -> Phonopy:
        """Construct a new instance with the same init parameters.

        Parameters
        ----------
        supercell_matrix : array_like or None, optional
            Supercell matrix can be specified. None gives the same supercell
            matrix as this Phonopy class instance.

        Returns
        -------
        ph : Phonopy
            New Phonopy class instance.

        """
        if supercell_matrix is None:
            smat = self._supercell_matrix
        else:
            smat = supercell_matrix
        if log_level is not None:
            _log_level = log_level
        else:
            _log_level = self._log_level
        ph = Phonopy(
            self._unitcell,
            supercell_matrix=smat,
            primitive_matrix=self._primitive_matrix,
            group_velocity_delta_q=self._gv_delta_q,
            symprec=self._symprec,
            is_symmetry=self._is_symmetry,
            use_SNF_supercell=self._use_SNF_supercell,
            calculator=self._calculator,
            log_level=_log_level,
        )
        if self._unit_conversion_factor_overridden:
            ph.unit_conversion_factor = self._unit_conversion_factor
        return ph

    def _run_force_constants_from_forces(
        self,
        is_compact_fc: bool = False,
        fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
        fc_calculator_options: str | None = None,
        decimals: int | None = None,
        log_level: int = 0,
    ) -> None:
        if self._dataset is None:
            return None

        # For mixed-species (site-mixture) supercells, the stored dataset
        # carries raw per-constituent forces shape (..., n_expanded, 3).
        # Reduce them to per-site forces here so the existing FC
        # calculator path remains unchanged. The raw forces in
        # ``self._dataset`` are kept intact. Reduction convention is
        # picked from the calculator: VASP uses a plain sum because its
        # vasprun.xml forces already carry the mixture weight factor.
        dataset_for_fc = self._dataset
        if self._supercell.has_mixtures:
            dataset_for_fc = _reduce_mixture_dataset_forces(
                self._dataset,
                self._supercell,
                mode=_mixture_reduce_mode_for_calculator(self._calculator),
            )
        # Non-merge site-mixture (weighted real species) cells use the
        # ordinary finite-difference path: the raw VASP forces and real
        # displacements give the symmetric force constants G that satisfy
        # the ordinary sum rule, so the standard symmetrizer applies. The
        # concentration weights x do not enter the force constants.

        self._force_constants = get_fc2(
            self._supercell,
            dataset_for_fc,
            primitive=self._primitive,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            is_compact_fc=is_compact_fc,
            symmetry=self._symmetry,
            log_level=log_level,
            lang=self._lang,
        )
        if decimals:
            self._force_constants = self._force_constants.round(decimals=decimals)

    def _set_dynamical_matrix(self) -> None:
        self._dynamical_matrix = None

        nac_params: NacParams | None
        if self._is_symmetry and self._nac_params is not None:
            if len(self._nac_params["born"]) != len(self._primitive):
                raise ValueError(
                    "Numbers of atoms in primitive cell and Born effective charges "
                    "are different."
                )
            borns, epsilon = symmetrize_borns_and_epsilon(
                self._nac_params["born"],
                self._nac_params["dielectric"],
                self._primitive,
                symprec=self._symprec,
                lang=self._lang,
            )
            nac_params = self._nac_params.copy()
            nac_params.update({"born": borns, "dielectric": epsilon})
        else:
            nac_params = self._nac_params

        if self._supercell is None or self._primitive is None:
            raise RuntimeError("Supercell or primitive is not created.")
        if self._force_constants is None:
            raise RuntimeError("Force constants are not prepared.")
        if self._primitive.masses is None:
            raise RuntimeError("Atomic masses are not correctly set.")
        self._dynamical_matrix = get_dynamical_matrix(
            self._force_constants,
            self._supercell,
            self._primitive,
            nac_params=nac_params,
            hermitianize=self._hermitianize_dynamical_matrix,
            log_level=self._log_level,
            use_openmp=c_use_openmp(),
            lang=self._lang,
        )
        # DynamialMatrix instance transforms force constants in correct
        # type of numpy array.
        self._force_constants = self._dynamical_matrix.force_constants

    def _invalidate_derived(
        self, level: Literal["dataset_inputs", "dm_inputs", "factor"]
    ) -> None:
        """Clear derived state after an input mutation.

        Levels cascade:

        - "dataset_inputs" clears force constants, then behaves as
          "dm_inputs".
        - "dm_inputs" clears the dynamical matrix and everything
          derived from it.
        - "factor" preserves the dynamical matrix (it does not depend
          on the unit conversion factor) but clears group velocity and
          all analyses.

        """
        if level == "dataset_inputs":
            self._force_constants = None
        if level in ("dataset_inputs", "dm_inputs"):
            self._dynamical_matrix = None
        self._group_velocity = None
        self._band_structure = None
        self._mesh = None
        self._total_dos = None
        self._pdos = None
        self._thermal_properties = None
        self._thermal_displacements = None
        self._thermal_displacement_matrices = None
        self._moment = None
        self._dynamic_structure_factor = None
        self._qpoints = None
        self._modulation = None
        self._irreps = None
        self._random_displacements = None

    def _set_group_velocity(self) -> None:
        if self._dynamical_matrix is None:
            raise RuntimeError("Dynamical matrix has not yet built.")

        self._group_velocity = GroupVelocity(
            self._dynamical_matrix,
            q_length=self._gv_delta_q,
            symmetry=self._primitive_symmetry,
            frequency_factor_to_THz=self._unit_conversion_factor,
        )

    def _search_symmetry(self) -> None:
        self._symmetry = Symmetry(
            self._supercell,
            self._symprec,
            self._is_symmetry,
            s2p_map=self._primitive.s2p_map,
            lang=self._lang,
            distinguish_symbol_index=self._distinguish_symbol_index,
        )

    def _search_primitive_symmetry(self) -> None:
        self._primitive_symmetry = Symmetry(
            self._primitive,
            self._symprec,
            self._is_symmetry,
            lang=self._lang,
            distinguish_symbol_index=self._distinguish_symbol_index,
        )

        if len(self._symmetry.pointgroup_operations) != len(
            self._primitive_symmetry.pointgroup_operations
        ):
            warnings.warn(
                "Warning: Point group symmetries of supercell and primitive"
                "cell are different.",
                UserWarning,
                stacklevel=2,
            )

    def _build_supercell(self) -> None:
        self._supercell = get_supercell(
            self._unitcell,
            self._supercell_matrix,
            is_old_style=(not self._use_SNF_supercell),
            symprec=self._symprec,
        )

    def _build_supercells_with_displacements(self) -> None:
        all_positions = []
        assert self._dataset is not None
        if "first_atoms" in self._dataset:  # type-1
            for disp in self._dataset["first_atoms"]:
                positions = self._supercell.positions
                positions[disp["number"]] += disp["displacement"]
                all_positions.append(positions)
        elif "displacements" in self._dataset:
            for disp in self._dataset["displacements"]:
                all_positions.append(self._supercell.positions + disp)
        else:
            raise RuntimeError("displacement_dataset is not set.")

        supercells = []
        for positions in all_positions:
            supercells.append(
                PhonopyAtoms(
                    species_table=self._supercell.species_table,
                    species_ids=self._supercell.species_ids,
                    masses=self._supercell.masses,
                    magnetic_moments=self._supercell.magnetic_moments,
                    positions=positions,
                    cell=self._supercell.cell,
                )
            )
        self._supercells_with_displacements = supercells

    def _build_primitive_cell(self) -> None:
        """Create primitive cell.

        primitive_matrix:
          Relative axes of primitive cell to the input unit cell.
          Relative axes to the supercell is calculated by:
             supercell_matrix^-1 * primitive_matrix
          Therefore primitive cell lattice is finally calculated by:
             (supercell_lattice * (supercell_matrix)^-1 * primitive_matrix)^T

        """
        inv_supercell_matrix = np.linalg.inv(self._supercell_matrix)
        trans_mat = np.dot(inv_supercell_matrix, self._primitive_matrix)

        try:
            self._primitive = get_primitive(
                self._supercell,
                trans_mat,
                self._symprec,
                lang=self._lang,
            )
        except ValueError as exc:
            msg = (
                "Creating primitive cell is failed. "
                "PRIMITIVE_AXIS may be incorrectly specified."
            )
            raise RuntimeError(msg) from exc

    def _get_forces_energies(
        self, target: Literal["forces", "supercell_energies"]
    ) -> NDArray[np.double]:
        """Return forces and supercell energies.

        Return None if tagert data is not found.

        """
        if self._dataset is None:
            raise RuntimeError("Displacement-force dataset is not set.")
        if "displacements" in self._dataset and target in self._dataset:  # type-2
            return self._dataset[target]  # type: ignore
        if "first_atoms" in self._dataset:  # type-1
            values = []
            for disp in self._dataset["first_atoms"]:
                if target == "forces":
                    if target in disp:
                        values.append(disp[target])
                elif target == "supercell_energies":
                    if "supercell_energy" in disp:
                        values.append(disp["supercell_energy"])
            if values:
                return np.array(values, dtype="double", order="C")
        raise RuntimeError(f"{target} is not found in displacement-force dataset.")

    def _set_forces_energies(
        self,
        values: Sequence[float]
        | NDArray[np.double]
        | Sequence[NDArray[np.double]]
        | Sequence[Sequence[Sequence[float]]],
        target: Literal["forces", "supercell_energies"],
    ) -> None:
        assert self._dataset is not None
        if "first_atoms" in self._dataset:  # type-1
            for disp, val in zip(self._dataset["first_atoms"], values, strict=True):  # type: ignore
                if target == "forces":
                    disp[target] = np.array(val, dtype="double", order="C")
                elif target == "supercell_energies":
                    disp["supercell_energy"] = float(val)  # type: ignore
        elif "displacements" in self._dataset:  # type-2
            _values = np.array(values, dtype="double", order="C")
            natom = len(self._supercell)
            ndisps = len(self._dataset["displacements"])
            if target == "forces" and (
                _values.ndim != 3 or _values.shape != (ndisps, natom, 3)
            ):
                raise RuntimeError(f"Array shape of input {target} is incorrect.")
            elif target == "supercell_energies":
                if _values.ndim != 1 or _values.shape != (ndisps,):
                    raise RuntimeError(f"Array shape of input {target} is incorrect.")
            self._dataset[target] = _values
        else:
            raise RuntimeError("Set of displacements is not available.")


def _reduce_mixture_dataset_forces(
    dataset: DisplacementDataset,
    supercell: PhonopyAtoms,
    mode: Literal["weighted_sum", "sum"] = "weighted_sum",
) -> DisplacementDataset:
    """Return a shallow-copied dataset with forces reduced to per-site values.

    Used to feed the FC calculator a per-site force tensor while leaving the
    raw expanded forces in the user-visible dataset. ``mode`` selects the
    per-site reduction convention; see ``reduce_mixture_forces``.

    """
    if "first_atoms" in dataset:
        d1 = cast(Type1DisplacementDataset, dataset)
        new_first_atoms = []
        for entry in d1["first_atoms"]:
            new_entry = dict(entry)
            if "forces" in entry:
                new_entry["forces"] = reduce_mixture_forces(
                    entry["forces"], supercell, mode=mode
                )
            new_first_atoms.append(new_entry)
        new_dataset: dict = {"first_atoms": new_first_atoms, "natom": d1["natom"]}
        return cast(DisplacementDataset, new_dataset)

    d2 = cast(Type2DisplacementDataset, dataset)
    new_dataset = dict(d2)
    if "forces" in d2:
        new_dataset["forces"] = reduce_mixture_forces(
            d2["forces"], supercell, mode=mode
        )
    return cast(DisplacementDataset, new_dataset)


def _mixture_reduce_mode_for_calculator(
    calculator: str | None,
) -> Literal["weighted_sum", "sum"]:
    """Return the mixture-reduce convention appropriate for ``calculator``.

    VASP folds the mixture weights into the SCF, so the per-row forces in
    vasprun.xml already carry the weight factor: a plain sum across
    constituents is correct. Other calculators have not yet been wired
    for site-mixture; for them we default to the explicit weighted sum
    so the per-site force matches the standard weighted-mixture expression.

    """
    if calculator is None or calculator == "vasp":
        return "sum"
    return "weighted_sum"


def set_data_to_phonopy_yaml(phpy_yaml: PhonopyYaml, self: Phonopy) -> None:
    """Set data to PhonopyYaml instance."""
    phpy_yaml.unitcell = self.unitcell
    phpy_yaml.primitive = self.primitive
    phpy_yaml.supercell = self.supercell
    phpy_yaml.version = self.version
    phpy_yaml.supercell_matrix = self.supercell_matrix
    phpy_yaml.symmetry = self.symmetry
    phpy_yaml.primitive_matrix = self.primitive_matrix
    phpy_yaml.nac_params = self.nac_params
    phpy_yaml.frequency_unit_conversion_factor = self.unit_conversion_factor
    phpy_yaml.calculator = self.calculator
    if self.force_constants is not None:
        phpy_yaml.force_constants = self.force_constants
    if self.dataset is not None:
        phpy_yaml.dataset = self.dataset
