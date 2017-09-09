Crystal symmetry
=================

Tolerance used in crystal symmetry search
------------------------------------------

Phonon calculation is based on the assumption that atoms have thier
own equilibrium positions where forces on these atoms are zero. In
addition, there is a unit cell that contains these atoms and the unit
cell are repeated in the direct space, i.e., it forms a lattice. The
lattice vectors (or basis vectors) and points of atoms in this unit
cell give the information of the crystal structure.

The crystal structure may have a specific symmetry. The categorization
of the crystal symmetried is achieved by the group theory about
symmetry operations, and there are the 230 different space group
types. In phonopy, the crystal symmetry is automatically analyzed from
the input unit cell structure file that doesn't contain the symmetry
information. Symmetries are searched by attemping possible symmetry
operations to the crsytal structure and cheking if the crystal
structure after the symmetry operation is overlapping to the original
crystal structures. In this analysis, a tolerance of distance is
used to tolerate small deviation of overlapping. This tolerance is
the user's choice. The default value is ``1e-5``.

Often we know the space group type of our crystal. Therefore it is
recommended to check whether the space group type of the input unit
cell is the correct one or not with very tight value such as
``--tolerance=1e-8`` using :ref:`tolerance_option` option. If an input
unit cell structure is naively distorted for which the distortion is
about the same order of the chosen tolerance, inconsistency in
symmetry handling may occur and it can result in a crash of the
calculation or induce a strange calculation result. This can be
checked by changing the tolerance value variously and watching the
obtained space group type. If an input unit cell structure is
distorted, different space group types are found with respect to the
different tolerance values.

The detailed space group information is obtained by
:ref:`symmetry_option` option. As a result of using this option and
the chosen tolerance value, ``BPOSCAR`` file is obtained. This is a
standardized conventional unit cell (see
https://atztogo.github.io/spglib/definition.html#conventions-of-standardized-unit-cell),
and its distortion against the crystal symmetry is very small. It is
recommended to used this structure as the starting point of phonopy
calculation.
