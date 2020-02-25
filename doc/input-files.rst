Input files
===========

.. contents::
   :depth: 2
   :local:

Setting file
-------------

A setting file contains phonopy settings which are summarized at
:ref:`setting_tags`. This file is passed to phonopy as an argument,
e.g.,

::

   % phonopy phonopy.conf

where the filename is arbitrary.

``phonopy.yaml`` and ``phonopy_disp.yaml``
------------------------------------------

These are output files after the calculation or creating the
displacements. These files contain the crystal structure information,
primitive cell and supercell sizes, and also the calculator
interface. Therefore with this file, users will not need to specify
those crystal sturcutre related tags. This file format can be used
with :ref:`cell_filename_tag` tag or ``-c`` option::

   $ phonopy -c phonopy_disp.yaml

``FORCE_SETS``, ``BORN``, and ``FORCE_CONSTANTS`` information can be
also stored in ``phonopy.yaml`` as the output after running phonopy, e.g.,

::

   $ phonopy -c phonopy_disp.yaml --include-all --nac

Structure file
--------------

Crystal structure is described by a file with specific format for each
calculator, though the default crystal structure is written in VASP
POSCAR format. See the detail of the calculator interfaces at
:ref:`calculator_interfaces`.

VASP POSCAR like format
~~~~~~~~~~~~~~~~~~~~~~~

In the following, the VASP POSCAR format that phonopy can parse is
explained. The format is simple. The first line is for your comment,
where you can write anything you want. The second line is the ratio
for lattice parameters. You can multiply by this number. The third to
fifth lines give the lattice parameters, *a*, *b*, and *c* for the
respective lines. The sixth line contains the number of atoms for each
atomic species, which have to correspond to the atomic positions in
the order. The seventh line should be written as ``Direct``. This
means that the atomic positions are represented in fractional
(reduced) coordinates. When you write chemical symbols in the first
line, they are read and those defined by the ``ATOM_NAME`` tag are
overwritten.

.. _example_POSCAR1:

Example of rutile-type silicon oxide crystal structure (VASP 4 style)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   Si O
      1.00000000000000
        4.2266540199664249    0.0000000000000000    0.0000000000000000
        0.0000000000000000    4.2266540199664249    0.0000000000000000
        0.0000000000000000    0.0000000000000000    2.6888359272289208
    2   4
   Direct
     0.0000000000000000  0.0000000000000000  0.0000000000000000
     0.5000000000000000  0.5000000000000000  0.5000000000000000
     0.3067891334429594  0.3067891334429594  0.0000000000000000
     0.6932108665570406  0.6932108665570406  0.0000000000000000
     0.1932108665570406  0.8067891334429594  0.5000000000000000
     0.8067891334429594  0.1932108665570406  0.5000000000000000

Example of rutile-type silicon oxide crystal structure (VASP 5 style)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The VASP 5.x style is also supported. Chemical symbols are inserted
just before the line of the numbers of atoms. The chemical symbols in
this line overwrite those defined by the ``ATOM_NAME`` tag and those
defined by the first line of ``POSCAR``.

::

   Stishovite
      1.00000000000000
        4.2266540199664249    0.0000000000000000    0.0000000000000000
        0.0000000000000000    4.2266540199664249    0.0000000000000000
        0.0000000000000000    0.0000000000000000    2.6888359272289208
   Si   O
    2   4
   Direct
     0.0000000000000000  0.0000000000000000  0.0000000000000000
     0.5000000000000000  0.5000000000000000  0.5000000000000000
     0.3067891334429594  0.3067891334429594  0.0000000000000000
     0.6932108665570406  0.6932108665570406  0.0000000000000000
     0.1932108665570406  0.8067891334429594  0.5000000000000000
     0.8067891334429594  0.1932108665570406  0.5000000000000000

.. _file_forces:

Force file (``FORCE_SETS``)
----------------------------

Two types of ``FORCE_SETS`` formats are supported.

.. _file_forces_type_1:

Type 1
~~~~~~

This format is the default format of phonopy and force constants can
be calculated by built-in force constants calculator of phonopy by
finite difference method, though external force constants calculator
can be also used to obtain force constants with this format by the
fitting approach.

This file gives sets of forces in supercells with finite atomic
displacements. Each supercell involves one displaced atom.  The first
line is the number of atoms in supercell. The second line gives number
of calculated supercells with displacements. Below the lines, sets of
forces with displacements are written. In each set, firstly the atom
number in supercell is written. Secondary, the atomic displacement in
**Cartesian coordinates** is written. Below the displacement line,
atomic forces in **Cartesian coordinates** are successively
written. This is repeated for the set of displacements. Blank likes
are simply ignored.

In the following example, the third line is the displaced atom number
that corresponds to the atom number in the supercell created by
phonopy. The fourth line gives the displacements in **Cartesian
coordinates**. The lines below, the atomic forces in **Cartesian
coordinates** are written. Once all the forces for a supercell are
written, the next set of forces are written. This routine is repeated
until the forces of all the displacements have been written.

Example
^^^^^^^
::

   48
   2

   1
     0.0050650623043761   0.0000000000000000   0.0086223630086415
     -0.0347116200   -0.0000026500   -0.0679795200
      0.0050392400   -0.0015711700   -0.0079514600
      0.0027380900   -0.0017851900   -0.0069206400
   ... (continue until all the forces for this displacement have written)

   25
     0.0050650623043761   0.0000000000000000   0.0086223630086415
     -0.0017134500   -0.0001539800    0.0017333400
      0.0013248100    0.0001984300   -0.0001203700
     -0.0001310200   -0.0007955600    0.0003889300
   ... (continue until all the forces for this displacement have written)

.. _file_forces_type_2:

Type 2
~~~~~~~

Equivalent to ``DFSET`` of `ALM code
<https://alm.readthedocs.io/en/develop/format-dfset.html#format-of-dfset>`_.

Each line has exactly 6 elements. The first three and second three
elements give displacement and force of an atom in a supercell,
respectively. One set with the number of lines of supercell atoms
corresponds to one supercell calculation and the number of supercell
calculations are concatenated as many as the user likes. This file is
parsed to finally get displacements and forces to have the array
shapes of ``displacements.shape = (num_supercells, num_atoms, 3)`` and
``forces.shape = (num_supercells, num_atoms, 3)``.

Force constants can be calculated by the fitting approach and this
force constants calculation requires external force constants
calculator such as `ALM
<https://alm.readthedocs.io/en/develop/index.html>`_ (invoked by
``--alm`` option). All the data are used for calculating force
constants in the fitting (usually least square fitting) by the force
constants calculator.

Example
^^^^^^^

::

     0.00834956     0.00506291     0.00215683    -0.01723508    -0.00418148    -0.00376513
    -0.00494556     0.00866021    -0.00073630     0.00849148    -0.01091833    -0.00458456
    -0.00403290    -0.00837741     0.00368169     0.00476247     0.00907379    -0.00210179
    -0.00462319     0.00361350    -0.00809745     0.00996582    -0.00320343     0.01904460
     0.00496785    -0.00596540    -0.00630352    -0.01882121    -0.00100787     0.01681980
    ...

.. _file_force_constants:

``FORCE_CONSTANTS`` and ``force_constants.hdf5``
--------------------------------------------------

If the force constants of a supercell are known, it is not necessary
to prepared ``FORCES``. Phonopy has an interface to read and write
``FORCE_CONSTANTS`` or ``force_constants.hdf5``.  To read and write
these files are controlled by :ref:`force constants tags
<force_constants_tag>` and :ref:`fc_format_tag`. VASP users can use
:ref:`VASP DFPT interface <vasp_force_constants>` to create
``FORCE_CONSTANTS`` from ``vasprun.xml``. Quantum ESPRESSO users can
use ``q2r.x`` to create force constants file by followng the
instraction shown at :ref:`qe_q2r`

Force constants are stored in either array shape of

- Compact format: ``(n_patom, n_satom, 3, 3)``
- Full format: ``(n_satom, n_satom, 3, 3)``

where ``n_satom`` and ``n_patom`` are the numbers of atoms in
supercell and primitive cell, respectively.

Format of ``FORCE_CONSTANTS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First line contains the first two elements of the shape of the force
constants array, i.e., for ``(n_satom, n_satom, 3, 3)``, the first and
second numbers are the same and are the number of atoms in the
supercell, and for ``(n_patom, n_satom, 3, 3)``, they are the numbers
of atoms in the primitive cell and supercell. If the first line
contains only one number, it is assumed same as that of the former case.

Below second line,
force constants between atoms are written by every four lines. In
first line of the four lines, anything can be written, i.e., just
ignored. Second to fourth lines of the four lines are for the second
rank tensor of force constant in Cartesian coordinates, i.e.:::

   xx xy xz
   yx yy yz
   zx zy zz

Example
~~~~~~~

::

   32  32
   1   1
     4.635786969900131    -0.000000000000000    -0.000000000000000
    -0.000000000000000     4.635786969900130    -0.000000000000000
    -0.000000000000000    -0.000000000000000     4.635786969900130
   1   2
    -0.246720998398056    -0.000000000000000    -0.000000000000000
    -0.000000000000000     0.018256999881458    -0.000000000000000
    -0.000000000000000    -0.000000000000000     0.018256999881458
   ...
   1  32
     0.002646999982813     0.018011999883049    -0.000000000000000
     0.018011999883049     0.002646999982813    -0.000000000000000
    -0.000000000000000    -0.000000000000000     0.035303999770773
   2   1
    -0.246720998398056     0.000000000000000     0.000000000000000
     0.000000000000000     0.018256999881458     0.000000000000000
     0.000000000000000     0.000000000000000     0.018256999881458
   ...
   32  32
     4.635786969900131     0.000000000000000     0.000000000000000
     0.000000000000000     4.635786969900130     0.000000000000000
     0.000000000000000     0.000000000000000     4.635786969900130

Format of ``force_constants.hdf5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an alternative of ``FORCE_CONSTANTS`` but the data is stored
in HDF5 format. See the detail of how to obtain this file,
:ref:`fc_format_tag`.

The data are stored as follows. ``p2s_map`` is introduced at version
1.12.6. Force constants data can be stored in the array shape of
either ``(n_satom, n_satom, 3, 3)`` or ``(n_patom, n_satom, 3, 3)``.
In the later case, ``p2s_map`` is necessary for the consistency check
and this gives the indices of atoms in the primitive cell in supercell
index system.

::

   In [1]: import h5py
   f
   In [2]: f = h5py.File("force_constants.hdf5", 'r')

   In [3]: list(f)
   Out[3]: ['force_constants', 'p2s_map']

   In [4]: f['force_constants'].shape
   Out[4]: (2, 64, 3, 3)

   In [5]: f['p2s_map'][:]
   Out[5]: array([ 0, 32], dtype=int32)

.. _qpoints_file:

``QPOINTS`` (optional)
-----------------------

Specific q-points are calculated using ``QPOINTS = .TRUE.`` tag and
``QPOINTS`` file. The file format of ``QPOINTS`` is as follows. The
first line gives the number of q-points. Then the successive lines
give q-points in reduced coordinate of reciprocal space of the input
unit cell.

Example
~~~~~~~
::

   512
   -0.437500000000000  -0.437500000000000  -0.437500000000000
   -0.312500000000000  -0.437500000000000  -0.437500000000000
   -0.187500000000000  -0.437500000000000  -0.437500000000000
   ...

.. _born_file:

``BORN`` (optional)
-----------------------

This file is used with the ``--nac`` option or ``NAC`` tag.

The formula implemented is refered to :ref:`non_analytical_term_correction_theory`.

Format
~~~~~~

In the first line, unit conversion factor is given. In versions 1.10.4
or later, the default value for each calculater can be used if
characters than numerical number are given. The default values for the
calculaters are found at :ref:`nac_default_value_interfaces`.

In the second line, dielectric constant :math:`\epsilon` is specifed
in Cartesian coordinates. The nine values correspond to the tensor
elements of xx, xy, xz, yx, yy, yz, zx, zy, and zz.

From the third line, Born effective charges :math:`Z` for the
independent atoms in the **primitive cell** have to be written in
Cartesian coordinates. The independent atoms can be found using the
``-v`` option. As shown below in the Al2O3 example, the independent
atoms are marked by ``*`` in front of atomic positions::

   % phonopy --dim="2 2 1" --pa="2/3 -1/3 -1/3  1/3 1/3 -2/3  1/3 1/3 1/3" -v
           _
     _ __ | |__   ___  _ __   ___   _ __  _   _
    | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
    | |_) | | | | (_) | | | | (_) || |_) | |_| |
    | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
    |_|                            |_|    |___/

                                        1.8.4.2

   Settings:
     Supercell:  [2 2 1]
     Primitive axis:
        [ 0.66666667 -0.33333333 -0.33333333]
        [ 0.33333333  0.33333333 -0.66666667]
        [ 0.33333333  0.33333333  0.33333333]
   Spacegroup:  R-3c (167)
   ---------------------------- primitive cell -------------------------------
   Lattice vectors:
     a    2.403817201137804    1.387844508159565    4.372423306604251
     b   -2.403817201137804    1.387844508159565    4.372423306604251
     c    0.000000000000000   -2.775689016319131    4.372423306604251
   Atomic positions (fractional):
      *1 Al  0.35218509422890  0.35218509422890  0.35218509422890  26.982
       2 Al  0.64781490577110  0.64781490577110  0.64781490577110  26.982
       3 Al  0.14781490577110  0.14781490577110  0.14781490577110  26.982
       4 Al  0.85218509422890  0.85218509422890  0.85218509422890  26.982
      *5 O   0.55616739064549  0.94383260935451  0.25000000000000  15.999
       6 O   0.44383260935451  0.05616739064549  0.75000000000000  15.999
       7 O   0.25000000000000  0.55616739064549  0.94383260935451  15.999
       8 O   0.75000000000000  0.44383260935451  0.05616739064549  15.999
       9 O   0.94383260935451  0.25000000000000  0.55616739064549  15.999
      10 O   0.05616739064549  0.75000000000000  0.44383260935451  15.999
   ------------------------------ unit cell ----------------------------------
   Lattice vectors:
     a    4.807634402275609    0.000000000000000    0.000000000000000
     b   -2.403817201137805    4.163533524478696    0.000000000000000
     c    0.000000000000000    0.000000000000000   13.117269919812754
   Atomic positions (fractional):
      *1 Al  0.00000000000000  0.00000000000000  0.35218509422890  26.982 > 1
       2 Al  0.66666666666666  0.33333333333334  0.68551842756224  26.982 > 1
       3 Al  0.33333333333334  0.66666666666666  0.01885176089557  26.982 > 1
       4 Al  0.00000000000000  0.00000000000000  0.64781490577110  26.982 > 2
       5 Al  0.66666666666666  0.33333333333334  0.98114823910443  26.982 > 2
       6 Al  0.33333333333334  0.66666666666666  0.31448157243776  26.982 > 2
       7 Al  0.00000000000000  0.00000000000000  0.14781490577110  26.982 > 3
       8 Al  0.66666666666666  0.33333333333334  0.48114823910443  26.982 > 3
       9 Al  0.33333333333334  0.66666666666666  0.81448157243776  26.982 > 3
      10 Al  0.00000000000000  0.00000000000000  0.85218509422890  26.982 > 4
      11 Al  0.66666666666666  0.33333333333334  0.18551842756224  26.982 > 4
      12 Al  0.33333333333334  0.66666666666666  0.51885176089557  26.982 > 4
     *13 O   0.30616739064549  0.00000000000000  0.25000000000000  15.999 > 5
      14 O   0.97283405731215  0.33333333333334  0.58333333333334  15.999 > 5
      15 O   0.63950072397883  0.66666666666666  0.91666666666666  15.999 > 5
      16 O   0.69383260935451  0.00000000000000  0.75000000000000  15.999 > 6
      17 O   0.36049927602117  0.33333333333334  0.08333333333334  15.999 > 6
      18 O   0.02716594268785  0.66666666666666  0.41666666666666  15.999 > 6
      19 O   0.00000000000000  0.30616739064549  0.25000000000000  15.999 > 7
      20 O   0.66666666666666  0.63950072397883  0.58333333333334  15.999 > 7
      21 O   0.33333333333334  0.97283405731215  0.91666666666666  15.999 > 7
      22 O   0.00000000000000  0.69383260935451  0.75000000000000  15.999 > 8
      23 O   0.66666666666666  0.02716594268785  0.08333333333334  15.999 > 8
      24 O   0.33333333333334  0.36049927602117  0.41666666666666  15.999 > 8
      25 O   0.69383260935451  0.69383260935451  0.25000000000000  15.999 > 9
      26 O   0.36049927602117  0.02716594268785  0.58333333333334  15.999 > 9
      27 O   0.02716594268785  0.36049927602117  0.91666666666666  15.999 > 9
      28 O   0.30616739064549  0.30616739064549  0.75000000000000  15.999 > 10
      29 O   0.97283405731215  0.63950072397883  0.08333333333334  15.999 > 10
      30 O   0.63950072397883  0.97283405731215  0.41666666666666  15.999 > 10
   ------------------------------ supercell ----------------------------------
   ...

If VASP is used as the calculator for Born effective charge, and the
hexagonal unit cell is used for the calculation, the Born effective
charge tensors of atoms No. 1 and 13 have to be written in ``BORN``
file.

Example
~~~~~~~
::

    14.400
    3.269  0.000  0.000  0.000  3.269  0.000  0.000  0.000  3.234
    2.981  0.000  0.000  0.000  2.981  0.000  0.000  0.000  2.952
   -1.935  0.000  0.000  0.000 -2.036 -0.261  0.000 -0.261 -1.968

or using the default NAC unit conversion factor (version 1.10.4 or later),

::

   default value
    3.269  0.000  0.000  0.000  3.269  0.000  0.000  0.000  3.234
    2.981  0.000  0.000  0.000  2.981  0.000  0.000  0.000  2.952
   -1.935  0.000  0.000  0.000 -2.036 -0.261  0.000 -0.261 -1.968
