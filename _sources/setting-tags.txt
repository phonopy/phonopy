.. _setting_tags:

Setting tags
============

Most of the setting tags have corresponding command-line options (:ref:`command_options`).

For specifying real and reciprocal points, fractional values
(e.g. ``1/3``) are accepted. However fractional values must not
have space among characters (e.g. ``1 / 3``) are not allowed.


Basic tags
----------

``DIM``
~~~~~~~~~~

The supercell is created from the input unit cell. When three integers
are specified, a supercell elongated along axes of unit cell is
created.

::

   DIM = 2 2 3

In this case, a 2x2x3 supercell is created.


When nine integers are specified, the supercell is created by
multiplying the supercell matrix :math:`M_\mathrm{s}` with the unit
cell. For example,

::

   DIM = 0 1 1  1 0 1  1 1 0

the supercell matrix is

.. math::

   M_\mathrm{s} = \begin{pmatrix}
   0 & 1 & 1 \\
   1 & 0 & 1 \\
   1 & 1 & 0 
   \end{pmatrix}

where the rows correspond to the first three, second three, and third
three sets of numbers, respectively. When lattice parameters of unit
cell are the column vectors of :math:`\mathbf{a}_\mathrm{u}`,
:math:`\mathbf{b}_\mathrm{u}`, and :math:`\mathbf{c}_\mathrm{u}`,
those of supercell, :math:`\mathbf{a}_\mathrm{s}`,
:math:`\mathbf{b}_\mathrm{s}`, :math:`\mathbf{c}_\mathrm{s}`, are
determined by,

.. math::

   ( \mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \; \mathbf{c}_\mathrm{s} )
   =  ( \mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u} \;
   \mathbf{c}_\mathrm{u} ) M_\mathrm{s} 

Be careful that the axes in ``POSCAR`` is defined by three row
vectors, i.e., :math:`( \mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u}
\; \mathbf{c}_\mathrm{u} )^T`.

.. _primitive_axis_tag:

``PRIMITIVE_AXIS``
~~~~~~~~~~~~~~~~~~
::

   PRIMITIVE_AXIS = 0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0

Likewise,

::

   PRIMITIVE_AXIS = 0 1/2 1/2  1/2 0 1/2  1/2 1/2 0

The primitive cell for building the dynamical matrix is created by
multiplying primitive-axis matrix :math:`M_\mathrm{p}`. Let the matrix
as,

.. math::

   M_\mathrm{p} = \begin{pmatrix}
   0.0 & 0.5 & 0.5 \\
   0.5 & 0.0 & 0.5 \\
   0.5 & 0.5 & 0.0
   \end{pmatrix}

where the rows correspond to the first three, second three, and
third three sets of numbers, respectively.

When lattice parameters of unit cell (set by ``POSCAR``) are the
column vectors of :math:`\mathbf{a}_\mathrm{u}`,
:math:`\mathbf{b}_\mathrm{u}`, and :math:`\mathbf{c}_\mathrm{u}`,
those of supercell, :math:`\mathbf{a}_\mathrm{p}`,
:math:`\mathbf{b}_\mathrm{p}`, :math:`\mathbf{c}_\mathrm{p}`, are
determined by,

.. math::

   ( \mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \; \mathbf{c}_\mathrm{p} )
   =  ( \mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u} \;
   \mathbf{c}_\mathrm{u} ) M_\mathrm{p} 

Be careful that the axes in ``POSCAR`` is defined by three row
vectors, i.e., :math:`( \mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u}
\; \mathbf{c}_\mathrm{u} )^T`.

``ATOM_NAME``
~~~~~~~~~~~~~

Chemical symbols ::

   ATOM_NAME = Si O

The number of chemical symbols have to be same as that of the numbers
in the sixth line of ``POSCAR``.

Chemical symbols read by phonopy are overwritten by those written in
``POSCAR``. See ``POSCAR`` examples. In WIEN2k mode,
you don't need to set this tag, i.e., chemical symbols are read from
the structure file.

``EIGENVECTORS``
~~~~~~~~~~~~~~~~

When this tag is '.TRUE.', eigenvectors are calculated. With ``-p``
option, partial density of states are calculated.

.. _mass_tag:

``MASS``
~~~~~~~~

This tag is not necessary to use usually, because atomic masses are
automatically set from the chemical symbols.

Atomic masses of a **primitive cell** are overwritten by the values
specified. The order of atoms in the primitive cell that is defined by
``PRIMITIVE_AXIS`` tag can be shown using ``-v`` option. It must be
noted that this tag does not affect to the symmetry search.

For example, when there are six atoms in a primitive cell, ``MASS`` is
set as follows ::

   MASS =   28.085 28.085 16.000 16.000 16.000 16.000

.. _magmom_tag:

``MAGMOM``
~~~~~~~~~~~

Symmetry of spin such as collinear magnetic moments is considered
using this tag. The number of values has to be equal to the number of
atoms in the unit cell, not the primitive cell or supercell. If this
tag is used with ``-d`` option (``CREATE_DISPLACEMENTS`` tag),
``MAGMOM`` file is created. This file contains the ``MAGMOM``
information of the supercell used for VASP. Unlike ``MAGMOM`` in VASP,
``*`` can not be used, i.e., all the values (the same number of times
to the number of atoms in unit cell) have to be explicitly written.

::

   MAGMOM = 1.0 1.0 -1.0 -1.0

.. _dimension_tag:

``CELL_FILENAME``
~~~~~~~~~~~~~~~~~~

::

   CELL_FILENAME = UPOSCAR

See :ref:`cell_filename_option`.
    

Displacement creation tags
--------------------------

``CREATE_DISPLACEMENTS``
~~~~~~~~~~~~~~~~~~~~~~~~~

Supercells with displacements are created.  This tag is used as the
post process of phonon calculation.

::

   CREATE_DISPLACEMENTS = .TRUE.
   DIM = 2 2 2

.. _displacement_distance_tag:

``DISPLACEMENT_DISTANCE``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Finite atomic displacement distance is set as specified value when
creating supercells with displacements. The default displacement
amplitude is 0.01 :math:`\textrm{\AA}`, but when the ``wien2k`` or
``abinit`` option is specified, the default value is 0.02 Bohr.

``DIAG``
~~~~~~~~~

When this tag is set ``.FALSE.``, displacements in diagonal directions
are not searched, i.e. all the displacements are along the lattice
vectors. ``DIAG = .FALSE.`` is recommended if one of the lattice
parameter of your supercell is much longer or much shorter than the
other lattice parameters.

``PM``
~~~~~~~

This tag specified how displacements are found. When ``PM = .FALSE.``,
least displacements that can calculate force constants are found. This
may cause less accurate result. When ``PM = .TRUE.``, all the
displacements that are opposite to the least displacements are found.
The default setting is ``PM = AUTO``. Plus-minus displacements are
considered with this tag. If the plus and minus displacements are
symmetrically equivalent, only the plus displacement is found. This
may be in between ``.FALSE.`` and ``.TRUE.``. You can check how it
works to see the file ``DISP`` where displacement directions on atoms
are written.

.. _band_structure_related_tags:

Band structure tags
----------------------------

``BAND``, ``BAND_POINTS``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BAND`` gives sampling band paths. The reciprocal points are
specified in reduced coordinates. The given points are connected for
defining band paths. When comma ``,`` is inserted between the points,
the paths are disconnected.

``BAND_POINTS`` gives the number of sampling points including the path
ends. The default value is ``BAND_POINTS = 51``.


An example of three paths, (0,0,0) to (1/2,0,1/2), (1/2,1/2,1) to
(0,0,0), and (0,0,0) to (1/2,1/2,1/2), with 101 sampling points of
each path are as follows:

::

   BAND = 0 0 0  1/2 0 1/2,  1/2 1/2 1  0 0 0   1/2 1/2 1/2
   BAND_POINTS = 101

.. _band_labels_tag:

``BAND_LABELS``
~~~~~~~~~~~~~~~~~~

Labels specified are depicted in band structure plot at the points of
band segments. The number of labels has to correspond to the
number of band paths specified by ``BAND`` plus one.

::

   BAND = 1/2 0 1/2   0 0 0   1/2 1/2 1/2
   BAND_LABELS = X \Gamma L

.. |bandlabels| image:: band-labels.png
                :scale: 50

|bandlabels|

The colors of curves are automatically determined by matplotlib. The
same color in a band segment shows the same kind of band. Between
different band segments, the correspondence of colors doesn't mean
anything.

.. _band_connection_tag:

``BAND_CONNECTION``
~~~~~~~~~~~~~~~~~~~~

With this option, band connections are estimated from eigenvectors and
band structure is drawn considering band crossings. In sensitive
cases, to obtain better band connections, it requires to increase
number of points calculated in band segments by the ``BAND_POINTS`` tag.

::

   BAND = 1/2 0 1/2   0 0 0   1/2 1/2 1/2
   BAND_POINTS = 101
   BAND_CONNECTION = .TRUE.

.. |bandconnection| image:: band-connection.png
                    :scale: 50

|bandconnection|


.. _mesh_sampling_tags:

Mesh sampling tags
-------------------

Mesh sampling tags are used commonly for calculations of thermal
properties and density of states.

.. _mp_tag:

``MP``, ``MESH``
~~~~~~~~~~~~~~~~~

``MP`` numbers give uniform meshes in each axis. As the default
behavior, the center of mesh is determined by the Monkhorst-Pack
scheme, i.e., for odd number, a point comes to the center, and for
even number, the center is shifted half in the distance between
neighboring mesh points.

Examples of an even mesh with :math:`\Gamma` center in two ways,

::

   MP = 8 8 8 
   GAMMA_CENTER = .TRUE.

::

   MP = 8 8 8 
   MP_SHIFT = 1/2 1/2 1/2

``MP_SHIFT``
~~~~~~~~~~~~~~~~~~

``MP_SHIFT`` gives the shifts in direction along the corresponding
reciprocal axes (:math:`a^*`, :math:`b^*`, :math:`c^*`). 0 or 1/2
(0.5) can be used as these values. 1/2 means the half mesh shift with
respect to neighboring grid points in each direction.

``GAMMA_CENTER``
~~~~~~~~~~~~~~~~~~

Instead of employing the Monkhorst-Pack scheme for the mesh sampling,
:math:`\Gamma` center mesh is used. The default value is ``.FALSE.``.

::

   GAMMA_CENTER = .TRUE.

.. _write_mesh_tag:

``WRITE_MESH``
~~~~~~~~~~~~~~~~~

With a dense mesh, with eigenvectors, without mesh symmetry, sometimes
its output file ``mesh.yaml`` or ``mesh.hdf5`` can be huge. However
when those files are not needed, e.g., in (P)DOS calculation,
``WRITE_MESH = .FALSE.`` can disable to write out those files. With
(P)DOS calculation, DOS output files are obtained even with
``WRITE_MESH = .FALSE.``. The default setting is ``.TRUE.``.

::

   WRITE_MESH = .FALSE.


Density of states (DOS) tags
-----------------------------

Density of states (DOS) is calcualted either with smearing method
(default) or tetrahedron method. The physical unit of horizontal axis
is that of frequency that the user employs, e.g., THz, and that of
vertical axis is {no. of states}/({unit cell} x {unit of the
horizontal axis}). If the DOS is integrated over the frequency range,
it will be :math:`3N_\mathrm{a}` states, where :math:`N_\mathrm{a}` is
the number of atoms in the unit cell.

.. _dos_related_tags:

``DOS``
~~~~~~~~

This tag enables to calculate DOS. This tag is automatically set when
``PDOS`` tag or ``-p`` option.

::

   DOS = .TRUE.


   
``DOS_RANGE``
~~~~~~~~~~~~~
::

   DOS_RANGE = 0 40 0.1

Total and partial density of states are drawn with some
parameters. The example makes DOS be calculated from frequency=0 to 40
with 0.1 pitch.

``PDOS``
~~~~~~~~
::

   PDOS = 1 2, 3 4 5 6

By setting this tag, ``EIGENVECTORS = .TRUE.`` and ``MESH_SYMMETRY =
.FALSE.`` are automatically set.  ``PDOS`` tag controls how elements
of eigenvectors are added. Each value gives the atom index in
primitive cell. ``,`` separates the atom sets. Therefore in the
example, atom 1 and 2 are summarized as one curve and atom 3, 4, 5,
and, 6 are summarized as the other curve.

The projection is applied along arbitrary direction using
``PROJECTION_DIRECTION`` tag.

.. _projection_direction_tag:

``PROJECTION_DIRECTION``
~~~~~~~~~~~~~~~~~~~~~~~~

Eigenvectors are projected along the direction specified by this tag.
Projection direction is specified in reduced coordinates, i.e., with
respect to *a*, *b*, *c* axes.

::

   PDOS = 1, 2   
   PROJECTION_DIRECTION = -1 1 1

.. _sigma_tag:

``SIGMA``
~~~~~~~~~

This tag specifies the deviation of a smearing function. The unit
is same as that of final result of DOS, i.e., for VASP without
``--factor`` option, it is THz. The default value is the value given
by the difference of maximum and minimum frequencies divided by 100.

::

   SIGMA = 0.1

.. _debye_model_tag:

``TETRAHEDRON``
~~~~~~~~~~~~~~~~

Tetrahedron method is used instead of smearing method.

``DEBYE_MODEL``
~~~~~~~~~~~~~~~~

By setting ``.TRUE.``, DOS at lower phonon frequencies are fit to a
Debye model. By default, the DOS from 0 to 1/4 of the maximum phonon
frequencies are used for the fitting. The function used to the fitting
is :math:`D(\omega)=a\omega^2` where :math:`a` is the parameter and
the Debye frequency is :math:`(9N/a)^{1/3}` where :math:`N` is the
number of atoms in unit cell. Users have to unserstand that this is
**not** a unique way to determine Debye frequency. Debye frequency is
dependent on how to parameterize it.

::

   DEBYE_MODEL = .TRUE.
   
.. _thermal_properties_tag:

Thermal properties related tags
--------------------------------

``TPROP``, ``TMIN``, ``TMAX``, ``TSTEP``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thermal properties, free energy, heat capacity, and entropy, are
calculated from their statistical thermodynamic expressions
(see :ref:`thermal_properties_expressions`). Thermal properties are
calculated from phonon frequencies on a sampling mesh in the
reciprocal space. Therefore These tags are used with ``MP`` tag and
their convergence with respect to the sampling mesh has to be
checked. Usually this calculation is not computationally demanding, so
the convergence is easily achieved with increasing the density of the
sampling mesh. ``-p`` option can be used together to plot the thermal
propreties. Phonon frequencies have to be calculated in THz. Therefore
unit conversion factor to THz may be specified using ``--factor``
option. The calculated values are written into
``thermal_properties.yaml``. The unit systems of free energy, heat
capacity, and entropy are kJ/mol, J/K/mol, and J/K/mol, respectively,
where 1 mol means :math:`\mathrm{N_A}\times` your input unit cell (not
formula unit), i.e. you have to divide the value by number of formula
unit in your unit cell by yourself. For example, in MgO (conventional)
unit cell, if you want to compare with experimental results in kJ/mol,
you have to divide the phonopy output by four.

``TMIN``, ``TMAX``, and ``TSTEP`` tags are used to specify the
temperature range to be calculated. The default values of them are 0,
1000, and 10, respectively.

::

   TPROP = .TRUE.
   TMAX = 2000


.. _thermal_atomic_displacements_tags:

Thermal displacements
---------------------

.. _thermal_displacements_tag:

``TDISP``, ``TMAX``, ``TMIN``, ``TSTEP``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mean square displacements projected to Cartesian axes as a function of
temperature are calculated from the number of phonon excitations. The
usages of ``TMAX``, ``TMIN``, ``TSTEP`` tags are same as those in
:ref:`thermal properties tags <thermal_properties_tag>`. The result is
writen into ``thermal_displacements.yaml``. See the detail of the
method, :ref:`thermal_displacement`.

The projection is applied along arbitrary direction using
``PROJECTION_DIRECTION`` tag (:ref:`projection_direction_tag`).

::

   TDISP = .TRUE.
   PROJECTION_DIRECTION = 1 1 0

.. _thermal_displacement_matrices_tag:

``TDISPMAT``, ``TMAX``, ``TMIN``, ``TSTEP``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mean square displacement matricies are calculated. The difinition is
shown at :ref:`thermal_displacement`. The result is
writen into ``thermal_displacement_matrices.yaml`` where six matrix
elements are given in the order of xx, yy, zz, yz, xz, xy.

::

   TDISPMAT = .TRUE.

``CUTOFF_FREQUENCY``
~~~~~~~~~~~~~~~~~~~~~

Frequencies lower than this cutoff frequency are not used to calculate
thermal displacements.

Specific q-points
-----------------

.. _qpoints_tag:

``QPOINTS``
~~~~~~~~~~~

When ``QPOINTS = .TRUE.``, ``QPOINTS`` file in your working directory
is read, and the q-points written in this file are calculated.

.. _writedm_tag:

``WRITEDM``
~~~~~~~~~~~~

::

   WRITEDM = .TRUE.

Dynamical matrices :math:`D` are written into ``qpoints.yaml``
in the following :math:`6N\times3N` format, where *N* is the number of atoms in
the primitive cell.

The physical unit of dynamical matrix is ``[unit of force] / ([unit of
displacement] * [unit of mass])``, i.e., square of the unit of phonon
frequency before multiplying the unit conversion factor
(see :ref:`unit_conversion_factor_option`).

.. math::

   D =
   \begin{pmatrix}
   D_{11} & D_{12} & D_{13} & \\
   D_{21} & D_{22} & D_{23} & \cdots \\
   D_{31} & D_{32} & D_{33} & \\
   & \vdots &  & \\
   \end{pmatrix},

and :math:`D_{jj'}` is

.. math::
   D_{jj'} = 
   \begin{pmatrix}
   Re(D_{jj'}^{xx}) & Im(D_{jj'}^{xx}) & Re(D_{jj'}^{xy}) &
   Im(D_{jj'}^{xy}) & Re(D_{jj'}^{xz}) & Im(D_{jj'}^{xz}) \\
   Re(D_{jj'}^{yx}) & Im(D_{jj'}^{yx}) & Re(D_{jj'}^{yy}) &
   Im(D_{jj'}^{yy}) & Re(D_{jj'}^{yz}) & Im(D_{jj'}^{yz}) \\
   Re(D_{jj'}^{zx}) & Im(D_{jj'}^{zx}) & Re(D_{jj'}^{zy}) &
   Im(D_{jj'}^{zy}) & Re(D_{jj'}^{zz}) & Im(D_{jj'}^{zz}) \\
   \end{pmatrix},

where *j* and *j'* are the atomic indices in the primitive cell. The
phonon frequencies may be recovered from ``qpoints.yaml`` by writing a
simple python script. For example, ``qpoints.yaml`` is obtained for
NaCl at :math:`q=(0, 0.5, 0.5)` by

::

   phonopy --dim="2 2 2" --pa="0 1/2 1/2  1/2 0 1/2  1/2 1/2 0" --qpoints="0 1/2 1/2" --writedm

and the dynamical matrix may be used as

.. code-block:: python

   #!/usr/bin/env python
   
   import yaml
   import numpy as np
   
   data = yaml.load(open("qpoints.yaml"))
   dynmat = []
   dynmat_data = data['phonon'][0]['dynamical_matrix']
   for row in dynmat_data:
       vals = np.reshape(row, (-1, 2))
       dynmat.append(vals[:, 0] + vals[:, 1] * 1j)
   dynmat = np.array(dynmat)
   
   eigvals, eigvecs, = np.linalg.eigh(dynmat)
   frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
   conversion_factor_to_THz = 15.633302
   print frequencies * conversion_factor_to_THz


.. _nac_tag:

Non-analytical term correction 
----------------------------------

``NAC``
~~~~~~~~~~

Non-analytical term correction is applied to dynamical
matrix. ``BORN`` file has to be prepared in the current directory. See
:ref:`born` and :ref:`non_analytical_term_correction_theory`.

::

   NAC = .TRUE.

``Q_DIRECTION``
~~~~~~~~~~~~~~~~

This tag is used to activate NAC at
:math:`\mathbf{q}\rightarrow\mathbf{0}`, i.e. practically
:math:`\Gamma`-point. Away from :math:`\Gamma`-point, this setting is
ignored and the specified **q**-point is used as the **q**-direction.

::

   MP = 1 1 1
   NAC = .TRUE.
   Q_DIRECTION = 1 0 0


.. _group_velocity_tag:

Group velocity
---------------

``GROUP_VELOCITY``
~~~~~~~~~~~~~~~~~~~

Group velocities at q-points are calculated by using this tag. The
group velocities are written into a yaml file corresponding to the run
mode in Cartesian coordinates. The physical unit depends on physical
units of input files and frequency conversion factor, but if VASP and
the default settings (e.g., THz for phonon frequency) are simply used,
then the physical unit will be Angstrom THz.

::

   GROUP_VELOCITY = .TRUE.

Technical details are shown at :ref:`group_velocity`.

``GV_DELTA_Q``
~~~~~~~~~~~~~~~

The reciprocal distance used for finite difference method is
specified. The default value is 1e-4.

::

   GV_DELTA_Q = 0.01

Symmetry
---------


.. _symmetry_tag:

``SYMMETRY``
~~~~~~~~~~~~~

P1 symmetry is enforced to the input unit cell by setting ``SYMMETRY = .FALSE``.

.. _nomeshsym_tag:

``MESH_SYMMETRY``
~~~~~~~~~~~~~~~~~~

Symmetry search on the reciprocal sampling mesh is disabled by setting
``MESH_SYMMETRY = .FALSE.``.


.. _fc_symmetry_tag:

``FC_SYMMETRY``
~~~~~~~~~~~~~~~~

This tag is used to symmetrize force constants partly. The number of
iteration of the following set of symmetrization applied to force
constants is specified. The default value is 0. In the case of VASP,
this tag is usually unnecessary to be specified.

::

   FC_SYMMETRY = 1


From the translation invariance condition,

.. math::

   \sum_i \Phi_{ij}^{\alpha\beta} = 0, \;\;\text{for all $j$, $\alpha$, $\beta$},

where *i* and *j* are the atom indices, and :math:`\alpha` and
:math:`\beta` are the Catesian indices for atoms *i* and *j*,
respectively. Force constants are symmetric in each pair as

.. math::

   \Phi_{ij}^{\alpha\beta}
        = \frac{\partial^2 U}{\partial u_i^\alpha \partial u_j^\beta} 
        = \frac{\partial^2 U}{\partial u_j^\beta \partial u_i^\alpha}
	= \Phi_{ji}^{\beta\alpha}

These symmetrizations break the symmetry conditions each other. Be
careful that the other symmetries of force constants, i.e., the
symmetry from crystal symmetry or rotational symmetry, are broken to
force applying ``FC_SYMMETRY``.

.. Tolerance of the crystal symmetry search is given by phonopy option of
.. ``--tolerance``.

.. ``TRANSLATION``
.. ~~~~~~~~~~~~~~~

.. Translational invariance is forced by setting ``TRANSLATION =
.. .TRUE.``. The default value is ``.FALSE.``. The input forces are
.. summed up in each Cartesian axis and the average are subtracted from
.. the forces.

.. ``PERMUTATION``
.. ~~~~~~~~~~~~~~~

.. Symmetry of force constants:

.. .. math::

..    \Phi_{ij}^{\alpha\beta}
..         = \frac{\partial^2 U}{\partial u_i^\alpha \partial u_j^\beta} 
..         = \frac{\partial^2 U}{\partial u_j^\beta \partial u_i^\alpha}
.. 	= \Phi_{ji}^{\beta\alpha}

.. is imposed with ``PERMUTATION = .TRUE.``. The default value is
.. ``.FALSE.``. This is not necessary to be set, because dynamical
.. matrix is always forced to be Hermitian in phonopy, i.e.,
.. :math:`D^{\alpha\beta}_{ij} = (D^{\beta\alpha}_{ji})^*`.

.. ``MP_REDUCE``
.. ~~~~~~~~~~~~~~

.. When mesh sampling, time reversal symmetry is imposed by setting
.. ``MP_REDUCE = .TRUE.``. The default value is ``.TRUE.``. If you don't
.. want time reversal symmetry, you have to set as ``MP_REDUCE =
.. .FALSE.``.


.. _force_constants_tag:

Force constants
---------------

``FORCE_CONSTANTS``
~~~~~~~~~~~~~~~~~~~

::

   FORCE_CONSTANTS = READ

There are three values to be set, which are ``READ`` and ``WRITE``,
and ``.FALSE.``. The default is ``.FALSE.``. When ``FORCE_CONSTANTS =
READ``, force constants are read from ``FORCE_CONSTANTS`` file. With
``FORCE_CONSTANTS = WRITE``, force constants calculated from
``FORCE_SETS`` are written to ``FORCE_CONSTANTS`` file.

The file format of ``FORCE_CONSTANTS`` is shown
:ref:`here <file_force_constants>`.



.. _animation_tag:

Create animation file
---------------------

``ANIME_TYPE``
~~~~~~~~~~~~~~~~

::

   ANIME_TYPE = JMOL

There are ``V_SIM``, ``ARC``, ``XYZ``, ``JMOL``, and ``POSCAR``
settings. Those may be viewed by ``v_sim``, ``gdis``, ``jmol``
(animation), ``jmol`` (vibration), respectively. For ``POSCAR``, a set
of ``POSCAR`` format structure files corresponding to respective
animation images are created such as ``APOSCAR-000``,
``APOSCAR-001``,.... 

There are several parameters to be set in the ``ANIME`` tag.

``ANIME``
~~~~~~~~~

**The format of ``ANIME`` tag was modified after ver. 0.9.3.3.**

For v_sim
^^^^^^^^^^

::

   ANIME = 0.5 0.5 0

The values are the *q*-point to be calculated. An animation file of
``anime.ascii`` is generated.

.. toctree::

   animation


For the other animation formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Phonon is only calculated at :math:`\Gamma` point. So *q*-point is not
necessary to be set.

``anime.arc``, ``anime.xyz``, ``anime.xyz_jmol``, or ``APOSCAR-*``
are generated according to the ``ANIME_TYPE`` setting.

::

   ANIME = 4 5 20  0.5 0.5 0

The values are as follows from left:

1. Band index given by ascending order in phonon frequency.

2. Magnitude to be multiplied. In the harmonic phonon calculation,
   there is no amplitude information obtained directly. The relative
   amplitude among atoms in primitive cell can be obtained from
   eigenvectors with the constraint of the norm or the eigenvectors
   equals one, i.e., number of atoms in the primitive is large, the
   displacements become small. Therefore this has to be adjusted to
   make the animation good looking.

3. Number of images in one phonon period.

4. (4-6) Shift of atomic points in reduced coordinate in real space. These
   values can be omitted and the default values are ``0 0 0``.

For ``anime.xyz_jmol``, the first and third values are not used,
however dummy values, e.g. 0, are required.

.. _modulation_tag:

Create modulated structure
----------------------------

``MODULATION``
~~~~~~~~~~~~~~

The ``MODULATION`` tag is used to create a crystal structure with
displacements along normal modes at q-point in the specified supercell
dimension.

Atomic displacement of the *j*-th atom is created from the real part
of the eigenvectors with amplitudes and phase factors as

.. math::

   \frac{A} { \sqrt{N_\mathrm{a}m_j} } \operatorname{Re} \left[ \exp(i\phi)
   \mathbf{e}_j \exp( \mathbf{q} \cdot \mathbf{r}_{jl} ) \right],

where :math:`A` is the amplitude, :math:`\phi` is the phase,
:math:`N_\mathrm{a}` is the number of atoms in the supercell specified
in this tag and :math:`m_j` is the mass of the *j*-th atom,
:math:`\mathbf{q}` is the q-point specified, :math:`\mathbf{r}_{jl}`
is the position of the *j*-th atom in the *l*-th unit cell, and
:math:`\mathbf{e}_j` is the *j*-th atom part of eigenvector. Convention of
eigenvector or dynamical matrix employed in phonopy is shown in
:ref:`dynacmial_matrix_theory`.

If several modes are specified as shown in the example above, they are
overlapped on the structure. The output filenames are
``MPOSCAR...``. Each modulated structure of a normal mode is written
in ``MPOSCAR-<number>`` where the numbers correspond to the order of
specified sets of modulations. ``MPOSCAR`` is the structure where all
the modulations are summed. ``MPOSCAR-orig`` is the structure without
containing modulation, but the dimension is the one that is specified.
Some information is written into ``modulation.yaml``.

Usage
^^^^^^^^^^^^^

The first three (nine) values correspond to supercell dimension
(supercell matrix) like the :ref:`dimension_tag` tag. The following
values are used to describe how the atoms are modulated. Multiple sets
of modulations can be specified by separating by comma ``,``. In each
set, the first three values give a Q-point in the reduced coordinates
in reciprocal space. Then the next three values are the band index
from the bottom with ascending order, amplitude, and phase factor in
degrees. The phase factor is optional. If it is not specified, 0 is
used.

Before multiplying user specified phase factor, the phase of
the modulation vector is adjusted as the largest absolute value,
:math:`\left|\mathbf{e}_j\right|/\sqrt{m_j}`, of element of
3N dimensional modulation vector to be real. The complex modulation
vector is shown in ``modulation.yaml``.

::

   MODULATION = 3 3 1, 1/3 1/3 0 1 2, 1/3 1/3 0 2 3.5

::

   MODULATION = 3 3 1, 1/3 1/3 0 1 2, 1/3 0 0 2 2

::

   MODULATION = 3 3 1, 1/3 1/3 0 1 1 0, 1/3 1/3 0 1 1 90

::

   MODULATION = -1 1 1 1 -1 1 1 1 -1, 1/2 1/2 0 1 2


.. _irreducible_representation_related_tags:

Characters of irreducible representations
------------------------------------------

.. _irreps_tag:

``IRREPS``
~~~~~~~~~~~~~~~~~~~~

Characters of irreducible representations (IRs) of phonon modes are
shown. For this calculation, a primitive cell has to be used. If the
input unit cell is a non-primitive cell, it has to be transformed to a
primitive cell using ``PRIMITIVE_AXIS`` tag.

The first three values gives a *q*-point in reduced coordinates
to be calculated.  The degenerated modes are searched only by the closeness of
frequencies. The frequency difference to be tolerated is specified by
the fourth value in the frequency unit that the user specified.

::

   IRREPS = 0 0 0 1e-3

Only the databases of IRs for a few point group types at the
:math:`\Gamma` point are implemented. If the database is available,
the symbols of the IRs and the rotation operations are shown.


``SHOW_IRREPS``
~~~~~~~~~~~~~~~~

Irreducible representations are shown along with character table.

::

   IRREPS = 1/3 1/3 0
   SHOW_IRREPS = .TRUE.   

``LITTLE_COGROUP``
~~~~~~~~~~~~~~~~~~~
Show irreps of little co-group (point-group of wavevector) instead of
little group.

::

   IRREPS = 0 0 1/8
   LITTLE_COGROUP = .TRUE.


