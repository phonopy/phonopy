x.. _command_options:

Command options
===============

Some of command-line options are equivalent to respective setting
tags:

* ``--amplitude`` (``DISPLACEMENT_DISTANCE``)
* ``--anime`` (``ANIME``)
* ``--band`` (``BAND``)
* ``--band_connection``  (``BAND_CONNECTION = .TRUE.``)
* ``--band_labels`` (``BAND_LABELS``)
* ``--band_points``  (``BAND_POINTS``)
* ``--cutoff_freq`` (``CUTOFF_FREQUENCY``)
* ``-c``, ``--cell`` (``CELL_FILENAME``)
* ``-d``  (``CREATE_DISPLACEMENTS = .TRUE.``
* ``--dim`` (``DIM``)
* ``--dos`` (``DOS = .TRUE.``)
* ``--eigvecs``, ``--eigenvectors`` (``EIGENVECTORS = .TRUE.``)
* ``--fits_debye_model`` (``DEBYE_MODEL = .TRUE.``)
* ``--gc``, ``--gamma_center`` (``GAMMA_CENTER``)
* ``--gv``, ``--group_velocity`` (``GROUP_VELOCITY = .TRUE.``)
* ``--gv_delta_q`` (``GV_DELTA_Q``)
* ``--irreps`` (``IRREPS``)
* ``--lcg``, ``--little_cogroup`` (``LITTLE_COGROUP``)
* ``--modulation`` (``MODULATION``)
* ``--mp``, ``--mesh`` (``MP``)
* ``--nac`` (``NAC = .TRUE.``)
* ``--nosym`` (``SYMMETRY = .FALSE.``)
* ``--nomeshsym`` (``MESH_SYMMETRY = .FALSE.``)
* ``--nowritemesh`` (``WRITE_MESH = .FALSE.``)
* ``--pa``, ``--primitive_axis`` (``PRIMITIVE_AXIS``)
* ``--pd``, ``--projection_direction`` (``PROJECTION_DIRECTION``)
* ``--pdos`` (``PDOS``)
* ``--q_direction`` (``Q_DIRECTION``)
* ``--readfc`` (``FORCE_CONSTANTS = READ``)
* ``--show_irreps`` (``SHOW_IRREPS``)
* ``--sigma`` (``SIGMA``)
* ``-t`` (``TPROP``)
* ``--td`` (``TDISP``)
* ``--tdm`` (``TDISPMAT``)
* ``--thm``, ``--tetrahedron_method`` (``TETRAHEDRON``)
* ``--tmin`` (``TMIN``)
* ``--tmax`` (``TMAX``)
* ``--tstep`` (``TSTEP``)
* ``--writedm`` (``WRITEDM = .TRUE.``)
* ``--writefc`` (``FORCE_CONSTANTS = WRITE``)

When both of command-line option and setting tag for the same purpose
are set simultaneously, the command-line options overide the setting
tags.

.. _force_calculators:

Force calculators
------------------

Currently interfaces for VASP, Wien2k, Pwscf, Abinit, and Elk are
prepared. Wien2k, Pwscf, Abinit and Elk interfaces are invoked with
``--wienk2``, ``--pwscf``, ``--abinit``, and ``--elk`` options,
respectively, and if none of these options is specified, VASP mode is invoked.

The details about these interfaces are found at :ref:`calculator_interfaces`.

.. _wien2k_mode:

``--wien2k``
~~~~~~~~~~~~

**Behavior is changed at phonopy 1.9.2.**

This option invokes the WIEN2k mode.In this mode. Usually this option
is used with ``--cell`` (``-c``) option or ``CELL_FILENAME`` tag to
read WIEN2k crystal structure file.

::

   % phonopy --wien2k -c NaCl.struct band.conf

**Only the WIEN2k struct with the P lattice is supported**.  See more
information :ref:`wien2k_interface`.

For previous versions than 1.9.1.3, this option is used as

::

   % phonopy --wien2k=NaCl.struct band.conf   (version <= 1.9.1.3)
   

.. _abinit_mode:

``--abinit``
~~~~~~~~~~~~

Abinit mode is invoked with this option. Usually this option is used
with ``--cell`` (``-c``) option or ``CELL_FILENAME`` tag to read
Abinit main input file that contains the unit cell crystal structure,
e.g.,

::

   % phonopy --abinit -c NaCl.in band.conf

.. _pwscf_mode:

``--pwscf``
~~~~~~~~~~~~

Pwscf mode is invoked with this option. Usually this option is used
with ``--cell`` (``-c``) option or ``CELL_FILENAME`` tag to read Pwscf
input file that contains the unit cell crystal structure, e.g.,

::

   % phonopy --pwscf -c NaCl.in band.conf

.. _siesta_mode:

``--siesta``
~~~~~~~~~~~~

Siesta mode is invoked with this option. Usually this option is used
with ``--cell`` (``-c``) option or ``CELL_FILENAME`` tag to read a Siesta
input file that contains the unit cell crystal structure, e.g.,

::

   % phonopy --siesta -c Si.fdf band.conf

.. _elk_mode:

``--elk``
~~~~~~~~~~~~

Pwscf mode is invoked with this option. Usually this option is used
with ``--cell`` (``-c``) option or ``CELL_FILENAME`` tag to read Elk
input file that contains the unit cell crystal structure, e.g.,

::

   % phonopy --elk -c elk-unitcell.in band.conf

  
.. _cell_filename_option:

Input cell
----------

``-c`` or ``--cell``
~~~~~~~~~~~~~~~~~~~~

Unit cell crystal structure file is specified with this tag.

::

   % phonopy --cell=POSCAR-unitcell band.conf

Without specifying this tag, default file name is searched in current
directory. The default file names for the calculators are as follows::

   VASP   | POSCAR     
   Wien2k | case.struct
   Abinit | unitcell.in
   Pwscf  | unitcell.in
   Elk    | elk.in

Create ``FORCE_SETS``
----------------------

``-f`` or ``--forces``
~~~~~~~~~~~~~~~~~~~~~~

.. _vasp_force_sets_option:

VASP interface
^^^^^^^^^^^^^^

``FORCE_SETS`` file is created from ``disp.yaml``, which is an output
file when creating supercell with displacements, and
``vasprun.xml``'s, which are the VASP output files. ``disp.yaml`` in
the current directory is automatically read. The order of
displacements written in ``disp.yaml`` file has to correpond to that of
``vasprun.xml`` files .

::

   % phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml ...

Attention:

* Site-projected wave function information (the same information as
  ``PROCAR``) siginificantly increases the size of ``vasprun.xml``. So
  parsing xml file uses huge memory space. It is recommended
* to switch off to calculate it.  If there are many displacements, shell
  expansions are useful, e.g., ``disp-*/vasprun.xml``, or
  ``disp-{001..128}/vasprun.xml`` (for zsh, and recent bash).

..
   ``--fz`` option is used to subtract residual forces in the equilibrium
   supercell.

   ::

      % phonopy --fz sposcar/vasprun.xml disp-001/vasprun.xml ...

   Usually the ``-f`` option is preferable to ``--fz``.

.. _abinit_force_sets_option:

Abinit interface
^^^^^^^^^^^^^^^^

``FORCE_SETS`` file is created from ``disp.yaml`` and Abinit output
files (``*.out``). In the reading of forces in Abinit output files,
forces in eV/Angstrom are read. The unit conversion factor is
determined with this unit.

::

   % phonopy --abinit -f disp-001/supercell.out disp-002/supercell.out  ...


.. _pwscf_force_sets_option:

Pwscf interface
^^^^^^^^^^^^^^^^

``FORCE_SETS`` file is created from ``disp.yaml`` and Pwscf output
files.

::

   % phonopy --pwscf -f disp-001/supercell.out disp-002/supercell.out  ...

Here ``*.out`` files are the saved texts of standard outputs of Pwscf calculations.
   
.. _wien2k_force_sets_option:

WIEN2k interface
^^^^^^^^^^^^^^^^

This is experimental support to generage ``FORCE_SETS``. Insted of
this, you can use the external tool called ``scf2forces`` to generate
``FORCE_SETS``. ``scf2forces`` is found at
http://www.wien2k.at/reg_user/unsupported/.


``FORCE_SETS`` file is created from ``disp.yaml``, which is an output
file when creating supercell with displacements, and
``case.scf``'s, which are the WIEN2k output files. The order of
displacements in ``disp.yaml`` file and the order of ``case.scf``'s
have to be same. **For Wien2k struct file, only negative atom index
with the P lattice format is supported.**

::

   % phonopy --wien2k -f case_001/case_001.scf case_002/case_002.scf ...

For more information, :ref:`wien2k_interface`.

.. _elk_force_sets_option:

Elk interface
^^^^^^^^^^^^^^^^

``FORCE_SETS`` file is created from ``disp.yaml`` and Elk output
files.

::

   % phonopy --elk -f disp-001/INFO.OUT disp-002/INFO.OUT  ...


Create ``FORCE_CONSTANTS``
--------------------------

.. _vasp_force_constants:

``--fc`` or ``--force_constants``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Currently this option supports only VASP output.**

VASP output of force constants is imported from
``vasprun.xml`` and ``FORCE_CONSTANTS`` is created.

::

   % phonopy --fc vasprun.xml

This ``FORCE_CONSTANTS`` can be used instead of ``FORCE_SETS``. For
more details, please refer :ref:`vasp_dfpt_interface`.

.. _graph_option:

Graph plotting
---------------

``-p``
~~~~~~

Result is plotted.

::

   % phonopy -p

.. _graph_save_option:

``-p -s``
~~~~~~~~~

Result is plotted (saved) to PDF file.

::

   % phonopy -p -s


Unit conversion factor
----------------------

.. _unit_conversion_factor_option:

``--factor``
~~~~~~~~~~~~

Unit conversion factor of frequency from input values to your favorite
unit is specified. The default value is that to convert to THz. The
default conversion factors for ``wien2k``, ``abinit``, ``pwscf``, and
``elk`` are 3.44595, 21.49068, 108.9708, and 154.1079
respectively. These are determined following the physical unit systems
of the calculators. How to calcualte these conversion factors is
explained at :ref:`physical_unit_conversion`.

When calculating thermal property, the factor to THz is
required. Otherwise the calculated thermal properties have wrong
units. In the case of band structure plot, any factor can be used,
where the frequency is simply shown in the unit you specified.

::

   % phonopy --factor=521.471



Log level
----------

``-v`` or ``--verbose``
~~~~~~~~~~~~~~~~~~~~~~~

More detailed log are shown

``-q`` or ``--quiet``
~~~~~~~~~~~~~~~~~~~~~

No log is shown.

Crystal symmetry
-----------------

``--tolerance``
~~~~~~~~~~~~~~~

The specified value is used as allowed tolerance to find symmetry of
crystal structure. The default value is 1e-5.

::

   % phonopy --tolerance=1e-3

``--symmetry``
~~~~~~~~~~~~~~

Using this option, various crystal symmetry information is just
printed out and phonopy stops without going to phonon analysis.

::

   % phonopy --symmetry

This tag can be used together with the ``--cell`` (``-c``),
``--abinit``, ``--pwscf``, ``--elk``, ``--wien2k``, or
``--primitive_axis`` option.

Input/Output file control
-------------------------

.. _hdf5_option:

``--hdf5``
~~~~~~~~~~~

The following input/output files are read/written in hdf5 format
instead of their original formats (in parenthesis).

* ``force_constants.hdf5`` (``FORCE_CONSTANTS``)
* ``mesh.hdf5`` (``mesh.yaml``)

``force_constants.hdf5``
^^^^^^^^^^^^^^^^^^^^^^^^^

With ``--hdf5`` option and ``FORCE_CONSTANTS = WRITE``
(``--writefc``), ``force_constants.hdf5`` is written.
With ``--hdf5`` option and ``FORCE_CONSTANTS = READ`` (``--readfc``),
``force_constants.hdf5`` is read.

``mesh.hdf5``
^^^^^^^^^^^^^^

In the mesh sampling calculations (see :ref:`mesh_sampling_tags`),
calculation results are written into ``mesh.hdf5`` but not into
``mesh.yaml``. Using this option may reduce the data output size and
thus writing time when ``mesh.yaml`` is huge, e.g., eigenvectors are
written on a dense sampling mesh.
