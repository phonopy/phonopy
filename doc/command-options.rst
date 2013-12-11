.. _command_options:

Command options
===============

Some of command-line options are equivalent to respective setting
tags:

* ``--amplitude`` (``DISPLACEMENT_DISTANCE``)
* ``--anime`` (``ANIME``)
* ``-d``  (``CREATE_DISPLACEMENTS = .TRUE.``
* ``--dim`` (``DIM``)
* ``--mp``, ``--mesh`` (``MP``)
* ``--band`` (``BAND``)
* ``--band_points``  (``BAND_POINTS``)
* ``--band_connection``  (``BAND_CONNECTION = .TRUE.``)
* ``--cutoff_freq`` (``CUTOFF_FREQUENCY``)
* ``--eigvecs``, ``--eigenvectors`` (``EIGENVECTORS = .TRUE.``)
* ``--fits_debye_model`` (``DEBYE_MODEL = .TRUE.``)
* ``--gc``, ``--gamma_center`` (``GAMMA_CENTER``)
* ``--gv``, ``--group_velocity`` (``GROUP_VELOCITY = .TRUE.``)
* ``--gv_delta_q`` (``GV_DELTA_Q``)
* ``--irreps`` (``IRREPS``)
* ``--show_irreps`` (``SHOW_IRREPS``)
* ``--modulation`` (``MODULATION``)
* ``--nac`` (``NAC = .TRUE.``)
* ``--nosym`` (``SYMMETRY = .FALSE.``)
* ``--nomeshsym`` (``MESH_SYMMETRY = .FALSE.``)
* ``--pa``, ``--primitive_axis`` (``PRIMITIVE_AXIS``)
* ``--pd``, ``--projection_direction`` (``PROJECTION_DIRECTION``)
* ``--pdos`` (``PDOS``)
* ``--readfc`` (``FORCE_CONSTANTS = READ``)
* ``--sigma`` (``SIGMA``)
* ``-t`` (``TPROP``)
* ``--td`` (``TDISP``)
* ``--tdm`` (``TDISPMAT``)
* ``--tmin`` (``TMIN``)
* ``--tmax`` (``TMAX``)
* ``--tstep`` (``TSTEP``)
* ``--writedm`` (``WRITEDM = .TRUE.``)
* ``--writefc`` (``FORCE_CONSTANTS = WRITE``)

When both of command-line option and setting tag for the same purpose
are set simultaneously, the command-line options overide the setting
tags.

Help (``-h`` or ``--help``)
---------------------------

Review of options is shown.

Create ``FORCE_SETS``
----------------------

``-f`` or ``--forces`` and ``--fz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

``--fz`` option is used to subtract residual forces in the equilibrium
supercell.

::

   % phonopy --fz sposcar/vasprun.xml disp-001/vasprun.xml ...

Usually the ``-f`` option is preferable to ``--fz``.

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

   % phonopy --wien2k=case.struct -f case_001/case_001.scf case_002/case_002.scf ...

For more information, :ref:`wien2k_interface`.


.. Though the ``--fz`` option is supported as well as the VASP interface,
.. usually the ``-f`` option is preferable to ``--fz``.

.. ::

..    % phonopy --wien2k=case.struct --fz case_000/case_000.scf case_001/case_001.scf ...



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

.. ``--fco``
.. ~~~~~~~~~~

.. Force constants are read from VASP ``OUTCAR`` file, instead of
.. ``vasprun.xml``. This option can be used as well as ``--fc`` tag.
.. ``--fc`` is recommended than ``--fco`` because ``vasprun.xml`` has
.. more digits than ``OUTCAR``.

.. ::

..    % phonopy --fco OUTCAR


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


Calculate DOS
-------------

``--dos``
~~~~~~~~~

Density of states are calculated using this option with ``MP``
tag. When ``-p`` option with ``MP`` tag is set, ``--dos`` is
automatically set. Therefore this tag is used when you want to
calculate DOS, but you don't need to plot.

Unit conversion factor
----------------------

.. _unit_conversion_factor_option:

``--factor``
~~~~~~~~~~~~

Unit conversion factor of frequency from input values to your favorite
unit is specified. The default value is that of VASP to THz, which is
calculated by
:math:`\sqrt{\text{eV/AMU}}`/(:math:`\text{\AA}\cdot2\pi\cdot10^{12}`)
(=15.633302) in SI base unit. When the ``wien2k`` option is specified,
the default value is changed to 3.44595, which is the factor to
convert from Wien2k to THz.

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

This tag can be used together with the ``--cell``, ``--wien2k``, or
``--primitive_axis`` option.


Input cell
----------

``-c`` or ``--cell``
~~~~~~~~~~~~~~~~~~~~
 
Phonopy searches the ``POSCAR`` file in the current directory. Using
this tag, you can specify another filename than ``POSCAR`` as the
input unit cell.

::

   % phonopy --cell=UPOSCAR

.. _wien2k_mode:

``--wien2k``
~~~~~~~~~~~~

This option with WIEN2k struct file, phonopy runs with the WIEN2k
mode. In this mode, you don't need to prepare ``POSCAR``. The
supercells with displacements in WIEN2k struct format are created
using ``-d`` option.  The physical
unit is changed to mRydberg and Bohr. **Only the WIEN2k struct with
the P lattice is supported**.  See more information
:ref:`wien2k_interface`.

::

   % phonopy --wien2k=case.struct


.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|
