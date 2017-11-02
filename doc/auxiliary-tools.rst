.. _auxiliary_tools:

Auxiliary tools
===============

A few auxiliary tools are prepared. They are stored in ``bin``
directory as well as ``phonopy``.

.. contents::
   :depth: 3
   :local:

.. _bandplot_tool:

``phonopy-bandplot``
---------------------

Band structure is plotted reading phonopy output in ``band.yaml``
format. ``-o`` option with a file name is used to save the plot into a
file in PDF format. A few more options are prepared and shown by
``-h`` option. If you specify more than two yaml files, they are
plotted together.

::

   phonopy-bandplot band.yaml

To obtain a simple text format data::

   phonopy-bandplot --gnuplot band.yaml

.. _pdosplot_tool:

``phonopy-pdosplot``
---------------------

Partial density of states (PDOS) are plotted.

``-i`` option is used as

::

   phonopy-pdosplot -i '1 2 4 5, 3 6' -o 'pdos.pdf' partial_dos.dat

The indices and comma in '1 2 3 4, 5 6' mean as follows. The indices
are separated into blocks by comma (1 2 4 5 and 3 6). PDOS specified
by the successive indices separated by space in each block are summed
up. The PDOS of blocks are drawn simultaneously. Indices usually
correspond to atoms.  A few more options are prepared and shown by
``-h`` option.

.. _propplot_tool:

``phonopy-propplot``
---------------------

Thermal properties are plotted. Options are prepared and shown by
``-h`` option. If you specify more than two yaml files, they are
plotted together.

::

   phonopy-proplot thermal_properties_A.yaml thermal_properties_B.yaml

.. ``tdplot``
.. ------------

.. Mean square displacements are plotted. Options are prepared and shown by
.. ``-h`` option. ``-i`` option may be important, which works such like
.. that of pdosplot.

.. ::

..    tdplot -i '1 2 4 5, 3 6' -o 'td.pdf' thermal_displacements.yaml

.. _dispmanager_tool:

``phonopy-dispmanager``
-------------------------

This is used for two purposes.

The first argument is the displacement file (``disp.yaml`` type). The
default file name is ``disp.yaml``.

``-a``, ``--amplitude``, ``-s``, ``-o``, ``--overwite``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``-o`` is used to specify the output file name of the new displacement
file and ``--overwrite`` is used to overwrite the displacement file.

``-a`` is specified with an atom index and a direction of displacement
as a character string. The first value is the atom index and remaining
three values are for direction. ``--amplitude`` is used with ``-a``
and specify the displacement amplitude. An example is as follows:

::

   phonopy-dispmanager disp.yaml -o disp-new.yaml -a "33 1 1 0" --amplitude 0.05

``disp-new.yaml`` is created from ``disp.yaml`` with a new
displacement of the thirty-third atom (index 33) with the direction of
(1,1,0) with the amplitude of 0.05. The direction is defined against
lattice vectors. The amplitude unit is same as the lattice vectors.

``-s`` is specified with displacement indices. For example when there
are four dependent displacements and only the first and third
displacements are needed, ``phonopy-dispmanager`` is used like

::

   phonopy-dispmanager disp.yaml -o disp-new.yaml -s "1 3"

``-w``
^^^^^^^

The option is used to create supercells with displacements in
``POSCAR`` format from a displacement file. ``DPOSCAR-xxx`` files are
created.

``--compatibility``
^^^^^^^^^^^^^^^^^^^^

The old style displacement file ``DISP`` is compared with
``disp.yaml`` whether the directions of the displacements are
equivalent or not.


``phonopy-vasp-born``
----------------------

This script is used to create a ``BORN`` style file from VASP output
file of ``vasprun.xml``.  The first argument is a ``vasprun.xml``
file.  If it is ommited, ``vasprun.xml`` at current directory are
read. The Born effective charges and dielectric tensor are symmetrized
as default. To prevent symmetrization, ``--nost`` option has to be
specified.

::

   phonopy-vasp-born

::

   phonopy-vasp-born --nost


``--pa``, ``--primitive_axis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is same as :ref:`primitive_axis_tag`.

``--dim``
^^^^^^^^^^

This is same as :ref:`dimension_tag`.

``--nost``
^^^^^^^^^^^

Dielectric constant and Born effective charge tensors are not
symmetrized.

``--outcar``
^^^^^^^^^^^^^^^^^

Read ``OUTCAR`` instead of ``vasprun.xml``. Without specifying
arguments, ``OUTCAR`` and ``POSCAR`` at current directory are
read. ``POSCAR`` information is necessary in contrast to reading
``vasprun.xml`` where the unit cell structure is also read from it.

::

   phonopy-vasp-born --outcar

::

   phonopy-vasp-born --nost --outcar OUTCAR POSCAR
