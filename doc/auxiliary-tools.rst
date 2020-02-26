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


``--pa``, ``--primitive-axes``
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
