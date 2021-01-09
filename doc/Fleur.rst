.. _Fleur_interface:

Fleur & phonopy calculation
=========================================

Setting up the inpgen file:

The preparation of a Fleur calculation takes two steps. First a basic input file
is written, that in converted to an inp.xml via the input generator inpgen.

The input file starts with a comment line describing the calculation. From there,
a multitude of options can be specified by namelists starting with '&' and ending
with '/'. Normally, the lattice information would be set preferably by tags like
'fcc' etc. and the corresponding necessary lattice constants. For further
information, consult the inpgen documentation under User Guide --> Reference -->
the input generator on flapw.de.

An important thing to note is:
The bravais matrix needs to be set directly via the second method proposed in
the reference. The complete overhead to constuct the bravais matrix from only
the lattice tag and constants in not (yet) programmed into the interface. The
supercell will be written in the same fashion, while the rest of the inpgen
file is merely copied over.

Also:
For phonopy to be able to correctly extract the necessary quantities from the
input file, a mandatory comment '! a1' needs to be put at the end of the first
line of the Bravais matrix and '! num atoms' after the atom count.

[TODO: specifying an fcc Bravais matrix directly instead of setting a simple
cubic cell with a 4 atom basis yielded construction errors and other lattices
are untested. Directly setting an fcc matrix in Elk failed identically.]

How to run
----------

A procedure of a Fleur-phonopy calculation is as follows:

1) Read a Fleur input file and create supercells with
   :ref:`Fleur_mode` option::

   % phonopy --fleur -d --dim="2 2 2" -c fleur_inpgen

   In this example a 2x2x2 supercells are created. ``supercell.in`` and
   ``supercell-001.in`` (``001`` for only one displacement) give the
   perfect and displaced supercell respectively. ``phonopy_disp.yaml``
   is also created. This file contains information on displacements.
   The supercell file of ``supercell-001.in`` should be stored in
   a subfolder e.g. ``disp-001``, then Fleur calculations are executed
   in this directory.

2) Calculate forces on atoms in the supercell with displacements.
   First run the fleur inpgen with the '-f fleur_inpgen' option
   (or any name you chose to give the file). The resulting inp.xml
   is then run by the fleur command itself. After successfull convergence,
   set the tag 'l_f' in the inp.xml to T and add 'f_level="n"' behind it
   with n from {0,1,2,3}. This optional tag ensures the write-out of the
   'FORCES' file and setting it to {1,2,3} calculated additional refined
   force contibutions.
   
   TODO: At the moment, for the force parser to work, an additional ' force'
   needs to be added after every force line. This will be done automatically
   by Fleur soon.

3) Create ``FORCE_SETS`` by

   ::

     % phonopy --fleur -f disp-001/FORCES

   To run this command, ``phonopy_disp.yaml`` has to be located in same
   directory because the atomic displacements are written into the
   FORCE_SETS file. See some more detail at
   :ref:`Fleur_force_sets_option`. An example is found in
   ``example/Al-Fleur``.

4) Run post-process of phonopy with the Elk input file for the
   unit cell used in the step 1::

   % phonopy --Fleur -c fleur_inpgen -p band.conf

   or::

   % phonopy --Fleur -c fleur_inpgen --dim="2 2 2" [other-OPTIONS] [setting-file]
