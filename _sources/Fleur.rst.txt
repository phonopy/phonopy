.. _Fleur_interface:

Fleur & phonopy calculation
=========================================

Setting up the inpgen file:

The preparation of a Fleur calculation takes two steps. First a basic input file
is written, that is converted to an inp.xml via the Fleur input generator inpgen.

The input file starts with a comment line describing the calculation. From there,
a multitude of options can be specified by namelists starting with '&' and ending
with '/'. Normally, the lattice information would be set preferably by tags like
'fcc' etc. and the corresponding necessary lattice constants. For further
information, consult the inpgen documentation under 'User Guide --> Reference -->
The input generator' on flapw.de.

An important thing to note is:
The bravais matrix needs to be set directly via the second method proposed in
the reference. The complete overhead to constuct the bravais matrix from only
the lattice tag and constants in not (yet) programmed into the phonopy interface.
The supercell will be written in the same fashion, while the rest of the inpgen
file is merely copied over.

Also:
For phonopy to be able to correctly extract the necessary quantities from the
input file, a mandatory comment '! a1' needs to be put at the end of the first
line of the Bravais matrix and '! num atoms' after the atom count.

How to run
----------

A procedure of a Fleur-phonopy calculation may look as follows:

1) Read a Fleur input file and create supercells with the
   :ref:`Fleur_mode` option::

   % phonopy --fleur -d --dim="a b c" -c fleur_inpgen

   In this example several axbxc supercells are created. ``supercell.in``
   and ``supercell-XXX.in`` (``XXX`` enumerating the necessary displacements) give
   the perfect and displaced supercells respectively. ``phonopy_disp.yaml``
   is also generated. This file contains information on displacements.
   The supercell files ``supercell-XXX.in`` should be stored in
   subfolders e.g. ``disp-XXX``, then Fleur calculations are executed
   in these directories.

2) Calculate forces on atoms in the supercells with displacements.
   First run the fleur inpgen with the '-f supercell-XXX.in' option
   The resulting inp.xml is then run by the fleur command itself.
   After successfull convergence, set the tag 'l_f' in the inp.xml to "T"
   and add 'f_level="n"' behind it with n from {0,1,2,3}. This optional
   tag ensures the write-out of the 'FORCES' file and setting it to
   {1,2,3} calculates additional refined force contibutions.

3) Create ``FORCE_SETS`` by::

     % phonopy --fleur -f disp-001/FORCES disp-002/FORCES ...

   To run this command, ``phonopy_disp.yaml`` has to be located in same
   directory because the atomic displacements need to be written into the
   FORCE_SETS file. See some more detail at
   :ref:`Fleur_force_sets_option`. An example for fcc Al with a 2x2x2
   supercell and only one necessary displacement is found in
   ``example/Al-Fleur``.

4) Run the post-process of phonopy with the Fleur input file for the
   unit cell used in step 1::

   % phonopy --fleur -c fleur_inpgen -p band.conf

   if you prepared a band.conf file or::

   % phonopy --fleur -c fleur_inpgen --dim="2 2 2" [other-OPTIONS] [setting-file]

   if you want to set the path directly or specify a different file.
