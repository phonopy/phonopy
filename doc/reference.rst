References
===========

Method used in phonopy
-----------------------

.. _reference_plk:

Parlinski-Li-Kawazoe method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- \K. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. 78, 4063 (1997)

Parlinski-Li-Kawazoe method is based on the supercell approach with
the finite displacement method. 

Force constants are calculated using Moore–Penrose pseudoinverse by
fitting symmetry reduced elements of force constans to the linear
relations between atomic forces and atomic displacements. The
pseudoinverse is easy to handle arbitrary number of displacements
amplitudes and directions, and can rely on the exisiting library,
e.g., LAPACK.

In phonopy, the symmetry reduced elements of force constants are not
prepared. Simply all the elements of force constans between atoms in
a primitive cell and the supercell are fit. 

.. _reference_NAC:

Non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- \R. M. Pick, M. H. Cohen, and R. M. Martin, Phys. Rev. B 1, 910, (1970)
- \P. Giannozzi, S. Degironcoli, P. Pavone, and S. Baroni,  Phys. Rev. B 43, 7231 (1991)
- \X. Gonze, and C. Lee, Phys. Rev. B 55, 10355 (1997)

.. _reference_wang_NAC:

Interpolation scheme at general *q*-points with non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- \Y Wang , J J Wang , W Y Wang , Z G Mei , S L Shang , L Q Chen and
  Z K Liu, J. Phys.: Condens. Matter. 22, 202201 (2010)

.. - \K. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. 81, 3298 (1998)

Interpolation scheme at getenral *q*-points with non-analytical term
correction is implemented according to Wang *et al* (``--nac``
option).

Other methods for calculating force constants
-----------------------------------------------

.. _reference_small_displacement:

Parlinsk-Li-Kawazoe method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`PHONON <http://wolf.ifj.edu.pl/phonon/>`_ is the original
implementation of the Parlinsk-Li-Kawazoe method.

Small displacement method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Dario Alfè, Computer Physics Communications, 180, 2622 (2009)

`PHON <http://www.homepages.ucl.ac.uk/~ucfbdxa/phon/>`_ is based on the
small displacement method.


.. _reference_dfpt:

DFPT
^^^^^^^^^^^^

- Xavier Gonze and Changyol Lee, Phys. Rev. B 55, 10355 (1997)

Currently there are several many implementations such as `Abinit
<http://www.abinit.org/>`_, `Quantum espresso
<http://www.quantum-espresso.org/>`_, `Elk
<http://elk.sourceforge.net/>`_, etc.  VASP can calculate force constants
using DFPT however only at Gamma-point.

For the study of basics
------------------------

- Introduction to Lattice Dynamics, Martin. T. Dove, Cambridge
  university press
- Thermodynamics of Crystals, Duane C. Wallace, Dover Publications

|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net
