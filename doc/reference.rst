References
===========

Method used in phonopy
-----------------------

.. _reference_plk:

Parlinski-Li-Kawazoe method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- \K. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. 78, 4063 (1997)

Parlinski-Li-Kawazoe method is based on the supercell approach with
the finite displacement method. The calculation and symmetrization of
force constants are executed by using singular-value decomposition
(pseudo-inverse). The key of this method would be the matrix
formulations of equations, which leads to the coherent and flexible
implementation.

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

Small displacement method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Dario Alfè, Computer Physics Communications, 180, 2622 (2009)

PHON is based on the small displacement method.

.. _reference_dfpt:

DFPT
^^^^^^^^^^^^

- Xavier Gonze and Changyol Lee, Phys. Rev. B 55, 10355 (1997)

The most famous implementation is Abinit. Currently there are many
implementations of DFPT. VASP can calculate force constants using DFPT
however only at Gamma-point.


For the study of basics
------------------------

- Introduction to Lattice Dynamics, Martin. T. Dove, Cambridge
  university press
- Thermodynamics of Crystals, Duane C. Wallace, Dover Publications

|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net
