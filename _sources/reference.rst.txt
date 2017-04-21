References
===========

Method used in phonopy
-----------------------

.. _reference_force_constants:

Generation of force constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In phonopy, force constants are generated based on finite displacement
method. Crystal symmetry is used to reduce the calculation cost and
numerical noise of the force constants. Firstly a symmetry reduced set
of atomic displacements is generated. After the atomic force
calculations, the set of atomic displacements are expanded using the
symmetry and then all the elements of force constans between atoms in
a primitive cell and the supercell are fit to the symmetry expanded
forces of atoms in supercells using Moore–Penrose pseudoinverse. This
procedure may considered as a variant of :ref:`reference_plk`. Some of
the details are found in the appendix of the following paper:

- \L. Chaput, A. Togo, I. Tanaka, and G. Hug, Phys. Rev. B, 84,
  094302 (2011)

.. _reference_plk:

Parlinski-Li-Kawazoe method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parlinski-Li-Kawazoe method is based on the supercell approach with
the finite displacement method. 

Force constants are calculated using Moore–Penrose pseudoinverse by
fitting symmetry reduced elements of force constans to the linear
relations between atomic forces and atomic displacements. The
pseudoinverse is easy to handle arbitrary number of displacements
amplitudes and directions, and can rely on the exisiting library,
e.g., LAPACK.

- \K. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. 78, 4063 (1997)


.. _reference_thermal_expansion:

Thermal expansion using quasi-harmonic approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In :ref:`phonopy-qha <phonopy_qha>`, thermal properties at constant
pressure is obtained from the thermodynamic definition.  To achieve
Legendre transformation, volume-energy function is generated from a
set of Helmholtz free energies and *pV* terms at volumes by fitting to
a smooth function for which equations of states are prepared in phonopy-qha.

The volume dependence of the Helmholtz free energy is included from
quasi-harmonicity. When using DFT-GGA (-LDA), often we should have
some amount of error in the absolute value since phonon frequencies
are underestimated (overestimated). However the value of some ratio
like thermal expansion coefficient is often very well estimated. An
example is shown in the following paper:

- \A. Togo, L. Chaput, I. Tanaka, G. Hug, Phys. Rev. B, 81, 174301-1-6 (2010)

.. _reference_NAC:

Non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-metallic crystals are polarized by atomic displacements and the
generated macroscopic field changes force constants near
:math:`\Gamma` point. This contribution is included through
non-analytical term correction.

- \R. M. Pick, M. H. Cohen, and R. M. Martin, Phys. Rev. B 1, 910, (1970)
- \P. Giannozzi, S. Degironcoli, P. Pavone, and S. Baroni,  Phys. Rev. B 43, 7231 (1991)
- \X. Gonze, and C. Lee, Phys. Rev. B 55, 10355 (1997)

.. _reference_wang_NAC:

Interpolation scheme at general *q*-points with non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above non-anarytical term correction can be applied only at
:math:`\mathbf{q}\rightarrow \mathbf{0}`. To connect smoothly to
general **q** points, the following interpolation scheme is employed
in phonopy (:ref:`nac_tag`).

- \Y. Wang , J. J. Wang , W. Y. Wang , Z. G. Mei , S. L. Shang , L. Q. Chen and
  Z K Liu, J. Phys.: Condens. Matter. 22, 202201 (2010)

The first derivative of this expression, which is for example used for
group velocity calclation, is described in the following paper:

- Atsushi Togo, Laurent Chaput, and Isao Tanaka, Phys. Rev. B, 91, 094306-1-31 (2015)

Other methods and software for calculating force constants
-----------------------------------------------------------

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

