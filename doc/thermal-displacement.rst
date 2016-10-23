.. _thermal_displacement:

Mean square displacement
--------------------------

From Eq. (10.71) in the book "Thermodynamics of Crystal", atomic
displacement, **u**, is written by

.. math::

   u^\alpha(jl,t) = \left(\frac{\hbar}{2Nm_j}\right)^{\frac{1}{2}}
   \sum_{\mathbf{q},\nu}\left[\omega_\nu(\mathbf{q})\right]^{-\frac{1}{2}}
   \left[\hat{a}_\nu(\mathbf{q})\exp(-i\omega_\nu(\mathbf{q})t)+
   \hat{a}^\dagger_\nu(\mathbf{-q})\exp({i\omega_\nu(\mathbf{q})}t)\right]
   \exp({i\mathbf{q}\cdot\mathbf{r}(jl)})
   e^\alpha_\nu(j,\mathbf{q})

where *j* and *l* are the labels for the *j*-th atomic position in the
*l*-th unit cell, *t* is the time, :math:`\alpha` is an axis (a
Cartesian axis in the default behavior of phonopy), *m* is the atomic
mass, *N* is the number of the unit cells, :math:`\mathbf{q}` is the
wave vector, :math:`\nu` is the index of phonon mode. *e* is the
polarization vector of the atom *jl* and the band :math:`\nu` at
:math:`\mathbf{q}`. :math:`\mathbf{r}(jl)` is the atomic position and
:math:`\omega` is the phonon frequency. :math:`\hat{a}^\dagger` and
:math:`\hat{a}` are the creation and annihilation operators of
phonon. The expectation value of the squared atomic displacement is
calculated as,

.. math::

   \left\langle |u^\alpha(jl, t)|^2 \right\rangle = \frac{\hbar}{2Nm_j}
   \sum_{\mathbf{q},\nu}\omega_\nu(\mathbf{q})^{-1}
   (1+2n_\nu(\mathbf{q},T))|e^\alpha_\nu(j,\mathbf{q})|^2,

where :math:`n_\nu(\mathbf{q},T)` is the phonon population, which is give by,

.. math::

   n_\nu(\mathbf{q},T) =
   \frac{1}{\exp(\hbar\omega_\nu(\mathbf{q})/\mathrm{k_B}T)-1},

where *T* is the temperature, and :math:`\mathrm{k_B}` is the
Boltzmann constant. The equation is calculated using the commutation
relation of the creation and annihilation operators and the 
expectation values of the combination of the operations, e.g.,

.. math::

   [ \hat{a}_\nu(\mathbf{q}), \hat{a}^\dagger_{\nu'}(\mathbf{q'}) ]
   = \delta(\mathbf{q}-\mathbf{q}')\delta_{\nu\nu'},

   [ \hat{a}_\nu(\mathbf{q}), \hat{a}_{\nu'}(\mathbf{q'}) ] = 0,

   [ \hat{a}^\dagger_\nu(\mathbf{q}), \hat{a}^\dagger_{\nu'}(\mathbf{q'}) ] = 0,

   \langle|\hat{a}_\nu(\mathbf{q})\hat{a}_{\nu'}(\mathbf{q'})|\rangle
   = 0,

   \langle|\hat{a}^\dagger_\nu(\mathbf{q})\hat{a}^\dagger_{\nu'}(\mathbf{q'})|\rangle
   = 0.

.. _thermal_displacement_matrix:

Mean square displacement matrix
--------------------------------

Mean square displacement matrix is defined as follows:

.. math::

   \mathrm{U}_\text{cart}(j, T) = \frac{\hbar}{2Nm_j}
   \sum_{\mathbf{q},\nu}\omega_\nu(\mathbf{q},T)^{-1}
   (1+2n_\nu(\mathbf{q}))
   \mathbf{e}_\nu(j,\mathbf{q}) \otimes \mathbf{e}^*_\nu(j,\mathbf{q}).

This is a symmetry matrix and diagonal elements are same as mean
square displacement calculated along Cartesian x, y, z directions.

Projection to an arbitrary axis from the Cartesian axes
--------------------------------------------------------

In phonopy, eigenvectors are calculated in the Cartesian axes that are
defined in the input structure file. Mean square displacement along an
arbitrary axis is obtained projecting eigenvectors in the Cartesian
axes as follows:

.. math::

   \left\langle |u(jl, t)|^2 \right\rangle = \frac{\hbar}{2Nm_j}
   \sum_{\mathbf{q},\nu}\omega_\nu(\mathbf{q})^{-1}
   (1+2n_\nu(\mathbf{q}))|
   \hat{\mathbf{n}}\cdot\mathbf{e}_\nu(j,\mathbf{q})|^2

where :math:`\hat{\mathbf{n}}` is an arbitrary unit direction.

Mean square displacement matrix in cif format
----------------------------------------------

According to the paper by Grosse-Kunstleve and Adams [J. Appl. Cryst.,
35, 477-480 (2002)], mean square displacement matrix in the cif
definition (``aniso_U``),
:math:`\mathrm{U}_\text{cif}`, is obtained by

.. math::

   \mathrm{U}_\text{cif} = (\mathrm{AN})^{-1}\mathrm{U}_\text{cart}
   (\mathrm{AN})^{-\mathrm{T}},

where :math:`\mathrm{A}` is the matrix to transform a point in fractional
coordinates to the Cartesian coordinates and :math:`\mathrm{N}` is the
diagonal matrix made of reciprocal basis vector lengths as follows:

.. math::

   \mathrm{A} = \begin{pmatrix}
   a_x & b_x & c_x \\
   a_y & b_y & c_y \\
   a_z & b_z & c_z
   \end{pmatrix}

and

.. math::

   \mathrm{N} = \begin{pmatrix}
   a^* & 0 & 0 \\
   0 & b^* & 0 \\
   0 & 0 & c^*
   \end{pmatrix}.

:math:`a^*`, :math:`b^*`, :math:`c^*` are defined without :math:`2\pi`.





