.. _formulations:

==============
 Formulations
==============

.. contents::
   :depth: 2
   :local:

Second-order force constants
============================

Potential energy of phonon system is represented as functions of atomic
positions:

.. math::

    V[\mathbf{r}(j_1 l_1),\ldots,\mathbf{r}(j_n l_N)],

where :math:`\mathbf{r}(jl)` is the point of the :math:`j`-th atom in
the :math:`l`-th unit cell and :math:`n` and :math:`N` are the number
of atoms in a unit cell and the number of unit cells, respectively.  A
force and a second-order force constant :math:`\Phi_{\alpha \beta}`
are given by

.. math::

   F_\alpha(jl) = -\frac{\partial V }{\partial r_\alpha(jl)}

and

.. math::

   \Phi_{\alpha\beta}(jl, j'l') = \frac{\partial^2
   V}{\partial r_\alpha(jl) \partial r_\beta(j'l')} =
   -\frac{\partial F_\beta(j'l')}{\partial r_\alpha(jl)},

respectively, where :math:`\alpha`, :math:`\beta`, ..., are the
Cartesian indices, :math:`j`, :math:`j'`, ..., are the indices of
atoms in a unit cell, and :math:`l`, :math:`l'`, ..., are
the indices of unit cells. In the finite displacement method, the
equation for the force constants is approximated as

.. math::

   \Phi_{\alpha\beta}(jl, j'l') \simeq -\frac{
   F_\beta(j'l';\Delta r_\alpha{(jl)}) - F_\beta(j'l')} {\Delta
   r_\alpha(jl)},

where :math:`F_\beta(j'l'; \Delta r_\alpha{(jl)})` are the forces on
atoms with a finite displacement :math:`\Delta r_\alpha{(jl)}` and
usually :math:`F_\beta(j'l') \equiv 0`.

.. _force_constants_solver_theory:
   
Modified Parlinski-Li-Kawazoe method
====================================

The following is a modified and simplified version of the
Parlinski-Li-Kawazoe method, which is just a numerical fitting
approach to obtain force constants from forces and displacements.

The last equation above is represented by matrices as

.. math::

   \mathbf{F} = - \mathbf{U} \mathbf{P},

where :math:`\mathbf{F}`, :math:`\mathbf{P}`, and :math:`\mathbf{U}`
for a pair of atoms, e.g. :math:`\{jl, j'l'\}`, are given by

.. math::

   \mathbf{F} = 
    \begin{pmatrix}
     F_{x} & F_{y} & F_{z} 
    \end{pmatrix},

.. math::
   \mathbf{P} =
     \begin{pmatrix}
     \Phi_{xx} & \Phi_{xy} & \Phi_{xz} \\
     \Phi_{yx} & \Phi_{yy} & \Phi_{yz} \\
     \Phi_{zx} & \Phi_{zy} & \Phi_{zz}
   \end{pmatrix},

.. math::

   \mathbf{U} =
    \begin{pmatrix}
      \Delta r_{x} & \Delta r_{y} & \Delta r_{z} \\
    \end{pmatrix}.

The matrix equation is expanded for number of
forces and displacements as follows:

.. math::

   \begin{pmatrix}
   \mathbf{F}_1 \\
   \mathbf{F}_2 \\
    \vdots
   \end{pmatrix}
   = -
   \begin{pmatrix}
   \mathbf{U}_1 \\
   \mathbf{U}_2 \\
    \vdots
   \end{pmatrix}
   \mathbf{P}.

With sufficient number of atomic displacements, this
may be solved by pseudo inverse such as

.. math::

   \mathbf{P} = -
   \begin{pmatrix}
   \mathbf{U}_1 \\
   \mathbf{U}_2 \\
    \vdots
   \end{pmatrix}^{+}
   \begin{pmatrix}
   \mathbf{F}_1 \\
   \mathbf{F}_2 \\
   \vdots
   \end{pmatrix}.

Required number of atomic displacements to solve the simultaneous
equations may be reduced using site-point symmetries. The matrix
equation can be written using a symmetry operation as

.. math::

  \hat{R}(\mathbf{F}) = -\hat{R}(\mathbf{U})\mathbf{P},

where :math:`\hat{R}` is the site symmetry
operation centring at
:math:`\mathbf{r}(jl)`. :math:`\hat{R}(\mathbf{F})` and :math:`\hat{R}(\mathbf{U})` are defined as
:math:`\mathbf{RF}(\hat{R^{-1}}(j'l'))` and :math:`\mathbf{RU}`,
respectively, where :math:`\mathbf{R}` is the matrix
representation of the rotation operation. The combined
simultaneous equations are built such as

.. math::

   \begin{pmatrix}
   \mathbf{F}^{(1)}_1 \\
   \mathbf{F}^{(2)}_1 \\
   \vdots \\
   \mathbf{F}^{(1)}_2 \\
   \mathbf{F}^{(2)}_2 \\
   \vdots \end{pmatrix} = -
   \begin{pmatrix}
   \mathbf{U}^{(1)}_1 \\ 
   \vdots \\
   \mathbf{U}^{(2)}_1 \\
   \mathbf{U}^{(1)}_2 \\
   \mathbf{U}^{(2)}_2 \\
   \vdots
   \end{pmatrix}
   \mathbf{P}.

where the superscript with parenthesis gives the index of
site-symmetry operations. This is solved by pseudo inverse.

.. _dynacmial_matrix_theory:

Dynamical matrix
================

In phonopy, a phase convention of dynamical matrix is used as follows:

.. math::

   D_{\alpha\beta}(jj',\mathbf{q}) = \frac{1}{\sqrt{m_j m_{j'}}}
    \sum_{l'}
    \Phi_{\alpha\beta}(j0, j'l')
    \exp(i\mathbf{q}\cdot[\mathbf{r}(j'l')-\mathbf{r}(j0)]),

where :math:`m` is the atomic mass and :math:`\mathbf{q}` is the wave
vector. An equation of motion is written as

.. math::

  \sum_{j'\beta} D_{\alpha\beta}(jj',\mathbf{q}) e_\beta(j', \mathbf{q}\nu) =
  m_j [ \omega(\mathbf{q}\nu) ]^2 e_\alpha(j, \mathbf{q}\nu).

where the eigenvector of the band index :math:`\nu` at
:math:`\mathbf{q}` is obtained by the diagonalization of
:math:`\mathbf{D}(\mathbf{q})`:

.. math::

   \sum_{j \alpha j' \beta}e_\alpha(j',\mathbf{q}\nu)^* D_{\alpha\beta}(jj',\mathbf{q})
   e_\beta(j',\mathbf{q}\nu') = [\omega(\mathbf{q}\nu)]^2 \delta_{\nu\nu'}.

  
The atomic displacements :math:`\mathbf{u}` are given as

.. math::

   u_\alpha(jl,t) = \left(\frac{\hbar}{2Nm_j}\right)^{\frac{1}{2}}
   \sum_{\mathbf{q},\nu}\left[\omega(\mathbf{q}\nu)\right]^{-\frac{1}{2}}
   \left[\hat{a}(\mathbf{q}\nu)\exp(-i\omega(\mathbf{q}\nu)t)+
   \hat{a}^\dagger(\mathbf{-q}\nu)\exp({i\omega(\mathbf{q}\nu)}t)\right]
   \exp({i\mathbf{q}\cdot\mathbf{r}(jl)})
   e_\alpha(j,\mathbf{q}\nu),

where :math:`\hat{a}^\dagger` and :math:`\hat{a}` are the creation and
annihilation operators of phonon, :math:`\hbar` is the reduced Planck
constant, and :math:`t` is the time.

.. _non_analytical_term_correction_theory:

Non-analytical term correction
==============================

To correct long range interaction of macroscopic electric field
induced by polarization of collective ionic motions near the
:math:`\Gamma`-point, non-analytical term is added to dynamical matrix
(:ref:`reference_NAC`).  At
:math:`\mathbf{q}\to\mathbf{0}`, the dynamical matrix with
non-analytical term is given by,

.. math::

   D_{\alpha\beta}(jj',\mathbf{q}\to \mathbf{0}) =
    D_{\alpha\beta}(jj',\mathbf{q}=\mathbf{0})
    + \frac{1}{\sqrt{m_j m_j}} \frac{4\pi}{\Omega_0}
    \frac{[\sum_{\gamma}q_{\gamma}Z^{*}_{j,\gamma\alpha}][\sum_{\gamma'}q_{\gamma'}Z^{*}_{j',\gamma'\beta}]}
    {\sum_{\alpha\beta}q_{\alpha}\epsilon_{\alpha\beta}^{\infty} q_{\beta}}.

Phonon frequencies at general **q**-points are interpolated by the
method of Wang *et al.* (:ref:`reference_wang_NAC`).



.. _thermal_properties_expressions:

Thermodynamic properties
========================

Phonon number
-------------

.. math::

   n = \frac{1}{\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)-1}

Harmonic phonon energy
----------------------

.. math::

   E = \sum_{\mathbf{q}\nu}\hbar\omega(\mathbf{q}\nu)\left[\frac{1}{2} +
    \frac{1}{\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)-1}\right]


Constant volume heat capacity
-----------------------------

.. math::

   C_V &= \left(\frac{\partial E}{\partial T} \right)_V \\
       &= \sum_{\mathbf{q}\nu} k_\mathrm{B}
    \left(\frac{\hbar\omega(\mathbf{q}\nu)}{k_\mathrm{B} T} \right)^2
    \frac{\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B}
    T)}{[\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)-1]^2} 

Partition function
------------------

.. math::

   Z = \exp(-\varphi/k_\mathrm{B} T) \prod_{\mathbf{q}\nu}
    \frac{\exp(-\hbar\omega(\mathbf{q}\nu)/2k_\mathrm{B}
    T)}{1-\exp(-\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)} 

Helmholtz free energy
---------------------

.. math::

    F &= -k_\mathrm{B} T \ln Z \\
      &= \varphi + \frac{1}{2} \sum_{\mathbf{q}\nu}
    \hbar\omega(\mathbf{q}\nu) + k_\mathrm{B} T \sum_{\mathbf{q}\nu} \ln
    \bigl[1 -\exp(-\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T) \bigr] 

Entropy
-------

.. math::

    S &= -\frac{\partial F}{\partial T} \\
      &= \frac{1}{2T}\sum_{\mathbf{q}\nu}\hbar\omega(\mathbf{q}\nu)\coth(\hbar\omega(\mathbf{q}\nu)/2k_\mathrm{B}T)-k_\mathrm{B} \sum_{\mathbf{q}\nu}\ln\left[2\sinh(\hbar\omega(\mathbf{q}\nu)/2k_\mathrm{B}T)\right]

Thermal displacement
====================

.. toctree::

   thermal-displacement

Group velocity
==============

.. toctree::

   group-velocity

.. _physical_unit_conversion:

Physical unit conversion
=========================

Phonopy calculates phonon frequencies based on input values from
users. In the default case, the physical units of distance, atomic
mass, force, and force constants are supposed to be
:math:`\text{\AA}`, :math:`\text{AMU}`, :math:`\text{eV/\AA}`, and
:math:`\text{eV/\AA}^2`, respectively, and the physical unit of the
phonon frequency is converted to THz. This conversion is made as
follows:

Internally phonon frequency has the physical unit of
:math:`\sqrt{\text{eV/}(\text{\AA}^2\cdot \text{AMU})}` in angular
frequency. To convert this unit to THz (not angular frequency), the
calculation of ``sqrt(EV/AMU)/Angstrom/(2*pi)/1e12`` is made. ``EV``,
``AMU``, ``Angstrom`` are the values to convert them to those in the
SI base unit, i.e., to Joule, kg, and metre, respectively. These values
implemented in phonopy are found at `a phonopy github page
<https://github.com/atztogo/phonopy/blob/master/phonopy/units.py>`_. This
unit conversion factor can be manually specified. See
:ref:`frequency_conversion_factor_tag`.

The unit conversion factor in the ``BORN`` file is multiplied with the second
term of the right hand side of the equation in
:ref:`non_analytical_term_correction_theory` where this equation is written
with atomic units (:ref:`Gonze and Lee, 1997 <reference_NAC>`).
The physical unit of the part of the equation corresponding to force
constants:

.. math::

    \frac{4\pi}{\Omega_0}
    \frac{[\sum_{\gamma}q_{\gamma}Z^{*}_{j,\gamma\alpha}]
    [\sum_{\gamma'}q_{\gamma'}Z^{*}_{j',\gamma'\beta}]}
    {\sum_{\alpha\beta}q_{\alpha}\epsilon_{\alpha\beta}^{\infty} q_{\beta}}.

is :math:`[\text{hartree}/\text{bohr}^2]`. In the default case for the
VASP interface, internally :math:`\Omega_0` is given in
:math:`\text{\AA}^3`. In total, the necessary unit conversion is
:math:`(\text{hartree} \rightarrow \text{eV}) \times (\text{bohr}
\rightarrow \text{\AA})=14.4`. In the default case of the Wien2k
interface, the conversion factor is :math:`(\text{hartree}
\rightarrow \text{mRy})=2000`. For the other interfaces, the
conversion factors are similarly calculated following the unit
systems employed in phonopy (:ref:`calculator_interfaces`).
      
.. _definition_of_commensurate_points:

Commensurate points
====================

In phonopy, so-called commensurate points mean the q-points whose waves are
confined in the supercell used in the phonon calculation.

To explain about the commensurate points, let basis vectors of a
primitive cell in direct space cell be the column vectors
:math:`(\mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \;
\mathbf{c}_\mathrm{p})` and those of the supercell be
:math:`(\mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \;
\mathbf{c}_\mathrm{s})`. The transformation of the basis vectors from
the primitive cell to the supercell is written as

.. math::

   ( \mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \; \mathbf{c}_\mathrm{s} )
   =  ( \mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \;
   \mathbf{c}_\mathrm{p} ) \boldsymbol{P}.

:math:`\boldsymbol{P}` is given as a :math:`3\times 3` matrix and its
elements are all integers, which is a constraint we have. The
resolution for q-points being the commensurate points is determined by
:math:`\boldsymbol{P}` since one period of a wave has to be bound by
any of lattice points inside the supercell. Therefore the number of
commensurate points becomes the same as the number of the primitive
cell that can be contained in the supercell, i.e.,
:math:`\det(\boldsymbol{P})`.

Then let the basis vectors in reciprocal space be the column vectors
:math:`(\mathbf{a}^*_\mathrm{p} \; \mathbf{b}^*_\mathrm{p} \;
\mathbf{c}^*_\mathrm{p})`. Note that often reciprocal vectors are
deifned by row vectors, but column vectors are chosen here to
formulate. Formally we see the set of besis vectors are :math:`3\times
3` matrices, we have the following relation:

.. math::

   ( \mathbf{a}^*_\mathrm{p} \;
   \mathbf{b}^*_\mathrm{p} \; \mathbf{c}^*_\mathrm{p} ) = 2\pi (
   \mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \;
   \mathbf{c}_\mathrm{p} )^{-\mathbf{T}}.

Similarly for the supercell, we define a relation

.. math::

   ( \mathbf{a}^*_\mathrm{s} \;
   \mathbf{b}^*_\mathrm{s} \; \mathbf{c}^*_\mathrm{s} ) = 2\pi (
   \mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \;
   \mathbf{c}_\mathrm{s} )^{-\mathbf{T}}.

Then 

.. math::

   ( \mathbf{a}^*_\mathrm{s} \; \mathbf{b}^*_\mathrm{s} \;
   \mathbf{c}^*_\mathrm{s} ) \boldsymbol{P}^{\mathrm{T}} = (
   \mathbf{a}^*_\mathrm{p} \; \mathbf{b}^*_\mathrm{p} \;
   \mathbf{c}^*_\mathrm{p} ).

To multiply an arbitrary q-point :math:`\mathbf{q}` on both sides

.. math::

   ( \mathbf{a}^*_\mathrm{s} \; \mathbf{b}^*_\mathrm{s} \;
   \mathbf{c}^*_\mathrm{s} ) \boldsymbol{P}^{\mathrm{T}} \mathbf{q} = (
   \mathbf{a}^*_\mathrm{p} \; \mathbf{b}^*_\mathrm{p} \;
   \mathbf{c}^*_\mathrm{p} ) \mathbf{q},

we find the constraint of a q-point being one of the commensurate points is
the elements of :math:`\boldsymbol{P}^{\mathrm{T}} \mathbf{q}` to be integers.

