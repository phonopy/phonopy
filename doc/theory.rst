==============================
Formulations
==============================

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

   
The Palinski-Li-Kawazoe method
==============================

The following is considered as the Parlinski-Li-Kawazoe method,
however the exact definition may be different. 

The last equation above is represented by matrices as

.. math::

   \mathbf{F} = - \mathbf{U} \mathbf{P},

where :math:`\mathbf{F}`, :math:`\mathbf{P}`, and :math:`\mathbf{U}`
are given by

.. math::

   \mathbf{F} = 
    \begin{pmatrix}
     F_{x} \\
     F_{y} \\
     F_{z} 
    \end{pmatrix},

.. math::
   \mathbf{P} = 
     \begin{pmatrix}
     \Phi_{xx} \\
     \Phi_{xy} \\
     \Phi_{xz} \\
     \Phi_{yx} \\
     \Phi_{yy} \\
     \Phi_{yz} \\
     \Phi_{zx} \\
     \Phi_{zy} \\
     \Phi_{zz}
   \end{pmatrix},

.. math::

   \mathbf{U} = 
    \begin{pmatrix}
      \Delta r_{x} & 0 & 0 & \Delta r_{y} & 0 & 0 & \Delta r_{z} & 0 & 0 \\
      0 & \Delta r_{x} & 0 & 0 & \Delta r_{y} & 0 & 0 & \Delta r_{z} & 0 \\
      0 & 0 & \Delta r_{x} & 0 & 0 & \Delta r_{y} & 0 & 0 & \Delta r_{z} \\
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
equation can be written using a site-point symmetry operation as

.. math::

  \mathbf{F}' = -\mathbf{U} \mathbf{A} \mathbf{P},

where :math:`\mathbf{F}'` is the force at the atomic site translated
from the original atomic site by the site-point symmetry operation,
and :math:`\mathbf{A}` is the :math:`9\times 9` site-point symmetry
matrix to rotate :math:`\mathbf{P}`.  Thus, the combined simultaneous
equations are built such as

.. math::

   \begin{pmatrix}
   \mathbf{F}^{(1)}_1 \\
   \mathbf{F}^{(2)}_1 \\
   \vdots \\
   \mathbf{F}^{(1)}_2 \\
   \mathbf{F}^{(2)}_2 \\
   \vdots \end{pmatrix} = -
   \begin{pmatrix}
   \mathbf{U}_1
   \mathbf{A}^{(1)} \\
   \vdots \\
   \mathbf{U}_1 \mathbf{A}^{(2)} \\
   \mathbf{U}_2 \mathbf{A}^{(1)} \\
   \mathbf{U}_2 \mathbf{A}^{(2)} \\
   \vdots
   \end{pmatrix}
   \mathbf{P}.

where the superscript with parenthesis gives the index of
site-symmetry operations. This is solved by pseudo inverse.

.. _dynacmial_matrix_theory:

Dynamical matrix
=================

The dynamical matrix is defined by

.. math::

   D_{\alpha\beta}(jj',\mathbf{q}) = \frac{1}{\sqrt{m_j m_{j'}}}
    \sum_{l'}
    \Phi_{\alpha\beta}(j0, j'l')
    \exp(i\mathbf{q}\cdot[\mathbf{r}(j'l')-\mathbf{r}(j0)]),

where :math:`m` is the atomic mass and :math:`\mathbf{q}` is the wave
vector. An equation of motion is writtein as

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


.. _thermal_properties_expressions:

Thermodynamic properties
=========================

Phonon number
--------------

.. math::

   n = \frac{1}{\exp(\hbar\omega(\mathbf{q}\nu)/k_B T)-1}

Harmonic phonon energy
-----------------------

.. math::

   E = \sum_{\mathbf{q}\nu}\hbar\omega(\mathbf{q}\nu)\left[\frac{1}{2} +
    \frac{1}{\exp(\hbar\omega(\mathbf{q}\nu)/k_B T)-1}\right]


Constant volume heat capacity
-------------------------------

.. math::

   C_V &= \left(\frac{\partial E}{\partial T} \right)_V \\
       &= \sum_{\mathbf{q}\nu} k_B
    \left(\frac{\hbar\omega(\mathbf{q}\nu)}{k_B T} \right)^2
    \frac{\exp(\hbar\omega(\mathbf{q}\nu)/k_B
    T)}{[\exp(\hbar\omega(\mathbf{q}\nu)/k_B T)-1]^2} 

Partition function
-------------------

.. math::

   Z = \exp(-\varphi/k_B T) \prod_{\mathbf{q}\nu}
    \frac{\exp(-\hbar\omega(\mathbf{q}\nu)/2k_B
    T)}{1-\exp(-\hbar\omega(\mathbf{q}\nu)/k_B T)} 

Helmholtz free energy
----------------------

.. math::

    F &= -k_B T \ln Z \\
      &= \varphi + \frac{1}{2} \sum_{\mathbf{q}\nu}
    \hbar\omega(\mathbf{q}\nu) + k_B T \sum_{\mathbf{q}\nu} \ln
    \bigl[1 -\exp(-\hbar\omega(\mathbf{q}\nu)/k_B T) \bigr] 

Entropy
---------

.. math::

    S &= -\frac{\partial F}{\partial T} \\
      &= -k_B \sum_{\mathbf{q}\nu} \ln
    \left[1 -\exp(-\hbar\omega(\mathbf{k},\nu)/k_B T) \right] -
    \frac{1}{T} \sum_{\mathbf{q}\nu} 
    \frac{\hbar\omega(\mathbf{q}\nu)}{\exp(\hbar\omega(\mathbf{q}\nu)/k_B T)-1}


.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|
