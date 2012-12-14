.. fc3 documentation master file, created by
   sphinx-quickstart on Mon May 25 14:36:19 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Third-order force constant document
===================================

Contents:

.. toctree::
   :maxdepth: 2

Method
======

Force :math:`F_\alpha` is first derivative of potential energy with
respect to an atomic displacement :math:`u`,

.. math::

   F_\alpha(j_nl_n) = -\frac{\partial V[\mathbf{u}(j_1l_1),\ldots,\mathbf{u}(j_Nl_N)]}{\partial u_\alpha(j_nl_n)},

where :math:`\alpha` gives the index of Cartesian axis,
:math:`\mathbf{u}(jl)` is the displacement from the equilibrium
position of :math:`j`-th atom in :math:`l`-th unit cell.  :math:`N`
are the number of atoms in unit cell and number of unit cells,
respectively. Second-order force constant :math:`\Phi_{\alpha\beta}`
is second derivative of potential energy as functions of
displacements,

.. math::

   \Phi_{\alpha\beta}(jl, j'l') = \frac{\partial^2 V}{\partial u_\alpha(jl) \partial
   u_\beta(j'l')}.

This is considered as being equivalent to first derivative of force
with respect to displacement,

.. math::

   \Phi_{\alpha\beta}(jl, j'l') = -\frac{\partial F_\beta(j'l'; \mathbf{u}(j_1l_1),\cdots,\mathbf{u}(j_Nl_N))}{\partial u_\alpha(jl)}.

Third-order force constant :math:`\Phi_{\alpha\beta\gamma}` is third
derivative of potential energy as functions of displacements,

.. math::

   \Phi_{\alpha\beta\gamma}(jl, j'l', j''l'') = & \frac{\partial^3
   V}{\partial u_\alpha(jl) \partial u_\beta(j'l') \partial
   u_\gamma(j''l'')} \\ = & \frac{\partial
   \Phi_{\beta\gamma}(j'l',j''l'';
   \mathbf{u}(j_1l_1),\cdots,\mathbf{u}(j_Nl_N))}{\partial
   u_\alpha(jl)}.
   
The third-order force constants are to be calculated if we know forces
on all atoms and two displacements. We employed finite displacements
for the partial differentiation as follows,

.. math::

   \Phi_{\alpha\beta\gamma} \simeq \frac{\Delta_\alpha
   \Phi_{\beta\gamma}}{\Delta u_\alpha} = \frac{
   \Phi_{\beta\gamma}|_\alpha - \Phi_{\beta\gamma}}{\Delta u_\alpha}
   \simeq \frac{\frac{F_\gamma|_{\alpha\beta} - F_\gamma|_{\alpha}}{
   \Delta u_\beta} - \Phi_{\beta\gamma}}{\Delta u_\alpha}

where :math:`X|_{\alpha,\cdots}` :math:`(X=\Phi, F)` means the variable
is calculated under :math:`\Delta u_\alpha(jl)`,...

In our system, the forces are obtained in periodic boundary
condition. This gives special constraints to derive the second- and
third-order force constants. The method to calculate the second-order
force constant is establisehd by Parlinski *et. al*. The third-order
foce constant can be obtained in similar manner to thier method, which
is described in the following sections.

Computational method
--------------------------------

:math:`\Phi_{\alpha\beta\gamma}(jl,j'l',j''l'')` should be obtained by
solving,

.. math::

   \Delta_{\alpha jl} \Phi_{\beta\gamma}(jl,j'l',j''l'') =
   \Phi_{\alpha\beta\gamma}(jl,j'l',j''l'')\Delta u_\alpha(jl),

where :math:`\Delta_{\alpha jl}\Phi_{\beta\gamma}` is the perturbed second-order
force constant by an additional displacement
:math:`u_\alpha(jl)`. :math:`\Delta_{\alpha jl}\Phi_{\beta\gamma}` is
calculated in the same manner as second-order force constant, however,
in keeping with the atom :math:`u_\alpha(jl)`
displaced. To handle this calculation using computer, this equation is
written with matrices:

.. math::

   \Delta \mathbf{P} = \mathbf{U} \cdot \mathbf{P}^{\mathrm{third}},

where :math:`\Delta \mathbf{P}`, :math:`\mathbf{U}`, and
:math:`\mathbf{P}^{\mathrm{third}}` are the matrices approximately
corresponding to :math:`\Delta_\alpha\Phi_{\beta\gamma}`,
:math:`\Delta u_{\alpha}`, and :math:`\Phi_{\alpha\beta\gamma}`,
respectively. The dimensions of the matrices are also modified to
the convenient forms,

.. math::

       \Delta\mathbf{P}=
        \begin{pmatrix}
      \Delta \Phi_{xx} \\
      \Delta \Phi_{xy} \\
      \Delta \Phi_{xz} \\
      \Delta \Phi_{yx} \\
      \Delta \Phi_{yy} \\
      \Delta \Phi_{yz} \\
      \Delta \Phi_{zx} \\
      \Delta \Phi_{zy} \\
      \Delta \Phi_{zz} \\
    \end{pmatrix},


.. math::

       \mathbf{P}^\mathrm{third}=
        \begin{pmatrix}
      \Phi_{xxx} \\
      \Phi_{xxy} \\
      \Phi_{xxz} \\
      \Phi_{xyx} \\
      \Phi_{xyy} \\
      \Phi_{xyz} \\
      \Phi_{xzx} \\
      \Phi_{xzy} \\
      \Phi_{xzz} \\
      \Phi_{yxx} \\
      \Phi_{yxy} \\
      \Phi_{yxz} \\
      \Phi_{yyx} \\
      \Phi_{yyy} \\
      \Phi_{yyz} \\
      \Phi_{yzx} \\
      \Phi_{yzy} \\
      \Phi_{yzz} \\
      \Phi_{zxx} \\
      \Phi_{zxy} \\
      \Phi_{zxz} \\
      \Phi_{zyx} \\
      \Phi_{zyy} \\
      \Phi_{zyz} \\
      \Phi_{zzx} \\
      \Phi_{zzy} \\
      \Phi_{zzz} \\
    \end{pmatrix},

and :math:`\mathbf{U}` is a :math:`9\times 27` matrix given by

.. math::

      \mathbf{U} =
      \left(\mathbf{U}_x\; \mathbf{U}_y\; \mathbf{U}_z \right)\\

where :math:`\mathbf{U}_x = \Delta u_x \cdot \mathbf{I}_9`, :math:`\mathbf{U}_y
= \Delta u_y \cdot \mathbf{I}_9`, and :math:`\mathbf{U}_z = \Delta u_z \cdot
\mathbf{I}_9`, and :math:`\mathbf{I}_9` is the identity matrix of size 9.

To fill the elements of the third-order force constant, at least three
linearly independent displacements for each displaced atom are
required. Combining these three or more than three sets of above
matrices in rows:

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1\\
   \Delta\mathbf{P}_2\\
   \Delta\mathbf{P}_3
   \end{pmatrix} =
   \begin{pmatrix}
   \mathbf{U}_1\\
   \mathbf{U}_2\\
   \mathbf{U}_3
   \end{pmatrix} \cdot 
   \mathbf{P}^\mathrm{third},

and the matrix inversion can be done as
   
.. math::

   \mathbf{P}^\mathrm{third} =
   \begin{pmatrix}
   \mathbf{U}_1\\
   \mathbf{U}_2\\
   \mathbf{U}_3
   \end{pmatrix}^{-1}
   \cdot
   \begin{pmatrix}
   \Delta\mathbf{P}_1\\
   \Delta\mathbf{P}_2\\
   \Delta\mathbf{P}_3
   \end{pmatrix}.


Crystal symmetry can reduce the number of displacements
:math:`\mathbf{U}_i` to be used to calculate respective
:math:`\Delta\mathbf{P}_i`. The reduced set of displacements is
obtained from the site-point symmetry and only
:math:`\Delta\mathbf{P}_i` with those :math:`\mathbf{U}_i` are
calculated. To solve above inversion, symmetry non-reduced pairs of
:math:`(\mathbf{U}_i,\Delta\mathbf{P}_i)` have to be recovered.

For example, for :math:`\Phi_{\alpha\beta\gamma}(1,2,3)` when we need
two displacement directions at atom 1 in x-y plane for the above
inversion, one of them (:math:`\Delta u_2`) may be recovered from the
other (:math:`\Delta u_1`) by symmetry operation.  As shown in the
figures below, :math:`\Delta u_2` is obtained by :math:`\mathbf{C}_4`
rotation of :math:`\Delta u_1` (Figs. (a) and (b)), which is active
transformation. Therefore

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \Delta\mathbf{P}_2 \\
   \Delta\mathbf{P}_3
   \end{pmatrix} =
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{third} \\
   \mathbf{U}_2 \cdot \mathbf{P}^\mathrm{third} \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{third}
   \end{pmatrix} =
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{third} \\
   \hat{C}_4(\mathbf{U}_1) \cdot \mathbf{P}^\mathrm{third} \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{third}
   \end{pmatrix}.

With passive transformation, :math:`\mathbf{C}_4`
rotation of :math:`\Delta u_1` is equivalent to
:math:`\mathbf{C}_4^{-1}` rotation of axes :math:`(x,y,z)
\xrightarrow[\mathbf{C}_4^{-1}]{} (x',y',z')` (Figs. (a) and
(c)). Therefore

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \hat{C}_4^{-1}(\Delta\mathbf{P}_2) \\
   \Delta\mathbf{P}_3
   \end{pmatrix} = 
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{third} \\
   \mathbf{U}_1 \cdot \hat{C}_4^{-1}(\mathbf{P}^\mathrm{third}) \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{third}
   \end{pmatrix}.

.. |c4| image:: rotation-C4.png
        :scale: 70

|c4|

Fig. (d) is just the figure of Fig. (b) that is drawn rotated
clockwise 90 degrees. To see Figs. (c) and (d), the values of
:math:`\hat{C}_4^{-1}(\Delta\mathbf{P}_2)` are obtained from
:math:`\Delta\mathbf{P}_1` by combination of rotation mapping of
atomic pairs, e.g., atomic pair (2, 3) is mapped onto that of (5, 2),
and rotation of second order tensor, e.g.,
:math:`\hat{C}_4^{-1}(\Delta_y\Phi_{xy}(2,3)) =
\Delta_{y'}\Phi_{x'y'}(5,2) = \Delta_x\Phi_{-yx}(5,2)`.
:math:`\hat{C}_4^{-1}(\mathbf{P}^\mathrm{third})` is treated as
follows. Since values of tensor elements of
:math:`\Phi_{\alpha\beta\gamma}(jl,j'l',j''l'')` are unknown, we want
to push the tensor rotation information of third-order force constants
into somewhere else. The way how the third-order force constants are
rotated is represented by a :math:`27\times 27` matrix
:math:`\mathbf{A}` and this matrix is put on the left-hand side of
:math:`\mathbf{P}^\mathrm{third}`. Then

.. math::

   \mathbf{U}\hat{R}(\mathbf{P}^\mathrm{third}) =
   \mathbf{U}(\mathbf{AP}^\mathrm{third}) = 
   (\mathbf{UA})\mathbf{P}^\mathrm{third}.

Thus the rotation is pushed into a displacement.
:math:`\mathbf{A}` is constructed using the definition of rotation of
tensor, :math:`\Phi_{lmn} =
\sum_{i}\sum_{j}\sum_{k}\mathbf{R}_{li}\mathbf{R}_{mj}\mathbf{R}_{nk}\Phi_{ijk}`,
i.e., :math:`[\mathbf{A}]_{9l+3m+n\; 9i+3j+k} =
\mathbf{R}_{li}\mathbf{R}_{mj}\mathbf{R}_{nk}`. The above example is also rewritten as:

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \hat{C}_4^{-1}(\Delta\mathbf{P}_2) \\
   \Delta\mathbf{P}_3
   \end{pmatrix} = 
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{third} \\
   \mathbf{U}_1 \cdot \mathbf{A^*}\mathbf{P}^\mathrm{third} \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{third}
   \end{pmatrix} = 
   \begin{pmatrix}
   \mathbf{U}_1 \\
   \mathbf{U}_1\mathbf{A}^* \\
   \mathbf{U}_3 
   \end{pmatrix} \cdot \mathbf{P}^\mathrm{third},

where :math:`\mathbf{A}^*` is the matrix corresponding to
:math:`\hat{C}_4^{-1}`. Finally we obtain
:math:`\mathbf{P}^\mathrm{third}` as

.. math::

   \mathbf{P}^\mathrm{third} = 
   \begin{pmatrix}
   \mathbf{U}_1 \\
   \mathbf{U}_1\mathbf{A}^* \\
   \mathbf{U}_3 
   \end{pmatrix}^{-1} \cdot
   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \hat{C}_4^{-1}(\Delta\mathbf{P}_2) \\
   \Delta\mathbf{P}_3
   \end{pmatrix}

In this example, number of displacements are three, but in phonopy
:math:`\hat{C}_4` and :math:`\hat{C}_2` are also recovered. Therefore
more pairs of :math:`(\mathbf{U}_i,\Delta\mathbf{P}_i)` are used to
the matrix inversion. In addition, more displacements can be involved,
e.g., when positive and negative displacements are not symmetrically
equivalent, those two are simultaneously employed in the default
behavior of phonopy, though one of them is enough to perform the above
inversion.

Generation of reduced pair-displacements
----------------------------------------

Using crystal symmetry, number of pairs of displacements, which are
required to calculate third-order force constants, are reduced.  A
pair of displacements is considered as a displacement after another
displacement. Least atomic positions for the first displacements can
be found using full symmetry including nonsymmorphic
operations. Directions of the first displacements are reduced using
site-point symmetry. The second displacement may not be found in the
same way as that of the first displacement. Due to the first
displacement, the crystal symmetry is broken and only a part of
site-point symmetry survives. Some of the atoms may still be
symmetrically equivalent each other. In this case, number of the
positions of the second displacements are reduced. Site-point symmetry
at the second atomic position is looked for among the limited symmetry
operations whose rotations do not change the direction
of the first displacement. Directions of the second
displacements are reduced using this site-point symmetry.

