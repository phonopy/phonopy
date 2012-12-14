Fourth-order force constant document
====================================

Contents:

.. toctree::
   :maxdepth: 2

Method
======

.. math::

   \Phi_{\alpha\beta\gamma\delta} \simeq \frac{\Delta_\alpha
   \Phi_{\beta\gamma\delta}}{\Delta u_\alpha} =
   \frac{\Phi_{\beta\gamma\delta}|_\alpha - \Phi_{\beta\gamma\delta}}
   {\Delta u_\alpha} \simeq 
   \frac{\frac{\Delta_\beta \Phi_{\gamma\delta}|_\alpha}{\Delta u_\beta} - \Phi_{\beta\gamma\delta}}
   {\Delta u_\alpha} =
   \frac{\frac{\Phi_{\gamma\delta}|_{\alpha\beta} -
   \Phi_{\gamma\delta}|_{\alpha}}{\Delta u_\beta} - \Phi_{\beta\gamma\delta}}
   {\Delta u_\alpha}


Computational method
--------------------------------

:math:`\Phi_{\alpha\beta\gamma\delta}(jl,j'l',j''l'',j'''l''')` should
be obtained by solving,

.. math::

   \Delta_\alpha \Phi_{\beta\gamma\delta}(jl,j'l',j''l'',j'''l''') =
   \Phi_{\alpha\beta\gamma\delta}(jl,j'l',j''l'',j'''l''')\Delta u_\alpha(jl),

.. math::

   \Delta \mathbf{P} = \mathbf{U} \cdot \mathbf{P}^{\mathrm{fourth}},

.. math::

       \Delta\mathbf{P}=
        \begin{pmatrix}
      \Delta \Phi_{xxx} \\
      \Delta \Phi_{xxy} \\
      \Delta \Phi_{xxz} \\
      \Delta \Phi_{xyx} \\
      \Delta \Phi_{xyy} \\
      \Delta \Phi_{xyz} \\
      \Delta \Phi_{xzx} \\
      \Delta \Phi_{xzy} \\
      \Delta \Phi_{xzz} \\
      \Delta \Phi_{yxx} \\
      \vdots \\
      \Delta \Phi_{zzz} \\
    \end{pmatrix},

.. math::

       \mathbf{P}^\mathrm{fourth}=
        \begin{pmatrix}
      \Phi_{xxxx} \\
      \Phi_{xxxy} \\
      \Phi_{xxxz} \\
      \Phi_{xxyx} \\
      \Phi_{xxyy} \\
      \Phi_{xxyz} \\
      \Phi_{xxzx} \\
      \Phi_{xxzy} \\
      \Phi_{xxzz} \\
      \Phi_{xyxx} \\
      \Phi_{xyxy} \\
      \Phi_{xyxz} \\
      \Phi_{xyyx} \\
      \Phi_{xyyy} \\
      \Phi_{xyyz} \\
      \Phi_{xyzx} \\
      \Phi_{xyzy} \\
      \Phi_{xyzz} \\
      \Phi_{xzxx} \\
      \Phi_{xzxy} \\
      \Phi_{xzxz} \\
      \Phi_{xzyx} \\
      \Phi_{xzyy} \\
      \Phi_{xzyz} \\
      \Phi_{xzzx} \\
      \Phi_{xzzy} \\
      \Phi_{xzzz} \\
      \Phi_{yxxx} \\
      \vdots \\
      \Phi_{zzzz} \\
    \end{pmatrix},

and :math:`\mathbf{U}` is a :math:`27\times 81` matrix given by

.. math::

      \mathbf{U} =
      \left(\mathbf{U}_x\; \mathbf{U}_y\; \mathbf{U}_z \right)\\

where :math:`\mathbf{U}_x = \Delta u_x \cdot \mathbf{I}_{27}`,
:math:`\mathbf{U}_y = \Delta u_y \cdot \mathbf{I}_{27}`, and
:math:`\mathbf{U}_z = \Delta u_z \cdot \mathbf{I}_{27}`, and
:math:`\mathbf{I}_{27}` is the identity matrix of size 27.

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
   \mathbf{P}^\mathrm{fourth},

and the matrix inversion can be done as
   
.. math::

   \mathbf{P}^\mathrm{fourth} =
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

.. |c4| image:: rotation-C4.png
        :scale: 70

|c4|

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \Delta\mathbf{P}_2 \\
   \Delta\mathbf{P}_3
   \end{pmatrix} =
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{fourth} \\
   \mathbf{U}_2 \cdot \mathbf{P}^\mathrm{fourth} \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{fourth}
   \end{pmatrix} =
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{fourth} \\
   \hat{C}_4(\mathbf{U}_1) \cdot \mathbf{P}^\mathrm{fourth} \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{fourth}
   \end{pmatrix}.

This is rewritten by rotated :math:`\mathbf{P}^\mathrm{fourth}` and
:math:`\Delta\mathbf{P}_i` instead of rotated :math:`\mathbf{U}_i`,

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \hat{C}_4^{-1}(\Delta\mathbf{P}_2) \\
   \Delta\mathbf{P}_3
   \end{pmatrix} = 
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{fourth} \\
   \mathbf{U}_1 \cdot \hat{C}_4^{-1}(\mathbf{P}^\mathrm{fourth}) \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{fourth}
   \end{pmatrix}


Fig. (d) is just the figure of Fig. (b) that is drawn rotated
clockwise 90 degrees. To see Figs. (c) and (d), the values of
:math:`\hat{C}_4^{-1}(\Delta\mathbf{P}_2)` are obtained from
:math:`\Delta\mathbf{P}_1` by combination of rotation mapping of
atomic pairs, e.g., atomic triplet (2, 3, 4) is mapped onto that of
(5, 2, 3), and rotation of third order tensor, e.g.,
:math:`\hat{C}_4^{-1}(\Delta_y\Phi_{xyz}(2,3,4)) =
\Delta_{y'}\Phi_{x'y'z'}(5,2,3) = \Delta_x\Phi_{-yxz}(5,2,3)`.
:math:`\hat{C}_4^{-1}(\mathbf{P}^\mathrm{fourth})` is treated as
follows. Since values of elements of
:math:`\Phi_{\alpha\beta\gamma\delta}(jl,j'l',j''l'',j'''l''')` are
unknown. Therefore we want to push the information of rotation of
fourth-order force constants into somewhere else. The way how the
fourth-order force constants are rotated can be represented by a
:math:`81\times 81` matrix :math:`\mathbf{A}` and this matrix is put
on the left-hand side of :math:`\mathbf{P}^\mathrm{fourth}`. Then

.. math::

   \mathbf{U}\hat{R}(\mathbf{P}^\mathrm{fourth}) =
   \mathbf{U}(\mathbf{AP}^\mathrm{fourth}) = 
   (\mathbf{UA})\mathbf{P}^\mathrm{fourth}.

Thus the rotation is pushed into a displacement.  :math:`\mathbf{A}`
is constructed using the definition of rotation of tensor,
:math:`\Phi_{mnpq} =
\sum_{i}\sum_{j}\sum_{k}\sum_{l}\mathbf{R}_{mi}\mathbf{R}_{nj}\mathbf{R}_{pk}\mathbf{R}_{ql}\Phi_{ijkl}`,
i.e., :math:`[\mathbf{A}]_{27m+9n+3p+q\; 27i+9j+3k+l} =
\mathbf{R}_{mi}\mathbf{R}_{nj}\mathbf{R}_{pk}\mathbf{R}_{ql}`. The
above example is also rewritten as:

.. math::

   \begin{pmatrix}
   \Delta\mathbf{P}_1 \\
   \hat{C}_4^{-1}(\Delta\mathbf{P}_2) \\
   \Delta\mathbf{P}_3
   \end{pmatrix} = 
   \begin{pmatrix}
   \mathbf{U}_1 \cdot \mathbf{P}^\mathrm{fourth} \\
   \mathbf{U}_1 \cdot \mathbf{A^*}\mathbf{P}^\mathrm{fourth} \\
   \mathbf{U}_3 \cdot \mathbf{P}^\mathrm{fourth}
   \end{pmatrix} = 
   \begin{pmatrix}
   \mathbf{U}_1 \\
   \mathbf{U}_1\mathbf{A}^* \\
   \mathbf{U}_3 
   \end{pmatrix} \cdot \mathbf{P}^\mathrm{fourth},

where :math:`\mathbf{A}^*` is the matrix corresponding to
:math:`\hat{C}_4^{-1}`. Finally we obtain
:math:`\mathbf{P}^\mathrm{fourth}` as

.. math::

   \mathbf{P}^\mathrm{fourth} = 
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
