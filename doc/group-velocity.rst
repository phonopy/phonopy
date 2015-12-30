.. _group_velocity:

Method
------------

Phonopy calculates group velocity of phonon as follows:

.. math::

   \mathbf{v}_\mathrm{g}(\mathbf{q}\nu) = & \nabla_\mathbf{q} \omega(\mathbf{q}\nu) \\
   =&\frac{\partial\omega(\mathbf{q}\nu)}{\partial \mathbf{q}} \\
   =&\frac{1}{2\omega(\mathbf{q}\nu)}\frac{\partial[\omega(\mathbf{q}\nu)]^2}{\partial
   \mathbf{q}} \\
   =&\frac{1}{2\omega(\mathbf{q}\nu)}\left<\mathbf{e}(\mathbf{q}\nu)\biggl|
   \frac{\partial D(\mathbf{q})} {\partial
   \mathbf{q}}\biggl|\mathbf{e}(\mathbf{q}\nu)\right>,
   
where the meanings of the variables are found at :ref:`formulations`.

Finite difference method
-------------------------

In the previous versions, group velocity was calculated using finite
difference method:

.. math::
   
   \mathbf{v}_\mathrm{g}(\mathbf{q}\nu) =
   \frac{1}{2\omega(\mathbf{q}\nu)}\left<\mathbf{e}(\mathbf{q}\nu)\biggl|
   \frac{\partial D(\mathbf{q})} {\partial
   \mathbf{q}}\biggl|\mathbf{e}(\mathbf{q}\nu)\right>
   \simeq \frac{1}{2\omega(\mathbf{q}\nu)}
   \left<\mathbf{e}(\mathbf{q}\nu)\biggl|
   \frac{\Delta D(\mathbf{q})}
   {\Delta \mathbf{q}}\biggl|\mathbf{e}(\mathbf{q}\nu)\right>.

Group velocity calculation with the finite difference method is still
able to be activated using ``GV_DELTA_Q`` tag or ``-gv_delta_q``
option.  :math:`\Delta\mathbf{q} = (\Delta q_x, \Delta q_y, \Delta
q_z)` is described in Cartesian coordinated in reciprocal space. In
the implementation, central difference is employed, and :math:`+\Delta
q_\alpha` and :math:`-\Delta q_\alpha` are taken to calculate group
velocity, where :math:`\alpha` is the Cartesian index in reciprocal
space. :math:`\Delta q_\alpha` is specified in the unit of reciprocal
space distance (:math:`\mathrm{\AA}^{-1}` for the default case) by
``--gv_delta_q`` option or ``GV_DELTA_Q`` tag.

