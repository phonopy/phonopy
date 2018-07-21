.. _dynamic_structure_factor:

Dynamic structure factor
-------------------------

**This is the feature under testing.**

From Eq. (3.120) in the book "Thermal of Neutron Scattering", coherent
one-phonon dynamic structure factor is given as

.. math::

   S(\mathbf{Q}, \nu, \omega)^{+1\text{ph}} =
   \frac{k'}{k} \frac{N}{\hbar}
   \sum_\mathbf{q} |F(\mathbf{Q}, \mathbf{q}\nu)|^2
   (n_{\mathbf{q}\nu} + 1) \delta(\omega - \omega_{\mathbf{q}\nu})
   \Delta(\mathbf{Q-q}),

and

.. math::

   S(\mathbf{Q}, \nu, \omega)^{-1\text{ph}} =
   \frac{k'}{k} \frac{N}{\hbar}
   \sum_\mathbf{q} |F(\mathbf{Q}, \mathbf{q}\nu)|^2
   n_{\mathbf{q}\nu} \delta(\omega + \omega_{\mathbf{q}\nu})
   \Delta(\mathbf{Q+q}),

with

.. math::

   F(\mathbf{Q}, \mathbf{q}\nu) =
   \sum_j \sqrt{\frac{\hbar}{2 m_j \omega_{\mathbf{q}\nu}}}
   \bar{b}_j \exp\left(
   -\frac{1}{2} \langle |\mathbf{Q}\cdot\mathbf{u}(j0)|^2 \rangle
   \right) \mathbf{Q}\cdot\mathbf{e}(j, \mathbf{q}\nu).

where :math:`\mathbf{Q}` is the scattering vector defined as
:math:`\mathbf{Q} = \mathbf{k} - \mathbf{k}'` with incident
wave vector :math:`\mathbf{k}` and final wavevector
:math:`\mathbf{k}'`
following the book "Thermal of Neutron Scattering". For
inelastic neutron scattering, :math:`\bar{b}_j` is the average
scattering length over isotopes and spins. For inelastic X-ray
scattering, :math:`\bar{b}_j` is replaced by atomic form factor
:math:`f_j(\mathbf{Q})` and :math:`k'/k \sim 1`.

Currently only :math:`S(\mathbf{Q}, \nu, \omega)^{+1\text{ph}}` is
calcualted with setting :math:`N k'/k = 1` and the physical unit is
:math:`\text{m}^2/\text{J}` when :math:`\bar{b}_j` is given in
Angstrom.
