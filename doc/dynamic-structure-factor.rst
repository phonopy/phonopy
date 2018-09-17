.. _dynamic_structure_factor:

Dynamic structure factor
========================

**This feature is under testing.**

From Eq. (3.120) in the book "Thermal of Neutron Scattering", coherent
one-phonon dynamic structure factor is given as

.. math::

   S(\mathbf{Q}, \nu, \omega)^{+1\text{ph}} =
   \frac{k'}{k} \frac{N}{\hbar}
   \sum_\mathbf{q} |F(\mathbf{Q}, \mathbf{q}\nu)|^2
   (n_{\mathbf{q}\nu} + 1) \delta(\omega - \omega_{\mathbf{q}\nu})
   \Delta(\mathbf{Q-q}),

.. math::

   S(\mathbf{Q}, \nu, \omega)^{-1\text{ph}} =
   \frac{k'}{k} \frac{N}{\hbar}
   \sum_\mathbf{q} |F(\mathbf{Q}, \mathbf{q}\nu)|^2
   n_{\mathbf{q}\nu} \delta(\omega + \omega_{\mathbf{q}\nu})
   \Delta(\mathbf{Q-q}),

with

.. math::

   F(\mathbf{Q}, \mathbf{q}\nu) = \sum_j \sqrt{\frac{\hbar}{2 m_j
   \omega_{\mathbf{q}\nu}}} \bar{b}_j \exp\left( -\frac{1}{2} \langle
   |\mathbf{Q}\cdot\mathbf{u}(j0)|^2 \rangle \right)
   \exp[-i(\mathbf{Q-q})\cdot\mathbf{r}(j0)]
   \mathbf{Q}\cdot\mathbf{e}(j, \mathbf{q}\nu).

where :math:`\mathbf{Q}` is the scattering vector defined as
:math:`\mathbf{Q} = \mathbf{k} - \mathbf{k}'` with incident wave
vector :math:`\mathbf{k}` and final wavevector
:math:`\mathbf{k}'`. Similarly, :math:`\omega=1/\hbar (E-E')` where
:math:`E` and :math:`E'` are the energies of the incident and final
particles. These follow the convention of the book "Thermal of Neutron
Scattering". In some other text books, their definitions have opposite
sign. :math:`\Delta(\mathbf{Q-q})` is defined so that
:math:`\Delta(\mathbf{Q-q})=1` with
:math:`\mathbf{Q}-\mathbf{q}=\mathbf{G}` and
:math:`\Delta(\mathbf{Q-q})=0` with :math:`\mathbf{Q}-\mathbf{q} \neq
\mathbf{G}` where :math:`\mathbf{G}` is any reciprocal lattice
vector. Other variables are refered to :ref:`formulations` page. Note
that the phase convention of the dynamical matrix given :ref:`here
<dynacmial_matrix_theory>` is used. This changes the representation of
the phase factor in :math:`F(\mathbf{Q}, \mathbf{q}\nu)` from that
given in the book "Thermal of Neutron Scattering", but the additional
term :math:`\exp(i\mathbf{q}\cdot\mathbf{r})` comes from the different
phase convention of the dynamical matrix or equivalently the
eigenvector. For inelastic neutron scattering, :math:`\bar{b}_j` is
the average scattering length over isotopes and spins. For inelastic
X-ray scattering, :math:`\bar{b}_j` is replaced by atomic form factor
:math:`f_j(\mathbf{Q})` and :math:`k'/k \sim 1`.

Currently only :math:`S(\mathbf{Q}, \nu, \omega)^{+1\text{ph}}` is
calcualted with setting :math:`N k'/k = 1` and the physical unit is
:math:`\text{m}^2/\text{J}` when :math:`\bar{b}_j` is given in
Angstrom.

Usage
-----

Currently this feature is usable only from API. The following example
runs with the input files in ``example/NaCl``.

::

   #!/usr/bin/env python

   import numpy as np
   from phonopy import load
   from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995
   from phonopy.phonon.degeneracy import degenerate_sets


   def get_func_AFF(f_params):
       def func(symbol, Q):
           return atomic_form_factor_WK1995(Q, f_params[symbol])
       return func


   def run(phonon,
           G_points_cubic,
           directions,
           temperature,
           func_AFF=None,
           scattering_lengths=None,
           n_points=51,
           verbose=False):
       P = phonon.primitive_matrix

       for G_cubic in np.array(G_points_cubic):
           G_prim = np.dot(G_cubic, P)
           for direction in directions:
               direction_prim = np.dot(direction, P)

               if verbose:
                   print("# %s to %s (Primitive: %s to %s)"
                         % (G_cubic, G_cubic + direction,
                            G_prim, G_prim + direction_prim))

               qpoints = np.array(
                   [direction_prim * x
                    for x in np.arange(n_points) / float(n_points - 1)])
               phonon.set_band_structure([qpoints])
               _, distances, frequencies, _ = phonon.get_band_structure()
               # Remove Gamma point because number of bands is different.
               qpoints = qpoints[1:]
               distances = distances[0][1:]
               frequencies = frequencies[0][1:]

               if func_AFF is not None:
                   phonon.set_dynamic_structure_factor(
                       qpoints,
                       G_prim,
                       temperature,
                       func_atomic_form_factor=func_AFF,
                       freq_min=1e-3,
                       run_immediately=False)
               elif scattering_lengths is not None:
                   phonon.set_dynamic_structure_factor(
                       qpoints,
                       G_prim,
                       temperature,
                       scattering_lengths=scattering_lengths,
                       freq_min=1e-3,
                       run_immediately=False)
               else:
                   raise SyntaxError
               dsf = phonon.dynamic_structure_factor
               for i, S in enumerate(dsf):  # Use as iterator
                   Q_cubic = np.dot(dsf.Qpoints[i], np.linalg.inv(P))

                   if verbose:
                       f = frequencies[i]
                       bi_sets = degenerate_sets(f)
                       text = "%f  " % distances[i]
                       text += "%f %f %f  " % tuple(Q_cubic)
                       text += " ".join(["%f" % (f[bi].sum() / len(bi))
                                         for bi in bi_sets])
                       text += "  "
                       text += " ".join(["%f" % (S[bi].sum()) for bi in bi_sets])
                       print(text)

               if verbose:
                   print("")
                   print("")


   if __name__ == '__main__':
       phonon = load(np.diag([2, 2, 2]),
                     primitive_matrix=[[0, 0.5, 0.5],
                                       [0.5, 0, 0.5],
                                       [0.5, 0.5, 0]],
                     unitcell_filename="POSCAR",
                     force_sets_filename="FORCE_SETS",
                     born_filename="BORN")
       phonon.symmetrize_force_constants()

       # Mesh sampling calculation is needed for Debye-Waller factor
       # This must be done with is_mesh_symmetry=False and is_eigenvectors=True.
       mesh = [11, 11, 11]
       phonon.set_mesh(mesh,
                       is_mesh_symmetry=False,
                       is_eigenvectors=True)

       # Gamma-L path i FCC conventional basis
       directions_to_L = [[0.5, 0.5, 0.5],
                          [-0.5, 0.5, 0.5]]
       G_points_cubic = ([3, 3, 3], )
       n_points = 51
       temperature = 300

       print("# Distance from Gamma point, 4 band frequencies in meV, "
             "4 dynamic structure factors")
       print("# For degenerate bands, summations are made.")
       print("# Gamma point is omitted due to different number of bands.")
       print("")

       # With scattering lengths
       print("# Running with scattering lengths")
       run(phonon,
           G_points_cubic,
           directions_to_L,
           temperature,
           scattering_lengths={'Na': 3.63, 'Cl': 9.5770},
           n_points=n_points,
           verbose=True)
       print("")

       # With atomic form factor
       print("# Running with atomic form factor")
       # D. Waasmaier and A. Kirfel, Acta Cryst. A51, 416 (1995)
       # f(Q) = \sum_i a_i \exp((-b_i Q^2) + c
       # Q is in angstron^-1
       # a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
       f_params = {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                          0.767888, 0.070139, 0.995612, 14.1226457,
                          0.968249, 0.217037, 0.045300],  # 1+
                   'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                          6.524271, 19.467656, 2.355626, 60.320301,
                          35.829404, 0.000436, -34.916604]}  # 1-
       run(phonon,
           G_points_cubic,
           directions_to_L,
           temperature,
           func_AFF=get_func_AFF(f_params),
           n_points=n_points,
           verbose=True)
