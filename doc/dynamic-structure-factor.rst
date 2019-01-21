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
   from phonopy.units import THzToEv


   def get_AFF_func(f_params):
       def func(symbol, Q):
           return atomic_form_factor_WK1995(Q, f_params[symbol])
       return func


   def run(phonon,
           Qpoints,
           temperature,
           atomic_form_factor_func=None,
           scattering_lengths=None):
       # Transformation to the Q-points in reciprocal primitive basis vectors
       Q_prim = np.dot(Qpoints, phonon.primitive_matrix)
       # Q_prim must be passed to the phonopy dynamical structure factor code.
       phonon.run_dynamic_structure_factor(
           Q_prim,
           temperature,
           atomic_form_factor_func=atomic_form_factor_func,
           scattering_lengths=scattering_lengths,
           freq_min=1e-3)
       dsf = phonon.dynamic_structure_factor
       q_cartesian = np.dot(dsf.qpoints,
                            np.linalg.inv(phonon.primitive.get_cell()).T)
       distances = np.sqrt((q_cartesian ** 2).sum(axis=1))

       print("# [1] Distance from Gamma point,")
       print("# [2-4] Q-points in cubic reciprocal space, ")
       print("# [5-8] 4 band frequencies in meV (becaues of degeneracy), ")
       print("# [9-12] 4 dynamic structure factors.")
       print("# For degenerate bands, dynamic structure factors are summed.")
       print("")

       # Use as iterator
       for Q, d, f, S in zip(Qpoints, distances, dsf.frequencies,
                             dsf.dynamic_structure_factors):
           bi_sets = degenerate_sets(f)  # to treat for band degeneracy
           text = "%f  " % d
           text += "%f %f %f  " % tuple(Q)
           text += " ".join(["%f" % (f[bi].sum() * THzToEv * 1000 / len(bi))
                             for bi in bi_sets])
           text += "  "
           text += " ".join(["%f" % (S[bi].sum()) for bi in bi_sets])
           print(text)


   if __name__ == '__main__':
       phonon = load(supercell_matrix=[2, 2, 2],
                     primitive_matrix='F',
                     unitcell_filename="POSCAR")

       # Q-points in reduced coordinates wrt cubic reciprocal space
       Qpoints = [[2.970000, -2.970000, 2.970000],
                  [2.950000, 2.950000, -2.950000],
                  [2.930000, -2.930000, 2.930000],
                  [2.905000, -2.905000, 2.905000],
                  [2.895000, -2.895000, 2.895000],
                  [2.880000, -2.880000, 2.880000],
                  [2.850000, -2.850000, 2.850000],
                  [2.810000, -2.810000, 2.810000],
                  [2.735000, -2.735000, 2.735000],
                  [2.660000, -2.660000, 2.660000],
                  [2.580000, -2.580000, 2.580000],
                  [2.500000, -2.500000, 2.500000]]

       # Mesh sampling phonon calculation is needed for Debye-Waller factor.
       # This must be done with is_mesh_symmetry=False and with_eigenvectors=True.
       mesh = [11, 11, 11]
       phonon.run_mesh(mesh,
                       is_mesh_symmetry=False,
                       with_eigenvectors=True)
       temperature = 300

       IXS = True

       if IXS:
           # For IXS, atomic form factor is needed and given as a function as
           # a parameter.
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
           AFF_func = get_AFF_func(f_params)
           run(phonon,
               Qpoints,
               temperature,
               atomic_form_factor_func=AFF_func)
       else:
           # For INS, scattering length has to be given.
           # The following values is obtained at (Coh b)
           # https://www.nist.gov/ncnr/neutron-scattering-lengths-list
           run(phonon,
               Qpoints,
               temperature,
               scattering_lengths={'Na': 3.63, 'Cl': 9.5770})

The output of the script is::

   # [1] Distance from Gamma point,
   # [2-4] Q-points in cubic reciprocal space,
   # [5-8] 4 band frequencies in meV (becaues of degeneracy),
   # [9-12] 4 dynamic structure factors.
   # For degenerate bands, dynamic structure factors are summed.

   0.009132  2.970000 -2.970000 2.970000  0.977517 1.648183 19.035705 30.535702  0.000000 706.475380 0.000000 16.137386
   0.015219  2.950000 2.950000 -2.950000  1.640522 2.747087 18.994893 30.479078  0.000000 262.113412 0.000000 16.366740
   0.021307  2.930000 -2.930000 2.930000  2.298710 3.841226 18.935185 30.395654  0.000000 138.116831 0.000000 16.619581
   0.028917  2.905000 -2.905000 2.905000  3.116160 5.200214 18.836546 30.256295  0.000000 78.225945 0.000000 16.965983
   0.031961  2.895000 -2.895000 2.895000  3.441401 5.740457 18.790421 30.190463  0.000000 65.174556 0.000000 17.112970
   0.036526  2.880000 -2.880000 2.880000  3.927209 6.546550 18.714922 30.081791  0.000000 51.288627 0.000000 17.341206
   0.045658  2.850000 -2.850000 2.850000  4.890522 8.140492 18.544488 29.832346  0.000000 34.845699 0.000000 17.819327
   0.057833  2.810000 -2.810000 2.810000  6.154512 10.217882 18.286153 29.444246  0.000000 23.864605 0.000000 18.476926
   0.080662  2.735000 -2.735000 2.735000  8.440068 13.902951 17.731951 28.589254  0.000000 15.762830 0.000000 19.591803
   0.103491  2.660000 -2.660000 2.660000  10.559231 17.071210 17.175478 27.602958  0.000000 0.000000 14.349345 20.000676
   0.127842  2.580000 -2.580000 2.580000  12.497611 16.203093 19.926659 26.474218  0.000000 0.000000 18.814845 17.496644
   0.152193  2.500000 -2.500000 2.500000  13.534679 15.548262 21.156819 25.813428  0.000000 0.000000 34.134746 6.765951
