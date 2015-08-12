.. _changelog:

Change Log
==========

Aug-12-2015: version 0.9.12
------------------------------

- Spglib update to version 1.8.2.1.
- Improve computational performance of ``kaccum`` and ``gaccum``.

Jun-18-2015: version 0.9.10.1
------------------------------

- Bug fix of ``gcaccum``

Jun-17-2015: version 0.9.10
----------------------------

- Fix bug in ``kaccum``. When using with ``--pa`` option, irreducible
  q-points were incorrectly indexed.
- ``gaccum`` is implemented. ``gaccum`` is very similar to ``kaccum``,
  but for :math:`\Gamma_\lambda(\omega_\lambda)`.
- spglib update.

Changes in version 0.9.7
-------------------------

- The definition of MSPP is modified so as to be averaged ph-ph
  interaction defined as :math:`P_{\mathbf{q}j}` in the arXiv
  manuscript. The key in the kappa hdf5 file is changed from ``mspp``
  to ``ave_pp``. The physical unit of :math:`P_{\mathbf{q}j}` is set
  to :math:`\text{eV}^2`.

Changes in version 0.9.6
------------------------

- Silicon example is put in ``example-phono3py`` directory.
- Accumulated lattice thermal conductivity is calculated by ``kaccum``
  script.
- JDOS output format was changed.

Changes in version 0.9.5
-------------------------

- In ``kappa-xxx.hdf5`` file, ``heat_capacity`` format was changed
  from ``(irreducible q-point, temperature, phonon band)`` to
  ``(temperature, irreducible q-point, phonon band)``. For ``gamma``,
  previous document was wrong in the array shape. It is
  ``(temperature, irreducible q-point, phonon band)``


Changes in version 0.9.4
------------------------

- The option of ``--cutoff_mfp`` is renamed to ``--boundary_mfp`` and
  now it's on the document.
- Detailed contribution of ``kappa`` at each **q**-point and phonon
  mode is output to .hdf5 with the keyword ``mode_kappa``.

Changes in version 0.8.11
-------------------------

- A new option of ``--cutoff_mfp`` for including effective boundary
  mean free path. 
- The option name ``--cutfc3`` is changed to ``--cutoff_fc3``. 
- The option name ``--cutpair`` is changed to ``--cutoff_pair``.
- A new option ``--ga`` is created.
- Fix spectrum plot of joint dos and imaginary part of self energy

Changes in version 0.8.10
-------------------------

- Different supercell size of fc2 from fc3 can be specified using
  ``--dim_fc2`` option.
- ``--isotope`` option is implemented. This is used instead of
  ``--mass_variances`` option without specifying the values. Mass
  variance parameters are read from database.

Changes in version 0.8.2
------------------------

- Phono3py python interface is rewritten and a lot of changes are
  introduced.
- ``FORCES_SECOND`` and ``FORCES_THIRD`` are no more used. Instead just
  one file of ``FORCES_FC3`` is used. Now ``FORCES_FC3`` is generated
  by ``--cf3`` option and the backward compatibility is simple: ``cat
  FORCES_SECOND FORCES_THIRD > FORCES_FC3``.
- ``--multiple_sigmas`` is removed. The same behavior is achieved by
  ``--sigma``.

Changes in version 0.8.0
------------------------

- ``--q_direction`` didn't work. Fix it.
- Implementation of tetrahedron method whcih is activated by
  ``--thm``.
- Grid addresses are written out by ``--wgp`` option.

Changes in version 0.7.6
------------------------

- Cut-off distance for fc3 is implemented. This is activated by
  ``--cutfc3`` option. FC3 elements where any atomic pair has larger
  distance than cut-off distance are set zero.
- ``--cutpair`` works only when creating displacements. The cut-off
  pair distance is written into ``disp_fc3.yaml`` and FC3 is created
  from ``FORCES_THIRD`` with this information. Usually sets of pair
  displacements are more redundant than that needed for creating fc3
  if index permutation symmetry is considered. Therefore using index
  permutation symmetry, some elements of fc3 can be recovered even if
  some of supercell force calculations are missing. In paticular, all
  pair distances among triplet atoms are larger than cutoff pair
  distance, any fc3 elements are not recovered, i.e., the element will
  be zero.

Changes in version 0.7.2
------------------------

- Default displacement distance is changed to 0.03.
- Files names of displacement supercells now have 5 digits numbering,
  ``POSCAR-xxxxx``.
- Cutoff distance between pair displacements is implemented. This is
  triggered by ``--cutpair`` option. This option works only for
  calculating atomic forces in supercells with configurations of pairs
  of displacements.

Changes in version 0.7.1
------------------------

- It is changed to sampling q-points in Brillouin zone. Previously
  q-points are sampled in reciprocal primitive lattice. Usually this
  change affects very little to the result.
- q-points of phonon triplets are more carefully sampled when a
  q-point is on Brillouin zone boundary. Usually this
  change affects very little to the result.
- Isotope effect to thermal conductivity is included.

Changes in version 0.6.0
------------------------

- ``disp.yaml`` is renamed to ``disp_fc3.yaml``. Old calculations with
  ``disp.yaml`` can be used without any problem just by changing the
  file name.
- Group velocity is calculated from analytical derivative of dynamical
  matrix.
- Group velocities at degenerate phonon modes are better handled.
  This improves the accuracy of group velocity and thus for thermal
  conductivity.
- Re-implementation of third-order force constants calculation from
  supercell forces, which makes the calculation much faster
- When any phonon of triplets can be on the Brillouin zone boundary, i.e.,
  when a mesh number is an even number, it is more carefully treated.


|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net
  
