.. _workflow:

Work flow
==========

Calculation procedure
----------------------

1. Create POSCARs with displacements

   This is the same way as usual phonopy::

      % phono3py -d --dim="2 2 2" -c POSCAR-unitcell

   ``disp_fc3.yaml`` and ``POSCAR-xxxxx`` files are created.

   If you want to use larger supercell size for
   second-order force constants (fc2) calculation than that
   for third-order force constants (fc3) calculation::

      % phono3py -d --dim_fc2="4 4 4" --dim="2 2 2" -c POSCAR-unitcell

   In this case, ``disp_fc2.yaml`` and ``POSCAR_FC2-xxxxx`` files are
   also created.

2. Run VASP for supercell force calculations 

   To calculate forces on atoms in supercells, ``POSCAR-xxxxx`` (and
   ``POSCAR_FC2-xxxxx`` if they exist) are used as VASP (or any force
   calculator) calculations.

   It is supposed that each force calculation is executed under the
   directory named ``disp-xxxxx`` (and ``disp_fc2-xxxxx``), where
   ``xxxxx`` is sequential number.

3. Collect ``vasprun.xml``'s

   When VASP is used as the force calculator, force sets to calculate
   fc3 and fc2 are created as follows.

   ::

      % phono3py --cf3 disp-{00001..00755}/vasprun.xml

   where 0755 is an example of the index of the last displacement
   supercell. To perform this collection, ``disp_fc3.yaml`` created at
   step 1 is required. Then ``FORCES_FC3`` is created.

   When you use larger supercell for fc2 calculation::

      % phono3py --cf2 disp_fc2-{00001..00002}/vasprun.xml

   ``disp_fc2.yaml`` is necessary in this case and ``FORCES_FC2`` is
   created.
   
4. Create fc2.hdf and fc3.hdf

   ::

      % phono3py --dim="2 2 2" -c POSCAR-unitcell

   ``fc2.hdf5`` and ``fc3.hdf5`` are created from ``FORCES_FC3`` and
   ``disp_fc3.yaml``. This step is not mandatory, but you can avoid
   calculating fc2 and fc3 at every run time.

   When you use larger supercell for fc2 calculation::
   
      % phono3py --dim_fc2="4 4 4" --dim="2 2 2" -c POSCAR-unitcell

   Similarly ``fc2.hdf5`` and ``fc3.hdf5`` are created from ``FORCES_FC3``,
   ``FORCES_FC2``, ``disp_fc3.yaml``, and ``disp_fc2.yaml``.

5. Thermal conductivity calculation

   An example of thermal conductivity calculation is::

      % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
        -c POSCAR-unitcell --br --thm

   or with larger supercell for fc2::

      % phono3py --fc3 --fc2 --dim_fc2="4 4 4" --dim="2 2 2" -v --mesh="11 11 11" \
        -c POSCAR-unitcell --br --thm

   This calculation may take very long time. ``--thm`` invokes a
   tetrahedron method for Brillouin zone integration for phonon
   lifetime calculation. Instead, ``--sigma`` option can be used with
   the smearing widths.

   In this command, phonon lifetimes at many grid points are
   calculated in series. The phonon lifetime calculation at each grid
   point can be separately calculated since they
   are independent and no communication is necessary at the
   computation. The procedure is as follows:

   First run the same command with the addition option of ``--wgp``::

      % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
        -c POSCAR-unitcell --br --thm --wgp

   ``ir_grid_points.yaml`` is obtained. In this file, irreducible
   q-points are shown. Then distribute calculations of phonon
   lifetimes on grid points with ``--write_gamma`` option by::

      % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
        -c POSCAR-unitcell --br --thm --write_gamma --gp="[grid ponit(s)]"

   After finishing all distributed calculations, run with
   ``--read_gamma`` option::

      % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
        -c POSCAR-unitcell --br --thm --read_gamma

   Once this calculation runs without problem, separately calculated
   hdf5 files on grid points are no more necessary and may be deleted.


Convergence check in calculation
---------------------------------

.. _brillouinzone_sum:

Brillouin zone summation
~~~~~~~~~~~~~~~~~~~~~~~~~

Brillouin zone sums appear at different two points for phonon lifetime
calculation. First it is used for the Fourier transform of force
constans, and then to obtain imaginary part of phonon-self-energy.  In
the numerical calculation, uniform sampling meshes are employed for
these summations. To obtain more accurate result, it is always better
to use denser meshes. But the denser mesh requires more
computationally demanding.

The second Brillouin zone sum contains delta functions. In phono3py
calculation, a linear tetrahedron method (``--thm``) and a smearing
method (``--sigma``) can be used for this Brillouin zone
integration. Smearing parameter is used to approximate delta
functions. Small ``sigma`` value is better to describe the detailed
structure of three-phonon-space, but it requires a denser mesh to
converge.

..
   The first and second meshes have to be same or the first
   mesh is integral multiple of the second mesh, i.e., the first and
   second meshes have to overlap and the first mesh is the same as or
   denser than the second mesh.

To check the convergence with respect to the ``sigma`` value, multiple
sigma values can be set. This can be computationally efficient, since
it is avoided to re-calculate phonon-phonon interaction strength for
different ``sigma`` values in this case.

Convergence with respect to the sampling mesh and smearing parameter
strongly depends on materials. A :math:`20\times 20\times 20` sampling
mesh (or 8000 reducible sampling points) and 0.1 THz smearing value
for reciprocal of the volume of an atom may be a good starting choice.

The tetrahedron method requires no parameter such as the smearing
width, therefore it is easier to use than the smearing method and
recommended to use. A drawback of using the tetrahedron method is that
it is slower and consumes more memory space.

Numerical quality of force constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Third-order force constants are much weaker to numerical noise of a
force calculator than second-order force constants. Therefore
supercell force calculations have to be done by enough high numerical
accuracy.

The phono3py default displacement distance is 0.03
:math:`\text{\AA}`. In some cases, accurate result may not be obtained
due to the numerical noise of the force calculator. Usually increasing
the displacement distance by ``--amplitude`` option reduces
the numerical noise, but increases error from higher order anharmonicity.

It is not easy to check the numerical quality of force constants. It
is suggested firstly to check deviation from the translational
invariance condition by watching output where the output lines start
with ``max drift of ...``. The drift value smaller than 1 may be
acceptable but of course it is dependent on cases. The most practical
way may be to compare thermal conductivities calculated with and
without symmetrizing third-order force constants by ``--sym_fc3r``,
``--sym_fc2``, and ``--tsym`` options.

Mode-Gruneisen-parameters calculated from third-order force constants
look very sensitive to numerical noise near the Gamma point. Therefore
symmetrization is recommended.


|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net
