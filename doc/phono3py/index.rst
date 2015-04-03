How to use phono3py
====================

Introduction
-------------

This software is used to calculate phonon-phonon interaction related
properties:

- Lattice thermal conductivity
- Phonon lifetime/linewidth
- Imaginary part of self energy at the lowest order
- Joint density of states (JDOS) and w-JDOS

The theoretical background is summarized in the paper found at
http://dx.doi.org/10.1103/PhysRevB.91.094306 or
http://arxiv.org/abs/1501.00691 .

Examples are found in ``example-phono3py`` directory.

Installation
--------------

System requirement
^^^^^^^^^^^^^^^^^^^

The following python libraries are required.

::

   python-dev python-numpy python-yaml python-h5py python-matplotlib 

``python-matplotlib`` is optional, but it is strongly recommended to
install it.  The OpenMP library is necessary for multithreding
support. The GNU OpenMP library is ``libgomp1``.  In the case of
ubuntu linux, these are installed using the package manager::

   % sudo apt-get install python-dev python-numpy python-matplotlib \
     python-yaml python-h5py libgomp1 liblapacke-dev

After the versions of Ubuntu-12.10, lapacke
(http://www.netlib.org/lapack/lapacke.html) can be installed from the
package manager (``liblapacke`` and ``liblapacke-dev``), but in older
versions of ubuntu, or in other environments, you may have to compile
lapacke by yourself. The compilation procedure is found at the lapacke
web site. After creating the lapacke library, ``liblapacke.a`` (or the
dynamic link library) ``setup3.py`` must be properly modified to link
it. As an example, the procedure of compiling lapacke is shown below.

::

   % tar xvfz lapack-3.5.0.tgz
   % cd lapack-3.5.0
   % cp make.inc.example make.inc
   % make lapackelib

Multithreading support
^^^^^^^^^^^^^^^^^^^^^^^

Phono3py supports OpenMP multithreading and most users will need it,
otherwise the calculation may take long time. However, without special
OpenMP environment variables (``-lgomp`` and ``-fopenmp`` in
``setup3.py``), phono3py will be compiled without the OpenMP
multithreding support.

Installation procedure
^^^^^^^^^^^^^^^^^^^^^^^

Download the latest version from
https://github.com/atztogo/phonopy/tags and extract it somewhere. The
version number here is not related to the version number of harmonic
(usual) phonopy. The harmonic phonopy included in this package is a
development version and can be different from that distributed at
sourceforge.net.

In the directory, open ``setup3.py`` and set the location of
lapacke. If you installed lapacke from the package manager, you can
remove the line related to lapacke. If you compiled it by yourself,
set the location of it. Then run ``setup.py`` (for harmonic phonopy)
and ``setup3.py`` (for anharmonic phonopy)::

   % python setup.py install --home=.
   % python setup3.py install --home=.

In this way to setup, ``PYTHONPATH`` has to be set so that python can
find harmonic and anharmonic phonopy libraries. If you have been
already a user of phonopy, the original phonopy version distributed at
sourceforge.net will be removed from the list of the ``PYTHONPATH``.
The ``PYTHONPATH`` setting depends on shells that you use. For example
in bash or zsh::

   export PYTHONPATH=~/phonopy-0.9.8/lib/python

or::

   export PYTHONPATH=$PYTHONPATH:~/phonopy-0.9.8/lib/python

   
   
Calculation flow
-----------------

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

How to read the results stored in hdf5 files
-----------------------------------------------

How to use HDF5 python library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is assumed that ``python-h5py`` is installed on the computer you
interactively use. In the following, how to see the contents of
``.hdf5`` files in the interactive mode of Python. Usually for running
interactive python, ``ipython`` is recommended to use but not the
plain python. In the following example, an MgO result of thermal
conductivity calculation is loaded and thermal conductivity tensor at
300 K is watched.

::
   

   In [1]: import h5py
   
   In [2]: f = h5py.File("kappa-m111111.hdf5")
   
   In [3]: f.keys()
   Out[3]:
   [u'frequency',
    u'gamma',
    u'group_velocity',
    u'heat_capacity',
    u'kappa',
    u'mode_kappa',
    u'ave_pp',
    u'qpoint',
    u'temperature',
    u'weight']
   
   In [4]: f['kappa'].shape
   Out[4]: (101, 6)
   
   In [5]: f['kappa'][:]
   Out[5]:
   array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
          [  5.86834069e+03,   5.86834069e+03,   5.86834069e+03,
             1.20936823e-15,   0.00000000e+00,  -2.05720313e-15],
          [  1.37552313e+03,   1.37552313e+03,   1.37552313e+03,
             2.81132320e-16,   0.00000000e+00,  -5.00076366e-16],
	  ...,
          [  6.56974871e+00,   6.56974871e+00,   6.56974871e+00,
             1.76632276e-18,   0.00000000e+00,  -2.30450472e-18],
          [  6.50316555e+00,   6.50316555e+00,   6.50316555e+00,
             1.74843437e-18,   0.00000000e+00,  -2.28116103e-18],
          [  6.43792061e+00,   6.43792061e+00,   6.43792061e+00,
             1.73090513e-18,   0.00000000e+00,  -2.25828616e-18]])
   
   In [6]: f['temperature'][:]
   Out[6]:
   array([    0.,    10.,    20.,    30.,    40.,    50.,    60.,    70.,
             80.,    90.,   100.,   110.,   120.,   130.,   140.,   150.,
            160.,   170.,   180.,   190.,   200.,   210.,   220.,   230.,
            240.,   250.,   260.,   270.,   280.,   290.,   300.,   310.,
            320.,   330.,   340.,   350.,   360.,   370.,   380.,   390.,
            400.,   410.,   420.,   430.,   440.,   450.,   460.,   470.,
            480.,   490.,   500.,   510.,   520.,   530.,   540.,   550.,
            560.,   570.,   580.,   590.,   600.,   610.,   620.,   630.,
            640.,   650.,   660.,   670.,   680.,   690.,   700.,   710.,
            720.,   730.,   740.,   750.,   760.,   770.,   780.,   790.,
            800.,   810.,   820.,   830.,   840.,   850.,   860.,   870.,
            880.,   890.,   900.,   910.,   920.,   930.,   940.,   950.,
            960.,   970.,   980.,   990.,  1000.])
   
   In [7]: f['kappa'][30]
   Out[7]:
   array([  2.18146513e+01,   2.18146513e+01,   2.18146513e+01,
            5.84389577e-18,   0.00000000e+00,  -7.63278476e-18])
   

Details of ``kappa-*.hdf5`` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Files name, e.g. ``kappa-m323220.hdf5``, is determined by some
specific options. ``mxxx``, show the numbers of sampling
mesh. ``sxxx`` and ``gxxx`` appear optionally. ``sxxx`` gives the
smearing width in the smearing method for Brillouin zone integration
for phonon lifetime, and ``gxxx`` denotes the grid number. Using the
command option of ``-o``, the file name can be modified slightly. For
example ``-o nac`` gives ``kappa-m323220.nac.hdf5`` to
memorize the option ``--nac`` was used.

Currently ``kappa-*.hdf5`` file (not for the specific grid points)
contains the properties shown below.

frequency
~~~~~~~~~

Phonon frequencies. The physical unit is THz (without :math:`2\pi`)

The array shape is (irreducible q-point, phonon band).

gamma
~~~~~
Imaginary part of self energy. The physical unit is THz
(without :math:`2\pi`).

The array shape for all grid-points (irreducible q-points) is
(temperature, irreducible q-point, phonon band).

The array shape for a specific grid-point is 
(temperature, phonon band).

This is read when ``--read_gamma`` option is specified.

gamma_isotope
~~~~~~~~~~~~~~

Isotope scattering of :math:`1/2\tau^\mathrm{iso}_\lambda`.
The physical unit is same as that of gamma.

The array shape is same as that of frequency.

This is NOT read even when ``--read_gamma`` option is specified.

group_velocity
~~~~~~~~~~~~~~

Phonon group velocity, :math:`\nabla_\mathbf{q}\omega_\lambda`. The
physical unit is :math:`\text{THz}\cdot\text{\AA}` (without
:math:`2\pi`).

The array shape is (irreducible q-point, phonon band, 3 = Cartesian coordinates).

heat_capacity
~~~~~~~~~~~~~

Mode-heat-capacity defined by

.. math::

    C_\lambda = k_\mathrm{B}
     \left(\frac{\hbar\omega_\lambda}{k_\mathrm{B} T} \right)^2
     \frac{\exp(\hbar\omega_\lambda/k_\mathrm{B}
     T)}{[\exp(\hbar\omega_\lambda/k_\mathrm{B} T)-1]^2}.

The physical unit is eV/K.

The array shape is (temperature, irreducible q-point, phonon band).

kappa
~~~~~

Thermal conductivity tensor. The physical unit is W/m-K.

The array shape is (temperature, 6 = (xx, yy, zz, yz, xz, xy)).

mode_kappa
~~~~~~~~~~

Thermal conductivity tensor at k-star. The physical unit is
W/m-K. Each tensor element is the sum of tensor elements on the
members of the k-star, i.e., equivalent q-points by crystallographic
point group and time reversal symmetry.

The array shape is (temperature, irreducible q-point, phonon band, 6 =
(xx, yy, zz, yz, xz, xy)).

q-point
~~~~~~~

Irreducible q-points in reduced coordinates.

The array shape is (irreducible q-point, 3 = reduced
coordinates in reciprocal space).

temperature
~~~~~~~~~~~

Temperatures where thermal conductivities are calculated. The physical
unit is K.

weight
~~~~~~

Weights corresponding to irreducible q-points. Sum of weights equals to
the number of (coarse) mesh grid points.

ave_pp
~~~~~~~

Averaged phonon-phonon interaction in :math:`\text{eV}^2`,
:math:`P_{\mathbf{q}j}`:

.. math::

   P_{\mathbf{q}j} = \frac{1}{(3n_\mathrm{a})^2} \sum_{\lambda'\lambda''}
   |\Phi_{\lambda\lambda'\lambda''}|^2.

	
Command options
----------------

Some of options are common to phonopy.

``-d``
^^^^^^^

Supercell with displacements are created. Using with ``--amplitude``
option, atomic displacement distances are controlled.

``--amplitude``
^^^^^^^^^^^^^^^^

Displacement distance. The default value is 0.03.

``--pa``, ``--primitive_axis``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transformation matrix from a non-primitive cell to the primitive
cell. See phonopy ``PRIMITIVE_AXIS`` tag (``--pa`` option) at
http://phonopy.sourceforge.net/setting-tags.html#primitive-axis

``--fc2``
^^^^^^^^^^^

Read ``fc2.hdf5``.

``--fc3``
^^^^^^^^^^

Read ``fc3.hdf5``.

``--sym_fc2``, ``--sym_fc3r``, ``--tsym``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are used to symmetrize second- and third-order force
constants. ``--sym_fc2`` and ``--sym_fc3r`` symmetrize those in real
space by the index exchange, respectively, and ``--tsym`` symmetrizes
by the translational invariance, respectively.

..
   ``--sym_fc3q`` symmetrizes third-order force constants in normal
   coordinates by the index exchange.

When those force constants are not read from the hdf5 files,
symmetrized force constants in real space are written into those hdf5
files.

``--dim``
^^^^^^^^^^

Supercell size is specified. See the
detail at http://phonopy.sourceforge.net/setting-tags.html#dim .

``--dim_fc2``
^^^^^^^^^^^^^^

Larger supercell size to calculate harmonic force constants can be
used with these options. The larger supercell size is specified by
``--dim_fc2``. When running with ``--dim_fc2`` option, a pair of
``FORCES_fC2`` and ``disp_fc2.yaml`` or ``fc2.hdf5`` has to be
prepared.

The larger supercells for fc2 in ``POSCAR`` format are created
specifying this option with ``-d`` option as
``POSCAR_FC2-xxxxx``. Simultaneously ``disp_fc2.yaml`` is created,
which is necessary to generate fc2 from ``FORCES_FC2``.

``--mesh``
^^^^^^^^^^^

Phonon triples are chosen on the grid points on the sampling mesh
specified by this option. This mesh is made along reciprocal
axes and is always Gamma-centered.

..
   ``--md``
   ^^^^^^^^^

   Divisors of mesh numbers. Another sampling mesh is used to calculate
   phonon lifetimes. :math:`8\times 8\times 8` mesh is used for the
   calculation of phonon lifetimes when it is specified, e.g.,
   ``--mesh="11 11 11" --md="2 2 2"``.

``--br``
^^^^^^^^^

Run calculation of lattice thermal conductivity tensor with the single
mode relaxation time approximation and linearized phonon Boltzmann
equation. Without specifying ``--gp`` option, thermal conductivity is
written into ``kappa-mxxxxxx.hdf5``.

``--sigma``
^^^^^^^^^^^^

:math:`\sigma` value of Gaussian function for smearing when
calculating imaginary part of self energy. See the detail at
:ref:`brillouinzone_sum`.

Multiple :math:`\sigma` values are also specified by space separated
numerical values. This is used when we want to test several
:math:`\sigma` values simultaneously.


``--thm``
^^^^^^^^^^^

Tetrahedron method is used for calculation of imaginary part of self
energy.

``--tmax``, ``--tmin``, ``--tstep``, ``--ts``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Temperatures at equal interval are specified by ``--tmax``,
``--tmin``, ``--tstep``. See phonopy ``TMAX``, ``TMIN``, ``TSTEP``
tags (``--tmax``, ``--tmin``, ``--tstep`` options) at
http://phonopy.sourceforge.net/setting-tags.html#tprop-tmin-tmax-tstep .

::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
     -c POSCAR-unitcell --br --thm --tmin=100 --tmax=1000 --tstep=50


Specific temperatures are given by ``--ts``.

::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="11 11 11" \
     -c POSCAR-unitcell --br --thm --ts="200 300 400"

``--gp``
^^^^^^^^^

Grid points where imaginary part of self energy is calculated. Indices
of grid points are specified by space separated numbers. The mapping
table between grid points to its indices is obtained by running with
``--loglevel=2`` option.

``--ga`` option can be used instead of ``--gp`` option. See ``--gp``
section.

``--ga``
^^^^^^^^^

This option is used to specify grid points like ``--gp`` option but in
the different way. For example with ``--mesh="16 16 16"``, a q-point
of (0.5, 0.5, 0.5) is given by ``--ga="8 8 8"``. The values have to be
integers. If you want to specify the point on a path, ``--ga="0 0 0 1
1 1 2 2 2 3 3 3 ..."``, where each three values are recogninzed as a
grid point. The grid points given by ``--ga`` option are translated to
grid point indices as given by ``--gp`` option, and the values given
by ``--ga`` option will not be shown in log files.

``--wgp``
^^^^^^^^^

Irreducible grid point indices are written into
``ir_grid_points.yaml``. This information may be used when we want to
calculate imaginary part of self energy at each grid point in
conjunction with ``--gp`` option. ``grid_address-mxxx.dat`` is also
written. This file contains all the grid points and their grid
addresses in integers.

``--nac``
^^^^^^^^^^

Non-analytical term correction for harmonic phonons. Like as phonopy,
``BORN`` file has to be put on the same directory.

``--q_direction``
^^^^^^^^^^^^^^^^^^

This is used with ``--nac`` to specify the direction to polarize in
reciprocal space. See the detail at
http://phonopy.sourceforge.net/setting-tags.html#q-direction .

``--isotope``
^^^^^^^^^^^^^^

Phonon-isotope scattering is calculated.. Mass variance parameters are
read from database of the natural abundance data for elements, which
refers Laeter *et al.*, Pure Appl. Chem., **75**, 683
(2003)

::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --thm --isotope

``--mass_variances`` or ``--mv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is used to include isotope effect by reading specified
mass variance parameters. For example of GaN, this may be set like
``--mv="1.97e-4 1.97e-4 0 0"``. The number of elements has to
correspond to the number of atoms in the primitive cell.

Isotope effect to thermal conductivity may be checked first running
without isotope calculation::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --thm

Then running with isotope calculation::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --thm \
     --read_gamma --mv="1.97e-4 1.97e-4 0 0"

In the result hdf5 file, currently isotope scattering strength is not
written out, i.e., ``gamma`` is still imaginary part of self energy of
ph-ph scattering.

``--boundary_mfp``, ``--bmfp``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A most simple boundary scattering treatment is
implemented. :math:`v_g/L` is just used as the scattering rate, where
:math:`v_g` is the group velocity and :math:`L` is the boundary mean
free path. The value is given in micrometre. The default value, 1
metre, is just used to avoid divergence of phonon lifetime and the
contribution to the thermal conducitivity is considered negligible.

``--cutoff_fc3`` or ``--cutoff_fc3_distance``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is used to set elements of third-order force constants
zero when any pair-distance among triplet of atoms is larger than the
cut-off distance. This option may be useful to check interaction range
of third-order force constants.

``--cutoff_pair`` or ``--cutoff_pair_distance``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is only used together with ``-d`` option. Using this
option, number of supercells with displacements is reduced and a
special ``disp_fc3.yaml`` is created.

Cut-off pair distance is used to cut-off configurations of pairs of
displacements. ``POSCAR-xxxxx`` are not created if distance between
pair of atoms to be displaced is larger than the cut-off pair
distance. The indexing of ``POSCAR-xxxxx`` files is same as the usual
case, i.e., without this option. But using this option, a lot of
indices are missing, which are not necessary to be put for creating
``FORCES_THIRD``. Only ``vasprun.xml``'s calculated for these
reduced number of ``POSCAR-xxxxx`` have to be given at ``phono3py --cf3
...``.

::

   phono3py -d --cutpair=4

After running VASP calculations,

::

   phono3py --cf3 all_calculated_vasprun_xmls

``disp_fc3.yaml`` may be readable and helpful to understand this procedure.

``--write_gamma``
^^^^^^^^^^^^^^^^^^

Imaginary part of self energy at harmonic phonon frequency
:math:`\Gamma(\omega_\lambda)` (or twice of inverse phonon lifetime)
is written into file in hdf5 format.  The result is written into
``kappa-mxxxxxx-dxxx-gxxxx-sxxx.hdf5``.

``--read_gamma``
^^^^^^^^^^^^^^^^

Imaginary part of self energy at harmonic phonon frequency
:math:`\Gamma(\omega_\lambda)` (or twice of inverse phonon lifetime)
is read from ``kappa`` file in hdf5 format.
Initially the usual result file of ``kappa-mxxxxxx-dxxx-sxxx.hdf5`` is
searched. Unless it is found, it tries to read ``kappa`` file for
each grid point, ``kappa-mxxxxxx-dxxx-gxxxx-sxxx.hdf5``.

..
   ``--write_amplitude``
   ^^^^^^^^^^^^^^^^^^^^^^^

   Interaction strengths of triplets are written into file in hdf5
   format. This file can be huge and usually it is not recommended to
   write it out.

``--ave_pp`` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Averaged phonon-phonon interaction strength (:math:`P_{\mathbf{q}j}`)
is used to calculate imaginary part of self energy. This option works
only when ``--read_gamma`` and ``--br`` options are activated where
the averaged phonon-phonon interaction that is read from
``kappa-mxxxxx.hdf5`` file is used. Therefore the averaged
phonon-phonon interaction has to be stored before using this
option. The calculation result **overwrites** ``kappa-mxxxxx.hdf5``
file. Therefore to use this option together with ``-o`` option is
strongly recommended.

First, run full conductivity calculation,

::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --thm

Then

::

   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --thm \
     --read_gamma --ave_pp -o ave_pp

``--const_ave_pp``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Averaged phonon-phonon interaction (:math:`P_{\mathbf{q}j}`) is
replaced by this constant value. Therefore third-order force constants
are not necessary to input.  The physical unit of the value is
:math:`\text{eV}^2`.

::
   
   % phono3py --dim="3 3 2" -v --mesh="32 32 20" -c POSCAR-unitcell --br --thm \
     --const_ave_pp=1e-10

The other command options
--------------------------

The ways to use and ways to output given by following command options
may change soon.

``--jdos``
^^^^^^^^^^^

Two classes of joint density of states (JDOS) are calculated. The
result is written into ``jdos-mxxxxxx-gx.dat``. The first column is
the frequency, and the second and third columns are the values given
as follows, respectively,

.. math::
   
   &D_2^{(1)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda_1,\lambda_2}
   \left[\delta(\omega+\omega_{\lambda_1}-\omega_{\lambda_2}) +
   \delta(\omega-\omega_{\lambda_1}+\omega_{\lambda_2}) \right], \\
   &D_2^{(2)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda_1,\lambda_2}\delta(\omega-\omega_{\lambda_1}
   -\omega_{\lambda_2}).

::

   % phono3py --fc2 --dim="2 2 2" -c POSCAR-unitcell --mesh="16 16 16" \
     --thm --nac --jdos --ga="0 0 0  8 8 8"

When temperatures are specified, two classes of weighted JDOS are
calculated. The result is written into ``jdos-mxxxxxx-gx-txxx.dat``,
where ``txxx`` shows the temperature. The first column is the
frequency, and the second and third columns are the values given as
follows, respectively,

.. math::

   &N_2^{(1)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda'\lambda''} \Delta(-\mathbf{q}+\mathbf{q}'+\mathbf{q}'')
   (n_{\lambda'} - n_{\lambda''}) [ \delta( \omega + \omega_{\lambda'} -
   \omega_{\lambda''}) - \delta( \omega - \omega_{\lambda'} +
   \omega_{\lambda''})], \\
   &N_2^{(2)}(\mathbf{q}, \omega) = \frac{1}{N}
   \sum_{\lambda'\lambda''} \Delta(-\mathbf{q}+\mathbf{q}'+\mathbf{q}'')
   (n_{\lambda'}+ n_{\lambda''}+1) \delta( \omega - \omega_{\lambda'} -
   \omega_{\lambda''}).

::

   % phono3py --fc2 --dim="2 2 2" -c POSCAR-unitcell --mesh="16 16 16" \
     --thm --nac --jdos --ga="0 0 0  8 8 8" --ts=300


``--bi``
^^^^^^^^^

Specify band indices. Imaginary part of self energy is calculated when
``--lw`` is not specified. The output file name is like
``gammas-mxxxxxx-gxx-bx.dat`` where ``bxbx...`` shows the band indices
used to be averaged. The calculated values at indices separated by
space are averaged, and those separated by comma are separately
calculated.

::

   % phono3py --fc3 --fc2 --dim="2 2 2" --mesh="16 16 16" \
     -c POSCAR-unitcell --thm --nac --gp="34" --bi="4 5, 6"


``--lw``
^^^^^^^^^

Linewidth calculation. The output is written to ``linewidth-mxxxx-gxx-bx.dat``.

::

   % phono3py --fc3 --fc2 --dim="2 2  2" --mesh="16 16 16" -c POSCAR-unitcell \
     --thm --nac --q_direction="1 0 0" --gp=0 --lw --bi="4 5, 6"
     

``--gruneisen``
^^^^^^^^^^^^^^^^

Mode-Gruneisen-parameters are calculated from fc3.

Mesh sampling mode::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v --mesh="16 16 16" 
     -c POSCAR-unitcell --nac --gruneisen

Band path mode::

   % phono3py --fc3 --fc2 --dim="2 2 2" -v \
     -c POSCAR-unitcell --nac --gruneisen --band="0 0 0  0 0 1/2"


Auxiliary tool
----------------

``kaccum``
^^^^^^^^^^^

**This command is under the development. The usage and file format of
the output may change in the future.**

Accumulated lattice thermal conductivity with respect to frequency is
calculated. The frequency derivative like density of states is also
calculated.

::

   % kaccum --mesh="11 11 11" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" \
     POSCAR-unitcell kappa-m111111.hdf5 |tee kaccum.dat

``--mesh`` option is mandatory and ``--pa`` option is optional. The
first and second arguments are the unit cell and ``kappa-***.hdf5`` files,
respectively. 

The format of the output is as follows: The first column gives
frequency, and the second to seventh columns give the accumulated

lattice thermal conductivity of 6 elements, xx, yy, zz, yz, xz,
xy. The eighth to 13th columns give the derivatives. There are sets of
frequencies, which are separated by blank lines. Each set is for a
temperature. There are the groups corresponding to the number of
temperatures calculated.

To plot the output by gnuplot at temperature index 30 that may
correspond to 300 K,

::

   % echo 'p "kaccum.dat" i 30 u 1:2 w l, "kaccum.dat" i 30 u 1:8 w l'|gnuplot -persist

This is the result of silicon.

.. |i0| image:: Si-kaccum.png
        :width: 50%

|i0|


Convergence check of calculation
---------------------------------

.. _brillouinzone_sum:

Brillouin zone sum
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Third-order force constants are much weaker to numerical noise of a
force calculator than second-order force constants. Therefore
supercell force calculations have to be done by enough high numerical
accuracy.

The phono3py default displacement distance is 0.01
:math:`\text{\AA}`. In some cases, accurate result may not be obtained
due to the numerical noise of the force calculator. Usually increasing
the displacement distance by ``--amplitude`` option reduces
the numerical noise, but increases error from higher order anharmonicity.

It is not easy to check the numerical quality of force constants. It
is suggested firstly to check deviation from the translational
invariance condition by watching output where the lines start with
"drift of ...". The drift value smaller than 1 may be acceptable but
of course it is dependent on cases. Most practical way may be to
compare thermal conductivities calculated with and without symmetrizing
third-order force constants (``--sym_fc3r``, ``--sym_fc2``, ``--tsym``
options).

Mode-Gruneisen-parameters calculated from third-order force constants
look very sensitive to numerical noise near the Gamma point. Therefore
symmetrization is recommended.

Change Log
-----------

Changes in version 0.9.7
^^^^^^^^^^^^^^^^^^^^^^^^^

- The definition of MSPP is modified so as to be averaged ph-ph
  interaction defined as :math:`P_{\mathbf{q}j}` in the arXiv
  manuscript. The key in the kappa hdf5 file is changed from ``mspp``
  to ``ave_pp``. The physical unit of :math:`P_{\mathbf{q}j}` is set
  to :math:`\text{eV}^2`.

Changes in version 0.9.6
^^^^^^^^^^^^^^^^^^^^^^^^^

- Silicon example is put in ``example-phono3py`` directory.
- Accumulated lattice thermal conductivity is calculated by ``kaccum``
  script.
- JDOS output format was changed.

Changes in version 0.9.5
^^^^^^^^^^^^^^^^^^^^^^^^^
- In ``kappa-xxx.hdf5`` file, ``heat_capacity`` format was changed
  from ``(irreducible q-point, temperature, phonon band)`` to
  ``(temperature, irreducible q-point, phonon band)``. For ``gamma``,
  previous document was wrong in the array shape. It is
  ``(temperature, irreducible q-point, phonon band)``


Changes in version 0.9.4
^^^^^^^^^^^^^^^^^^^^^^^^^

- The option of ``--cutoff_mfp`` is renamed to ``--boundary_mfp`` and
  now it's on the document.
- Detailed contribution of ``kappa`` at each **q**-point and phonon
  mode is output to .hdf5 with the keyword ``mode_kappa``.

Changes in version 0.8.11
^^^^^^^^^^^^^^^^^^^^^^^^^^
- A new option of ``--cutoff_mfp`` for including effective boundary
  mean free path. 
- The option name ``--cutfc3`` is changed to ``--cutoff_fc3``. 
- The option name ``--cutpair`` is changed to ``--cutoff_pair``.
- A new option ``--ga`` is created.
- Fix spectrum plot of joint dos and imaginary part of self energy

Changes in version 0.8.10
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Different supercell size of fc2 from fc3 can be specified using
  ``--dim_fc2`` option.
- ``--isotope`` option is implemented. This is used instead of
  ``--mass_variances`` option without specifying the values. Mass
  variance parameters are read from database.

Changes in version 0.8.2
^^^^^^^^^^^^^^^^^^^^^^^^^

- Phono3py python interface is rewritten and a lot of changes are
  introduced.
- ``FORCES_SECOND`` and ``FORCES_THIRD`` are no more used. Instead just
  one file of ``FORCES_FC3`` is used. Now ``FORCES_FC3`` is generated
  by ``--cf3`` option and the backward compatibility is simple: ``cat
  FORCES_SECOND FORCES_THIRD > FORCES_FC3``.
- ``--multiple_sigmas`` is removed. The same behavior is achieved by
  ``--sigma``.

Changes in version 0.8.0
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--q_direction`` didn't work. Fix it.
- Implementation of tetrahedron method whcih is activated by
  ``--thm``.
- Grid addresses are written out by ``--wgp`` option.

Changes in version 0.7.6
^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^

- Default displacement distance is changed to 0.03.
- Files names of displacement supercells now have 5 digits numbering,
  ``POSCAR-xxxxx``.
- Cutoff distance between pair displacements is implemented. This is
  triggered by ``--cutpair`` option. This option works only for
  calculating atomic forces in supercells with configurations of pairs
  of displacements.

Changes in version 0.7.1
^^^^^^^^^^^^^^^^^^^^^^^^^

- It is changed to sampling q-points in Brillouin zone. Previously
  q-points are sampled in reciprocal primitive lattice. Usually this
  change affects very little to the result.
- q-points of phonon triplets are more carefully sampled when a
  q-point is on Brillouin zone boundary. Usually this
  change affects very little to the result.
- Isotope effect to thermal conductivity is included.

Changes in version 0.6.0
^^^^^^^^^^^^^^^^^^^^^^^^^

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
