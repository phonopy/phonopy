.. _changelog:

Change Log
==========

Jan-25-2016: Version 1.10.2
----------------------------

* Python 3 support
* Many fixes
* spglib update to v1.9.0

Oct-20-2015: Version 1.10.0 (release for testing)
--------------------------------------------------

* An experimental release for testing python 3 support. Bug reports
  are very appreciated.

Oct-20-2015: Version 1.9.7 
-----------------------------

* Siesta interface (``--elk`` option) was added (:ref:`siesta_interface`)
  by Henrique Pereira Coutada Miranda.
* ``WRITE_MESH = .FALSE.`` (``--nowritemesh``) tag was added
  (:ref:`write_mesh_tag`).


Aug-12-2015: Version 1.9.6
-----------------------------

* ``--hdf5`` option. Some output files can be written in hdf5
  format. See :ref:`hdf5_option`.
* Improve tetrahedron method performance in the calculation of DOS and
  PDOS.
* Spglib update to version 1.8.2.1.


July-11-2015: Version 1.9.5
-----------------------------

* Elk interface (``--elk`` option) was added (:ref:`elk_interface`).
* Spglib update to version 1.8.1. 


Feb-18-2015: Version 1.9.4
-----------------------------

* Fixed to force setting ``MESH_SYMMETRY = .FALSE.`` (``--nomeshsym``)
  when PDOS is invoked.

Feb-10-2015: Version 1.9.3
-----------------------------

* ``MAGMOM`` tag is now available (:ref:`magmom_tag`).
* Spglib update.

Jan-4-2015: Version 1.9.2
-----------------------------

* Behaviors of ``--wien2k``, ``--abinit``, ``--pwscf`` options have
  been changed. Now they are just to invoke those calculator modes
  without a unit cell structure file. The unit cell structure file is
  specified using ``--cell`` (``-c``) option or ``CELL_FILENAME``
  tag. See :ref:`force_calculators`, :ref:`wien2k_interface`,
  :ref:`abinit_interface`, and :ref:`pwscf_interface`.
* For the ``gruneisen`` command, ``--factor``, ``--nomeshsym``,
  ``--wien2k``, ``--abinit``, and ``--pwscf`` options are
  implemented. See :ref:`gruneisen_calculators` and
  :ref:`gruneisen_command_options`.
* In phonopy-API, timing to call Phonopy._set_dynamical_matrix is
  changed to reduce the number of calls of this function. This may
  raise timing issue to phonopy-API users.
* Band-DOS (band-PDOS) plot is implemented.

Oct-30-2014: Version 1.9.1.3
-----------------------------

* Experimental support for Abinit. See :ref:`pwscf_mode` and
  :ref:`pwscf_force_sets_option`.

Oct-29-2014: Version 1.9.1.2
-----------------------------

* Experimental support for Abinit. See :ref:`abinit_mode` and
  :ref:`abinit_force_sets_option`.
* FHI-aims modulation output. Some more examples for ``phonopy-FHI-aims``.

Oct-17-2014: Version 1.9.1.1
-----------------------------

* Supercell matrix support (9 elements) for the ``MODULATION`` tag.
* Improve the speed to create supercell.
* Many minor changes to clean-up code badly written.

Aug-28-2014: Version 1.9.0
-----------------------------

* Use a native cElementTree of Python as VASP XML parser and stop
  using lxml. The native cElementTree is built in after Python 2.5. So 
  Python 2.4 or before will not be suppored from this phonopy
  version. This migration to cElementTree was made by shyuep.

Aug-12-2014: Version 1.8.5
-----------------------------

* Supercell creation behavior may change due to the change of
  algorithm. Though it used its own independent routine, now it uses
  the ``Supercell`` class to build supercell
* Spglib update (1.6.1-alpha)
* Experimental option ``--fc_computation_algorithm="regression"`` to
  compute force constants from forces and displacements using linear
  regression with displaying standard deviation, by KL(m).

June-23-2014: Version 1.8.4.2
-----------------------------

* Symmetrization of Born effective charge of ``outcar-born`` is
  improved.
* ``-v`` option shows Born effective charges and dielectric constants
  when NAC is set.
* Bug fix to include cutoff frequency to the displacement distance
  matrix.
* Yaml output formats for band, mesh, q-points, modulation modes were
  slightly modified.
* Bug fix in finding equivalent atoms in supercell that has lower
  symmetry than the original unit cell.

Apr-5-2014: Version 1.8.4.1
-----------------------------

* Fix irreps for non-zero q-point of nonsymmorphic case

Mar-31-2014: Version 1.8.4
---------------------------

* Implementation of analytical derivative of dynamical matrix in C,
  which improves the performance of group velocity calculation.
* Minor change of python module for group velocity.

Feb-17-2014: Version 1.8.3
---------------------------

* A collection of small fixes in interface.
* Spglib update (spglib-1.6.0)
* Change in ``phonopy/file_IO/write_FORCE_SETS_*``.

Feb-8-2014: Version 1.8.2
---------------------------

* ``vasprun.xml`` of VASP 5.2.8 is only specially parsed to treat
  special wrong character.
* Python module interface is updated and modified. ``set_post_process``,
  ``set_force_sets`` will be obsolete. ``set_displacements`` is
  obsolete.

Jan-9-2014: Version 1.8.0
---------------------------

This version is dangerous. A lot of code is modified internally.

* Tetrahedron method is implemented for total and partial DOS.
  This is activated by ``--thm`` option.
* The display output with ``-v`` option is enriched.
* Symmetrization for ``outcar-born`` is implemented (Experimental).
* Cutoff-frequency ``CUTOFF_FREQUENCY`` (``--cutoff_freq``) is
  implemented to ignore frequencies lower than this cutoff frequency
  to calculate thermal properties and thermal displacements.

Dec-4-2013: Version 1.7.5
---------------------------

* ``--mass`` option is created to shortcut of the ``MASS`` tag.
* ``--fc_spg_symmetry`` option is created to symmetrize force
  constants.
* Symmetry finder update (spglib version 1.5.2)

Oct-3-2013: Version 1.7.4
---------------------------

* Thermal displacement matrix is implemented. See
  :ref:`thermal_displacement_matrices_tag` and :ref:`thermal_displacement`.
* PDOS with projection along arbitrary direction was implemented. See
  :ref:`projection_direction_tag`. 
* ``partial_dos.dat`` format was changed. XYZ projected PDOS is not
  output. Instead atom projected PDOS (sum of XYZ projected PDOS)
  is written. See :ref:`output_files`.
* DOS and PDOS python interface was modified. The keyword of
  ``omega_something`` is changed to ``freq_something``.
* ``gruneisen`` didn't run because it didn't follow the move of
  the ``file_IO.py`` file location. This is fixed.
* The formula of non-analytical term correction implemented in phonopy
  is not translational invariant in reciprocal space. This induces
  tiny difference of the choice of equivalent q-points being different
  by reciprocal primitive vectors. Now in the mesh sampling mode
  (``MP``), q-points are automatically moved to inside
  first-Brillouin-zone.
* In the mesh sampling mode, consistency of symmetry of mesh numbers
  to crystal symmetry is checked. If the symmetry disagrees with
  crystal symmetry, mesh symmetrization (equivalent to ``MESH_SYMMETRY
  = .FALSE.``) is disabled.
* Wien2k interface is updated to adapt Wien2k-13.
* Fix the problem that only Vinet EOS worked in phonopy-qha.

Sep-17-2013: Version 1.7.3
---------------------------

* Fix. Segmentation fault happens in some specific systems
  (e.g. Kubuntu 12.04 32bit) due to a different behavior of numpy
  array creation.
* Group velocity for degenerate phonon mode is calculated slightly
  different from older version and now it is symmetrized by
  site-symmetry of q-point.

Aug-4-2013: Version 1.7.2
---------------------------

* ``group_velocity/__init__.py`` is moved to ``phonon`` directory.
* ``hphonopy/file_IO.py`` is moved to top directory.
* New ``harmonic/derivative_dynmat.py``: Analytical derivatives of
  dynamical matrix
* Group velocity is computed by analytical derivatives of dynamical
  matrix in the default configuration instead of previous finite
  difference method. Group velocity calculation with the finite
  difference method can be still activated by ``--gv_delta_q`` option.
* Force constants solver was partially rewritten. The order and shape
  of matrices in the formula is rearranged
  (:ref:`force_constants_solver_theory`).

July-14-2013: Version 1.7.1
---------------------------

* ``--pdos`` option was created. This is same as ``PDOS`` tag.
* Group velocity with degenerate modes was improved.

Jun-21-2013: Version 1.7
---------------------------

* The tag ``CHARACTER_TABLE`` was renamed to ``IRREPS``
  (:ref:`irreps_tag`), and the option of ``--ct`` was renamed to
  ``--irreps`` as well. To show Ir-representations along with
  characters, ``SHOW_IRREPS`` tag (or ``--show_irreps`` option) is
  used. The output file name was also renamed to ``irreps.yaml``. In
  the ir-reps calculation, display and file outputs were modified to
  show the arguments of complex value characters.
* Numpy array types of 'double' and 'intc' for those arrays
  passed to numpy C-API are used.
* ``thermal_displacement.py`` is slightly modified for the preparation
  to include thermal displacement matrix.
* Symmetry finder update (spglib 1.4.2).

Apr-13-2013: Version 1.6.4
---------------------------

* Group velocity can be calculated using ``GROUP_VELOCITY`` tag or
  ``--gv`` option (:ref:`group_velocity_tag`).
* Non-analytical term correction is implemented in C, which
  accelerates the calculation speed.

Feb-7-2013: Version 1.6.3
----------------------------

* Arbitral projection direction is allowed for thermal displacements
  calculation. (:ref:`thermal_displacements_tag`)
* A new tag `WRITEDM` and an option `--writedm` are
  implemented. Dynamical matrices are written into ``qpoints.yaml``
  when this is used togather with the ``QPOINTS`` mode. (:ref:`writedm_tag`)

Nov-13-2012: Version 1.6.2
----------------------------

* A small fix of FHIaims.py.

Nov-4-2012: Version 1.6.1
----------------------------

* Implementation of database of character table for another type
  of point group -3m.
* A new option ``--irreps`` or ``IRREPS`` tag (Experimental).
* ``character_table.yaml`` output.
* Eigenvectors output in``modulation.yaml`` was recovered.


Oct-22-2012: Version 1.6
----------------------------

* Experimental support of band connection. (:ref:`band_connection_tag`)
* Experimental support of mode Grüneisen parameter calculation. (:ref:`phonopy_gruneisen`)
* Format of ``MODULATION`` tag was modified. (:ref:`modulation_tag`)
* Phonopy is controlled by command line options more than
  before. ``--qpoints``, ``--modulation`` and ``--anime`` options are prepared.
* Symmetry finder update.
* Implementation of database of character table for the point
  group 32. Fix -3m database.

June-29-2012: Version 1.5
-------------------------

* Bug fix on plotting PDOS with labels.
* The array structures of qpoints, distances, frequencies, eigenvalues,
  eigenvectors in BandStructure are changed to the lists of those
  values of segments of band paths. For qpoints, frequencies,
  eigenvalues, eigenvectors, the previous array structures are
  recovered by numpy.vstack and for distances, numpy.hstack.
* Experimental support on thermal displacement.
* Experimental support on fitting DOS to a Debye model
  (:ref:`debye_model_tag`) implemented by Jörg Meyer.

May-22-2012: Version 1.4.2
---------------------------

* Bug fix on showing the values of thermal properties. No bug in plot
  and yaml.

May-21-2012: Version 1.4.1
---------------------------

* Avoid list comprehension with else statement, because it is not
  supported in old python versions.

May-13-2012: Version 1.4
---------------------------

* ``--writefc`` option is implemented.
* In using ``MODULATION`` tag, phase factor for each mode can be
  specified as the third value of each mode in degrees.
* Arguments of ``get_modulation`` in Phonopy module were modified.
  The phase factor is now included in ``phonon_modes``.
* Class ``Phonopy`` was refactored. All private variables were renamed
  as those starting with an underscore. Some basic variables are
  obtained with the same variable names without the underscode, which
  was implemented by the function ``property``.
* The labels of segments of band structure plot are specified by
  ``BAND_LABELS`` (:ref:`band_labels_tag`).
* ``--band`` option is implemented.
* ``GAMMA_CENTER`` tag and ``--gc``, ``--gamma_center`` option are
  implemented (:ref:`mp_tag`).
* ``phonopy-qha`` was polished. Most of the code was moved to
  ``phonopy/qha/__init__.py``.
* ``Phonopy::get_mesh`` and ``Phonopy::get_band_structure`` were
  modified. Instead of eigenvalues, frequencies are returned.
* The order of return values of ``Phonopy::get_thermal_properties``
  was changed as numpy arrays of temperatures, Helmhotlz free
  energies, entropies, and heat capacities at constant volume.
* Arguments of the class ``ThermalProperties``, ``Dos``, and
  ``PartialDOS`` were changed. Instead of eigenvalues, frequencies are
  used.
* The default sigma value used for total and partial DOS was changed
  to (max_frequency - min_frequency) / 100.
* Symmetry finder update.

Mar-20-2012: Version 1.3
---------------------------

* C implementations of a few parts of ``force_constants.py`` to speed
  up.
* spglib update.
* Many small modifications.
* License is changed to the new BSD from the LGPL.

Oct-13-2011: Version 1.2.1
---------------------------

* Bug fix of the option ``--dim`` with 9 elements.

Oct-12-2011: Version 1.2
---------------------------

* Closing support of the ``--nac_old`` option.
* The option ``--nomeshsym`` is available on the manual.
* Symmetry finder update that includes the bug fix of Wyckoff letter
  assignment.
* Showing site-symmetry symbols with respective orientations in the output of
  ``--symmetry`` option.
* Code cleanings of ``settings.py``, ``force_constant.py``, etc.
* Starting implementation of ``character_table.py`` (:ref:`irreps_tag`).

Sep-19-2011: Version 1.1
---------------------------

* ``--readfc`` option is implemented.
* A bit of clean-up of the code ``dynamical_matrix.py``,
  ``force_constant.py`` and ``_phonopy.c`` to make implementations
  similar to the formulations often written in text books.

Sep-5-2011: Version 1.0
---------------------------

* ``settings.py`` is moved to ``phonopy/cui/Phonopy``. The configure
  parser from a file and options is modified.
* Usage of ``MODULATION`` tag was changed.
* The option ``--nosym`` is available on the manual.

Aug-8-2011: Version 0.9.6
---------------------------

* Symmetry finder update
* Wyckoff positions are shown with ``--symmetry`` option

Jun-7-2011: Version 0.9.5.1
----------------------------------

* Bug fix of ``get_surrounding_frame`` in ``cells.py`` by Jörg Meyer and Christian Carbogno.

Errata of document
-----------------------------

The cell matrix definition of ``Atoms`` class was transposed.

Jun-3-2011: Version 0.9.5
----------------------------------

* Wien2k interface is updated (:ref:`wien2k_interface`), but this is
  still quite experimental support.
* More information is involved in ``disp.yaml``. Along this
  modification, supercells with displacements can be created solely
  from ``disp.yaml`` using ``dispmanager``.
* Instead of ``TRANSLATION`` tag, ``FC_SYMMETRY`` is created
  (:ref:`fc_symmetry_tag`).
* Closing support of ``--fco`` option.
* Add a few more examples in the ``example`` directory.
* Symmetry finder update
* ``propplot`` is updated for the ``--gnuplot`` option.

Errata of document
-----------------------------

The example of ``FORCE_SETS`` was wrong and was fixed. The explanation
of the document is correct.

Apr-18-2011: Version 0.9.4.2
-----------------------------

* In the setting tag ``BAND``, now comma ``,`` can be used to
  disconnect the sequence of band paths
  (:ref:`band_structure_related_tags`).

* ``dispmanager``, an auxiliary tool for modifying ``disp.yaml``, is
  developed (:ref:`dispmanager_tool`).

* Symmetry finder update to spglib-1.0.3.1. Almost perfect casting to
  a Bravais lattice is achieved using ``--symmetry`` option.

* The setting tags ``TRANSLATION``, ``PERMUTATION``, and ``MP_REDUCE``
  are ceased.


Feb-26-2011: Version 0.9.4.1
-----------------------------

* Wien2k interface bug fix

Feb-20-2011: Version 0.9.4
-----------------------------

* Big phonopy-interface change was imposed. Some of filenames and
  formats of input and output files are modified. **There is no
  default setting filename** like ``INPHON`` (setting file is passed
  as the first argument). Some of tag names and those usage are also
  modified. Please first check :ref:`examples_link` for the new usage.

  List of changes:

  - Setting file has to be passed to phonopy as the first argunment.
  - FORCES is replaced by FORCE_SETS (:ref:`file_forces`).
  - DISP is replaced by disp.yaml.
  - LSUPER tag is removed. Please use -d option.
  - NDIM and MATDIM tags are replaced by DIM tag (:ref:`dimension_tag`).
  - Band structure setting tags are changed to BAND tag
    (:ref:`band_structure_related_tags`).
  - DOS tag is renamed to DOS_RANGE tag (:ref:`dos_related_tags`).
  
  These changes are applied only for the phonopy interface. Internal
  simulation code has not been touched, so **physical results would not
  be affected**. If you have any questions, please send e-mail to
  phonopy :ref:`mailinglist`.

* ``phonopy-FHI-aims`` had not worked in some of previous
  versions. Now it works by Jörg Meyer and Christian Carbogno.
  
* Directory structure of the code was changed.

* Symmetry finder update to spglib-1.0.2

* [**Experimental**]  Finding Bravais lattice using
  ``--symmetry`` option.

* [**Experimental**] Modulated structure along specified phonon modes
  by ``MODULATION`` tag (:ref:`modulation_tag`).

Jan-21-2011: Version 0.9.3.3
-----------------------------

* Animation file output update (:ref:`animation_tag`). The ``ANIME``
  tag format was changed.

Jan-12-2011: Version 0.9.3.2
-----------------------------

* ``phonopy-qha`` is updated. A few options are added
  (:ref:`phonopy_qha_options`). Calculation under pressure is
  supported by ``--pressure`` option.

* Primitive cell search and Bravais lattice output are integrated into
  the symmetry search with ``--symmetry`` option.

Errata of document
-----------------------------

* There were mistakes in the documents for the ``PRIMITIVE_AXIS`` and
  ``MATDIM``. The 9 values are read from the first three to
  the last three as respective rows of the matrices defined.

Dec-30-2010: Version 0.9.3.1
-----------------------------

* Bug fix of ``-f`` option.
* The output filenames of ``phonopy-qha`` are modified and summarized
  at :ref:`phonopy_qha_output_files`.

Dec-5-2010: Version 0.9.3
------------------------------------

* The license is changed to LGPL.
* ``MASS`` tag is recreated (:ref:`mass_tag`).
* ``--mp`` option is created. This works like the ``MP`` tag.
* Improvement of ``phonopy-qha`` both in the code and :ref:`manual
  <phonopy_qha>`.
* The bug in ``--fco`` option was fixed.

Nov-26-2010: Version 0.9.2
------------------------------------

* spglib update (ver. 1.0.0)
* ASE.py is removed. Compatible class and functions, Atoms,
  write_vasp, and read_vasp, are implemented.
* A ``vasprun.xml`` parser wrapper is implemened to avoid the broken
  ``PRECFOCK`` in vasprun.xml of VASP 5.2.8.

Sep-22-2010: Version 0.9.1.4
------------------------------------

* The new tag ``ANIME_TYPE`` supports ``xyz`` and ``xyz_jmol`` formats
  by Jörg Meyer and Christian Carbogno, and also `A set of ``POSCAR``
  files corresponding to animation frames.

* Fix bugs in ``trim_cell`` and
  ``Primitive.__supercell_to_primitive_map`` in ``cells.py``. When
  :math:`M_s^{-1}M_p` is not symmetric, the supercell was not created
  correctly.
 
* ``phonopy-FHI-aims`` update by jm.


Aug-24-2010: Version 0.9.1.3
------------------------------------

* Update symmetry finder of spglib. Now precision is in Cartesian distance.

* The animation output for ``arc`` didn't work. Now it works.

* Qpoint mode didn't work with bugs. Now it works.

* ``--vasp`` option is renamed to ``--cell`` or ``-c``.

* The new options ``--symmetry``, ``--displacement`` or ``-d``,
  ``--dim``, ``--primitive_axis`` are implemented.

* The option ``--ndim`` is replaced with ``--dim`` with ``-d`` option.

June-10-2010: Version 0.9.1.2
------------------------------------

* The code on non-analytical term correction is included in the
  ``DynamicalMatrix`` class. Data sets read by ``parse_BORN`` are set
  by ``set_non_analytical_term`` and gotten by
  ``get_non_analytical_term``. The q-vector direction (only direction
  is used in the non-analytical term correction) is set by
  ``set_q_non_analytical_term``. However for emprical damping
  function, some distance is used, i.e., when a q-point is getting
  away, non-analytical term is weaken. For this purpose, the second
  argument of ``set_q_non_analytical_term`` is used.

  At the same time, a small problem on the previous implementation was
  found. When a reduced q-point is out of the first Brillouin zone, 
  it is not correctly handled. Currently it is fixed so as that when
  absolute values of elements of the reduced q-point are over 0.5, they
  are reduced into -0.5 < q < 0.5.


  [**Attention**] The previous ``--nac`` option is moved to
  ``--nac_old``. ``--nac`` is used for different method of the
  non-analytical term correction at general q-points. This will be
  documented soon.

* Bug fix on ``write_FORCES`` in ``file_IO.py``. When order of
  displacements in ``DISP`` file is not ascending order of atom indices,
  it was not correctly re-ordered. Because the default order of
  phonopy is ascending order, usually there is no problem for the most
  users.

* ``phonopy-FHI-aims``

  - adapted to extensions of dynamical_matrix with respect to
    non-analytical corrections
  - added support for animation infrastructure
  - moved several options to control.in

  by Jörg Meyer and Christian Carbogno

May-11-2010: Version 0.9.1.1
------------------------------------

* ``phonopy-FHI-aims`` adapted to split of dos array into the two
  seperate omega, dos arrays in TotalDOS class by Jörg Meyer.

May-10-2010: Version 0.9.1
------------------------------------

* The methods of get_partial_DOS and get_total_DOS are added to the
  Phonopy class.
  
Apr-12-2010: Version 0.9.0.2
------------------------------------

* spglib bug was fixed. If the crystal structure has non-standard origin,
  the translation was not correctly handled. This problem happened
  after version 0.9.0.

Apr-12-2010: Version 0.9.0.1
------------------------------------

* spglib update

Apr-10-2010: Version 0.9.0
------------------------------------

* Phonopy module (``__init.py__``) is heavily revised and the script
  ``phonopy`` is rewritten using the phonopy module.  Therefore there
  may be bugs. Be careful. Document of the phonopy module will be
  updated gradually.
* A small Wien2k interface document is added (:ref:`wien2k_interface`).
* A script ``phonopy-FHI-aims`` and its examples are added by
  Jörg Meyer. 
* spglib update
  

Mar-10-2010: Version 0.7.4
------------------------------------

* spglib update
* Animation mode (:ref:`animation_tag`)

Feb-10-2010: Version 0.7.3
------------------------------------

* Bug fix for Wien2k mode

Jan-12-2010: Version 0.7.2
------------------------------------
* [**Experimental**] Non-analytical term correction
  was implemented.

Dec-8-2009: Version 0.7.1 released
------------------------------------

* :ref:`auxiliary_tools` ``propplot`` is added.
* Memory consumption is reduced when using ``-f`` option to handle
  large vasprun.xml files.

Nov-24-2009: Version 0.7.0 released
------------------------------------

* :ref:`auxiliary_tools` ``bandplot`` and ``pdosplot`` are prepared.
* Formats of `band.yaml`, `mesh.yaml`, and `qpoints.yaml` are slightly
  modified.
* There was bug in ``PERMUTATION`` tag to calculate symmetrized force
  constants. Now it is fixed. Usually this is not necessary to set
  because this does not affect to result.
* Symmetry finder spglib is updated.
* ``PM`` tag is implemented. See :ref:`setting_tags`. Behaviors in
  the previous versions are ``PM = AUTO``.

Oct-14-2009: Version 0.6.2 released
------------------------------------

* Installation process was changed slightly.
  See :ref:`install`.
* The command ``phonopy`` is stored in the ``bin``
  directory. ``phonopy.py`` is renamed to ``phonopy``.
* setup system is improved by Maxim V. Losev.
* ``--fz`` tag was implemented experimentally. This is supposed to
  enable to subtract residual forces on atoms in equilibrium structure
  from those in structure with atomic displacements.
	
