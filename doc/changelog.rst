.. _changelog:

Change Log
==========

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
* Experimental support on thermal displacement
  (:ref:`thermal_displacements_option`).
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

* ``--writefc`` option is implemented (:ref:`writefc_option`)
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
* The option ``--nomeshsym`` is available on the manual (:ref:`nomeshsym_option`).
* Symmetry finder update that includes the bug fix of Wyckoff letter
  assignment.
* Showing site-symmetry symbols with respective orientations in the output of
  ``--symmetry`` option.
* Code cleanings of ``settings.py``, ``force_constant.py``, etc.
* Starting implementation of ``character_table.py`` (:ref:`character_table_tag`).

Sep-19-2011: Version 1.1
---------------------------

* ``--readfc`` option is implemented (:ref:`readfc_option`)
* A bit of clean-up of the code ``dynamical_matrix.py``,
  ``force_constant.py`` and ``_phonopy.c`` to make implementations
  similar to the formulations often written in text books.

Sep-5-2011: Version 1.0
---------------------------

* ``settings.py`` is moved to ``phonopy/cui/Phonopy``. The configure
  parser from a file and options is modified.
* Usage of ``MODULATION`` tag was changed.
* The option ``--nosym`` is available on the manual (:ref:`nosym_option`).

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
  - LSUPER tag is removed. Please use -d option
    (:ref:`create_displacement_option`).
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
  was implemented. (:ref:`nac_option`)

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
	
.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|
