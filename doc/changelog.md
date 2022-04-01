(changelog)=

# Change Log

## Feb-13-2022: Version 2.13.1

- Bug fix of thermal property calculation (PR #184).

## Feb-12-2022: Version 2.13.0

- Dorp python 3.6 support, and dependencies of numpy and matplotlib versions are
  updated:

  - Python >= 3.7
  - numpy >= 1.15.0
  - matplotlib >= 2.2.2

## Oct-28-2021: Version 2.12.0

- Matplotlib built-in mathtext replaced LaTeX renderer with `text.usetex` to
  avoid requiring latex installation.
- Aiming modernizing phonopy code, required python version and package versions
  were changed to

  - Python >= 3.6
  - numpy >= 1.11
  - matplotlib >= 2.0

- For developers, flake8, black, pydocstyle, and isort were introduced. See
  `REAEME.md` and `.pre-commit-config.yaml`.

## Jul-8-2021: Version 2.11.0

- Maintenance release. C-methods were changed to use dense shortest vectors
  (`Primitive.get_smallest_vectors()`) format. But the front end still uses
  current format `shape=(size_super, size_prim, 27, 3)`.

## Jun-10-2021: Version 2.10.0

- Fix the contents of `entropy-volume.dat` and `Cv-volume.dat` in QHA were
  swapped. (Issue #144 by @prnvrvs)
- Implementation of writing `helmholtz-volume_fitted.dat` in QHA. (PR #149 by
  @kgmat)

## Mar-17-2021: Version 2.9.3

- Fix `MANIFEST.in` to provide necessary files to compile.

## Mar-17-2021: Version 2.9.2

- Fix a bug to make the initial symmetry search slow.

## Jan-29-2021: Version 2.9.1

- Release for making pypi wheel of py3.9

## Jan-28-2021: Version 2.9.0

- {ref}`Fleur interface <Fleur_interface>` was made by @neukirchen-212.
- Improvement of `phonopy-bandplot` by @kgmat (PR#131, #132, #134)

## Sep-29-2020: Version 2.8.1

- Fix vasprun.xml expat parser (issues #129)

## Sep-22-2020: Version 2.8.0

- {ref}`CASTEP interface <castep_interface>` was made by @ladyteam.
- It was made to give more information in `gruneisen.yaml` (@ab5424, PR#127).
- Fix and update of `phonopy-bandplot` (@kgmat PR#125, commit 816a12a9)

## July-31-2020: Version 2.7.1

- Release for pypi and conda packaging

## July-22-2020: Version 2.7.0

- Spglib was removed from phonopy source code tree. Now phonopy depends on
  spglib. So spglib has to be installed separately. But for normal cases, it is
  handled by the package manager.
- A new way of using phonopy from command line is proposed at
  {ref}`phonopy_load_command`.
- Castep interface was added by @ladyteam.

## May-3-2020: Version 2.6.1

- Release for pypi and conda packaging

## May-2-2020: Version 2.6.0

- Collection of minor fixes and internal updates

## Mar-29-2020: Version 2.5.0

- New options, `--include-*`, to write summary in `phonopy.yaml`. See
  {ref}`summary_tag`.
- FHI-aims interface (`--aims`) was created by Florian Knoop.
- `phonopy-gruneisen` and `--symmetry` option were updated to be able to handle
  most of build-in calculator interfaces.
- Update spglib version to v1.14.1.
- `phonopy-dispmanager` was removed.
- Let new force calculator interface be implemented easier by centralizing those
  interface related stuffs in `phonopy.interface.calculator`.

## Dec-22-2019: Version 2.4.2

- Collection of minor updates: adding docstrings, refactoring group velocity
  code, and updating examples.

## Nov-16-2019: Version 2.4.1

- Fix a bug on Phonopy.save reported by PR#104

## Nov-13-2019: Version 2.4.0

- CP2K interface is provided by Tiziano Müller.
- Many minor bug fixes and updates.

## Sep-1-2019: Version 2.3.2

- It was forgotten to increment version number at v2.3.1. Now it is made to be
  v2.3.2 just for incrementing the number.

## Aug-29-2019: Version 2.3.1

- Fix alm wrapper to follow the lastest basis vector matrix form.

## Aug-29-2019: Version 2.3.0

- New setting tag {ref}`fc_calculator_tag` was implemented. With this, an
  external force constants calculator can be used. Currently only ALM
  (`FC_CALCULATOR = alm` or `--alm` option) is supported. By using ALM, we can
  compute force constants with any number of displacements in each supercell.
  Any number of supercells with displacements can be also handled with it if
  ordinary least square fitting can be solved for force constants.
  {ref}`random_displacements_tag` and {ref}`random_seed_tag` were implemented to
  use with ALM.
- For the external force constants calculator, new file format of `FORCE_SETS`
  was introduced. See {ref}`file_forces_type_2` `FORCE_SETS` format.

## Jul-21-2019: Version 2.2.0

- Collection of minor updates.
- Spglib update to v1.13.0.

## Jun-19-2019: Version 2.1.4

- [Bug fix](https://github.com/atztogo/phonopy/pull/96) of
  `Cp-temperature_polyfit.dat` calculation in QHA (by `@ezanardi`).

## Apr-17-2019: Version 2.1.3

- TURBOMOLE interface is provided by Antti Karttunen (`--turbomole`).

## Mar-24-2019: Version 2.1.2

- `PDOS = AUTO` (`--pdos="auto"`) was implemented.

## Feb-27-2019: Version 2.1.1

- User interface bug fix release.

## Feb-26-2019: Version 2.1

- Spglib update to v1.12.1.
- (Experimental) `phonopy` command accepts `phonopy.yaml` type file as an input
  crystal structure by `-c` option. When `DIM` and any structure file are not
  given, `phonopy_disp.yaml` (primary) or `phonopy.yaml` (secondary) is searched
  in the current directory. Then `phonopy.yaml` type file is used as the input,
  semi-automatic phonopy mode is invoked, which means:

  (1) `supercell_matrix` in the `phonopy.yaml` type file is used if it exists.
  (2) `primitive_matrix` in the `phonopy.yaml` type file is used if it exists.
  Otherwise, set `PRIMITIVE_AXES = AUTO` when `PRIMITIVE_AXES` is not given. (3)
  NAC params are read (`NAC = .TRUE.`) if NAC params are contained (primary) in
  the `phonopy.yaml` type file or if `BORN` file exists in the current directory
  (secondary). (4) Forces and displacements are read from the `phonopy.yaml`
  type file if those exist instead of reading `FORCE_SETS` in the current
  directory. (5) Calculator name (such as `qe`) is read if it is contained in
  the `phonopy.yaml` type file.

  Possible usages are:

  - For PhononDB at Kyoto-U
    (http://phonondb.mtl.kyoto-u.ac.jp/ph20180417/index.html) raw data, phonons
    are easily calculated, e.g., by
    `% phonopy -c phonon.yaml --band auto --mesh 100 -p`.
  - If `phonopy_params.yaml` is created using API of `phonopy.save()`
    ({ref}`phonopy_save_parameters`), phonopy's essential data may be easily
    passed to other people only by this file.
  - `phonopy_disp.yaml` is used instead of calculator option and input structure
    file. For example `--qe -c NaCl.in` is replaced by `-c phonopy_disp.yaml`.

## Jan-16-2019: Version 2.0

- `disp.yaml` is replaced by `phonopy_disp.yaml`. For the backward
  compatibility, when `phonopy_disp.yaml` is not found, `disp.yaml` is used.
- New calculator interface for DFTB+ written by Ben Hourahine.
- Minor update of the look of band structure plot. The format in `band.yaml` for
  treating labels was changed.
- `MESH` accepts a length value, which works in the same way as VASP automatic
  k-mesh generation (see {ref}`mp_tag`).
- For plotting DOS, it is changed to choose linear tetrahedron method as
  default, but not smearing method.
- Output file name of projected DOS was renamed from `partial_dos.dat` to
  `projected_dos.dat`.

### API change at version 2.0

- `Phonopy.get_band_structure()` is deprecated. Instead use
  `Phonopy.get_band_structure_dict()`.
- `Phonopy.get_mesh()` is deprecated. Instead use `Phonopy.get_mesh_dict()`.
- `Phonopy.set_band_structure()` is deprecated. Instead use
  `Phonopy.run_band_structure()` where `is_eigenvectors` keyword argument is
  replaced by `with_eigenvectors`.
- `Phonopy.set_mesh()` is deprecated. Instead use `Phonopy.run_mesh()` where
  `is_eigenvectors` keyword argument is replaced by `with_eigenvectors`.
- Previous behaviour of `Phonopy.run_mesh()` is achieved by
  `phonopy.mesh.run()`.
- `Phonopy.set_qpoints_phonon()` is deprecated. Use `Phonopy.run_qpoints()`
  where `is_eigenvectors` keyword argument is replaced by `with_eigenvectors`.
- `Phonopy.get_qpoints_phonon()` is deprecated. Instead use
  `Phonopy.get_qpoints_dict()`.
- `Phonopy.get_group_velocity()` is deprecated. Use
  `Phonopy.mode.group_velocities` attribute or
  `Phonopy.get_*mode*_dict()['group_velocities']`, where `*mode*` is
  `band_structure`, `mesh`, or `qpoints`.
- `Phonopy.get_group_velocities_on_bands()` is deprecated.
- `Phonopy.get_mesh_grid_info()` is deprecated.
- `Phonopy.set_iter_mesh()` is deprecated. Use `Phonopy.mesh()` with
  `use_iter_mesh=True`.
- `Phonopy.itermesh` was removed. IterMesh instance is stored in phonopy.mesh.
- `Phonopy.set_group_velocity()` is deprecated. No need to call. `gv_delta_q`
  can be set at `Phonopy.__init__()`.
- `Phonopy.set_unitcell()` is deprecated.
- `Phonopy.set_total_DOS()` is deprecated. Use `Phonopy.run_total_dos()`.
- `Phonopy.get_total_DOS()` is deprecated. Use `Phonopy.get_total_dos_dict()`.
- `Phonopy.write_total_DOS()` is deprecated. Use `Phonopy.write_total_dos()`.
- `Phonopy.plot_total_DOS()` is deprecated. Use `Phonopy.plot_total_dos()`.
- `Phonopy.set_partial_DOS()` is deprecated. Use `Phonopy.run_projected_dos()`.
- `Phonopy.get_partial_DOS()` is deprecated. Use
  `Phonopy.get_projected_dos_dict()`.
- `Phonopy.write_partial_DOS()` is deprecated. Use
  `Phonopy.write_projected_dos()`.
- `Phonopy.plot_partial_DOS()` is deprecated. Use
  `Phonopy.plot_projected_dos()`.
- `Phonopy.partial_dos` attribute is deprecated. Use `Phonopy.projected_dos`
  attribute.
- `Phonopy.set_thermal_properties()` is deprecated. Use
  `Phonopy.run_thermal_properties()`.
- `Phonopy.get_thermal_properties()` is deprecated. Use
  `Phonopy.get_thermal_properties_dict()`.
- `Phonopy.set_thermal_displacements()` is deprecated. Use
  `Phonopy.run_thermal_displacements()`.
- `Phonopy.get_thermal_displacements()` is deprecated. Use
  `Phonopy.get_thermal_displacements_dict()`.
- `Phonopy.set_thermal_displacement_matrices()` is deprecated. Use
  `Phonopy.run_thermal_displacement_matrices()`.
- `Phonopy.get_thermal_displacement_matrices()` is deprecated. Use
  `Phonopy.get_thermal_displacements_matrices_dict()`.
- New `Phonopy.auto_total_dos()`.
- New `Phonopy.auto_partial_dos()`.

## Nov-22-2018: Version 1.14.2

- This is the release for preparing features for future and is not well tested.
- Code refactoring to ease the development of calculator interface. Most of
  calculator interface dependent codes are moved to
  `phonopy.interface.__init__.py`.
- For eary adaptors, two experimental features were made:

  - Convenient Phonopy instance loader and crystal structure yaml parser
    (`phonopy.load()` and `phonopy.read_cell_yaml()`).
  - Automatic band paths generation using SeeKpath
    (https://seekpath.readthedocs.io/) (`BAND = AUTO`). Installation of SeekPath
    is needed.

## Nov-17-2018: Version 1.14.0

- From this version, a trial to follow Semantic Versioning starts.
- Automatic determination of transformation matrix from the input unit cell to
  the primitive is implemented (`PRIMITIVE_AXES = AUTO` or `--pa='auto'`).
- Spglib update to v1.11.1.
- Experimental support for dynamical structure factor
  ({ref}`dynamic_structure_factor`).
- Experimental support in phonopy-QHA for temperature dependent energy input
  ({ref}`--efe <phonopy_qha_efe_option>` option) from a file. To create this
  input file for a simple electron free energy was made for VASP
  (`phonopy-vasp-efe`).

## Jun-20-2018: Version 1.13.2

- `FULL_FORCE_CONSTANTS` tag is created ({ref}`full_force_constants_tag`).
- Minor update of script to use QE's `q2r.x` output ({ref}`this <qe_q2r>`).
- Updates and fixes of CRYSTAL and SIESTA interfaces.
- Updates of labels of irreducible representations of crystallographic point
  groups.

## Apr-21-2018: Version 1.13.0

- Non-analytical term correction by Gonze _et al._ ({ref}`reference_dp_dp_NAC`)
  is implemented and now this is the default NAC method. The former default
  method by Wang _et al._ ({ref}`reference_wang_NAC`) can be invoked by using
  `NAC_METHOD` tag ({ref}`nac_method_tag`).

## Jan-31-2018: Version 1.12.6

- Force constants file formats of `FORCE_CONSTANTS` and `force_constants.hdf5`
  are extended to support smaller size force constants. Backward compatibility
  is preserved. See {ref}`file_force_constants`. To write out force constants,
  the compact format is chosen as the default for usual cases.
- Performance improvement of phonopy and spglib by Michael Lamparski which is
  effective especially for larger unit cell systems.

## Jan-7-2018: Version 1.12.4

- For thermal displacements (and its matrices), `FMIN` (`--fmin`) and `FMAX`
  (`--fmax`) can be used to limit phonons included to the summation as the
  minimum and maximum frequencies, respectively. Instead, `CUTOFF_FREQUENCY`
  (`--cutoff-freq`) does not work for thermal displacements.
- The way of symmetrization for translation invariance is modified. As a result,
  `FC_SYMMETRY` became a Boolean, i.e., `FC_SYMMETRY = .TRUE.`
  (`--fc-symmetry`), and no need to specify a number.
- Experimental support to parse Quantum ESPRESSO `q2r.x` output from python
  script.

## Nov-8-2017: Version 1.12.2

- Command option parser of the phonopy tools is replaced from `optparse` to
  `argparse`.
- The behaviours of `-f`, `--fz`, and `--fc` options are slightly changed. The
  filenames used with these options were the positional arguments previously.
  Now they are the command-line arguments, i.e., filenames have to be put just
  after the option name like `-f vasprun.xml-001 vasprun.xml-002 ...`.
- New tags (options), `FC_FORMAT` (`--fc-format`), `READFC_FORMAT`
  (`read-fc-format`), `WRITEFC_FORMAT` (`write-fc-format`), `BAND_FORMAT`
  (`--band-format`), `MESH_FORMAT` (`--mesh-format`), and `QPOINTS_FORMAT`
  (`--qpoints-format`) were implemented. `HDF5` tag is deprecated.
- New tags `READ_FORCE_CONSTANTS` and `WRITE_FORCE_CONSTANTS` were made. They
  are equivalent to existing options of `--readfc` and `--writefc`. Using them,
  reading and writing force constants are separably controlled.

## Oct-19-2017: Version 1.12.0

- The names of auxiliary tools and `gruneisen` are changed, for which the prefix
  `phonopy-` is added to the old names to avoid accidental conflict with other
  filenames already existing under `bin`. `outcar-born` is renamed to
  `phonopy-vasp-born`. Similarly `gruneisen` is renamed `phonopy-gruneisen`.
  Please find these changes at {ref}`auxiliary_tools` and
  {ref}`phonopy_gruneisen`.

## Oct-2-2017: Version 1.11.14

- 6/m and 1 point groups are added for irreps dataset.
- `band.hdf5` is output instead of `band.yaml` when using `--hdf5` option
  together.
- Spglib update to v1.9.10. By this, symmetry search for supercells with large
  number of dimensions may become significantly faster.
- It is changed so that `mesh.yaml` or `mesh.hdf5` is not written out in thermal
  displacements calculations (`TDISP`, `TDISPMAT`, `TDISPMAT_CIF`). This is done
  to reduce the memory consumption of this calculation with dense mesh sampling.
- And many minor updates.

## June-18-2017: Version 1.11.12

- Maintenance release with many minor fixes after v1.11.10.
- -1,and -3 point groups are added for irreps dataset.
- {ref}`pretend_real_tags` was made.
- `--vasprunxml` option for `outcar-born`

## Mar-31-2017: Version 1.11.10

- Maintenance release with many fixes.

## Feb-7-2017: Version 1.11.8

- CRYSTAL code interface ({ref}`crystal_mode`) is added by Antti Karttunen.
- Different vasprun.xml (expat) parser is under testing.

## Dec-14-2016: Version 1.11.6

- `--fz` option comes back. See {ref}`fz_force_sets_option`.
- spglib update to v1.9.9

## Oct-23-2016: Version 1.11.2

- `TDISPMAT_CIF` (`--tdm_cif`) for CIF output for thermal displacement is added
  ({ref}`thermal_displacement_cif_tag`).
- spglib update to v1.9.7

## Aug-29-2016: Version 1.11.0

- `FPITCH` (`--fpicth`) is made ({ref}`dos_fmin_fmax_tags`).
- Minor update of `gruneisen`.
- Tentatively `phonopy.yaml` and `phonopy_disp.yaml` are written when running
  phonopy.
- In Phonopy-API, from this version, to create displacements in supercells
  internally the phonopy object, the `generate_displacements` method has to be
  called explicitly along with the `distance` argument. See
  {ref}`phonopy_module`.

## Jul-17-2016: Version 1.10.10

- {ref}`dos_moment_tag` tags are implemented to calculate phonon moments.
- `qpoints.hdf5` is written with the `--hdf5` option. Dynamical matrices are
  also stored in `qpoints.hdf5` with `WRITEDM = .TRUE.` (`--writedm` option).

## Apr-22-2016: Version 1.10.8

- {ref}`xyz_projection_tag` tag is created for PDOS.
- {ref}`vasp_mode` option is created to explicitly show VASP is used to generate
  `band.yaml` as the calculator.
- spglib update to v1.9.2

## Feb-7-2016: Version 1.10.4

- More keywords are written in `band.yaml`.
- Default NAC unit conversion factors ({ref}`born_file`)
- Collection of many minor fixes and updates

## Jan-25-2016: Version 1.10.2

- Python 3 support
- Many fixes
- spglib update to v1.9.0

## Oct-20-2015: Version 1.10.0 (release for testing)

- An experimental release for testing python 3 support. Bug reports are very
  appreciated.

## Oct-20-2015: Version 1.9.7

- Siesta interface (`--elk` option) was added ({ref}`siesta_interface`) by
  Henrique Pereira Coutada Miranda.
- `WRITE_MESH = .FALSE.` (`--nowritemesh`) tag was added
  ({ref}`write_mesh_tag`).

## Aug-12-2015: Version 1.9.6

- `--hdf5` option. Some output files can be written in hdf5 format. See
  {ref}`hdf5_tag`.
- Improve tetrahedron method performance in the calculation of DOS and PDOS.
- Spglib update to version 1.8.2.1.

## July-11-2015: Version 1.9.5

- Elk interface (`--elk` option) was added ({ref}`elk_interface`).
- Spglib update to version 1.8.1.

## Feb-18-2015: Version 1.9.4

- Fixed to force setting `MESH_SYMMETRY = .FALSE.` (`--nomeshsym`) when PDOS is
  invoked.

## Feb-10-2015: Version 1.9.3

- `MAGMOM` tag is now available ({ref}`magmom_tag`).
- Spglib update.

## Jan-4-2015: Version 1.9.2

- Behaviors of `--wien2k`, `--abinit`, `--pwscf` options have been changed. Now
  they are just to invoke those calculator modes without a unit cell structure
  file. The unit cell structure file is specified using `--cell` (`-c`) option
  or `CELL_FILENAME` tag. See {ref}`force_calculators`, {ref}`wien2k_interface`,
  {ref}`abinit_interface`, and {ref}`qe_interface`.
- For the `gruneisen` command, `--factor`, `--nomeshsym`, `--wien2k`,
  `--abinit`, and `--pwscf` options are implemented. See
  {ref}`gruneisen_calculators` and {ref}`gruneisen_command_options`.
- In phonopy-API, timing to call `Phonopy.set_dynamical_matrix` is changed to
  reduce the number of calls of this function. This may raise timing issue to
  phonopy-API users.
- Band-DOS (band-PDOS) plot is implemented.

## Oct-30-2014: Version 1.9.1.3

- Experimental support for Abinit. See {ref}`qe_mode` and
  {ref}`qe_force_sets_option`.

## Oct-29-2014: Version 1.9.1.2

- Experimental support for Abinit. See {ref}`abinit_mode` and
  {ref}`abinit_force_sets_option`.
- FHI-aims modulation output. Some more examples for `phonopy-FHI-aims`.

## Oct-17-2014: Version 1.9.1.1

- Supercell matrix support (9 elements) for the `MODULATION` tag.
- Improve the speed to create supercell.
- Many minor changes to clean-up code badly written.

## Aug-28-2014: Version 1.9.0

- Use a native cElementTree of Python as VASP XML parser and stop using lxml.
  The native cElementTree is built in after Python 2.5. So Python 2.4 or before
  will not be supported from this phonopy version. This migration to
  cElementTree was made by shyuep.

## Aug-12-2014: Version 1.8.5

- Supercell creation behavior may change due to the change of algorithm. Though
  it used its own independent routine, now it uses the `Supercell` class to
  build supercell
- Spglib update (1.6.1-alpha)
- Experimental option `--fc_computation_algorithm="regression"` to compute force
  constants from forces and displacements using linear regression with
  displaying standard deviation, by KL(m).

## June-23-2014: Version 1.8.4.2

- Symmetrization of Born effective charge of `outcar-born` is improved.
- `-v` option shows Born effective charges and dielectric constants when NAC is
  set.
- Bug fix to include cutoff frequency to the displacement distance matrix.
- Yaml output formats for band, mesh, q-points, modulation modes were slightly
  modified.
- Bug fix in finding equivalent atoms in supercell that has lower symmetry than
  the original unit cell.

## Apr-5-2014: Version 1.8.4.1

- Fix irreps for non-zero q-point of nonsymmorphic case

## Mar-31-2014: Version 1.8.4

- Implementation of analytical derivative of dynamical matrix in C, which
  improves the performance of group velocity calculation.
- Minor change of python module for group velocity.

## Feb-17-2014: Version 1.8.3

- A collection of small fixes in interface.
- Spglib update (spglib-1.6.0)
- Change in `phonopy/file_IO/write_FORCE_SETS_*`.

## Feb-8-2014: Version 1.8.2

- `vasprun.xml` of VASP 5.2.8 is only specially parsed to treat special wrong
  character.
- Python module interface is updated and modified. `set_post_process`,
  `set_force_sets` will be obsolete. `set_displacements` is obsolete.

## Jan-9-2014: Version 1.8.0

This version is dangerous. A lot of code is modified internally.

- Tetrahedron method is implemented for total and partial DOS. This is activated
  by `--thm` option.
- The display output with `-v` option is enriched.
- Symmetrization for `outcar-born` is implemented (Experimental).
- Cutoff-frequency `CUTOFF_FREQUENCY` (`--cutoff_freq`) is implemented to ignore
  frequencies lower than this cutoff frequency to calculate thermal properties
  and thermal displacements.

## Dec-4-2013: Version 1.7.5

- `--mass` option is created to shortcut of the `MASS` tag.
- `--fc_spg_symmetry` option is created to symmetrize force constants.
- Symmetry finder update (spglib version 1.5.2)

## Oct-3-2013: Version 1.7.4

- Thermal displacement matrix is implemented. See
  {ref}`thermal_displacement_matrices_tag` and {ref}`thermal_displacement`.
- PDOS with projection along arbitrary direction was implemented. See
  {ref}`projection_direction_tag`.
- `partial_dos.dat` format was changed. XYZ projected PDOS is not output.
  Instead atom projected PDOS (sum of XYZ projected PDOS) is written. See
  {ref}`output_files`.
- DOS and PDOS python interface was modified. The keyword of `omega_something`
  is changed to `freq_something`.
- `gruneisen` didn't run because it didn't follow the move of the `file_IO.py`
  file location. This is fixed.
- The formula of non-analytical term correction implemented in phonopy is not
  translational invariant in reciprocal space. This induces tiny difference of
  the choice of equivalent q-points being different by reciprocal primitive
  vectors. Now in the mesh sampling mode (`MP`), q-points are automatically
  moved to inside first-Brillouin-zone.
- In the mesh sampling mode, consistency of symmetry of mesh numbers to crystal
  symmetry is checked. If the symmetry disagrees with crystal symmetry, mesh
  symmetrization (equivalent to `MESH_SYMMETRY = .FALSE.`) is disabled.
- Wien2k interface is updated to adapt Wien2k-13.
- Fix the problem that only Vinet EOS worked in phonopy-qha.

## Sep-17-2013: Version 1.7.3

- Fix. Segmentation fault happens in some specific systems (e.g. Kubuntu 12.04
  32bit) due to a different behavior of numpy array creation.
- Group velocity for degenerate phonon mode is calculated slightly different
  from older version and now it is symmetrized by site-symmetry of q-point.

## Aug-4-2013: Version 1.7.2

- `group_velocity/__init__.py` is moved to `phonon` directory.
- `hphonopy/file_IO.py` is moved to top directory.
- New `harmonic/derivative_dynmat.py`: Analytical derivatives of dynamical
  matrix
- Group velocity is computed by analytical derivatives of dynamical matrix in
  the default configuration instead of previous finite difference method. Group
  velocity calculation with the finite difference method can be still activated
  by `--gv_delta_q` option.
- Force constants solver was partially rewritten. The order and shape of
  matrices in the formula is rearranged ({ref}`force_constants_solver_theory`).

## July-14-2013: Version 1.7.1

- `--pdos` option was created. This is same as `PDOS` tag.
- Group velocity with degenerate modes was improved.

## Jun-21-2013: Version 1.7

- The tag `CHARACTER_TABLE` was renamed to `IRREPS` ({ref}`irreps_tag`), and the
  option of `--ct` was renamed to `--irreps` as well. To show Ir-representations
  along with characters, `SHOW_IRREPS` tag (or `--show_irreps` option) is used.
  The output file name was also renamed to `irreps.yaml`. In the ir-reps
  calculation, display and file outputs were modified to show the arguments of
  complex value characters.
- Numpy array types of 'double' and 'intc' for those arrays passed to numpy
  C-API are used.
- `thermal_displacement.py` is slightly modified for the preparation to include
  thermal displacement matrix.
- Symmetry finder update (spglib 1.4.2).

## Apr-13-2013: Version 1.6.4

- Group velocity can be calculated using `GROUP_VELOCITY` tag or `--gv` option
  ({ref}`group_velocity_tag`).
- Non-analytical term correction is implemented in C, which accelerates the
  calculation speed.

## Feb-7-2013: Version 1.6.3

- Arbitral projection direction is allowed for thermal displacements
  calculation. ({ref}`thermal_displacements_tag`)
- A new tag `WRITEDM` and an option `--writedm` are implemented. Dynamical
  matrices are written into `qpoints.yaml` when this is used together with the
  `QPOINTS` mode. ({ref}`writedm_tag`)

## Nov-13-2012: Version 1.6.2

- A small fix of FHIaims.py.

## Nov-4-2012: Version 1.6.1

- Implementation of database of character table for another type of point group
  -3m.
- A new option `--irreps` or `IRREPS` tag (Experimental).
- `character_table.yaml` output.
- Eigenvectors output in`modulation.yaml` was recovered.

## Oct-22-2012: Version 1.6

- Experimental support of band connection. ({ref}`band_connection_tag`)
- Experimental support of mode Grüneisen parameter calculation.
  ({ref}`phonopy_gruneisen`)
- Format of `MODULATION` tag was modified. ({ref}`modulation_tag`)
- Phonopy is controlled by command line options more than before. `--qpoints`,
  `--modulation` and `--anime` options are prepared.
- Symmetry finder update.
- Implementation of database of character table for the point group 32. Fix -3m
  database.

## June-29-2012: Version 1.5

- Bug fix on plotting PDOS with labels.
- The array structures of qpoints, distances, frequencies, eigenvalues,
  eigenvectors in BandStructure are changed to the lists of those values of
  segments of band paths. For qpoints, frequencies, eigenvalues, eigenvectors,
  the previous array structures are recovered by numpy.vstack and for distances,
  numpy.hstack.
- Experimental support on thermal displacement.
- Experimental support on fitting DOS to a Debye model ({ref}`debye_model_tag`)
  implemented by Jörg Meyer.

## May-22-2012: Version 1.4.2

- Bug fix on showing the values of thermal properties. No bug in plot and yaml.

## May-21-2012: Version 1.4.1

- Avoid list comprehension with else statement, because it is not supported in
  old python versions.

## May-13-2012: Version 1.4

- `--writefc` option is implemented.
- In using `MODULATION` tag, phase factor for each mode can be specified as the
  third value of each mode in degrees.
- Arguments of `get_modulation` in Phonopy module were modified. The phase
  factor is now included in `phonon_modes`.
- Class `Phonopy` was refactored. All private variables were renamed as those
  starting with an underscore. Some basic variables are obtained with the same
  variable names without the underscore, which was implemented by the function
  `property`.
- The labels of segments of band structure plot are specified by `BAND_LABELS`
  ({ref}`band_labels_tag`).
- `--band` option is implemented.
- `GAMMA_CENTER` tag and `--gc`, `--gamma_center` option are implemented
  ({ref}`mp_tag`).
- `phonopy-qha` was polished. Most of the code was moved to
  `phonopy/qha/__init__.py`.
- `Phonopy::get_mesh` and `Phonopy::get_band_structure` were modified. Instead
  of eigenvalues, frequencies are returned.
- The order of return values of `Phonopy::get_thermal_properties` was changed as
  numpy arrays of temperatures, Helmhotlz free energies, entropies, and heat
  capacities at constant volume.
- Arguments of the class `ThermalProperties`, `Dos`, and `PartialDOS` were
  changed. Instead of eigenvalues, frequencies are used.
- The default sigma value used for total and partial DOS was changed to
  (max_frequency - min_frequency) / 100.
- Symmetry finder update.

## Mar-20-2012: Version 1.3

- C implementations of a few parts of `force_constants.py` to speed up.
- spglib update.
- Many small modifications.
- License is changed to the new BSD from the LGPL.

## Oct-13-2011: Version 1.2.1

- Bug fix of the option `--dim` with 9 elements.

## Oct-12-2011: Version 1.2

- Closing support of the `--nac_old` option.
- The option `--nomeshsym` is available on the manual.
- Symmetry finder update that includes the bug fix of Wyckoff letter assignment.
- Showing site-symmetry symbols with respective orientations in the output of
  `--symmetry` option.
- Code cleanings of `settings.py`, `force_constant.py`, etc.
- Starting implementation of `character_table.py` ({ref}`irreps_tag`).

## Sep-19-2011: Version 1.1

- `--readfc` option is implemented.
- A bit of clean-up of the code `dynamical_matrix.py`, `force_constant.py` and
  `_phonopy.c` to make implementations similar to the formulations often written
  in text books.

## Sep-5-2011: Version 1.0

- `settings.py` is moved to `phonopy/cui/Phonopy`. The configure parser from a
  file and options is modified.
- Usage of `MODULATION` tag was changed.
- The option `--nosym` is available on the manual.

## Aug-8-2011: Version 0.9.6

- Symmetry finder update
- Wyckoff positions are shown with `--symmetry` option

## Jun-7-2011: Version 0.9.5.1

- Bug fix of `get_surrounding_frame` in `cells.py` by Jörg Meyer and Christian
  Carbogno.

## Errata of document

The cell matrix definition of `Atoms` class was transposed.

## Jun-3-2011: Version 0.9.5

- Wien2k interface is updated ({ref}`wien2k_interface`), but this is still quite
  experimental support.
- More information is involved in `disp.yaml`. Along this modification,
  supercells with displacements can be created solely from `disp.yaml` using
  `dispmanager`.
- Instead of `TRANSLATION` tag, `FC_SYMMETRY` is created
  ({ref}`fc_symmetry_tag`).
- Closing support of `--fco` option.
- Add a few more examples in the `example` directory.
- Symmetry finder update
- `propplot` is updated for the `--gnuplot` option.

## Errata of document

The example of `FORCE_SETS` was wrong and was fixed. The explanation of the
document is correct.

## Apr-18-2011: Version 0.9.4.2

- In the setting tag `BAND`, now comma `,` can be used to disconnect the
  sequence of band paths ({ref}`band_structure_related_tags`).

- `dispmanager`, an auxiliary tool for modifying `disp.yaml`, is developed.

- Symmetry finder update to spglib-1.0.3.1. Almost perfect casting to a Bravais
  lattice is achieved using `--symmetry` option.

- The setting tags `TRANSLATION`, `PERMUTATION`, and `MP_REDUCE` are ceased.

## Feb-26-2011: Version 0.9.4.1

- Wien2k interface bug fix

## Feb-20-2011: Version 0.9.4

- Big phonopy-interface change was imposed. Some of filenames and formats of
  input and output files are modified. **There is no default setting filename**
  like `INPHON` (setting file is passed as the first argument). Some of tag
  names and those usage are also modified. Please first check
  {ref}`examples_link` for the new usage.

  List of changes:

  - Setting file has to be passed to phonopy as the first argument.
  - FORCES is replaced by FORCE_SETS ({ref}`file_forces`).
  - DISP is replaced by disp.yaml.
  - LSUPER tag is removed. Please use -d option.
  - NDIM and MATDIM tags are replaced by DIM tag ({ref}`dimension_tag`).
  - Band structure setting tags are changed to BAND tag
    ({ref}`band_structure_related_tags`).
  - DOS tag is renamed to DOS_RANGE tag ({ref}`dos_related_tags`).

  These changes are applied only for the phonopy interface. Internal simulation
  code has not been touched, so **physical results would not be affected**. If
  you have any questions, please send e-mail to phonopy {ref}`mailinglist`.

- `phonopy-FHI-aims` had not worked in some of previous versions. Now it works
  by Jörg Meyer and Christian Carbogno.

- Directory structure of the code was changed.

- Symmetry finder update to spglib-1.0.2

- [**Experimental**] Finding Bravais lattice using `--symmetry` option.

- [**Experimental**] Modulated structure along specified phonon modes by
  `MODULATION` tag ({ref}`modulation_tag`).

## Jan-21-2011: Version 0.9.3.3

- Animation file output update ({ref}`animation_tag`). The `ANIME` tag format
  was changed.

## Jan-12-2011: Version 0.9.3.2

- `phonopy-qha` is updated. A few options are added
  ({ref}`phonopy_qha_options`). Calculation under pressure is supported by
  `--pressure` option.

- Primitive cell search and Bravais lattice output are integrated into the
  symmetry search with `--symmetry` option.

## Errata of document

- There were mistakes in the documents for the `PRIMITIVE_AXIS` and `MATDIM`.
  The 9 values are read from the first three to the last three as respective
  rows of the matrices defined.

## Dec-30-2010: Version 0.9.3.1

- Bug fix of `-f` option.
- The output filenames of `phonopy-qha` are modified and summarized at
  {ref}`phonopy_qha_output_files`.

## Dec-5-2010: Version 0.9.3

- The license is changed to LGPL.
- `MASS` tag is recreated ({ref}`mass_tag`).
- `--mp` option is created. This works like the `MP` tag.
- Improvement of `phonopy-qha` both in the code and {ref}`manual <phonopy_qha>`.
- The bug in `--fco` option was fixed.

## Nov-26-2010: Version 0.9.2

- spglib update (ver. 1.0.0)
- ASE.py is removed. Compatible class and functions, Atoms, write_vasp, and
  read_vasp, are implemented.
- A `vasprun.xml` parser wrapper is implemened to avoid the broken `PRECFOCK` in
  vasprun.xml of VASP 5.2.8.

## Sep-22-2010: Version 0.9.1.4

- The new tag `ANIME_TYPE` supports `xyz` and `xyz_jmol` formats by Jörg Meyer
  and Christian Carbogno, and also `A set of `POSCAR` files corresponding to
  animation frames.

- Fix bugs in `trim_cell` and `Primitive.__supercell_to_primitive_map` in
  `cells.py`. When :math:`M_s^{-1}M_p` is not symmetric, the supercell was not
  created correctly.

- `phonopy-FHI-aims` update by jm.

## Aug-24-2010: Version 0.9.1.3

- Update symmetry finder of spglib. Now precision is in Cartesian distance.

- The animation output for `arc` didn't work. Now it works.

- Qpoint mode didn't work with bugs. Now it works.

- `--vasp` option is renamed to `--cell` or `-c`.

- The new options `--symmetry`, `--displacement` or `-d`, `--dim`,
  `--primitive_axis` are implemented.

- The option `--ndim` is replaced with `--dim` with `-d` option.

## June-10-2010: Version 0.9.1.2

- The code on non-analytical term correction is included in the
  `DynamicalMatrix` class. Data sets read by `parse_BORN` are set by
  `set_non_analytical_term` and gotten by `get_non_analytical_term`. The
  q-vector direction (only direction is used in the non-analytical term
  correction) is set by `set_q_non_analytical_term`. However for emprical
  damping function, some distance is used, i.e., when a q-point is getting away,
  non-analytical term is weaken. For this purpose, the second argument of
  `set_q_non_analytical_term` is used.

  At the same time, a small problem on the previous implementation was found.
  When a reduced q-point is out of the first Brillouin zone, it is not correctly
  handled. Currently it is fixed so as that when absolute values of elements of
  the reduced q-point are over 0.5, they are reduced into -0.5 < q < 0.5.

  [**Attention**] The previous `--nac` option is moved to `--nac_old`. `--nac`
  is used for different method of the non-analytical term correction at general
  q-points. This will be documented soon.

- Bug fix on `write_FORCES` in `file_IO.py`. When order of displacements in
  `DISP` file is not ascending order of atom indices, it was not correctly
  re-ordered. Because the default order of phonopy is ascending order, usually
  there is no problem for the most users.

- `phonopy-FHI-aims`

  - adapted to extensions of dynamical_matrix with respect to non-analytical
    corrections
  - added support for animation infrastructure
  - moved several options to control.in

  by Jörg Meyer and Christian Carbogno

## May-11-2010: Version 0.9.1.1

- `phonopy-FHI-aims` adapted to split of dos array into the two seperate omega,
  dos arrays in TotalDOS class by Jörg Meyer.

## May-10-2010: Version 0.9.1

- The methods of get_partial_DOS and get_total_DOS are added to the Phonopy
  class.

## Apr-12-2010: Version 0.9.0.2

- spglib bug was fixed. If the crystal structure has non-standard origin, the
  translation was not correctly handled. This problem happened after version
  0.9.0.

## Apr-12-2010: Version 0.9.0.1

- spglib update

## Apr-10-2010: Version 0.9.0

- Phonopy module (`__init.py__`) is heavily revised and the script `phonopy` is
  rewritten using the phonopy module. Therefore there may be bugs. Be careful.
  Document of the phonopy module will be updated gradually.
- A small Wien2k interface document is added ({ref}`wien2k_interface`).
- A script `phonopy-FHI-aims` and its examples are added by Jörg Meyer.
- spglib update

## Mar-10-2010: Version 0.7.4

- spglib update
- Animation mode ({ref}`animation_tag`)

## Feb-10-2010: Version 0.7.3

- Bug fix for Wien2k mode

## Jan-12-2010: Version 0.7.2

- [**Experimental**] Non-analytical term correction was implemented.

## Dec-8-2009: Version 0.7.1 released

- {ref}`auxiliary_tools` `propplot` is added.
- Memory consumption is reduced when using `-f` option to handle large
  vasprun.xml files.

## Nov-24-2009: Version 0.7.0 released

- {ref}`auxiliary_tools` `bandplot` and `pdosplot` are prepared.
- Formats of `band.yaml`, `mesh.yaml`, and `qpoints.yaml` are slightly modified.
- There was bug in `PERMUTATION` tag to calculate symmetrized force constants.
  Now it is fixed. Usually this is not necessary to set because this does not
  affect to result.
- Symmetry finder spglib is updated.
- `PM` tag is implemented. See {ref}`setting_tags`. Behaviors in the previous
  versions are `PM = AUTO`.

## Oct-14-2009: Version 0.6.2 released

- Installation process was changed slightly. See {ref}`install`.
- The command `phonopy` is stored in the `bin` directory. `phonopy.py` is
  renamed to `phonopy`.
- setup system is improved by Maxim V. Losev.
- `--fz` tag was implemented experimentally. This is supposed to enable to
  subtract residual forces on atoms in equilibrium structure from those in
  structure with atomic displacements.
