(command_options)=
# Command options

From phonopy v1.12.2, the command option names with underscores `_` are replaced
by those with dashes `-`. Those tag names are unchanged.

Some of command-line options are equivalent to respective setting tags:

- `--alm` (`FC_CALCULATOR = ALM`) [{ref}`fc_calculator_alm_tag`]
- `--amplitude` (`DISPLACEMENT_DISTANCE`) [{ref}`displacement_distance_tag`]
- `--anime` (`ANIME`) [{ref}`anime_tag`]
- `--band` (`BAND`) [{ref}`band_tag`]
- `--band-connection` (`BAND_CONNECTION = .TRUE.`) [{ref}`band_connection_tag`]
- `--band-format` (`BAND_FORMAT`) [{ref}`band_format_tag`]
- `--band-labels` (`BAND_LABELS`) [{ref}`band_labels_tag`]
- `--band-points` (`BAND_POINTS`) [{ref}`band_points_tag`]
- `--cutoff-freq` (`CUTOFF_FREQUENCY`) [{ref}`cutoff_frequency_tags`]
- `-c`, `--cell` (`CELL_FILENAME`) [{ref}`cell_filename_tag`]
- `-d` (`CREATE_DISPLACEMENTS = .TRUE.`) [{ref}`create_displacements_tag`]
- `--dim` (`DIM`) [{ref}`dimension_tag`]
- `--dos` (`DOS = .TRUE.`) [{ref}`dos_tag`]
- `--eigvecs`, `--eigenvectors` (`EIGENVECTORS = .TRUE.`)
  [{ref}`eigenvectors_tag`]
- `--factor` (`FREQUENCY_CONVERSION_FACTOR`)
  [{ref}`frequency_conversion_factor_tag`]
- `--fc-calc` (`FC_CALCULATOR`) [{ref}`fc_calculator_tag`]
- `--fc-calc-opt` (`FC_CALCULATOR_OPTIONS`) [{ref}`fc_calculator_options_tag`]
- `--fc-symmetry` (`FC_SYMMETRY = .TRUE.`) [{ref}`fc_symmetry_tag`]
- `--fits-debye-model` (`DEBYE_MODEL = .TRUE.`) [{ref}`debye_model_tag`]
- `--fmax` (`FMAX`) [{ref}`FMAX <dos_fmin_fmax_tags>`]
- `--fmin` (`FMIN`) [{ref}`FMIN <dos_fmin_fmax_tags>`]
- `--fpitch` (`FPITCH`) [{ref}`FPITCH <dos_fmin_fmax_tags>`]
- `--full-fc` (`FULL_FORCE_CONSTANTS`) [{ref}`full_force_constants_tag`]
- `--gc`, `--gamma_center` (`GAMMA_CENTER`) [{ref}`gamma_center_tag`]
- `--gv`, `--group_velocity` (`GROUP_VELOCITY = .TRUE.`)
  [{ref}`group_velocity_tag`]
- `--gv-delta-q` (`GV_DELTA_Q`) [{ref}`gv_delta_q_tag`]
- `--hdf5` (`HDF5 = .TRUE.`) [{ref}`hdf5_tag`]
- `--irreps` (`IRREPS`) [{ref}`irreps_tag`]
- `--include-fc` (`INCLUDE_FC = .TRUE.`) [{ref}`include_fc_tag`]
- `--include-fs` (`INCLUDE_FS = .TRUE.`) [{ref}`include_fs_tag`]
- `--include-born` (`INCLUDE_BORN = .TRUE.`) [{ref}`include_born_tag`]
- `--include-disp` (`INCLUDE_DISP = .TRUE.`) [{ref}`include_disp_tag`]
- `--include-all` (`INCLUDE_ALL = .TRUE.`) [{ref}`include_all_tag`]
- `--lcg`, `--little_cogroup` (`LITTLE_COGROUP`) [{ref}`little_cogroup_tag`]
- `--magmom` (`MAGMOM`) [{ref}`magmom_tag`]
- `--modulation` (`MODULATION`) [{ref}`modulation_tag`]
- `--moment` (`MOMENT = .TRUE.`) [{ref}`MOMENT <dos_moment_tag>`]
- `--moment_order` (`MOMENT_ORDER`) [{ref}`MOMENT_ORDER <dos_moment_tag>`]
- `--mesh-format` (`MESH_FORMAT`) [{ref}`mesh_format_tag`]
- `--mp`, `--mesh` (`MP` or `MESH`) [{ref}`MESH <mesh_tag>`]
- `--nac` (`NAC = .TRUE.`) [{ref}`nac_tag`]
- `--nac-method` (`NAC_METHOD`) [{ref}`nac_method_tag`]
- `--nosym` (`SYMMETRY = .FALSE.`) [{ref}`symmetry_tag`]
- `--nomeshsym` (`MESH_SYMMETRY = .FALSE.`) [{ref}`nomeshsym_tag`]
- `--nowritemesh` (`WRITE_MESH = .FALSE.`) [{ref}`write_mesh_tag`]
- `--pa`, `--primitive-axes` (`PRIMITIVE_AXES`) [{ref}`primitive_axes_tag`]
- `--pd`, `--projection-direction` (`PROJECTION_DIRECTION`)
  [{ref}`projection_direction_tag`]
- `--pdos` (`PDOS`) [{ref}`pdos_tag`]
- `--pr`, `--pretend-real` (`PRETEND_REAL = .TRUE.`) [{ref}`pretend_real_tags`]
- `--q-direction` (`Q_DIRECTION`) [{ref}`q_direction_tag`]
- `--qpoints` (`QPOINTS`) [{ref}`qpoints_tag`]
- `--qpoints-format` (`QPOINTS_FORMAT`) [{ref}`qpoints_format_tag`]
- `--rd` (`RANDOM_DISPLACEMENTS`) [{ref}`random_displacements_tag`]
- `--rd-temperature` (`RANDOM_DISPLACEMENT_TEMPERATURE`)
  [{ref}`random_displacement_temperature_tag`]
- `--readfc` (`READ_FORCE_CONSTANTS = .TRUE.`) [{ref}`read_force_constants_tag`]
- `--readfc-format` (`READFC_FORMAT`) [{ref}`READFC_FORMAT <fc_format_tag>`]
- `--read-qpoints` (`QPOINTS = .TRUE.`) [{ref}`qpoints_tag`]
- `--show-irreps` (`SHOW_IRREPS`) [{ref}`show_irreps_tag`]
- `--sigma` (`SIGMA`) [{ref}`sigma_tag`]
- `--symfc` (`FC_CALCULATOR = SYMFC`) [{ref}`fc_calculator_symfc_tag`]
- `-t` (`TPROP = .TRUE.`) [{ref}`thermal_properties_tag`]
- `--td` (`TDISP = .TRUE.`) [{ref}`thermal_displacements_tag`]
- `--tdm` (`TDISPMAT = .TRUE.`) [{ref}`thermal_displacement_matrices_tag`]
- `--tdm-cif` (`TDISPMAT_CIF = .TRUE.`)
- `--tmin` (`TMIN`) [{ref}`TMIN <thermal_property_temperatures>`]
- `--tmax` (`TMAX`) [{ref}`TMAX <thermal_property_temperatures>`]
- `--tolerance` (`SYMMETRY_TOLERANCE`) [{ref}`tolerance_tag`]
- `--tstep` (`TSTEP`) [{ref}`TSTEP <thermal_property_temperatures>`]
- `--writedm` (`WRITEDM = .TRUE.`) [{ref}`writedm_tag`]
- `--writefc` (`WRITE_FORCE_CONSTANTS = .TRUE.`) [{ref}`write_force_constants_tag`]
- `--writefc-format` (`WRITEFC_FORMAT`) [{ref}`WRITEFC_FORMAT <fc_format_tag>`]
- `--xyz-projection` (`XYZ_PROJECTION = .TRUE.`) [{ref}`xyz_projection_tag`]

When both of equivalent command-line option and setting tag are set
simultaneously, the command-line option supersedes the setting tag. The
configuration file is recommended to place at the first position for the mixed
use of setting tags and command-line options, i.e.,

```bash
% phonopy-load --config setting.conf [OPTIONS]
```

(force_calculators)=
## Choice of force calculator

Currently interfaces for VASP, WIEN2k, Quantum ESPRESSO (QE), ABINIT, Elk,
SIESTA, CRYSTAL, TURBOMOLE, Fleur and CP2K are prepared. These interfaces are
invoked with `--vasp`, `--wienk2`, `--qe`, `--abinit`, `--elk`, `--siesta`,
`--crystal`, `--turbomole`, `--fleur` and `--cp2k` options, respectively. When
no interface is specified, `--vasp` is selected as the default interface.

The details about these interfaces are found at {ref}`calculator_interfaces`.

(abinit_mode)=
### `--abinit`

Abinit mode is invoked with this option.

(cp2k_mode)=
### `--cp2k`

CP2K mode is invoked with this option.

(crystal_mode)=
### `--crystal`

CRYSTAL mode is invoked with this option.

(elk_mode)=
### `--elk`

Elk mode is invoked with this option.

(fleur_mode)=
### `--fleur`

Fleur mode is invoked with this option.

(qe_mode)=
### `--qe`

Quantum ESPRESSO mode is invoked with this option.

(siesta_mode)=
### `--siesta`

Siesta mode is invoked with this option.

(turbomole_mode)=
### `--turbomole`

TURBOMOLE mode is invoked with this option.

(vasp_mode)=
### `--vasp`

With this option, the calculator name `vasp` will appear in `phonopy.yaml` type
files.

(wien2k_mode)=
### `--wien2k`

This option invokes the WIEN2k mode.

**Only the WIEN2k struct with the P lattice is supported**. See more information
{ref}`wien2k_interface`.

(cell_filename_option)=
## Input cell

### `-c` or `--cell`

**`phonopy-load` doesn't have this option.**

Unit cell crystal structure file is specified with this tag.

```bash
% phonopy -c POSCAR-unitcell [OPTIONS]
```

Without specifying this tag, default file name is searched in current directory.
The default file names for the calculators are as follows:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Calculator
  - Default crystal structure file name
* - VASP
  - `POSCAR`
* - WIEN2k
  - `case.struct`
* - ABINIT
  - `unitcell.in`
* - PWscf
  - `unitcell.in`
* - Elk
  - `elk.in`
* - CRYSTAL
  - `crystal.o`
* - TURBOMOLE
  - `control`
* - Fleur
  - `fleur.in`
* - CP2K
  - `unitcell.inp`
```

## Create `FORCE_SETS`

(f_force_sets_option)=
### `-f` or `--forces`

(vasp_force_sets_option)=
#### VASP interface

`FORCE_SETS` file is created from `phonopy_disp.yaml`, which is an output file
when creating supercells with displacements, and `vasprun.xml`'s, which are the
VASP output files. `phonopy_disp.yaml` in the current directory is automatically
read. The order of displacements written in `phonopy_disp.yaml` file has to
correpond to that of `vasprun.xml` files .

```bash
% phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml ...
```

```{note}
- Site-projected wave function information (the same information as `PROCAR`)
  significantly increases the size of `vasprun.xml`. So parsing xml file uses
  huge memory space. It is recommended
- to switch off to calculate it. If there are many displacements, shell
  expansions are useful, e.g., `disp-*/vasprun.xml`, or
  `disp-{001..128}/vasprun.xml` (for zsh, and recent bash).
```

(abinit_force_sets_option)=
#### ABINIT interface

`FORCE_SETS` file is created from `phonopy_disp.yaml` and ABINIT output files
(`*.out`). In the reading of forces in ABINIT output files, forces in
eV/Angstrom are read. The unit conversion factor is determined with this unit.

```bash
% phonopy -f disp-001/supercell.out disp-002/supercell.out ...
```

(qe_force_sets_option)=
#### Quantum ESPRESSO interface

`FORCE_SETS` file is created from `phonopy_disp.yaml` and QE-PW output files.

```bash
% phonopy -f disp-001/supercell.out disp-002/supercell.out ...
```

Here `*.out` files are the saved texts of standard outputs of PWscf
calculations.

(wien2k_force_sets_option)=

#### WIEN2k interface
This is experimental support to generage `FORCE_SETS`. Insted of this, you can
use the external tool called `scf2forces` to generate `FORCE_SETS`. `scf2forces`
is found at <http://www.wien2k.at/reg_user/unsupported/>.

`FORCE_SETS` file is created from `phonopy_disp.yaml`, which is an output file
when creating supercell with displacements, and `case.scf`'s, which are the
WIEN2k output files. The order of displacements in `phonopy_disp.yaml` file and
the order of `case.scf`'s have to be same. **For WIEN2k struct file, only
negative atom index with the P lattice format is supported.**

```bash
% phonopy -f case_001/case_001.scf case_002/case_002.scf ...
```

For more information, {ref}`wien2k_interface`.

(elk_force_sets_option)=
#### Elk interface

`FORCE_SETS` file is created from `phonopy_disp.yaml` and Elk output files.

```
% phonopy -f disp-001/INFO.OUT disp-002/INFO.OUT ...
```

(crystal_force_sets_option)=

#### CRYSTAL interface
`FORCE_SETS` file is created from `phonopy_disp.yaml` and CRYSTAL output files.

```bash
% phonopy -f supercell-001.o supercell-002.o ...
```

(turbomole_force_sets_option)=
#### TURBOMOLE interface

`FORCE_SETS` file is created from `phonopy_disp.yaml` and TURBOMOLE output
files.

```bash
% phonopy -f supercell-001 supercell-002 ...
```

(fleur_force_sets_option)=
#### Fleur interface

`FORCE_SETS` file is created from `phonopy_disp.yaml` and special Fleur FORCES
output files.

```bash
% phonopy -f disp-001/FORCES disp-002/FORCES ...
```

(cp2k_force_sets_option)=
#### CP2K interface

`FORCE_SETS` file is created from `phonopy_disp.yaml` and CP2K output files,
with:

```bash
% phonopy -f supercell-001-forces-1_0.xyz supercell-002-forces-1_0.xyz ...
```

Please note: the files containing the forces can be prefixed with the
`PROJECT_NAME` as specified in the original CP2K input file.

(fz_force_sets_option)=
### `--fz`

**`phonopy-load` doesn't have this option.**

`--fz` option is used to subtract residual forces frown the forces calculated
for the supercells with displacements. Here the residual forces mean that the
forces calculated for the perfect supercell for which the number of atoms has to
be the same as those for the supercells with displacements. If the forces are
accurately calculated by calculators, the residual forces should be canceled
when plus-minus displacements are employed (see {ref}`pm_displacement_tag`),
that is the default option in phonopy. Therefore `--fz` option is expected to be
useful when `PM = .FALSE.` is set in the phonopy setting file.

The usage of this option is almost the same as that of `-f` option except that
one more argument is inserted at the front. Mind that `--fz` is exclusively used
with `-f` option. The example for the VASP interface is shown below:

```bash
% phonopy --fz sposcar/vasprun.xml disp-001/vasprun.xml ...
```

where `sposcar/vasprun.xml` assumes the output file for the perfect supercell
containing residual forces.

This option perhaps works for the other calculator interfaces than the VASP
interface, but it is not tested yet. It would be appreciated if you report it to
the phonopy mailing list when you find it does/doesn't work for any other
calculator interfaces.

## Create `FORCE_CONSTANTS`

(vasp_force_constants)=

### `--fc` or `--force_constants`

**Currently this option supports only VASP output.**

VASP output of force constants is imported from `vasprun.xml` and
`FORCE_CONSTANTS` is created.

```bash
% phonopy --fc vasprun.xml
```

This `FORCE_CONSTANTS` can be used instead of `FORCE_SETS`. For more details,
please refer {ref}`vasp_dfpt_interface`.

(graph_option)=

(save_params_option)=
## Create `phonopy_params.yaml`

`phonopy_params.yaml` thus created by `--sp` option contains the necessary
information to run phonopy. Therefore only this file can be given to other user
as the phonopy input file.

### `--sp` or `--save-params`

After running phonopy, displacement-force dataset or force constants and
parameters for non-analytical term correction are written in
`phonopy_params.yaml` if these data are used.

When using with `-f`, displacement-force dataset are stored in
`phonopy_params.yaml` instead of `FORCE_SETS`. When `BORN` file is found in the
current directory, the parameters are also stored in `phonopy_params.yaml`.

## Graph plotting

### `-p`

Result is plotted.

```bash
% phonopy-load -p
```

(graph_save_option)=

### `-p -s`

Result is plotted (saved) to PDF file.

```bash
% phonopy-load -p -s
```

## Log level

### `-v` or `--verbose`

More detailed log are shown

### `-q` or `--quiet`

No log is shown.

## Crystal symmetry

(symmetry_option)=

### `--symmetry`

Using this option, various crystal symmetry information is just printed out and
phonopy stops without going to phonon analysis.

```bash
% phonopy --symmetry
```

This tag can be used together with the `--cell` (`-c`), `--abinit`, `--qe`,
`--elk`, `--wien2k`, `--siesta`, `--crystal` or `--primitive-axes` option.

After running this, `BPOSCAR` and `PPOSCAR` files are written, which are the
symmetrized conventional unit cell and primitive cell, respectively, in the VASP
style format.
