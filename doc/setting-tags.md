(setting_tags)=

# Setting tags

```{contents}
:depth: 2
:local:
```

Most of the setting tags have respective command-line options
({ref}`command_options`). When both of equivalent command-line option and
setting tag are set simultaneously, the command-line option supersedes the
setting tag. The configuration file is recommended to place at the first
position for the mixed use of setting tags and command-line options, i.e.,

```bash
% phonopy setting.conf [command-line-options]
```

For specifying real and reciprocal points, fractional values (e.g. `1/3`) are
accepted. However fractional values must not have space among characters (e.g.
`1 / 3`) are not allowed.

## Basic tags

(dimension_tag)=

### `DIM`

The supercell is created from the input unit cell. When three integers are
specified, a supercell elongated along axes of unit cell is created.

```
DIM = 2 2 3
```

In this case, a 2x2x3 supercell is created.

When nine integers are specified, the supercell is created by multiplying the
supercell matrix {math}`\mathrm{M}_\mathrm{s}` with the unit cell. For example,

```
DIM = 0 1 1 1 0 1 1 1 0
```

the supercell matrix is

```{math}
\mathrm{M}_\mathrm{s} = \begin{pmatrix}
0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0
\end{pmatrix}
```

where the rows correspond to the first three, second three, and third three sets
of numbers, respectively. When lattice parameters of unit cell are the column
vectors of {math}`\mathbf{a}_\mathrm{u}`, {math}`\mathbf{b}_\mathrm{u}`, and
{math}`\mathbf{c}_\mathrm{u}`, those of supercell,
{math}`\mathbf{a}_\mathrm{s}`, {math}`\mathbf{b}_\mathrm{s}`,
{math}`\mathbf{c}_\mathrm{s}`, are determined by,

```{math}
( \mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \; \mathbf{c}_\mathrm{s} ) = (
\mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u} \; \mathbf{c}_\mathrm{u} )
M_\mathrm{s}
```

Be careful that the axes in `POSCAR` is defined by three row vectors, i.e.,
{math}`( \mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u} \; \mathbf{c}_\mathrm{u} )^T`.

(primitive_axis_tag)=

### `PRIMITIVE_AXES` or `PRIMITIVE_AXIS`

When specified, transformation from the input unit cell to the primitive cell is
performed. With this, the primitive cell basis vectors are used as the
coordinate system for the phonon calculation. The transformation matrix is
specified by nine values. The first, second, and third three values give the
rows of the 3x3 matrix as follows:

```
PRIMITIVE_AXES = 0.0 0.5 0.5 0.5 0.0 0.5 0.5 0.5 0.0
```

Likewise,

```
PRIMITIVE_AXES = 0 1/2 1/2 1/2 0 1/2 1/2 1/2 0
```

The primitive cell for building the dynamical matrix is created by multiplying
primitive-axis matrix {math}`\mathrm{M}_\mathrm{p}`. Let the matrix as,

```{math}
\mathrm{M}_\mathrm{p} = \begin{pmatrix}
0.0 & 0.5 & 0.5 \\ 0.5 & 0.0 & 0.5 \\ 0.5 & 0.5 & 0.0
\end{pmatrix}
```

where the rows correspond to the first three, second three, and third three sets
of numbers, respectively.

When lattice parameters of unit cell (set by `POSCAR`) are the column vectors of
{math}`\mathbf{a}_\mathrm{u}`, {math}`\mathbf{b}_\mathrm{u}`, and
{math}`\mathbf{c}_\mathrm{u}`, those of supercell,
{math}`\mathbf{a}_\mathrm{p}`, {math}`\mathbf{b}_\mathrm{p}`,
{math}`\mathbf{c}_\mathrm{p}`, are determined by,

```{math}
( \mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \; \mathbf{c}_\mathrm{p} ) = (
\mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u} \; \mathbf{c}_\mathrm{u} )
\mathrm{M}\_\mathrm{p}.
```

{math}`\mathrm{M}_\mathrm{p}` is a change of basis matrix and so
{math}`\mathrm{M}_\mathrm{p}^{-1}` must be an integer matrix. Be careful that
{math}the axes in `POSCAR` is defined by three row vectors, i.e.,
{math}`( \mathbf{a}_\mathrm{u} \; \mathbf{b}_\mathrm{u} \; \mathbf{c}_\mathrm{u} )^T`.

**New in v1.14.0** `PRIMITIVE_AXES = AUTO` is supported. This enables to choose
the transformation matrix automatically. Since the choice of the primitive cell
is arbitrary, it is recommended to use `PRIMITIVE_AXES = AUTO` to check if a
possible transformation matrix exists or not.

### `ATOM_NAME`

When a crystal structure format has no information about chemical symbols, this
tag is used to specify chemical symbols.

```
ATOM_NAME = Si O
```

### `EIGENVECTORS`

When this tag is `.TRUE.`, eigenvectors are calculated.

(mass_tag)=

### `MASS`

This tag is not necessary to use usually, because atomic masses are
automatically set from the chemical symbols.

Atomic masses of a **primitive cell** are overwritten by the values specified.
The order of atoms in the primitive cell that is defined by `PRIMITIVE_AXIS` tag
can be shown using `-v` option. It must be noted that this tag does not affect
to the symmetry search.

For example, when there are six atoms in a primitive cell, `MASS` is set as
follows :

```
MASS = 28.085 28.085 16.000 16.000 16.000 16.000
```

(magmom_tag)=

### `MAGMOM`

Symmetry of spin such as collinear magnetic moments is considered using this
tag. The number of values has to be equal to the number of atoms in the unit
cell, not the primitive cell or supercell. If this tag is used with `-d` option
(`CREATE_DISPLACEMENTS` tag), `MAGMOM` file is created. This file contains the
`MAGMOM` information of the supercell used for VASP. Unlike `MAGMOM` in VASP,
`*` can not be used, i.e., all the values (the same number of times to the
number of atoms in unit cell) have to be explicitly written.

```
MAGMOM = 1.0 1.0 -1.0 -1.0
```

(cell_filename_tag)=

### `CELL_FILENAME`

```
CELL_FILENAME = POSCAR-unitcell
```

See {ref}`cell_filename_option`.

(frequency_conversion_factor_tag)=

### `FREQUENCY_CONVERSION_FACTOR`

Unit conversion factor of frequency from input values to your favorite unit can
be specified, but the use should be limited and it is recommended to use this
tag to convert the frequency unit to THz in some exceptional case, for example,
a special force calculator whose physical unit system is different from the
default setting of phonopy is used. If the frequency unit is different from THz,
though it works just for seeing results of frequencies, e.g., band structure or
DOS, it doesn't work for derived values like thermal properties and mean square
displacements.

The default values for calculators are those to convert frequency units to THz.
The default conversion factors are shown at
{ref}`frequency_default_value_interfaces`. These are determined following the
physical unit systems of the calculators. How to calculated these conversion
factors is explained at {ref}`physical_unit_conversion`.

## Displacement creation tags

### `CREATE_DISPLACEMENTS`

Supercells with displacements are created. This tag is used as the post process
of phonon calculation.

```
CREATE_DISPLACEMENTS = .TRUE.
DIM = 2 2 2
```

(displacement_distance_tag)=

### `DISPLACEMENT_DISTANCE`

Finite atomic displacement distance is set as specified value when creating
supercells with displacements. The default displacement amplitude is 0.01
Angstrom, but when the `wien2k`, `abinit`, `Fleur` or `turbomole` option is
specified, the default value is 0.02 Bohr.

### `DIAG`

When this tag is set `.FALSE.`, displacements in diagonal directions are not
searched, i.e. all the displacements are along the lattice vectors.
`DIAG = .FALSE.` is recommended if one of the lattice parameter of your
supercell is much longer or much shorter than the other lattice parameters.

(pm_displacement_tag)=

### `PM`

This tag specified how displacements are found. When `PM = .FALSE.`, least
displacements that can calculate force constants are found. This may cause less
accurate result. When `PM = .TRUE.`, all the displacements that are opposite
directions to the least displacements are also found, which is called plus-minus
displacements here. The default setting is `PM = AUTO`. Plus-minus displacements
are considered with this tag. If the plus and minus displacements are
symmetrically equivalent, only the plus displacement is found. This may be in
between `.FALSE.` and `.TRUE.`. You can check how it works to see the file
`DISP` where displacement directions on atoms are written.

(random_displacements_tag)=

### `RANDOM_DISPLACEMENTS`

The number of random displacement supercells are created by the specified
positive integer values. In each supercell, all atoms are displaced in random
direction with a constant displacement distance specified by
{ref}`displacement_distance_tag` tag. The random seed can be specified by
{ref}`random_seed_tag` tag.

To obtain force constants with random displacements and respective forces,
external force constants calculator is necessary.

```
CREATE_DISPLACEMENTS = .TRUE.
DIM = 2 2 2
RANDOM_DISPLACEMENTS = 20
DISPLACEMENT_DISTANCE = 0.03
```

(random_seed_tag)=

### `RANDOM_SEED`

The random seed used for creating random displacements by
{ref}`random_displacements_tag` tag. The value has to be 32bit unsigned int. The
random seed is useful for crating the same random displacements with using the
same number.

(band_structure_related_tags)=

## Band structure tags

### `BAND` and `BAND_POINTS`

`BAND` gives sampling band paths. The reciprocal points are specified in reduced
coordinates. The given points are connected for defining band paths. When comma
`,` is inserted between the points, the paths are disconnected.

`BAND_POINTS` gives the number of sampling points including the path ends. The
default value is `BAND_POINTS = 51`.

An example of three paths, (0,0,0) to (1/2,0,1/2), (1/2,1/2,1) to (0,0,0), and
(0,0,0) to (1/2,1/2,1/2), with 101 sampling points of each path are as follows:

```
BAND = 0 0 0 1/2 0 1/2, 1/2 1/2 1 0 0 0 1/2 1/2 1/2
BAND_POINTS = 101
```

(band_labels_tag)=

### `BAND_LABELS`

Labels specified are depicted in band structure plot at the points of band
segments. The number of labels has to correspond to the number of band paths
specified by `BAND` plus one. When LaTeX math style expression such as
{math}`\Gamma` (`\Gamma`) is expected, it is probably necessary to place it
between two $ characters.

```
BAND = 1/2 0 1/2 0 0 0 1/2 1/2 1/2
BAND_LABELS = X $\Gamma$ L
```

```{image} band-labels.png
:scale: 25
```

The colors of curves are automatically determined by matplotlib. The same color
in a band segment shows the same kind of band. Between different band segments,
the correspondence of colors doesn't mean anything.

(band_connection_tag)=

### `BAND_CONNECTION`

With this option, band connections are estimated from eigenvectors and band
structure is drawn considering band crossings. In sensitive cases, to obtain
better band connections, it requires to increase number of points calculated in
band segments by the `BAND_POINTS` tag.

```
BAND = 1/2 0 1/2 0 0 0 1/2 1/2 1/2
BAND_POINTS = 101
BAND_CONNECTION = .TRUE.
```

```{image} band-connection.png
:scale: 25
```

(mesh_sampling_tags)=

## Mesh sampling tags

Mesh sampling tags are used commonly for calculations of thermal properties and
density of states.

(mp_tag)=

### `MESH`, `MP`, or `MESH_NUMBERS`

`MESH` numbers give uniform meshes in each axis. As the default behavior, the
center of mesh is determined by the Monkhorst-Pack scheme, i.e., for odd number,
a point comes to the center, and for even number, the center is shifted half in
the distance between neighboring mesh points.

Examples of an even mesh with {math}`\Gamma` center in two ways,

```
MESH = 8 8 8
GAMMA_CENTER = .TRUE.
```

```
MESH = 8 8 8
MP_SHIFT = 1/2 1/2 1/2
```

If only one float value is given, e.g., `MESH = 100.0`, {math}`\Gamma` centred
sampling mesh is generated with the mesh numbers
{math}`(N_{\mathbf{a}^*}, N_{\mathbf{b}^*}, N_{\mathbf{c}^*})` computed
following the convention of the VASP automatic k-point generation, which is

```{math}
N_{\mathbf{a}^*} = \max[1, \mathrm{nint}(l|\mathbf{a}^*|)], \; N_{\mathbf{b}^*}
= \max[1, \mathrm{nint}(l|\mathbf{b}^*|)], \; N_{\mathbf{c}^*} = \max[1,
\mathrm{nint}(l|\mathbf{c}^*|)],
```

where {math}`l` is the value to be specified. With this, `GAMMA_CENTER` becomes
simply ignored, but `MP_SHIFT` works on top of the {math}`\Gamma` centred
sampling mesh.

```
MESh = 100
```

### `MP_SHIFT`

`MP_SHIFT` gives the shifts in direction along the corresponding reciprocal axes
({math}`a^*`, {math}`b^*`, {math}`c^*`). 0 or 1/2 (0.5) can be used as these
values. 1/2 means the half mesh shift with respect to neighboring grid points in
each direction.

### `GAMMA_CENTER`

Instead of employing the Monkhorst-Pack scheme for the mesh sampling,
{math}`\Gamma` center mesh is used. The default value is `.FALSE.`.

```
GAMMA_CENTER = .TRUE.
```

(write_mesh_tag)=

### `WRITE_MESH`

With a dense mesh, with eigenvectors, without mesh symmetry, sometimes its
output file `mesh.yaml` or `mesh.hdf5` can be huge. However when those files are
not needed, e.g., in (P)DOS calculation, `WRITE_MESH = .FALSE.` can disable to
write out those files. With (P)DOS calculation, DOS output files are obtained
even with `WRITE_MESH = .FALSE.`. The default setting is `.TRUE.`.

```
WRITE_MESH = .FALSE.
```

(dos_related_tags)=

## Phonon density of states (DOS) tags

Phonon density of states (DOS) is calculated either with a linear tetrahedron
method (default) or smearing method. Phonons are calculated on a sampling mesh,
therefore these tags must be used with {ref}`mesh_sampling_tags`. The physical
unit of horizontal axis is that of frequency that the user employs, e.g., THz,
and that of vertical axis is {no. of states}/({unit cell} x {unit of the
horizontal axis}). If the DOS is integrated over the frequency range, it will be
{math}`3N_\mathrm{a}` states, where {math}`N_\mathrm{a}` is the number of atoms
in the unit cell.

Phonon-DOS is formally defined as

```{math}
g(\omega) = \frac{1}{N} \sum_\lambda \delta(\omega - \omega_\lambda)
```

where {math}`N` is the number of unit cells and
{math}`\lambda = (\nu, \mathbf{q})` with {math}`\nu` as the band index and
{math}`\mathbf{q}` as the q-point. This is computed on a set of descritized
sampling frequency points for which {math}`\omega` is specified arbitrary using
{ref}`dos_range_tag`. The phonon frequencies {math}`\omega_\lambda` are obtained
on a sampling mesh whose the number of grid points being {math}`N`. In the
smearing method, the delta function is replaced by normal distribution (Gaussian
function) with the standard deviation specified by {ref}`sigma_tag`. In the
tetrahedron method, the Brillouin integration is made analytically within
tetrahedra in reciprocal space.

### `DOS`

This tag enables to calculate DOS. This tag is automatically set when `PDOS` tag
or `-p` option.

```
DOS = .TRUE.
```

(dos_range_tag)=

### `DOS_RANGE`

```
DOS_RANGE = 0 40 0.1
```

Total and partial density of states are drawn with some parameters. The example
makes DOS be calculated from frequency=0 to 40 with 0.1 pitch.

{ref}`dos_fmin_fmax_tags` can be alternatively used to specify the minimum and
maximum frequencies (the first and second values).

(dos_fmin_fmax_tags)=

### `FMIN`, `FMAX`, and `FPITCH`

The uniform frequency sampling points for phonon-DOS calculation are specified.
`FMIN` and `FMAX` give the minimum, maximum frequencies of the range,
respectively, and `FPITCH` gives the frequency pitch to be sampled. These three
values are the same as those that can be specified by `DOS_RANGE`.

### `PDOS`

Projected DOS is calculated using this tag. The formal definition is written as

```{math}
g^j(\omega, \hat{\mathbf{n}}) = \frac{1}{N} \sum_\lambda \delta(\omega -
\omega_\lambda) |\hat{\mathbf{n}} \cdot \mathbf{e}^j_\lambda|^2,
```

where {math}`j` is the atom indices and {math}`\hat{\mathbf{n}}` is the unit
projection direction vector. Without specifying {ref}`projection_direction_tag`
or {ref}`xyz_projection_tag`, PDOS is computed as sum of
{math}`g^j(\omega, \hat{\mathbf{n}})` projected onto Cartesian axes
{math}`x,y,z`, i.e.,

```{math}
g^j(\omega) = \sum_{\hat{\mathbf{n}} = \{x, y, z\}} g^j(\omega,
\hat{\mathbf{n}}).
```

The atom indices {math}`j` are specified by

```
PDOS = 1 2, 3 4 5 6
```

These numbers are those in the primitive cell. `,` separates the atom sets. In
this example, atom 1 and 2 are summarized as one curve and atom 3, 4, 5, and, 6
are summarized as another curve.

`PDOS = AUTO` is supported To group symmetrically equivalent atoms
automatically.

```
PDOS = AUTO
```

`EIGENVECTORS = .TRUE.` and `MESH_SYMMETRY = .FALSE.` are automatically set,
therefore the calculation takes much more time than usual DOS calculation. With
a very dense sampling mesh, writing data into `mesh.yaml` or `mesh.hdf5` can be
unexpectedly huge. If only PDOS is necessary but these output files are
unnecessary, then it is good to consider using `WRITE_MESH = .FALSE.`
({ref}`write_mesh_tag`).

(projection_direction_tag)=

### `PROJECTION_DIRECTION`

Eigenvectors are projected along the direction specified by this tag. Projection
direction is specified in reduced coordinates, i.e., with respect to _a_, _b_,
_c_ axes.

```
PDOS = 1, 2
PROJECTION_DIRECTION = -1 1 1
```

(xyz_projection_tag)=

### `XYZ_PROJECTION`

PDOS is calculated using eigenvectors projected along x, y, and z Cartesian
coordinates. The format of output file `projected_dos.dat` becomes different
when using this tag, where phonon-mode-frequency and x, y, and z components of
PDOS are written out in the order:

```
frequency atom1_x atom1_y atom1_z atom2_x atom2_y atom2_z ...
```

With `-p` option, three curves are drawn. These correspond to sums of all
projections to x, sums of all projections to y, and sums of all projections to z
components of eigenvectors, respectively.

```
XYZ_PROJECTION = .TRUE.
```

(sigma_tag)=

### `SIGMA`

A smearing method is used instead of a linear tetrahedron method. This tag also
specifies the smearing width. The unit is same as that used for phonon
frequency. The default value is the value given by the difference of maximum and
minimum frequencies divided by 100.

```
SIGMA = 0.1
```

(debye_model_tag)=

### `DEBYE_MODEL`

By setting `.TRUE.`, DOS at lower phonon frequencies are fit to a Debye model.
By default, the DOS from 0 to 1/4 of the maximum phonon frequencies are used for
the fitting. The function used to the fitting is {math}`D(\omega)=a\omega^2`
where {math}`a` is the parameter and the Debye frequency is {math}`(9N/a)^{1/3}`
where {math}`N` is the number of atoms in unit cell. Users have to unserstand
that this is **not** a unique way to determine Debye frequency. Debye frequency
is dependent on how to parameterize it.

```
DEBYE_MODEL = .TRUE.
```

(dos_moment_tag)=

### `MOMEMT` and `MOMENT_ORDER`

Phonon moments for DOS and PDOS defined below are calculated using these tags up
to arbitrary order. The order is specified with `MOMENT_ORDER` ({math}`n` in the
formula). Unless `MOMENT_ORDER` specified, the first and second moments are
calculated.

The moments for DOS are given as

```{math}
M_n(\omega_\text{min}, \omega_\text{max})
=\frac{\int_{\omega_\text{min}}^{\omega_\text{max}} \omega^n g(\omega) d\omega}
{\int_{\omega_\text{min}}^{\omega\_\text{max}} g(\omega) d\omega}.
```

The moments for PDOS are given as

```{math}
M_n^j(\omega_\text{min}, \omega_\text{max})
=\frac{\int_{\omega_\text{min}}^{\omega_\text{max}} \omega^n g^j(\omega)
d\omega} {\int_{\omega_\text{min}}^{\omega\_\text{max}} g^j(\omega) d\omega}.
```

{math}`\omega_\text{min}` and {math}`\omega_\text{max}` are specified :using
ref:`dos_fmin_fmax_tags` tags. When these are not specified, the moments are
computed with the range of {math}`\epsilon < \omega < \infty`, where
{math}`\epsilon` is a small positive value. Imaginary frequencies are treated as
negative real values in this computation, therefore it is not a good idea to set
negative {math}`\omega_\text{min}`.

```
MOMENT = .TRUE.
MOMENT_ORDER = 3
```

(thermal_properties_tag)=

## Thermal properties related tags

See {ref}`cutoff_frequency_tags` on the treatment of the imaginary modes.

### `TPROP`, `TMIN`, `TMAX`, and `TSTEP`

Thermal properties, free energy, heat capacity, and entropy, are calculated from
their statistical thermodynamic expressions (see
{ref}`thermal_properties_expressions`). Thermal properties are calculated from
phonon frequencies on a sampling mesh in the reciprocal space. Therefore these
tags must be used with {ref}`mesh_sampling_tags` and their convergence with
respect to the sampling mesh has to be checked. Usually this calculation is not
computationally demanding, so the convergence is easily achieved with increasing
the density of the sampling mesh. `-p` option can be used together to plot the
thermal properties.

Phonon frequencies in THz, which is the default setting of phonopy, are used to
obtain the thermal properties, therefore physical units have to be set properly
for it (see {ref}`calculator_interfaces`.)

The calculated values are written into `thermal_properties.yaml`. The unit
systems of free energy, heat capacity, and entropy are kJ/mol, J/K/mol, and
J/K/mol, respectively, where 1 mol means {math}`\mathrm{N_A}\times` your input
unit cell (not formula unit), i.e. you have to divide the value by number of
formula unit in your unit cell by yourself. For example, in MgO (conventional)
unit cell, if you want to compare with experimental results in kJ/mol, you have
to divide the phonopy output by four.

`TMIN`, `TMAX`, and `TSTEP` tags are used to specify the temperature range to be
calculated. The default values of them are 0, 1000, and 10, respectively.

```
TPROP = .TRUE.
TMAX = 2000
```

(pretend_real_tags)=

### `PRETEND_REAL`

This enables to take imaginary frequencies as real for thermal property
calculation. This does give false thermal properties, therefore for a testing
purpose only, when a small amount of imaginary branches obtained.

```
TPROP = .TRUE.
PRETEND_REAL = .TRUE.
```

(cutoff_frequency_tags)=

### `CUTOFF_FREQUENCY`

This is given by a real value and the default value is 0. This tag works as
follows.

Phonon thermal properties are computed as sum over phonon modes. See
{ref}`thermal_properties_expressions`. When we treat imaginary frequencies as
negative values by
{math}`\text{sgn}(\nu^2) |\nu| \rightarrow \nu_\text{phonopy}`, all phonon modes
with {math}`\nu_\text{phonopy}` smaller than this `CUTOFF_FREQUENCY` are simply
excluded in the summation.

In the `thermal_properties.yaml`, the total number of calculated phonon modes
and the number of phonon modes included for the thermal property calculation are
shown as `num_modes:` and `num_integrated_modes:`, respectively.

```
CUTOFF_FREQUENCY = 0.1
```

(thermal_atomic_displacements_tags)=

## Thermal displacements

(thermal_displacements_tag)=

### `TDISP`, `TMAX`, `TMIN`, and `TSTEP`

Mean square displacements projected to Cartesian axes as a function of
temperature are calculated from the number of phonon excitations. The usages of
`TMAX`, `TMIN`, `TSTEP` tags are same as those in
{ref}`thermal properties tags <thermal_properties_tag>`. Phonon frequencies in
THz, which is the default setting of phonopy, are used to obtain the mean square
displacements, therefore physical units have to be set properly for it (see
{ref}`calculator_interfaces`.) The result is given in {math}`\text{Angstrom}^2`
and writen into `thermal_displacements.yaml`. See the detail of the method,
{ref}`thermal_displacement`. These tags must be used with
{ref}`mesh_sampling_tags`

Optionally, `FMIN` tag (`--fmin` option) with a small value is recommened to be
set when q-points at {math}`\Gamma` point or near {math}`\Gamma` point (e.g.
using very dense sampling mesh) are sampled to avoid divergence. `FMAX` tag
(`--fmax` option) can be used to specify an upper bound of phonon frequencies
where the phonons are considered in the summation. The projection is applied
along arbitrary direction using `PROJECTION_DIRECTION` tag
({ref}`projection_direction_tag`).

`mesh.yaml` or `mesh.hdf5` is not written out from phonopy-1.11.14.

```
TDISP = .TRUE.
PROJECTION_DIRECTION = 1 1 0
```

(thermal_displacement_matrices_tag)=

### `TDISPMAT`, `TMAX`, `TMIN`, and `TSTEP`

Mean square displacement matrices are calculated. The definition is shown at
{ref}`thermal_displacement`. Phonon frequencies in THz, which is the default
setting of phonopy, are used to obtain the mean square displacement matrices,
therefore physical units have to be set properly for it (see
{ref}`calculator_interfaces`.) The result is given in {math}`\text{Angstrom}^2`
and written into `thermal_displacement_matrices.yaml` where six matrix elements
are given in the order of xx, yy, zz, yz, xz, xy. In this yaml file,
`displacement_matrices` and `displacement_matrices_cif` correspond to
{math}`\mathrm{U}_\text{cart}` and {math}`\mathrm{U}_\text{cif}` defined at
{ref}`thermal_displacement_matrix`, respectively.

Optionally, `FMIN` tag (`--fmin` option) with a small value is recommended to be
set when q-points at {math}`\Gamma` point or near {math}`\Gamma` point (e.g.
using very dense sampling mesh) are sampled to avoid divergence. `FMAX` tag
(`--fmax` option) can be used to specify an upper bound of phonon frequencies
where the phonons are considered in the summation.

The 3x3 matrix restricts distribution of each atom around the equilibrium
position to be ellipsoid. But the distribution is not necessarily to be so.

`mesh.yaml` or `mesh.hdf5` is not written out from phonopy-1.11.14.

```
TDISPMAT = .TRUE.
```

(thermal_displacement_cif_tag)=

### `TDISPMAT_CIF`

This tag specifies a temperature (K) at which thermal displacement is calculated
and the mean square displacement matrix is written to the cif file
`tdispmat.cif` with the dictionary item `aniso_U`. Phonon frequencies in THz,
which is the default setting of phonopy, are used to obtain the mean square
displacement matrices, therefore physical units have to be set properly for it
(see {ref}`calculator_interfaces`.) The result is given in
{math}`\textrm{Angstrom}^2`.

`mesh.yaml` or `mesh.hdf5` is not written out from phonopy-1.11.14.

```
TDISPMAT_CIF = 1273.0
```

## Specific q-points

(qpoints_tag)=

### `QPOINTS`

When q-points are supplied, those phonons are calculated. Q-points are specified
successive values separated by spaces and collected by every three values as
vectors in reciprocal reduced coordinates.

```
QPOINTS = 0 0 0 1/2 1/2 1/2 1/2 0 1/2
```

With `QPOINTS = .TRUE.`, q-points are read from `QPOITNS` file (see the file
format at {ref}`QPOINTS<qpoints_file>`) in current directory phonons at the
q-points are calculated.

```
QPOINTS = .TRUE.
```

(writedm_tag)=

### `WRITEDM`

```
WRITEDM = .TRUE.
```

Dynamical matrices {math}`D` are written into `qpoints.yaml` in the following
{math}`6N\times3N` format, where _N_ is the number of atoms in the primitive
cell.

The physical unit of dynamical matrix is
`[unit of force] / ([unit of displacement] * [unit of mass])`, i.e., square of
the unit of phonon frequency before multiplying the unit conversion factor (see
{ref}`frequency_conversion_factor_tag`).

```{math}
D = \begin{pmatrix} D_{11} & D_{12} & D_{13} & \\ D_{21} & D_{22} & D_{23} &
\cdots \\ D_{31} & D_{32} & D_{33} & \\ & \vdots & & \\ \end{pmatrix},
```

and {math}`D_{jj'}` is

```{math}
D_{jj'} = \begin{pmatrix} Re(D_{jj'}^{xx}) & Im(D_{jj'}^{xx}) &
Re(D_{jj'}^{xy}) & Im(D_{jj'}^{xy}) & Re(D_{jj'}^{xz}) & Im(D_{jj'}^{xz}) \\
Re(D_{jj'}^{yx}) & Im(D_{jj'}^{yx}) & Re(D_{jj'}^{yy}) & Im(D_{jj'}^{yy}) &
Re(D_{jj'}^{yz}) & Im(D_{jj'}^{yz}) \\ Re(D_{jj'}^{zx}) & Im(D_{jj'}^{zx}) &
Re(D_{jj'}^{zy}) & Im(D_{jj'}^{zy}) & Re(D_{jj'}^{zz}) & Im(D_{jj'}^{zz}) \\
\end{pmatrix},
```

where _j_ and _j'_ are the atomic indices in the primitive cell. The phonon
frequencies may be recovered from `qpoints.yaml` by writing a simple python
script. For example, `qpoints.yaml` is obtained for NaCl at
{math}`q=(0, 0.5, 0.5)` by

```
phonopy --qpoints="0 1/2 1/2" --writedm
```

and the dynamical matrix may be used as

```python
import yaml
import numpy as np

data = yaml.load(open("qpoints.yaml"))
dynmat = []
dynmat_data = data['phonon'][0]['dynamical_matrix']
for row in dynmat_data:
    vals = np.reshape(row, (-1, 2))
    dynmat.append(vals[:, 0] + vals[:, 1] * 1j)
dynmat = np.array(dynmat)

eigvals, eigvecs, = np.linalg.eigh(dynmat)
frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
conversion_factor_to_THz = 15.633302
print frequencies * conversion_factor_to_THz
```

(nac_tag)=

## Non-analytical term correction

### `NAC`

Non-analytical term correction is applied to dynamical matrix. `BORN` file has
to be prepared in the current directory. See {ref}`born_file` and
{ref}`non_analytical_term_correction_theory`. The default method is
`NAC_METHOD = GONZE` after v1.13.0.

```
NAC = .TRUE.
```

(nac_method_tag)=

### `NAC_METHOD`

The method of non-analytical term correction is chosen by this tag between two,
`NAC_METHOD = GONZE` ({ref}`reference_dp_dp_NAC`) and `NAC_METHOD = WANG`
({ref}`reference_wang_NAC`), and the default is the former after v1.13.0.

### `Q_DIRECTION`

This tag is used to activate non-analytical term correction (NAC) at
{math}`\mathbf{q}\rightarrow\mathbf{0}`, i.e. practically {math}`\Gamma`-point,
because NAC is direction dependent. With this tag, {math}`\mathbf{q}` is
specified in the fractional coordinates of the reciprocal basis vectors. Only
the direction has the meaning. Therefore `Q_DIRECTION = 1 1 1` and
`Q_DIRECTION = 2 2 2` give the same result. This tag is valid for `QPOINTS`,
`IRREPS`, and `MODULATION` tags.

Away from {math}`\Gamma`-point, this setting is ignored and the specified
**q**-point is used as the **q**-direction.

```
QPOINTS = 0 0 0 NAC = .TRUE.
Q_DIRECTION = 1 0 0
```

(group_velocity_tag)=

## Group velocity

### `GROUP_VELOCITY`

Group velocities at q-points are calculated by using this tag. The group
velocities are written into a yaml file corresponding to the run mode in
Cartesian coordinates. The physical unit depends on physical units of input
files and frequency conversion factor, but if VASP and the default settings
(e.g., THz for phonon frequency) are simply used, then the physical unit will be
Angstrom THz.

```
GROUP_VELOCITY = .TRUE.
```

Technical details are shown at {ref}`group_velocity`.

### `GV_DELTA_Q`

The reciprocal distance used for finite difference method is specified. The
default value is `1e-5` for the method of non-analytical term correction by
Gonze _et al._. In other case, unless this tag is specified, analytical
derivative is used instead of the finite difference method.

```
GV_DELTA_Q = 0.01
```

## Symmetry

(tolerance_tag)=

### `SYMMETRY_TOLERANCE`

This is used to set geometric tolerance to find symmetry of crystal structure.
The default value is `1e-5`. In general, it is not a good idea to loosen the
tolerance. It is recommended to symmetrize crystal structure before starting
phonon calculation, e.g., using {ref}`symmetry_option` option.

```
SYMMETRY_TOLERANCE = 1e-3
```

(symmetry_tag)=

### `SYMMETRY`

P1 symmetry is enforced to the input unit cell by setting `SYMMETRY = .FALSE`.

(nomeshsym_tag)=

### `MESH_SYMMETRY`

Symmetry search on the reciprocal sampling mesh is disabled by setting
`MESH_SYMMETRY = .FALSE.`. In some case such as hexagonal systems or primitive
cells of cubic systems having F and I-centrings, the results with and without
mesh symmetry give slightly different values for those properties that can
employ mesh symmetry. This happens when the uniform sampling mesh made along
basis vectors doesn't have the same crystallographic point group as the crystal
itself. This symmetry breaking may be also seen by the fact that `weight`
written in `mesh.yaml` can be different from possible order of product group of
site-symmetry group and time reversal symmetry. Generally the difference becomes
smaller when increasing the sampling mesh numbers.

(fc_symmetry_tag)=

### `FC_SYMMETRY`

**Changed at v1.12.3**

Previously this tag required a number for the iteration. From version 1.12.3,
the way of symmetrization for translation invariance is modified and this number
became unnecessary.

This tag is used to symmetrize force constants by translational symmetry and
permutation symmetry with `.TRUE.` or `.FALSE.`.

```
FC_SYMMETRY = .TRUE.
```

From the translation invariance condition,

```{math}
\sum_i \Phi_{ij}^{\alpha\beta} = 0, \;\;\text{for all $j$, $\alpha$, $\beta$},
```

where _i_ and _j_ are the atom indices, and {math}`\alpha` and {math}`\beta` are
the Cartesian indices for atoms _i_ and _j_, respectively. When this condition
is broken, the sum gives non-zero value. This value is subtracted from the
diagonal blocks. Force constants are symmetric in each pair as

```{math}
\Phi_{ij}^{\alpha\beta} = \frac{\partial^2 U}{\partial u_i^\alpha \partial
u_j^\beta} = \frac{\partial^2 U}{\partial u_j^\beta \partial u_i^\alpha} =
\Phi_{ji}^{\beta\alpha}
```

Mind that the other symmetries of force constants, i.e., the symmetry from
crystal symmetry or rotational symmetry, are broken to use `FC_SYMMETRY`.

(force_constants_tag)=

## Force constants

### `FORCE_CONSTANTS`

```
FORCE_CONSTANTS = READ
```

There are three values to be set, which are `READ` and `WRITE`, and `.FALSE.`.
The default is `.FALSE.`. When `FORCE_CONSTANTS = READ`, force constants are
read from `FORCE_CONSTANTS` file. With `FORCE_CONSTANTS = WRITE`, force
constants calculated from `FORCE_SETS` are written to `FORCE_CONSTANTS` file.

The file format of `FORCE_CONSTANTS` is shown
{ref}`here <file_force_constants>`.

(full_force_constants_tag)=

### `FULL_FORCE_CONSTANTS`

`FULL_FORCE_CONSTANTS = .TRUE.` is used to compute full supercell constants
matrix. The default setting is `.FALSE.`. By `.TRUE.` or `.FALSE.`, the array
shape becomes `(n_patom, n_satom, 3, 3)` or `(n_satom, n_satom, 3, 3)`,
respectively. The detail is found at {ref}`file_force_constants`.

### `READ_FORCE_CONSTANTS`

`READ_FORCE_CONSTANTS = .TRUE.` is equivalent to `FORCE_CONSTANTS = READ`.

### `WRITE_FORCE_CONSTANTS`

`WRITE_FORCE_CONSTANTS = .TRUE.` is equivalent to `FORCE_CONSTANTS = WRITE`.

(fc_calculator_tag)=

### `FC_CALCULATOR`

External force constants calculator can be used using this tag. Currently `ALM`
is supported. The phonopy's default force constants calculator is based on
finite difference method, for which atomic displacements are made
systematically. The following is the list of the force constants calculator
currently possible to be invoked from phonopy.

(fc_calculator_alm_tag)=

#### `ALM`

**New in v2.3** ALM (https://github.com/ttadano/ALM) is based on fitting
approach and any displacements set of atoms in supercell can be handled. For
example, random displacements generated by {ref}`random_displacements_tag` can
be used to compute force constants. To use ALM, its python module has to be
installed via conda-forge or building it. The installation instruction is found
[here](https://alm.readthedocs.io/en/develop/compile-with-conda-packages.html).

When ALM is used, please cite the paper: T. Tadano and S. Tsuneyuki, J. Phys.
Soc. Jpn. **87**, 041015 (2018).

```
FC_CALCULATOR = ALM
```

(animation_tag)=

## Create animation file

### `ANIME_TYPE`

```
ANIME_TYPE = JMOL
```

There are `V_SIM`, `ARC`, `XYZ`, `JMOL`, and `POSCAR` settings. Those may be
viewed by `v_sim`, `gdis`, `jmol` (animation), `jmol` (vibration), respectively.
For `POSCAR`, a set of `POSCAR` format structure files corresponding to
respective animation images are created such as `APOSCAR-000`,
`APOSCAR-001`,....

There are several parameters to be set in the `ANIME` tag.

### `ANIME`

**The format of `ANIME` tag was modified after ver. 0.9.3.3.**

#### For v_sim

```
ANIME = 0.5 0.5 0
```

The values are the _q_-point to be calculated. An animation file of
`anime.ascii` is generated.

```{toctree}
animation
```

#### For the other animation formats

Phonon is only calculated at {math}`\Gamma` point. So _q_-point is not necessary
to be set.

`anime.arc`, `anime.xyz`, `anime.xyz_jmol`, or `APOSCAR-*` are generated
according to the `ANIME_TYPE` setting.

```
ANIME = 4 5 20 0.5 0.5 0
```

The values are as follows from left:

1. Band index given by ascending order in phonon frequency.

2. Magnitude to be multiplied. In the harmonic phonon calculation, there is no
   amplitude information obtained directly. The relative amplitude among atoms
   in primitive cell can be obtained from eigenvectors with the constraint of
   the norm or the eigenvectors equals one, i.e., number of atoms in the
   primitive is large, the displacements become small. Therefore this has to be
   adjusted to make the animation good looking.

3. Number of images in one phonon period.

4. (4-6) Shift of atomic points in reduced coordinate in real space. These
   values can be omitted and the default values are `0 0 0`.

For `anime.xyz_jmol`, the first and third values are not used, however dummy
values, e.g. 0, are required.

(modulation_tag)=

## Create modulated structure

### `MODULATION`

The `MODULATION` tag is used to create a crystal structure with displacements
along normal modes at q-point in the specified supercell dimension.

Atomic displacement of the _j_-th atom is created from the real part of the
eigenvectors with amplitudes and phase factors as

```{math}
\frac{A} { \sqrt{N_\mathrm{a}m_j} } \operatorname{Re} \left[ \exp(i\phi)
\mathbf{e}_j \exp( \mathbf{q} \cdot \mathbf{r}_{jl} ) \right],
```

where {math}`A` is the amplitude, {math}`\phi` is the phase,
{math}`N_\mathrm{a}` is the number of atoms in the supercell specified in this
tag and {math}`m_j` is the mass of the _j_-th atom, {math}`\mathbf{q}` is the
q-point specified, {math}`\mathbf{r}_{jl}` is the position of the _j_-th atom in
the _l_-th unit cell, and {math}`\mathbf{e}_j` is the _j_-th atom part of
eigenvector. Convention of eigenvector or dynamical matrix employed in phonopy
is shown in {ref}`dynacmial_matrix_theory`.

If several modes are specified as shown in the example above, they are
overlapped on the structure. The output filenames are `MPOSCAR` and
`MPOSCAR-<number>`. Each modulated structure of a normal mode is written in
`MPOSCAR-<number>` where the numbers correspond to the order of specified sets
of modulations. `MPOSCAR` is the structure where all the modulations are summed.
`MPOSCAR-orig` is the structure without containing modulation, but the dimension
is the one that is specified. Some information is written into
`modulation.yaml`.

#### Usage

The first three (nine) values correspond to supercell dimension (supercell
matrix) like the {ref}`dimension_tag` tag. The following values are used to
describe how the atoms are modulated. Multiple sets of modulations can be
specified by separating by comma `,`. In each set, the first three values give a
Q-point in the reduced coordinates in reciprocal space. Then the next three
values are the band index from the bottom with ascending order, amplitude, and
phase factor in degrees. The phase factor is optional. If it is not specified, 0
is used.

Before multiplying user specified phase factor, the phase of the modulation
vector is adjusted as the largest absolute value,
{math}`\left|\mathbf{e}_j\right|/\sqrt{m_j}`, of element of 3N dimensional
modulation vector to be real. The complex modulation vector is shown in
`modulation.yaml`.

```
MODULATION = 3 3 1, 1/3 1/3 0 1 2, 1/3 1/3 0 2 3.5
```

```
MODULATION = 3 3 1, 1/3 1/3 0 1 2, 1/3 0 0 2 2
```

```
MODULATION = 3 3 1, 1/3 1/3 0 1 1 0, 1/3 1/3 0 1 1 90
```

```
MODULATION = -1 1 1 1 -1 1 1 1 -1, 1/2 1/2 0 1 2
```

(irreducible_representation_related_tags)=

## Characters of irreducible representations

(irreps_tag)=

### `IRREPS`

Characters of irreducible representations (Irreps) of phonon modes are shown.
For this calculation, a primitive cell has to be used. If the input unit cell is
a non-primitive cell, it has to be transformed to a primitive cell using
`PRIMITIVE_AXES` tag.

The first three values gives a _q_-point in reduced coordinates to be
calculated. The degenerated modes are searched only by the closeness of
frequencies. The frequency difference to be tolerated is specified by the fourth
value in the frequency unit that the user specified.

```
IRREPS = 0 0 0 1e-3
```

Symbols of Irreps for the 32 point group types at the {math}`\Gamma` point are
shown but not at non-{math}`\Gamma` point.

### `SHOW_IRREPS`

Irreducible representations are shown along with character table.

```
IRREPS = 1/3 1/3 0
SHOW_IRREPS = .TRUE.
```

### `LITTLE_COGROUP`

Show irreps of little co-group (point-group of wavevector) instead of little
group.

```
IRREPS = 0 0 1/8
LITTLE_COGROUP = .TRUE.
```

## Input/Output file control

(fc_format_tag)=

### `FC_FORMAT`, `READFC_FORMAT`, `WRITEFC_FORMAT`

There are two file-formats to store force constants. Currently
{ref}`text style<file_force_constants>` (`TEXT`) and hdf5 (`HDF5`) formats are
supported. The default file format is the
{ref}`text style<file_force_constants>`. Reading and writing force constants are
invoked by {ref}`FORCE_CONSTANTS tag<force_constants_tag>`. Using these tags,
the input/output formats are switched.

`FC_FORMAT` affects to both input and output, e.g.

```
FORCE_CONSTANTS = WRITE
FC_FORMAT = HDF5
```

`READFC_FORMAT` and `WRITEFC_FORMAT` can be used to control input and output
formats separately, i.e., the following setting to convert force constants
format is possible:

```
READ_FORCE_CONSTANTS = .TRUE.
WRITE_FORCE_CONSTANTS = .TRUE.
WRITEFC_FORMAT = HDF5
```

(band_format_tag)=

### `BAND_FORMAT`, `MESH_FORMAT`, `QPOINTS_FORMAT`

There are two file-formats to write the results of band structure, mesh, and
q-points calculations. Currently YAML (`YAML`) and hdf5 (`HDF5`) formats are
supported. The default file format is the YAML format. The file format is
changed as follows:

```
BAND_FORMAT = HDF5
```

```
MESH_FORMAT = HDF5
```

```
QPOINTS_FORMAT = HDF5
```

(hdf5_tag)=

### `HDF5`

The following output files are written in hdf5 format instead of their original
formats (in parenthesis) by `HDF5 = .TRUE.`. In addition, `force_constants.hdf5`
is read with this tag.

- `force_constants.hdf5` (`FORCE_CONSTANTS`)
- `mesh.hdf5` (`mesh.yaml`)
- `band.hdf5` (`band.yaml`)
- `qpoints.hdf5` (`qpoints.yaml`)

```
HDF5 = .TRUE.
```

#### `force_constants.hdf5`

With `--hdf5` option and `FORCE_CONSTANTS = WRITE` (`--writefc`),
`force_constants.hdf5` is written. With `--hdf5` option and
`FORCE_CONSTANTS = READ` (`--readfc`), `force_constants.hdf5` is read.

#### `mesh.hdf5`

In the mesh sampling calculations (see {ref}`mesh_sampling_tags`), calculation
results are written into `mesh.hdf5` but not into `mesh.yaml`. Using this option
may reduce the data output size and thus writing time when `mesh.yaml` is huge,
e.g., eigenvectors are written on a dense sampling mesh.

#### `qpoints.hdf5`

In the specific q-points calculations ({ref}`qpoints_tag`), calculation results
are written into `qpoints.hdf5` but not into `qpoints.yaml`. With
{ref}`writedm_tag`, dynamical matrices are also stored in `qpoints.hdf5`. Using
this option may be useful with large set of q-points with including eigenvector
or dynamical matrix output.

#### `band.hdf5`

In the band structure calculations ({ref}`band_structure_related_tags`),
calculation results are written into `band.hdf5` but not into `band.yaml`.

(summary_tag)=

### `summary`

The following data may be optionally included in the summary yaml file called
`phonopy_disp.yaml`/`phonopy.yaml` in addition to other file output settings.
This happens at the end of the pre/post-process (after running the `phonopy`
script):

- `force constants`
- `force sets`
- `dielectric constant`
- `born effective charge`
- `displacements`
- `[all]`

Including all relevant data in a single output file allows for a human readable
convenient file format.

#### `force constants`

The `--include-fc` flag or setting `INCLUDE_FC = .TRUE.` will cause the force
constants (if available) to be written as an entry in the yaml summary file. The
written force constants will reflect the required/available format used during
processing. So if `--full-fc` is set the entire matrix will be written.

#### `force sets`

The `--include-fs` flag or setting `INCLUDE_FS = .TRUE.` will cause the force
sets (if available) to be written as an entry in the yaml summary file.

#### `dielectric constant` and `born effective charge`

The `--include-born` flag or setting `INCLUDE_BORN = .TRUE.` will cause the born
effective charges and dielectric tensor (if available) to be written as an entry
in the yaml summary file. The values will only be written if non-analytical term
correction is set with the `--nac` flag or by setting `NAC = .TRUE.`.

This is more convenient than keeping track of the `BORN` file created by the
user.

#### `displacements`

The `--include-disp` flag or setting `INCLUDE_DISP = .TRUE.` will cause
displacements data (if available) to be written as an entry in the yaml summary
file.

This is set by default when the `phonopy` script is run in `displacements` mode.

#### `all`

All available data covered by the other `include` flags can be written to the
yaml summary file using the `--include-all` flag or by setting
`INCLUDE_ALL = .TRUE.`. Force constants are not stored when force sets are
stored.
