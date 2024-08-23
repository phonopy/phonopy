(formulations)=

# Formulations

## Second-order force constants

Potential energy of phonon system is represented as functions of atomic
positions:

```{math}
V[\mathbf{r}(j_1 l_1),\ldots,\mathbf{r}(j_n l_N)],
```

where {math}`\mathbf{r}(jl)` is the point of the {math}`j`-th atom in the
{math}`l`-th unit cell and {math}`n` and {math}`N` are the number of atoms in a
unit cell and the number of unit cells, respectively. A force and a second-order
force constant {math}`\Phi_{\alpha \beta}` are given by

```{math}
F_\alpha(jl) = -\frac{\partial V }{\partial r_\alpha(jl)}
```

and

```{math}
\Phi_{\alpha\beta}(jl, j'l') = \frac{\partial^2 V}{\partial r_\alpha(jl)
\partial r_\beta(j'l')} = -\frac{\partial F_\beta(j'l')}{\partial
r_\alpha(jl)},
```

respectively, where {math}`\alpha`, {math}`\beta`, ..., are the Cartesian
indices, {math}`j`, {math}`j'`, ..., are the indices of atoms in a unit cell,
and {math}`l`, {math}`l'`, ..., are the indices of unit cells. In the finite
displacement method, the equation for the force constants is approximated as

```{math}
\Phi_{\alpha\beta}(jl, j'l') \simeq -\frac{ F_\beta(j'l';\Delta
r_\alpha{(jl)}) - F_\beta(j'l')} {\Delta r_\alpha(jl)},
```

where {math}`F_\beta(j'l'; \Delta r_\alpha{(jl)})` are the forces on atoms with
a finite displacement {math}`\Delta r_\alpha{(jl)}` and usually
{math}`F_\beta(j'l') \equiv 0`.

(force_constants_solver_theory)=

## Modified Parlinski-Li-Kawazoe method

The following is a modified and simplified version of the Parlinski-Li-Kawazoe
method, which is just a numerical fitting approach to obtain force constants
from forces and displacements.

The last equation above is represented by matrices as

```{math}
\mathbf{F} = - \mathbf{U} \mathbf{P},
```

where {math}`\mathbf{F}`, {math}`\mathbf{P}`, and {math}`\mathbf{U}` for a pair
of atoms, e.g. {math}`\{jl, j'l'\}`, are given by

```{math}
\mathbf{F} = \begin{pmatrix} F_{x} & F_{y} & F_{z} \end{pmatrix},
```

```{math}
\mathbf{P} = \begin{pmatrix} \Phi_{xx} & \Phi_{xy} & \Phi_{xz} \\
\Phi_{yx} & \Phi_{yy} & \Phi_{yz} \\ \Phi_{zx} & \Phi_{zy} & \Phi_{zz}
\end{pmatrix},
```

```{math}
\mathbf{U} = \begin{pmatrix} \Delta r_{x} & \Delta r_{y} & \Delta r_{z} \\
\end{pmatrix}.
```

The matrix equation is expanded for number of forces and displacements as
follows:

```{math}
\begin{pmatrix} \mathbf{F}_1 \\ \mathbf{F}_2 \\ \vdots \end{pmatrix} = -
\begin{pmatrix} \mathbf{U}_1 \\ \mathbf{U}_2 \\ \vdots \end{pmatrix}
\mathbf{P}.
```

With sufficient number of atomic displacements, this may be solved by pseudo
inverse such as

```{math}
\mathbf{P} = - \begin{pmatrix} \mathbf{U}_1 \\ \mathbf{U}_2 \\ \vdots
\end{pmatrix}^{+} \begin{pmatrix} \mathbf{F}_1 \\ \mathbf{F}_2 \\ \vdots
\end{pmatrix}.
```

Required number of atomic displacements to solve the simultaneous equations may
be reduced using site-point symmetries. The matrix equation can be written using
a symmetry operation as

```{math}
\hat{R}(\mathbf{F}) = -\hat{R}(\mathbf{U})\mathbf{P},
```

where {math}`\hat{R}` is the site symmetry operation centring at
{math}`\mathbf{r}(jl)`. {math}`\hat{R}(\mathbf{F})` and
{math}`\hat{R}(\mathbf{U})` are defined as
{math}`\mathbf{RF}(\hat{R^{-1}}(j'l'))` and {math}`\mathbf{RU}`, respectively,
where {math}`\mathbf{R}` is the matrix representation of the rotation operation.
The combined simultaneous equations are built such as

```{math}
\begin{pmatrix} \mathbf{F}^{(1)}_1 \\ \mathbf{F}^{(2)}_1 \\ \vdots \\
\mathbf{F}^{(1)}_2 \\ \mathbf{F}^{(2)}_2 \\ \vdots \end{pmatrix} = -
\begin{pmatrix} \mathbf{U}^{(1)}_1 \\ \vdots \\ \mathbf{U}^{(2)}_1 \\
\mathbf{U}^{(1)}_2 \\ \mathbf{U}^{(2)}_2 \\ \vdots \end{pmatrix} \mathbf{P}.
```

where the superscript with parenthesis gives the index of site-symmetry
operations. This is solved by pseudo inverse.

(dynacmial_matrix_theory)=

## Dynamical matrix

In phonopy, a phase convention of dynamical matrix is used as follows:

```{math}
:label: eq_dynmat
D_{\alpha\beta}(jj',\mathbf{q}) = \frac{1}{\sqrt{m_j m_{j'}}} \sum_{l'}
\Phi_{\alpha\beta}(j0, j'l')
\exp(i\mathbf{q}\cdot[\mathbf{r}(j'l')-\mathbf{r}(j0)]),
```

where {math}`m` is the atomic mass and {math}`\mathbf{q}` is the wave vector. An
equation of motion is written as

```{math}
\sum_{j'\beta} D_{\alpha\beta}(jj',\mathbf{q}) e_\beta(j', \mathbf{q}\nu) =
[ \omega(\mathbf{q}\nu) ]^2 e_\alpha(j, \mathbf{q}\nu).
```

where the eigenvector of the band index {math}`\nu` at {math}`\mathbf{q}` is
obtained by the diagonalization of {math}`\mathbf{D}(\mathbf{q})`:

```{math}
\sum_{j \alpha j' \beta}e_\alpha(j',\mathbf{q}\nu)^*
D_{\alpha\beta}(jj',\mathbf{q}) e_\beta(j',\mathbf{q}\nu') =
[\omega(\mathbf{q}\nu)]^2 \delta_{\nu\nu'}.
```

The atomic displacements {math}`\mathbf{u}` are given as

```{math}
u_\alpha(jl,t) = \left(\frac{\hbar}{2Nm_j}\right)^{\frac{1}{2}}
\sum_{\mathbf{q},\nu}\left[\omega(\mathbf{q}\nu)\right]^{-\frac{1}{2}}
\left[\hat{a}(\mathbf{q}\nu)\exp(-i\omega(\mathbf{q}\nu)t)+
\hat{a}^\dagger(\mathbf{-q}\nu)\exp({i\omega(\mathbf{q}\nu)}t)\right]
\exp({i\mathbf{q}\cdot\mathbf{r}(jl)}) e_\alpha(j,\mathbf{q}\nu),
```

where {math}`\hat{a}^\dagger` and {math}`\hat{a}` are the creation and
annihilation operators of phonon, {math}`\hbar` is the reduced Planck constant,
and {math}`t` is the time.

(non_analytical_term_correction_theory)=

## Non-analytical term correction

To treat long range interaction of macroscopic electric field induced by
polarization of collective ionic motions near the {math}`\Gamma`-point,
non-analytical term is added to dynamical matrix ({ref}`reference_NAC`). At
{math}`\mathbf{q}\to\mathbf{0}`, the dynamical matrix with non-analytical term
is given by,

```{math}
D_{\alpha\beta}(jj',\mathbf{q}\to \mathbf{0}) =
D_{\alpha\beta}(jj',\mathbf{q}=\mathbf{0}) + \frac{1}{\sqrt{m_j m_{j'}}}
\frac{4\pi}{\Omega_0}
\frac{\left[\sum_{\gamma}q_{\gamma}Z^{*}_{j,\gamma\alpha}\right]
\left[\sum_{\gamma'}q_{\gamma'}Z^{*}_{j',\gamma'\beta}\right]}
{\sum_{\alpha\beta}q_{\alpha}\epsilon_{\alpha\beta}^{\infty} q_{\beta}}.
```

Phonon frequencies at general **q**-points with long-range dipole-dipole
interaction are calculated by the method of Gonze _et al._
({ref}`reference_dp_dp_NAC`).

(thermal_properties_expressions)=

## Thermodynamic properties

### Phonon number

```{math}
n = \frac{1}{\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)-1}
```

### Harmonic phonon energy

```{math}
E = \sum_{\mathbf{q}\nu}\hbar\omega(\mathbf{q}\nu)\left[\frac{1}{2} +
\frac{1}{\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)-1}\right]
```

### Constant volume heat capacity

```{math}
C_V &= \left(\frac{\partial E}{\partial T} \right)_V \\
      &= \sum_{\mathbf{q}\nu} k_\mathrm{B}
   \left(\frac{\hbar\omega(\mathbf{q}\nu)}{k_\mathrm{B} T} \right)^2
   \frac{\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B}
   T)}{[\exp(\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)-1]^2}
```

### Partition function

```{math}
Z = \exp(-\varphi/k_\mathrm{B} T) \prod_{\mathbf{q}\nu}
   \frac{\exp(-\hbar\omega(\mathbf{q}\nu)/2k_\mathrm{B}
   T)}{1-\exp(-\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T)}
```

### Helmholtz free energy

```{math}
F &= -k_\mathrm{B} T \ln Z \\
&= \varphi + \frac{1}{2} \sum_{\mathbf{q}\nu}
\hbar\omega(\mathbf{q}\nu) + k_\mathrm{B} T \sum_{\mathbf{q}\nu} \ln
\bigl[1 -\exp(-\hbar\omega(\mathbf{q}\nu)/k_\mathrm{B} T) \bigr]
```

### Entropy

```{math}
S &= -\frac{\partial F}{\partial T} \\ &= \frac{1}{2T}
\sum_{\mathbf{q}\nu} \hbar\omega(\mathbf{q}\nu)
\coth(\hbar\omega(\mathbf{q}\nu)/2k_\mathrm{B}T)-k_\mathrm{B}
\sum_{\mathbf{q}\nu}
\ln\left[2\sinh(\hbar\omega(\mathbf{q}\nu)/2k_\mathrm{B}T)\right]
```

(thermal_displacement)=

## Thermal displacement

### Mean square displacement

From Eq. (10.71) in the book "Thermodynamics of Crystal", atomic displacement,
**u**, is written by

```{math}
u^\alpha(jl,t) = \left(\frac{\hbar}{2Nm_j}\right)^{\frac{1}{2}}
\sum_{\mathbf{q},\nu}\left[\omega_\nu(\mathbf{q})\right]^{-\frac{1}{2}}
\left[\hat{a}_\nu(\mathbf{q})\exp(-i\omega_\nu(\mathbf{q})t)+
\hat{a}^\dagger_\nu(\mathbf{-q})\exp({i\omega_\nu(\mathbf{q})}t)\right]
\exp({i\mathbf{q}\cdot\mathbf{r}(jl)})
e^\alpha_\nu(j,\mathbf{q})
```

where _j_ and _l_ are the labels for the _j_-th atomic position in the _l_-th
unit cell, _t_ is the time, {math}`\alpha` is an axis (a Cartesian axis in the
default behavior of phonopy), _m_ is the atomic mass, _N_ is the number of the
unit cells, {math}`\mathbf{q}` is the wave vector, {math}`\nu` is the index of
phonon mode. _e_ is the polarization vector of the atom _jl_ and the band
{math}`\nu` at {math}`\mathbf{q}`. {math}`\mathbf{r}(jl)` is the atomic position
and {math}`\omega` is the phonon frequency. {math}`\hat{a}^\dagger` and
{math}`\hat{a}` are the creation and annihilation operators of phonon. The
expectation value of the squared atomic displacement is calculated as,

```{math}
\left\langle |u^\alpha(jl, t)|^2 \right\rangle = \frac{\hbar}{2Nm_j}
\sum_{\mathbf{q},\nu}\omega_\nu(\mathbf{q})^{-1}
(1+2n_\nu(\mathbf{q},T))|e^\alpha_\nu(j,\mathbf{q})|^2,
```

where {math}`n_\nu(\mathbf{q},T)` is the phonon population, which is give by,

```{math}
n_\nu(\mathbf{q},T) =
\frac{1}{\exp(\hbar\omega_\nu(\mathbf{q})/\mathrm{k_B}T)-1},
```

where _T_ is the temperature, and {math}`\mathrm{k_B}` is the Boltzmann
constant. The equation is calculated using the commutation relation of the
creation and annihilation operators and the expectation values of the
combination of the operations, e.g.,

```{math}
[ \hat{a}_\nu(\mathbf{q}), \hat{a}^\dagger_{\nu'}(\mathbf{q'}) ] &=
\delta(\mathbf{q}-\mathbf{q}')\delta_{\nu\nu'},\\ [ \hat{a}_\nu(\mathbf{q}),
\hat{a}_{\nu'}(\mathbf{q'}) ] &= 0,\\ [ \hat{a}^\dagger_\nu(\mathbf{q}),
\hat{a}^\dagger_{\nu'}(\mathbf{q'}) ] &= 0,\\
\langle|\hat{a}_\nu(\mathbf{q})\hat{a}_{\nu'}(\mathbf{q'})|\rangle &= 0,\\
\langle|\hat{a}^\dagger_\nu(\mathbf{q})\hat{a}^\dagger_{\nu'}(\mathbf{q'})|\rangle
&= 0.
```

(thermal_displacement_matrix)=

### Mean square displacement matrix

Mean square displacement matrix is defined as follows:

```{math}
\mathrm{U}_\text{cart}(j, T) = \frac{\hbar}{2Nm_j}
\sum_{\mathbf{q},\nu}\omega_\nu(\mathbf{q})^{-1} (1+2n_\nu(\mathbf{q},T))
\mathbf{e}_\nu(j,\mathbf{q}) \otimes \mathbf{e}^*_\nu(j,\mathbf{q}).
```

This is a symmetry matrix and diagonal elements are same as mean square
displacement calculated along Cartesian x, y, z directions.

### Projection to an arbitrary axis

In phonopy, eigenvectors are calculated in the Cartesian axes that are defined
in the input structure file. Mean square displacement along an arbitrary axis is
obtained projecting eigenvectors in the Cartesian axes as follows:

```{math}
\left\langle |u(jl, t)|^2 \right\rangle = \frac{\hbar}{2Nm_j}
\sum_{\mathbf{q},\nu}\omega_\nu(\mathbf{q})^{-1} (1+2n_\nu(\mathbf{q},T))|
\hat{\mathbf{n}}\cdot\mathbf{e}_\nu(j,\mathbf{q})|^2
```

where {math}`\hat{\mathbf{n}}` is an arbitrary unit direction.

### Mean square displacement matrix in cif format

According to the paper by Grosse-Kunstleve and Adams [J. Appl. Cryst., 35,
477-480 (2002)], mean square displacement matrix in the cif definition
(`aniso_U`), {math}`\mathrm{U}_\text{cif}`, is obtained by

```{math}
\mathrm{U}_\text{cif} = (\mathrm{AN})^{-1}\mathrm{U}_\text{cart}
(\mathrm{AN})^{-\mathrm{T}},
```

where {math}`\mathrm{A}` is the matrix to transform a point in fractional
coordinates to the Cartesian coordinates and {math}`\mathrm{N}` is the diagonal
matrix made of reciprocal basis vector lengths as follows:

```{math}
\mathrm{A} = \begin{pmatrix} a_x & b_x & c_x \\ a_y & b_y & c_y \\ a_z & b_z &
c_z \end{pmatrix}
```

and

```{math}
\mathrm{N} = \begin{pmatrix} a^* & 0 & 0 \\ 0 & b^* & 0 \\ 0 & 0 & c^*
\end{pmatrix}.
```

{math}`a^*`, {math}`b^*`, {math}`c^*` are defined without {math}`2\pi`.

(group_velocity)=

## Group velocity

### Method

Phonopy calculates group velocity of phonon as follows:

```{math}
\mathbf{v}_\mathrm{g}(\mathbf{q}\nu) = & \nabla_\mathbf{q} \omega(\mathbf{q}\nu) \\
=&\frac{\partial\omega(\mathbf{q}\nu)}{\partial \mathbf{q}} \\
=&\frac{1}{2\omega(\mathbf{q}\nu)}\frac{\partial[\omega(\mathbf{q}\nu)]^2}{\partial
\mathbf{q}} \\
=&\frac{1}{2\omega(\mathbf{q}\nu)}\left<\mathbf{e}(\mathbf{q}\nu)\biggl|
\frac{\partial D(\mathbf{q})} {\partial
\mathbf{q}}\biggl|\mathbf{e}(\mathbf{q}\nu)\right>,
```

where the meanings of the variables are found at {ref}`formulations`.

### Finite difference method

In the previous versions, group velocity was calculated using finite difference
method:

```{math}
\mathbf{v}_\mathrm{g}(\mathbf{q}\nu) =
\frac{1}{2\omega(\mathbf{q}\nu)}\left<\mathbf{e}(\mathbf{q}\nu)\biggl|
\frac{\partial D(\mathbf{q})} {\partial
\mathbf{q}}\biggl|\mathbf{e}(\mathbf{q}\nu)\right> \simeq
\frac{1}{2\omega(\mathbf{q}\nu)} \left<\mathbf{e}(\mathbf{q}\nu)\biggl|
\frac{\Delta D(\mathbf{q})} {\Delta
\mathbf{q}}\biggl|\mathbf{e}(\mathbf{q}\nu)\right>.
```

Group velocity calculation with the finite difference method is still able to be
activated using `GV_DELTA_Q` tag or `--gv_delta_q` option.
{math}`\Delta\mathbf{q} = (\Delta q_x, \Delta q_y, \Delta q_z)` is described in
Cartesian coordinated in reciprocal space. In the implementation, central
difference is employed, and {math}`+\Delta q_\alpha` and
{math}`-\Delta q_\alpha` are taken to calculate group velocity, where
{math}`\alpha` is the Cartesian index in reciprocal space.
{math}`\Delta q_\alpha` is specified in the unit of reciprocal space distance
 by `--gv_delta_q` option or
`GV_DELTA_Q` tag.

(physical_unit_conversion)=

## Physical unit conversion

Phonopy calculates phonon frequencies based on input values from users. In the
default case, the physical units of distance, atomic mass, force, and force
constants are supposed to be {math}`\text{Angstrom}`, {math}`\text{AMU}`,
{math}`\text{eV/Angstrom}`, and {math}`\text{eV/Angstrom}^2`, respectively, and
the physical unit of the phonon frequency is converted to THz. This conversion
is made as follows:

Internally phonon frequency has the physical unit of
{math}`\sqrt{\text{eV/}(\text{Angstrom}^2\cdot \text{AMU})}` in angular
frequency. To convert this unit to THz (not angular frequency), the calculation
of `sqrt(EV/AMU)/Angstrom/(2*pi)/1e12` is made. `EV`, `AMU`, `Angstrom` are the
values to convert them to those in the SI base unit, i.e., to Joule, kg, and
metre, respectively. These values implemented in phonopy are found at
[a phonopy github page](https://github.com/phonopy/phonopy/blob/master/phonopy/units.py).
This unit conversion factor can be manually specified. See
{ref}`frequency_conversion_factor_tag`.

The unit conversion factor in the `BORN` file is multiplied with the second term
of the right hand side of the equation in
{ref}`non_analytical_term_correction_theory` where this equation is written in
atomic units ({ref}`Gonze and Lee, 1997 <reference_NAC>`). The physical unit of
the part of the equation corresponding to force constants:

```{math}
\frac{4\pi}{\Omega_0}
\frac{[\sum_{\gamma}q_{\gamma}Z^{*}_{j,\gamma\alpha}]
[\sum_{\gamma'}q_{\gamma'}Z^{*}_{j',\gamma'\beta}]}
{\sum_{\alpha\beta}q_{\alpha}\epsilon_{\alpha\beta}^{\infty} q_{\beta}}.
```

is {math}`[\text{hartree}/\text{bohr}^2]`. In the default case for the VASP
interface, internally {math}`\Omega_0` is given in {math}`\text{Angstrom}^3`,
while Born charges {math}`Z^{*}` have physical unit of charge ({math}`e_{0}`).
Normally, physical units of Born charges obtained in many calculators are atomic
ones. In atomic units Hartree energy is equal to
{math}`1 \text{Ha} = e_{0}^{2}/a_{0}` ({math}`a_{0}` is Bohr radius), therefore
the units of charge squared is
{math}`e_{0}^{2}=[\text{hartree}\cdot \text{bohr}]`. In case for the VASP in
order to convert the nonanalytical term, one has to convert units of energy (Ha
to eV) and units of distance (Bohr to Angstrom). In total, the necessary unit
conversion is
{math}`(\text{hartree} \rightarrow \text{eV}) \times (\text{bohr} \rightarrow \text{Angstrom})=14.4`.
In the default case of the Wien2k interface, the conversion factor is
{math}`(\text{hartree} \rightarrow \text{mRy})=2000`. For the other interfaces,
the conversion factors are similarly calculated following the unit systems
employed in phonopy ({ref}`calculator_interfaces`).

(definition_of_commensurate_points)=

## Crystal structure

### Coordinates in direct and reciprocal spaces

As usual, in phonopy, the Born-von Karman boundary condition is assumed. Basis
vectors of a primitive lattice are defined in three column vectors
{math}`( \mathbf{a} \; \mathbf{b} \; \mathbf{c} )`. Coordinates of a point in
the direct space {math}`\mathbf{r}` is represented with respect to these basis
vectors. The direct lattice points are given by
{math}`i \mathbf{a} + j \mathbf{b} + k \mathbf{a}, \{i, j, k \in \mathbb{Z}\}`,
and the points for atoms in a unit cell
{math}`x \mathbf{a} + y \mathbf{b} + z \mathbf{a}, \{0 \le x, y, z < 1\}`. Basis
vectors of the reciprocal lattice may be given by three row vectors,
{math}`( \mathbf{a}^{*T} /\; \mathbf{b}^{*T} /\; \mathbf{c}^{*T} )`, but here
they are defined as three column vectors as
{math}`( \mathbf{a}^{*} \; \mathbf{b}^{*} \; \mathbf{c}^{*} )` with

```{math}
:label: eq_rec_basis_vectors

\mathbf{a}^{*} &= \frac{\mathbf{b} \times \mathbf{c}}{\mathbf{a} \cdot
(\mathbf{b} \times \mathbf{c})}, \\ \mathbf{b}^{*} &= \frac{\mathbf{c} \times
\mathbf{a}}{\mathbf{b} \cdot (\mathbf{c} \times \mathbf{a})}, \\ \mathbf{c}^{*}
&= \frac{\mathbf{a} \times \mathbf{b}}{\mathbf{c} \cdot (\mathbf{a} \times
\mathbf{b})}.
```

Coordinates of a point in the reciprocal space {math}`\mathbf{q}` is represented
with respect to these basis vectors, therefore
{math}`q_x \mathbf{a}^{*} + q_y \mathbf{b}^{*} + q_z \mathbf{c}^{*}`. The
reciprocal lattice points are given by
{math}`G_x\mathbf{a}^{*} + G_y \mathbf{b}^{*} + G_z \mathbf{c}^{*}, \{G_x, G_y, G_z \in \mathbb{Z}\}`.
Following these definition, phase factor should be represented as
{math}`\exp(2\pi i\mathbf{q}\cdot\mathbf{r})`, however in phonopy documentation,
{math}`2\pi` is implicitly included and not shown, i.e., it is represented like
{math}`\exp(i\mathbf{q}\cdot\mathbf{r})` (e.g., see Eq. {eq}`eq_dynmat`). In the
output of the reciprocal basis vectors, {math}`2\pi` is not included, e.g., in
`band.yaml`.

In phonopy, unless {ref}`primitive_axes_tag` (or `--pa` option) is specified,
basis vectors in direct space {math}`( \mathbf{a} \; \mathbf{b} \; \mathbf{c})`
are set from the input unit cell structure even if it is a supercell or a
conventional unit cell having centring, therefore the basis vectors in the
reciprocal space are given by Eq. {eq}`eq_rec_basis_vectors`. When using
{ref}`primitive_axes_tag`, {math}`( \mathbf{a} \; \mathbf{b} \; \mathbf{c})` are
set from those transformed by the transformation matrix {math}`M_\text{p}` as
written at {ref}`primitive_axes_tag`, therefore
{math}`( \mathbf{a}^{*} \; \mathbf{b}^{*} \; \mathbf{c}^{*} )` are given by
those calculated following Eq. {eq}`eq_rec_basis_vectors` with this
{math}`( \mathbf{a} \; \mathbf{b} \; \mathbf{c})`.

### Commensurate points

In phonopy, so-called commensurate points mean the q-points whose waves are
confined in the supercell used in the phonon calculation.

To explain about the commensurate points, let basis vectors of a primitive cell
in direct space cell be the column vectors
{math}`(\mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \; \mathbf{c}_\mathrm{p})`
and those of the supercell be
{math}`(\mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \; \mathbf{c}_\mathrm{s})`.
The transformation of the basis vectors from the primitive cell to the supercell
is written as

```{math}
( \mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \; \mathbf{c}_\mathrm{s} )
=  ( \mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \;
\mathbf{c}_\mathrm{p} ) \boldsymbol{P}.
```

{math}`\boldsymbol{P}` is given as a {math}`3\times 3` matrix and its elements
are all integers, which is a constraint we have. The resolution for q-points
being the commensurate points is determined by {math}`\boldsymbol{P}` since one
period of a wave has to be bound by any of lattice points inside the supercell.
Therefore the number of commensurate points becomes the same as the number of
the primitive cell that can be contained in the supercell, i.e.,
{math}`\det(\boldsymbol{P})`.

Then let the basis vectors in reciprocal space be the column vectors
{math}`(\mathbf{a}^*_\mathrm{p} \; \mathbf{b}^*_\mathrm{p} \; \mathbf{c}^*_\mathrm{p})`.
Note that often reciprocal vectors are deifned by row vectors, but column
vectors are chosen here to formulate. Formally we see the set of besis vectors
are {math}`3\times 3` matrices, we have the following relation:

```{math}
( \mathbf{a}^*_\mathrm{p} \;
\mathbf{b}^*_\mathrm{p} \; \mathbf{c}^*_\mathrm{p} ) = (
\mathbf{a}_\mathrm{p} \; \mathbf{b}_\mathrm{p} \;
\mathbf{c}_\mathrm{p} )^{-\mathbf{T}}.
```

Similarly for the supercell, we define a relation

```{math}
( \mathbf{a}^*_\mathrm{s} \;
\mathbf{b}^*_\mathrm{s} \; \mathbf{c}^*_\mathrm{s} ) = (
\mathbf{a}_\mathrm{s} \; \mathbf{b}_\mathrm{s} \;
\mathbf{c}_\mathrm{s} )^{-\mathbf{T}}.
```

Then

```{math}
( \mathbf{a}^*_\mathrm{s} \; \mathbf{b}^*_\mathrm{s} \;
\mathbf{c}^*_\mathrm{s} ) \boldsymbol{P}^{\mathrm{T}} = (
\mathbf{a}^*_\mathrm{p} \; \mathbf{b}^*_\mathrm{p} \;
\mathbf{c}^*_\mathrm{p} ).
```

To multiply an arbitrary q-point {math}`\mathbf{q}` on both sides

```{math}
( \mathbf{a}^*_\mathrm{s} \; \mathbf{b}^*_\mathrm{s} \;
\mathbf{c}^*_\mathrm{s} ) \boldsymbol{P}^{\mathrm{T}} \mathbf{q} = (
\mathbf{a}^*_\mathrm{p} \; \mathbf{b}^*_\mathrm{p} \;
\mathbf{c}^*_\mathrm{p} ) \mathbf{q},
```

we find the constraint of a q-point being one of the commensurate points is the
elements of {math}`\boldsymbol{P}^{\mathrm{T}} \mathbf{q}` to be integers.
