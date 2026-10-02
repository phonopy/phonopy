(symmetry_adapted_modes)=
# Symmetry-adapted phonon modes

This page gives the equations that `phonopy/phonon/symmetry_adapted_modes.py`
implements. The module takes the dynamical matrix at one q-point and returns the
phonon frequencies, the eigenvectors and the degenerate sets. The degenerate
sets are decided by the symmetry of the q-point, including time reversal.
Closeness of frequencies is not used.

The module is used by the irreducible-representation calculation when the
`IRREPS_SYMMETRY_ADAPTED` tag ({ref}`irreps_symmetry_adapted_tag`) is `.TRUE.`,
the `--irreps-symmetry-adapted` option is given, or `Phonopy.run_irreps` is
called with `symmetry_adapted=True`.

Frequencies are a poor guide to degeneracy. The three acoustic modes at
{math}`\Gamma` of a crystal with space group P2 have frequencies that differ by
a small amount after the force constants are symmetrized, and a frequency
tolerance can either merge them or split a two-dimensional representation. The
representation of the little group of q separates the two cases, because modes
that belong to different representations never mix under the symmetry
operations.

## Notation

The symbols on this page are defined here and are used only here.

- {math}`N` is the number of atoms in the primitive cell. The dynamical matrix
  is a {math}`3N\times3N` matrix.
- {math}`\mathbf{r}(jl)=\mathbf{r}(l)+\mathbf{r}_{j0}` is the position of atom
  {math}`j` in unit cell {math}`l`. {math}`\mathbf{r}(l)` is the lattice point
  and {math}`\mathbf{r}_{j0}` is the position of the atom relative to the lattice
  point.
- {math}`L=(\mathbf{a}_1\ \mathbf{a}_2\ \mathbf{a}_3)` is the matrix whose
  columns are the basis vectors. In phonopy, `primitive.cell` stores the basis
  vectors as rows, so {math}`L` is `primitive.cell.T`.
- {math}`x_j` is the position of atom {math}`j` in crystallographic coordinates
  (a column vector), so that {math}`\mathbf{r}_{j0}=Lx_j`. It is row
  {math}`j` of `primitive.scaled_positions`.
- {math}`\tilde q` is the q-point in crystallographic coordinates of the
  reciprocal basis vectors (a row vector). It is the `qpoint` argument. The
  reciprocal basis vectors {math}`\mathbf{b}_k` satisfy
  {math}`\mathbf{a}_i\cdot\mathbf{b}_k=2\pi\delta_{ik}`, and
  {math}`\mathbf{q}\cdot\mathbf{r}_{j0}=2\pi\,\tilde q\,x_j`.
- {math}`\mathbf{G}` is a reciprocal lattice vector. Its crystallographic
  coordinates {math}`\tilde G` are integers.
- {math}`\mathrm{S}=\{\mathrm{R}|\tau\}` is a space-group operation,
  {math}`\mathbf{r}\mapsto\mathrm{R}\mathbf{r}+\tau`, with the rotation
  {math}`\mathrm{R}` and the translation {math}`\tau` in Cartesian coordinates.
  spglib returns the integer matrix {math}`\tilde R` and the translation
  {math}`t` in crystallographic coordinates. They are related by
  {math}`\mathrm{R}=L\tilde RL^{-1}` and {math}`\tau=Lt`.
- The rotated q-point {math}`\mathrm{R}\mathbf{q}` has the crystallographic
  coordinates {math}`\tilde q\tilde R^{-1}`.
- {math}`\Delta(\mathbf{L})` is 1 when {math}`\mathbf{L}` is a lattice vector
  and 0 otherwise.
- {math}`\pi_{\mathrm{S}}(j')=j` is the atom to which {math}`\mathrm{S}` sends
  atom {math}`j'`, defined by
  {math}`\Delta(\mathbf{r}_{j0}-\mathrm{S}\mathbf{r}_{j'0})=1`.
- {math}`\Theta` is complex conjugation.

## Phase convention of the dynamical matrix

phonopy builds the dynamical matrix with a phase that contains the atomic
positions. This convention is called C-type:

```{math}
D^{\mathrm{C}}_{\alpha\beta}(jj',\mathbf{q})=\frac{1}{\sqrt{m_jm_{j'}}}
\sum_{l'}\Phi_{\alpha\beta}(j0,j'l')
\exp\bigl(i\mathbf{q}\cdot[\mathbf{r}(j'l')-\mathbf{r}(j0)]\bigr).
```

The eigenvectors {math}`\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q})` satisfy
{math}`D^{\mathrm{C}}(\mathbf{q})\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q})=\omega_\nu^2\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q})`,
and the atomic displacements are proportional to
{math}`e^{\mathrm{C}}_\alpha(j,\mathbf{q})\exp(i\mathbf{q}\cdot\mathbf{r}(jl))`.
`symmetry_adapted_modes.py` expects a C-type dynamical matrix and returns C-type eigenvectors.

The other common convention puts only the lattice vectors in the phase. It is
called D-type:

```{math}
D^{\mathrm{D}}_{\alpha\beta}(jj',\mathbf{q})=\frac{1}{\sqrt{m_jm_{j'}}}
\sum_{l'}\Phi_{\alpha\beta}(j0,j'l')
\exp\bigl(i\mathbf{q}\cdot[\mathbf{r}(l')-\mathbf{r}(0)]\bigr).
```

`symmetry_adapted_modes.py` does not use the D-type matrix. The D-type matrix is defined here
because `RandomDisplacements` uses it internally, and a D-type matrix must not
be passed to `symmetry_adapted_modes.py`.

Let {math}`V(\mathbf{q})` be the {math}`3N\times3N` diagonal matrix

```{math}
V(\mathbf{q})=\operatorname{diag}\bigl(
e^{i\mathbf{q}\cdot\mathbf{r}_{10}},e^{i\mathbf{q}\cdot\mathbf{r}_{10}},e^{i\mathbf{q}\cdot\mathbf{r}_{10}},
\ldots,
e^{i\mathbf{q}\cdot\mathbf{r}_{N0}},e^{i\mathbf{q}\cdot\mathbf{r}_{N0}},e^{i\mathbf{q}\cdot\mathbf{r}_{N0}}
\bigr),
\qquad
V_{j\alpha,j'\beta}(\mathbf{q})=e^{i\mathbf{q}\cdot\mathbf{r}_{j0}}\,\delta_{jj'}\delta_{\alpha\beta}.
```

Each atom {math}`j` has the same factor on its three Cartesian components. The
two types are related by

```{math}
D^{\mathrm{C}}(\mathbf{q})=V^\dagger(\mathbf{q})\,D^{\mathrm{D}}(\mathbf{q})\,V(\mathbf{q}),
\qquad
\mathbf{e}^{\mathrm{D}}_\nu(\mathbf{q})=V(\mathbf{q})\,\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q}).
```

The D-type matrix is periodic in {math}`\mathbf{q}`,
{math}`D^{\mathrm{D}}(\mathbf{q}+\mathbf{G})=D^{\mathrm{D}}(\mathbf{q})`, because
{math}`e^{i\mathbf{G}\cdot\mathbf{r}(l')}=1`. The C-type matrix is not periodic.
Writing the same displacement pattern with the label
{math}`\mathbf{q}+\mathbf{G}` or with the label {math}`\mathbf{q}` gives

```{math}
:label: symmodes_ec_shift
\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q})
=V(\mathbf{G})\,\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q}+\mathbf{G}),
\qquad
D^{\mathrm{C}}(\mathbf{q})
=V(\mathbf{G})\,D^{\mathrm{C}}(\mathbf{q}+\mathbf{G})\,V^\dagger(\mathbf{G}).
```

The factor {math}`e^{i\mathbf{G}\cdot\mathbf{r}_{j0}}` differs from atom to atom,
so it is not an overall phase.

The force constants are real. Therefore the dynamical matrix satisfies

```{math}
:label: symmodes_dc_tr
D^{\mathrm{C}}(\mathbf{q})^*=D^{\mathrm{C}}(-\mathbf{q}),
```

and the complex conjugate of an eigenvector at {math}`\mathbf{q}` is an
eigenvector at {math}`-\mathbf{q}`. This is time-reversal symmetry.

(symmodes_little_group_of_q)=
## Little group of q

The little group of {math}`\mathbf{q}` is the set of space-group operations
{math}`\mathrm{S}` with {math}`\mathrm{R}\mathbf{q}=\mathbf{q}+\mathbf{G}`. The
module keeps one operation for each rotation, as spglib returns them for the
primitive cell. Lattice translations act on an eigenvector only as an overall
phase, so these representative operations are enough.

Time reversal adds operations. When
{math}`\mathrm{R}(-\mathbf{q})=\mathbf{q}+\mathbf{G}`, the product
{math}`\mathrm{S}\Theta` also leaves {math}`\mathbf{q}` unchanged.
{math}`\mathrm{S}\Theta` is antiunitary: it acts on an eigenvector as a matrix
times the complex conjugate.

Both kinds are written with a sign {math}`\eta`. {math}`\eta=+1` for a unitary
operation and {math}`\eta=-1` for an antiunitary one. An operation belongs to the
little group when

```{math}
:label: symmodes_little_group
\mathrm{R}(\eta\mathbf{q})=\mathbf{q}+\mathbf{G}_{\mathrm{S}},
\qquad\text{or}\qquad
\eta\,\tilde q\tilde R^{-1}-\tilde q\in\mathbb{Z}^3.
```

{math}`G_{\mathbf{q}}` denotes the unitary operations and
{math}`A_{\mathbf{q}}` the antiunitary ones.
`get_little_group_operations` returns {math}`G_{\mathbf{q}}` first and then
{math}`A_{\mathbf{q}}`.

{math}`G_{\mathbf{q}}` has one operation per rotation, so
{math}`|G_{\mathbf{q}}|` is the order of the point group of the little group.
It divides the order of the point group of the crystal, which is at most 48.
At a general q-point, {math}`G_{\mathbf{q}}` has only the identity.

{math}`|A_{\mathbf{q}}|` is either zero or {math}`|G_{\mathbf{q}}|`. When one
operation {math}`\mathrm{S}_0` sends {math}`\mathbf{q}` to
{math}`-\mathbf{q}+\mathbf{G}`, the antiunitary operations are exactly
{math}`\mathrm{S}_0\mathrm{S}\Theta` with {math}`\mathrm{S}\in G_{\mathbf{q}}`.
For any {math}`\mathrm{S}'\Theta` in {math}`A_{\mathbf{q}}`, the product
{math}`\mathrm{S}_0^{-1}\mathrm{S}'` leaves {math}`\mathbf{q}` unchanged and is
therefore in {math}`G_{\mathbf{q}}`. When no such {math}`\mathrm{S}_0` exists,
{math}`A_{\mathbf{q}}` is empty.

- In a crystal with inversion, the inversion is such an {math}`\mathrm{S}_0`
  at every q-point, and {math}`|A_{\mathbf{q}}|=|G_{\mathbf{q}}|`.
- At a q-point with {math}`-\mathbf{q}=\mathbf{q}+\mathbf{G}`, such as
  {math}`\Gamma` and the points with half-integer coordinates, the identity is
  such an {math}`\mathrm{S}_0`, and {math}`|A_{\mathbf{q}}|=|G_{\mathbf{q}}|`.
- In a crystal without inversion, at a q-point where {math}`\mathbf{q}` and
  {math}`-\mathbf{q}` are not equivalent, {math}`A_{\mathbf{q}}` is empty.

The table lists values from the structures in `test/phonon/test_symmetry_adapted_modes.py`.

| Space group | q-point                         | {math}`\|G_{\mathbf{q}}\|` | {math}`\|A_{\mathbf{q}}\|` |
| ----------- | ------------------------------- | -------------------------- | -------------------------- |
| P-43m       | {math}`\Gamma`                  | 24                         | 24                         |
| P-43m       | (0.1, 0.1, 0.1)                 | 6                          | 0                          |
| Pa-3        | {math}`\Gamma`                  | 24                         | 24                         |
| P-3m1       | (1/3, 1/3, 0)                   | 6                          | 6                          |
| P222_1      | (0.2, 0.3, 0.5)                 | 1                          | 1                          |
| P222_1      | (0.2, 0.3, 0.4)                 | 1                          | 0                          |

P-43m has no inversion, and along (0.1, 0.1, 0.1) the directions
{math}`[111]` and {math}`[\bar1\bar1\bar1]` are not equivalent under its point
group. At (0.2, 0.3, 0.5) of P222_1, the only unitary operation is the identity,
and the one antiunitary operation makes every band doubly degenerate.

## Representation matrix

A space-group operation sends a C-type eigenvector at {math}`\mathbf{q}` to an
eigenvector at {math}`\mathrm{R}\mathbf{q}` by the {math}`3N\times3N` matrix

```{math}
:label: symmodes_gamma
\Gamma^{\mathrm{C},\mathbf{q}}_{j\alpha,j'\beta}(\mathrm{S})
=\mathrm{R}_{\alpha\beta}\exp(-i\,\mathrm{R}\mathbf{q}\cdot\tau)\,
\Delta(\mathbf{r}_{j0}-\mathrm{S}\mathbf{r}_{j'0}),
\qquad
D^{\mathrm{C}}(\mathrm{R}\mathbf{q})
=\Gamma^{\mathrm{C},\mathbf{q}}(\mathrm{S})\,D^{\mathrm{C}}(\mathbf{q})\,
\Gamma^{\mathrm{C},\mathbf{q}}(\mathrm{S})^\dagger.
```

The vector {math}`\Gamma^{\mathrm{C},\mathbf{q}}(\mathrm{S})\mathbf{e}^{\mathrm{C}}_\nu(\mathbf{q})`
is an eigenvector of {math}`D^{\mathrm{C}}(\mathrm{R}\mathbf{q})` with the same
eigenvalue {math}`\omega_\nu^2`. It can differ from the eigenvector that a
diagonalization at {math}`\mathrm{R}\mathbf{q}` returns by an overall phase, or
by a unitary rotation inside a degenerate subspace.

For an operation of the little group, {math}`\mathrm{R}(\eta\mathbf{q})` is
{math}`\mathbf{q}+\mathbf{G}_{\mathrm{S}}`, and Eq. {eq}`symmodes_ec_shift`
brings the label back to {math}`\mathbf{q}`. The representation matrix used in
the module is

```{math}
:label: symmodes_t
T(\mathrm{S})=V(\mathbf{G}_{\mathrm{S}})\,\Gamma^{\mathrm{C},\eta\mathbf{q}}(\mathrm{S}).
```

It acts on eigenvectors and on matrices as follows.

| Operation  | Eigenvector                                            | Matrix                          |
| ---------- | ------------------------------------------------------ | ------------------------------- |
| Unitary    | {math}`\mathbf{e}\mapsto T\mathbf{e}`                  | {math}`M\mapsto TMT^\dagger`    |
| Antiunitary | {math}`\mathbf{e}\mapsto T\mathbf{e}^*`               | {math}`M\mapsto TM^*T^\dagger`  |

For an antiunitary operation, the complex conjugate moves the eigenvector from
{math}`\mathbf{q}` to {math}`-\mathbf{q}` by Eq. {eq}`symmodes_dc_tr`,
{math}`\Gamma^{\mathrm{C},-\mathbf{q}}(\mathrm{S})` moves it to
{math}`\mathbf{q}+\mathbf{G}_{\mathrm{S}}`, and {math}`V(\mathbf{G}_{\mathrm{S}})`
brings the label back to {math}`\mathbf{q}`. In both cases the dynamical matrix
is unchanged:
{math}`D^{\mathrm{C}}(\mathbf{q})=TD^{\mathrm{C}}(\mathbf{q})T^\dagger` or
{math}`D^{\mathrm{C}}(\mathbf{q})=TD^{\mathrm{C}}(\mathbf{q})^*T^\dagger`.

The matrix elements of {math}`T` are

```{math}
T_{j\alpha,j'\beta}=\mathrm{R}_{\alpha\beta}\,
\exp\bigl(i\,[\mathbf{G}_{\mathrm{S}}\cdot\mathbf{r}_{j0}-\mathbf{q}'\cdot\tau]\bigr)\,
\Delta(\mathbf{r}_{j0}-\mathrm{S}\mathbf{r}_{j'0}),
\qquad \mathbf{q}'=\mathrm{R}(\eta\mathbf{q})=\mathbf{q}+\mathbf{G}_{\mathrm{S}}.
```

In crystallographic coordinates, {math}`\tilde q'=\eta\,\tilde q\tilde R^{-1}`,
and the phase of the element that moves atom {math}`j'` to atom
{math}`j=\pi_{\mathrm{S}}(j')` is

```{math}
:label: symmodes_phase
\exp\bigl(2\pi i\,[(\tilde q'-\tilde q)\,x_j-\tilde q'\,t]\bigr).
```

This is `LittleGroupOperation.phases[j']`. The same phase can be written without
{math}`\tau`. With {math}`\mathrm{S}\mathbf{r}_{j'0}=\mathbf{r}_{j0}+\mathbf{L}`,
where {math}`\mathbf{L}` is the lattice vector that brings the image back into
the unit cell,

```{math}
T_{j\alpha,j'\beta}=\mathrm{R}_{\alpha\beta}\,
\exp\bigl(-i\,\mathbf{q}\cdot(\mathrm{S}\mathbf{r}_{j'0}-\eta\,\mathbf{r}_{j'0})\bigr)\,
\Delta(\mathbf{r}_{j0}-\mathrm{S}\mathbf{r}_{j'0}).
```

For a unitary operation, {math}`\mathrm{S}\mathbf{r}_{j'0}-\mathbf{r}_{j'0}` is
the whole vector by which {math}`\mathrm{S}` moves atom {math}`j'`, including the
lattice vector.

`LittleGroupOperation` stores {math}`T` without building the dense matrix.

| Field                | Content                                                         |
| -------------------- | --------------------------------------------------------------- |
| `permutation[j']`    | {math}`\pi_{\mathrm{S}}(j')`                                    |
| `phases[j']`         | Eq. {eq}`symmodes_phase`                                         |
| `rotation_cartesian` | {math}`\mathrm{R}`                                              |
| `rotation`           | {math}`\tilde R`                                                |
| `translation`        | {math}`t`                                                       |
| `is_antiunitary`     | `True` for {math}`\eta=-1`                                      |

## Symmetrized geometry

The matrices {math}`T` must form a group to round-off. Positions in an input
file are rounded, for example a coordinate of one third written with eight
digits, and the rounding breaks the group closure by a small amount. That amount
can exceed the tolerance used to find degenerate eigenvalues in the next
section, so the lattice and the positions are symmetrized before {math}`T` is
built. The sums below run over all {math}`n_{\mathrm{op}}` operations of the
space group, not only over the little group.

The lattice is symmetrized through its metric tensor
{math}`g=L^{\mathsf{T}}L`, whose elements are
{math}`g_{ik}=\mathbf{a}_i\cdot\mathbf{a}_k`. The Cartesian rotation
{math}`\mathrm{R}=L\tilde RL^{-1}` is orthogonal exactly when
{math}`\tilde R^{\mathsf{T}}g\tilde R=g`, so the metric tensor is averaged over
the rotations:

```{math}
g_{\mathrm{sym}}=\frac{1}{n_{\mathrm{op}}}\sum_s\tilde R_s^{\mathsf{T}}\,g\,\tilde R_s.
```

The basis vectors are then changed through the polar decomposition of
{math}`L`. The polar decomposition writes {math}`L=QP` with an orthogonal matrix
{math}`Q` and the symmetric positive-definite matrix
{math}`P=(L^{\mathsf{T}}L)^{1/2}=g^{1/2}`, so {math}`Q=Lg^{-1/2}`. The rotation
part {math}`Q` is kept, and the stretch part {math}`g^{1/2}` is replaced by
{math}`g_{\mathrm{sym}}^{1/2}`:

```{math}
L_{\mathrm{sym}}=Q\,g_{\mathrm{sym}}^{1/2}=L\,g^{-1/2}\,g_{\mathrm{sym}}^{1/2}.
```

The metric tensor of {math}`L_{\mathrm{sym}}` is {math}`g_{\mathrm{sym}}`, and
{math}`L_{\mathrm{sym}}` differs from {math}`L` by an amount of the order of
{math}`g_{\mathrm{sym}}-g`. It is not in general the basis closest to {math}`L`
among those with the metric tensor {math}`g_{\mathrm{sym}}`; that basis is given by the
orthogonal Procrustes problem and differs from {math}`L_{\mathrm{sym}}` when
{math}`g` and {math}`g_{\mathrm{sym}}` do not commute. The difference does not
matter here, because only the orthogonality of the rotations is needed. With
{math}`L_{\mathrm{sym}}`, the Cartesian rotation
{math}`\mathrm{R}=L_{\mathrm{sym}}\tilde RL_{\mathrm{sym}}^{-1}` is orthogonal to
round-off.

Each position is replaced by the average of its images over the space group,
which is the projection onto the totally symmetric part:

```{math}
\bar x_j=\frac{1}{n_{\mathrm{op}}}\sum_s
\bigl[\tilde R_sx_{j'}+t_s+n_s\bigr],\qquad j'=\pi_s^{-1}(j),
```

where the integer vector {math}`n_s=\operatorname{rint}(x_j-\tilde R_sx_{j'}-t_s)`
moves each image next to {math}`x_j`. The translations {math}`t_s` returned by
spglib are used as they are. The dynamical matrix is computed from the original
structure, and the difference is removed by the symmetrization of the dynamical
matrix described below.

## Decomposition into degenerate sets

`SymmetryAdaptedModes` finds the degenerate sets from the representation, then
diagonalizes the dynamical matrix inside each of them.

The group average of a {math}`3N\times3N` matrix {math}`M` is

```{math}
\langle M\rangle=\frac{1}{|G_{\mathbf{q}}|+|A_{\mathbf{q}}|}
\Bigl(\sum_{G_{\mathbf{q}}}T\,M\,T^\dagger+\sum_{A_{\mathbf{q}}}T\,M^*\,T^\dagger\Bigr).
```

Since {math}`|A_{\mathbf{q}}|` is either {math}`|G_{\mathbf{q}}|` or zero (see
{ref}`symmodes_little_group_of_q`), the denominator
{math}`|G_{\mathbf{q}}|+|A_{\mathbf{q}}|` is {math}`2|G_{\mathbf{q}}|` or
{math}`|G_{\mathbf{q}}|`. When {math}`A_{\mathbf{q}}` is empty, the second sum is
absent and time reversal adds no condition.

{math}`\langle M\rangle` is the projection of {math}`M` onto its totally
symmetric part, that is, the Wigner projection operator for the totally
symmetric irreducible representation, whose characters are 1 for all
operations. This group average is also called the Reynolds operator. Unlike the
projection operators for other irreducible representations, it needs no
character table, and it can include the antiunitary operations.
{math}`\langle M\rangle` commutes with every operation of the little group. The
steps are listed below.

1. A random Hermitian matrix {math}`Y` is drawn with a fixed seed, and
   {math}`X=\langle Y\rangle` is formed. The dynamical matrix is averaged in the
   same way, {math}`D_{\mathrm{sym}}=\langle D^{\mathrm{C}}(\mathbf{q})\rangle`.
2. {math}`X` is diagonalized. Eigenvalues that differ by less than a relative
   tolerance are grouped, and each group of eigenvectors spans one subspace
   {math}`U_k` ({math}`3N\times d_k`). {math}`X` commutes with the operations,
   so each subspace is invariant under them. For a random {math}`Y`, each
   subspace carries one irreducible representation of the little group, or one
   pair of representations joined by time reversal. The reason is given in
   {ref}`symmodes_why_irreducible`.
3. The character of subspace {math}`k` is computed for the unitary operations,
   {math}`\chi_k(\mathrm{S})=\operatorname{Tr}(U_k^\dagger T(\mathrm{S})U_k)`.
   Subspaces with the same dimension and the same characters carry the same
   representation and are collected into one type {math}`\mu`. The type has
   {math}`m_\mu` subspaces of dimension {math}`d_\mu`.
4. For each type, the subspaces are joined into
   {math}`B_\mu=(U_{k_1},\ldots,U_{k_{m_\mu}})` ({math}`3N\times m_\mu d_\mu`), and
   {math}`B_\mu^\dagger D_{\mathrm{sym}}B_\mu` is diagonalized. Its eigenvalues
   come in runs of {math}`d_\mu` equal values. Each run of {math}`d_\mu`
   consecutive eigenvalues is one degenerate set.
5. Within each type, the eigenvectors of every set except the first are
   rotated inside the set, so that all sets of the type are transformed by the
   same matrices. The procedure is given in {ref}`symmodes_aligning_sets`.
6. The sets of all types are sorted by eigenvalue, and the frequencies are
   {math}`\operatorname{sgn}(\omega^2)\sqrt{|\omega^2|}` times the unit
   conversion factor.

(symmodes_why_irreducible)=
### Why the eigenspaces of X are irreducible

The goal of steps 1 and 2 is to split the {math}`3N`-dimensional space of
eigenvectors into subspaces that each carry one irreducible representation. A
projection operator for each irreducible representation would do this, but it
needs the character table of the little group. The random matrix
{math}`X` does the same without a character table.

{math}`X` commutes with every operation, {math}`TXT^\dagger=X`. If
{math}`X\mathbf{u}=\lambda\mathbf{u}`, then

```{math}
X(T\mathbf{u})=TX\mathbf{u}=\lambda\,T\mathbf{u},
```

so {math}`T\mathbf{u}` is an eigenvector with the same eigenvalue. Each
eigenspace of {math}`X` is therefore closed under all the operations.

The eigenspaces are also irreducible when {math}`X` is generic. By Schur's
lemma, a matrix that commutes with a unitary representation has the block form

```{math}
X=\bigoplus_\mu X_\mu\otimes I_{d_\mu},
```

where {math}`\mu` runs over the irreducible representations, {math}`d_\mu` is
the dimension of {math}`\mu`, and {math}`X_\mu` is a Hermitian
{math}`m_\mu\times m_\mu` matrix, with {math}`m_\mu` the number of times
{math}`\mu` appears in the {math}`3N`-dimensional space. Each appearance of
{math}`\mu` is one irreducible component of the representation. At
{math}`\Gamma` of NaCl, for example, the six-dimensional representation is
{math}`T_{1u}\oplus T_{1u}`: the three acoustic modes and the three optical
modes are two irreducible components of the same irreducible representation
{math}`T_{1u}`.

Each eigenvalue of {math}`X_\mu` appears {math}`d_\mu` times in {math}`X`, and
its eigenspace is one irreducible component of {math}`\mu`. Two components merge
into one eigenspace only when two eigenvalues coincide, either inside one
{math}`X_\mu` or between {math}`X_\mu` and {math}`X_\nu`. For a random
{math}`X` this happens with probability zero.

The group average makes a random {math}`X` of this form. Every matrix
{math}`X'` that commutes with the operations satisfies
{math}`\langle X'\rangle=X'`, so the average maps a random Hermitian
{math}`Y` onto a random element among all the commuting matrices.

A small case shows the result. Two atoms on a line are exchanged by inversion,
and {math}`T` is the matrix that swaps them. The Hermitian matrices that commute
with the swap have the form {math}`\begin{pmatrix}a&b\\b&a\end{pmatrix}`. For any
{math}`a` and any {math}`b\neq0`, the eigenvectors are
{math}`(1,1)/\sqrt2` and {math}`(1,-1)/\sqrt2`, the mode symmetric under
inversion and the mode antisymmetric under it. Each of them carries one
irreducible representation.

The dynamical matrix also commutes with the operations, but it is not generic.
Accidental degeneracies, near-degeneracies close to a band crossing, and the
small symmetry breaking from numerical noise in the force constants all make
coincidences in its eigenvalues. The random {math}`X` has none of these, so the
subspaces are taken from {math}`X`, and the dynamical matrix is diagonalized
inside them in step 4.

{math}`X` also commutes with the antiunitary operations,
{math}`TX^*T^\dagger=X`. When time reversal joins two unitary irreducible
representations into one pair, {math}`X` has equal eigenvalues on the two, and
both lie in one eigenspace of dimension {math}`2d_\mu`. The two bands that stick
together on a Brillouin-zone boundary plane of a nonsymmorphic space group are
found as one subspace in this way.

Schur's lemma covers the unitary operations. For the antiunitary ones, the
statement that each eigenspace carries one pair was checked for the structures
in `test/phonon/test_symmetry_adapted_modes.py` and is not proven here.

(symmodes_aligning_sets)=
### Aligning the sets of one type

At {math}`\Gamma` of NaCl, the acoustic set and the optical set both carry
{math}`T_{1u}`. After step 4, each set has the basis that `numpy.linalg.eigh`
returns for it. The two sets are therefore transformed by two different
{math}`3\times3` matrices, although the two representations are the same. Step
5 rotates the basis of the optical set so that both sets are transformed by the
same matrices.

For a set with eigenvectors {math}`E` ({math}`3N\times d_\mu`), the matrix of an
operation {math}`g` of the little group is

```{math}
\Gamma_E(g)=E^\dagger\,T(g)\,E^{(*)},
```

where {math}`E^{(*)}` is {math}`E^*` for an antiunitary operation and {math}`E`
otherwise. Let {math}`E_0` be the first set of a type and {math}`E_c` another
set of the same type. The {math}`d_\mu\times d_\mu` matrix

```{math}
J=\sum_{g}\Gamma_{E_c}(g)\,R\,\Gamma_{E_0}(g)^\dagger
```

is summed over all operations, the antiunitary ones included. {math}`R` is a
{math}`d_\mu\times d_\mu` matrix with a single element equal to one, and among
these {math}`d_\mu^2` choices the one that gives {math}`J` of the largest norm
is used.
By Schur's lemma, {math}`J` is a multiple of a unitary matrix. With the singular
value decomposition {math}`J=W\Sigma V^\dagger`, the set is rotated to
{math}`E_cWV^\dagger`, and after the rotation
{math}`\Gamma_{E_c}(g)=\Gamma_{E_0}(g)` for every operation.

The rotation mixes eigenvectors of one set only. All of them have the same
eigenvalue, so the frequencies, the degenerate sets and the characters do not
change. `get_representation_matrices` returns the same matrices for all sets of
one type.

The antiunitary operations are included in the sum because the unitary ones
alone leave a phase undetermined. With the unitary operations only,
{math}`J` is a multiple of a unitary matrix by a complex number, and after the
rotation the matrices of the antiunitary operations can still differ by a
phase between the two sets. That {math}`J` is a multiple of a unitary matrix
when the antiunitary operations are included was checked for the structures in
`test/phonon/test_symmetry_adapted_modes.py`.

The common matrices {math}`\Gamma_{E_0}(g)` are not a standard form of the
irreducible representation. They depend on {math}`X` and on the basis that
`numpy.linalg.eigh` returns for the first set, so they cannot be compared with
the matrices in a table or with the matrices at another q-point. Only the
statement that all sets of one type are transformed by the same matrices holds.

### Remarks

The size of a degenerate set is the dimension of a subspace, so no frequency
tolerance enters. Characters are computed for the unitary operations only. An
antiunitary operation multiplies a scalar by its complex conjugate, so the trace
of its matrix depends on the basis and is not a character.

The dynamical matrix must be C-type. A D-type matrix, as used internally by
`RandomDisplacements`, is in general not left unchanged by {math}`T` at a
q-point other than {math}`\Gamma`, and the module does not check this. Pass the matrix
that `DynamicalMatrix.run` produces.
