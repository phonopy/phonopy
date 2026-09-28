---
orphan: true
---

(spacegroup_reps)=
# Space-group representations of phonon modes

This page gives the equations that `phonopy/phonon/spgreps.py` implements. The
module takes the dynamical matrix at one q-point and returns the phonon
frequencies, the eigenvectors and the degenerate sets. The degenerate sets are
decided by the symmetry of the q-point, including time reversal. Closeness of
frequencies is not used.

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
`spgreps.py` expects a C-type dynamical matrix and returns C-type eigenvectors.

The C-type dynamical matrix is not periodic in {math}`\mathbf{q}`. Let
{math}`V(\mathbf{q})` be the diagonal matrix with
{math}`e^{i\mathbf{q}\cdot\mathbf{r}_{j0}}` repeated three times for each atom
{math}`j`. Writing the same displacement pattern with the label
{math}`\mathbf{q}+\mathbf{G}` or with the label {math}`\mathbf{q}` gives

```{math}
:label: spgreps_ec_shift
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
:label: spgreps_dc_tr
D^{\mathrm{C}}(\mathbf{q})^*=D^{\mathrm{C}}(-\mathbf{q}),
```

and the complex conjugate of an eigenvector at {math}`\mathbf{q}` is an
eigenvector at {math}`-\mathbf{q}`. This is time-reversal symmetry.

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
:label: spgreps_little_group
\mathrm{R}(\eta\mathbf{q})=\mathbf{q}+\mathbf{G}_{\mathrm{S}},
\qquad\text{or}\qquad
\eta\,\tilde q\tilde R^{-1}-\tilde q\in\mathbb{Z}^3.
```

{math}`G_{\mathbf{q}}` denotes the unitary operations and
{math}`A_{\mathbf{q}}` the antiunitary ones.
`get_little_group_operations` returns {math}`G_{\mathbf{q}}` first and then
{math}`A_{\mathbf{q}}`.

## Representation matrix

A space-group operation sends a C-type eigenvector at {math}`\mathbf{q}` to an
eigenvector at {math}`\mathrm{R}\mathbf{q}` by the {math}`3N\times3N` matrix

```{math}
:label: spgreps_gamma
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
{math}`\mathbf{q}+\mathbf{G}_{\mathrm{S}}`, and Eq. {eq}`spgreps_ec_shift`
brings the label back to {math}`\mathbf{q}`. The representation matrix used in
the module is

```{math}
:label: spgreps_t
T(\mathrm{S})=V(\mathbf{G}_{\mathrm{S}})\,\Gamma^{\mathrm{C},\eta\mathbf{q}}(\mathrm{S}).
```

It acts on eigenvectors and on matrices as follows.

| Operation  | Eigenvector                                            | Matrix                          |
| ---------- | ------------------------------------------------------ | ------------------------------- |
| Unitary    | {math}`\mathbf{e}\mapsto T\mathbf{e}`                  | {math}`M\mapsto TMT^\dagger`    |
| Antiunitary | {math}`\mathbf{e}\mapsto T\mathbf{e}^*`               | {math}`M\mapsto TM^*T^\dagger`  |

For an antiunitary operation, the complex conjugate moves the eigenvector from
{math}`\mathbf{q}` to {math}`-\mathbf{q}` by Eq. {eq}`spgreps_dc_tr`,
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
:label: spgreps_phase
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
| `phases[j']`         | Eq. {eq}`spgreps_phase`                                         |
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

The metric {math}`g=L^{\mathsf{T}}L` is averaged, and the basis vectors are
replaced by the nearest ones with the averaged metric:

```{math}
g_{\mathrm{sym}}=\frac{1}{n_{\mathrm{op}}}\sum_s\tilde R_s^{\mathsf{T}}\,g\,\tilde R_s,
\qquad
L_{\mathrm{sym}}=L\,g^{-1/2}\,g_{\mathrm{sym}}^{1/2}.
```

With {math}`L_{\mathrm{sym}}`, the Cartesian rotation
{math}`\mathrm{R}=L_{\mathrm{sym}}\tilde RL_{\mathrm{sym}}^{-1}` is orthogonal to
round-off.

Each position is replaced by the average of its images:

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

{math}`\langle M\rangle` commutes with every operation of the little group. The
steps are listed below.

1. A random Hermitian matrix {math}`Y` is drawn with a fixed seed, and
   {math}`X=\langle Y\rangle` is formed. The dynamical matrix is averaged in the
   same way, {math}`D_{\mathrm{sym}}=\langle D^{\mathrm{C}}(\mathbf{q})\rangle`.
2. {math}`X` is diagonalized. Eigenvalues that differ by less than a relative
   tolerance are grouped, and each group of eigenvectors spans one subspace
   {math}`U_k` ({math}`3N\times d_k`). {math}`X` commutes with the operations,
   so each subspace is invariant under them. For a random {math}`Y`, each
   subspace is expected to carry one irreducible representation of the little
   group, or one pair of representations joined by time reversal. This was
   checked for the structures in `test/phonon/test_spgreps.py`; it is not proven
   in general.
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
5. The sets of all types are sorted by eigenvalue, and the frequencies are
   {math}`\operatorname{sgn}(\omega^2)\sqrt{|\omega^2|}` times the unit
   conversion factor.

The size of a degenerate set is the dimension of a subspace, so no frequency
tolerance enters. Characters are computed for the unitary operations only. An
antiunitary operation multiplies a scalar by its complex conjugate, so the trace
of its matrix depends on the basis and is not a character.

The dynamical matrix must be C-type. A D-type matrix, as used internally by
`RandomDisplacements`, is in general not left unchanged by {math}`T` at a
q-point other than {math}`\Gamma`, and the module does not check this. Pass the matrix
that `DynamicalMatrix.run` produces.
