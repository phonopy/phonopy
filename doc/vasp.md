(vasp_interface)=
# VASP & phonopy calculation

## Pre-process

The input structure of `POSCAR` ({ref}`this <example_POSCAR1>`) is used as an
example here. Most files are found at [SiO2-HP
example](https://github.com/phonopy/phonopy/tree/master/example/SiO2-HP/).

In the pre-process, supercell structures with (or without)
displacements are created from a unit cell fully considering crystal
symmetry.

To obtain supercells ({math}`2\times 2\times 3`) with displacements,
run phonopy,

```bash
% phonopy -d --dim 2 2 3 --pa auto
```

You should find the files, `SPOSCAR`, `phonopy_disp.yaml`, and
`POSCAR-{number}` as follows:

```bash
% ls
phonopy_disp.yaml  POSCAR  POSCAR-001  POSCAR-002  POSCAR-003  SPOSCAR
```

`SPOSCAR` is the perfect supercell structure, `phonopy_disp.yaml`
contains the information on displacements, and `POSCAR-{number}` are
the supercells with atomic displacements. `POSCAR-{number}`
corresponds to the different atomic displacements written in
`phonopy_disp.yaml`.

## Calculation of sets of forces

Force constants are calculated using the structure files
`POSCAR-{number}` (from forces on atoms) or using the `SPOSCAR`
file. In the case of VASP, the calculations for the finite
displacement method can be proceeded just using the
`POSCAR-{number}` files as `POSCAR` of VASP calculations. An
example of the `INCAR` is as follows:

```
   PREC = Accurate
 IBRION = -1
  ENCUT = 500
  EDIFF = 1.0e-08
 ISMEAR = 0; SIGMA = 0.01
  IALGO = 38
  LREAL = .FALSE.
  LWAVE = .FALSE.
 LCHARG = .FALSE.
```

Be careful not to relax the structures. Then create `FORCE_SETS`
file using {ref}`vasp_force_sets_option`:

```bash
% phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml disp-003/vasprun.xml
```

or

```bash
% phonopy -f disp-{001..003}/vasprun.xml
```

If you want to calculate force constants by VASP-DFPT directory, see
{ref}`vasp_dfpt_interface`.

## Post-process

In the post-process,

1. Force constants are calculated from the sets of forces
2. A part of dynamical matrix is built from the force constants
3. Phonon frequencies and eigenvectors are calculated from the
   dynamical matrices with the specified *q*-points.

The density of states (DOS) is plotted by

```bash
% phonopy-load --mesh 20 20 20 -p
```

Thermal properties are calculated with the sampling mesh by
```bash
% phonopy-load --mesh 20 20 20 -t
```

You should check the convergence with respect to the mesh numbers.
Thermal properties can be plotted by

```bash
% phonopy-load --mesh 20 20 20 -t -p
```

Projected DOS is calculated and plotted by

```bash
% phonopy-load --mesh 20 20 20 --pdos "1 2, 3 4 5 6" -p
```

Band structure is plotted by

```bash
% phonopy-load --band "0.5 0.5 0.5  0.0 0.0 0.0  0.5 0.5 0.0  0.0 0.5 0.0" -p
```

In either case, by setting the `-s` option, the plot is going to be
saved in the PDF format. If you don't need to plot DOS, the (partial)
DOS is just calculated using the `--dos` option.

## Non-analytical term correction (Optional)

To activate non-analytical term correction, {ref}`born_file` is
required. This file contains the information of Born effective charge
and dielectric constant. These physical values are also obtained from
the first-principles calculations, e.g., by using VASP, pwscf, etc. In
the case of VASP, an example of `INCAR` will be as shown below

```
    PREC = Accurate
  IBRION = -1
  NELMIN = 5
   ENCUT = 500
   EDIFF = 1.000000e-08
  ISMEAR = 0
   SIGMA = 1.000000e-02
   IALGO = 38
   LREAL = .FALSE.
   LWAVE = .FALSE.
  LCHARG = .FALSE.
LEPSILON = .TRUE.
```

In addition, it is recommended to increase the number of k-points to
be sampled. Twice the number for each axis may be a choice. After
running this VASP calculation, `BORN` file has to be created
following the `BORN` format ({ref}`born_file`). However for VASP, an
auxiliary tool is prepared, which is `phonopy-vasp-born`. There is
an option `--pa` for this command to set a transformation matrix
from supercell or unit cell with centring to the primitive cell. Since
this rutile-type SiO2 has the primitive lattice, it is unnecessary to
set this option. Running `phonopy-vasp-born` in the directory
containing `vasprun.xml` (or `OUTCAR`) of this VASP calculation:

```bash
% phonopy-vasp-born
# epsilon and Z* of atoms 1 3
   3.2605670   0.0000000   0.0000000   0.0000000   3.2605670   0.0000000   0.0000000   0.0000000   3.4421330
   3.7558600   0.3020100   0.0000000   0.3020100   3.7558600   0.0000000   0.0000000   0.0000000   3.9965200
  -1.8783900  -0.5270900   0.0000000  -0.5270900  -1.8783900   0.0000000   0.0000100   0.0000100  -1.9987900
```

To employ symmetry constraints, `--st` option may used as follows:

```bash
% phonopy-vasp-born --st
# epsilon and Z* of atoms 1 3
   3.2605670   0.0000000   0.0000000   0.0000000   3.2605670   0.0000000   0.0000000   0.0000000   3.4421330
   3.7561900   0.3020100   0.0000000   0.3020100   3.7561900   0.0000000   0.0000000   0.0000000   3.9968733
  -1.8780950  -0.5270900   0.0000000  -0.5270900  -1.8780950   0.0000000   0.0000000   0.0000000  -1.9984367
```

The values are slightly modified by symmetry, but we can see the
original values obtained directly from VASP was already very good.

To put `BORN` file in the current directly, non-analytical term correction is
activated.
