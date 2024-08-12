(examples_link)=

# Examples

Phonopy supports various external force calculators (mainly for abinitio codes).
The examples below are given for the default system that is equivalent to VASP
style. Most of usage is universal among the force calculators. So it is also
useful for non-VASP users to see the examples below. The list of the force
calculators and the links to their specific usages are shown at
{ref}`calculator_interfaces`.

Example files are found at
https://github.com/phonopy/phonopy/tree/master/example. The same are found in
the example directory of the phonopy package downloaded at
https://github.com/phonopy/phonopy/archive/master.zip. The followings show how
some of those examples work. Note that sometimes the followings are outdated
than the examples in the phonopy package. So the results or displace outputs can
be different.

## Si

### `FORCE_SETS` file creation for VASP

~~~bash
% phonopy -f vasprun.xml
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.26.6

Compiled with OpenMP support (max 10 threads).
Python version 3.12.4
Spglib version 2.4.0

Displacements were read from "phonopy_disp.yaml".
counter (file index): 1
"FORCE_SETS" has been created.
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
~~~
where `vasprun.xml` is the VASP output.

### DOS

~~~bash
% phonopy-load --mesh 31 31 31 -p
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.26.6

Compiled with OpenMP support (max 10 threads).
Running in phonopy.load mode.
Python version 3.12.4
Spglib version 2.4.0

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Mesh sampling mode
Settings:
  Sampling mesh: [31 31 31]
  Supercell: [2 2 2]
  Primitive matrix:
    [1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]
Spacegroup: Fd-3m (227)
Number of symmetry operations in supercell: 384
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were read from "FORCE_SETS".
Displacements were overwritten by "FORCE_SETS".
Max drift of force constants: -0.000001 (zz) -0.000001 (zz)
Max drift after symmetrization by translation: -0.000000 (xx) -0.000000 (xx)

Mesh numbers: [31 31 31]
Number of irreducible q-points on sampling mesh: 816/29791
Calculating phonons on sampling mesh...
Calculating DOS...

Summary of calculation was written in "phonopy.yaml".
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
~~~

```{image} Si-DOS.png
:width: 50%
```

### Thermal properties

~~~bash
% phonopy-load --mesh 31 31 31 -t -p
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                     2.26.6-rdev26+g1f6a3f81

Compiled with OpenMP support (max 10 threads).
Running in phonopy.load mode.
Python version 3.12.4
Spglib version 2.4.0

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Mesh sampling mode
Settings:
  Sampling mesh: [31 31 31]
  Supercell: [2 2 2]
  Primitive matrix:
    [1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]
Spacegroup: Fd-3m (227)
Number of symmetry operations in supercell: 384
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were read from "FORCE_SETS".
Displacements were overwritten by "FORCE_SETS".
Max drift of force constants: -0.000001 (zz) -0.000001 (zz)
Max drift after symmetrization by translation: -0.000000 (xx) -0.000000 (xx)

Mesh numbers: [31 31 31]
Number of irreducible q-points on sampling mesh: 816/29791
Calculating phonons on sampling mesh...
Calculating thermal properties...
Cutoff frequency: 0.00000
Number of phonon frequencies less than cutoff frequency: 1/178746
#      T [K]      F [kJ/mol]    S [J/K/mol]  C_v [J/K/mol]     E [kJ/mol]
       0.000      11.7110492      0.0000000      0.0000000     11.7110492
      10.000      11.7109211      0.0292204      0.0657586     11.7112133
      20.000      11.7100041      0.1915595      0.5807546     11.7138352
      30.000      11.7060581      0.6585603      1.9572223     11.7258149
      40.000      11.6956193      1.4847934      3.9396878     11.7550111
      50.000      11.6754205      2.5932055      6.0735527     11.8050808
      60.000      11.6431482      3.8848280      8.1404132     11.8762379
      70.000      11.5973537      5.2885744     10.1087510     11.9675539
      80.000      11.5371421      6.7633329     12.0156965     12.0782088
      90.000      11.4619236      8.2874371     13.8993871     12.2077929
     100.000      11.3712686      9.8490970     15.7769306     12.3561783
...
~~~

```{image} Si-props.png
:width: 50%
```

## NaCl

### Band structure

This requires to prepare `BORN` file.

~~~bash
% phonopy-load --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" -p
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.26.6

Compiled with OpenMP support (max 10 threads).
Running in phonopy.load mode.
Python version 3.12.4
Spglib version 2.4.0

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Band structure mode
Settings:
  Supercell: [2 2 2]
  Primitive matrix (Auto):
    [0.  0.5 0.5]
    [0.5 0.  0.5]
    [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Number of symmetry operations in supercell: 1536
Use -v option to watch primitive cell, unit cell, and supercell structures.

NAC params were read from "BORN".
Force sets were read from "FORCE_SETS".
Displacements were overwritten by "FORCE_SETS".
Max drift of force constants: 0.040159 (xx) 0.000009 (xx)
Max drift after symmetrization by translation: 0.000000 (zz) 0.000000 (zz)

Reciprocal space paths in reduced coordinates:
[ 0.000  0.000  0.000] --> [ 0.500  0.000  0.000]
[ 0.500  0.000  0.000] --> [ 0.500  0.500  0.000]
[ 0.500  0.500  0.000] --> [ 0.000  0.000  0.000]
[ 0.000  0.000  0.000] --> [ 0.500  0.500  0.500]

Summary of calculation was written in "phonopy.yaml".
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
~~~

```{image} NaCl-band-NAC.png
:width: 50%
```

(example_pdos)=

### PDOS

~~~bash
% phonopy-load --mesh 41 41 41 --pdos "1, 2" -p
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.26.6

Compiled with OpenMP support (max 10 threads).
Running in phonopy.load mode.
Python version 3.12.4
Spglib version 2.4.0

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Mesh sampling mode
Settings:
  Sampling mesh: [41 41 41]
  Supercell: [2 2 2]
  Primitive matrix (Auto):
    [0.  0.5 0.5]
    [0.5 0.  0.5]
    [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Number of symmetry operations in supercell: 1536
Use -v option to watch primitive cell, unit cell, and supercell structures.

NAC params were read from "BORN".
Force sets were read from "FORCE_SETS".
Displacements were overwritten by "FORCE_SETS".
Max drift of force constants: 0.040159 (xx) 0.000009 (xx)
Max drift after symmetrization by translation: 0.000000 (zz) 0.000000 (zz)

Mesh numbers: [41 41 41]
Number of q-points on sampling mesh: 68921
Calculating phonons on sampling mesh...
Calculating projected DOS...

Summary of calculation was written in "phonopy.yaml".
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
~~~

```{image} NaCl-PDOS-nac.png
:width: 50%
```

### Plot band structure and DOS at once

Band structure and DOS or PDOS can be plotted on one figure together by

~~~bash
% phonopy-load --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" --mesh 41 41 41 --pdos "1, 2" -p
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.26.6

Compiled with OpenMP support (max 10 threads).
Running in phonopy.load mode.
Python version 3.12.4
Spglib version 2.4.0

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Band structure and mesh sampling mode
Settings:
  Sampling mesh: [41 41 41]
  Supercell: [2 2 2]
  Primitive matrix (Auto):
    [0.  0.5 0.5]
    [0.5 0.  0.5]
    [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Number of symmetry operations in supercell: 1536
Use -v option to watch primitive cell, unit cell, and supercell structures.

NAC params were read from "BORN".
Force sets were read from "FORCE_SETS".
Displacements were overwritten by "FORCE_SETS".
Max drift of force constants: 0.040159 (xx) 0.000009 (xx)
Max drift after symmetrization by translation: 0.000000 (zz) 0.000000 (zz)

Reciprocal space paths in reduced coordinates:
[ 0.000  0.000  0.000] --> [ 0.500  0.000  0.000]
[ 0.500  0.000  0.000] --> [ 0.500  0.500  0.000]
[ 0.500  0.500  0.000] --> [ 0.000  0.000  0.000]
[ 0.000  0.000  0.000] --> [ 0.500  0.500  0.500]
Mesh numbers: [41 41 41]
Number of q-points on sampling mesh: 68921
Calculating phonons on sampling mesh...
Calculating projected DOS...

Summary of calculation was written in "phonopy.yaml".
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
~~~

```{image} NaCl-band-PDOS-NAC.png
:width: 50%
```

## MgB2 characters of ireducible representations

~~~bash
% phonopy-load --irreps 0 0 0
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                     2.26.6-rdev26+g1f6a3f81

Compiled with OpenMP support (max 10 threads).
Running in phonopy.load mode.
Python version 3.12.4
Spglib version 2.4.0

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Ir-representation mode
Settings:
  Supercell: [3 3 2]
Spacegroup: P6/mmm (191)
Number of symmetry operations in supercell: 432
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were read from "FORCE_SETS".
Displacements were overwritten by "FORCE_SETS".
Max drift of force constants: -0.039930 (zz) -0.000007 (zz)
Max drift after symmetrization by translation: 0.000000 (yy) 0.000000 (yy)


-------------------------------
  Irreducible representations
-------------------------------
q-point: [0. 0. 0.]
Point group: 6/mmm

Original rotation matrices:

     1         2         3         4         5         6
 --------  --------  --------  --------  --------  --------
  1  0  0  -1  0  0   1 -1  0  -1  1  0   0 -1  0   0  1  0
  0  1  0   0 -1  0   1  0  0  -1  0  0   1 -1  0  -1  1  0
  0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1

     7         8         9        10        11        12
 --------  --------  --------  --------  --------  --------
 -1  0  0   1  0  0  -1  1  0   1 -1  0   0  1  0   0 -1  0
  0 -1  0   0  1  0  -1  0  0   1  0  0  -1  1  0   1 -1  0
  0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1

    13        14        15        16        17        18
 --------  --------  --------  --------  --------  --------
  0 -1  0   0  1  0  -1  0  0   1  0  0  -1  1  0   1 -1  0
 -1  0  0   1  0  0  -1  1  0   1 -1  0   0  1  0   0 -1  0
  0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1

    19        20        21        22        23        24
 --------  --------  --------  --------  --------  --------
  0  1  0   0 -1  0   1  0  0  -1  0  0   1 -1  0  -1  1  0
  1  0  0  -1  0  0   1 -1  0  -1  1  0   0 -1  0   0  1  0
  0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1

Transformation matrix:

 1.000 -0.000  0.000
 0.000  1.000  0.000
 0.000  0.000  1.000

Rotation matrices by transformation matrix:

     E         i        C6        S3        C3        S6
 --------  --------  --------  --------  --------  --------
  1  0  0  -1  0  0   1 -1  0  -1  1  0   0 -1  0   0  1  0
  0  1  0   0 -1  0   1  0  0  -1  0  0   1 -1  0  -1  1  0
  0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1

    C2        sgh       C3        S6        C6        S3
 --------  --------  --------  --------  --------  --------
 -1  0  0   1  0  0  -1  1  0   1 -1  0   0  1  0   0 -1  0
  0 -1  0   0  1  0  -1  0  0   1  0  0  -1  1  0   1 -1  0
  0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1

   C2''       sgv       C2'       sgd      C2''       sgv
 --------  --------  --------  --------  --------  --------
  0 -1  0   0  1  0  -1  0  0   1  0  0  -1  1  0   1 -1  0
 -1  0  0   1  0  0  -1  1  0   1 -1  0   0  1  0   0 -1  0
  0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1

    C2'       sgd      C2''       sgv       C2'       sgd
 --------  --------  --------  --------  --------  --------
  0  1  0   0 -1  0   1  0  0  -1  0  0   1 -1  0  -1  1  0
  1  0  0  -1  0  0   1 -1  0  -1  1  0   0 -1  0   0  1  0
  0  0 -1   0  0  1   0  0 -1   0  0  1   0  0 -1   0  0  1

Character table:

  1 (  -0.000): Not found. Try adjusting tolerance value in IRREPS.
    ( 3,   0.0) ( 3, 180.0) ( 2,   0.0) ( 2, 180.0) ( 0,   0.0) ( 0,   0.0)
    ( 1, 180.0) ( 1,   0.0) ( 0,   0.0) ( 0,   0.0) ( 2,   0.0) ( 2, 180.0)
    ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0)
    ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0)

  4 (   9.953): E1u
    ( 2,   0.0) ( 2, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1,   0.0)
    ( 2, 180.0) ( 2,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0) ( 1, 180.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)

  6 (  11.975): A2u
    ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0)
    ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0)
    ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0)
    ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0)

  7 (  17.269): E2g
    ( 2,   0.0) ( 2,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1, 180.0) ( 1, 180.0)
    ( 2,   0.0) ( 2,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1, 180.0) ( 1, 180.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)

  9 (  20.565): B1g
    ( 1,   0.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0)
    ( 1, 180.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0)
    ( 1, 180.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0)
    ( 1,   0.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0)


Summary of calculation was written in "phonopy.yaml".
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
~~~

## Al-QHA

```
% phonopy-qha e-v.dat thermal_properties.yaml-{-{5..1},{0..5}} --sparse=50 -p
# Vinet EOS
#          T           E_0           B_0          B'_0           V_0
      0.000000    -14.814330     75.358945      4.746862     66.684166
      2.000000    -14.814330     75.358944      4.746862     66.684166
      4.000000    -14.814330     75.358934      4.746864     66.684167
      6.000000    -14.814330     75.358891      4.746869     66.684169
      8.000000    -14.814330     75.358779      4.746883     66.684174
     10.000000    -14.814331     75.358553      4.746911     66.684185
...
```

```{image} Al-QHA.png
:width: 50%
```

## Si-gruneisen

See {ref}`phonopy_gruneisen`.
