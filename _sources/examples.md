(examples_link)=

# Examples

```{contents}
:depth: 2
:local:
```

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

```
% phonopy -f vasprun.xml
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                       2.7.0

Python version 3.7.6
Spglib version 1.14.1

Displacements were read from "phonopy_disp.yaml".
counter (file index): 1
FORCE_SETS has been created.
                 _
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
```

where `vasprun.xml` is the VASP output.

### DOS

```
% phonopy -p mesh.conf
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                       2.7.0

Python version 3.7.6
Spglib version 1.14.1

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
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: -0.000001 (yy) -0.000001 (yy)

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
```

```{image} Si-DOS.png
:width: 50%
```

### Thermal properties

```
% phonopy -t -p mesh.conf
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Phonopy configuration was read from "mesh.conf".
Crystal structure was read from "phonopy_params.yaml".
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
Use -v option to watch primitive cell, unit cell, and supercell structures.

Forces and displacements were read from "phonopy_params.yaml".
Computing force constants...
Max drift of force constants: -0.000001 (zz) -0.000001 (zz)

Mesh numbers: [31 31 31]
Number of irreducible q-points on sampling mesh: 816/29791
Calculating phonons on sampling mesh...
Calculating thermal properties...
Cutoff frequency: 0.00000
Number of phonon frequencies less than cutoff frequency: 3/178746
# T [K] F [kJ/mol] S [J/K/mol] C_v [J/K/mol] E [kJ/mol]

       0.000      11.7110491      0.0000000      0.0000000     11.7110491
      10.000      11.7110004      0.0207133      0.0652014     11.7112076
      20.000      11.7101706      0.1826665      0.5801980     11.7138239
      30.000      11.7063148      0.6494417      1.9566658     11.7257980
      40.000      11.6959680      1.4755146      3.9391312     11.7549886
      50.000      11.6758626      2.5838026      6.0729959     11.8050528
      60.000      11.6436849      3.8753235      8.1398561     11.8762043
      70.000      11.5979858      5.2789840     10.1081937     11.9675147
      80.000      11.5378706      6.7536681     12.0151391     12.0781640
      90.000      11.4627490      8.2777067     13.8988296     12.2077426
     100.000      11.3721917      9.8393078     15.7763730     12.3561224
...
```

```{image} Si-props.png
:width: 50%
```

## NaCl

### Band structure

```
% phonopy -p band.conf
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Phonopy configuration was read from "band.conf".
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
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: 0.040159 (yy) 0.000009 (xx)

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
```

```{image} NaCl-band.png
:width: 50%
```

### Band structure with non-analytical term correction

This requires to prepare BORN file.

```
% phonopy -p --nac band.conf
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Phonopy configuration was read from "band.conf".
Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Band structure mode
Settings:
  Non-analytical term correction (NAC): on
  Supercell: [2 2 2]
  Primitive matrix (Auto):
    [0.  0.5 0.5]
    [0.5 0.  0.5]
    [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: 0.040159 (yy) 0.000009 (xx)

NAC parameters were not found in "phonopy_disp.yaml".
NAC parameters were read from "BORN".
Use NAC by Gonze et al. (no real space sum in current implementation)
  PRB 50, 13035(R) (1994), PRB 55, 10355 (1997)
  G-cutoff distance: 1.16, Number of G-points: 307, Lambda: 0.19

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
```

```{image} NaCl-band-NAC.png
:width: 50%
```

(example_pdos)=

### PDOS

```
% phonopy -p pdos.conf
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Phonopy configuration was read from "pdos.conf".
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
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: 0.040159 (yy) 0.000009 (xx)

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
```

```{image} NaCl-PDOS.png
:width: 50%
```

With non-analytical term correction, the PDOS may not change very much because
it mainly affects phonon modes in the reciprocal region close to {math}`\Gamma`
point.

```
% phonopy --nac -p pdos.conf
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Phonopy configuration was read from "pdos.conf".
Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Mesh sampling mode
Settings:
  Non-analytical term correction (NAC): on
  Sampling mesh: [41 41 41]
  Supercell: [2 2 2]
  Primitive matrix (Auto):
    [0.  0.5 0.5]
    [0.5 0.  0.5]
    [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: 0.040159 (yy) 0.000009 (xx)

NAC parameters were not found in "phonopy_disp.yaml".
NAC parameters were read from "BORN".
Use NAC by Gonze et al. (no real space sum in current implementation)
  PRB 50, 13035(R) (1994), PRB 55, 10355 (1997)
  G-cutoff distance: 1.16, Number of G-points: 307, Lambda: 0.19

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
```

```{image} NaCl-PDOS-nac.png
:width: 50%
```

### Plot band structure and DOS at once

Band structure and DOS or PDOS can be plotted on one figure together by

```
% phonopy band-pdos.conf --nac -p
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Phonopy configuration was read from "band-pdos.conf".
Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Band structure and mesh sampling mode
Settings:
  Non-analytical term correction (NAC): on
  Sampling mesh: [41 41 41]
  Supercell: [2 2 2]
  Primitive matrix (Auto):
    [0.  0.5 0.5]
    [0.5 0.  0.5]
    [0.5 0.5 0. ]
Spacegroup: Fm-3m (225)
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: 0.040159 (yy) 0.000009 (xx)

NAC parameters were not found in "phonopy_disp.yaml".
NAC parameters were read from "BORN".
Use NAC by Gonze et al. (no real space sum in current implementation)
  PRB 50, 13035(R) (1994), PRB 55, 10355 (1997)
  G-cutoff distance: 1.16, Number of G-points: 307, Lambda: 0.19

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
```

```{image} NaCl-band-PDOS-NAC.png
:width: 50%
```

## MgB2 characters of ireducible representations

```
% phonopy -f vasprun.xml-{001,002}
% phonopy --dim="3 3 2" --irreps="0 0 0"
        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/
                                      2.12.0

Python version 3.9.6
Spglib version 1.16.2

Crystal structure was read from "phonopy_disp.yaml".
Unit of length: angstrom
Ir-representation mode
Settings:
  Supercell: [3 3 2]
  Primitive matrix (Auto):
    [1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]
Spacegroup: P6/mmm (191)
Use -v option to watch primitive cell, unit cell, and supercell structures.

Force sets were not found in "phonopy_disp.yaml".
Forces and displacements were read from "FORCE_SETS".
Computing force constants...
Max drift of force constants: -0.039930 (zz) -0.000007 (zz)


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

 1.000  0.000  0.000
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

  1 (  -0.019): A2u
    ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0)
    ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0)
    ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0)
    ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1,   0.0)

  2 (   0.004): E1u
    ( 2,   0.0) ( 2, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1,   0.0)
    ( 2, 180.0) ( 2,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0) ( 1, 180.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)

  4 (   9.953): E1u
    ( 2,   0.0) ( 2, 180.0) ( 1,   0.0) ( 1, 180.0) ( 1, 180.0) ( 1,   0.0)
    ( 2, 180.0) ( 2,   0.0) ( 1, 180.0) ( 1,   0.0) ( 1,   0.0) ( 1, 180.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)
    ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0) ( 0,   0.0)

  6 (  11.982): A2u
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
```

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
