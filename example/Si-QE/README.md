# QE-PW interface with Si example

To create supercells with displacements:

```bash
% phonopy --qe -c Si.in -d --dim 2 2 2 --pa auto
```

A perfect 2x2x2 supercell (supercell.in) and one 2x2x2 supercells
(supercell-xxx.in) of the conventional unit cell written in Si.in are
created. In addition, disp.yaml file is created. After force
calculation with the crystal structure in supercell-001.in, it is
needed to create FORCE_SETS file by

```bash
% phonopy --qe -f supercell-001.out
```

Here .out file is the output of the PW calculation and is
supposed to contain the forces on atoms calculated by PW. The
disp.yaml file has to be put in the current directory. Now you can run
phonon calculation, e.g.,

```bash
% phonopy-load --band "1/2 1/2 1/2 0 0 0 1/2 0 1/2" -p
```
