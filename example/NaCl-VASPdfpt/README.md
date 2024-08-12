# NaCl phonon calculation with DFPT functional of VASP code

`POSCAR` in this directory is created as `SPOSCAR` using the unit cell
`POSCAR-unitcell` by

```bash
% phonopy -d --dim 2 2 2 -c POSCAR-unitcell
% phonopy -d --dim 1 1 1 -c SPOSCAR --pa auto
% mv SPOSCAR POSCAR
```

The second phonopy command is for making `phonopy_disp.yaml` corresponding to
`SPOSCAR`

To obtain `FORCE_CONSTANTS`

```bash
% phonopy --fc vasprun.xml
```

Phonon analysis is done such as

```bash
% phonopy-load --readfc --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" -p
```
