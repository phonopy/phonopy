# Cr with collinear magnetic moments

This is an example to calculate phonon with magnetic symmetry by collinear
magnetic moments. Supercells with displacements can be created by running
following command:

```bash
% phonopy -d --dim 2 2 2 --pa "-1/2 1/2 1/2 1/2 -1/2 1/2 1/2 1/2 -1/2" --magmom "1 -1" -c POSCAR-unitcell
```

`--magmom` option has to be specified to correctly recognize the magnetic
symmetry. `--pa` option is added because this system has the type-4 magnetic
space group and this choice of the unit cell. Currently, `--pa auto` cannot be
used for magnetic space groups.

A file named `MAGMOM` is created together with supercell `POSCAR`s. In the
`MAGMOM` file, a line of `MAGMOM` is written, which can be used in VASP `INCAR`.
Running force calculation (350 eV, 0.1 eV smearing, 20x20x20 k-points for unit
cell), we obtain vasprun.xml. With this, FORCE_SETS is created by

```bash
% phonopy -f vasprun.xml.xz
```

Then phonon calculation is achieved by, e.g.,

```bash
% phonopy-load --band "1/2 -1/2 1/2 0 0 0 1/4 1/4 1/4" -p
```
