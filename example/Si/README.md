# Si example

To create supercell(s) with displacement(s):

```bash
% phonopy -d --dim 2 2 2 --pa auto -c POSCAR-unitcell
```

For this example, only one displaced supercell is made. Running VASP with this
displaced supercell, `vasprun.xml` is obtained. To create `FORCE_SETS` that is
the phonopy's default dataset to calculate force constants by

```bash
% phonopy -f vasprun.xml
```

To draw DOS,

```bash
% phonopy-load --mesh 31 31 31 -p
```

To calculate thermal properties

```bash
% phonopy-load --mesh 31 31 31 -p -t
```
