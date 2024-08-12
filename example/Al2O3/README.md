# Corundum Al2O3

Forces were calculated in the 2x2x1 supercell of the hexagonal lattice by VASP
with 500eV cutoff energy and 6x6x1 k-point mesh. Dielectric constant and Born
effective charge tensors were calculated with the hexagonal unit cell.
The Wyckoff positions are c and e, respectively, which can be found by

```bash
% phonopy -c POSCAR-unitcell --symmetry --pa=auto
```

`phonopy_disp.yaml` is created by

```bash
% phonopy -d --dim 2 2 1 --pa auto -c POSCAR-unitcell
```

The band structure is plotted using seekpath by

```bash
% phonopy-load --band auto -p
```

where seekpath can be installed by `pip install seekpath`.
