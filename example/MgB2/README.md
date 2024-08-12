# MgB2 example

Supercell dimension is 3x3x2. Forces were calculated using VASP with 500 eV
cutoff energy, 0.2 eV smearing, and k-point mesh of 18x18x18 for the unit cell.

```bash
% phonopy -f vaspruns/vasprun-{001..002}.xml
% phonopy-load --band "0 0 0  1/3 1/3 0  1/2 0 0  0 0 0  0 0 1/2  1/3 1/3 1/2  1/2 0 1/2  0 0 1/2" -p
```

The band path refers to Bohnen et al., PRL 86, 5771 (2001).
