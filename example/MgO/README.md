# MgO example

The rocksalt-type MgO phonon calculation. Forces were calculated using VASP with
500 eV cutoff energy and 4x4x4 k-point mesh for the conventional unit cell
(`POSCAR-unitcell`).

Phonon band structure is drawn using Seek-path (`pip install seekpath`) by

```bash
% phonopy-load --band auto -p
```
