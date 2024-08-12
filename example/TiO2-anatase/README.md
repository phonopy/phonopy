# Phonon calculation for TiO2-anatase

Supercell dimension is 4x4x2 and forces were calculated using VASP with 500 eV
cutoff energy and 4x4x1 k-point mesh for the unit cell.

Band structure is drawn with Seek-Path (pip install seekpath is needed) by

```bash
% phonopy-load --band auto -p
```
