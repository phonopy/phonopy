# SiO2 high pressure phase of Stishovite

The unit cell was relaxed using VASP with `PSTRESS = 400` in `INCAR`. In `BORN`
file, Born effective charges of atom 1 and 3 are written. Dielectric and Born
effective charge tensors were calculated using VASP with `LEPSILON = .TRUE.` in
`INCAR`. Band structure with LO-TO splitting is plotted by

```bash
% phonopy-load --band "0.5 0.5 0.5  0.0 0.0 0.0  0.5 0.5 0.0  0.0 0.5 0.0" -p
```

Small imaginary acoustic mode may appear due to the shortage of supercell size.

An animation file for the v_sim viewer is created by

```bash
% phonopy-load --anime 0 0 0
```

`INCAR` file in this directory gives an example of setting for calculating Born
effective charge and dielectric constant.
