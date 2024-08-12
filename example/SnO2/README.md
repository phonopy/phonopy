# SnO2 rutile structure

Forces were calculated using VASP with 500 eV cutoff energy and 4x4x6 k-point
mesh for the unit cell.

The space group type is P4_2/mnm (136). This is a non-symmorphic space group and
a screw axis of c exists. Usually slope of phonon band normal to the Brillouin
zone boundary becomes zero, but there are some positive and negative slopes are
found at the Brillouin zone boundary due to the screw symmetry operation. This
can be watched by

```bash
% phonopy-load --band "0 0 0  0 0 1" -p
```

where Z=(0, 0, 1/2). Non-zero slope indicates non-zero group velocity. But
phonopy is usually not able to calculate these group velocity correctly. An
example of wrongly calculated group velocity is shown as follows. We obtain the
group velocity by

```bash
% phonopy-load --band="0 0 0  0 0 1" --gv
```

For the group velocity calculation, depending on cases, the degenerated bands
are decomposed in an unexpected way and the group velocity can be calculated
incorrectly. So the check by eyes is always recommended.
