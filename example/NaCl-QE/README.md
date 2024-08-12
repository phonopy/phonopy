This is an example of QE-PW interface.

To create supercells with displacements:

```bash
% phonopy --qe -c NaCl.in -d --dim 2 2 2 --pa auto
```

A perfect 2x2x2 supercell (`supercell.in`) and two 2x2x2 supercells
(`supercell-xxx.in`) of the conventional unit cell written in NaCl.in are
created. In addition, `phonopy_disp.yaml` file is created. After force
calculations with the crystal structures in `supercell-xxx.in`, it is needed to
create `FORCE_SETS` file by

```bash
% phonopy -f NaCl-001.out NaCl-002.out
```

Here `*.out` files are the output of the PW calculations and are supposed to
contain the forces on atoms calculated by PW. The `phonopy_disp.yaml` file has
to be put in the current directory. Now you can run phonon calculation, e.g.,

```bash
% phonopy-load --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" -p
```

`BORN` file is created running DFPT calculation using `ph.x` (phonon) code in
the quantum espresso package. The details are found in the phonopy
documentation. The input and output files of the `ph.x` DFPT calculation are
`NaCl.ph.in` and `NaCl.ph.out`.

Thermal properties at constant volume are calculated by setting regular grid
with `--mesh` and `-t` option:

```bash
% phonopy-load -t --mesh 31 31 31
```

The thermal properties are written in `thermal_properties.yaml`.
