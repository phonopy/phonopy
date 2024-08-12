# LiF with `--nosym` option

This is an example of using `--nosym` option in LiF.

If non-analytical term correction is used, `BORN` file has to be created without
considering symmetry. In the case of this example, 8 Born effective charge
tensors (lines) have to be written in BORN file.

The displacements are created by

```bash
% phonopy -d --dim="2 2 2" --nosym -c POSCAR-unitcell
```

48 `POSCAR`s are created. `FORCE_SETS` is created from the forces on atoms of
these structures. Then DOS is calculated by

```bash
% phonopy-load --mesh 50 --nosym -p
```
