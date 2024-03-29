This is an example of PWmat interface.

To create supercells with displacements:

```bash
% phonopy --pwmat -c atom.config -d --dim 2 2 2 --pa F
```

A perfect 2x2x2 supercell (`supercell.config`) and two 2x2x2 supercells
(`supercell-xxx.config`) of the conventional unit cell written in NaCl.in are
created. In addition, `phonopy_disp.yaml` file is created. After force
calculations with the crystal structures in `supercell-xxx.config`, it is needed to
create `FORCE_SETS` file by

```bash
% phonopy -f OUT.FORCE-001
```

Here `OUT.FORCE-*` files are the output of the PWmat calculations and are supposed to
contain the forces on atoms calculated by PWmat. The `phonopy_disp.yaml` file has
to be put in the current directory. Now you can run phonon calculation, e.g.,

```bash
% phonopy -p band.conf
```
with `mesh` tag or `--mesh` option and `-t` option:

```bash
% phonopy -t --mesh 31 31 31
```
