# NaCl example

Details are presented on the phonopy document at
http://phonopy.github.io/phonopy/examples.html. Shortly the usage is shown here,
too.

The supercells with displacements were created by

```bash
% phonopy -d --dim 2 2 2 --pa auto -c POSCAR-unitcell
```

`FORSE_SETS` is obtained by

```bash
% phonopy -f vasprun.xml-001 vasprun.xml-002
```

Band structure is plotted by

```bash
% phonopy-load -p --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5"
```

PDOS is plotted by:

```bash
% phonopy-load -p --mesh 15 15 15 --pdos "1, 2"
```

Both are plotted together by:

```bash
% phonopy-load -p --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" --mesh 15 15 15 --pdos "1, 2"
```

NaCl.py, NaCl-yaml.py, and NaCl-read_write_fc.py are phonopy API examples, which are executed by, e.g.,

```bash
% python NaCl.py
```
