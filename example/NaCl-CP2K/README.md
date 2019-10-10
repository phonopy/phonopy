# Example of the CP2K Phonopy interface for NaCl

To create supercells with displacements:

```console
$ phonopy --cp2k -c NaCl.inp -d --dim="2 2 2"
```

A perfect 2x2x2 supercell (`NaCl-supercell.inp`) and two 2x2x2 supercells
(`NaCl-supercell-001.inp`, resp. `NaCl-supercell-002.inp`) of the conventional
unit cell written in `NaCl.inp` are created.
In addition, `phonopy_disp.yaml` file is created.

The `amplitude` option is optionally specified for atomic displacement distance
in Angstrom. After force the calculations with the crystal structures in
NaCl-supercell-xxx.inp, it is needed to create `FORCE_SETS` file by running

```console
$ phonopy --cp2k -f NaCl-supercell-001-forces-1_0.xyz NaCl-supercell-002-forces-1_0.xyz
```

Here the XYZ files are supposed to contain the forces on atoms calculated
by CP2K. The `phonopy_disp.yaml` file has to be put in the current directory.
Now you can run phonon calculation, e.g.,

```console
$ phonopy --cp2k -c NaCl.inp -p band.conf
```
