# Example for the CP2K Phonopy interface using bulk silicon

To create supercells with displacements:

```console
$ phonopy --cp2k -c Si.inp -d --dim="2 2 2"
```

A perfect 2x2x2 supercell (`Si-supercell-000.inp`) and one 2x2x2 supercells
(`supercell-001.inp`) of the conventional unit cell written in `Si.inp` are
created. In addition, a `phonopy_disp.yaml` file is created. After the force
calculation with the crystal structure in `supercell-001.inp`, it is
needed to create `FORCE_SETS` file by running:

```console
$ phonopy --cp2k -f Si-supercell-001-forces-1_0.xyz
```

Here the `.xyz` files are supposed to contain the forces on atoms calculated
by CP2K.

To plot the phonon band structure:

```console
$ phonopy --cp2k -c Si.inp -p --dim="2 2 2" --pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0" --band="1/2 1/2 1/2 0 0 0 1/2 0 1/2"
```
