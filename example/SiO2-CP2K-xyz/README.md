This is example of PHONOPY calculations using CP2K including BORN charges calculations.
The example is tested with version 8 and 2024.1.
First one have to construct structures with displacements stored in xyz file format.
Use --born option if BORN charges calculations required.

```bash
% python gen_supercell_xyz.py -i Punitcell.inp --dim="2 2 2" --born
```
Lattice parameters are printed in the output and in each supercell.xyz structure. Put
lattice parameters in force.inp pattern file. Edit the file according to your
calculations (Potentials, cutoff energy and so on). Edit path where potential files
are stored (@SET BASISDIR /home/USER/cp2k/Potentials in force.inp).
Check parameters in polar.inp correspond to parameters in force.inp.
Linear response calculations (polar.inp) require tight convergence
```EPS_SCF 1.0E-8``` recomended.

Generate directories using ```make_displ_dirs.sh``` bash script,  then run CP2K
calculations.
To speed up the process make calculations for ideals structure (DISP-0000 dir) first,
then copy WFN file in all DISP directories to use as initial guess.

Collect forces after CP2K calculations is done:

```bash
%phonopy --cp2k -f DISP-{0001..0012}/SiO2-forces-1_0.xyz
```

Generate BORN file to take non-analitical term into account:

```bash
python get_born_cp2k.py -i DISP-0000/polar.out -m force.out
```
Plot bandstrucutre

```bash
phonopy --cp2k -c Punitcell.inp --dim="2 2 2" --band=auto -s -p --factor=3739.4256801 --nac
```
