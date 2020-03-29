# Calling hiphive through phonopys API

The reference structures and vasprun.xml files for this example can be found in `reference_data`.

First two training structures are generated with displacement amplitude of 0.03

    phonopy -d --dim 5 5 5 --rd 2 -c reference_data/POSCAR_prim --amplitude 0.03 --random-seed 42

Next the vasprun.xml files for these two structures are used to create the `FORCE_SETS`

    phonopy -f reference_data/vasprun.xml-{001..2}

Then, finally we can compute harmonic properties, e.g. DOS, via

    phonopy --hiphive -c phonopy_disp.yaml --mesh 40 -p

And if you want to e.g. change the cutoff used in hiphive, it can be done via

    phonopy --hiphive -c phonopy_disp.yaml --mesh 40 -p --fc-calc-opt cutoff=5.0
