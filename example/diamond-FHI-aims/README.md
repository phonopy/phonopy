This is a minimal example on how to run `phonopy` with `FHI-aims` as the force calculator:

1) Create the (one) displaced supercell structure(s):
   ```phonopy -d --dim="2 2 2" --aims```

2) Copy the generated supercells into folders containing one `geometry.in` (rename!) and one `control.in`. The `control.in` should contain `compute_forces .true.` or similar. Run the calculations in each folder, producing, e.g., one `aims.out` file in each. In this example we provide an example calculation for the generated displacement in `disp-001/aims.out`

3) Collect the forces:
   ```phonopy -f disp-???/aims.out```

4) Calculate phonon dispersion data into band.yaml and save band.pdf:
   ```phonopy -p -s band.conf```

Have fun and read the docs: [https://phonopy.github.io/phonopy/](https://phonopy.github.io/phonopy/)
