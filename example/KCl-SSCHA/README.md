# Example of KCl SSCHA calculation

## How to run

The displacements for the training data used in polynomial machine learning
potentials (MLPs) are generated with a command like the following:

```
% phonopy --pa auto --rd 1000 -c POSCAR-unitcell --dim 2 2 2 --amin 0.03 --amax 1.5
```

After calculating forces using the VASP code, the `phonopy_params.yaml` file,
which contains displacements, forces, and supercell energies, is created with
the following command:

```
% phonopy --sp -f vasprun-{001..120}.xml
```

In this example, the file `phonopy_mlpsscha_params_KCl-120.yaml.xz` in the
example directory serves as precomputed data for KCl. To develop the MLPs using
this file, run:

```
% phonopy-load phonopy_mlpsscha_params_KCl-120.yaml.xz --pypolymlp --mlp-params="ntrain=100, ntest=20"
```

This command generates the `phonopy.pmlp` file, which contains the developed
MLPs. This file under current directory is read when running phonopy with the
`--pypolymlp` option. To perform the SSCHA calculation, execute:

```
% phonopy-load phonopy_mlpsscha_params_KCl-120.yaml.xz --pypolymlp --sscha 10 --rd-temperature 300
```

The calculated SSCHA force constants are stored in
`phonopy_sscha_fc_10.yaml.xz`, where `10` indicates the final iteration number.
These SSCHA force constants can be compared with `phonopy_fc_JPCM2022.yaml.xz`,
which is explained in the next section. The phonon band structures corresponding
to these force constants can be plotted and compared using the following
command:

```
% phonopy-load phonopy_sscha_fc_JPCM2022.yaml.xz --band auto
% mv band.yaml band-JPCM2022.yaml
% phonopy-load phonopy_sscha_fc_10.yaml.xz --band auto
% phonopy-bandplot band.yaml band-JPCM2022.yaml
```

## Comparison with the reported result

The file `phonopy_fc_JPCM2022.yaml.xz` contains the full *ab initio* SSCHA
force constants for a 2×2×2 supercell of the conventional unit cell, as
calculated in the study by A. Togo *et al.*, J. Phys.: Condens. Matter **34**,
365401 (2022). The SSCHA force constants from the final iteration are expected
to closely resemble these force constants if the MLPs are good.

The files `phonopy_fc222_JPCM2022.yaml.xz` and `phonopy_fc444_JPCM2022.yaml.xz`
provide the harmonic force constants for 2×2×2 and 4×4×4 supercells,
respectively, as calculated in the same study. To compute a
temperature-dependent phonon band structure for a 4×4×4 supercell, an
approximation was employed where the SSCHA force constants for the 2×2×2
supercell were embedded into the harmonic force constants of the 4×4×4
supercell.

This embedding process involved first interpolating the SSCHA and harmonic force
constants of the 2×2×2 supercell to match those of the 4×4×4 supercell. The
difference between the interpolated SSCHA and harmonic force constants was then
added to the harmonic force constants for the 4x4x4 supercell. The detailed
procedure is described in A. Togo *et al.*, *J. Phys.: Condens. Matter* **35**,
353001 (2023). The Python script `embed_sscha_fc.py` automates this embedding
process.

Finally, for KCl, the difference in the phonon band structure between the 2×2×2
and 4×4×4 supercells was negligible. Consequently, this technique was not
particularly necessary.
