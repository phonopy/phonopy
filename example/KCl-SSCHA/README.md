# Example of KCl SSCHA calculation

Developing polynomial machine learning potentials (MLPs) by

```
% phonopy-load phonopy_mlpsscha_params_KCl-120.yaml.xz --pypolymlp --mlp-params="ntrain=100, ntest=20"
```

The `phonopy.pmlp` file is generated. This file contains the developed MLPs and
read when running phonopy with `--pypolymlp`. The SSCHA calculation is peformed
by

```
% phonopy-load phonopy_mlpsscha_params_KCl-120.yaml.xz --pypolymlp --sscha 10 --rd-temperature 300
```

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
