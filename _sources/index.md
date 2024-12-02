# Welcome to phonopy

**Phonopy** is an open source package for phonon calculations at harmonic and
quasi-harmonic levels.

[Phono3py](http://phonopy.github.io/phono3py/) is another open source package
for phonon-phonon interaction and lattice thermal conductivity calculations.

**PhononDB**: URL links of first principles phonon calculation data at
[nims.4197](https://doi.org/10.48505/nims.4197) are found at
<https://github.com/atztogo/phonondb/blob/main/mdr/phonondb/README.md>.
See {ref}`phonopy_yaml_phonondb`.


The following features of phonopy are highlighted:

- {ref}`Phonon band structure <band_structure_related_tags>`,
  {ref}`phonon DOS and partial-DOS <dos_related_tags>`
- {ref}`Phonon thermal properties <thermal_properties_tag>`: Free energy, heat
  capacity (Cv), and entropy
- {ref}`Phonon group velocity <group_velocity>`
- {ref}`Thermal ellipsoids <thermal_displacement_matrices_tag>` /
  {ref}`Mean square displacements <thermal_displacements_tag>`
- {ref}`Irreducible representations of normal modes <irreducible_representation_related_tags>`
- {ref}`Dynamic structure factor for INS and IXS <dynamic_structure_factor>`
- {ref}`Non-analytical-term correction <nac_tag>`: LO-TO splitting
  ({ref}`Born effective charges and dielectric constant are required. <born_file>`)
- {ref}`Mode Grüneisen parameters <phonopy_gruneisen>`
- {ref}`Quasi-harmonic approximation <phonopy_qha>`: Thermal expansion, heat
  capacity at constant pressure (Cp)
- {ref}`Interfaces to calculators <calculator_interfaces>`: {ref}`VASP
<vasp_interface>`, {ref}`VASP DFPT <vasp_dfpt_interface>`, {ref}`ABINIT
<abinit_interface>`, {ref}`Quantu ESPRESSO <qe_interface>`, {ref}`SIESTA
<siesta_interface>`, {ref}`Elk <elk_interface>`, {ref}`WIEN2k
<wien2k_interface>`, {ref}`CRYSTAL <crystal_interface>`, {ref}`DFTB+
<dftbp_interface>`, {ref}`TURBOMOLE <turbomole_interface>`, {ref}`CP2K
<cp2k_interface>`, {ref}`FHI-aims <FHI_aims_interface>`, {ref}`CASTEP
<castep_interface>`, {ref}`Fleur <Fleur_interface>`, {ref}`ABACUS
<abacus_interface>`, {ref}`LAMMPS <lammps_interface>`, {ref}`LAMMPS (external)
<external_tools_phonolammps>`
- {ref}`Phonopy API for Python <phonopy_module>`

```{toctree}
:hidden:
install
symmetry
workflow
examples
input-files
output-files
setting-tags
command-options
dynamic-structure-factor
Mode Grüneisen parameters <gruneisen>
qha
random-displacements
interfaces
auxiliary-tools
external-tools
phonopy-module
phonopy-yaml
phonopy-load
mlp-sscha
formulation
citation
reference
changelog
```

<!-- Latex master doc is documentation.md. But documentation.md is not included
for html. Uncomment below when generating latex documentation. -->

<!-- ```{toctree}
:hidden:
documentation
```
-->

(mailinglist)=

## Mailing list

For questions, bug reports, and comments, please visit
<https://lists.sourceforge.net/lists/listinfo/phonopy-users> to subscribe the
phonopy mailing list and send them to <phonopy-users@lists.sourceforge.net>.
Message body including attached files has to be smaller than 300 KB.

## License

New BSD from version 1.3.

(LGPL from ver. 0.9.3 to version 1.2.1., GPL to version 0.9.2.)

## Contributors

- Atsushi Togo, National Institute for Materials Science

## Acknowledgements

Phonopy development is supported by:

- National Institute for Materials Science
