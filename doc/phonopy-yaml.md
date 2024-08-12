(phonopy_yaml_format)=

# phonopy-yaml format

`phonopy_xxx.yaml` type file is a structure text format in YAML to store data
for phonopy. This file can contain the following information:

- Unit cell, primitive cell, and supercell crystal structures.
- Displacement in supercells
- Forces in supercells
- Energies of supercells
- Parameters used for non-analytical term correction

`phonopy-load` command ({ref}`phonopy_load_command`) can read this type of file, e.g.,

```bash
% phonopy-load phonopy_params.yaml [OPTIONS]
```

`phonopy` command can read this file with `-c` option

```bash
% phonopy -c phonopy_params.yaml CONFIG_FILE [OPTIONS]
```

## Compression support

`phonopy_xxx.yaml.gz`, `phonopy_xxx.yaml.bz2`, and `phonopy_xxx.yaml.xz` can be
read in the same way as shown above.

(phonopy_yaml_phonondb)=
## How to use PhononDB data

PhononDB data downloaded from <https://doi.org/10.48505/nims.4197> (its list of
crystals <https://github.com/atztogo/phonondb/blob/main/mdr/phonondb/README.md>)
are given in `phonopy_xxx.yaml.xz`. Therefore these are readily used with
`phonopy-load` or `phonopy -c` commands. For example,

```bash
% wget https://mdr.nims.go.jp/download_all/9306t5111.zip
% unzip -d mp-661 9306t5111.zip
% cd mp-661
% phonopy-load phonopy_params.yaml.xz --band auto -p --pdos auto
```

```{image} phonondb-mp-661.png
:width: 50%
```
