(phonopy_command)=

# phonopy command

The `phonopy` command runs the phonon calculation step of the workflow.  It
reads a `phonopy.yaml`-like file ({ref}`phonopy_yaml_format`), constructs the
phonon, and emits the requested property (band structure, mesh sampling, DOS,
thermal properties, group velocity, ...).

To prepare displacements and to convert calculator results into the phonopy
input format (`FORCE_SETS`, `FORCE_CONSTANTS`) use {ref}`phonopy_init_command`.

## Example

In the NaCl example for the VASP calculator,

```bash
% phonopy-init -d --dim 2 2 2 --pa auto -c POSCAR-unitcell
% phonopy-init --sp -f vasprun.xml-00{1,2}
```

In the NaCl-qe example for the QE calculator,

```bash
% phonopy-init --qe -d --dim 2 2 2 --pa auto -c NaCl.in
% phonopy-init --sp -f NaCl-00{1,2}.out
```

The first and second commands create `phonopy_disp.yaml` and
`phonopy_params.yaml`, respectively.

Once the `phonopy_xxx.yaml` file is available, the phonon calculation is run as
the post-process:

```bash
% phonopy --band auto -p phonopy_params.yaml
```

`phonopy` can read files with the following extensions: `xz`, `lzma`, `gz`, and
`bz2`. Therefore when the file size of `phonopy_params.yaml` is large, it is
recommended to compress it.

```bash
% xz phonopy_params.yaml
% phonopy --band auto -p phonopy_params.yaml.xz
```

(phonopy_command_behaviour)=
## Behaviour

- `phonopy_xxx.yaml`-like file ({ref}`phonopy_yaml_format`) is always required,
  provided in either of two ways:

  1. `phonopy_xxx.yaml`-like file is given as the first argument of the
     command.
  2. `phonopy_disp.yaml` or `phonopy.yaml` is placed in the current directory.
     The search preference is `phonopy_disp.yaml` > `phonopy.yaml`.

- The `-c` option (read crystal structure separately) does not exist — the
  crystal structure is read from the yaml file.

- Use of command options is recommended, but a phonopy configuration file
  ({ref}`configuration_file`) can be read through `--config` option.

- If parameters for non-analytical term correction (NAC) are found, NAC is
  automatically enabled. This can be disabled by `--nonac` option.

- When force constants are calculated from displacements and forces dataset,
  force constants are automatically symmetrized. From phonopy v2.30.0, `symfc`
  is used for the symmetrization. From phonopy v2.41.0, symfc-projector is used
  to symmetrize force constants calculated by the finite difference approach.
  The old behavior of the symmetrization can be performed by `--fc-calculator
  traditional` option. The `--no-sym-fc` option can be used to calculate force
  constants in the traditional force constants calculator without
  symmetrization.

## Relation to `phonopy-load`

`phonopy-load` is the historical name of this command and is kept as a
deprecated alias.  It emits a warning and otherwise behaves identically to
`phonopy`.  Use `phonopy` in new scripts.
