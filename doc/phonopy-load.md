(phonopy_load_command)=

# phonopy-load command

At phonopy v2.7.0, `phonopy-load` command is installed. This behaves similarly
to `phonopy.load` ({ref}`phonopy_load`) in the phonopy python module. The main
aim of introducing this command is to provide uniform usage over many different
force calculators. Once `phonopy_disp.yaml` is created, the following operations
will be the same using this command.

The following default behaviours are different from that of those of `phonopy`
command:

1. `phonopy_xxx.yaml` type file is always necessary in either of two ways:

   - `phonopy_xxx.yaml` type file is given as the first argument of the command.
   - `phonopy_xxx.yaml` type file is put in the current directory with one of
     the default filenames of `phonopy_params.yaml`, `phonopy_disp.yaml`,
     `phonopy.yaml`. The searching preference order is `phonopy_params.yaml` >
     `phonopy_disp.yaml` > `phonopy.yaml`.

2. `-c` option (read crystal structure) does not exist.

3. Use of command options is recommended, but phonopy configuration file can be
   read through `--config` option.

4. If parameters for non-analytical term correction (NAC) are found, NAC is
   automatically enabled. This can be disabled by `--nonac` option.

5. When force constants are calculated from displacements and forces dataset,
   force constants are automatically symmetrized. To disable this, `--no-sym-fc`
   option is used.

6. `--save-params` option is added. With this option, `phonopy_params.yaml` that
   contains most of the information to run phonopy, i.e., crystal structure,
   displacement-force dataset, and non-analytical term correction parameters.
   When displacement-force dataset didn't exist, force constants are written.

## Examples

In the NaCl-qe example,

```bash
% phonopy --qe -d --dim 2 2 2 --pa auto -c NaCl.in % phonopy-load -f NaCl-00{1,2}.out
```

With these commands, `phonopy_disp.yaml` and `FORCE_SETS` are created. After
this step, it is unnecessary to specify `--qe` option to run with
`phonopy-load`. The following command works to draw band structure.

```bash
% phonopy-load --band auto -p
```

Data in `FORCE_SETS` and `BORN` can be saved in `phonopy_params.yaml` using
`--save-params` option. Then phonons can be calculated only with this file as
fillows:

```
% phonopy-load --save-params % mkdir test && cd test % mv ../phonopy_params.yaml .
% phonopy-load --band auto -p
```

In the last line, `phonopy_params.yaml` is read without specifying because this
filename is reserved name to be searched.
