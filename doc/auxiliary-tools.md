(auxiliary_tools)=

# Auxiliary tools

A few auxiliary tools are prepared. They are stored in `bin` directory as well
as `phonopy`.

(bandplot_tool)=

## `phonopy-bandplot`

Band structure is plotted reading phonopy output in `band.yaml` format. `-o`
option with a file name is used to save the plot into a file in PDF format. A
few more options are prepared and shown by `-h` option. If you specify more than
two yaml files, they are plotted together.

```bash
% phonopy-bandplot band.yaml
```

To obtain a simple text format data:

```bash
% phonopy-bandplot --gnuplot band.yaml
```

(pdosplot_tool)=

## `phonopy-pdosplot`

Partial density of states (PDOS) are plotted.

`-i` option is used as

```bash
% phonopy-pdosplot -i '1 2 4 5, 3 6' -o 'pdos.pdf' partial_dos.dat
```

The indices and comma in `1 2 3 4, 5 6` mean as follows. The indices are
separated into blocks by comma (1 2 4 5 and 3 6). PDOS specified by the
successive indices separated by space in each block are summed up. The PDOS of
blocks are drawn simultaneously. Indices usually correspond to atoms. A few more
options are prepared and shown by `-h` option.

(propplot_tool)=

## `phonopy-propplot`

Thermal properties are plotted. Options are prepared and shown by `-h` option.
If you specify more than two yaml files, they are plotted together.

```
% phonopy-proplot thermal_properties_A.yaml thermal_properties_B.yaml
```

## `phonopy-vasp-born`

This script is used to create a `BORN` style file from VASP output file of
`vasprun.xml`. The first argument is a `vasprun.xml` file. If it is omitted,
`vasprun.xml` at current directory are read. The Born effective charges and
dielectric tensor are symmetrized as default. To prevent symmetrization,
`--nost` option has to be specified.

```bash
% phonopy-vasp-born
```

```bash
% phonopy-vasp-born --nost
```

### `--pa`, `--primitive-axes`

This is same as {ref}`primitive_axes_tag`.

### `--dim`

This is same as {ref}`dimension_tag`.

### `--nost`

Dielectric constant and Born effective charge tensors are not symmetrized.

### `--outcar`

Read `OUTCAR` instead of `vasprun.xml`. Without specifying arguments, `OUTCAR`
and `POSCAR` at current directory are read. `POSCAR` information is necessary in
contrast to reading `vasprun.xml` where the unit cell structure is also read
from it.

```bash
% phonopy-vasp-born --outcar
```

```bash
% phonopy-vasp-born --nost --outcar OUTCAR POSCAR
```
