(phonopy_init_command)=

# phonopy-init command

The `phonopy-init` command handles the setup steps that happen before phonon
calculation:

- generate supercells with displacements (`-d`, `--rd`),
- collect calculator results into `FORCE_SETS` (`-f`, `--fz`) or convert them
  into `FORCE_CONSTANTS` (`--fc`),
- inspect the crystal symmetry of the input cell (`--symmetry`).

After this step, run the phonon calculation with {ref}`phonopy_command`.

## Examples

VASP:

```bash
% phonopy-init -d --dim 2 2 2 --pa auto -c POSCAR-unitcell
% phonopy-init --sp -f vasprun.xml-00{1,2}
```

Quantum ESPRESSO:

```bash
% phonopy-init --qe -d --dim 2 2 2 --pa auto -c NaCl.in
% phonopy-init --sp -f NaCl-00{1,2}.out
```

Symmetry inspection:

```bash
% phonopy-init --symmetry -c POSCAR-unitcell
```

Random displacements with a finite snapshot count:

```bash
% phonopy-init --rd 100 --dim 2 2 2 --pa auto -c POSCAR-unitcell
```

Finite-temperature random displacements require phonon information and belong
to the phonon-calculation step, so they are handled by {ref}`phonopy_command`
rather than `phonopy-init`.

## Relation to the legacy `phonopy` command

Earlier versions of phonopy used a single `phonopy` command to cover both the
setup and the phonon-calculation steps.  The two responsibilities have been
split:

| Old invocation                   | New invocation                      |
| -------------------------------- | ----------------------------------- |
| `phonopy -d --dim ... -c POSCAR` | `phonopy-init -d --dim ... -c POSCAR` |
| `phonopy --sp -f vasprun.xml*`   | `phonopy-init --sp -f vasprun.xml*`   |
| `phonopy --symmetry -c POSCAR`   | `phonopy-init --symmetry -c POSCAR`   |
| `phonopy-load --band auto ...`   | `phonopy --band auto ...`             |

The new `phonopy` command rejects setup flags (`-d`, `--rd`, `-f`, `--fz`,
`--fc`, `--symmetry`) and points the user to `phonopy-init`.
