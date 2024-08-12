# NaCl phonon calculation with Wien2k

NaCl phonon calculation with Wien2k output files. The supercell structures are
made by

```bash
% phonopy --wien2k -c NaCl.struct -d --dim="2 2 2" --pa auto
```

This is also done for restoring `phonopy_disp.yaml` that is used in the next step.
`FORCE_SETS` is obtained by

```bash
% phonopy --wien2k -f NaCl-001.scf NaCl-002.scf
```

Phonon analysis is done such as

```bash
% phonopy-load --wien2k --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" -p
```
