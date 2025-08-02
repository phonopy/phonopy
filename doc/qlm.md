(qlm_interface)=

# Questaal & phonopy calculation

This is a tutorial for the interface between phonopy and Questaal
[https://www.questaal.org/](https://www.questaal.org/).

The interface uses the site files. Such files have only a small amount of
information within them relating to the calculation. Extention to support
ctrl files is planned.

## How to run

The following is a walkthrough for a phonopy calculation with LM:

1.  Setup an `lmf` calculation with external site file instead of a `site`
    section in the ctrl file. Both relative and cartesian coordinates are
    supported. Remember to set `forces=1` in section `ham` of the ctrl file.

2.  Read the LM site file and create supercells with a command of the form,

    ```bash
        phonopy --qlm -d --dim='2 2 2' -c site.lm
    ```

    In this example, 2x2x2 supercells are created. `supercell.lm` and
    `supercell-{id}.lm` correspond to the perfect supercell and supercells
    with displacements, respectively. These supercell files are LM site files
    ready to be used in `ctrl.lm`. The original `alat` present in site.lm is
    preserved. A file named `phonopy_disp.yaml` is also created in the current
    directory. This file contains information pertaining to displacements.

3.  Next, run the calculations for the generated displacements and be sure to
    use the `--wforce=force` flag. This will write a file containing forces.
    Phonopy will require this file to be read in the next step.

4.  Create `FORCE_SETS` by:

    ```bash
        phonopy --qlm -f supercell-001/force.lm supercell-002/force.lm  ...
    ```

    To run this command, `phonopy_disp.yaml` has to be located in the current
    directory because the atomic displacements are written into the
    `FORCE_SETS` file.

5. Run post-process of phonopy with the LM site file for the unit cell used in
    the first step

    ```bash
        phonopy --qlm -c site.lm -p band.conf
    ```

    or

    ```bash
        phonopy --qlm -c site.lm --dim="2 2 2" [other-OPTIONS] [setting-file]
    ```
