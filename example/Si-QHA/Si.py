"""QHA example of Si."""

import tarfile

import matplotlib.pyplot as plt
import numpy as np

from phonopy import Phonopy
from phonopy.interface.vasp import Vasprun, read_vasp


def get_force_sets(index):
    """Parse vasprun.xml and return forces."""
    with tarfile.open("vasprun_xmls.tar.lzma", "r:xz") as tr:
        with tr.extractfile("vasprun_xmls/vasprun.xml-%d" % index) as fp:
            vasprun = Vasprun(fp, use_expat=True)
            forces = vasprun.read_forces()
    return forces


def get_frequency(poscar_filename, force_sets):
    """Calculate phonons and return frequencies."""
    unitcell = read_vasp(poscar_filename)
    volume = unitcell.volume
    phonon = Phonopy(
        unitcell,
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    disps = np.zeros_like(force_sets)
    disps[0] = [0.01, 0, 0]
    phonon.dataset = {
        "natom": len(force_sets),
        "first_atoms": [
            {"number": 0, "displacement": [0.01, 0, 0], "forces": force_sets}
        ],
    }
    phonon.produce_force_constants()
    return phonon.get_frequencies([0.5, 0.5, 0]), volume


def main():
    """Run QHA."""
    frequencies = []
    volumes = []
    for i in range(-5, 6):
        poscar_filename = "POSCAR-%d" % i
        force_sets = get_force_sets(i)
        fs, v = get_frequency(poscar_filename, force_sets)
        frequencies.append(fs)
        volumes.append(v)

    for freq_at_X in np.array(frequencies).T:
        freq_squared = freq_at_X**2 * np.sign(freq_at_X)
        # np.sign is used to treat imaginary mode, since
        # imaginary frequency is returned as a negative value from phonopy.
        plt.plot(volumes, freq_squared, "o-")
        # for v, f2 in zip(volumes, freq_squared):
        #      print("%f %f" % (v, f2))
        # print('')
        # print('')
    plt.title("Frequeny squared (THz^2) at X-point vs volume (A^3)")
    plt.show()


if __name__ == "__main__":
    main()
