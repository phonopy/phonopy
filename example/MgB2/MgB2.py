"""Example of calculation of irreps of MgB2."""

import numpy as np

import phonopy

phonon = phonopy.load(unitcell_filename="POSCAR-unitcell", supercell_matrix=[3, 3, 2])
print("Space group: %s" % phonon.symmetry.get_international_table())

# Character table
phonon.set_irreps([1.0 / 3, 1.0 / 3, 0], 1e-4)
ct = phonon.get_irreps()
band_indices = ct.get_band_indices()
characters = np.rint(ct.get_characters()).real
for bi, cts in zip(band_indices, characters):
    print("%s %s" % (np.array(bi) + 1, cts))
# phonon.show_character_table()
