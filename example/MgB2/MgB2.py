"""Example of calculation of irreps of MgB2."""

import numpy as np

import phonopy

phonon = phonopy.load(unitcell_filename="POSCAR-unitcell", supercell_matrix=[3, 3, 2])
print("Space group: %s" % phonon.symmetry.get_international_table())

# Character table
phonon.set_irreps([1.0 / 3, 1.0 / 3, 0], degeneracy_tolerance=1e-4)
ct = phonon.irreps
assert ct is not None
band_indices = ct.band_indices
characters = np.rint(ct.characters).real  # type: ignore
for bi, cts in zip(band_indices, characters, strict=True):
    print("%s %s" % (np.array(bi) + 1, cts))
# phonon.show_character_table()
