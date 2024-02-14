"""NaCl band structure calculation example."""

import phonopy
from phonopy.phonon.band_structure import get_band_qpoints

phonon = phonopy.load(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    unitcell_filename="POSCAR-unitcell",
    force_sets_filename="FORCE_SETS",
    born_filename="BORN",
)
points = get_band_qpoints(
    [[[0.5, 0, 0.5], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.25, 0.75]]], 51
)
phonon.run_band_structure(points, labels=["X", r"$\Gamma$", "L", "W"])
phonon.plot_band_structure().show()
