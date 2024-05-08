"""A script to generate supercells with displacements for LAMMPS."""

import phonopy
from phonopy.interface.calculator import write_supercells_with_displacements
from phonopy.interface.phonopy_yaml import read_cell_yaml

cell = read_cell_yaml("phonopy_unitcell.yaml")
ph = phonopy.load(
    unitcell=cell,
    primitive_matrix="auto",
    supercell_matrix=[2, 2, 2],
    calculator="lammps",
)
ph.generate_displacements()
ph.save("phonopy_disp.yaml")
write_supercells_with_displacements(
    "lammps", ph.supercell, ph.supercells_with_displacements
)
