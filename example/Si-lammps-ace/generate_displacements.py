"""A script to generate supercells with displacements for LAMMPS.

Generating the supercells from a yaml unit cell keeps the cell in its original
(unrotated) Cartesian orientation in ``phonopy_disp.yaml``, in contrast to
reading a LAMMPS structure file, which is given in the rotated triclinic
convention. The generated ``supercell-001`` follows the LAMMPS structure input
format; the forces obtained from LAMMPS are rotated back automatically when
FORCE_SETS is created.

"""

import phonopy
from phonopy.interface.calculator import write_supercells_with_displacements
from phonopy.interface.phonopy_yaml import read_cell_yaml

cell = read_cell_yaml("phonopy_unitcell.yaml")
ph = phonopy.load(
    unitcell=cell,
    supercell_matrix=[2, 2, 2],  # use a larger size (e.g. [4, 4, 4]) for convergence
    calculator="lammps",
)
ph.generate_displacements()
ph.save("phonopy_disp.yaml")
write_supercells_with_displacements(
    "lammps", ph.supercell, ph.supercells_with_displacements
)
