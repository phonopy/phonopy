"""Example to obtain PhonopyYaml instance."""

import phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml

phonon = phonopy.load(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    unitcell_filename="POSCAR-unitcell",
    force_sets_filename="FORCE_SETS",
    born_filename="BORN",
)
phpy_yaml = PhonopyYaml(calculator="vasp", settings={"force_constants": True})
phpy_yaml.set_phonon_info(phonon)
print(phpy_yaml)
