"""Example to obtain PhonopyYaml instance."""

import phonopy

phonon = phonopy.load(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    unitcell_filename="POSCAR-unitcell",
    force_sets_filename="FORCE_SETS",
    born_filename="BORN",
)
phpy_yaml = phonon.to_phonopy_yaml(settings={"force_constants": True})
print(phpy_yaml)
