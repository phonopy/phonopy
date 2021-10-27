"""NaCl example to obtain dynamical matrix."""

import numpy as np
import yaml

import phonopy

phonon = phonopy.load(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    unitcell_filename="POSCAR-unitcell",
    force_sets_filename="FORCE_SETS",
    born_filename="BORN",
)

q = [0.1, 0.1, 0.1]
dynmat = phonon.get_dynamical_matrix_at_q(q)
print(dynmat)
phonon.run_qpoints(q, with_dynamical_matrices=True)
print(phonon.get_qpoints_dict()["frequencies"][0])
phonon.write_yaml_qpoints_phonon()

data = yaml.load(open("qpoints.yaml"), Loader=yaml.FullLoader)
dynmat_from_yaml = []
dynmat_data = data["phonon"][0]["dynamical_matrix"]
for row in dynmat_data:
    vals = np.reshape(row, (-1, 2))
    dynmat_from_yaml.append(vals[:, 0] + vals[:, 1] * 1j)
dynmat_from_yaml = np.array(dynmat_from_yaml)
print(dynmat_from_yaml)
(
    eigvals,
    eigvecs,
) = np.linalg.eigh(dynmat)
frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
conversion_factor_to_THz = 15.633302
print(frequencies * conversion_factor_to_THz)
