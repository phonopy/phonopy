"""Group velocity example by NaCl."""

import matplotlib.pyplot as plt
import numpy as np

import phonopy

phonon = phonopy.load(
    unitcell_filename="POSCAR-unitcell",
    born_filename="BORN",
    force_sets_filename="FORCE_SETS",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
)
phonon.run_mesh([31, 31, 31], with_group_velocities=True)
phonon_mesh = phonon.get_mesh_dict()
frequencies = phonon_mesh["frequencies"]
group_velocity = phonon_mesh["group_velocities"]
gv_norm = np.sqrt((group_velocity**2).sum(axis=2))
for i, (f, g) in enumerate(zip(frequencies.T, gv_norm.T)):
    plt.plot(f, g, "o", label=("band%d" % (i + 1)))
plt.legend()
plt.xlabel("Frequency (THz)")
plt.ylabel("|group-velocity| (THz.A)")
plt.show()
