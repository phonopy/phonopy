import sys
import numpy as np
from phonopy import Phonopy
from phonopy.structure.symmetry import Symmetry
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.structure.spglib import get_stabilized_reciprocal_mesh
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh

cell = read_vasp(sys.argv[1])
phonon = Phonopy(cell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]], is_auto_displacements=False)
force_sets = parse_FORCE_SETS()
phonon.set_force_sets(force_sets)
phonon.set_post_process([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
primitive = phonon.get_primitive()
born = parse_BORN(primitive)
phonon.set_nac_params(born)
symmetry = phonon.get_primitive_symmetry()
mesh = [20, 20, 20]
# phonon.set_mesh(mesh)
# phonon.set_total_DOS(sigma=0.1)
# phonon.plot_total_DOS().show()

rotations = symmetry.get_pointgroup_operations()
thm = TetrahedronMesh(phonon.get_dynamical_matrix(),
                      mesh,
                      rotations,
                      is_gamma_center=True)
                      
thm.run_dos()
for f, iw in zip(thm.get_frequency_points(), thm.get_integration_weights()):
    print f, iw
