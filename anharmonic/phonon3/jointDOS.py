import sys
import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.file_IO import parse_BORN
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.units import VaspToTHz
from anharmonic.phonon3.triplets import get_triplets_at_q, get_nosym_triplets_at_q
from anharmonic.file_IO import write_jointDOS

#
# Joint density of states
#    
def get_jointDOS(fixed_grid_points,
                 mesh,
                 primitive,
                 supercell,
                 fc2,
                 nac_params=None,
                 sigma=0.2,
                 frequency_step=0.1,
                 factor=VaspToTHz,
                 frequency_factor=1.0,
                 frequency_scale=1.0,
                 is_nosym=False,
                 symprec=1e-5,
                 filename=None,
                 log_level=False):

    try:
        import anharmonic._phono3py as phono3c
    except ImportError:
        print "Joint density of states in python is not implemented."
        return None, None
    
    symmetry = Symmetry(primitive, symprec)

    if nac_params==None:
        dm = DynamicalMatrix(supercell,
                             primitive,
                             fc2,
                             frequency_scale_factor=frequency_scale,
                             symprec=symprec)
    else:
        dm = DynamicalMatrixNAC(supercell,
                                primitive,
                                fc2,
                                frequency_scale_factor=frequency_scale,
                                symprec=symprec)
        dm.set_nac_params(nac_params)

    jointDOS = []
    omegas = []
    reciprocal_lattice = np.linalg.inv(primitive.get_cell())
    num_band = primitive.get_number_of_atoms() * 3
    for gp in fixed_grid_points:
        if is_nosym:
            if log_level:
                print "Triplets at q without considering symmetry"
                sys.stdout.flush()
            
            (triplets_at_q,
             weights_at_q,
             grid_points) = get_nosym_triplets_at_q(
                 gp,
                 mesh,
                 reciprocal_lattice)
        else:
            triplets_at_q, weights_at_q, grid_points = get_triplets_at_q(
                gp,
                mesh,
                symmetry.get_pointgroup_operations(),
                reciprocal_lattice)

        if log_level:
            print "Grid point (%d):" % gp,  grid_points[gp]
            if is_nosym:
                print "Number of ir triplets:",
            else:
                print "Number of triplets:",
            print (len(weights_at_q))
            print "Sum of weights:", weights_at_q.sum()
            sys.stdout.flush()

        freqs = np.zeros((len(triplets_at_q), 3, num_band), dtype='double')
        
        for i, g3 in enumerate(triplets_at_q):
            q3 = grid_points[g3] / np.array(mesh, dtype='double')
            for j, q in enumerate(q3):
                dm.set_dynamical_matrix(q)
                val = np.linalg.eigvalsh(dm.get_dynamical_matrix())
                freqs[i, j] = np.sqrt(np.abs(val)) * factor * frequency_factor

        omegas_at_gp = np.arange(0, np.max(freqs) * 2 + sigma * 4,
                                 frequency_step)
        jointDOS_at_gp = np.zeros(len(omegas_at_gp), dtype='double')

        phono3c.joint_dos(jointDOS_at_gp,
                          omegas_at_gp,
                          weights_at_q,
                          freqs,
                          sigma)
        
        jdos = jointDOS_at_gp / weights_at_q.sum()
        jointDOS.append(jdos)
        omegas.append(omegas_at_gp)

        write_jointDOS(gp,
                       mesh,
                       omegas_at_gp,
                       jdos,
                       filename=filename,
                       is_nosym=is_nosym)

    return jointDOS, omegas

    
