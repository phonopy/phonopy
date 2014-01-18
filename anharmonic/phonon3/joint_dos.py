import sys
import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.file_IO import parse_BORN
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.units import VaspToTHz
from anharmonic.phonon3.triplets import get_triplets_at_q, get_nosym_triplets_at_q, get_tetrahedra_vertices
from anharmonic.phonon3.interaction import get_dynamical_matrix, set_phonon_c
from phonopy.structure.tetrahedron_method import TetrahedronMethod

class JointDos:
    def __init__(self,
                 mesh,
                 primitive,
                 supercell,
                 fc2,
                 nac_params=None,
                 sigma=0.1,
                 tetrahedron_method=False,
                 frequency_step=0.1,
                 frequency_factor_to_THz=VaspToTHz,
                 frequency_scale_factor=1.0,
                 is_nosym=False,
                 symprec=1e-5,
                 filename=None,
                 log_level=False,
                 lapack_zheev_uplo='L'):

        self._grid_point = None
        self._mesh = np.array(mesh, dtype='intc')
        self._primitive = primitive
        self._supercell = supercell
        self._fc2 = fc2
        self._nac_params = nac_params
        self.set_sigma(sigma)

        self._frequency_step = frequency_step
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._filename = filename
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._num_band = self._primitive.get_number_of_atoms() * 3
        self._reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        self._set_dynamical_matrix()
        self._symmetry = Symmetry(primitive, symprec)

        if tetrahedron_method:
            self._tetrahedron_method = TetrahedronMethod(
                self._reciprocal_lattice)
        else:
            self._tetrahedron_method = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._nac_q_direction = None
            
        self._joint_dos = None
        self._frequency_points = None

    def run(self, grid_point):
        self._grid_point = grid_point

        mesh_with_boundary = self._mesh + 1
        num_grid = np.prod(mesh_with_boundary)
        num_band = self._num_band
        if self._phonon_done is None:
            self._phonon_done = np.zeros(num_grid, dtype='byte')
            self._frequencies = np.zeros((num_grid, num_band), dtype='double')
            self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                          dtype='complex128')
            
        try:
            import anharmonic._phono3py as phono3c
            self._run_c()
        except ImportError:
            print "Joint density of states in python is not implemented."
            return None, None

    def get_joint_dos(self):
        return self._joint_dos

    def get_frequency_points(self):
        return self._frequency_points
        
    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)
            
    def _run_c(self):
        gp = self._grid_point
        self._joint_dos = []
        self._frequency_points = []
        (self._triplets_at_q,
         self._weights_at_q,
         self._grid_address,
         self._bz_map) = self._get_triplets(gp)
        
        if self._log_level:
            print "Grid point (%d):" % gp,  self._grid_address[gp]
            if self._tetrahedron_method is None:
                print "Sigma:", self._sigma
            else:
                print("Tetrahedron method is used for " 
                      "Brillouin zone integration.")
            if self._is_nosym:
                print "Number of ir triplets:",
            else:
                print "Number of triplets:",
            print (len(self._weights_at_q))
            # print "Sum of weights:", self._weights_at_q.sum()
            sys.stdout.flush()

        if self._tetrahedron_method is None:
            self._run_smearing_method()
        else:
            self._run_tetrahedron_method()

    def _run_tetrahedron_method(self):
        self._vertices = get_tetrahedra_vertices(
            self._tetrahedron_method.get_tetrahedra(),
            self._mesh,
            self._triplets_at_q,
            self._grid_address,
            self._bz_map)
        self._set_phonon_at_grid_points(self._vertices.ravel())
        thm = self._tetrahedron_method
        f_max = np.max(self._frequencies) * 2 + self._frequency_step / 10
        f_min = np.min(self._frequencies) * 2
        freq_points = np.arange(f_min, f_max, self._frequency_step,
                                dtype='double')
        jdos = np.zeros_like(freq_points)
        for vertices, w in zip(self._vertices, self._weights_at_q):
            for i, j in list(np.ndindex(self._num_band, self._num_band)):
                f1 = self._frequencies[vertices[0], i]
                f2 = self._frequencies[vertices[1], j]
                thm.set_tetrahedra_omegas(f1 + f2)
                thm.run(freq_points)
                iw = thm.get_integration_weight()
                jdos += iw * w

        self._joint_dos = jdos / np.prod(self._mesh)
        self._frequency_points = freq_points

    def _run_smearing_method(self):
        import anharmonic._phono3py as phono3c

        self._set_phonon_at_grid_points(self._triplets_at_q.ravel())
        f_max = np.max(self._frequencies) * 2 + self._sigma * 4
        f_min = np.min(self._frequencies) * 2 - self._sigma * 4
        freq_points = np.arange(f_min, f_max, self._frequency_step,
                                dtype='double')
        jdos = np.zeros_like(freq_points)
        phono3c.joint_dos(jdos,
                          freq_points,
                          self._triplets_at_q,
                          self._weights_at_q,
                          self._frequencies,
                          self._sigma)
        jdos /= np.prod(self._mesh)
        self._joint_dos = jdos
        self._frequency_points = freq_points
        
    def _set_dynamical_matrix(self):
        self._dm = get_dynamical_matrix(
            self._fc2,
            self._supercell,
            self._primitive,
            nac_params=self._nac_params,
            frequency_scale_factor=self._frequency_scale_factor,
            symprec=self._symprec)
        
    def _get_triplets(self, gp):
        if self._is_nosym:
            if self._log_level:
                print "Triplets at q without considering symmetry"
                sys.stdout.flush()
            
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map) = get_nosym_triplets_at_q(
                 gp,
                 self._mesh,
                 self._reciprocal_lattice,
                 with_bz_map=True)
        else:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map) = get_triplets_at_q(
                 gp,
                 self._mesh,
                 self._symmetry.get_pointgroup_operations(),
                 self._reciprocal_lattice,
                 with_bz_map=True)

        return triplets_at_q, weights_at_q, grid_address, bz_map

    def _set_phonon_at_grid_points(self, grid_points):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
                     self._phonon_done,
                     grid_points,
                     self._grid_address,
                     self._mesh,
                     self._frequency_factor_to_THz,
                     self._nac_q_direction,
                     self._lapack_zheev_uplo)
