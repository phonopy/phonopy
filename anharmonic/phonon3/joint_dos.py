import sys
import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz
from anharmonic.phonon3.triplets import get_triplets_at_q, get_nosym_triplets_at_q, get_tetrahedra_vertices, get_triplets_integration_weights
from anharmonic.other.phonon import get_dynamical_matrix, set_phonon_c
from phonopy.structure.tetrahedron_method import TetrahedronMethod

class JointDos:
    def __init__(self,
                 mesh,
                 primitive,
                 supercell,
                 fc2,
                 nac_params=None,
                 nac_q_direction=None,
                 sigma=None,
                 frequency_step=None,
                 num_frequency_points=None,
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
        self._nac_q_direction = None
        self.set_nac_q_direction(nac_q_direction)
        self._sigma = None
        self.set_sigma(sigma)

        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
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

        self._tetrahedron_method = None
        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
            
        self._joint_dos = None
        self._frequency_points = None

    def run(self):
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

    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done

    def get_primitive(self):
        return self._primitive

    def get_mesh_numbers(self):
        return self._mesh
        
    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    def set_grid_point(self, grid_point):
        self._grid_point = grid_point
        self._set_triplets()
        num_grid = np.prod(len(self._grid_address))
        num_band = self._num_band
        if self._phonon_done is None:
            self._phonon_done = np.zeros(num_grid, dtype='byte')
            self._frequencies = np.zeros((num_grid, num_band), dtype='double')
            self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                          dtype='complex128')
            
        self._joint_dos = None
        self._frequency_points = None
        self.set_phonon(np.array([grid_point], dtype='intc'))

    def get_triplets_at_q(self):
        return self._triplets_at_q, self._weights_at_q
        
    def get_grid_address(self):
        return self._grid_address

    def get_bz_map(self):
        return self._bz_map
    
    def _run_c(self, lang='C'):
        if self._sigma is None:
            if lang == 'C':
                self._run_c_tetrahedron_method()
            else:
                self._run_py_tetrahedron_method()
        else:
            self._run_smearing_method()

    def _run_c_tetrahedron_method(self):
        """
        This is not very faster than _run_c_tetrahedron_method and
        use much more memory space. So this function is not set as default.
        """

        self.set_phonon(self._triplets_at_q.ravel())
        f_max = np.max(self._frequencies) * 2
        f_max *= 1.005
        f_min = 0
        self._set_frequency_points(f_min, f_max)

        num_band = self._num_band
        num_triplets = len(self._triplets_at_q)
        num_freq_points = len(self._frequency_points)
        num_mesh = np.prod(self._mesh)
        jdos = np.zeros((num_freq_points, 2), dtype='double')
        
        for i, freq_point in enumerate(self._frequency_points):
            g = get_triplets_integration_weights(self,
                                                 [freq_point],
                                                 None,
                                                 is_collision_matrix=True,
                                                 neighboring_phonons=(i == 0))
            jdos[i, 0] = np.sum(
                np.tensordot(g[0, :, 0], self._weights_at_q, axes=(0, 0)))
            gx = g[2] - g[0]
            jdos[i, 1] = np.sum(
                np.tensordot(gx[:, 0], self._weights_at_q, axes=(0, 0)))
            if self._log_level > 1:
                print "%4d %f %e %e" % ((i + 1, freq_point,) +
                                        tuple(jdos[i] / num_mesh))

        self._joint_dos = jdos / num_mesh
    
    def _run_py_tetrahedron_method(self):
        thm = TetrahedronMethod(self._reciprocal_lattice, mesh=self._mesh)
        self._vertices = get_tetrahedra_vertices(
            thm.get_tetrahedra(),
            self._mesh,
            self._triplets_at_q,
            self._grid_address,
            self._bz_map)
        self.set_phonon(self._vertices.ravel())
        f_max = np.max(self._frequencies) * 2
        f_max *= 1.005
        f_min = np.min(self._frequencies) * 2
        self._set_frequency_points(f_min, f_max)

        num_freq_points = len(self._frequency_points)
        jdos = np.zeros((num_freq_points, 2), dtype='double')
        for vertices, w in zip(self._vertices, self._weights_at_q):
            for i, j in list(np.ndindex(self._num_band, self._num_band)):
                f1 = self._frequencies[vertices[0], i]
                f2 = self._frequencies[vertices[1], j]
                thm.set_tetrahedra_omegas(f1 + f2)
                thm.run(self._frequency_points)
                iw = thm.get_integration_weight()
                jdos[:, 0] += iw * w

                thm.set_tetrahedra_omegas(f1 - f2)
                thm.run(self._frequency_points)
                iw = thm.get_integration_weight()
                jdos[:, 1] += iw * w

                thm.set_tetrahedra_omegas(-f1 + f2)
                thm.run(self._frequency_points)
                iw = thm.get_integration_weight()
                jdos[:, 1] += iw * w

        self._joint_dos = jdos / np.prod(self._mesh)

    def _run_smearing_method(self):
        import anharmonic._phono3py as phono3c

        self.set_phonon(self._triplets_at_q.ravel())
        f_max = np.max(self._frequencies) * 2 + self._sigma * 4
        f_min = np.min(self._frequencies) * 2 - self._sigma * 4
        self._set_frequency_points(f_min, f_max)
        jdos = np.zeros((len(self._frequency_points), 2), dtype='double')
        phono3c.joint_dos(jdos,
                          self._frequency_points,
                          self._triplets_at_q,
                          self._weights_at_q,
                          self._frequencies,
                          self._sigma)
        jdos /= np.prod(self._mesh)
        self._joint_dos = jdos
        
    def _set_dynamical_matrix(self):
        self._dm = get_dynamical_matrix(
            self._fc2,
            self._supercell,
            self._primitive,
            nac_params=self._nac_params,
            frequency_scale_factor=self._frequency_scale_factor,
            symprec=self._symprec)
        
    def _set_triplets(self):
        if self._is_nosym:
            if self._log_level:
                print "Triplets at q without considering symmetry"
                sys.stdout.flush()
            
            (self._triplets_at_q,
             self._weights_at_q,
             self._grid_address,
             self._bz_map,
             map_triplets,
             map_q) = get_nosym_triplets_at_q(
                 self._grid_point,
                 self._mesh,
                 self._reciprocal_lattice,
                 with_bz_map=True)
        else:
            (self._triplets_at_q,
             self._weights_at_q,
             self._grid_address,
             self._bz_map,
             map_triplets,
             map_q) = get_triplets_at_q(
                 self._grid_point,
                 self._mesh,
                 self._symmetry.get_pointgroup_operations(),
                 self._reciprocal_lattice)

    def set_phonon(self, grid_points):
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

    def _set_frequency_points(self, f_min, f_max):
        if self._num_frequency_points is None:
            if self._frequency_step is not None:
                self._frequency_points = np.arange(
                    f_min, f_max, self._frequency_step, dtype='double')
            else:
                self._frequency_points = np.array(np.linspace(
                    f_min, f_max, 201), dtype='double')
        else:
            self._frequency_points = np.array(np.linspace(
                f_min, f_max, self._num_frequency_points), dtype='double')
