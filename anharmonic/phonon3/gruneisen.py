import sys
import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC, get_smallest_vectors
from phonopy.structure.cells import get_supercell
from anharmonic.file_IO import write_fc3_dat, write_fc2_dat
from phonopy.units import VaspToTHz
from phonopy.structure.grid_points import get_qpoints

def get_gruneisen_parameters(fc2,
                             fc3,
                             supercell,
                             primitive,
                             band_paths,
                             mesh,
                             qpoints,
                             nac_params=None,
                             nac_q_direction=None,
                             ion_clamped=False,
                             factor=None,
                             symprec=1e-5,
                             output_filename=None,
                             log_level=True):
    if log_level:
        print("-" * 23 + " Phonon Gruneisen parameter " + "-" * 23)
        if mesh is not None:
            print("Mesh sampling: [ %d %d %d ]" % tuple(mesh))
        elif band_paths is not None:
            print("Paths in reciprocal reduced coordinates:")
            for path in band_paths:
                print("[%5.2f %5.2f %5.2f] --> [%5.2f %5.2f %5.2f]" %
                      (tuple(path[0]) + tuple(path[-1])))
        if ion_clamped:
            print("To be calculated with ion clamped.")
            
        sys.stdout.flush()

    gruneisen = Gruneisen(fc2,
                          fc3,
                          supercell,
                          primitive,
                          nac_params=nac_params,
                          nac_q_direction=nac_q_direction,
                          ion_clamped=ion_clamped,
                          factor=factor,
                          symprec=symprec)

    if mesh is not None:
        gruneisen.set_sampling_mesh(mesh, is_gamma_center=True)
    elif band_paths is not None:
        gruneisen.set_band_structure(band_paths)
    elif qpoints is not None:
        gruneisen.set_qpoints(qpoints)
    gruneisen.run()

    if output_filename is None:
        filename = 'gruneisen3.yaml'
    else:
        filename = 'gruneisen3.' + output_filename + '.yaml'
    gruneisen.write_yaml(filename=filename)

class Gruneisen(object):
    def __init__(self,
                 fc2,
                 fc3,
                 supercell,
                 primitive,
                 nac_params=None,
                 nac_q_direction=None,
                 ion_clamped=False,
                 factor=VaspToTHz,
                 symprec=1e-5):
        self._fc2 = fc2
        self._fc3 = fc3
        self._scell = supercell
        self._pcell = primitive
        self._ion_clamped = ion_clamped
        self._factor = factor
        self._symprec = symprec
        if nac_params is None:
            self._dm = DynamicalMatrix(self._scell,
                                       self._pcell,
                                       self._fc2,
                                       symprec=self._symprec)
        else:
            self._dm = DynamicalMatrixNAC(self._scell,
                                          self._pcell,
                                          self._fc2,
                                          symprec=self._symprec)
            self._dm.set_nac_params(nac_params)
        self._nac_q_direction = nac_q_direction
        self._shortest_vectors, self._multiplicity = get_smallest_vectors(
            self._scell, self._pcell, self._symprec)

        if self._ion_clamped:
            num_atom_prim = self._pcell.get_number_of_atoms()
            self._X = np.zeros((num_atom_prim, 3, 3, 3), dtype=float)
        else:
            self._X = self._get_X()
        self._dPhidu = self._get_dPhidu()

        self._gruneisen_parameters = None
        self._frequencies = None
        self._qpoints = None
        self._mesh = None
        self._band_paths = None
        self._band_distances = None
        self._run_mode = None
        self._weights = None

    def run(self):
        if self._run_mode == 'band':
            (self._gruneisen_parameters,
             self._frequencies) = self._calculate_band_paths()
        elif self._run_mode == 'qpoints' or self._run_mode == 'mesh':
            (self._gruneisen_parameters,
             self._frequencies) = self._calculate_at_qpoints(self._qpoints)
        else:
            sys.stderr.write('Q-points are not specified.\n')

    def get_gruneisen_parameters(self):
        return self._gruneisen_parameters

    def set_qpoints(self, qpoints):
        self._run_mode = 'qpoints'
        self._qpoints = qpoints

    def set_sampling_mesh(self,
                          mesh,
                          shift=None,
                          is_gamma_center=False):
        self._run_mode = 'mesh'
        self._mesh = mesh
        self._qpoints, self._weights = get_qpoints(
            self._mesh,
            np.linalg.inv(self._pcell.get_cell()),
            q_mesh_shift=shift,
            is_gamma_center=is_gamma_center)

    def set_band_structure(self, paths):
        self._run_mode = 'band'
        self._band_paths = paths
        rec_lattice = np.linalg.inv(self._pcell.get_cell())
        self._band_distances = []
        for path in paths:
            distances_at_path = [0.]
            for i in range(len(path) - 1):
                distances_at_path.append(np.linalg.norm(
                        np.dot(rec_lattice, path[i + 1] - path[i])) +
                                         distances_at_path[-1])
            self._band_distances.append(distances_at_path)

    def write_yaml(self, filename="gruneisen3.yaml"):
        if self._gruneisen_parameters is not None:
            f = open(filename, 'w')
            if self._run_mode == 'band':
                self._write_band_yaml(f)
            elif self._run_mode == 'qpoints' or self._run_mode == 'mesh':
                self._write_yaml(f)
            f.close()

    def _write_yaml(self, f):
        if self._run_mode == 'mesh':
            f.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
        f.write("nqpoint: %d\n" % len(self._qpoints))
        f.write("phonon:\n")
        for i, (q, g_at_q, freqs_at_q) in enumerate(
            zip(self._qpoints,
                self._gruneisen_parameters,
                self._frequencies)):
            f.write("- q-position: [ %10.7f, %10.7f, %10.7f ]\n" % tuple(q))
            if self._weights is not None:
                f.write("  multiplicity: %d\n" % self._weights[i])
            f.write("  band:\n")
            for j, (g, freq) in enumerate(zip(g_at_q, freqs_at_q)):
                f.write("  - # %d\n" % (j + 1))
                f.write("    frequency: %15.10f\n" % freq)
                f.write("    gruneisen: %15.10f\n" % (g.trace() / 3))
                f.write("    gruneisen_tensor:\n")
                for g_xyz in g:
                    f.write("    - [ %10.7f, %10.7f, %10.7f ]\n" %
                            tuple(g_xyz))
        
    def _write_band_yaml(self, f):
        f.write("path:\n\n")
        for path, distances, gs, fs in zip(self._band_paths,
                                           self._band_distances,
                                           self._gruneisen_parameters,
                                           self._frequencies):
            f.write("- nqpoint: %d\n" % len(path))
            f.write("  phonon:\n")
            for i, (q, d, g_at_q, freqs_at_q) in enumerate(
                zip(path, distances, gs, fs)):
                f.write("  - q-position: [ %10.7f, %10.7f, %10.7f ]\n"
                        % tuple(q))
                f.write("    distance: %10.7f\n" % d)
                f.write("    band:\n")
                for j, (g, freq) in enumerate(zip(g_at_q, freqs_at_q)):
                    f.write("    - # %d\n" % (j + 1))
                    f.write("      frequency: %15.10f\n" % freq)
                    f.write("      gruneisen: %15.10f\n" % (g.trace() / 3))
                    f.write("      gruneisen_tensor:\n")
                    for g_xyz in g:
                        f.write("      - [ %10.7f, %10.7f, %10.7f ]\n" %
                                tuple(g_xyz))
                f.write("\n")
                        
        
    def _calculate_at_qpoints(self, qpoints):
        gruneisen_parameters = []
        frequencies = []
        for i, q in enumerate(qpoints):
            if self._dm.is_nac():
                if (np.abs(q) < 1e-5).all(): # If q is almost at Gamma
                    if self._run_mode == 'band': 
                        # Direction estimated from neighboring point
                        if i > 0:
                            q_dir = qpoints[i] - qpoints[i - 1]
                        elif i == 0 and len(qpoints) > 1:
                            q_dir = qpoints[i + 1] - qpoints[i]
                        else:
                            q_dir = None
                        g, omega2 = self._get_gruneisen_tensor(
                            q, nac_q_direction=q_dir)
                    else: # Specified q-vector
                        g, omega2 = self._get_gruneisen_tensor(
                            q, nac_q_direction=self._nac_q_direction)
                else: # If q is away from Gamma-point, then q-vector
                    g, omega2 = self._get_gruneisen_tensor(q, nac_q_direction=q)
            else: # Without NAC
                g, omega2 = self._get_gruneisen_tensor(q)
            gruneisen_parameters.append(g)
            frequencies.append(
                np.sqrt(abs(omega2)) * np.sign(omega2) * self._factor)

        return gruneisen_parameters, frequencies
            
    def _calculate_band_paths(self):
        gruneisen_parameters = []
        frequencies = []
        for path in self._band_paths:
            (gruneisen_at_path,
             frequencies_at_path) = self._calculate_at_qpoints(path)
            gruneisen_parameters.append(gruneisen_at_path)
            frequencies.append(frequencies_at_path)

        return gruneisen_parameters, frequencies

    def _get_gruneisen_tensor(self, q, nac_q_direction=None):
        if nac_q_direction is None:
            self._dm.set_dynamical_matrix(q)
        else:
            self._dm.set_dynamical_matrix(q, nac_q_direction)
        omega2, w = np.linalg.eigh(self._dm.get_dynamical_matrix())
        g = np.zeros((len(omega2), 3, 3), dtype=float)
        num_atom_prim = self._pcell.get_number_of_atoms()
        dDdu = self._get_dDdu(q)

        for s in range(len(omega2)):
            if (np.abs(q) < 1e-5).all() and s < 3:
                continue
            for i in range(3):
                for j in range(3):
                    for nu in range(num_atom_prim):
                        for pi in range(num_atom_prim):
                            g[s] += (w[nu * 3 + i, s].conjugate() * 
                                     dDdu[nu, pi, i, j] * w[pi * 3 + j, s]).real

            g[s] *= -1.0 / 2 / omega2[s]

        return g, omega2

    def _get_dDdu(self, q):
        num_atom_prim = self._pcell.get_number_of_atoms()
        num_atom_super = self._scell.get_number_of_atoms()
        p2s = self._pcell.get_primitive_to_supercell_map()
        s2p = self._pcell.get_supercell_to_primitive_map()
        vecs = self._shortest_vectors
        multi = self._multiplicity
        m = self._pcell.get_masses()
        dPhidu = self._dPhidu
        dDdu = np.zeros((num_atom_prim, num_atom_prim, 3, 3, 3, 3), dtype=complex)
        
        for nu in range(num_atom_prim):
            for pi, p in enumerate(p2s):
                for Ppi, s in enumerate(s2p):
                    if not s==p:
                        continue
                    phase = np.exp(2j * np.pi * np.dot(
                            vecs[Ppi,nu,:multi[Ppi, nu], :], q)
                                   ).sum() / multi[Ppi, nu]
                    dDdu[nu, pi] += phase * dPhidu[nu, Ppi]
                dDdu[nu, pi] /= np.sqrt(m[nu] * m[pi])

        return dDdu
                                    
    def _get_dPhidu(self):
        fc3 = self._fc3
        num_atom_prim = self._pcell.get_number_of_atoms()
        num_atom_super = self._scell.get_number_of_atoms()
        p2s = self._pcell.get_primitive_to_supercell_map()
        dPhidu = np.zeros((num_atom_prim, num_atom_super, 3, 3, 3, 3),
                          dtype=float)

        for nu in range(num_atom_prim):
            Y = self._get_Y(nu)
            for pi in range(num_atom_super):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                for m in range(3):
                                    dPhidu[nu, pi, i, j, k, l] = (
                                        fc3[p2s[nu], pi, :, i, j, :] *
                                        Y[:, :, k, l]).sum()
                                    # (Y[:,:,k,l] + Y[:,:,l,k]) / 2).sum() # Symmetrization?

        return dPhidu

    def _get_Y(self, nu):
        P = self._fc2
        X = self._X
        vecs = self._shortest_vectors
        multi = self._multiplicity
        lat = self._pcell.get_cell()
        num_atom_super = self._scell.get_number_of_atoms()
        R = np.array(
            [np.dot(vecs[Npi, nu, :multi[Npi,nu], :].sum(axis=0) /
                    multi[Npi,nu], lat) for Npi in range(num_atom_super)])

        p2s = self._pcell.get_primitive_to_supercell_map()
        s2p = self._pcell.get_supercell_to_primitive_map()
        p2p = self._pcell.get_primitive_to_primitive_map()

        Y = np.zeros((num_atom_super, 3, 3, 3), dtype=float)

        for Mmu in range(num_atom_super):
            for i in range(3):
                Y[Mmu, i, i, :] = R[Mmu, :]
            Y[Mmu] += X[p2p[s2p[Mmu]]]
            
        return Y

    def _get_X(self):
        num_atom_super = self._scell.get_number_of_atoms()
        num_atom_prim = self._pcell.get_number_of_atoms()
        p2s = self._pcell.get_primitive_to_supercell_map()
        lat = self._pcell.get_cell()
        vecs = self._shortest_vectors
        multi = self._multiplicity
        X = np.zeros((num_atom_prim, 3, 3, 3), dtype=float)
        G = self._get_Gamma()
        P = self._fc2

        for mu in range(num_atom_prim):
            for nu in range(num_atom_prim):
                R = np.array(
                    [np.dot(vecs[Npi, nu, :multi[Npi, nu], :].sum(axis=0) /
                            multi[Npi, nu], lat)
                     for Npi in range(num_atom_super)])
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                X[mu, i, j, k] -= G[mu, nu, i, l] * \
                                    np.dot(P[p2s[nu], :, l, j], R[:, k])

        return X

    def _get_Gamma(self):
        num_atom_prim = self._pcell.get_number_of_atoms()
        m = self._pcell.get_masses()
        self._dm.set_dynamical_matrix([0, 0, 0])
        vals, vecs = np.linalg.eigh(self._dm.get_dynamical_matrix().real)
        G = np.zeros((num_atom_prim, num_atom_prim, 3, 3), dtype=float)

        for pi in range(num_atom_prim):
            for mu in range(num_atom_prim):
                for k in range(3):
                    for i in range(3):
                        # Eigenvectors are real.
                        # 3: means optical modes
                        G[pi, mu, k, i] = (
                            1.0 / np.sqrt(m[pi] * m[mu]) *
                            (vecs[pi * 3 + k, 3:] * vecs[mu * 3 + i, 3:] /
                             vals[3:]).sum())
        return G

            
        
