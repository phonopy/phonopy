import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, get_smallest_vectors
from anharmonic.fc_interpolate import get_fc_interpolation
from phonopy.structure.cells import get_supercell, Primitive, print_cell
from anharmonic.file_IO import write_fc3_dat, write_fc2_dat
from anharmonic.fc_tools import expand_fc2, expand_fc3
from phonopy.units import VaspToTHz
from phonopy.phonon.mesh import get_qpoints

class Gruneisen:
    def __init__(self,
                 fc2,
                 fc3,
                 supercell,
                 primitive,
                 is_ion_clamped=False,
                 factor=VaspToTHz,
                 symprec=1e-5):
        self._fc2 = fc2
        self._fc3 = fc3
        self._scell = supercell
        self._pcell = primitive
        self._is_ion_clamped = is_ion_clamped
        self._factor = factor
        self._symprec = symprec
        self._dm = DynamicalMatrix(self._scell,
                                   self._pcell,
                                   self._fc2,
                                   symprec=self._symprec)
        self._shortest_vectors, self._multiplicity = get_smallest_vectors(
            self._scell, self._pcell, self._symprec)

        if self._is_ion_clamped:
            num_atom_prim = self._pcell.get_number_of_atoms()
            self._X = np.zeros((num_atom_prim, 3, 3, 3), dtype=float)
        else:
            self._X = self._get_X()
        self._dPhidu = self._get_dPhidu()

        self._gruneisen_parameters = None
        self._frequencies = None
        self._qpoints = None
        self._mesh = None
        self._weights = None

    def run(self):
        self._gruneisen_parameters = []
        self._frequencies = []
        for q in self._qpoints:
            g, omega2 = self._get_gruneisen_tensor(q)
            self._gruneisen_parameters.append(g)
            self._frequencies.append(
                np.sqrt(abs(omega2)) * np.sign(omega2) * self._factor)

    def get_gruneisen_parameters(self):
        return self._gruneisen_parameters

    def set_qpoints(self, qpoints):
        self._qpoints = qpoints

    def set_sampling_mesh(self,
                          mesh,
                          grid_shift=None,
                          is_gamma_center=False):
        self._mesh = mesh
        self._qpoints, self._weights = get_qpoints(self._mesh,
                                                   self._pcell,
                                                   grid_shift,
                                                   is_gamma_center)

    def write_yaml(self, filename="gruneisen3.yaml"):
        if (self._qpoints is not None and
            self._gruneisen_parameters is not None):
            f = open(filename, 'w')
            if self._mesh is not None:
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
            f.close()
        
    def _get_gruneisen_tensor(self, q):
        self._dm.set_dynamical_matrix(q)
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

            g[s] *= -1.0/2/omega2[s]

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
                        G[pi, mu, k, i] = 1.0 / np.sqrt(m[pi] * m[mu]) * \
                            (vecs[pi * 3 + k, 3:] * vecs[mu * 3 + i, 3:] /
                             vals[3:]).sum()
        return G

            
        
