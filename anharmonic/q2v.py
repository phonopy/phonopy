import sys
import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC, get_smallest_vectors
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz, PlanckConstant, Kb, THzToCm, EV, AMU, Hbar, THz, Angstrom
from anharmonic.file_IO import write_triplets, write_grid_address, write_amplitudes, parse_triplets, parse_grid_address
from anharmonic.triplets import get_triplets_at_q, get_nosym_triplets
from anharmonic.r2q import get_fc3_reciprocal
from anharmonic.shortest_distance import get_shortest_vectors

def print_log(text):
    sys.stdout.write(text)
    sys.stdout.flush()

class PhononPhonon:
    """
    This code expects phonon frequecies to be in THz unit.
    To make frequencies in THz, 'factor' is used.
    'freq_factor' is used to output in different unit, e.g., cm-1.
    'freq_scale' is multiplied to adjust frequencies,
       e.g., to correct underestimation of frequencies by GGA.
    """
    def __init__(self, 
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 factor=VaspToTHz,
                 freq_factor=1.0, # Convert from THz to another (e.g., cm-1)
                 symprec=1e-5,
                 is_read_triplets=False,
                 r2q_TI_index=None,
                 is_symmetrize_fc3_q=False,
                 is_Peierls=False,
                 log_level=False,
                 is_nosym=False):
    
        self._freq_factor = freq_factor
        self._cutoff_frequency = 0.01 * self._freq_factor
        self._factor = factor
        self._primitive = primitive
        self._mesh = np.array(mesh)
        self._fc3 = fc3
        self._is_read_triplets = is_read_triplets
        if r2q_TI_index == None or r2q_TI_index > 2 or r2q_TI_index < 0:
            self._r2q_TI_index = 0
        else:
            self._r2q_TI_index = r2q_TI_index
        self._symprec = symprec
        self._log_level = log_level
        self._is_Peierls = is_Peierls
        self._is_symmetrize_fc3_q = is_symmetrize_fc3_q
        self._is_nosym = is_nosym
        self._p2s_map = primitive.get_primitive_to_supercell_map()
        self._s2p_map = primitive.get_supercell_to_primitive_map()
        self._conversion_factor = get_unit_conversion_factor(freq_factor)

        # self._shortest_vectors, self._multiplicity = get_shortest_vectors(
        #     supercell, primitive, symprec)
        self._shortest_vectors, self._multiplicity = get_smallest_vectors(
            supercell, primitive, symprec)
        self._symmetry = Symmetry(primitive, symprec)

        # set_interaction_strength
        self._amplitude_at_q = None
        self._frequencies_at_q = None
        self._band_indices = None

        # set_triplets_at_q
        self._grid_point = None
        self._grid_address = None
        self._triplets_at_q = None
        self._weights_at_q = None

        # Dynamical matrix has to be set by calling self._set_dynamical matrix
        self._dm = None
        self._q_direction = None

        # Frequencies and eigenvectors at the grid point
        self._frequencies = None
        self._eigenvectors = None

    def get_amplitude(self):
        return (self._amplitude_at_q,
                self._weights_at_q,
                self._frequencies_at_q)

    def get_band_indices(self):
        return self._band_indices

    def get_cutoff_frequency(self):
        return self._cutoff_frequency
    
    def get_dynamical_matrix(self):
        return self._dm

    def get_frequencies(self):
        return self._frequencies

    def get_eigenvectors(self):
        return self._eigenvectors
    
    def get_frequency_factor_to_THz(self):
        return self._factor

    def get_frequency_unit_conversion_factor(self):
        return self._freq_factor

    def get_grid_point(self):
        return self._grid_point

    def get_grid_address(self):
        return self._grid_address

    def get_mesh_numbers(self):
        return self._mesh

    def get_primitive(self):
        return self._primitive

    def get_qpoint(self):
        return self._q

    def get_q_direction(self):
        return self._q_direction

    def get_symmetry(self):
        return self._symmetry

    def get_triplets_at_q(self):
        return self._triplets_at_q
    
    def get_unit_conversion_factor(self):
        return self._conversion_factor

    def is_nosym(self):
        return self._is_nosym
    
    def set_triplets_at_q(self, gp):
        self.print_log("----- Triplets -----\n")

        mesh = self._mesh
        if self._is_nosym:
            self.print_log("Triplets at q without considering symmetry\n")
            (triplets_at_q,
             weights_at_q,
             self._grid_address) = get_nosym_triplets(mesh, gp)
        elif self._is_read_triplets:
            self.print_log("Reading ir-triplets at %d\n" % gp)
            self._grid_address = parse_grid_address(
                "grids-%d%d%d.dat" % tuple(mesh))
            triplets_at_q, weights_at_q = parse_triplets(
                "triplets_q-%d%d%d-%d.dat" % (tuple(mesh) + (gp,)))
        else:
            self.print_log("Finding ir-triplets at %d\n" % gp)
            
            (triplets_at_q,
             weights_at_q,
             self._grid_address) = get_triplets_at_q(
                gp,
                mesh,
                self._primitive.get_cell(),
                self._symmetry.get_pointgroup_operations(),
                is_time_reversal=True)

            if self._log_level > 1:
                t_filename = "triplets_q-%d%d%d-%d.dat" % (tuple(mesh) + (gp,))
                write_triplets(triplets_at_q, weights_at_q, mesh, t_filename)
                self.print_log("Ir-triplets at %d were written into %s.\n" %
                               (gp, t_filename))

            if self._log_level:
                g_filename = "grids-%d%d%d.dat" % tuple(mesh)
                write_grid_address(self._grid_address, mesh, g_filename)
            self.print_log("Mesh points were written into %s.\n" % g_filename)
            self.print_log("Grid point (%d): " % gp)
            self.print_log("[ %d %d %d ]\n" % tuple(self._grid_address[gp]))
            self.print_log("Number of ir triplets: %d\n" % len(weights_at_q))

        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._q = self._grid_address[gp].astype(float) / mesh
        self._grid_point = gp
        
    def set_interaction_strength(self, band_indices=None):
        self.print_log("----- phonon-phonon interaction strength ------\n")

        num_atom = self._primitive.get_number_of_atoms()

        if band_indices == None:
            self._band_indices = np.arange(num_atom * 3, dtype=int)
        else:
            self._band_indices = np.array(band_indices)

        self.print_log(("Band indices: [" + " %d" * len(self._band_indices) +
                         " ]\n") % tuple(self._band_indices + 1))

        self.set_harmonic_phonons(self._q)
        
        self._amplitude_at_q = np.zeros((len(self._weights_at_q),
                                         len(self._band_indices),
                                         num_atom * 3,
                                         num_atom * 3), dtype=float)
        self._frequencies_at_q = np.zeros(
            (len(self._weights_at_q), 3, num_atom * 3), dtype=float)

        for i, (q3, w) in enumerate(zip(self._triplets_at_q,
                                        self._weights_at_q)):
            (self._amplitude_at_q[i],
             self._frequencies_at_q[i]) = get_triplet_interaction_strength(
                i,
                len(self._triplets_at_q),
                q3,
                w,
                self._mesh,
                self._grid_address,
                self._shortest_vectors,
                self._multiplicity,
                self._fc3,
                self._dm,
                self._q_direction,
                self._frequencies,
                self._eigenvectors,
                self._primitive,
                self._band_indices,
                self._factor,
                self._freq_factor,
                self._cutoff_frequency,
                self._symprec,
                self._is_symmetrize_fc3_q,
                self._is_Peierls,
                self._r2q_TI_index,
                self._log_level)

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None,
                             frequency_scale_factor=None):
        if nac_params==None:
            self._dm = DynamicalMatrix(
                supercell,
                primitive,
                fc2,
                frequency_scale_factor=frequency_scale_factor,
                symprec=self._symprec)
        else:
            self._dm = DynamicalMatrixNAC(
                supercell,
                primitive,
                fc2,
                frequency_scale_factor=frequency_scale_factor,
                symprec=self._symprec)
            self._dm.set_nac_params(nac_params)

        if nac_q_direction is not None:
            self._q_direction = nac_q_direction

    def set_harmonic_phonons(self, q):
        if ((self._q_direction is not None) and
            (q < 0.1 / max(self._mesh)).all()):
            self._dm.set_dynamical_matrix(q, self._q_direction)
        else:
            self._dm.set_dynamical_matrix(q)
        vals, self._eigenvectors = np.linalg.eigh(
            self._dm.get_dynamical_matrix())
        vals = vals.real
        factor = self._factor * self._freq_factor
        self._frequencies = np.sqrt(abs(vals)) * np.sign(vals) * factor
            
    def print_log(self, text):
        if self._log_level:
            print_log(text)

def get_triplet_interaction_strength(triplet_number,
                                     num_triplets,
                                     q3,
                                     w,
                                     mesh,
                                     grid_address,
                                     shortest_vectors,
                                     multiplicity,
                                     fc3,
                                     dm,
                                     q_direction,
                                     frequencies,
                                     eigenvectors,
                                     primitive,
                                     band_indices,
                                     factor,
                                     freq_factor,
                                     cutoff_frequency,
                                     symprec, 
                                     is_symmetrize_fc3_q,
                                     is_Peierls,
                                     r2q_TI_index,
                                     log_level):
    q_set = []
    for q in q3:
        q_set.append(grid_address[q].astype(float) / mesh)
    q_set = np.array(q_set)

    show_interaction_strength_progress(log_level,
                                       triplet_number,
                                       num_triplets,
                                       w,
                                       q_set)

    num_atom = primitive.get_number_of_atoms()

    # Solve dynamical matrix
    freqs = np.zeros((3, num_atom * 3), dtype=float)
    freqs[0] = frequencies.copy()
    eigvecs = np.zeros((3, num_atom * 3, num_atom * 3), dtype=complex)
    eigvecs[0] = eigenvectors.copy()
    for i, q in enumerate(q_set[1:]):
        if (q_direction is not None) and q3[i + 1] == 0:
            dm.set_dynamical_matrix(q, q_direction)
        else:
            dm.set_dynamical_matrix(q)

        vals, eigvecs[i + 1] = np.linalg.eigh(dm.get_dynamical_matrix())
        vals = vals.real
        total_factor = factor * freq_factor
        freqs[i + 1] = np.sqrt(np.abs(vals)) * np.sign(vals) * total_factor

    # Calculate interaction strength
    amplitude = np.zeros((len(band_indices), num_atom * 3, num_atom * 3),
                         dtype=float)

    try:
        import anharmonic._phono3py as phono3c
        get_c_triplet_interaction_strength(amplitude,
                                           freqs,
                                           eigvecs,
                                           shortest_vectors,
                                           multiplicity,
                                           q_set,
                                           fc3,
                                           primitive,
                                           band_indices,
                                           cutoff_frequency,
                                           is_symmetrize_fc3_q,
                                           r2q_TI_index,
                                           symprec)
    except ImportError:
        get_py_triplet_interaction_strength(amplitude,
                                            freqs,
                                            eigvecs,
                                            shortest_vectors,
                                            multiplicity,
                                            q_set,
                                            fc3,
                                            primitive,
                                            band_indices,
                                            cutoff_frequency,
                                            is_symmetrize_fc3_q,
                                            is_Peierls,
                                            r2q_TI_index,
                                            symprec)

    return amplitude / np.prod(mesh), freqs

def show_interaction_strength_progress(log_level,
                                       triplet_number,
                                       num_triplets,
                                       w,
                                       q_set):
    if log_level:
        if int(log_level) > 1:
            print "%d/%d: Weight %d" % (triplet_number+1, num_triplets, w)
            for q in q_set:
                print "     " + ("%7.4f " * 3) % tuple(q)
            print "     -----------------------"
            print "Sum: " + ("%7.4f " * 3) % tuple(q_set.sum(axis=0))
            print ""
            sys.stdout.flush()
        else:
            progress = ((triplet_number + 1) * 20) / num_triplets
            last_progress = (triplet_number * 20) / num_triplets

            if not progress == last_progress:
                print "> %d%%" % (progress * 5)
                last_progress = progress
            else:
                progress = ((triplet_number + 1) * 100) / num_triplets
                last_progress = (triplet_number * 100) / num_triplets
                if not progress == last_progress:
                    sys.stdout.write("^")
                else:
                    progress = ((triplet_number + 1) * 1000) / num_triplets
                    last_progress = (triplet_number * 1000) / num_triplets
                    if not progress == last_progress:
                        sys.stdout.write("-")
            sys.stdout.flush()


    

def get_c_triplet_interaction_strength(amplitude,
                                       freqs,
                                       eigvecs,
                                       shortest_vectors,
                                       multiplicity,
                                       q_set,
                                       fc3,
                                       primitive,
                                       band_indices,
                                       cutoff_frequency,
                                       is_symmetrize_fc3_q,
                                       r2q_TI_index,
                                       symprec):
    
    import anharmonic._phono3py as phono3c
    p2s_map = primitive.get_primitive_to_supercell_map()
    s2p_map = primitive.get_supercell_to_primitive_map()

    phono3c.triplet_interaction_strength(amplitude,
                                         freqs,
                                         eigvecs,
                                         shortest_vectors,
                                         multiplicity,
                                         q_set,
                                         np.array(p2s_map),
                                         np.array(s2p_map),
                                         fc3,
                                         primitive.get_masses(),
                                         band_indices,
                                         cutoff_frequency,
                                         is_symmetrize_fc3_q * 1,
                                         r2q_TI_index,
                                         symprec)

def get_py_triplet_interaction_strength(amplitude,
                                        freqs,
                                        eigvecs,
                                        shortest_vectors,
                                        multiplicity,
                                        q_set,
                                        fc3,
                                        primitive,
                                        band_indices,
                                        cutoff_frequency,
                                        is_symmetrize_fc3_q,
                                        is_Peierls,
                                        r2q_TI_index,
                                        symprec):

    # fc3 from real space to reciprocal space
    #
    # \sum_{NP} \Phi(M\mu,N\nu,P\pi) e^{i\mathbf{q}'\cdot
    # [\mathbf{R}(N\nu)-\mathbf{R}(M\mu)]}
    # e^{i\mathbf{q}''\cdot [\mathbf{R}(P\pi)-\mathbf{R}(M\mu)]}

    num_atom = primitive.get_number_of_atoms()
    p2s_map = primitive.get_primitive_to_supercell_map()
    s2p_map = primitive.get_supercell_to_primitive_map()

    if (not is_symmetrize_fc3_q) and (not is_Peierls):
        fc3_q = get_fc3_reciprocal(shortest_vectors,
                                   multiplicity,
                                   q_set,
                                   p2s_map,
                                   s2p_map,
                                   fc3,
                                   symprec=symprec,
                                   r2q_TI_index=r2q_TI_index)
    elif is_symmetrize_fc3_q:
        index_exchange = ((0, 1, 2),
                          (2, 0, 1),
                          (1, 2, 0),
                          (2, 1, 0),
                          (0, 2, 1),
                          (1, 0, 2))
        fc3_q = []
        for k in index_exchange:
            fc3_q.append(
                get_fc3_reciprocal(shortest_vectors,
                                   multiplicity,
                                   np.array([q_set[k[0]],
                                             q_set[k[1]],
                                             q_set[k[2]]]),
                                   p2s_map,
                                   s2p_map,
                                   fc3,
                                   symprec=symprec,
                                   r2q_TI_index=r2q_TI_index))
    # e[q, eigvec, eigvec_index]
    # e[3, num_atom*3, num_atom*3]
    e = eigvecs

    # \sum_{\mu\nu\pi}
    j = [0, 0, 0]
    for i, j[0] in enumerate(band_indices):
        for j[1] in range(num_atom * 3):
            for j[2] in range(num_atom * 3):
                if (freqs[0][j[0]] < cutoff_frequency or
                    freqs[1][j[1]] < cutoff_frequency or
                    freqs[2][j[2]] < cutoff_frequency):
                    continue

                if is_Peierls:
                    vv = 1.0 / np.prod(mesh)
                    vv /= freqs[0][j[0]] * freqs[1][j[1]] * freqs[2][j[2]]
                else:
                    if not is_symmetrize_fc3_q:
                        vv = get_sum_in_primitive(fc3_q,
                                                  e[0,:, j[0]], # q
                                                  e[1,:, j[1]], # q'
                                                  e[2,:, j[2]], # q''
                                                  primitive)
                        vv /= freqs[0][j[0]] * freqs[1][j[1]] * freqs[2][j[2]]
                    else:
                        vv = 0.0
                        for l, k in enumerate(index_exchange):
                            vv += get_sum_in_primitive(
                                fc3_q[l],
                                e[k[0],:, j[k[0]]], # q
                                e[k[1],:, j[k[1]]], # q'
                                e[k[2],:, j[k[2]]], # q''
                                primitive)
                        vv /= 6 * freqs[0][j[0]] * freqs[1][j[1]] * freqs[2][j[2]]
                amplitude[i, j[1], j[2]] = vv

def get_sum_in_primitive(fc3_q, e1, e2, e3, primitive):
    
    try:
        import anharmonic._phono3py as phono3c
        return get_c_sum_in_primitive(fc3_q,
                                      e1.copy(),
                                      e2.copy(),
                                      e3.copy(),
                                      primitive)
               
    except ImportError:
        return get_py_sum_in_primitive(fc3_q, e1, e2, e3, primitive)
               


def get_c_sum_in_primitive(fc3_q, e1, e2, e3, primitive):
    import anharmonic._phono3py as phono3c
    return phono3c.sum_in_primitive(fc3_q,
                                    e1,
                                    e2,
                                    e3,
                                    primitive.get_masses())

def get_py_sum_in_primitive(fc3_q, e1, e2, e3, primitive):

    num_atom = primitive.get_number_of_atoms()
    m = primitive.get_masses()
    sum = 0

    for i1 in range(num_atom):
        for i2 in range(num_atom):
            for i3 in range(num_atom):
                sum += get_sum_in_cartesian(
                    fc3_q[i1, i2, i3],
                    e1[i1*3:(i1 + 1)*3],
                    e2[i2*3:(i2 + 1)*3],
                    e3[i3*3:(i3 + 1)*3]) / np.sqrt(m[i1] * m[i2] * m[i3])
                
    return abs(sum) ** 2



def get_sum_in_cartesian(f3, e1, e2, e3):
    """
    i1, i2, i3 are the atom indices.
    """
    sum = 0
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            for c in (0, 1, 2):
                sum += f3[a, b, c] * e1[a] * e2[b] * e3[c]

    return sum

def get_unit_conversion_factor(freq_factor):
    """
    Input:
      Frequency: THz * freq_factor
      Mass: AMU
      Force constants: eV/A^3

    """
    # omega => 2pi * freq * THz
    # Frequency unit to angular frequency (rad/s)
    # Mass unit to kg
    # Force constants to J/m^3
    # hbar to J s
    unit_in_angular_freq = np.pi * (Hbar * EV) / 16 * EV ** 2 / Angstrom ** 6 / (2 * np.pi * (THz / freq_factor)) ** 4 / AMU ** 3

    # Frequency unit in some favorite one
    return unit_in_angular_freq / (2 * np.pi) / THz * freq_factor

        
#
#  Functions used for debug
#    
def print_smallest_vectors(smallest_vectors, multiplicity, p2s_map):
    
    print 
    print "Shortest distances"
    nums = multiplicity.shape
    for i in range(nums[0]):
        for j in range(nums[1]):
            for k in range(multiplicity[i, j]):
                print "(%3d -%3d)   %5.2f %5.2f %5.2f" % (
                    i + 1,
                    p2s_map[j] + 1,
                    smallest_vectors[i, j, k][0],
                    smallest_vectors[i, j, k][1],
                    smallest_vectors[i, j, k][2])
        
def print_eigenvectors(dm, q):

    from cmath import phase
    dm.set_dynamical_matrix([0.1, 0.3, 0.9])
    val, vec = np.linalg.eigh(dm.get_dynamical_matrix())

    for i in range(6):
        print "band ", i+1
        for v in vec[:, i]:
            varg = phase(v)-phase(vec[0, i])
            if varg < -np.pi:
                varg += np.pi
            if varg > np.pi:
                varg -= np.pi
        
            print "%10.5f %10.5f %10.5f %10.5f" % (v.real, v.imag, abs(v), varg)
        
    
def print_fc3_q(fc3, q_set, shortest_vectors, multiplicity, primitive, symprec=1e-5):
    
    p2s_map = primitive.get_primitive_to_supercell_map()
    s2p_map = primitive.get_supercell_to_primitive_map()

    fc3_q = get_fc3_reciprocal(shortest_vectors, multiplicity,
                               q_set, p2s_map, s2p_map, fc3, symprec=symprec) 

    num_atom = primitive.get_number_of_atoms()
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                print "%2d -%2d -%2d" % (i+1, j+1, k+1)
                for t2 in fc3_q[i,j,k]:
                    for v in t2:
                        print "%10.5f %10.5f   %10.5f %10.5f   %10.5f %10.5f" % (v[0].real, v[0].imag, v[1].real, v[1].imag, v[2].real, v[2].imag)
                print
