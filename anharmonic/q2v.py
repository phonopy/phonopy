import sys
import os
import numpy as np
import phonopy.structure.spglib as spg
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.mesh import Mesh
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz, PlanckConstant, Kb, THzToCm, EV, AMU, Hbar, THz, Angstrom
from anharmonic.file_IO import write_triplets, write_grid_points, write_amplitudes, write_damping_functions, parse_triplets, parse_grid_points, write_fwhm, write_decay_channels
from anharmonic.r2q import get_fc3_reciprocal
from anharmonic.shortest_distance import get_shortest_vectors

class PhononPhonon:
    def __init__(self, 
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 sigma=0.2,
                 omega_step=0.1,
                 factor=VaspToTHz,
                 freq_factor=1.0,
                 freq_scale=1.0,
                 symprec=1e-5,
                 is_read_triplets=False,
                 r2q_TI_index=None,
                 is_symmetrize_fc3_q=False,
                 is_Peierls=False,
                 verbose=False,
                 is_nosym=False):
    
        self.freq_factor = freq_factor
        self.cutoff_frequency = 1.0
        self.sigma = sigma
        self.omega_step = omega_step
        self.freq_scale = freq_scale
        self.factor = factor
        self.primitive = primitive
        self.mesh = mesh
        self.fc3 = fc3
        self.is_read_triplets = is_read_triplets
        if r2q_TI_index == None or r2q_TI_index > 2 or r2q_TI_index < 0:
            self.r2q_TI_index = 0
        else:
            self.r2q_TI_index = r2q_TI_index
        self.symprec = symprec
        self.verbose = verbose
        self.is_Peierls = is_Peierls
        self.is_symmetrize_fc3_q = is_symmetrize_fc3_q
        self.is_nosym = is_nosym
        self.p2s_map = primitive.get_primitive_to_supercell_map()
        self.s2p_map = primitive.get_supercell_to_primitive_map()
        self.num_atom = primitive.get_number_of_atoms()
        N0 = len(self.s2p_map) / len(self.p2s_map)

        self.unit_conversion = get_unit_conversion_factor(freq_factor, N0)

        self.shortest_vectors, self.multiplicity = \
            get_shortest_vectors(supercell, primitive, symprec)
        self.symmetry = Symmetry(primitive, symprec)

        # set_interaction_strength
        self.amplitude_at_q = None
        self.frequencies_at_q = None
        self.band_indices = None

        # set_triplets_at_q
        self.grid_point = None
        self.triplets_at_q = None
        self.weights_at_q = None

        # Dynamical matrix has to be set by calling self.set_dynamical matrix
        self.dm = None
        self.q_direction = None

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             q_direction=None):
        self.dm = get_dynamical_matrix(fc2,
                                       supercell,
                                       primitive,
                                       nac_params,
                                       self.symprec)
        if not q_direction==None:
            self.q_direction = q_direction

    def get_life_time(self,
                      tmax,
                      tmin,
                      tstep,
                      gamma_option=0,
                      filename = None):
        
        if self.verbose:
            print "---- Life time calculation ----"
            sys.stdout.flush()

        # A set of phonon modes where the fwhm is calculated.
        q = self.grid_points[self.grid_point].astype(float) / self.mesh
        if (not self.q_direction==None) and self.grid_point==0:
            self.dm.set_dynamical_matrix(q, self.q_direction)
        else:
            self.dm.set_dynamical_matrix(q)
        vals = np.linalg.eigvalsh(self.dm.get_dynamical_matrix())
        fs = np.sqrt(abs(vals)) * self.factor * self.freq_factor * self.freq_scale
        omegas = np.array([fs[x] for x in self.band_indices])
        fwhms = []
        temps = []

        
        for t in np.arange(tmin, tmax + float(tstep) / 2, tstep):
            g_sum = 0.0
            temps.append(t)
            for i in range(len(self.band_indices)):
                g = get_gamma(self.amplitude_at_q,
                              omegas,
                              self.weights_at_q,
                              self.frequencies_at_q,
                              i,
                              t,
                              self.sigma,
                              self.freq_factor,
                              gamma_option).sum() / len(omegas) * self.unit_conversion * 2 
                g_sum += g
            fwhms.append(g_sum / len(self.band_indices))

        write_fwhm(self.grid_point,
                   self.band_indices + 1,
                   temps,
                   fwhms,
                   self.mesh,
                   is_nosym=self.is_nosym,
                   filename=filename)

        return fwhms, temps, omegas


    def get_damping_function(self,
                             temperature = None,
                             filename = None,
                             gamma_option = 0):
        """
        Units of inputs are supposed:
          energy: eV
          distance: Angstrom
          frequency: THz
        """

        omegas = get_omegas(np.max(self.frequencies_at_q),
                            self.omega_step,
                            self.sigma)

        # Calculate damping function at each band
        damps_bands = []
        for i, band_index in enumerate(self.band_indices):
            if (self.grid_point == 0 and band_index < 3) or \
                    band_index < 0 or band_index > self.num_atom * 3 - 1:
                print "The band index %d can not be calculated." % band_index
                continue

            # Unit: frequency^{-1} 
            #   frequency THz
            # 18\pi / \hbar^2 to be multiplied
            dampings = get_gamma(self.amplitude_at_q,
                                 omegas,
                                 self.weights_at_q,
                                 self.frequencies_at_q,
                                 i,
                                 temperature,
                                 self.sigma,
                                 self.freq_factor,
                                 gamma_option) * self.unit_conversion

            write_damping_functions(self.grid_point,
                                    band_index + 1,
                                    self.mesh,
                                    omegas,
                                    dampings,
                                    filename=filename,
                                    is_nosym=self.is_nosym)

    def get_decay_channels(self, temperature):
        if self.verbose:
            print "---- Decay channels ----"
            if temperature==None:
                print "Temperature: 0K"
            else:
                print "Temperature: %10.3fK" % temperature
            sys.stdout.flush()

        q = self.grid_points[self.grid_point].astype(float) / self.mesh 
        if (not self.q_direction==None) and self.grid_point==0:
            self.dm.set_dynamical_matrix(q, self.q_direction)
        else:
            self.dm.set_dynamical_matrix(q)
        vals = np.linalg.eigvalsh(self.dm.get_dynamical_matrix())
        fs = np.sqrt(abs(vals)) * self.factor * self.freq_factor * self.freq_scale
        omegas = np.array([fs[x] for x in self.band_indices])

        if temperature==None:
            t = -1
        else:
            t = temperature

        import anharmonic._phono3py as phono3c

        decay_channels = np.zeros((len(self.weights_at_q),
                                   self.num_atom*3,
                                   self.num_atom*3), dtype=float)

        phono3c.decay_channel(decay_channels,
                              self.amplitude_at_q,
                              self.frequencies_at_q,
                              omegas,
                              self.freq_factor,
                              float(t),
                              float(self.sigma))

        decay_channels *= self.unit_conversion / np.sum(self.weights_at_q) / len(omegas)
        filename = write_decay_channels(decay_channels,
                                        self.amplitude_at_q,
                                        self.frequencies_at_q,
                                        self.triplets_at_q,
                                        self.weights_at_q,
                                        self.grid_points,
                                        self.mesh,
                                        self.band_indices,
                                        omegas,
                                        self.grid_point,
                                        is_nosym=self.is_nosym)

        decay_channels_sum = \
            np.array([d.sum() * w for d, w in
                      zip(decay_channels, self.weights_at_q)]).sum()
        if self.verbose:
            print "FWHM: %f" % (decay_channels_sum * 2)
            print "Decay channels are written into %s." % filename


    def set_triplets_at_q(self, gp):
        mesh = self.mesh

        if self.verbose:
            print "----- Triplets -----"

        # Determine triplets with a specific q-point among ir-triplets
        if self.is_nosym:
            if self.verbose:
                print "Triplets at q without considering symmetry"
                sys.stdout.flush()
            
            triplets_at_q, weights_at_q, self.grid_points = get_nosym_triplets(mesh, gp)

        elif self.is_read_triplets:
            if self.verbose:
                print "Reading ir-triplets at %d" % gp
                sys.stdout.flush()

            self.grid_points = parse_grid_points("grids-%d%d%d.dat" % tuple(mesh))
            triplets_at_q, weights_at_q = parse_triplets(
                "triplets_q-%d%d%d-%d.dat" % (mesh[0], mesh[1], mesh[2], gp ))
        else:
            if self.verbose:
                print "Finding ir-triplets at %d" % gp
                sys.stdout.flush()
            
            triplets_at_q, weights_at_q, self.grid_points = \
                get_triplets_at_q(gp,
                                  mesh,
                                  self.primitive.get_cell(),
                                  self.symmetry.get_pointgroup_operations(),
                                  True,
                                  self.symprec)

            t_filename = "triplets_q-%d%d%d-%d.dat" % (mesh[0], mesh[1], mesh[2], gp)
            write_triplets(triplets_at_q, weights_at_q, mesh, t_filename)
            g_filename = "grids-%d%d%d.dat" % tuple(mesh)
            write_grid_points(self.grid_points, mesh, g_filename)
            if self.verbose:
                print "Ir-triplets at %d were written into %s." % (gp, t_filename)
                print "Mesh points were written into %s." % (g_filename)
                sys.stdout.flush()

        if self.verbose:
            print "Grid point (%d):" % gp,  self.grid_points[gp]
            print "Number of ir triplets:", (len(weights_at_q))
            print "Sum of weights:", weights_at_q.sum()
            sys.stdout.flush()

        self.triplets_at_q = triplets_at_q
        self.weights_at_q = weights_at_q
        self.grid_point = gp

    def set_interaction_strength(self, band_indices=None):
        if self.verbose:
            print "----- phonon-phonon interaction strength ------"

        if band_indices == None:
            self.band_indices = np.arange(self.num_atom * 3, dtype=int)
        else:
            self.band_indices = np.array(band_indices) - 1

        if self.verbose:
            print "Band indices: ", self.band_indices+1

        
        # \Phi^2(set_of_q's, s, s', s'')
        # Unit: mass^{-3} \Phi^2(real space)
        #   \Phi(real space) = eV/A^3
        #   frequency THz
        #   mass AMU
        # 1/36 * (\hbar/2N0)^3 * N0^2 to be multiplied somewhere else.
        self.amplitude_at_q = np.zeros((len(self.weights_at_q),
                                        len(self.band_indices),
                                        self.num_atom*3,
                                        self.num_atom*3), dtype=float)
        self.frequencies_at_q = np.zeros((len(self.weights_at_q),
                                          3,
                                          self.num_atom*3), dtype=float)

        for i, (q3, w) in enumerate(zip(self.triplets_at_q, self.weights_at_q)):
            self.amplitude_at_q[i], self.frequencies_at_q[i] = \
                get_interaction_strength(i,
                                         len(self.triplets_at_q),
                                         q3,
                                         w,
                                         self.mesh,
                                         self.grid_points,
                                         self.shortest_vectors,
                                         self.multiplicity,
                                         self.fc3,
                                         self.dm,
                                         self.q_direction,
                                         self.primitive,
                                         self.band_indices,
                                         self.factor,
                                         self.freq_factor,
                                         self.freq_scale,
                                         self.cutoff_frequency,
                                         self.symprec,
                                         self.is_symmetrize_fc3_q,
                                         self.is_Peierls,
                                         self.r2q_TI_index,
                                         self.verbose)

def get_interaction_strength(triplet_number,
                             num_triplets,
                             q3,
                             w,
                             mesh,
                             grid_points,
                             shortest_vectors,
                             multiplicity,
                             fc3,
                             dm,
                             q_direction,
                             primitive,
                             band_indices,
                             factor,
                             freq_factor,
                             freq_scale,
                             cutoff_frequency,
                             symprec, 
                             is_symmetrize_fc3_q,
                             is_Peierls,
                             r2q_TI_index,
                             verbose):

    q_set = []
    for q in q3:
        q_set.append(grid_points[q].astype(float) / mesh)
    q_set = np.array(q_set)

    show_interaction_strength_progress(verbose,
                                       triplet_number,
                                       num_triplets,
                                       w,
                                       q_set)

    num_atom = primitive.get_number_of_atoms()

    # Solve dynamical matrix
    freqs = np.zeros((3, num_atom*3), dtype=float)
    eigvecs = np.zeros((3, num_atom*3, num_atom*3), dtype=complex)
    for i, q in enumerate(q_set):
        if (not q_direction==None) and q3[i]==0:
            dm.set_dynamical_matrix(q, q_direction)
        else:
            dm.set_dynamical_matrix(q)
        # dm.set_dynamical_matrix(q)
        val, eigvecs[i] = np.linalg.eigh(dm.get_dynamical_matrix())
        freqs[i] = np.sqrt(np.abs(val)) * factor * freq_factor * freq_scale

    # Calculate interaction strength
    amplitude = np.zeros((len(band_indices), num_atom*3, num_atom*3), dtype=float)
    try:
        import anharmonic._phono3py as phono3c
        get_c_interaction_strength(amplitude,
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
        get_py_interaction_strength(amplitude,
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

    return amplitude, freqs

def show_interaction_strength_progress(verbose,
                                       triplet_number,
                                       num_triplets,
                                       w,
                                       q_set):
    if verbose:
        if int(verbose) > 1:
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


    

def get_c_interaction_strength(amplitude,
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

    phono3c.interaction_strength(amplitude,
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

def get_py_interaction_strength(amplitude,
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

def get_gamma(amplitudes,
              omegas,
              weights,
              frequencies,
              band_index,
              temperature,
              sigma,
              freq_factor,
              gamma_option=0):
    
    gammas = np.zeros(len(omegas), dtype=float)
    if temperature==None:
        t = -1 # Means 0 K
    else:
        t = temperature

    try:
        import anharmonic._phono3py as phono3c
        phono3c.gamma(gammas,
                      omegas,
                      amplitudes,
                      weights,
                      frequencies,
                      band_index,
                      float(sigma),
                      freq_factor,
                      float(t),
                      gamma_option)

    except ImportError:
        get_py_gamma(gammas,
                     omegas,
                     amplitudes,
                     weights,
                     frequencies,
                     band_index,
                     sigma,
                     freq_factor,
                     t)

    return gammas / weights.sum()
               

def get_py_gamma(gammas,
                 omegas,
                 amplitudes,
                 weights,
                 frequencies,
                 band_index,
                 sigma,
                 freq_factor,
                 t):

    num_band = frequencies.shape[2]
    sum_ir_triplet = np.zeros((len(weights), len(omegas)), dtype=float)

    for i, omega in enumerate(omegas):
        for l, (a, w, freqs) in enumerate(zip(amplitudes, weights, frequencies)):
            sum_local = 0.0
            for j in range(num_band):
                for k in range(num_band):
                    f2 = freqs[1, j]
                    f3 = freqs[2, k]
                    vv = a[band_index, j, k]

                    if t > 0:
                        n2 = bs(f2 / freq_factor, t)
                        n3 = bs(f3 / freq_factor, t)
                        sum_local += ((1 + n2 + n3) * gauss(f2 + f3 - omega, sigma) +
                                      2 * (n3 - n2) * gauss(f2 - f3 - omega, sigma)
                                      ) * vv * w
                    else:
                        sum_local += gauss(f2 + f3 - omega, sigma) * vv * w

            sum_ir_triplet[l, i] = sum_local
            gammas[i] += sum_local


def get_sum_in_primitive(fc3_q, e1, e2, e3, primitive):
    
    try:
        import anharmonic._phono3py as phono3c
        return get_c_sum_in_primitive(fc3_q, e1.copy(), e2.copy(), e3.copy(), primitive)
               
    except ImportError:
        return get_py_sum_in_primitive(fc3_q, e1, e2, e3, primitive)
               


def get_c_sum_in_primitive(fc3_q, e1, e2, e3, primitive):
    import anharmonic._phono3py as phono3c
    return phono3c.sum_in_primitive(fc3_q,
                                    e1, e2, e3,
                                    primitive.get_masses())

def get_py_sum_in_primitive(fc3_q, e1, e2, e3, primitive):

    num_atom = primitive.get_number_of_atoms()
    m = primitive.get_masses()
    sum = 0

    for i1 in range(num_atom):
        for i2 in range(num_atom):
            for i3 in range(num_atom):
                sum += get_sum_in_cartesian(fc3_q[i1, i2, i3],
                                            e1[i1*3:(i1+1)*3],
                                            e2[i2*3:(i2+1)*3],
                                            e3[i3*3:(i3+1)*3]) \
                                            / np.sqrt(m[i1] * m[i2] * m[i3])
                
    return abs(sum)**2



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
        

def gauss(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2.0 / sigma**2)

def bs(x, t): # Bose Einstein distribution (For frequency THz)
    return 1.0 / (np.exp(PlanckConstant * 1e12 * x / (Kb * t)) - 1)
    
def get_grid_points(mesh):
    grid_points = np.zeros((np.prod(mesh), 3), dtype=int)
    count = 0
    for i in range(mesh[0]):
        for j in range(mesh[1]):
            for k in range(mesh[2]):
                grid_points[count] = [k - (k > (mesh[2] // 2)) * mesh[2],
                                      j - (j > (mesh[1] // 2)) * mesh[1],
                                      i - (i > (mesh[0] // 2)) * mesh[0]]
                count += 1
    
    return grid_points

def get_nosym_triplets(mesh, grid_point0):
    grid_points = get_grid_points(mesh)
    triplets = np.zeros((len(grid_points), 3), dtype=int)
    for i, g1 in enumerate(grid_points):
        g2 = - (grid_points[grid_point0] + g1)
        triplets[i] = [grid_point0, i, get_address(g2, mesh)]
    weights = np.ones(len(grid_points), dtype=int)

    return triplets, weights, grid_points

def get_address(grid, mesh):
    return ((grid[0] + mesh[0]) % mesh[0] +
            ((grid[1] + mesh[1]) % mesh[1]) * mesh[0] +
            ((grid[2] + mesh[2]) % mesh[2]) * mesh[0] * mesh[1])

def get_triplets_at_q(gp,
                      mesh,
                      primitive_lattice,
                      rotations,
                      is_time_reversal=True,
                      symprec=1e-5):

    weights, third_q, grid_points = \
        spg.get_triplets_reciprocal_mesh_at_q(gp,
                                              mesh,
                                              primitive_lattice,
                                              rotations,
                                              is_time_reversal,
                                              symprec)

    weights_at_q = []
    triplets_at_q = []
    for i, (w, q) in enumerate(zip(weights, third_q)):
        if w > 0:
            weights_at_q.append(w)
            triplets_at_q.append([gp, i, q])

    return np.array(triplets_at_q), np.array(weights_at_q), grid_points


def get_dynamical_matrix(fc2,
                         supercell,
                         primitive,
                         nac_params=None,
                         symprec=1e-5):
    if nac_params==None:
        dm = DynamicalMatrix(supercell,
                             primitive,
                             fc2,
                             symprec=symprec)
    else:
        dm = DynamicalMatrixNAC(supercell,
                                primitive,
                                fc2,
                                nac_params=nac_params,
                                symprec=symprec)
    return dm
        
def get_unit_conversion_factor(freq_factor, N0):
    """
    The first two lines are used to convert to SI unit
    and the third line gives conversion to frequncy unit
    specified by freq_factor.
    """
    return  1.0 / 36 / 8 * (Hbar * EV)**3 / N0 / ((2 * np.pi * THz / freq_factor)**3) / AMU**3 * (EV / Angstrom**3) ** 2 \
        * 18 * np.pi / (2 * np.pi * THz / freq_factor) / ((Hbar * EV) **2 ) * N0 / (2 * np.pi) \
        / THz * freq_factor
        
def get_omegas(max_omega, omega_step, sigma):
    return np.array(range(int((max_omega * 2 + sigma * 4) / omega_step + 1)),
                    dtype=float) * omega_step
    

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
    
    
    
