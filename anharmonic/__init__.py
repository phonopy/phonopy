import sys
from phonopy.harmonic.force_constants import get_force_constants, symmetrize_force_constants
from anharmonic.q2v import PhononPhonon
from anharmonic.jointDOS import get_jointDOS
from anharmonic.gruneisen import Gruneisen

class JointDOS:
    def __init__(self,
                 supercell=None,
                 primitive=None,
                 mesh=None,
                 fc2=None,
                 nac_params=None,
                 sigma=None,
                 omega_step=None,
                 factor=None,
                 freq_factor=None,
                 freq_scale=None,
                 is_nosym=False,
                 symprec=1e-5,
                 log_level=0):
        self.supercell = supercell
        self.primitive = primitive
        self.mesh = mesh
        self.fc2 = fc2
        self.nac_params = nac_params
        self.sigma = sigma
        self.omega_step = omega_step
        self.factor = factor
        self.freq_factor = freq_factor
        self.freq_scale = freq_scale
        self.is_nosym = is_nosym
        self.symprec = symprec
        self.log_level = log_level

    def get_jointDOS(self, grid_points, filename=None):
        get_jointDOS(grid_points,
                     self.mesh,
                     self.primitive,
                     self.supercell,
                     self.fc2,
                     self.nac_params,
                     self.sigma,
                     self.omega_step,
                     self.factor,
                     self.freq_factor,
                     self.freq_scale,
                     self.is_nosym,
                     self.symprec,
                     filename,
                     self.log_level)

class Phono3py:
    def __init__(self,
                 supercell=None,
                 primitive=None,
                 mesh=None,
                 fc3=None,
                 sigma=None,
                 omega_step=None,
                 factor=None,
                 freq_factor=None,
                 freq_scale=None,
                 is_nosym=False,
                 is_symmetrize_fc3_q=False,
                 is_read_triplets=False,
                 r2q_TI_index=None,
                 is_Peierls=False,
                 symprec=1e-5,
                 log_level=0):

        self.supercell = supercell
        self.primitive = primitive
        self.mesh = mesh
        self.fc3 = fc3
        self.sigma = sigma
        self.omega_step = omega_step
        self.factor = factor
        self.freq_factor = freq_factor
        self.freq_scale = freq_scale
        self.is_nosym = is_nosym
        self.is_symmetrize_fc3_q = is_symmetrize_fc3_q
        self.symprec = symprec
        self.log_level = log_level
        self.is_read_triplets = is_read_triplets
        self.r2q_TI_index = r2q_TI_index
        self.is_Peierls = is_Peierls

        # Phonon-phonon
        self.pp = None

    def set_phonon_phonon(self):
        self.pp = PhononPhonon(self.fc3,
                               self.supercell,
                               self.primitive,
                               self.mesh,
                               self.sigma,
                               self.omega_step,
                               self.factor,
                               self.freq_factor,
                               self.freq_scale,
                               self.symprec,
                               self.is_read_triplets,
                               self.r2q_TI_index,
                               self.is_symmetrize_fc3_q,
                               self.is_Peierls,
                               self.log_level,
                               self.is_nosym)

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None):
        self.pp.set_dynamical_matrix(fc2,
                                     supercell,
                                     primitive,
                                     nac_params,
                                     nac_q_direction)
                           
    def get_damping_function(self,
                             grid_points=None,
                             sets_of_band_indices=None,
                             temperature=None,
                             filename=None,
                             gamma_option=0):

        if grid_points==None:
            print "Grid points are not specified."
            return False

        
        for gp in grid_points:
            self.pp.set_triplets_at_q(gp)

            if sets_of_band_indices == None:
                if gp==0:
                    self.pp.set_interaction_strength(
                        range(4, self.primitive.get_number_of_atoms() * 3 + 1))
                else:
                    self.pp.set_interaction_strength(
                        range(1, self.primitive.get_number_of_atoms() * 3 + 1))

                self.pp.get_damping_function(temperature=temperature,
                                             filename=filename,
                                             gamma_option=gamma_option)
            else:
                for band_indices in sets_of_band_indices:
                    self.pp.set_interaction_strength(band_indices)
                    self.pp.get_damping_function(temperature=temperature,
                                                 filename=filename,
                                                 gamma_option=gamma_option)
                

    def get_fwhm(self,
                 grid_points,
                 sets_of_band_indices,
                 tmax,
                 tmin,
                 tstep,
                 gamma_option=0,
                 filename=None):

        if grid_points==None:
            print "Grid points are not specified."
            return False

        if sets_of_band_indices==None:
            print "Band indices are not specified."
            return False

        for gp in grid_points:
            self.pp.set_triplets_at_q(gp)
            for band_indices in sets_of_band_indices:
                self.pp.set_interaction_strength(band_indices)
                self.pp.get_damping_function(temperature=None,
                                             filename=filename,
                                             gamma_option=gamma_option)

                fwhms, temps, omegas = \
                    self.pp.get_life_time(tmax,
                                          tmin,
                                          tstep,
                                          gamma_option,
                                          filename=filename)


                print "# Grid point:", gp
                print "# Frequencies:", omegas
                for fwhm, t in zip(fwhms, temps):
                    print t, fwhm
                print

    def get_decay_channels(self,
                           grid_points,
                           sets_of_band_indices,
                           temperature=None):

        if grid_points==None:
            print "Grid points are not specified."
            return False

        if sets_of_band_indices==None:
            print "Band indices are not specified."
            return False

        for gp in grid_points:
            self.pp.set_triplets_at_q(gp)
            for band_indices in sets_of_band_indices:
                self.pp.set_interaction_strength(band_indices)
                self.pp.get_decay_channels(temperature)

def get_gruneisen_parameters(fc2,
                             fc3,
                             supercell,
                             primitive,
                             mesh,
                             symprec):
    gr = Gruneisen(fc2,
                   fc3,
                   supercell,
                   primitive,
                   mesh,
                   symprec)

