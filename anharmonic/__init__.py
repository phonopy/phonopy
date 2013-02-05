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
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._fc2 = fc2
        self._nac_params = nac_params
        self._sigma = sigma
        self._omega_step = omega_step
        self._factor = factor
        self._freq_factor = freq_factor
        self._freq_scale = freq_scale
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._log_level = log_level

    def get_jointDOS(self, grid_points, filename=None):
        get_jointDOS(grid_points,
                     self._mesh,
                     self._primitive,
                     self._supercell,
                     self._fc2,
                     self._nac_params,
                     self._sigma,
                     self._omega_step,
                     self._factor,
                     self._freq_factor,
                     self._freq_scale,
                     self._is_nosym,
                     self._symprec,
                     filename,
                     self._log_level)

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

        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._fc3 = fc3
        self._sigma = sigma
        self._omega_step = omega_step
        self._factor = factor
        self._freq_factor = freq_factor
        self._freq_scale = freq_scale
        self._is_nosym = is_nosym
        self._is_symmetrize_fc3_q = is_symmetrize_fc3_q
        self._symprec = symprec
        self._log_level = log_level
        self._is_read_triplets = is_read_triplets
        self._r2q_TI_index = r2q_TI_index
        self._is_Peierls = is_Peierls

        # Phonon-phonon
        self._pp = None

    def set_phonon_phonon(self):
        self._pp = PhononPhonon(self._fc3,
                                self._supercell,
                                self._primitive,
                                self._mesh,
                                self._sigma,
                                self._omega_step,
                                self._factor,
                                self._freq_factor,
                                self._freq_scale,
                                self._symprec,
                                self._is_read_triplets,
                                self._r2q_TI_index,
                                self._is_symmetrize_fc3_q,
                                self._is_Peierls,
                                self._log_level,
                                self._is_nosym)

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None):
        self._pp.set_dynamical_matrix(fc2,
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
            self._pp.set_triplets_at_q(gp)

            if sets_of_band_indices == None:
                if gp==0:
                    self._pp.set_interaction_strength(
                        range(4, self._primitive.get_number_of_atoms() * 3 + 1))
                else:
                    self._pp.set_interaction_strength(
                        range(1, self._primitive.get_number_of_atoms() * 3 + 1))

                self._pp.get_damping_function(temperature=temperature,
                                              filename=filename,
                                              gamma_option=gamma_option)
            else:
                for band_indices in sets_of_band_indices:
                    self._pp.set_interaction_strength(band_indices)
                    self._pp.get_damping_function(temperature=temperature,
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
            self._pp.set_triplets_at_q(gp)
            for band_indices in sets_of_band_indices:
                self._pp.set_interaction_strength(band_indices)
                self._pp.get_damping_function(temperature=None,
                                             filename=filename,
                                             gamma_option=gamma_option)

                fwhms, temps, omegas = self._pp.get_life_time(
                    tmax,
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
            self._pp.set_triplets_at_q(gp)
            for band_indices in sets_of_band_indices:
                self._pp.set_interaction_strength(band_indices)
                self._pp.get_decay_channels(temperature)

def get_gruneisen_parameters(fc2,
                             fc3,
                             supercell,
                             primitive,
                             factor=None,
                             is_ion_clamped=False,
                             symprec=1e-5):
    return Gruneisen(fc2,
                     fc3,
                     supercell,
                     primitive,
                     factor=factor,
                     is_ion_clamped=is_ion_clamped,
                     symprec=symprec)
