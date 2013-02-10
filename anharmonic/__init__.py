import sys
import phonopy.structure.spglib as spg
import numpy as np
from anharmonic.q2v import PhononPhonon
from anharmonic.im_self_energy import ImSelfEnergy
from anharmonic.linewidth import Linewidth
from anharmonic.BTE_RTA import BTE_RTA
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

        self._grid_points = None # The grid points to be calculated on.
        (grid_mapping_table,
         self._grid_address) = spg.get_ir_reciprocal_mesh(mesh,
                                                          primitive)
        self._ir_grid_indices = np.unique(grid_mapping_table)
        self._ir_weights = [np.sum(grid_mapping_table == 9)
                            for g in self._ir_grid_indices]
        
        self._pp = PhononPhonon(fc3,
                                supercell,
                                primitive,
                                mesh,
                                factor=self._factor,
                                freq_factor=self._freq_factor,
                                freq_scale=self._freq_scale,
                                symprec=self._symprec,
                                is_read_triplets=self._is_read_triplets,
                                r2q_TI_index=self._r2q_TI_index,
                                is_symmetrize_fc3_q=self._is_symmetrize_fc3_q,
                                is_Peierls=self._is_Peierls,
                                verbose=self._log_level,
                                is_nosym=self._is_nosym)
        
    def get_ir_grid_indices(self):
        return self._ir_grid_indices

    def get_grid_address(self):
        return self._grid_address
        
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
                             grid_points,
                             sets_of_band_indices,
                             sigma=None,
                             omega_step=None,
                             temperature=None,
                             filename=None,
                             gamma_option=0):

        self._sigma = sigma
        self._omega_step = omega_step
        self._im_self_energy = ImSelfEnergy(self._pp,
                                            sigma=self._sigma,
                                            omega_step=self._omega_step,
                                            verbose=self._log_level)

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

            else:
                for band_indices in sets_of_band_indices:
                    self._pp.set_interaction_strength(band_indices)
                    self._im_self_energy.get_damping_function(
                        temperature=temperature,
                        filename=filename,
                        gamma_option=gamma_option)

    def get_linewidth(self,
                      grid_points,
                      sets_of_band_indices,
                      sigma=0.2,
                      t_max=1000,
                      t_min=0,
                      t_step=10,
                      gamma_option=0,
                      filename=None):

        lw = Linewidth(self._pp,
                       sigma=sigma,
                       t_max=t_max,
                       t_min=t_min,
                       t_step=t_step)
        
        if grid_points is None:
            print "Grid points are not specified."
            return False

        if sets_of_band_indices is None:
            print "Band indices are not specified."
            return False

        for gp in grid_points:
            self._pp.set_triplets_at_q(gp)
            for band_indices in sets_of_band_indices:
                self._pp.set_interaction_strength(band_indices)
                fwhms, temps, omegas = lw.get_linewidth(
                    gamma_option,
                    filename=filename)

                print "# Grid point:", gp
                print "# Frequencies:", omegas
                for fwhm, t in zip(fwhms.T, temps):
                    print t, fwhm
                print

    def get_thermal_conductivity(self,
                                 sigma=0.2,
                                 t_max=1000,
                                 t_min=0,
                                 t_step=10,
                                 gamma_option=0,
                                 filename=None):

        lt = BTE_RTA(self._pp,
                     sigma=sigma,
                     t_max=t_max,
                     t_min=t_min,
                     t_step=t_step)
        partial_k = lt.get_kappa(gamma_option=gamma_option)
        temperatures = lt.get_temperatures()
        for t, k in zip(temperatures, partial_k.sum(axis=0)):
            print t, k
                
    # def get_decay_channels(self,
    #                        grid_points,
    #                        sets_of_band_indices,
    #                        temperature=None):

    #     if grid_points==None:
    #         print "Grid points are not specified."
    #         return False

    #     if sets_of_band_indices==None:
    #         print "Band indices are not specified."
    #         return False

    #     for gp in grid_points:
    #         self._pp.set_triplets_at_q(gp)
    #         for band_indices in sets_of_band_indices:
    #             self._pp.set_interaction_strength(band_indices)
    #             self._pp.get_decay_channels(temperature)

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
