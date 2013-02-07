import numpy as np
from anharmonic.file_IO import write_fwhm

class LineWidth:
    def __init__(self):
        pass

    def get_fwhm(self,
                 tmax,
                 tmin,
                 tstep,
                 gamma_option=0,
                 filename=None):
        
        self._print_log("---- FWHM calculation ----\n")

        # A set of phonon modes where the fwhm is calculated.
        q = self._grid_points[self._grid_point].astype(float) / self._mesh
        if (not self._q_direction==None) and self._grid_point==0:
            self._dm.set_dynamical_matrix(q, self._q_direction)
        else:
            self._dm.set_dynamical_matrix(q)
        vals = np.linalg.eigvalsh(self._dm.get_dynamical_matrix())
        factor = self._factor * self._freq_factor * self._freq_scale
        freqs = np.sqrt(abs(vals)) * np.sign(vals) * factor
        omegas = np.array([freqs[x] for x in self._band_indices])
        fwhms = []
        temps = []

        for t in np.arange(tmin, tmax + float(tstep) / 2, tstep):
            g_sum = 0.0
            temps.append(t)
            for i in range(len(self._band_indices)):
                g = get_gamma(self._amplitude_at_q,
                              omegas,
                              self._weights_at_q,
                              self._frequencies_at_q,
                              i,
                              t,
                              self._sigma,
                              self._freq_factor,
                              gamma_option).sum() * self._unit_conversion * 2
                g /= len(self._band_indices)
                g_sum += g
            fwhms.append(g_sum / len(self._band_indices))

        write_fwhm(self._grid_point,
                   self._band_indices + 1,
                   temps,
                   fwhms,
                   self._mesh,
                   is_nosym=self._is_nosym,
                   filename=filename)

        return fwhms, temps, omegas

