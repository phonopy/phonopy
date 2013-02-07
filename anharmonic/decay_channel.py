class DecayChannel:
    def __init__(self):
        pass

    def get_decay_channels(self, temperature):
        self._print_log("---- Decay channels ----\n")
        if temperature==None:
            self._print_log("Temperature: 0K\n")
        else:
            self._print_log("Temperature: %10.3fK\n" % temperature)

        q = self._grid_points[self._grid_point].astype(float) / self._mesh 
        if (not self._q_direction==None) and self._grid_point==0:
            self._dm.set_dynamical_matrix(q, self._q_direction)
        else:
            self._dm.set_dynamical_matrix(q)
        vals = np.linalg.eigvalsh(self._dm.get_dynamical_matrix())
        factor = self._factor * self._freq_factor * self._freq_scale
        freqs = np.sqrt(abs(vals)) * np.sign(vals) * factor
        omegas = np.array([freqs[x] for x in self._band_indices])

        if temperature==None:
            t = -1
        else:
            t = temperature

        import anharmonic._phono3py as phono3c

        decay_channels = np.zeros((len(self._weights_at_q),
                                   self._num_atom*3,
                                   self._num_atom*3), dtype=float)

        phono3c.decay_channel(decay_channels,
                              self._amplitude_at_q,
                              self._frequencies_at_q,
                              omegas,
                              self._freq_factor,
                              float(t),
                              float(self._sigma))

        decay_channels *= self._unit_conversion / np.sum(self._weights_at_q) / len(omegas)
        filename = write_decay_channels(decay_channels,
                                        self._amplitude_at_q,
                                        self._frequencies_at_q,
                                        self._triplets_at_q,
                                        self._weights_at_q,
                                        self._grid_points,
                                        self._mesh,
                                        self._band_indices,
                                        omegas,
                                        self._grid_point,
                                        is_nosym=self._is_nosym)

        decay_channels_sum = np.array(
            [d.sum() * w for d, w in
             zip(decay_channels, self._weights_at_q)]).sum()

        self._print_log("FWHM: %f\n" % (decay_channels_sum * 2))
        self._print_log( "Decay channels are written into %s.\n" % filename)

