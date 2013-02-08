class Lifetime:
    def __init__(self,
                 interaction_strength,
                 grid_points,
                 sigma=0.2):
        self._pp = interaction_strength
        self._grid_points = grid_points
        self._sigma = sigma

    def get_lifetime(self,
                     gamma_option=0,
                     filename=None):
        pass
