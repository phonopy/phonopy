# Copyright (C) 2015 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from phonopy.units import VaspToTHz
from anharmonic.phonon3.triplets import get_grid_point_from_address

def get_phono3py_configurations(settings):
    primitive_matrix = settings.get_primitive_matrix()
    supercell_matrix = settings.get_supercell_matrix()
    phonon_supercell_matrix = settings.get_phonon_supercell_matrix()
    masses = settings.get_masses()
    mesh = settings.get_mesh_numbers()
    mesh_divs = settings.get_mesh_divisors()
    coarse_mesh_shifts = settings.get_coarse_mesh_shifts()
    grid_points = settings.get_grid_points()
    grid_addresses = settings.get_grid_addresses()
    if grid_addresses is not None:
        grid_points = [get_grid_point_from_address(ga, mesh)
                       for ga in grid_addresses]
    band_indices = settings.get_band_indices()
    
    # Brillouin zone integration: Tetrahedron (default) or smearing method
    sigma = settings.get_sigma()
    if sigma is None:
        sigmas = []
    elif isinstance(sigma, float):
        sigmas = [sigma]
    else:
        sigmas = sigma
    if settings.get_is_tetrahedron_method():
        sigmas = [None] + sigmas
    if len(sigmas) == 0:
        sigmas = [None]

    if settings.get_temperatures() is None:
        if settings.get_is_joint_dos():
            temperature_points = None
            temperatures = None
        else:
            t_max = settings.get_max_temperature()
            t_min = settings.get_min_temperature()
            t_step = settings.get_temperature_step()
            temperature_points = [0.0, 300.0] # For spectra
            temperatures = np.arange(t_min, t_max + float(t_step) / 10, t_step)
    else:
        temperature_points = settings.get_temperatures() # For spectra
        temperatures = settings.get_temperatures() # For others

    if settings.get_frequency_conversion_factor() is None:
        frequency_factor_to_THz = VaspToTHz
    else:
        frequency_factor_to_THz = settings.get_frequency_conversion_factor()

    if settings.get_num_frequency_points() is None:
        if settings.get_frequency_pitch() is None:
            num_frequency_points = 201
            frequency_step = None
        else:
            num_frequency_points = None
            frequency_step = settings.get_frequency_pitch()
    else:
        num_frequency_points = settings.get_num_frequency_points()
        frequency_step = None

    if settings.get_frequency_scale_factor() is None:
        frequency_scale_factor = 1.0
    else:
        frequency_scale_factor = settings.get_frequency_scale_factor()

    if settings.get_cutoff_frequency() is None:
        cutoff_frequency = 1e-2
    else:
        cutoff_frequency = settings.get_cutoff_frequency()

    conf = {}
    conf['primitive_matrix'] = primitive_matrix
    conf['supercell_matrix'] = supercell_matrix
    conf['phonon_supercell_matrix'] = phonon_supercell_matrix
    conf['masses'] = masses
    conf['mesh'] = mesh
    conf['mesh_divs'] = mesh_divs
    conf['coarse_mesh_shifts'] = coarse_mesh_shifts
    conf['grid_points'] = grid_points
    conf['band_indices'] = band_indices
    conf['sigmas'] = sigmas
    conf['temperature_points'] = temperature_points
    conf['temperatures'] = temperatures
    conf['frequency_factor_to_THz'] = frequency_factor_to_THz
    conf['num_frequency_points'] = num_frequency_points
    conf['frequency_step'] = frequency_step
    conf['frequency_scale_factor'] = frequency_scale_factor
    conf['cutoff_frequency'] = cutoff_frequency

    return conf
