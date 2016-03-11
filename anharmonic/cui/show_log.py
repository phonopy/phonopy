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

import os
import sys
import numpy as np
from phonopy.structure.cells import print_cell

def file_exists(filename, log_level):
    if os.path.exists(filename):
        return True
    else:
        error_text = "%s not found." % filename
        print(error_text)
        if log_level > 0:
            print_error()
        sys.exit(1)

# AA is created at http://www.network-science.de/ascii/.
def print_phono3py():
    print("""        _                      _____             
  _ __ | |__   ___  _ __   ___|___ / _ __  _   _ 
 | '_ \| '_ \ / _ \| '_ \ / _ \ |_ \| '_ \| | | |
 | |_) | | | | (_) | | | | (_) |__) | |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___/____/| .__/ \__, |
 |_|                                |_|    |___/ """)

def print_version(version):
    print(" " * 42 + "%s" % version)
    print('')

def print_end():
    print("""                 _ 
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
""")

def print_error():
    print("""  ___ _ __ _ __ ___  _ __ 
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |   
 \___|_|  |_|  \___/|_|
""")

def print_error_message(message):
    print('')
    print(message)

def show_phono3py_cells(symmetry,
                        primitive,
                        supercell,
                        phonon_primitive,
                        phonon_supercell,
                        settings):
    print("Spacegroup: %s" % symmetry.get_international_table())
    print("-" * 30 + " primitive cell " + "-" * 30)
    print_cell(primitive)
    print("-" * 32 + " super cell " + "-" * 32)
    print_cell(supercell, mapping=primitive.get_supercell_to_primitive_map())
    print("-" * 19 + " ratio (supercell for fc)/(primitive) " + "-" * 19)
    for vec in np.dot(supercell.get_cell(),
                      np.linalg.inv(primitive.get_cell())):
        print(("%5.2f" * 3) % tuple(vec))
    if settings.get_phonon_supercell_matrix() is not None:
        print("-" * 19 + " primitive cell for harmonic phonon " + "-" * 20)
        print_cell(phonon_primitive)
        print("-" * 21 + " supercell for harmonic phonon " + "-" * 22)
        print_cell(phonon_supercell,
                   mapping=phonon_primitive.get_supercell_to_primitive_map())
        print("-" * 15 + " ratio (phonon supercell)/(phonon primitive) " +
              "-" * 15)
        for vec in np.dot(phonon_supercell.get_cell(),
                          np.linalg.inv(phonon_primitive.get_cell())):
            print(("%5.2f" * 3) % tuple(vec))

def show_phono3py_force_constants_settings(read_fc3,
                                           read_fc2,
                                           tsym_type,
                                           is_symmetrize_fc3_r,
                                           is_symmetrize_fc3_q,
                                           is_symmetrize_fc2,
                                           settings):
    print("-" * 29 + " Force constants " + "-" * 30)
    if not read_fc2:
        print("Imposing translational symmetry to fc2: %s" % 
              (tsym_type > 0))
        print("Imposing symmetry of index exchange to fc2: %s" %
              is_symmetrize_fc2)
        
    if not (read_fc3 or
            settings.get_is_isotope() or
            settings.get_is_joint_dos()):
        print("Imposing translational symmetry to fc3: %s" %
              (tsym_type > 0))
        print("Imposing symmetry of index exchange to fc3 in real space: %s" %
              is_symmetrize_fc3_r)
        print(("Imposing symmetry of index exchange to fc3 in reciprocal space: "
              "%s") % is_symmetrize_fc3_q)
        
    if settings.get_cutoff_fc3_distance() is not None:
        print("FC3 cutoff distance: %s" % settings.get_cutoff_fc3_distance())

def show_phono3py_settings(settings,
                           mesh,
                           mesh_divs,
                           band_indices,
                           sigmas,
                           temperatures,
                           temperature_points,
                           grid_points,
                           cutoff_frequency,
                           frequency_factor_to_THz,
                           frequency_step,
                           num_frequency_points,
                           log_level):
    print("-" * 33 + " Settings " + "-" * 33)
    if settings.get_is_nac():
        print("Non-analytical term correction: %s" % settings.get_is_nac())
    if mesh is not None:
        print("Mesh sampling: [ %d %d %d ]" % tuple(mesh))
    if mesh_divs is not None and settings.get_is_bterta():
        print("Mesh divisors: [ %d %d %d ]" % tuple(mesh_divs))
    if band_indices is not None and not settings.get_is_bterta():
        print(("Band indices: [" + " %s" * len(band_indices) + " ]") %
              tuple([np.array(bi) + 1 for bi in band_indices]))
    if sigmas:
        text = "BZ integration: "
        for i, sigma in enumerate(sigmas):
            if sigma:
                text += "Smearing=%s" % sigma
            else:
                text += "Tetrahedron-method"
            if i < len(sigmas) - 1:
                text += ", "
        print(text)
        
    if (settings.get_is_linewidth() or
        settings.get_is_frequency_shift() or
        settings.get_is_bterta() or
        settings.get_is_lbte()):
        if len(temperatures) > 5:
            text = (" %.1f " * 5 + "...") % tuple(temperatures[:5])
            text += " %.1f" % temperatures[-1]
        else:
            text = (" %.1f " * len(temperatures)) % tuple(temperatures)
        print("Temperature: " + text)
    elif temperature_points is not None:
        print(("Temperatures:" + " %.1f " * len(temperature_points))
              % tuple(temperature_points))
        if settings.get_scattering_event_class() is not None:
            print("Scattering event class: %s" %
                  settings.get_scattering_event_class())
            
    if grid_points is not None:
        text = "Grid point to be calculated: "
        if len(grid_points) > 8:
            for i, gp in enumerate(grid_points):
                if i % 10 == 0:
                    text += "\n"
                    text += " "
                text += "%d " % gp
        else:
            for gp in grid_points:
                text += "%d " % gp
        print(text)
            
    if cutoff_frequency:
        print("Cutoff frequency: %s" % cutoff_frequency)

    if (settings.get_use_ave_pp() and
        (settings.get_is_bterta() or settings.get_is_lbte())):
        print("Use averaged ph-ph interaction")

    if log_level > 1:
        print("Frequency factor to THz: %s" % frequency_factor_to_THz)
        if frequency_step is not None:
            print("Frequency step for spectrum: %s" % frequency_step)
        if num_frequency_points is not None:
            print("Number of frequency sampling points: %d" %
                  num_frequency_points)
    sys.stdout.flush()
