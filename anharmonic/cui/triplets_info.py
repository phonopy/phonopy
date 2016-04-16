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

import sys
import numpy as np
from anharmonic.file_IO import (write_ir_grid_points,
                                write_grid_address_to_hdf5)
from anharmonic.phonon3.triplets import (get_coarse_ir_grid_points,
                                         get_number_of_triplets)

def write_grid_points(primitive,
                      mesh,
                      mesh_divs,
                      coarse_mesh_shifts,
                      is_kappa_star,
                      symprec,
                      log_level):
    print("-" * 76)
    if mesh is None:
        print("To write grid points, mesh numbers have to be specified.")
    else:
        (ir_grid_points,
         grid_weights,
         bz_grid_address,
         grid_mapping_table) = get_coarse_ir_grid_points(
             primitive,
             mesh,
             mesh_divs,
             coarse_mesh_shifts,
             is_kappa_star=is_kappa_star,
             symprec=symprec)
        write_ir_grid_points(mesh,
                             mesh_divs,
                             ir_grid_points,
                             grid_weights,
                             bz_grid_address,
                             np.linalg.inv(primitive.get_cell()))
        gadrs_hdf5_fname = write_grid_address_to_hdf5(bz_grid_address,
                                                      mesh,
                                                      grid_mapping_table)

        print("Ir-grid points are written into \"ir_grid_points.yaml\".")
        print("Grid addresses are written into \"%s\"." % gadrs_hdf5_fname)

def show_num_triplets(primitive,
                      mesh,
                      mesh_divs,
                      grid_points,
                      coarse_mesh_shifts,
                      is_kappa_star,
                      symprec,
                      log_level):
    print("-" * 76)

    ir_grid_points, _, grid_address, _ = get_coarse_ir_grid_points(
        primitive,
        mesh,
        mesh_divs,
        coarse_mesh_shifts,
        is_kappa_star=is_kappa_star,
        symprec=symprec)

    if grid_points:
        _grid_points = grid_points
    else:
        _grid_points = ir_grid_points

    print("Grid point        q-point        No. of triplets")
    for gp in _grid_points:
        num_triplets = get_number_of_triplets(primitive,
                                              mesh,
                                              gp,
                                              symprec=symprec)
        q = grid_address[gp] / np.array(mesh, dtype='double')
        print("  %5d     (%5.2f %5.2f %5.2f)  %8d" %
              (gp, q[0], q[1], q[2], num_triplets))
