# Copyright (C) 2011 Atsushi Togo
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

class Forces(object):
    """
    forces: Forces on atoms in a supercell with a displacement in Cartesian coordinate
      [ [ F_1x, F_1y, F_1z ], 
        [ F_2x, F_2y, F_2z ], 
        ... ]
    displacement: An atomic displacement in Cartesian coordiante
      [ d_x, d_y, d_z ]
    """
    
    def __init__(self, atom_number, displacement, forces,
                 is_translational_invariance=False):
        self.atom_number = atom_number
        self.displacement = displacement
        self.forces = np.array(forces)
        if is_translational_invariance:
            self.set_translational_invariance()

    def get_atom_number(self):
        return self.atom_number

    def get_displacement(self):
        return self.displacement

    def get_forces(self):
        return self.forces

    def set_translational_invariance(self):
        self.forces = (self.forces - 
                       np.sum(self.forces, axis=0) / self.forces.shape[0])
        
