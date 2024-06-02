import sys
from phonopy.interface.qe import read_pwscf, PH_Q2R

primcell_filename = sys.argv[1]
q2r_filename = sys.argv[2]
cell, _ = read_pwscf(primcell_filename)
q2r = PH_Q2R(q2r_filename)
q2r.run(cell)
q2r.write_force_constants()
