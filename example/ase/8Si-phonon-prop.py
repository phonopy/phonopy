from gpaw import GPAW
from ase import *
from ase.calculators import numeric_force
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
from phonopy.units import VaspToTHz
import numpy as np

# GPAW setting
a = 5.404
bulk = PhonopyAtoms(symbols=['Si']*8,
                    scaled_positions=[(0, 0, 0),
                                      (0, 0.5, 0.5),
                                      (0.5, 0, 0.5),
                                      (0.5, 0.5, 0),
                                      (0.25, 0.25, 0.25),
                                      (0.25, 0.75, 0.75),
                                      (0.75, 0.25, 0.75),
                                      (0.75, 0.75, 0.25)] )
bulk.set_cell(np.diag((a, a, a)))

n = 24
calc = GPAW(gpts=(n, n, n),
            nbands=8*3,
            width=0.01,
            kpts=(2, 2, 2),
            convergence={'eigenstates': 1e-9}
            )

# Phonopy pre-process
print "------"
print "Phonon"
print "------"
# 1st arg. is the input unit cell.
# 2nd arg. is the supercell lattice relative to the unit cell.
# 'distance' is the distance of displacements.
# Default symmetry tolerance is 1e-5 in fractional coordinates.
phonon = Phonopy(bulk, [[1,0,0],[0,1,0],[0,0,1]], distance=0.01, factor=VaspToTHz)
phonon.print_displacements()
supercells = phonon.get_supercells_with_displacements()

# Force calculations by calculator
set_of_forces = []
for scell in supercells:
    cell = Atoms( symbols=scell.get_chemical_symbols(),
                  scaled_positions=scell.get_scaled_positions(),
                  cell=scell.get_cell(),
                  pbc=True )
    cell.set_calculator(calc)
    forces = cell.get_forces()
    drift_force = forces.sum(axis=0)
    print "        ---------------------------------"
    print "     ", "%11.5f"*3 % tuple(drift_force)
    # Simple translational invariance
    for force in forces:
        force -= drift_force / forces.shape[0]
    set_of_forces.append(forces)

# Phonopy post-process
# 1st arg. is a relative lattice to the input unit cell.
# 2nd arg. is bunch of the calculated forces.
phonon.set_post_process([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]], set_of_forces)

# Thermal properties
# mesh: Monkhorst-Pack
# shift: mesh shift relative to the grid distance
mesh = [ 10, 10, 10 ]
shift = [ 0.5, 0.5, 0.5 ]
print "\nThermal properties :"
print "%12s %15s%15s%15s" % ('T [K]',
                              'F [kJ/mol]',
                              'S [J/K/mol]',
                              'C_v [J/K/mol]')

# get_thermal_properties returns numpy array of
#
# [[ temperature, free energy, entropy, heat capacity ],
#  [ temperature, free energy, entropy, heat capacity ],...,]
#
# Frequency has to be given in THz internally. Therefore unit
# conversion factor may be specified when calling Phonon class. The
# unit of frequency in a calculator is square root of the unit of
# dynamical matrix, i.e.,
#
# /      [energy]       \^(1/2)
# | ------------------- |
# \ [mass] [distance]^2 /
#
# THz is the value above divided by 2pi*1e12 (2pi comes from the
# factor between angular frequency and ordinary frequency). See
# units.py in the phonopy directory.
phonon.set_mesh( mesh, shift )
phonon.set_thermal_properties( t_step=10,
                               t_max=1000,
                               t_min=0 )
for t, free_energy, entropy, cv in phonon.get_thermal_properties():
    print ("%12.3f " + "%15.7f" * 3) % ( t, free_energy, entropy, cv )

phonon.plot_thermal_properties().show()
