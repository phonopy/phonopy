from gpaw import GPAW
from ase import *
from ase.calculators import numeric_force
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms
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

n = 32
calc = GPAW(gpts=(n, n, n),
            nbands=8*3,
            width=0.01,
            kpts=(2, 2, 2),
            convergence={'eigenstates': 1e-7}
            )

# Phonopy pre-process
print "------"
print "Phonon"
print "------"
# 1st arg. is the input unit cell.
# 2nd arg. is the supercell lattice relative to the unit cell.
# 'distance' is the distance of displacements.
# Default symmetry tolerance is 1e-5 in fractional coordinates.
phonon = Phonopy(bulk, [[1,0,0],[0,1,0],[0,0,1]], distance=0.01)
symmetry = phonon.get_symmetry()
print "Space group:", symmetry.get_international_table()


phonon.print_displacements()
supercells = phonon.get_supercells_with_displacements()

# Force calculations
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

# Phonon dispersion relation
bands = []

q_start  = np.array([0.5,0.5,0.0])
q_end    = np.array([0.0,0.0,0.0])
band = []
for i in range(51):
    band.append( q_start + ( q_end - q_start ) / 50 * i )

bands.append(band)

q_start  = np.array([0.0,0.0,0.0])
q_end    = np.array([0.5,0.5,0.5])
band = []
for i in range(51):
    band.append( q_start + ( q_end - q_start ) / 50 * i )
    
bands.append(band)

#*********************
# Matplotlib required
#*********************
print "\nPhonon dispersion:"
phonon.set_band_structure( bands )
phonon.plot_band_structure( ["X", "$\Gamma$", "L"] ).show()
