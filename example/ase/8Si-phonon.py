from gpaw import GPAW, PW
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

def get_gpaw(kpts_size=None):
    if kpts_size is None:
        calc = GPAW(mode=PW(300),
                    kpts={'size': (2, 2, 2)})
    else:
        calc = GPAW(mode=PW(300),
                    kpts={'size': kpts_size})
    return calc

def get_crystal():
    a = 5.404
    cell = PhonopyAtoms(symbols=(['Si'] * 8),
                        cell=np.diag((a, a, a)),
                        scaled_positions=[(0, 0, 0),
                                          (0, 0.5, 0.5),
                                          (0.5, 0, 0.5),
                                          (0.5, 0.5, 0),
                                          (0.25, 0.25, 0.25),
                                          (0.25, 0.75, 0.75),
                                          (0.75, 0.25, 0.75),
                                          (0.75, 0.75, 0.25)])
    return cell

def phonopy_pre_process(cell, supercell_matrix=None):

    if supercell_matrix is None:
        smat = [[2,0,0], [0,2,0], [0,0,2]],
    else:
        smat = supercell_matrix
    phonon = Phonopy(cell,
                     smat,
                     primitive_matrix=[[0, 0.5, 0.5],
                                       [0.5, 0, 0.5],
                                       [0.5, 0.5, 0]])
    phonon.generate_displacements(distance=0.03)
    print("[Phonopy] Atomic displacements:")
    disps = phonon.get_displacements()
    for d in disps:
        print("[Phonopy] %d %s" % (d[0], d[1:]))
    return phonon

def run_gpaw(calc, phonon):
    supercells = phonon.get_supercells_with_displacements()
    # Force calculations by calculator
    set_of_forces = []
    for scell in supercells:
        cell = Atoms(symbols=scell.get_chemical_symbols(),
                     scaled_positions=scell.get_scaled_positions(),
                     cell=scell.get_cell(),
                     pbc=True)
        cell.set_calculator(calc)
        forces = cell.get_forces()
        drift_force = forces.sum(axis=0)
        print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
        # Simple translational invariance
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    return set_of_forces

def phonopy_post_process(phonon, set_of_forces):
    phonon.produce_force_constants(forces=set_of_forces)
    print('')
    print("[Phonopy] Phonon frequencies at Gamma:")
    for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
        print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq)) # THz
    
    # DOS
    phonon.set_mesh([21, 21, 21])
    phonon.set_total_DOS(tetrahedron_method=True)
    print('')
    print("[Phonopy] Phonon DOS:")
    for omega, dos in np.array(phonon.get_total_DOS()).T:
        print("%15.7f%15.7f" % (omega, dos))

def main():
    cell = get_crystal()

    # 1x1x1 supercell of conventional unit cell
    calc = get_gpaw(kpts_size=(4, 4, 4))
    phonon = phonopy_pre_process(cell, supercell_matrix=np.eye(3, dtype='intc'))

    # # 2x2x2 supercell of conventional unit cell
    # calc = get_gpaw(kpts_size=(2, 2, 2))
    # phonon = phonopy_pre_process(cell,
    #                              supercell_matrix=(np.eye(3, dtype='intc') * 2))

    set_of_forces = run_gpaw(calc, phonon)
    phonopy_post_process(phonon, set_of_forces)

if __name__ == "__main__":
    main()
