import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC

def get_dynamical_matrix(fc2,
                         supercell,
                         primitive,
                         nac_params=None,
                         frequency_scale_factor=None,
                         decimals=None,
                         symprec=1e-5):
    if nac_params is None:
        dm = DynamicalMatrix(
            supercell,
            primitive,
            fc2,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=symprec)
    else:
        dm = DynamicalMatrixNAC(
            supercell,
            primitive,
            fc2,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=symprec)
        dm.set_nac_params(nac_params)
    return dm

def set_phonon_c(dm,
                 frequencies,
                 eigenvectors,
                 phonon_done,
                 grid_points,
                 grid_address,
                 mesh,
                 frequency_factor_to_THz,
                 nac_q_direction,
                 lapack_zheev_uplo):
    import anharmonic._phono3py as phono3c
    svecs, multiplicity = dm.get_shortest_vectors()
    masses = np.array(dm.get_primitive().get_masses(), dtype='double')
    rec_lattice = np.array(
        np.linalg.inv(dm.get_primitive().get_cell()), dtype='double', order='C')
    if dm.is_nac():
        born = dm.get_born_effective_charges()
        nac_factor = dm.get_nac_factor()
        dielectric = dm.get_dielectric_constant()
    else:
        born = None
        nac_factor = 0
        dielectric = None

    phono3c.phonons_at_gridpoints(
        frequencies,
        eigenvectors,
        phonon_done,
        grid_points,
        grid_address,
        np.array(mesh, dtype='intc'),
        dm.get_force_constants(),
        svecs,
        multiplicity,
        masses,
        dm.get_primitive_to_supercell_map(),
        dm.get_supercell_to_primitive_map(),
        frequency_factor_to_THz,
        born,
        dielectric,
        rec_lattice,
        nac_q_direction,
        nac_factor,
        lapack_zheev_uplo)

def set_phonon_py(grid_point,
                  phonon_done,
                  frequencies,
                  eigenvectors,
                  grid_address,
                  mesh,
                  dynamical_matrix,
                  frequency_factor_to_THz,                  
                  lapack_zheev_uplo):
    gp = grid_point
    if phonon_done[gp] == 0:
        phonon_done[gp] = 1
        q = grid_address[gp].astype('double') / mesh
        dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm, UPLO=lapack_zheev_uplo)
        eigvals = eigvals.real
        frequencies[gp] = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                           * frequency_factor_to_THz)
        eigenvectors[gp] = eigvecs

