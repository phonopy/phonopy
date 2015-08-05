import numpy as np

def get_phonons_at_qpoints(frequencies,
                           eigenvectors,
                           dm,
                           qpoints,
                           frequency_factor_to_THz,
                           nac_q_direction=None,
                           lapack_zheev_uplo='L'):
    import phonopy._lapackepy as lapackepy

    (svecs,
     multiplicity,
     masses,
     rec_lattice,
     born,
     nac_factor,
     dielectric) = _extract_params(dm)

    lapackepy.phonons_at_qpoints(
        frequencies,
        eigenvectors,
        np.array(qpoints, dtype='double', order='C'),
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
    import phonopy._lapackepy as lapackepy

    (svecs,
     multiplicity,
     masses,
     rec_lattice,
     born,
     nac_factor,
     dielectric) = _extract_params(dm)

    lapackepy.phonons_at_gridpoints(
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

def _extract_params(dm):
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

    return (svecs,
            multiplicity,
            masses,
            rec_lattice,
            born,
            nac_factor,
            dielectric)
