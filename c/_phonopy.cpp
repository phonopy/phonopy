#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "phonopy.h"

namespace nb = nanobind;

void py_transform_dynmat_to_fc(nb::ndarray<> py_force_constants,
                               nb::ndarray<> py_dynamical_matrices,
                               nb::ndarray<> py_commensurate_points,
                               nb::ndarray<> py_svecs, nb::ndarray<> py_multi,
                               nb::ndarray<> py_masses,
                               nb::ndarray<> py_s2pp_map,
                               nb::ndarray<> py_fc_index_map, long use_openmp) {
    double *fc;
    double(*dm)[2];
    double(*comm_points)[3];
    double(*svecs)[3];
    double *masses;
    long(*multi)[2];
    long *s2pp_map;
    long *fc_index_map;
    long num_patom;
    long num_satom;

    fc = (double *)py_force_constants.data();
    dm = (double(*)[2])py_dynamical_matrices.data();
    comm_points = (double(*)[3])py_commensurate_points.data();
    svecs = (double(*)[3])py_svecs.data();
    masses = (double *)py_masses.data();
    multi = (long(*)[2])py_multi.data();
    s2pp_map = (long *)py_s2pp_map.data();
    fc_index_map = (long *)py_fc_index_map.data();
    num_patom = py_multi.shape(1);
    num_satom = py_multi.shape(0);

    phpy_transform_dynmat_to_fc(fc, dm, comm_points, svecs, multi, masses,
                                s2pp_map, fc_index_map, num_patom, num_satom,
                                use_openmp);
};

void py_perm_trans_symmetrize_fc(nb::ndarray<> py_force_constants, long level) {
    double *fc;
    int n_satom;

    fc = (double *)py_force_constants.data();
    n_satom = py_force_constants.shape(0);

    phpy_perm_trans_symmetrize_fc(fc, n_satom, level);
}

void py_perm_trans_symmetrize_compact_fc(nb::ndarray<> py_force_constants,
                                         nb::ndarray<> py_permutations,
                                         nb::ndarray<> py_s2pp_map,
                                         nb::ndarray<> py_p2s_map,
                                         nb::ndarray<> py_nsym_list,
                                         long level) {
    double *fc;
    int *perms;
    int *s2pp;
    int *p2s;
    int *nsym_list;
    int n_patom, n_satom;

    fc = (double *)py_force_constants.data();
    perms = (int *)py_permutations.data();
    s2pp = (int *)py_s2pp_map.data();
    p2s = (int *)py_p2s_map.data();
    nsym_list = (int *)py_nsym_list.data();
    n_patom = py_force_constants.shape(0);
    n_satom = py_force_constants.shape(1);

    phpy_perm_trans_symmetrize_compact_fc(fc, p2s, s2pp, nsym_list, perms,
                                          n_satom, n_patom, level);
}

void py_transpose_compact_fc(nb::ndarray<> py_force_constants,
                             nb::ndarray<> py_permutations,
                             nb::ndarray<> py_s2pp_map,
                             nb::ndarray<> py_p2s_map,
                             nb::ndarray<> py_nsym_list) {
    double *fc;
    int *s2pp;
    int *p2s;
    int *nsym_list;
    int *perms;
    int n_patom, n_satom;

    fc = (double *)py_force_constants.data();
    perms = (int *)py_permutations.data();
    s2pp = (int *)py_s2pp_map.data();
    p2s = (int *)py_p2s_map.data();
    nsym_list = (int *)py_nsym_list.data();
    n_patom = py_force_constants.shape(0);
    n_satom = py_force_constants.shape(1);

    phpy_set_index_permutation_symmetry_compact_fc(fc, p2s, s2pp, nsym_list,
                                                   perms, n_satom, n_patom, 1);
}

void py_get_dynamical_matrices_with_dd_openmp_over_qpoints(
    nb::ndarray<> py_dynamical_matrix, nb::ndarray<> py_qpoints,
    nb::ndarray<> py_force_constants, nb::ndarray<> py_svecs,
    nb::ndarray<> py_multi, nb::ndarray<> py_positions, nb::ndarray<> py_masses,
    nb::ndarray<> py_s2p_map, nb::ndarray<> py_p2s_map,
    nb::ndarray<> py_q_direction, nb::ndarray<> py_born,
    nb::ndarray<> py_dielectric, nb::ndarray<> py_reciprocal_lattice,
    double nac_factor, nb::ndarray<> py_dd_q0, nb::ndarray<> py_G_list,
    double lambda, long is_nac, long is_nac_q_zero, long use_Wang_NAC) {
    double(*dm)[2];
    double *fc;
    double *q_direction;
    double(*qpoints)[3];
    double(*svecs)[3];
    long(*multi)[2];
    double(*positions)[3];
    double *masses;
    double(*born)[3][3];
    double(*dielectric)[3];
    double(*reciprocal_lattice)[3];
    double(*dd_q0)[2];
    double(*G_list)[3];

    long *s2p_map;
    long *p2s_map;
    long num_patom;
    long num_satom;
    long n_qpoints;
    long n_Gpoints;

    dm = (double(*)[2])py_dynamical_matrix.data();
    qpoints = (double(*)[3])py_qpoints.data();
    n_qpoints = py_qpoints.shape(0);
    fc = (double *)py_force_constants.data();
    svecs = (double(*)[3])py_svecs.data();
    multi = (long(*)[2])py_multi.data();
    masses = (double *)py_masses.data();
    s2p_map = (long *)py_s2p_map.data();
    p2s_map = (long *)py_p2s_map.data();
    born = (double(*)[3][3])py_born.data();
    dielectric = (double(*)[3])py_dielectric.data();
    reciprocal_lattice = (double(*)[3])py_reciprocal_lattice.data();

    if (use_Wang_NAC || (!is_nac)) {
        positions = NULL;
        dd_q0 = NULL;
        G_list = NULL;
        n_Gpoints = 0;
    } else {
        positions = (double(*)[3])py_positions.data();
        dd_q0 = (double(*)[2])py_dd_q0.data();
        G_list = (double(*)[3])py_G_list.data();
        n_Gpoints = py_G_list.shape(0);
    }

    if (is_nac_q_zero || (!is_nac)) {
        q_direction = NULL;
    } else {
        q_direction = (double *)py_q_direction.data();
    }

    num_patom = py_p2s_map.shape(0);
    num_satom = py_s2p_map.shape(0);

    phpy_dynamical_matrices_with_dd_openmp_over_qpoints(
        dm, qpoints, n_qpoints, fc, svecs, multi, positions, num_patom,
        num_satom, masses, p2s_map, s2p_map, born, dielectric,
        reciprocal_lattice, q_direction, nac_factor, dd_q0, G_list, n_Gpoints,
        lambda, use_Wang_NAC);
}

void py_get_recip_dipole_dipole(
    nb::ndarray<> py_dd, nb::ndarray<> py_dd_q0, nb::ndarray<> py_G_list,
    nb::ndarray<> py_q_cart, nb::ndarray<> py_q_direction,
    nb::ndarray<> py_born, nb::ndarray<> py_dielectric,
    nb::ndarray<> py_positions, long is_nac_q_zero, double factor,
    double lambda, double tolerance, long use_openmp) {
    double(*dd)[2];
    double(*dd_q0)[2];
    double(*G_list)[3];
    double *q_vector;
    double *q_direction;
    double(*born)[3][3];
    double(*dielectric)[3];
    double(*pos)[3];

    long num_patom, num_G;

    dd = (double(*)[2])py_dd.data();
    dd_q0 = (double(*)[2])py_dd_q0.data();
    G_list = (double(*)[3])py_G_list.data();
    if (is_nac_q_zero) {
        q_direction = NULL;
    } else {
        q_direction = (double *)py_q_direction.data();
    }
    q_vector = (double *)py_q_cart.data();
    born = (double(*)[3][3])py_born.data();
    dielectric = (double(*)[3])py_dielectric.data();
    pos = (double(*)[3])py_positions.data();
    num_G = py_G_list.shape(0);
    num_patom = py_positions.shape(0);

    phpy_get_recip_dipole_dipole(dd,    /* [natom, 3, natom, 3, (real, imag)] */
                                 dd_q0, /* [natom, 3, 3, (real, imag)] */
                                 G_list, /* [num_kvec, 3] */
                                 num_G, num_patom, q_vector, q_direction, born,
                                 dielectric, pos, /* [natom, 3] */
                                 factor,          /* 4pi/V*unit-conv */
                                 lambda,          /* 4 * Lambda^2 */
                                 tolerance, use_openmp);
}

void py_get_recip_dipole_dipole_q0(nb::ndarray<> py_dd_q0,
                                   nb::ndarray<> py_G_list,
                                   nb::ndarray<> py_born,
                                   nb::ndarray<> py_dielectric,
                                   nb::ndarray<> py_positions, double lambda,
                                   double tolerance, long use_openmp) {
    double(*dd_q0)[2];
    double(*G_list)[3];
    double(*born)[3][3];
    double(*dielectric)[3];
    double(*pos)[3];

    long num_patom, num_G;

    dd_q0 = (double(*)[2])py_dd_q0.data();
    G_list = (double(*)[3])py_G_list.data();
    born = (double(*)[3][3])py_born.data();
    dielectric = (double(*)[3])py_dielectric.data();
    pos = (double(*)[3])py_positions.data();
    num_G = py_G_list.shape(0);
    num_patom = py_positions.shape(0);

    phpy_get_recip_dipole_dipole_q0(dd_q0,  /* [natom, 3, 3, (real, imag)] */
                                    G_list, /* [num_kvec, 3] */
                                    num_G, num_patom, born, dielectric,
                                    pos,    /* [natom, 3] */
                                    lambda, /* 4 * Lambda^2 */
                                    tolerance, use_openmp);
}

void py_get_derivative_dynmat(
    nb::ndarray<> py_derivative_dynmat, nb::ndarray<> py_force_constants,
    nb::ndarray<> py_q_vector, nb::ndarray<> py_lattice,
    nb::ndarray<> py_reclat, nb::ndarray<> py_svecs, nb::ndarray<> py_multi,
    nb::ndarray<> py_masses, nb::ndarray<> py_s2p_map, nb::ndarray<> py_p2s_map,
    double nac_factor, nb::ndarray<> py_born, nb::ndarray<> py_dielectric,
    nb::ndarray<> py_q_direction, long is_nac, long is_nac_q_zero,
    long use_openmp) {
    double(*ddm)[2];
    double *fc;
    double *q_vector;
    double *lattice;
    double *reclat;
    double(*svecs)[3];
    double *masses;
    long(*multi)[2];
    long *s2p_map;
    long *p2s_map;
    long num_patom;
    long num_satom;

    double *born;
    double *epsilon;
    double *q_dir;

    ddm = (double(*)[2])py_derivative_dynmat.data();
    fc = (double *)py_force_constants.data();
    q_vector = (double *)py_q_vector.data();
    lattice = (double *)py_lattice.data();
    reclat = (double *)py_reclat.data();
    svecs = (double(*)[3])py_svecs.data();
    masses = (double *)py_masses.data();
    multi = (long(*)[2])py_multi.data();
    s2p_map = (long *)py_s2p_map.data();
    p2s_map = (long *)py_p2s_map.data();
    num_patom = py_p2s_map.shape(0);
    num_satom = py_s2p_map.shape(0);

    epsilon = (double *)py_dielectric.data();
    born = (double *)py_born.data();
    if (is_nac_q_zero) {
        q_dir = NULL;
    } else {
        q_dir = (double *)py_q_direction.data();
    }

    phpy_get_derivative_dynmat_at_q(ddm, num_patom, num_satom, fc, q_vector,
                                    lattice, reclat, svecs, multi, masses,
                                    s2p_map, p2s_map, nac_factor, born, epsilon,
                                    q_dir, is_nac, use_openmp);
}

void py_get_thermal_properties(nb::ndarray<> py_thermal_props,
                               nb::ndarray<> py_temperatures,
                               nb::ndarray<> py_frequencies,
                               nb::ndarray<> py_weights,
                               double cutoff_frequency, int classical) {
    double *temperatures;
    double *freqs;
    double *thermal_props;
    long *weights;
    long num_qpoints;
    long num_bands;
    long num_temp;

    thermal_props = (double *)py_thermal_props.data();
    temperatures = (double *)py_temperatures.data();
    num_temp = (long)py_temperatures.shape(0);
    freqs = (double *)py_frequencies.data();
    num_qpoints = (long)py_frequencies.shape(0);
    weights = (long *)py_weights.data();
    num_bands = (long)py_frequencies.shape(1);

    phpy_get_thermal_properties(thermal_props, temperatures, freqs, weights,
                                num_temp, num_qpoints, num_bands,
                                cutoff_frequency, classical);
}

void py_distribute_fc2(nb::ndarray<> py_force_constants,
                       nb::ndarray<> py_atom_list,
                       nb::ndarray<> py_fc_indices_of_atom_list,
                       nb::ndarray<> py_rotations_cart,
                       nb::ndarray<> py_permutations,
                       nb::ndarray<> py_map_atoms, nb::ndarray<> py_map_syms) {
    double(*r_carts)[3][3];
    double(*fc2)[3][3];
    int *permutations;
    int *map_atoms;
    int *map_syms;
    int *atom_list;
    int *fc_indices_of_atom_list;
    long num_pos, num_rot, len_atom_list;

    fc2 = (double(*)[3][3])py_force_constants.data();
    atom_list = (int *)py_atom_list.data();
    len_atom_list = py_atom_list.shape(0);
    fc_indices_of_atom_list = (int *)py_fc_indices_of_atom_list.data();
    permutations = (int *)py_permutations.data();
    map_atoms = (int *)py_map_atoms.data();
    map_syms = (int *)py_map_syms.data();
    r_carts = (double(*)[3][3])py_rotations_cart.data();
    num_rot = py_permutations.shape(0);
    num_pos = py_permutations.shape(1);

    phpy_distribute_fc2(fc2, atom_list, len_atom_list, fc_indices_of_atom_list,
                        r_carts, permutations, map_atoms, map_syms, num_rot,
                        num_pos);
}

bool py_compute_permutation(nb::ndarray<> permutation, nb::ndarray<> lattice,
                            nb::ndarray<> positions,
                            nb::ndarray<> permuted_positions, double symprec) {
    int *rot_atoms;
    double(*lat)[3];
    double(*pos)[3];
    double(*rot_pos)[3];
    int num_pos;

    int is_found;

    rot_atoms = (int *)permutation.data();
    lat = (double(*)[3])lattice.data();
    pos = (double(*)[3])positions.data();
    rot_pos = (double(*)[3])permuted_positions.data();
    num_pos = positions.shape(0);

    is_found = phpy_compute_permutation(rot_atoms, lat, pos, rot_pos, num_pos,
                                        symprec);

    if (is_found) {
        return true;
    } else {
        return false;
    }
}

void py_gsv_set_smallest_vectors_sparse(
    nb::ndarray<> py_smallest_vectors, nb::ndarray<> py_multiplicity,
    nb::ndarray<> py_pos_to, nb::ndarray<> py_pos_from,
    nb::ndarray<> py_lattice_points, nb::ndarray<> py_reduced_basis,
    nb::ndarray<> py_trans_mat, double symprec) {
    double(*smallest_vectors)[27][3];
    int *multiplicity;
    double(*pos_to)[3];
    double(*pos_from)[3];
    int(*lattice_points)[3];
    double(*reduced_basis)[3];
    int(*trans_mat)[3];
    int num_pos_to, num_pos_from, num_lattice_points;

    smallest_vectors = (double(*)[27][3])py_smallest_vectors.data();
    multiplicity = (int *)py_multiplicity.data();
    pos_to = (double(*)[3])py_pos_to.data();
    pos_from = (double(*)[3])py_pos_from.data();
    num_pos_to = py_pos_to.shape(0);
    num_pos_from = py_pos_from.shape(0);
    lattice_points = (int(*)[3])py_lattice_points.data();
    num_lattice_points = py_lattice_points.shape(0);
    reduced_basis = (double(*)[3])py_reduced_basis.data();
    trans_mat = (int(*)[3])py_trans_mat.data();

    phpy_set_smallest_vectors_sparse(smallest_vectors, multiplicity, pos_to,
                                     num_pos_to, pos_from, num_pos_from,
                                     lattice_points, num_lattice_points,
                                     reduced_basis, trans_mat, symprec);
}

void py_gsv_set_smallest_vectors_dense(
    nb::ndarray<> py_smallest_vectors, nb::ndarray<> py_multiplicity,
    nb::ndarray<> py_pos_to, nb::ndarray<> py_pos_from,
    nb::ndarray<> py_lattice_points, nb::ndarray<> py_reduced_basis,
    nb::ndarray<> py_trans_mat, long initialize, double symprec) {
    double(*smallest_vectors)[3];
    long(*multiplicity)[2];
    double(*pos_to)[3];
    double(*pos_from)[3];
    long(*lattice_points)[3];
    double(*reduced_basis)[3];
    long(*trans_mat)[3];
    long num_pos_to, num_pos_from, num_lattice_points;

    smallest_vectors = (double(*)[3])py_smallest_vectors.data();
    multiplicity = (long(*)[2])py_multiplicity.data();
    pos_to = (double(*)[3])py_pos_to.data();
    pos_from = (double(*)[3])py_pos_from.data();
    num_pos_to = py_pos_to.shape(0);
    num_pos_from = py_pos_from.shape(0);
    lattice_points = (long(*)[3])py_lattice_points.data();
    num_lattice_points = py_lattice_points.shape(0);
    reduced_basis = (double(*)[3])py_reduced_basis.data();
    trans_mat = (long(*)[3])py_trans_mat.data();

    phpy_set_smallest_vectors_dense(
        smallest_vectors, multiplicity, pos_to, num_pos_to, pos_from,
        num_pos_from, lattice_points, num_lattice_points, reduced_basis,
        trans_mat, initialize, symprec);
}

void py_thm_relative_grid_address(nb::ndarray<> py_relative_grid_address,
                                  nb::ndarray<> py_reciprocal_lattice_py) {
    long(*relative_grid_address)[4][3];
    double(*reciprocal_lattice)[3];

    relative_grid_address = (long(*)[4][3])py_relative_grid_address.data();
    reciprocal_lattice = (double(*)[3])py_reciprocal_lattice_py.data();

    phpy_get_relative_grid_address(relative_grid_address, reciprocal_lattice);
}

void py_thm_all_relative_grid_address(nb::ndarray<> py_relative_grid_address) {
    long(*relative_grid_address)[24][4][3];

    relative_grid_address = (long(*)[24][4][3])py_relative_grid_address.data();

    phpy_get_all_relative_grid_address(relative_grid_address);
}

double py_thm_integration_weight(double omega,
                                 nb::ndarray<> py_tetrahedra_omegas,
                                 const char *function) {
    double(*tetrahedra_omegas)[4];
    double iw;

    tetrahedra_omegas = (double(*)[4])py_tetrahedra_omegas.data();

    iw = phpy_get_integration_weight(omega, tetrahedra_omegas, function[0]);

    return iw;
}

void py_thm_integration_weight_at_omegas(nb::ndarray<> py_integration_weights,
                                         nb::ndarray<> py_omegas,
                                         nb::ndarray<> py_tetrahedra_omegas,
                                         const char *function) {
    double *omegas;
    double *iw;
    long num_omegas;
    double(*tetrahedra_omegas)[4];

    long i;

    omegas = (double *)py_omegas.data();
    iw = (double *)py_integration_weights.data();
    num_omegas = (long)py_omegas.shape(0);
    tetrahedra_omegas = (double(*)[4])py_tetrahedra_omegas.data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_omegas; i++) {
        iw[i] = phpy_get_integration_weight(omegas[i], tetrahedra_omegas,
                                            function[0]);
    }
}

void py_get_tetrahedra_frequenies(nb::ndarray<> py_freq_tetras,
                                  nb::ndarray<> py_grid_points,
                                  nb::ndarray<> py_mesh,
                                  nb::ndarray<> py_grid_address,
                                  nb::ndarray<> py_gp_ir_index,
                                  nb::ndarray<> py_relative_grid_address,
                                  nb::ndarray<> py_frequencies) {
    double *freq_tetras;
    long *grid_points;
    long *mesh;
    long(*grid_address)[3];
    long *gp_ir_index;
    long(*relative_grid_address)[3];
    double *frequencies;

    long num_gp_in, num_band;

    freq_tetras = (double *)py_freq_tetras.data();
    grid_points = (long *)py_grid_points.data();
    num_gp_in = py_grid_points.shape(0);
    mesh = (long *)py_mesh.data();
    grid_address = (long(*)[3])py_grid_address.data();
    gp_ir_index = (long *)py_gp_ir_index.data();
    relative_grid_address = (long(*)[3])py_relative_grid_address.data();
    frequencies = (double *)py_frequencies.data();
    num_band = py_frequencies.shape(1);

    phpy_get_tetrahedra_frequenies(freq_tetras, mesh, grid_points, grid_address,
                                   relative_grid_address, gp_ir_index,
                                   frequencies, num_band, num_gp_in);
}

void py_tetrahedron_method_dos(nb::ndarray<> py_dos, nb::ndarray<> py_mesh,
                               nb::ndarray<> py_freq_points,
                               nb::ndarray<> py_frequencies,
                               nb::ndarray<> py_coef,
                               nb::ndarray<> py_grid_address,
                               nb::ndarray<> py_grid_mapping_table,
                               nb::ndarray<> py_relative_grid_address) {
    double *dos;
    long *mesh;
    double *freq_points;
    double *frequencies;
    double *coef;
    long(*grid_address)[3];
    long num_gp, num_ir_gp, num_band, num_freq_points, num_coef;
    long *grid_mapping_table;
    long(*relative_grid_address)[4][3];

    /* dos[num_ir_gp][num_band][num_freq_points][num_coef] */
    dos = (double *)py_dos.data();
    mesh = (long *)py_mesh.data();
    freq_points = (double *)py_freq_points.data();
    num_freq_points = (long)py_freq_points.shape(0);
    frequencies = (double *)py_frequencies.data();
    num_ir_gp = (long)py_frequencies.shape(0);
    num_band = (long)py_frequencies.shape(1);
    coef = (double *)py_coef.data();
    num_coef = (long)py_coef.shape(1);
    grid_address = (long(*)[3])py_grid_address.data();
    num_gp = (long)py_grid_address.shape(0);
    grid_mapping_table = (long *)py_grid_mapping_table.data();
    relative_grid_address = (long(*)[4][3])py_relative_grid_address.data();

    phpy_tetrahedron_method_dos(dos, mesh, grid_address, relative_grid_address,
                                grid_mapping_table, freq_points, frequencies,
                                coef, num_freq_points, num_ir_gp, num_band,
                                num_coef, num_gp);
}

NB_MODULE(_phonopy, m) {
    m.def("transform_dynmat_to_fc", &py_transform_dynmat_to_fc);
    m.def("perm_trans_symmetrize_fc", &py_perm_trans_symmetrize_fc);
    m.def("perm_trans_symmetrize_compact_fc",
          &py_perm_trans_symmetrize_compact_fc);
    m.def("transpose_compact_fc", &py_transpose_compact_fc);
    m.def("dynamical_matrices_with_dd_openmp_over_qpoints",
          &py_get_dynamical_matrices_with_dd_openmp_over_qpoints);
    m.def("recip_dipole_dipole", &py_get_recip_dipole_dipole);
    m.def("recip_dipole_dipole_q0", &py_get_recip_dipole_dipole_q0);
    m.def("derivative_dynmat", &py_get_derivative_dynmat);
    m.def("thermal_properties", &py_get_thermal_properties);
    m.def("distribute_fc2", &py_distribute_fc2);
    m.def("compute_permutation", &py_compute_permutation);
    m.def("gsv_set_smallest_vectors_sparse",
          &py_gsv_set_smallest_vectors_sparse);
    m.def("gsv_set_smallest_vectors_dense", &py_gsv_set_smallest_vectors_dense);
    m.def("tetrahedra_relative_grid_address", &py_thm_relative_grid_address);
    m.def("all_tetrahedra_relative_grid_address",
          &py_thm_all_relative_grid_address);
    m.def("tetrahedra_integration_weight", &py_thm_integration_weight);
    m.def("tetrahedra_integration_weight_at_omegas",
          &py_thm_integration_weight_at_omegas);
    m.def("tetrahedra_frequencies", &py_get_tetrahedra_frequenies);
    m.def("tetrahedron_method_dos", &py_tetrahedron_method_dos);
    m.def("use_openmp", &phpy_use_openmp);
    m.def("omp_max_threads", &phpy_get_max_threads);
}
