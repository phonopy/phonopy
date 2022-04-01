/* Copyright (C) 2021 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include <stddef.h>

#ifndef __phonopy_H__
#define __phonopy_H__

#ifndef PHPYCONST
#define PHPYCONST
#endif

void phpy_transform_dynmat_to_fc(double *fc, const double *dm,
                                 const double (*comm_points)[3],
                                 const double (*svecs)[3],
                                 const long (*multi)[2], const double *masses,
                                 const long *s2pp_map, const long *fc_index_map,
                                 const long num_patom, const long num_satom);
long phpy_get_dynamical_matrix_at_q(double *dynamical_matrix,
                                    const long num_patom, const long num_satom,
                                    const double *fc, const double q[3],
                                    const double (*svecs)[3],
                                    const long (*multi)[2], const double *mass,
                                    const long *s2p_map, const long *p2s_map,
                                    const double (*charge_sum)[3][3],
                                    const long with_openmp);
void phpy_get_charge_sum(
    double (*charge_sum)[3][3], const long num_patom,
    const double factor, /* 4pi/V*unit-conv and denominator */
    const double q_cart[3], const double (*born)[3][3]);
void phpy_get_recip_dipole_dipole(
    double *dd,                /* [natom, 3, natom, 3, (real,imag)] */
    const double *dd_q0,       /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double q_cart[3],
    const double *q_direction_cart, /* must be pointer */
    const double (*born)[3][3], const double dielectric[3][3],
    const double (*pos)[3], /* [num_patom, 3] */
    const double factor,    /* 4pi/V*unit-conv */
    const double lambda, const double tolerance);
void phpy_get_recip_dipole_dipole_q0(
    double *dd_q0,             /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double (*born)[3][3],
    const double dielectric[3][3], const double (*pos)[3], /* [num_patom, 3] */
    const double lambda, const double tolerance);
void phpy_get_derivative_dynmat_at_q(
    double *derivative_dynmat, const long num_patom, const long num_satom,
    const double *fc, const double *q,
    const double *lattice, /* column vector */
    const double (*svecs)[3], const long (*multi)[2], const double *mass,
    const long *s2p_map, const long *p2s_map, const double nac_factor,
    const double *born, const double *dielectric, const double *q_direction);
void phpy_get_neighboring_grid_points(size_t neighboring_grid_points[],
                                      const size_t grid_point,
                                      PHPYCONST int relative_grid_address[][3],
                                      const int num_relative_grid_address,
                                      const int mesh[3],
                                      PHPYCONST int bz_grid_address[][3],
                                      const size_t bz_map[]);
void phpy_get_relative_grid_address(long relative_grid_address[24][4][3],
                                    PHPYCONST double reciprocal_lattice[3][3]);
void phpy_get_all_relative_grid_address(
    long relative_grid_address[4][24][4][3]);
double phpy_get_integration_weight(const double omega,
                                   PHPYCONST double tetrahedra_omegas[24][4],
                                   const char function);
void phpy_get_tetrahedra_frequenies(double *freq_tetras, const long mesh[3],
                                    const long *grid_points,
                                    PHPYCONST long (*grid_address)[3],
                                    PHPYCONST long (*relative_grid_address)[3],
                                    const long *gp_ir_index,
                                    const double *frequencies,
                                    const long num_band, const long num_gp);
void phpy_tetrahedron_method_dos(double *dos, const long mesh[3],
                                 PHPYCONST long (*grid_address)[3],
                                 PHPYCONST long (*relative_grid_address)[4][3],
                                 const long *grid_mapping_table,
                                 const double *freq_points,
                                 const double *frequencies, const double *coef,
                                 const long num_freq_points,
                                 const long num_ir_gp, const long num_band,
                                 const long num_coef, const long num_gp);
void phpy_get_thermal_properties(double *thermal_props,
                                 const double *temperatures,
                                 const double *freqs, const long *weights,
                                 const long num_temp, const long num_qpoints,
                                 const long num_bands,
                                 const double cutoff_frequency);
void phpy_distribute_fc2(double (*fc2)[3][3], const int *atom_list,
                         const int len_atom_list,
                         PHPYCONST double (*r_carts)[3][3],
                         const int *permutations, const int *map_atoms,
                         const int *map_syms, const int num_rot,
                         const int num_pos);
int phpy_compute_permutation(int *rot_atom, PHPYCONST double lat[3][3],
                             PHPYCONST double (*pos)[3],
                             PHPYCONST double (*rot_pos)[3], const int num_pos,
                             const double symprec);
void phpy_set_smallest_vectors_sparse(
    double (*smallest_vectors)[27][3], int *multiplicity,
    PHPYCONST double (*pos_to)[3], const int num_pos_to,
    PHPYCONST double (*pos_from)[3], const int num_pos_from,
    PHPYCONST int (*lattice_points)[3], const int num_lattice_points,
    PHPYCONST double reduced_basis[3][3], PHPYCONST int trans_mat[3][3],
    const double symprec);
void phpy_set_smallest_vectors_dense(
    double (*smallest_vectors)[3], long (*multiplicity)[2],
    PHPYCONST double (*pos_to)[3], const long num_pos_to,
    PHPYCONST double (*pos_from)[3], const long num_pos_from,
    PHPYCONST long (*lattice_points)[3], const long num_lattice_points,
    PHPYCONST double reduced_basis[3][3], PHPYCONST long trans_mat[3][3],
    const long initialize, const double symprec);

void phpy_perm_trans_symmetrize_fc(double *fc, const int nsatom,
                                   const int level);
void phpy_perm_trans_symmetrize_compact_fc(double *fc, const int p2s[],
                                           const int s2pp[],
                                           const int nsym_list[],
                                           const int perms[], const int n_satom,
                                           const int n_patom, const int level);
void phpy_set_index_permutation_symmetry_compact_fc(
    double *fc, const int p2s[], const int s2pp[], const int nsym_list[],
    const int perms[], const int n_satom, const int n_patom,
    const int is_transpose);

#endif
