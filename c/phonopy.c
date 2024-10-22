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

#include "phonopy.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "derivative_dynmat.h"
#include "dynmat.h"
#include "rgrid.h"
#include "tetrahedron_method.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define KB 8.6173382568083159E-05

static void set_index_permutation_symmetry_fc(double *fc, const int natom);
static void set_translational_symmetry_fc(double *fc, const int natom);
static void set_translational_symmetry_compact_fc(double *fc, const int p2s[],
                                                  const int n_satom,
                                                  const int n_patom);
static double get_free_energy(const double temperature,
                              const double f,
                              const int classical);
static double get_entropy(const double temperature,
                          const double f,
                          const int classical);
static double get_heat_capacity(const double temperature,
                                const double f,
                                const int classical);
/* static double get_energy(double temperature, double f); */
static void distribute_fc2(double (*fc2)[3][3], const int *atom_list,
                           const int len_atom_list,
                           const int *fc_indices_of_atom_list,
                           const double (*r_carts)[3][3],
                           const int *permutations, const int *map_atoms,
                           const int *map_syms, const int num_rot,
                           const int num_pos);
static int nint(const double a);

void phpy_transform_dynmat_to_fc(double *fc, const double (*dm)[2],
                                 const double (*comm_points)[3],
                                 const double (*svecs)[3],
                                 const long (*multi)[2], const double *masses,
                                 const long *s2pp_map, const long *fc_index_map,
                                 const long num_patom, const long num_satom,
                                 const long use_openmp) {
    dym_transform_dynmat_to_fc(fc, dm, comm_points, svecs, multi, masses,
                               s2pp_map, fc_index_map, num_patom, num_satom,
                               use_openmp);
}

long phpy_dynamical_matrices_with_dd_openmp_over_qpoints(
    double (*dynamical_matrices)[2], const double (*qpoints)[3],
    const long n_qpoints, const double *fc, const double (*svecs)[3],
    const long (*multi)[2], const double (*positions)[3], const long num_patom,
    const long num_satom, const double *masses, const long *p2s_map,
    const long *s2p_map, const double (*born)[3][3],
    const double dielectric[3][3], const double (*reciprocal_lattice)[3],
    const double *q_direction, const double nac_factor,
    const double (*dd_q0)[2], const double (*G_list)[3],
    const long num_G_points, const double lambda, const long use_Wang_NAC) {
    return dym_dynamical_matrices_with_dd_openmp_over_qpoints(
        dynamical_matrices, qpoints, n_qpoints, fc, svecs, multi, positions,
        num_patom, num_satom, masses, p2s_map, s2p_map, born, dielectric,
        reciprocal_lattice, q_direction, nac_factor, dd_q0, G_list,
        num_G_points, lambda, use_Wang_NAC);
}

void phpy_get_charge_sum(
    double (*charge_sum)[3][3], const long num_patom,
    const double factor, /* 4pi/V*unit-conv and denominator */
    const double q_cart[3], const double (*born)[3][3]) {
    dym_get_charge_sum(charge_sum, num_patom, factor, q_cart, born);
}

void phpy_get_recip_dipole_dipole(
    double (*dd)[2],           /* [natom, 3, natom, 3, (real,imag)] */
    const double (*dd_q0)[2],  /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double q_cart[3],
    const double *q_direction_cart, /* must be pointer */
    const double (*born)[3][3], const double dielectric[3][3],
    const double (*pos)[3], /* [num_patom, 3] */
    const double factor,    /* 4pi/V*unit-conv */
    const double lambda, const double tolerance, const long use_openmp) {
    dym_get_recip_dipole_dipole(dd, dd_q0, G_list, num_G, num_patom, q_cart,
                                q_direction_cart, born, dielectric, pos, factor,
                                lambda, tolerance, use_openmp);
}

void phpy_get_recip_dipole_dipole_q0(
    double (*dd_q0)[2],        /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double (*born)[3][3],
    const double dielectric[3][3], const double (*pos)[3], /* [num_patom, 3] */
    const double lambda, const double tolerance, const long use_openmp) {
    dym_get_recip_dipole_dipole_q0(dd_q0, G_list, num_G, num_patom, born,
                                   dielectric, pos, lambda, tolerance,
                                   use_openmp);
}

void phpy_get_derivative_dynmat_at_q(
    double (*derivative_dynmat)[2], const long num_patom, const long num_satom,
    const double *fc, const double *q,
    const double *lattice, /* column vector */
    const double *reclat,  /* column vector */
    const double (*svecs)[3], const long (*multi)[2], const double *mass,
    const long *s2p_map, const long *p2s_map, const double nac_factor,
    const double *born, const double *dielectric, const double *q_direction,
    const long is_nac, const long use_openmp) {
    ddm_get_derivative_dynmat_at_q(derivative_dynmat, num_patom, num_satom, fc,
                                   q, lattice, reclat, svecs, multi, mass,
                                   s2p_map, p2s_map, nac_factor, born,
                                   dielectric, q_direction, is_nac, use_openmp);
}

void phpy_get_relative_grid_address(long relative_grid_address[24][4][3],
                                    const double reciprocal_lattice[3][3]) {
    thm_get_relative_grid_address(relative_grid_address, reciprocal_lattice);
}

void phpy_get_all_relative_grid_address(
    long relative_grid_address[4][24][4][3]) {
    thm_get_all_relative_grid_address(relative_grid_address);
}

double phpy_get_integration_weight(const double omega,
                                   const double tetrahedra_omegas[24][4],
                                   const char function) {
    return thm_get_integration_weight(omega, tetrahedra_omegas, function);
}

void phpy_get_tetrahedra_frequenies(double *freq_tetras, const long mesh[3],
                                    const long *grid_points,
                                    const long (*grid_address)[3],
                                    const long (*relative_grid_address)[3],
                                    const long *gp_ir_index,
                                    const double *frequencies,
                                    const long num_band, const long num_gp) {
    long is_shift[3] = {0, 0, 0};
    long i, j, k, gp;
    long g_addr[3];
    long address_double[3];

    /* relative_grid_address[4, 24, 3] is viewed as [96, 3]. */
    for (i = 0; i < num_gp; i++) {
#ifdef _OPENMP
#pragma omp parallel for private(k, g_addr, gp, address_double)
#endif
        for (j = 0; j < num_band * 96; j++) {
            for (k = 0; k < 3; k++) {
                g_addr[k] = grid_address[grid_points[i]][k] +
                            relative_grid_address[j % 96][k];
            }
            rgd_get_double_grid_address(address_double, g_addr, mesh, is_shift);
            gp = rgd_get_double_grid_index(address_double, mesh);
            freq_tetras[i * num_band * 96 + j] =
                frequencies[gp_ir_index[gp] * num_band + j / 96];
        }
    }
}

void phpy_tetrahedron_method_dos(
    double *dos, const long mesh[3], const long (*grid_address)[3],
    const long (*relative_grid_address)[4][3], const long *grid_mapping_table,
    const double *freq_points, const double *frequencies, const double *coef,
    const long num_freq_points, const long num_ir_gp, const long num_band,
    const long num_coef, const long num_gp) {
    long is_shift[3] = {0, 0, 0};
    long i, j, k, l, m, q, r, count;
    long ir_gps[24][4];
    long g_addr[3];
    double tetrahedra[24][4];
    long address_double[3];
    long *gp2ir, *ir_grid_points;
    long *weights;
    double iw;

    gp2ir = NULL;
    ir_grid_points = NULL;
    weights = NULL;

    gp2ir = (long *)malloc(sizeof(long) * num_gp);
    ir_grid_points = (long *)malloc(sizeof(long) * num_ir_gp);
    weights = (long *)malloc(sizeof(long) * num_ir_gp);

    count = 0;
    for (i = 0; i < num_gp; i++) {
        if (grid_mapping_table[i] == i) {
            gp2ir[i] = count;
            ir_grid_points[count] = i;
            weights[count] = 1;
            count++;
        } else {
            gp2ir[i] = gp2ir[grid_mapping_table[i]];
            weights[gp2ir[i]]++;
        }
    }

    if (num_ir_gp != count) {
        printf("Something is wrong!\n");
    }

#ifdef _OPENMP
#pragma omp parallel for private(j, k, l, m, q, r, iw, ir_gps, g_addr, \
                                     tetrahedra, address_double)
#endif
    for (i = 0; i < num_ir_gp; i++) {
        /* set 24 tetrahedra */
        for (l = 0; l < 24; l++) {
            for (q = 0; q < 4; q++) {
                for (r = 0; r < 3; r++) {
                    g_addr[r] = grid_address[ir_grid_points[i]][r] +
                                relative_grid_address[l][q][r];
                }
                rgd_get_double_grid_address(address_double, g_addr, mesh,
                                            is_shift);
                ir_gps[l][q] =
                    gp2ir[rgd_get_double_grid_index(address_double, mesh)];
            }
        }

        for (k = 0; k < num_band; k++) {
            for (l = 0; l < 24; l++) {
                for (q = 0; q < 4; q++) {
                    tetrahedra[l][q] = frequencies[ir_gps[l][q] * num_band + k];
                }
            }
            for (j = 0; j < num_freq_points; j++) {
                iw = thm_get_integration_weight(freq_points[j], tetrahedra,
                                                'I') *
                     weights[i];
                for (m = 0; m < num_coef; m++) {
                    dos[i * num_band * num_freq_points * num_coef +
                        k * num_coef * num_freq_points + j * num_coef + m] +=
                        iw * coef[i * num_coef * num_band + m * num_band + k];
                }
            }
        }
    }

    free(gp2ir);
    gp2ir = NULL;
    free(ir_grid_points);
    ir_grid_points = NULL;
    free(weights);
    weights = NULL;
}

void phpy_get_thermal_properties(double *thermal_props,
                                 const double *temperatures,
                                 const double *freqs, const long *weights,
                                 const long num_temp, const long num_qpoints,
                                 const long num_bands,
                                 const double cutoff_frequency,
                                 const int classical
) {
    long i, j, k;
    double f;
    double *tp;

    tp = (double *)malloc(sizeof(double) * num_qpoints * num_temp * 3);
    for (i = 0; i < num_qpoints * num_temp * 3; i++) {
        tp[i] = 0;
    }

#ifdef _OPENMP
#pragma omp parallel for private(j, k, f)
#endif
    for (i = 0; i < num_qpoints; i++) {
        for (j = 0; j < num_temp; j++) {
            for (k = 0; k < num_bands; k++) {
                f = freqs[i * num_bands + k];
                if (temperatures[j] > 0 && f > cutoff_frequency) {
                    tp[i * num_temp * 3 + j * 3] +=
                        get_free_energy(temperatures[j], f, classical) * weights[i];
                    tp[i * num_temp * 3 + j * 3 + 1] +=
                        get_entropy(temperatures[j], f, classical) * weights[i];
                    tp[i * num_temp * 3 + j * 3 + 2] +=
                        get_heat_capacity(temperatures[j], f, classical) * weights[i];
                }
            }
        }
    }

    for (i = 0; i < num_qpoints; i++) {
        for (j = 0; j < num_temp * 3; j++) {
            thermal_props[j] += tp[i * num_temp * 3 + j];
        }
    }

    free(tp);
    tp = NULL;
}

void phpy_distribute_fc2(double (*fc2)[3][3], const int *atom_list,
                         const int len_atom_list,
                         const int *fc_indices_of_atom_list,
                         const double (*r_carts)[3][3], const int *permutations,
                         const int *map_atoms, const int *map_syms,
                         const int num_rot, const int num_pos) {
    distribute_fc2(fc2, atom_list, len_atom_list, fc_indices_of_atom_list,
                   r_carts, permutations, map_atoms, map_syms, num_rot,
                   num_pos);
}

int phpy_compute_permutation(int *rot_atom, const double lat[3][3],
                             const double (*pos)[3], const double (*rot_pos)[3],
                             const int num_pos, const double symprec) {
    int i, j, k, l;
    int search_start;
    double distance2, diff_cart;
    double diff[3];

    for (i = 0; i < num_pos; i++) {
        rot_atom[i] = -1;
    }

    /* optimization: Iterate primarily by pos instead of rot_pos. */
    /*  (find where 0 belongs in rot_atom, then where 1 belongs, etc.) */
    /*  Then track the first unassigned index. */
    /* */
    /* This works best if the permutation is close to the identity. */
    /* (more specifically, if the max value of 'rot_atom[i] - i' is small)
     */
    search_start = 0;
    for (i = 0; i < num_pos; i++) {
        while (rot_atom[search_start] >= 0) {
            search_start++;
        }
        for (j = search_start; j < num_pos; j++) {
            if (rot_atom[j] >= 0) {
                continue;
            }

            for (k = 0; k < 3; k++) {
                diff[k] = pos[i][k] - rot_pos[j][k];
                diff[k] -= nint(diff[k]);
            }
            distance2 = 0;
            for (k = 0; k < 3; k++) {
                diff_cart = 0;
                for (l = 0; l < 3; l++) {
                    diff_cart += lat[k][l] * diff[l];
                }
                distance2 += diff_cart * diff_cart;
            }

            if (sqrt(distance2) < symprec) {
                rot_atom[j] = i;
                break;
            }
        }
    }

    for (i = 0; i < num_pos; i++) {
        if (rot_atom[i] < 0) {
            return 0;
        }
    }
    return 1;
}

void phpy_set_smallest_vectors_sparse(
    double (*smallest_vectors)[27][3], int *multiplicity,
    const double (*pos_to)[3], const int num_pos_to,
    const double (*pos_from)[3], const int num_pos_from,
    const int (*lattice_points)[3], const int num_lattice_points,
    const double reduced_basis[3][3], const int trans_mat[3][3],
    const double symprec) {
    int i, j, k, l, count;
    double length_tmp, minimum, vec_xyz;
    double *length;
    double(*vec)[3];

    length = (double *)malloc(sizeof(double) * num_lattice_points);
    vec = (double(*)[3])malloc(sizeof(double[3]) * num_lattice_points);

    for (i = 0; i < num_pos_to; i++) {
        for (j = 0; j < num_pos_from; j++) {
            for (k = 0; k < num_lattice_points; k++) {
                length[k] = 0;
                for (l = 0; l < 3; l++) {
                    vec[k][l] =
                        pos_to[i][l] - pos_from[j][l] + lattice_points[k][l];
                }
                for (l = 0; l < 3; l++) {
                    length_tmp = (reduced_basis[l][0] * vec[k][0] +
                                  reduced_basis[l][1] * vec[k][1] +
                                  reduced_basis[l][2] * vec[k][2]);
                    length[k] += length_tmp * length_tmp;
                }
                length[k] = sqrt(length[k]);
            }

            minimum = DBL_MAX;
            for (k = 0; k < num_lattice_points; k++) {
                if (length[k] < minimum) {
                    minimum = length[k];
                }
            }

            count = 0;
            for (k = 0; k < num_lattice_points; k++) {
                if (length[k] - minimum < symprec) {
                    for (l = 0; l < 3; l++) {
                        /* Transform back to supercell coordinates */
                        vec_xyz = (trans_mat[l][0] * vec[k][0] +
                                   trans_mat[l][1] * vec[k][1] +
                                   trans_mat[l][2] * vec[k][2]);
                        smallest_vectors[i * num_pos_from + j][count][l] =
                            vec_xyz;
                    }
                    count++;
                }
            }
            if (count > 27) { /* should not be greater than 27 */
                printf("Warning (gsv_set_smallest_vectors_sparse): ");
                printf("number of shortest vectors is out of range,\n");
                break;
            } else {
                multiplicity[i * num_pos_from + j] = count;
            }
        }
    }

    free(length);
    length = NULL;
    free(vec);
    vec = NULL;
}

void phpy_set_smallest_vectors_dense(
    double (*smallest_vectors)[3], long (*multiplicity)[2],
    const double (*pos_to)[3], const long num_pos_to,
    const double (*pos_from)[3], const long num_pos_from,
    const long (*lattice_points)[3], const long num_lattice_points,
    const double reduced_basis[3][3], const long trans_mat[3][3],
    const long initialize, const double symprec) {
    long i, j, k, l, count, adrs;
    double length_tmp, minimum, vec_xyz;
    double *length;
    double(*vec)[3];

    length = (double *)malloc(sizeof(double) * num_lattice_points);
    vec = (double(*)[3])malloc(sizeof(double[3]) * num_lattice_points);

    adrs = 0;

    for (i = 0; i < num_pos_to; i++) {
        for (j = 0; j < num_pos_from; j++) {
            for (k = 0; k < num_lattice_points; k++) {
                length[k] = 0;
                for (l = 0; l < 3; l++) {
                    vec[k][l] =
                        pos_to[i][l] - pos_from[j][l] + lattice_points[k][l];
                }
                for (l = 0; l < 3; l++) {
                    length_tmp = (reduced_basis[l][0] * vec[k][0] +
                                  reduced_basis[l][1] * vec[k][1] +
                                  reduced_basis[l][2] * vec[k][2]);
                    length[k] += length_tmp * length_tmp;
                }
                length[k] = sqrt(length[k]);
            }

            minimum = DBL_MAX;
            for (k = 0; k < num_lattice_points; k++) {
                if (length[k] < minimum) {
                    minimum = length[k];
                }
            }

            count = 0;
            for (k = 0; k < num_lattice_points; k++) {
                if (length[k] - minimum < symprec) {
                    if (!initialize) {
                        for (l = 0; l < 3; l++) {
                            /* Transform back to supercell coordinates */
                            vec_xyz = (trans_mat[l][0] * vec[k][0] +
                                       trans_mat[l][1] * vec[k][1] +
                                       trans_mat[l][2] * vec[k][2]);
                            smallest_vectors[adrs + count][l] = vec_xyz;
                        }
                    }
                    count++;
                }
            }
            if (initialize) {
                multiplicity[i * num_pos_from + j][0] = count;
                multiplicity[i * num_pos_from + j][1] = adrs;
            }
            adrs += count;
        }
    }

    free(length);
    length = NULL;
    free(vec);
    vec = NULL;
}

void phpy_perm_trans_symmetrize_fc(double *fc, const int n_satom,
                                   const int level) {
    int i, j, k, l, iter;
    double sum;

    for (iter = 0; iter < level; iter++) {
        /* Subtract drift along column */
        for (j = 0; j < n_satom; j++) {
            for (k = 0; k < 3; k++) {
                for (l = 0; l < 3; l++) {
                    sum = 0;
                    for (i = 0; i < n_satom; i++) {
                        sum += fc[i * n_satom * 9 + j * 9 + k * 3 + l];
                    }
                    sum /= n_satom;
                    for (i = 0; i < n_satom; i++) {
                        fc[i * n_satom * 9 + j * 9 + k * 3 + l] -= sum;
                    }
                }
            }
        }
        /* Subtract drift along row */
        for (i = 0; i < n_satom; i++) {
            for (k = 0; k < 3; k++) {
                for (l = 0; l < 3; l++) {
                    sum = 0;
                    for (j = 0; j < n_satom; j++) {
                        sum += fc[i * n_satom * 9 + j * 9 + k * 3 + l];
                    }
                    sum /= n_satom;
                    for (j = 0; j < n_satom; j++) {
                        fc[i * n_satom * 9 + j * 9 + k * 3 + l] -= sum;
                    }
                }
            }
        }
        set_index_permutation_symmetry_fc(fc, n_satom);
    }
    set_translational_symmetry_fc(fc, n_satom);
}

void phpy_perm_trans_symmetrize_compact_fc(double *fc, const int p2s[],
                                           const int s2pp[],
                                           const int nsym_list[],
                                           const int perms[], const int n_satom,
                                           const int n_patom, const int level) {
    int i, j, k, l, n, iter;
    double sum;

    for (iter = 0; iter < level; iter++) {
        for (n = 0; n < 2; n++) {
            /* transpose only */
            phpy_set_index_permutation_symmetry_compact_fc(
                fc, p2s, s2pp, nsym_list, perms, n_satom, n_patom, 1);
            for (i = 0; i < n_patom; i++) {
                for (k = 0; k < 3; k++) {
                    for (l = 0; l < 3; l++) {
                        sum = 0;
                        for (j = 0; j < n_satom; j++) {
                            sum += fc[i * n_satom * 9 + j * 9 + k * 3 + l];
                        }
                        sum /= n_satom;
                        for (j = 0; j < n_satom; j++) {
                            fc[i * n_satom * 9 + j * 9 + k * 3 + l] -= sum;
                        }
                    }
                }
            }
        }

        phpy_set_index_permutation_symmetry_compact_fc(
            fc, p2s, s2pp, nsym_list, perms, n_satom, n_patom, 0);
    }

    set_translational_symmetry_compact_fc(fc, p2s, n_satom, n_patom);
}

void phpy_set_index_permutation_symmetry_compact_fc(
    double *fc, const int p2s[], const int s2pp[], const int nsym_list[],
    const int perms[], const int n_satom, const int n_patom,
    const int is_transpose) {
    int i, j, k, l, m, n, i_p, j_p, i_trans;
    double fc_elem;
    char *done;

    done = NULL;
    done = (char *)malloc(sizeof(char) * n_satom * n_patom);
    for (i = 0; i < n_satom * n_patom; i++) {
        done[i] = 0;
    }

    for (j = 0; j < n_satom; j++) {
        j_p = s2pp[j];
        for (i_p = 0; i_p < n_patom; i_p++) {
            i = p2s[i_p];
            if (i == j) { /* diagnoal part */
                for (k = 0; k < 3; k++) {
                    for (l = 0; l < 3; l++) {
                        if (l > k) {
                            m = i_p * n_satom * 9 + i * 9 + k * 3 + l;
                            n = i_p * n_satom * 9 + i * 9 + l * 3 + k;
                            if (is_transpose) {
                                fc_elem = fc[m];
                                fc[m] = fc[n];
                                fc[n] = fc_elem;
                            } else {
                                fc[m] = (fc[m] + fc[n]) / 2;
                                fc[n] = fc[m];
                            }
                        }
                    }
                }
            }
            if (!done[i_p * n_satom + j]) {
                /* (j, i) -- nsym_list[j] --> (j', i') */
                /* nsym_list[j] translates j to j' where j' is in */
                /* primitive cell. The same translation sends i to i' */
                /* where i' is not necessarily to be in primitive cell. */
                /* Thus, i' = perms[nsym_list[j] * n_satom + i] */
                i_trans = perms[nsym_list[j] * n_satom + i];
                done[i_p * n_satom + j] = 1;
                done[j_p * n_satom + i_trans] = 1;
                for (k = 0; k < 3; k++) {
                    for (l = 0; l < 3; l++) {
                        m = i_p * n_satom * 9 + j * 9 + k * 3 + l;
                        n = j_p * n_satom * 9 + i_trans * 9 + l * 3 + k;
                        if (is_transpose) {
                            fc_elem = fc[m];
                            fc[m] = fc[n];
                            fc[n] = fc_elem;
                        } else {
                            fc[m] = (fc[n] + fc[m]) / 2;
                            fc[n] = fc[m];
                        }
                    }
                }
            }
        }
    }

    free(done);
    done = NULL;
}

long phpy_use_openmp(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

long phpy_get_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

static void set_index_permutation_symmetry_fc(double *fc, const int natom) {
    int i, j, k, l, m, n;

    for (i = 0; i < natom; i++) {
        /* non diagonal part */
        for (j = i + 1; j < natom; j++) {
            for (k = 0; k < 3; k++) {
                for (l = 0; l < 3; l++) {
                    m = i * natom * 9 + j * 9 + k * 3 + l;
                    n = j * natom * 9 + i * 9 + l * 3 + k;
                    fc[m] += fc[n];
                    fc[m] /= 2;
                    fc[n] = fc[m];
                }
            }
        }

        /* diagnoal part */
        for (k = 0; k < 2; k++) {
            for (l = k + 1; l < 3; l++) {
                m = i * natom * 9 + i * 9 + k * 3 + l;
                n = i * natom * 9 + i * 9 + l * 3 + k;
                fc[m] += fc[n];
                fc[m] /= 2;
                fc[n] = fc[m];
            }
        }
    }
}

static void set_translational_symmetry_fc(double *fc, const int natom) {
    int i, j, k, l, m;
    double sums[3][3];

    for (i = 0; i < natom; i++) {
        for (k = 0; k < 3; k++) {
            for (l = 0; l < 3; l++) {
                sums[k][l] = 0;
                m = i * natom * 9 + k * 3 + l;
                for (j = 0; j < natom; j++) {
                    if (i != j) {
                        sums[k][l] += fc[m];
                    }
                    m += 9;
                }
            }
        }
        for (k = 0; k < 3; k++) {
            for (l = 0; l < 3; l++) {
                fc[i * natom * 9 + i * 9 + k * 3 + l] =
                    -(sums[k][l] + sums[l][k]) / 2;
            }
        }
    }
}

static void set_translational_symmetry_compact_fc(double *fc, const int p2s[],
                                                  const int n_satom,
                                                  const int n_patom) {
    int j, k, l, m, i_p;
    double sums[3][3];

    for (i_p = 0; i_p < n_patom; i_p++) {
        for (k = 0; k < 3; k++) {
            for (l = 0; l < 3; l++) {
                sums[k][l] = 0;
                m = i_p * n_satom * 9 + k * 3 + l;
                for (j = 0; j < n_satom; j++) {
                    if (p2s[i_p] != j) {
                        sums[k][l] += fc[m];
                    }
                    m += 9;
                }
            }
        }
        for (k = 0; k < 3; k++) {
            for (l = 0; l < 3; l++) {
                fc[i_p * n_satom * 9 + p2s[i_p] * 9 + k * 3 + l] =
                    -(sums[k][l] + sums[l][k]) / 2;
            }
        }
    }
}

static double get_free_energy(const double temperature, const double f, const int classical) {
    /* temperature is defined by T (K) */
    /* 'f' must be given in eV. */
    if (classical) {
        return KB * temperature * log(f / (KB * temperature));
    } else {
        return KB * temperature * log(1 - exp(-f / (KB * temperature)));
    }
}

static double get_entropy(const double temperature, const double f, const int classical) {
    /* temperature is defined by T (K) */
    /* 'f' must be given in eV. */
    double val;
    if (classical) {
        return KB - KB * log(f / (KB * temperature));
    } else {
        val = f / (2 * KB * temperature);
        return 1 / (2 * temperature) * f * cosh(val) / sinh(val) -
               KB * log(2 * sinh(val));
    }
}

static double get_heat_capacity(const double temperature, const double f, const int classical) {
    /* temperature is defined by T (K) */
    /* 'f' must be given in eV. */
    /* If val is close to 1. Then expansion is used. */
    double val, val1, val2;
    if (classical) {
        return KB;
    } else {
        val = f / (KB * temperature);
        val1 = exp(val);
        val2 = (val) / (val1 - 1);
        return KB * val1 * val2 * val2;
    }
}

static void distribute_fc2(double (*fc2)[3][3], /* shape[n_pos][n_pos] */
                           const int *atom_list, const int len_atom_list,
                           const int *fc_indices_of_atom_list,
                           const double (*r_carts)[3][3], /* shape[n_rot] */
                           const int *permutations, /* shape[n_rot][n_pos] */
                           const int *map_atoms,    /* shape [n_pos] */
                           const int *map_syms,     /* shape [n_pos] */
                           const int num_rot, const int num_pos) {
    int i, j, k, l, m;
    int atom_todo, atom_done, atom_other;
    int sym_index;
    int *atom_list_reverse;
    double(*fc2_done)[3];
    double(*fc2_todo)[3];
    const double(*r_cart)[3];
    const int *permutation;

    atom_list_reverse = NULL;
    atom_list_reverse = (int *)malloc(sizeof(int) * num_pos);
    /* atom_list_reverse[!atom_done] is undefined. */
    for (i = 0; i < len_atom_list; i++) {
        atom_done = map_atoms[atom_list[i]];
        if (atom_done == atom_list[i]) {
            atom_list_reverse[atom_done] = i;
        }
    }

    for (i = 0; i < len_atom_list; i++) {
        /* look up how this atom maps into the done list. */
        atom_todo = atom_list[i];
        atom_done = map_atoms[atom_todo];
        sym_index = map_syms[atom_todo];

        /* skip the atoms in the done list, */
        /* which are easily identified because they map to themselves. */
        if (atom_todo == atom_done) {
            continue;
        }

        /* look up information about the rotation */
        r_cart = r_carts[sym_index];
        permutation = &permutations[sym_index * num_pos]; /* shape[num_pos] */

        /* distribute terms from atom_done to atom_todo */
        for (atom_other = 0; atom_other < num_pos; atom_other++) {
            fc2_done =
                fc2[fc_indices_of_atom_list[atom_list_reverse[atom_done]] *
                        num_pos +
                    permutation[atom_other]];
            fc2_todo = fc2[fc_indices_of_atom_list[i] * num_pos + atom_other];
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++) {
                    for (l = 0; l < 3; l++) {
                        for (m = 0; m < 3; m++) {
                            /* P' = R^-1 P R */
                            fc2_todo[j][k] +=
                                r_cart[l][j] * r_cart[m][k] * fc2_done[l][m];
                        }
                    }
                }
            }
        }
    }

    free(atom_list_reverse);
    atom_list_reverse = NULL;
}

/* static double get_energy(double temperature, double f){ */
/*   /\* temperature is defined by T (K) *\/ */
/*   /\* 'f' must be given in eV. *\/ */
/*   return f / (exp(f / (KB * temperature)) - 1); */
/* } */

static int nint(const double a) {
    if (a < 0.0)
        return (int)(a - 0.5);
    else
        return (int)(a + 0.5);
}
