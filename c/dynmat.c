/* Copyright (C) 2015 Atsushi Togo */
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

#include "dynmat.h"

#include <math.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

static void add_dynmat_dd_at_q(
    double (*dynamical_matrices)[2], const double q[3], const double *fc,
    const double (*positions)[3], const long num_patom, const double *masses,
    const double (*born)[3][3], const double dielectric[3][3],
    const double reciprocal_lattice[3][3], const double *q_dir_cart,
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const long num_G_points, const double lambda,
    const double tolerance);
static void get_dynmat_ij(double (*dynamical_matrix)[2], const long num_patom,
                          const long num_satom, const double *fc,
                          const double q[3], const double (*svecs)[3],
                          const long (*multi)[2], const double *mass,
                          const long *s2p_map, const long *p2s_map,
                          const double (*charge_sum)[3][3], const long i,
                          const long j);
static void get_dm(double dm[3][3][2], const long num_patom,
                   const long num_satom, const double *fc, const double q[3],
                   const double (*svecs)[3], const long (*multi)[2],
                   const long *p2s_map, const double (*charge_sum)[3][3],
                   const long i, const long j, const long k);
static double get_dielectric_part(const double q_cart[3],
                                  const double dielectric[3][3]);
static void get_dd(double (*dd_part)[2], /* [natom, 3, natom, 3, (real,imag)] */
                   const double (*G_list)[3], /* [num_G, 3] */
                   const long num_G, const long num_patom,
                   const double q_cart[3], const double *q_direction_cart,
                   const double dielectric[3][3],
                   const double (*pos)[3], /* [num_patom, 3] */
                   const double lambda, const double tolerance,
                   const long use_openmp);
static void get_dd_at_g(
    double (*dd_part)[2], /* [natom, 3, natom, 3, (real,imag)] */
    const long i, const long j, const double G[3], const long num_patom,
    const double (*pos)[3], /* [num_patom, 3] */
    const double KK[3][3]);
static void make_Hermitian(double (*mat)[2], const long num_band);
static void multiply_borns(double (*dd)[2], const double (*dd_in)[2],
                           const long num_patom, const double (*born)[3][3],
                           const long use_openmp);
static void multiply_borns_at_ij(double (*dd)[2], const long i, const long j,
                                 const double (*dd_in)[2], const long num_patom,
                                 const double (*born)[3][3]);
static void transform_dynmat_to_fc_ij(
    double *fc, const double (*dm)[2], const long i, const long j,
    const double (*comm_points)[3], const double (*svecs)[3],
    const long (*multi)[2], const double *masses, const long *s2pp_map,
    const long *fc_index_map, const long num_patom, const long num_satom);
static void get_q_cart(double q_cart[3], const double q[3],
                       const double reciprocal_lattice[3][3]);
static void get_dynmat_want(
    double (*dynamical_matrices)[2], const double qpoint[3], const double *fc,
    const double (*svecs)[3], const long (*multi)[2], const long num_patom,
    const long num_satom, const double *masses, const long *p2s_map,
    const long *s2p_map, const double (*born)[3][3],
    const double dielectric[3][3], const double (*reciprocal_lattice)[3],
    const double *q_direction, const double *q_dir_cart,
    const double nac_factor, const double q_zero_tolerance);

/// @brief Calculate dynamical matrices with openmp over q-points
/// use_Wang_NAC: Wang et al.
/// !use_Wang_NAC and dd_0 is NULL: no-NAC
/// !use_Wang_NAC and dd_0 is not NULL: NAC by Gonze and Lee.
/// @param reciprocal_lattice in column vectors
/// @param q_direction in Crystallographic coordinates.
long dym_dynamical_matrices_with_dd_openmp_over_qpoints(
    double (*dynamical_matrices)[2], const double (*qpoints)[3],
    const long n_qpoints, const double *fc, const double (*svecs)[3],
    const long (*multi)[2], const double (*positions)[3], const long num_patom,
    const long num_satom, const double *masses, const long *p2s_map,
    const long *s2p_map, const double (*born)[3][3],
    const double dielectric[3][3], const double (*reciprocal_lattice)[3],
    const double *q_direction, const double nac_factor,
    const double (*dd_q0)[2], const double (*G_list)[3],
    const long num_G_points, const double lambda, const long use_Wang_NAC) {
    long i, n, adrs_shift;
    double *q_dir_cart;
    double q_zero_tolerance;

    q_zero_tolerance = 1e-5;
    q_dir_cart = NULL;
    adrs_shift = num_patom * num_patom * 9;

    if (q_direction) {
        q_dir_cart = (double *)malloc(sizeof(double) * 3);
        get_q_cart(q_dir_cart, q_direction, reciprocal_lattice);
    }

    if (use_Wang_NAC) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (i = 0; i < n_qpoints; i++) {
            get_dynmat_want(dynamical_matrices + adrs_shift * i, qpoints[i], fc,
                            svecs, multi, num_patom, num_satom, masses, p2s_map,
                            s2p_map, born, dielectric, reciprocal_lattice,
                            q_direction, q_dir_cart, nac_factor,
                            q_zero_tolerance);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (i = 0; i < n_qpoints; i++) {
            dym_get_dynamical_matrix_at_q(
                dynamical_matrices + adrs_shift * i, num_patom, num_satom, fc,
                qpoints[i], svecs, multi, masses, s2p_map, p2s_map, NULL, 0);
            if (dd_q0) {  // NAC by Gonze and Lee if dd_in is not NULL
                add_dynmat_dd_at_q(dynamical_matrices + adrs_shift * i,
                                   qpoints[i], fc, positions, num_patom, masses,
                                   born, dielectric, reciprocal_lattice,
                                   q_dir_cart, nac_factor, dd_q0, G_list,
                                   num_G_points, lambda, q_zero_tolerance);
            }
        }
    }

    if (q_direction) {
        free(q_dir_cart);
        q_dir_cart = NULL;
    }
    return 0;
}

static void get_dynmat_want(
    double (*dynamical_matrices)[2], const double qpoint[3], const double *fc,
    const double (*svecs)[3], const long (*multi)[2], const long num_patom,
    const long num_satom, const double *masses, const long *p2s_map,
    const long *s2p_map, const double (*born)[3][3],
    const double dielectric[3][3], const double (*reciprocal_lattice)[3],
    const double *q_direction, const double *q_dir_cart,
    const double nac_factor, const double q_zero_tolerance) {
    double(*charge_sum)[3][3];
    double q_cart[3];
    double q_norm;
    long n;

    charge_sum = NULL;
    n = num_satom / num_patom;

    get_q_cart(q_cart, qpoint, reciprocal_lattice);
    q_norm = sqrt(q_cart[0] * q_cart[0] + q_cart[1] * q_cart[1] +
                  q_cart[2] * q_cart[2]);
    charge_sum =
        (double(*)[3][3])malloc(sizeof(double[3][3]) * num_patom * num_patom);

    if (q_norm < q_zero_tolerance) {
        if (q_direction) {
            dym_get_charge_sum(
                charge_sum, num_patom,
                nac_factor / n / get_dielectric_part(q_dir_cart, dielectric),
                q_dir_cart, born);
            dym_get_dynamical_matrix_at_q(
                dynamical_matrices, num_patom, num_satom, fc, qpoint, svecs,
                multi, masses, s2p_map, p2s_map, charge_sum, 0);

        } else {
            dym_get_dynamical_matrix_at_q(dynamical_matrices, num_patom,
                                          num_satom, fc, qpoint, svecs, multi,
                                          masses, s2p_map, p2s_map, NULL, 0);
        }
    } else {
        dym_get_charge_sum(
            charge_sum, num_patom,
            nac_factor / n / get_dielectric_part(q_cart, dielectric), q_cart,
            born);
        dym_get_dynamical_matrix_at_q(dynamical_matrices, num_patom, num_satom,
                                      fc, qpoint, svecs, multi, masses, s2p_map,
                                      p2s_map, charge_sum, 0);
    }

    free(charge_sum);
    charge_sum = NULL;
}

static void add_dynmat_dd_at_q(
    double (*dynamical_matrices)[2], const double q[3], const double *fc,
    const double (*positions)[3], const long num_patom, const double *masses,
    const double (*born)[3][3], const double dielectric[3][3],
    const double reciprocal_lattice[3][3], const double *q_dir_cart,
    const double nac_factor, const double (*dd_q0)[2],
    const double (*G_list)[3], const long num_G_points, const double lambda,
    const double tolerance) {
    long i, j, k, l, adrs;
    double(*dd)[2];
    double q_cart[3];
    double mm;

    dd = (double(*)[2])malloc(sizeof(double[2]) * num_patom * num_patom * 9);
    get_q_cart(q_cart, q, reciprocal_lattice);
    dym_get_recip_dipole_dipole(dd, dd_q0, G_list, num_G_points, num_patom,
                                q_cart, q_dir_cart, born, dielectric, positions,
                                nac_factor, lambda, tolerance, 0);

    for (i = 0; i < num_patom; i++) {
        for (j = 0; j < num_patom; j++) {
            mm = sqrt(masses[i] * masses[j]);
            for (k = 0; k < 3; k++) {
                for (l = 0; l < 3; l++) {
                    adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
                    dynamical_matrices[adrs][0] += dd[adrs][0] / mm;
                    dynamical_matrices[adrs][1] += dd[adrs][1] / mm;
                }
            }
        }
    }

    free(dd);
    dd = NULL;
}

/// @brief charge_sum is NULL if G-L NAC or no-NAC.
long dym_get_dynamical_matrix_at_q(double (*dynamical_matrix)[2],
                                   const long num_patom, const long num_satom,
                                   const double *fc, const double q[3],
                                   const double (*svecs)[3],
                                   const long (*multi)[2], const double *mass,
                                   const long *s2p_map, const long *p2s_map,
                                   const double (*charge_sum)[3][3],
                                   const long use_openmp) {
    long i, j, ij;

    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (ij = 0; ij < num_patom * num_patom; ij++) {
            get_dynmat_ij(dynamical_matrix, num_patom, num_satom, fc, q, svecs,
                          multi, mass, s2p_map, p2s_map, charge_sum,
                          ij / num_patom,  /* i */
                          ij % num_patom); /* j */
        }
    } else {
        for (i = 0; i < num_patom; i++) {
            for (j = 0; j < num_patom; j++) {
                get_dynmat_ij(dynamical_matrix, num_patom, num_satom, fc, q,
                              svecs, multi, mass, s2p_map, p2s_map, charge_sum,
                              i, j);
            }
        }
    }

    make_Hermitian(dynamical_matrix, num_patom * 3);

    return 0;
}

void dym_get_recip_dipole_dipole(
    double (*dd)[2],           /* [natom, 3, natom, 3, (real,imag)] */
    const double (*dd_q0)[2],  /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double q_cart[3],
    const double *q_direction_cart, /* must be pointer */
    const double (*born)[3][3], const double dielectric[3][3],
    const double (*pos)[3], /* [num_patom, 3] */
    const double factor,    /* 4pi/V*unit-conv */
    const double lambda, const double tolerance, const long use_openmp) {
    long i, k, l, adrs, adrs_sum;
    double(*dd_tmp)[2];

    dd_tmp =
        (double(*)[2])malloc(sizeof(double[2]) * num_patom * num_patom * 9);

    for (i = 0; i < num_patom * num_patom * 9; i++) {
        dd[i][0] = 0;
        dd[i][1] = 0;
        dd_tmp[i][0] = 0;
        dd_tmp[i][1] = 0;
    }

    get_dd(dd_tmp, G_list, num_G, num_patom, q_cart, q_direction_cart,
           dielectric, pos, lambda, tolerance, use_openmp);

    multiply_borns(dd, dd_tmp, num_patom, born, use_openmp);

    for (i = 0; i < num_patom; i++) {
        for (k = 0; k < 3; k++) {     /* alpha */
            for (l = 0; l < 3; l++) { /* beta */
                adrs = i * num_patom * 9 + k * num_patom * 3 + i * 3 + l;
                adrs_sum = i * 9 + k * 3 + l;
                dd[adrs][0] -= dd_q0[adrs_sum][0];
                dd[adrs][1] -= dd_q0[adrs_sum][1];
            }
        }
    }

    for (i = 0; i < num_patom * num_patom * 9; i++) {
        dd[i][0] *= factor;
        dd[i][1] *= factor;
    }

    /* This may not be necessary. */
    /* make_Hermitian(dd, num_patom * 3); */

    free(dd_tmp);
    dd_tmp = NULL;
}

void dym_get_recip_dipole_dipole_q0(
    double (*dd_q0)[2],        /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double (*born)[3][3],
    const double dielectric[3][3], const double (*pos)[3], /* [num_patom, 3] */
    const double lambda, const double tolerance, const long use_openmp) {
    long i, j, k, l, adrs_tmp, adrs, adrsT;
    double zero_vec[3];
    double(*dd_tmp1)[2], (*dd_tmp2)[2];

    dd_tmp1 =
        (double(*)[2])malloc(sizeof(double[2]) * num_patom * num_patom * 9);
    dd_tmp2 =
        (double(*)[2])malloc(sizeof(double[2]) * num_patom * num_patom * 9);

    for (i = 0; i < num_patom * num_patom * 9; i++) {
        dd_tmp1[i][0] = 0;
        dd_tmp1[i][1] = 0;
        dd_tmp2[i][0] = 0;
        dd_tmp2[i][1] = 0;
    }

    zero_vec[0] = 0;
    zero_vec[1] = 0;
    zero_vec[2] = 0;

    get_dd(dd_tmp1, G_list, num_G, num_patom, zero_vec, NULL, dielectric, pos,
           lambda, tolerance, use_openmp);

    multiply_borns(dd_tmp2, dd_tmp1, num_patom, born, use_openmp);

    for (i = 0; i < num_patom * 9; i++) {
        dd_q0[i][0] = 0;
        dd_q0[i][1] = 0;
    }

    for (i = 0; i < num_patom; i++) {
        for (k = 0; k < 3; k++) {     /* alpha */
            for (l = 0; l < 3; l++) { /* beta */
                adrs = i * 9 + k * 3 + l;
                for (j = 0; j < num_patom; j++) {
                    adrs_tmp =
                        i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
                    dd_q0[adrs][0] += dd_tmp2[adrs_tmp][0];
                    dd_q0[adrs][1] += dd_tmp2[adrs_tmp][1];
                }
            }
        }
    }

    /* Summation over another atomic index */
    /* for (j = 0; j < num_patom; j++) { */
    /*   for (k = 0; k < 3; k++) {   /\* alpha *\/ */
    /*     for (l = 0; l < 3; l++) { /\* beta *\/ */
    /*       adrs = j * 9 + k * 3 + l; */
    /*       for (i = 0; i < num_patom; i++) { */
    /*         adrs_tmp = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l
     * ; */
    /*         dd_q0[adrs][0] += dd_tmp2[adrs_tmp][0]; */
    /*         dd_q0[adrs][1] += dd_tmp2[adrs_tmp][1]; */
    /*       } */
    /*     } */
    /*   } */
    /* } */

    for (i = 0; i < num_patom; i++) {
        for (k = 0; k < 3; k++) {     /* alpha */
            for (l = 0; l < 3; l++) { /* beta */
                adrs = i * 9 + k * 3 + l;
                adrsT = i * 9 + l * 3 + k;
                dd_q0[adrs][0] += dd_q0[adrsT][0];
                dd_q0[adrs][0] /= 2;
                dd_q0[adrsT][0] = dd_q0[adrs][0];
                dd_q0[adrs][1] -= dd_q0[adrsT][1];
                dd_q0[adrs][1] /= 2;
                dd_q0[adrsT][1] = -dd_q0[adrs][1];
            }
        }
    }

    free(dd_tmp1);
    dd_tmp1 = NULL;
    free(dd_tmp2);
    dd_tmp2 = NULL;
}

void dym_get_charge_sum(
    double (*charge_sum)[3][3], const long num_patom,
    const double factor, /* 4pi/V*unit-conv and denominator */
    const double q_cart[3], const double (*born)[3][3]) {
    long i, j, k, a, b;
    double(*q_born)[3];

    q_born = (double(*)[3])malloc(sizeof(double[3]) * num_patom);
    for (i = 0; i < num_patom; i++) {
        for (j = 0; j < 3; j++) {
            q_born[i][j] = 0;
        }
    }

    for (i = 0; i < num_patom; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                q_born[i][j] += q_cart[k] * born[i][k][j];
            }
        }
    }

    for (i = 0; i < num_patom; i++) {
        for (j = 0; j < num_patom; j++) {
            for (a = 0; a < 3; a++) {
                for (b = 0; b < 3; b++) {
                    charge_sum[i * num_patom + j][a][b] =
                        q_born[i][a] * q_born[j][b] * factor;
                }
            }
        }
    }

    free(q_born);
    q_born = NULL;
}

/* fc[num_patom, num_satom, 3, 3] */
/* dm[num_comm_points, num_patom * 3, num_patom *3] */
/* comm_points[num_satom / num_patom, 3] */
/* shortest_vectors[:, 3] */
/* multiplicities[num_satom, num_patom, 2] */
void dym_transform_dynmat_to_fc(double *fc, const double (*dm)[2],
                                const double (*comm_points)[3],
                                const double (*svecs)[3],
                                const long (*multi)[2], const double *masses,
                                const long *s2pp_map, const long *fc_index_map,
                                const long num_patom, const long num_satom,
                                const long use_openmp) {
    long i, j, ij;

    for (i = 0; i < num_patom * num_satom * 9; i++) {
        fc[i] = 0;
    }

    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (ij = 0; ij < num_patom * num_satom; ij++) {
            transform_dynmat_to_fc_ij(
                fc, dm, ij / num_satom, ij % num_satom, comm_points, svecs,
                multi, masses, s2pp_map, fc_index_map, num_patom, num_satom);
        }
    } else {
        for (i = 0; i < num_patom; i++) {
            for (j = 0; j < num_satom; j++) {
                transform_dynmat_to_fc_ij(fc, dm, i, j, comm_points, svecs,
                                          multi, masses, s2pp_map, fc_index_map,
                                          num_patom, num_satom);
            }
        }
    }
}

/// @brief charge_sum is NULL if G-L NAC or no-NAC.
static void get_dynmat_ij(double (*dynamical_matrix)[2], const long num_patom,
                          const long num_satom, const double *fc,
                          const double q[3], const double (*svecs)[3],
                          const long (*multi)[2], const double *mass,
                          const long *s2p_map, const long *p2s_map,
                          const double (*charge_sum)[3][3], const long i,
                          const long j) {
    long k, l, adrs;
    double mass_sqrt;
    double dm[3][3][2];  // [3][3][(real, imag)]

    mass_sqrt = sqrt(mass[i] * mass[j]);

    for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
            dm[k][l][0] = 0;
            dm[k][l][1] = 0;
        }
    }

    for (k = 0; k < num_satom; k++) { /* Lattice points of right index of fc */
        if (s2p_map[k] != p2s_map[j]) {
            continue;
        }
        get_dm(dm, num_patom, num_satom, fc, q, svecs, multi, p2s_map,
               charge_sum, i, j, k);
    }

    for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
            adrs = (i * 3 + k) * num_patom * 3 + j * 3 + l;
            dynamical_matrix[adrs][0] = dm[k][l][0] / mass_sqrt;
            dynamical_matrix[adrs][1] = dm[k][l][1] / mass_sqrt;
        }
    }
}

static void get_dm(double dm[3][3][2], const long num_patom,
                   const long num_satom, const double *fc, const double q[3],
                   const double (*svecs)[3], const long (*multi)[2],
                   const long *p2s_map, const double (*charge_sum)[3][3],
                   const long i, const long j, const long k) {
    long l, m, i_pair, m_pair, adrs;
    double phase, cos_phase, sin_phase, fc_elem;

    cos_phase = 0;
    sin_phase = 0;

    i_pair = k * num_patom + i;
    m_pair = multi[i_pair][0];
    adrs = multi[i_pair][1];

    for (l = 0; l < m_pair; l++) {
        phase = 0;
        for (m = 0; m < 3; m++) {
            phase += q[m] * svecs[adrs + l][m];
        }
        cos_phase += cos(phase * 2 * PI) / m_pair;
        sin_phase += sin(phase * 2 * PI) / m_pair;
    }

    for (l = 0; l < 3; l++) {
        for (m = 0; m < 3; m++) {
            if (charge_sum) {
                fc_elem = (fc[p2s_map[i] * num_satom * 9 + k * 9 + l * 3 + m] +
                           charge_sum[i * num_patom + j][l][m]);
            } else {
                fc_elem = fc[p2s_map[i] * num_satom * 9 + k * 9 + l * 3 + m];
            }
            dm[l][m][0] += fc_elem * cos_phase;
            dm[l][m][1] += fc_elem * sin_phase;
        }
    }
}

static double get_dielectric_part(const double q_cart[3],
                                  const double dielectric[3][3]) {
    long i, j;
    double sum;

    sum = 0;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            sum += q_cart[i] * dielectric[i][j] * q_cart[j];
        }
    }
    return sum;
}

static void get_dd(double (*dd_part)[2], /* [natom, 3, natom, 3, (real,imag)] */
                   const double (*G_list)[3], /* [num_G, 3] */
                   const long num_G, const long num_patom,
                   const double q_cart[3], const double *q_direction_cart,
                   const double dielectric[3][3],
                   const double (*pos)[3], /* [num_patom, 3] */
                   const double lambda, const double tolerance,
                   const long use_openmp) {
    long i, j, g;
    double q_K[3];
    double norm, dielectric_part, L2;
    double(*KK)[3][3];

    KK = (double(*)[3][3])malloc(sizeof(double[3][3]) * num_G);
    L2 = 4 * lambda * lambda;

    /* sum over K = G + q and over G (i.e. q=0) */
    /* q_direction has values for summation over K at Gamma point. */
    /* q_direction is NULL for summation over G */
#ifdef _OPENMP
#pragma omp parallel for private(i, j, q_K, norm, \
                                     dielectric_part) if (use_openmp)
#endif
    for (g = 0; g < num_G; g++) {
        norm = 0;
        for (i = 0; i < 3; i++) {
            q_K[i] = G_list[g][i] + q_cart[i];
            norm += q_K[i] * q_K[i];
        }

        if (sqrt(norm) < tolerance) {
            if (!q_direction_cart) {
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++) {
                        KK[g][i][j] = 0;
                    }
                }
                continue;
            } else {
                dielectric_part =
                    get_dielectric_part(q_direction_cart, dielectric);
                for (i = 0; i < 3; i++) {
                    for (j = 0; j < 3; j++) {
                        KK[g][i][j] = q_direction_cart[i] *
                                      q_direction_cart[j] / dielectric_part;
                    }
                }
            }
        } else {
            dielectric_part = get_dielectric_part(q_K, dielectric);
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    KK[g][i][j] = q_K[i] * q_K[j] / dielectric_part *
                                  exp(-dielectric_part / L2);
                }
            }
        }
    }

    // OpenMP should not be used here.
    // Due to race condition in dd_part and bad performance.
    for (g = 0; g < num_G; g++) {
        for (i = 0; i < num_patom; i++) {
            for (j = 0; j < num_patom; j++) {
                get_dd_at_g(dd_part, i, j, G_list[g], num_patom, pos, KK[g]);
            }
        }
    }

    free(KK);
    KK = NULL;
}

static void get_dd_at_g(
    double (*dd_part)[2], /* [natom, 3, natom, 3, (real,imag)] */
    const long i, const long j, const double G[3], const long num_patom,
    const double (*pos)[3], /* [num_patom, 3] */
    const double KK[3][3]) {
    long k, l, adrs;
    double cos_phase, sin_phase, phase;

    phase = 0;
    for (k = 0; k < 3; k++) {
        /* For D-type dynamical matrix */
        /* phase += (pos[i][k] - pos[j][k]) * q_K[k]; */
        /* For C-type dynamical matrix */
        phase += (pos[i][k] - pos[j][k]) * G[k];
    }
    phase *= 2 * PI;
    cos_phase = cos(phase);
    sin_phase = sin(phase);
    for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
            adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
            dd_part[adrs][0] += KK[k][l] * cos_phase;
            dd_part[adrs][1] += KK[k][l] * sin_phase;
        }
    }
}

static void make_Hermitian(double (*mat)[2], const long num_band) {
    long i, j, adrs, adrsT;

    for (i = 0; i < num_band; i++) {
        for (j = i; j < num_band; j++) {
            adrs = i * num_band + j;
            adrsT = j * num_band + i;
            /* real part */
            mat[adrs][0] += mat[adrsT][0];
            mat[adrs][0] /= 2;
            /* imaginary part */
            mat[adrs][1] -= mat[adrsT][1];
            mat[adrs][1] /= 2;
            /* store */
            mat[adrsT][0] = mat[adrs][0];
            mat[adrsT][1] = -mat[adrs][1];
        }
    }
}

static void multiply_borns(double (*dd)[2], const double (*dd_in)[2],
                           const long num_patom, const double (*born)[3][3],
                           const long use_openmp) {
    long i, j, ij;

    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (ij = 0; ij < num_patom * num_patom; ij++) {
            multiply_borns_at_ij(dd, ij / num_patom, ij % num_patom, dd_in,
                                 num_patom, born);
        }
    } else {
        for (i = 0; i < num_patom; i++) {
            for (j = 0; j < num_patom; j++) {
                multiply_borns_at_ij(dd, i, j, dd_in, num_patom, born);
            }
        }
    }
}

static void multiply_borns_at_ij(double (*dd)[2], const long i, const long j,
                                 const double (*dd_in)[2], const long num_patom,
                                 const double (*born)[3][3]) {
    long k, l, m, n, adrs, adrs_in;
    double zz;

    for (k = 0; k < 3; k++) {     /* alpha */
        for (l = 0; l < 3; l++) { /* beta */
            adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
            for (m = 0; m < 3; m++) {     /* alpha' */
                for (n = 0; n < 3; n++) { /* beta' */
                    adrs_in = i * num_patom * 9 + m * num_patom * 3 + j * 3 + n;
                    zz = born[i][m][k] * born[j][n][l];
                    dd[adrs][0] += dd_in[adrs_in][0] * zz;
                    dd[adrs][1] += dd_in[adrs_in][1] * zz;
                }
            }
        }
    }
}

static void transform_dynmat_to_fc_ij(
    double *fc, const double (*dm)[2], const long i, const long j,
    const double (*comm_points)[3], const double (*svecs)[3],
    const long (*multi)[2], const double *masses, const long *s2pp_map,
    const long *fc_index_map, const long num_patom, const long num_satom) {
    long k, l, m, N, adrs, m_pair, i_pair, svecs_adrs;
    double coef, phase, cos_phase, sin_phase;

    N = num_satom / num_patom;
    i_pair = j * num_patom + i;
    m_pair = multi[i_pair][0];
    svecs_adrs = multi[i_pair][1];
    coef = sqrt(masses[i] * masses[s2pp_map[j]]) / N;
    for (k = 0; k < N; k++) {
        cos_phase = 0;
        sin_phase = 0;
        for (l = 0; l < m_pair; l++) {
            phase = 0;
            for (m = 0; m < 3; m++) {
                phase -= comm_points[k][m] * svecs[svecs_adrs + l][m];
            }
            cos_phase += cos(phase * 2 * PI);
            sin_phase += sin(phase * 2 * PI);
        }
        cos_phase /= m_pair;
        sin_phase /= m_pair;
        for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
                adrs = k * num_patom * num_patom * 9 + i * num_patom * 9 +
                       l * num_patom * 3 + s2pp_map[j] * 3 + m;
                fc[fc_index_map[i] * num_satom * 9 + j * 9 + l * 3 + m] +=
                    (dm[adrs][0] * cos_phase - dm[adrs][1] * sin_phase) * coef;
            }
        }
    }
}

/// @brief
/// @param q_cart q-point in Cartesian coordinates
/// @param q q-point in crystallographic coordinates
/// @param reciprocal_lattice in column vectors
static void get_q_cart(double q_cart[3], const double q[3],
                       const double reciprocal_lattice[3][3]) {
    long i, j;

    for (i = 0; i < 3; i++) {
        q_cart[i] = 0;
        for (j = 0; j < 3; j++) {
            q_cart[i] += reciprocal_lattice[i][j] * q[j];
        }
    }
}
