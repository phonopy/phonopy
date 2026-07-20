/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __dynmat_H__
#define __dynmat_H__

#include <stdint.h>

int64_t dym_dynamical_matrices_with_dd_openmp_over_qpoints(
    double (*dynamical_matrices)[2], const double (*qpoints)[3],
    const int64_t n_qpoints, const double *fc, const double (*svecs)[3],
    const int64_t (*multi)[2], const double (*positions)[3],
    const int64_t num_patom, const int64_t num_satom, const double *masses,
    const int64_t *p2s_map, const int64_t *s2p_map, const double (*born)[3][3],
    const double dielectric[3][3], const double (*reciprocal_lattice)[3],
    const double *q_direction, const double nac_factor,
    const double (*dd_q0)[2], const double (*G_list)[3],
    const int64_t num_G_points, const double lambda, const int64_t use_Wang_NAC,
    const int64_t hermitianize);
void dym_get_recip_dipole_dipole(
    double (*dd)[2],           /* [natom, 3, natom, 3, (real,imag)] */
    const double (*dd_q0)[2],  /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const int64_t num_G, const int64_t num_patom, const double q_cart[3],
    const double *q_direction_cart, /* must be pointer */
    const double (*born)[3][3], const double dielectric[3][3],
    const double (*pos)[3], /* [num_patom, 3] */
    const double factor,    /* 4pi/V*unit-conv */
    const double lambda, const double tolerance, const int64_t use_openmp);
void dym_get_recip_dipole_dipole_q0(
    double (*dd_q0)[2],        /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const int64_t num_G, const int64_t num_patom, const double (*born)[3][3],
    const double dielectric[3][3], const double (*pos)[3], /* [natom, 3] */
    const double lambda, const double tolerance, const int64_t use_openmp);
void dym_get_charge_sum(double (*charge_sum)[3][3], const int64_t num_patom,
                        const double factor, const double q_cart[3],
                        const double (*born)[3][3]);
/* fc[num_patom, num_satom, 3, 3] */
/* dm[num_comm_points, num_patom * 3, num_patom *3] */
/* comm_points[num_satom / num_patom, 3] */
/* shortest_vectors[:, 3] */
/* multiplicities[num_satom, num_patom, 2] */
void dym_transform_dynmat_to_fc(
    double *fc, const double (*dm)[2], const double (*comm_points)[3],
    const double (*svecs)[3], const int64_t (*multi)[2], const double *masses,
    const int64_t *s2pp_map, const int64_t *fc_index_map,
    const int64_t num_patom, const int64_t num_satom, const int64_t use_openmp);

#endif
