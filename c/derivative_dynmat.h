/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __derivative_dynmat_H__
#define __derivative_dynmat_H__

#include <stdint.h>

void ddm_get_derivative_dynmat_at_q(
    double (*derivative_dynmat)[2], const int64_t num_patom,
    const int64_t num_satom, const double *fc, const double *q,
    const double *lattice, /* column vector */
    const double *reclat,  /* column vector */
    const double (*svecs)[3], const int64_t (*multi)[2], const double *mass,
    const int64_t *s2p_map, const int64_t *p2s_map, const double nac_factor,
    const double *born, const double *dielectric, const double *q_direction,
    const int64_t is_nac, const int64_t use_openmp);

#endif
