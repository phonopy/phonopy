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

#ifndef __dynmat_H__
#define __dynmat_H__

long dym_dynamical_matrices_with_dd_openmp_over_qpoints(
    double (*dynamical_matrices)[2], const double (*qpoints)[3],
    const long n_qpoints, const double *fc, const double (*svecs)[3],
    const long (*multi)[2], const double (*positions)[3], const long num_patom,
    const long num_satom, const double *masses, const long *p2s_map,
    const long *s2p_map, const double (*born)[3][3],
    const double dielectric[3][3], const double (*reciprocal_lattice)[3],
    const double *q_direction, const double nac_factor,
    const double (*dd_q0)[2], const double (*G_list)[3],
    const long num_G_points, const double lambda, const long use_Wang_NAC);
long dym_get_dynamical_matrices_openmp_over_qpoints(
    double (*dynamical_matrices)[2], const long num_patom, const long num_satom,
    const double *fc, const double (*qpoints)[3], const long n_qpoints,
    const double (*svecs)[3], const long (*multi)[2], const double *mass,
    const long *s2p_map, const long *p2s_map, const double (*charge_sum)[3][3]);
long dym_get_dynamical_matrix_at_q(double (*dynamical_matrix)[2],
                                   const long num_patom, const long num_satom,
                                   const double *fc, const double q[3],
                                   const double (*svecs)[3],
                                   const long (*multi)[2], const double *mass,
                                   const long *s2p_map, const long *p2s_map,
                                   const double (*charge_sum)[3][3],
                                   const long use_openmp);
void dym_get_recip_dipole_dipole(
    double (*dd)[2],           /* [natom, 3, natom, 3, (real,imag)] */
    const double (*dd_q0)[2],  /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double q_cart[3],
    const double *q_direction_cart, /* must be pointer */
    const double (*born)[3][3], const double dielectric[3][3],
    const double (*pos)[3], /* [num_patom, 3] */
    const double factor,    /* 4pi/V*unit-conv */
    const double lambda, const double tolerance, const long use_openmp);
void dym_get_recip_dipole_dipole_q0(
    double (*dd_q0)[2],        /* [natom, 3, 3, (real,imag)] */
    const double (*G_list)[3], /* [num_G, 3] */
    const long num_G, const long num_patom, const double (*born)[3][3],
    const double dielectric[3][3], const double (*pos)[3], /* [natom, 3] */
    const double lambda, const double tolerance, const long use_openmp);
void dym_get_charge_sum(double (*charge_sum)[3][3], const long num_patom,
                        const double factor, const double q_cart[3],
                        const double (*born)[3][3]);
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
                                const long use_openmp);

#endif
