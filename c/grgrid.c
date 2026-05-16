/* Copyright (C) 2020 Atsushi Togo */
/* All rights reserved. */

/* This file is part of kspclib. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the kspclib project nor the names of its */
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

#include "grgrid.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "lagrid.h"
#include "snf3x3.h"

#define IDENTITY_TOL 1e-5

static void reduce_grid_address(int64_t address[3], const int64_t D_diag[3]);
static int64_t get_double_grid_index(const int64_t address_double[3],
                                     const int64_t D_diag[3],
                                     const int64_t PS[3]);
static int64_t get_grid_index_from_address(const int64_t address[3],
                                           const int64_t D_diag[3]);
static void get_all_grid_addresses(int64_t grid_address[][3],
                                   const int64_t D_diag[3]);
static void get_grid_address_from_index(int64_t address[3],
                                        const int64_t grid_index,
                                        const int64_t D_diag[3]);
static void get_grid_address(int64_t address[3],
                             const int64_t address_double[3],
                             const int64_t PS[3]);
static void get_double_grid_address(int64_t address_double[3],
                                    const int64_t address[3],
                                    const int64_t PS[3]);
static int64_t rotate_grid_index(const int64_t grid_index,
                                 const int64_t rotation[3][3],
                                 const int64_t D_diag[3], const int64_t PS[3]);
static void get_ir_grid_map(int64_t *ir_grid_map,
                            const int64_t (*rotations)[3][3],
                            const int64_t num_rot, const int64_t D_diag[3],
                            const int64_t PS[3]);

int64_t grg_get_snf3x3(int64_t D_diag[3], int64_t P[3][3], int64_t Q[3][3],
                       const int64_t A[3][3]) {
    int64_t i, j, succeeded;
    int64_t D[3][3];

    succeeded = 0;

    if (lagmat_get_determinant_l3(A) == 0) {
        goto err;
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            D[i][j] = A[i][j];
        }
    }

    succeeded = snf3x3(D, P, Q);
    for (i = 0; i < 3; i++) {
        D_diag[i] = D[i][i];
    }

err:
    return succeeded;
}

/*----------------------------------------*/
/* Transform rotations by D(Q^-1)RQ(D^-1) */
/*----------------------------------------*/
/* transformed_rots : D(Q^-1)RQ(D^-1) */
/* rotations : [num_rot][3][3] */
/*    Defined as q' = Rq where q is in the reciprocal primitive basis */
/*    vectors. */
/* num_rot : Number of rotations */
int64_t grg_transform_rotations(int64_t (*transformed_rots)[3][3],
                                const int64_t (*rotations)[3][3],
                                const int64_t num_rot, const int64_t D_diag[3],
                                const int64_t Q[3][3]) {
    int64_t i, j, k;
    double r[3][3], Q_double[3][3];

    /* Compute D(Q^-1)RQ(D^-1) by three steps */
    /* It is assumed that |det(Q)|=1 and Q^-1 has relatively small round-off */
    /* error, and we want to divide by D carefully. */
    /* 1. Compute (Q^-1)RQ */
    /* 2. Compute D(Q^-1)RQ */
    /* 3. Compute D(Q^-1)RQ(D^-1) */
    lagmat_cast_matrix_3l_to_3d(Q_double, Q);
    for (i = 0; i < num_rot; i++) {
        lagmat_get_similar_matrix_ld3(r, rotations[i], Q_double, 0);
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                r[j][k] *= D_diag[j];
                r[j][k] /= D_diag[k];
            }
        }
        lagmat_cast_matrix_3d_to_3l(transformed_rots[i], r);
        if (!lagmat_check_identity_matrix_ld3(transformed_rots[i], r,
                                              IDENTITY_TOL)) {
            return 0;
        }
    }

    return 1;
}

/* -------------------------------*/
/* Get all address in single grid */
/* -------------------------------*/
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
void grg_get_all_grid_addresses(int64_t (*grid_address)[3],
                                const int64_t D_diag[3]) {
    get_all_grid_addresses(grid_address, D_diag);
}

/* -------------------------------------------------------*/
/* Get address in double grid from address in single grid */
/* -------------------------------------------------------*/
/* This function doubles single-grid address and shifts it by PS. */
/* No modulo operation is applied to returned double-grid address. */
/* address_double : Double grid address. */
/* address : Single grid address. */
/* PS : Shifts transformed by P. s_i is 0 or 1. */
void grg_get_double_grid_address(int64_t address_double[3],
                                 const int64_t address[3],
                                 const int64_t PS[3]) {
    get_double_grid_address(address_double, address, PS);
}

/* -------------------------------------------------------*/
/* Get address in single grid from address in double grid */
/* -------------------------------------------------------*/
/* This function shifts double-grid address by PS and divides it by 2. */
/* No modulo operation is applied to returned single-grid address. */
/* address : Single grid address. */
/* address_double : Double grid address. */
/* PS : Shifts transformed by P. s_i is 0 or 1. */
void grg_get_grid_address(int64_t address[3], const int64_t address_double[3],
                          const int64_t PS[3]) {
    get_grid_address(address, address_double, PS);
}

/* -------------------------------------------------*/
/* Get grid point index from address in double grid */
/* -------------------------------------------------*/
/* address_double : Double grid address. */
/* D_diag : Diagnal elements of D. */
/* PS : Shifts transformed by P. s_i is 0 or 1. */
int64_t grg_get_double_grid_index(const int64_t address_double[3],
                                  const int64_t D_diag[3],
                                  const int64_t PS[3]) {
    return get_double_grid_index(address_double, D_diag, PS);
}

/* -------------------------------------------------*/
/* Get grid point index from address in single grid */
/* -------------------------------------------------*/
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
int64_t grg_get_grid_index(const int64_t address[3], const int64_t D_diag[3]) {
    int64_t red_adrs[3];

    lagmat_copy_vector_l3(red_adrs, address);
    reduce_grid_address(red_adrs, D_diag);
    return get_grid_index_from_address(red_adrs, D_diag);
}

/* ---------------------------------------*/
/* Get grid address from grid point index */
/* ---------------------------------------*/
/* address : Single grid address. */
/* D_diag : Diagnal elements of D. */
void grg_get_grid_address_from_index(int64_t address[3],
                                     const int64_t grid_index,
                                     const int64_t D_diag[3]) {
    get_grid_address_from_index(address, grid_index, D_diag);
}

/* ---------------------------*/
/* Rotate grid point by index */
/* ---------------------------*/
int64_t grg_rotate_grid_index(const int64_t grid_index,
                              const int64_t rotation[3][3],
                              const int64_t D_diag[3], const int64_t PS[3]) {
    return rotate_grid_index(grid_index, rotation, D_diag, PS);
}

/* -----------------------------*/
/* Find irreducible grid points */
/* -----------------------------*/
void grg_get_ir_grid_map(int64_t *ir_grid_map, const int64_t (*rotations)[3][3],
                         const int64_t num_rot, const int64_t D_diag[3],
                         const int64_t PS[3]) {
    get_ir_grid_map(ir_grid_map, rotations, num_rot, D_diag, PS);
}

/* Unique reciprocal rotations are collected from input rotations. */
/* is_transpose == 0 : Input rotations are considered those for */
/* reciprocal space. */
/* is_transpose != 0 : Input rotations are considered those for */
/* direct space, i.e., the rotation matrices are transposed. */
/* is_time_reversal controls if inversion is added in the group of */
/* reciprocal space rotations. */
/* Return 0 if failed */
int64_t grg_get_reciprocal_point_group(int64_t rec_rotations[48][3][3],
                                       const int64_t (*rotations)[3][3],
                                       const int64_t num_rot,
                                       const int64_t is_time_reversal,
                                       const int64_t is_transpose)

{
    int64_t i, j, num_rot_ret, inv_exist;
    const int64_t inversion[3][3] = {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};

    /* Collect unique rotations */
    num_rot_ret = 0;
    for (i = 0; i < num_rot; i++) {
        for (j = 0; j < num_rot_ret; j++) {
            if (lagmat_check_identity_matrix_l3(rotations[i],
                                                rec_rotations[j])) {
                goto escape;
            }
        }
        if (num_rot_ret == 48) {
            goto err;
        }
        lagmat_copy_matrix_l3(rec_rotations[num_rot_ret], rotations[i]);
        num_rot_ret++;
    escape:;
    }

    if (is_transpose) {
        for (i = 0; i < num_rot_ret; i++) {
            lagmat_transpose_matrix_l3(rec_rotations[i], rec_rotations[i]);
        }
    }

    if (is_time_reversal) {
        inv_exist = 0;
        for (i = 0; i < num_rot_ret; i++) {
            if (lagmat_check_identity_matrix_l3(inversion, rec_rotations[i])) {
                inv_exist = 1;
                break;
            }
        }

        if (!inv_exist) {
            if (num_rot_ret > 24) {
                goto err;
            }

            for (i = 0; i < num_rot_ret; i++) {
                lagmat_multiply_matrix_l3(rec_rotations[num_rot_ret + i],
                                          inversion, rec_rotations[i]);
            }
            num_rot_ret *= 2;
        }
    }

    return num_rot_ret;
err:
    return 0;
}

static void reduce_grid_address(int64_t address[3], const int64_t D_diag[3]) {
    int64_t i;

    for (i = 0; i < 3; i++) {
        address[i] = lagmat_modulo_l(address[i], D_diag[i]);
    }
}

static int64_t get_double_grid_index(const int64_t address_double[3],
                                     const int64_t D_diag[3],
                                     const int64_t PS[3]) {
    int64_t address[3];

    get_grid_address(address, address_double, PS);
    reduce_grid_address(address, D_diag);
    return get_grid_index_from_address(address, D_diag);
}

/* Here address elements have to be zero or positive. */
/* Therefore reduction to interval [0, D_diag[i]) has to be */
/* done outside of this function. */
/* See kgrid.h about GRID_ORDER_XYZ information. */
static int64_t get_grid_index_from_address(const int64_t address[3],
                                           const int64_t D_diag[3]) {
#ifndef GRID_ORDER_XYZ
    return (address[2] * D_diag[0] * D_diag[1] + address[1] * D_diag[0] +
            address[0]);
#else
    return (address[0] * D_diag[1] * D_diag[2] + address[1] * D_diag[2] +
            address[2]);
#endif
}

static void get_all_grid_addresses(int64_t grid_address[][3],
                                   const int64_t D_diag[3]) {
    int64_t i, j, k, grid_index;
    int64_t address[3];

    for (i = 0; i < D_diag[0]; i++) {
        address[0] = i;
        for (j = 0; j < D_diag[1]; j++) {
            address[1] = j;
            for (k = 0; k < D_diag[2]; k++) {
                address[2] = k;
                grid_index = get_grid_index_from_address(address, D_diag);
                lagmat_copy_vector_l3(grid_address[grid_index], address);
            }
        }
    }
}

/* See grg_get_grid_address_from_index */
static void get_grid_address_from_index(int64_t address[3],
                                        const int64_t grid_index,
                                        const int64_t D_diag[3]) {
    int64_t nn;

#ifndef GRID_ORDER_XYZ
    nn = D_diag[0] * D_diag[1];
    address[0] = grid_index % D_diag[0];
    address[2] = grid_index / nn;
    address[1] = (grid_index - address[2] * nn) / D_diag[0];
#else
    nn = D_diag[1] * D_diag[2];
    address[2] = grid_index % D_diag[2];
    address[0] = grid_index / nn;
    address[1] = (grid_index - address[0] * nn) / D_diag[2];
#endif
}

/* Usually address has to be reduced to [0, D_diag[i]) */
/* by calling reduce_grid_address after this operation. */
static void get_grid_address(int64_t address[3],
                             const int64_t address_double[3],
                             const int64_t PS[3]) {
    int64_t i;

    for (i = 0; i < 3; i++) {
        address[i] = (address_double[i] - PS[i]) / 2;
    }
}

/* Usually address_double has to be reduced to [0, 2*D_diag[i]) */
/* by calling reduce_double_grid_address after this operation. */
static void get_double_grid_address(int64_t address_double[3],
                                    const int64_t address[3],
                                    const int64_t PS[3]) {
    int64_t i;

    for (i = 0; i < 3; i++) {
        address_double[i] = address[i] * 2 + PS[i];
    }
}

static int64_t rotate_grid_index(const int64_t grid_index,
                                 const int64_t rotation[3][3],
                                 const int64_t D_diag[3], const int64_t PS[3]) {
    int64_t adrs[3], dadrs[3], dadrs_rot[3];

    get_grid_address_from_index(adrs, grid_index, D_diag);
    get_double_grid_address(dadrs, adrs, PS);
    lagmat_multiply_matrix_vector_l3(dadrs_rot, rotation, dadrs);
    return get_double_grid_index(dadrs_rot, D_diag, PS);
}

/* Find ir-grid points. */
/* This algorithm relies on the ir-grid index is always smallest */
/* number among symmetrically equivalent grid points. */
static void get_ir_grid_map(int64_t *ir_grid_map,
                            const int64_t (*rotations)[3][3],
                            const int64_t num_rot, const int64_t D_diag[3],
                            const int64_t PS[3]) {
    int64_t gp, num_gp, r_gp;
    int64_t i;

    num_gp = D_diag[0] * D_diag[1] * D_diag[2];

    for (gp = 0; gp < num_gp; gp++) {
        ir_grid_map[gp] = num_gp;
    }

    /* Do not simply multithreaded this for-loop. */
    /* This algorithm contains race condition in different gp's. */
    for (gp = 0; gp < num_gp; gp++) {
        for (i = 0; i < num_rot; i++) {
            r_gp = rotate_grid_index(gp, rotations[i], D_diag, PS);
            if (r_gp < gp) {
                ir_grid_map[gp] = ir_grid_map[r_gp];
                break;
            }
        }
        if (ir_grid_map[gp] == num_gp) {
            ir_grid_map[gp] = gp;
        }
    }
}
