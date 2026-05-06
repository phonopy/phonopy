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

#include "recgrid.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bzgrid.h"
#include "grgrid.h"
#include "lagrid.h"

#define GRID_TOLERANCE_FACTOR 0.01

void recgrid_get_all_grid_addresses(int64_t (*gr_grid_addresses)[3],
                                    const int64_t D_diag[3]) {
    grg_get_all_grid_addresses(gr_grid_addresses, D_diag);
}

void recgrid_get_double_grid_address(int64_t address_double[3],
                                     const int64_t address[3],
                                     const int64_t PS[3]) {
    grg_get_double_grid_address(address_double, address, PS);
}

void recgrid_get_grid_address_from_index(int64_t address[3],
                                         const int64_t grid_index,
                                         const int64_t D_diag[3]) {
    grg_get_grid_address_from_index(address, grid_index, D_diag);
}

int64_t recgrid_get_double_grid_index(const int64_t address_double[3],
                                      const int64_t D_diag[3],
                                      const int64_t PS[3]) {
    return grg_get_double_grid_index(address_double, D_diag, PS);
}

int64_t recgrid_get_grid_index_from_address(const int64_t address[3],
                                            const int64_t D_diag[3]) {
    return grg_get_grid_index(address, D_diag);
}

int64_t recgrid_rotate_grid_index(const int64_t grid_index,
                                  const int64_t rotation[3][3],
                                  const int64_t D_diag[3],
                                  const int64_t PS[3]) {
    return grg_rotate_grid_index(grid_index, rotation, D_diag, PS);
}

int64_t recgrid_get_reciprocal_point_group(int64_t rec_rotations[48][3][3],
                                           const int64_t (*rotations)[3][3],
                                           const int64_t num_rot,
                                           const int64_t is_time_reversal,
                                           const int64_t is_transpose) {
    return grg_get_reciprocal_point_group(rec_rotations, rotations, num_rot,
                                          is_time_reversal, is_transpose);
}

int64_t recgrid_get_snf3x3(int64_t D_diag[3], int64_t P[3][3], int64_t Q[3][3],
                           const int64_t A[3][3]) {
    return grg_get_snf3x3(D_diag, P, Q, A);
}

int64_t recgrid_transform_rotations(int64_t (*transformed_rots)[3][3],
                                    const int64_t (*rotations)[3][3],
                                    const int64_t num_rot,
                                    const int64_t D_diag[3],
                                    const int64_t Q[3][3]) {
    int64_t succeeded;
    succeeded = grg_transform_rotations(transformed_rots, rotations, num_rot,
                                        D_diag, Q);
    return succeeded;
}

int64_t recgrid_get_ir_grid_map(int64_t *ir_grid_map,
                                const int64_t (*rotations)[3][3],
                                const int64_t num_rot, const int64_t D_diag[3],
                                const int64_t PS[3]) {
    int64_t num_ir, i;

    grg_get_ir_grid_map(ir_grid_map, rotations, num_rot, D_diag, PS);

    num_ir = 0;
    for (i = 0; i < D_diag[0] * D_diag[1] * D_diag[2]; i++) {
        if (ir_grid_map[i] == i) {
            num_ir++;
        }
    }
    return num_ir;
}

int64_t recgrid_get_bz_grid_addresses(
    int64_t (*bz_grid_addresses)[3], int64_t *bz_map, int64_t *bzg2grg,
    const int64_t D_diag[3], const int64_t Q[3][3], const int64_t PS[3],
    const double rec_lattice[3][3], const int64_t bz_grid_type) {
    RecgridBZGrid *bzgrid;
    int64_t i, j, size;

    if ((bzgrid = (RecgridBZGrid *)malloc(sizeof(RecgridBZGrid))) == NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->bzg2grg = bzg2grg;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = PS[i];
        for (j = 0; j < 3; j++) {
            bzgrid->Q[i][j] = Q[i][j];
            bzgrid->reclat[i][j] = rec_lattice[i][j];
        }
    }

    if (bzg_get_bz_grid_addresses(bzgrid)) {
        size = bzgrid->size;
    } else {
        size = 0;
    }

    free(bzgrid);
    bzgrid = NULL;

    return size;
}

int64_t recgrid_rotate_bz_grid_index(
    const int64_t bz_grid_index, const int64_t rotation[3][3],
    const int64_t (*bz_grid_addresses)[3], const int64_t *bz_map,
    const int64_t D_diag[3], const int64_t PS[3], const int64_t bz_grid_type) {
    RecgridConstBZGrid *bzgrid;
    int64_t i, rot_bz_gp;

    if ((bzgrid = (RecgridConstBZGrid *)malloc(sizeof(RecgridConstBZGrid))) ==
        NULL) {
        warning_print("Memory could not be allocated.");
        return 0;
    }

    bzgrid->addresses = bz_grid_addresses;
    bzgrid->gp_map = bz_map;
    bzgrid->type = bz_grid_type;
    for (i = 0; i < 3; i++) {
        bzgrid->D_diag[i] = D_diag[i];
        bzgrid->PS[i] = PS[i];
    }

    rot_bz_gp = bzg_rotate_grid_index(bz_grid_index, rotation, bzgrid);

    free(bzgrid);
    bzgrid = NULL;

    return rot_bz_gp;
}

/* Note: Tolerance in squared distance. */
double recgrid_get_tolerance_for_BZ_reduction(const RecgridBZGrid *bzgrid) {
    int64_t i, j;
    double tolerance;
    double length[3];

    for (i = 0; i < 3; i++) {
        length[i] = 0;
        for (j = 0; j < 3; j++) {
            length[i] += bzgrid->reclat[j][i] * bzgrid->reclat[j][i];
        }
        length[i] /= bzgrid->D_diag[i] * bzgrid->D_diag[i];
    }
    tolerance = length[0];
    for (i = 1; i < 3; i++) {
        if (tolerance < length[i]) {
            tolerance = length[i];
        }
    }
    tolerance *= GRID_TOLERANCE_FACTOR;

    return tolerance;
}

RecgridMats *recgrid_alloc_RotMats(const int64_t size) {
    RecgridMats *rotmats;

    rotmats = NULL;

    if ((rotmats = (RecgridMats *)malloc(sizeof(RecgridMats))) == NULL) {
        warning_print("Memory could not be allocated.");
        return NULL;
    }

    rotmats->size = size;
    if (size > 0) {
        if ((rotmats->mat = (int64_t (*)[3][3])malloc(sizeof(int64_t[3][3]) *
                                                      size)) == NULL) {
            warning_print("Memory could not be allocated ");
            warning_print("(RecgridMats, line %d, %s).\n", __LINE__, __FILE__);
            free(rotmats);
            rotmats = NULL;
            return NULL;
        }
    }
    return rotmats;
}

void recgrid_free_RotMats(RecgridMats *rotmats) {
    if (rotmats->size > 0) {
        free(rotmats->mat);
        rotmats->mat = NULL;
    }
    free(rotmats);
}
