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

#ifndef __grgrid_H__
#define __grgrid_H__

#include <stddef.h>
#include <stdint.h>

int64_t grg_get_snf3x3(int64_t D_diag[3], int64_t P[3][3], int64_t Q[3][3],
                       const int64_t A[3][3]);
int64_t grg_transform_rotations(int64_t (*transformed_rots)[3][3],
                                const int64_t (*rotations)[3][3],
                                const int64_t num_rot, const int64_t D_diag[3],
                                const int64_t Q[3][3]);
void grg_get_all_grid_addresses(int64_t (*grid_address)[3],
                                const int64_t D_diag[3]);
void grg_get_double_grid_address(int64_t address_double[3],
                                 const int64_t address[3], const int64_t PS[3]);
void grg_get_grid_address(int64_t address[3], const int64_t address_double[3],
                          const int64_t PS[3]);
int64_t grg_get_grid_index(const int64_t address[3], const int64_t D_diag[3]);
int64_t grg_get_double_grid_index(const int64_t address_double[3],
                                  const int64_t D_diag[3], const int64_t PS[3]);
void grg_get_grid_address_from_index(int64_t address[3],
                                     const int64_t grid_index,
                                     const int64_t D_diag[3]);
int64_t grg_rotate_grid_index(const int64_t grid_index,
                              const int64_t rotations[3][3],
                              const int64_t D_diag[3], const int64_t PS[3]);
void grg_get_ir_grid_map(int64_t *ir_grid_map, const int64_t (*rotations)[3][3],
                         const int64_t num_rot, const int64_t D_diag[3],
                         const int64_t PS[3]);
int64_t grg_get_reciprocal_point_group(int64_t rec_rotations[48][3][3],
                                       const int64_t (*rotations)[3][3],
                                       const int64_t num_rot,
                                       const int64_t is_time_reversal,
                                       const int64_t is_transpose);

#endif
