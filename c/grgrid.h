/* SPDX-License-Identifier: BSD-3-Clause */

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
