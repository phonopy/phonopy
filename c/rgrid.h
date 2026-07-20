/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __rgrid_H__
#define __rgrid_H__

#include <stdint.h>

/* #define GRID_ORDER_XYZ */
/* This changes behaviour of index order of address. */
/* Without GRID_ORDER_XYZ, left most element of address runs first. */
/* grid_address (e.g. 4x4x4 mesh, unless GRID_ORDER_XYZ is defined) */
/*    [[ 0  0  0]                                                   */
/*     [ 1  0  0]                                                   */
/*     [ 2  0  0]                                                   */
/*     [-1  0  0]                                                   */
/*     [ 0  1  0]                                                   */
/*     [ 1  1  0]                                                   */
/*     [ 2  1  0]                                                   */
/*     [-1  1  0]                                                   */
/*     ....      ]                                                  */
/*                                                                  */
/* With GRID_ORDER_XYZ, right most element of address runs first.   */
/* grid_address (e.g. 4x4x4 mesh, if GRID_ORDER_XYZ is defined)     */
/*    [[ 0  0  0]                                                   */
/*     [ 0  0  1]                                                   */
/*     [ 0  0  2]                                                   */
/*     [ 0  0 -1]                                                   */
/*     [ 0  1  0]                                                   */
/*     [ 0  1  1]                                                   */
/*     [ 0  1  2]                                                   */
/*     [ 0  1 -1]                                                   */
/*     ....      ]                                                  */

/* #define GRID_BOUNDARY_AS_NEGATIVE */
/* This changes the behaviour of address elements on the surface of  */
/* parallelepiped. */
/* For odd mesh number, this affects nothing, e.g., [-2, -1, 0, 1, 2]. */
/* regardless of with and without GRID_BOUNDARY_AS_NEGATIVE. */
/* For even mesh number, this affects as follows: */
/* without GRID_BOUNDARY_AS_NEGATIVE, e.g., [-2, -1, 0, 1, 2, 3]. */
/* with GRID_BOUNDARY_AS_NEGATIVE, e.g., [-3, -2, -1, 0, 1, 2]. */

int64_t rgd_get_double_grid_index(const int64_t address_double[3],
                                  const int64_t mesh[3]);
void rgd_get_double_grid_address(int64_t address_double[3],
                                 const int64_t address[3],
                                 const int64_t mesh[3],
                                 const int64_t is_shift[3]);

#endif
