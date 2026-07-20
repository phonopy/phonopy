/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __tetrahedron_method_H__
#define __tetrahedron_method_H__

#include <stddef.h>
#include <stdint.h>

void thm_get_relative_grid_address(int64_t relative_grid_address[24][4][3],
                                   const double rec_lattice[3][3]);
void thm_get_all_relative_grid_address(
    int64_t relative_grid_address[4][24][4][3]);
double thm_get_integration_weight(const double omega,
                                  const double tetrahedra_omegas[24][4],
                                  const char function);
int64_t thm_in_tetrahedra(const double f0, const double freq_vertices[24][4]);

#endif
