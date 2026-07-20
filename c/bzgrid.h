/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __bzgrid_H__
#define __bzgrid_H__

#include <stdint.h>

#include "recgrid.h"

int64_t bzg_rotate_grid_index(const int64_t grid_index,
                              const int64_t rotation[3][3],
                              const RecgridConstBZGrid *bzgrid);
int64_t bzg_get_bz_grid_addresses(RecgridBZGrid *bzgrid);

#endif
