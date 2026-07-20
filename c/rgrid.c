/* SPDX-License-Identifier: BSD-3-Clause */

#include "rgrid.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static int64_t get_double_grid_index(const int64_t address_double[3],
                                     const int64_t mesh[3]);
static int64_t get_grid_index_single_mesh(const int64_t address[3],
                                          const int64_t mesh[3]);
static void reduce_double_grid_address(int64_t address[3],
                                       const int64_t mesh[3]);
static int64_t mat_modulo_l(const int64_t a, const int64_t b);

int64_t rgd_get_double_grid_index(const int64_t address_double[3],
                                  const int64_t mesh[3]) {
    return get_double_grid_index(address_double, mesh);
}

void rgd_get_double_grid_address(int64_t address_double[3],
                                 const int64_t address[3],
                                 const int64_t mesh[3],
                                 const int64_t is_shift[3]) {
    int64_t i;

    for (i = 0; i < 3; i++) {
        address_double[i] = address[i] * 2 + (is_shift[i] != 0);
    }
    reduce_double_grid_address(address_double, mesh);
}

static int64_t get_double_grid_index(const int64_t address_double[3],
                                     const int64_t mesh[3]) {
    int64_t i;
    int64_t address[3];

    for (i = 0; i < 3; i++) {
        if (address_double[i] % 2 == 0) {
            address[i] = address_double[i] / 2;
        } else {
            address[i] = (address_double[i] - 1) / 2;
        }
        address[i] = mat_modulo_l(address[i], mesh[i]);
    }

    return get_grid_index_single_mesh(address, mesh);
}

static int64_t get_grid_index_single_mesh(const int64_t address[3],
                                          const int64_t mesh[3]) {
#ifndef GRID_ORDER_XYZ
    return (address[2] * mesh[0] * (int64_t)(mesh[1]) + address[1] * mesh[0] +
            address[0]);
#else
    return (address[0] * mesh[1] * (int64_t)(mesh[2]) + address[1] * mesh[2] +
            address[2]);
#endif
}

static void reduce_double_grid_address(int64_t address[3],
                                       const int64_t mesh[3]) {
    int64_t i;

    for (i = 0; i < 3; i++) {
#ifndef GRID_BOUNDARY_AS_NEGATIVE
        address[i] -= 2 * mesh[i] * (address[i] > mesh[i]);
#else
        address[i] -= 2 * mesh[i] * (address[i] > mesh[i] - 1);
#endif
    }
}

static int64_t mat_modulo_l(const int64_t a, const int64_t b) {
    int64_t c;
    c = a % b;
    if (c < 0) {
        c += b;
    }
    return c;
}
