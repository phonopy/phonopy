/* Copyright (C) 2014 Atsushi Togo */
/* All rights reserved. */

/* This file was originally part of spglib and is part of kspclib. */

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
/* tetrahedron_method.c */
/* Copyright (C) 2014 Atsushi Togo */

#include "tetrahedron_method.h"

#include <stddef.h>

#ifdef THMWARNING
#include <stdio.h>
#define warning_print(...) fprintf(stderr, __VA_ARGS__)
#else
#define warning_print(...)
#endif

#ifdef THM_EPSILON
#include <math.h>
#endif

/*      6-------7             */
/*     /|      /|             */
/*    / |     / |             */
/*   4-------5  |             */
/*   |  2----|--3             */
/*   | /     | /              */
/*   |/      |/                       */
/*   0-------1                */
/*                            */
/*  i: vec        neighbours  */
/*  0: O          1, 2, 4     */
/*  1: a          0, 3, 5     */
/*  2: b          0, 3, 6     */
/*  3: a + b      1, 2, 7     */
/*  4: c          0, 5, 6     */
/*  5: c + a      1, 4, 7     */
/*  6: c + b      2, 4, 7     */
/*  7: c + a + b  3, 5, 6     */

static long main_diagonals[4][3] = {{1, 1, 1},   /* 0-7 */
                                    {-1, 1, 1},  /* 1-6 */
                                    {1, -1, 1},  /* 2-5 */
                                    {1, 1, -1}}; /* 3-4 */

static long db_relative_grid_address[4][24][4][3] = {
    {
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 1, 0},
            {1, 1, 1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 1, 1},
            {1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {1, 0, 1},
            {1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 1},
            {1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 1, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 0, 1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {1, 0, 1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {-1, -1, 0},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {-1, -1, 0},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 1, 0},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {-1, 0, -1},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {-1, 0, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, -1, -1},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, -1, -1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, -1},
            {0, -1, -1},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {-1, -1, -1},
            {0, -1, -1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, -1},
            {-1, 0, -1},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {-1, -1, -1},
            {-1, 0, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, -1},
            {-1, -1, 0},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, -1},
            {-1, -1, 0},
            {-1, 0, 0},
        },
    },
    {
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 1, 1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 0, 1},
            {0, 1, 1},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {-1, 1, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, 0, 1},
            {-1, 1, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {0, 1, 0},
            {-1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {-1, 1, 1},
            {0, 1, 1},
        },
        {
            {0, 0, 0},
            {-1, 0, 1},
            {0, 0, 1},
            {-1, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {-1, 1, 1},
            {0, 1, 1},
        },
        {
            {0, 0, 0},
            {0, 0, 1},
            {0, -1, 0},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 0, 1},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, 0, 1},
            {0, -1, 0},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, 0, 1},
            {0, 0, 1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 0, -1},
            {1, 0, -1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {1, 0, -1},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {0, 0, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {0, 1, 0},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {0, -1, -1},
            {1, -1, -1},
            {0, 0, -1},
        },
        {
            {0, 0, 0},
            {0, -1, -1},
            {1, -1, -1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {1, -1, -1},
            {0, 0, -1},
            {1, 0, -1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, -1, -1},
            {1, 0, -1},
        },
        {
            {0, 0, 0},
            {1, -1, -1},
            {0, -1, 0},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, -1, -1},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {0, -1, -1},
            {0, 0, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, -1, -1},
            {0, -1, 0},
            {-1, 0, 0},
        },
    },
    {
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {1, 0, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 0, 1},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {0, 0, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {1, -1, 1},
            {0, -1, 0},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {0, -1, 1},
            {1, -1, 1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, -1, 1},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, -1, 1},
            {1, 0, 1},
        },
        {
            {0, 0, 0},
            {0, -1, 1},
            {1, -1, 1},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {1, -1, 1},
            {0, 0, 1},
            {1, 0, 1},
        },
        {
            {0, 0, 0},
            {0, -1, 1},
            {0, -1, 0},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, -1, 1},
            {0, 0, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 0, -1},
            {0, 1, -1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 1, -1},
        },
        {
            {0, 0, 0},
            {-1, 0, -1},
            {0, 0, -1},
            {-1, 1, -1},
        },
        {
            {0, 0, 0},
            {-1, 0, -1},
            {-1, 1, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {-1, 1, -1},
            {0, 1, -1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {-1, 1, -1},
            {0, 1, -1},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {-1, 1, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, 1, 0},
            {0, 1, 0},
            {-1, 1, -1},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {0, -1, 0},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, 0, -1},
            {1, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, 0, -1},
            {0, 0, -1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, 0, -1},
            {0, -1, 0},
            {-1, 0, 0},
        },
    },
    {
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 1, 0},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {-1, 0, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {-1, 0, 1},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, -1, 1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {0, -1, 1},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {-1, -1, 1},
            {-1, -1, 0},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, 1},
            {-1, -1, 0},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, 1},
            {0, -1, 1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, 1},
            {-1, 0, 1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {-1, -1, 1},
            {0, -1, 1},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {-1, -1, 1},
            {-1, 0, 1},
            {0, 0, 1},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {1, 0, -1},
            {1, 1, -1},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {0, 1, -1},
            {1, 1, -1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 0, -1},
            {1, 1, -1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 1, -1},
            {1, 1, -1},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 1, 0},
            {1, 1, -1},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {1, 1, -1},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {0, 1, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, 1, 0},
            {0, 1, -1},
            {-1, 0, 0},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {1, 0, -1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {1, 0, 0},
            {1, 0, -1},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {-1, -1, 0},
            {0, -1, 0},
        },
        {
            {0, 0, 0},
            {0, 0, -1},
            {-1, -1, 0},
            {-1, 0, 0},
        },
    },
};

static double get_integration_weight(
    const double omega, const double tetrahedra_omegas[24][4],
    double (*gn)(const long, const double, const double[4]),
    double (*IJ)(const long, const long, const double, const double[4]));
static long get_main_diagonal(const double rec_lattice[3][3]);
static long sort_omegas(double v[4]);
static double norm_squared_d3(const double a[3]);
static void multiply_matrix_vector_dl3(double v[3], const double a[3][3],
                                       const long b[3]);
static double _f(const long n, const long m, const double omega,
                 const double vertices_omegas[4]);
static double _J(const long i, const long ci, const double omega,
                 const double vertices_omegas[4]);
static double _I(const long i, const long ci, const double omega,
                 const double vertices_omegas[4]);
static double _n(const long i, const double omega,
                 const double vertices_omegas[4]);
static double _g(const long i, const double omega,
                 const double vertices_omegas[4]);
static double _n_0(void);
static double _n_1(const double omega, const double vertices_omegas[4]);
static double _n_2(const double omega, const double vertices_omegas[4]);
static double _n_3(const double omega, const double vertices_omegas[4]);
static double _n_4(void);
static double _g_0(void);
static double _g_1(const double omega, const double vertices_omegas[4]);
static double _g_2(const double omega, const double vertices_omegas[4]);
static double _g_3(const double omega, const double vertices_omegas[4]);
static double _g_4(void);
static double _J_0(void);
static double _J_10(const double omega, const double vertices_omegas[4]);
static double _J_11(const double omega, const double vertices_omegas[4]);
static double _J_12(const double omega, const double vertices_omegas[4]);
static double _J_13(const double omega, const double vertices_omegas[4]);
static double _J_20(const double omega, const double vertices_omegas[4]);
static double _J_21(const double omega, const double vertices_omegas[4]);
static double _J_22(const double omega, const double vertices_omegas[4]);
static double _J_23(const double omega, const double vertices_omegas[4]);
static double _J_30(const double omega, const double vertices_omegas[4]);
static double _J_31(const double omega, const double vertices_omegas[4]);
static double _J_32(const double omega, const double vertices_omegas[4]);
static double _J_33(const double omega, const double vertices_omegas[4]);
static double _J_4(void);
static double _I_0(void);
static double _I_10(const double omega, const double vertices_omegas[4]);
static double _I_11(const double omega, const double vertices_omegas[4]);
static double _I_12(const double omega, const double vertices_omegas[4]);
static double _I_13(const double omega, const double vertices_omegas[4]);
static double _I_20(const double omega, const double vertices_omegas[4]);
static double _I_21(const double omega, const double vertices_omegas[4]);
static double _I_22(const double omega, const double vertices_omegas[4]);
static double _I_23(const double omega, const double vertices_omegas[4]);
static double _I_30(const double omega, const double vertices_omegas[4]);
static double _I_31(const double omega, const double vertices_omegas[4]);
static double _I_32(const double omega, const double vertices_omegas[4]);
static double _I_33(const double omega, const double vertices_omegas[4]);
static double _I_4(void);

void thm_get_relative_grid_address(long relative_grid_address[24][4][3],
                                   const double rec_lattice[3][3]) {
    long i, j, k, main_diag_index;

    main_diag_index = get_main_diagonal(rec_lattice);

    for (i = 0; i < 24; i++) {
        for (j = 0; j < 4; j++) {
            for (k = 0; k < 3; k++) {
                relative_grid_address[i][j][k] =
                    db_relative_grid_address[main_diag_index][i][j][k];
            }
        }
    }
}

void thm_get_all_relative_grid_address(
    long relative_grid_address[4][24][4][3]) {
    long i, j, k, main_diag_index;

    for (main_diag_index = 0; main_diag_index < 4; main_diag_index++) {
        for (i = 0; i < 24; i++) {
            for (j = 0; j < 4; j++) {
                for (k = 0; k < 3; k++) {
                    relative_grid_address[main_diag_index][i][j][k] =
                        db_relative_grid_address[main_diag_index][i][j][k];
                }
            }
        }
    }
}

double thm_get_integration_weight(const double omega,
                                  const double tetrahedra_omegas[24][4],
                                  const char function) {
    if (function == 'I') {
        return get_integration_weight(omega, tetrahedra_omegas, _g, _I);
    } else {
        return get_integration_weight(omega, tetrahedra_omegas, _n, _J);
    }
}

long thm_in_tetrahedra(const double f0, const double freq_vertices[24][4]) {
    long i, j;
    double fmin, fmax;

    fmin = freq_vertices[0][0];
    fmax = freq_vertices[0][0];

    for (i = 0; i < 24; i++) {
        for (j = 0; j < 4; j++) {
            if (fmin > freq_vertices[i][j]) {
                fmin = freq_vertices[i][j];
            }
            if (fmax < freq_vertices[i][j]) {
                fmax = freq_vertices[i][j];
            }
        }
    }

    if (fmin > f0 || fmax < f0) {
        return 0;
    } else {
        return 1;
    }
}

static double get_integration_weight(
    const double omega, const double tetrahedra_omegas[24][4],
    double (*gn)(const long, const double, const double[4]),
    double (*IJ)(const long, const long, const double, const double[4])) {
    long i, j, ci;
    double sum;
    double v[4];

    sum = 0;
    for (i = 0; i < 24; i++) {
        for (j = 0; j < 4; j++) {
            v[j] = tetrahedra_omegas[i][j];
        }
        ci = sort_omegas(v);
        if (omega < v[0]) {
            sum += IJ(0, ci, omega, v) * gn(0, omega, v);
        } else {
            if (v[0] < omega && omega < v[1]) {
                sum += IJ(1, ci, omega, v) * gn(1, omega, v);
            } else {
                if (v[1] < omega && omega < v[2]) {
                    sum += IJ(2, ci, omega, v) * gn(2, omega, v);
                } else {
                    if (v[2] < omega && omega < v[3]) {
                        sum += IJ(3, ci, omega, v) * gn(3, omega, v);
                    } else {
                        if (v[3] < omega) {
                            sum += IJ(4, ci, omega, v) * gn(4, omega, v);
                        }
                    }
                }
            }
        }
    }
    return sum / 6;
}

static long sort_omegas(double v[4]) {
    long i;
    double w[4];

    i = 0;

    if (v[0] > v[1]) {
        w[0] = v[1];
        w[1] = v[0];
        i = 1;
    } else {
        w[0] = v[0];
        w[1] = v[1];
    }

    if (v[2] > v[3]) {
        w[2] = v[3];
        w[3] = v[2];
    } else {
        w[2] = v[2];
        w[3] = v[3];
    }

    if (w[0] > w[2]) {
        v[0] = w[2];
        v[1] = w[0];
        if (i == 0) {
            i = 4;
        }
    } else {
        v[0] = w[0];
        v[1] = w[2];
    }

    if (w[1] > w[3]) {
        v[3] = w[1];
        v[2] = w[3];
        if (i == 1) {
            i = 3;
        }
    } else {
        v[3] = w[3];
        v[2] = w[1];
        if (i == 1) {
            i = 5;
        }
    }

    if (v[1] > v[2]) {
        w[1] = v[1];
        v[1] = v[2];
        v[2] = w[1];
        if (i == 4) {
            i = 2;
        }
        if (i == 5) {
            i = 1;
        }
    } else {
        if (i == 4) {
            i = 1;
        }
        if (i == 5) {
            i = 2;
        }
    }
    return i;
}

static long get_main_diagonal(const double rec_lattice[3][3]) {
    long i, shortest;
    double length, min_length;
    double main_diag[3];

    shortest = 0;
    multiply_matrix_vector_dl3(main_diag, rec_lattice, main_diagonals[0]);
    min_length = norm_squared_d3(main_diag);
    for (i = 1; i < 4; i++) {
        multiply_matrix_vector_dl3(main_diag, rec_lattice, main_diagonals[i]);
        length = norm_squared_d3(main_diag);
        if (min_length > length) {
            min_length = length;
            shortest = i;
        }
    }
    return shortest;
}

static double norm_squared_d3(const double a[3]) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

static void multiply_matrix_vector_dl3(double v[3], const double a[3][3],
                                       const long b[3]) {
    long i;
    double c[3];

    for (i = 0; i < 3; i++) {
        c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
    }

    for (i = 0; i < 3; i++) {
        v[i] = c[i];
    }
}

static double _f(const long n, const long m, const double omega,
                 const double vertices_omegas[4]) {
    double delta;
    delta = vertices_omegas[n] - vertices_omegas[m];

#ifdef THM_EPSILON
    if (fabs(delta) < THM_EPSILON) {
        return 0;
    }
#endif

    return ((omega - vertices_omegas[m]) / delta);
}

static double _J(const long i, const long ci, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _J_0();
        case 1:
            switch (ci) {
                case 0:
                    return _J_10(omega, vertices_omegas);
                case 1:
                    return _J_11(omega, vertices_omegas);
                case 2:
                    return _J_12(omega, vertices_omegas);
                case 3:
                    return _J_13(omega, vertices_omegas);
            }
        case 2:
            switch (ci) {
                case 0:
                    return _J_20(omega, vertices_omegas);
                case 1:
                    return _J_21(omega, vertices_omegas);
                case 2:
                    return _J_22(omega, vertices_omegas);
                case 3:
                    return _J_23(omega, vertices_omegas);
            }
        case 3:
            switch (ci) {
                case 0:
                    return _J_30(omega, vertices_omegas);
                case 1:
                    return _J_31(omega, vertices_omegas);
                case 2:
                    return _J_32(omega, vertices_omegas);
                case 3:
                    return _J_33(omega, vertices_omegas);
            }
        case 4:
            return _J_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" J is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

static double _I(const long i, const long ci, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _I_0();
        case 1:
            switch (ci) {
                case 0:
                    return _I_10(omega, vertices_omegas);
                case 1:
                    return _I_11(omega, vertices_omegas);
                case 2:
                    return _I_12(omega, vertices_omegas);
                case 3:
                    return _I_13(omega, vertices_omegas);
            }
        case 2:
            switch (ci) {
                case 0:
                    return _I_20(omega, vertices_omegas);
                case 1:
                    return _I_21(omega, vertices_omegas);
                case 2:
                    return _I_22(omega, vertices_omegas);
                case 3:
                    return _I_23(omega, vertices_omegas);
            }
        case 3:
            switch (ci) {
                case 0:
                    return _I_30(omega, vertices_omegas);
                case 1:
                    return _I_31(omega, vertices_omegas);
                case 2:
                    return _I_32(omega, vertices_omegas);
                case 3:
                    return _I_33(omega, vertices_omegas);
            }
        case 4:
            return _I_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" I is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

static double _n(const long i, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _n_0();
        case 1:
            return _n_1(omega, vertices_omegas);
        case 2:
            return _n_2(omega, vertices_omegas);
        case 3:
            return _n_3(omega, vertices_omegas);
        case 4:
            return _n_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" n is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

static double _g(const long i, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _g_0();
        case 1:
            return _g_1(omega, vertices_omegas);
        case 2:
            return _g_2(omega, vertices_omegas);
        case 3:
            return _g_3(omega, vertices_omegas);
        case 4:
            return _g_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" g is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

/* omega < omega1 */
static double _n_0(void) { return 0.0; }

/* omega1 < omega < omega2 */
static double _n_1(const double omega, const double vertices_omegas[4]) {
    return (_f(1, 0, omega, vertices_omegas) *
            _f(2, 0, omega, vertices_omegas) *
            _f(3, 0, omega, vertices_omegas));
}

/* omega2 < omega < omega3 */
static double _n_2(const double omega, const double vertices_omegas[4]) {
    return (
        _f(3, 1, omega, vertices_omegas) * _f(2, 1, omega, vertices_omegas) +
        _f(3, 0, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas) *
            _f(2, 1, omega, vertices_omegas) +
        _f(3, 0, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) *
            _f(1, 2, omega, vertices_omegas));
}

/* omega2 < omega < omega3 */
static double _n_3(const double omega, const double vertices_omegas[4]) {
    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas));
}

/* omega4 < omega */
static double _n_4(void) { return 1.0; }

/* omega < omega1 */
static double _g_0(void) { return 0.0; }

/* omega1 < omega < omega2 */
static double _g_1(const double omega, const double vertices_omegas[4]) {
    return (3 * _f(1, 0, omega, vertices_omegas) *
            _f(2, 0, omega, vertices_omegas) /
            (vertices_omegas[3] - vertices_omegas[0]));
}

/* omega2 < omega < omega3 */
static double _g_2(const double omega, const double vertices_omegas[4]) {
    return (
        3 / (vertices_omegas[3] - vertices_omegas[0]) *
        (_f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) +
         _f(2, 1, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas)));
}

/* omega3 < omega < omega4 */
static double _g_3(const double omega, const double vertices_omegas[4]) {
    return (3 * _f(1, 3, omega, vertices_omegas) *
            _f(2, 3, omega, vertices_omegas) /
            (vertices_omegas[3] - vertices_omegas[0]));
}

/* omega4 < omega */
static double _g_4(void) { return 0.0; }

static double _J_0(void) { return 0.0; }

static double _J_10(const double omega, const double vertices_omegas[4]) {
    return (1.0 + _f(0, 1, omega, vertices_omegas) +
            _f(0, 2, omega, vertices_omegas) +
            _f(0, 3, omega, vertices_omegas)) /
           4;
}

static double _J_11(const double omega, const double vertices_omegas[4]) {
    return _f(1, 0, omega, vertices_omegas) / 4;
}

static double _J_12(const double omega, const double vertices_omegas[4]) {
    return _f(2, 0, omega, vertices_omegas) / 4;
}

static double _J_13(const double omega, const double vertices_omegas[4]) {
    return _f(3, 0, omega, vertices_omegas) / 4;
}

static double _J_20(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (1.0 + _f(0, 3, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                (1.0 + _f(0, 3, omega, vertices_omegas) +
                 _f(0, 2, omega, vertices_omegas))) /
           4 / n;
}

static double _J_21(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (1.0 + _f(1, 3, omega, vertices_omegas) +
                 _f(1, 2, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (_f(1, 3, omega, vertices_omegas) +
                 _f(1, 2, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas)) /
           4 / n;
}

static double _J_22(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                (_f(2, 1, omega, vertices_omegas) +
                 _f(2, 0, omega, vertices_omegas))) /
           4 / n;
}

static double _J_23(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                _f(3, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (_f(3, 1, omega, vertices_omegas) +
                 _f(3, 0, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                _f(3, 0, omega, vertices_omegas)) /
           4 / n;
}

static double _J_30(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas)) /
           4 / n;
}

static double _J_31(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas)) /
           4 / n;
}

static double _J_32(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas)) /
           4 / n;
}

static double _J_33(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas) *
                      (1.0 + _f(3, 0, omega, vertices_omegas) +
                       _f(3, 1, omega, vertices_omegas) +
                       _f(3, 2, omega, vertices_omegas))) /
           4 / n;
}

static double _J_4(void) { return 0.25; }

static double _I_0(void) { return 0.0; }

static double _I_10(const double omega, const double vertices_omegas[4]) {
    return (_f(0, 1, omega, vertices_omegas) +
            _f(0, 2, omega, vertices_omegas) +
            _f(0, 3, omega, vertices_omegas)) /
           3;
}

static double _I_11(const double omega, const double vertices_omegas[4]) {
    return _f(1, 0, omega, vertices_omegas) / 3;
}

static double _I_12(const double omega, const double vertices_omegas[4]) {
    return _f(2, 0, omega, vertices_omegas) / 3;
}

static double _I_13(const double omega, const double vertices_omegas[4]) {
    return _f(3, 0, omega, vertices_omegas) / 3;
}

static double _I_20(const double omega, const double vertices_omegas[4]) {
    double f12_20, g;
    f12_20 =
        _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas);
    g = f12_20 +
        _f(2, 1, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas);

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(0, 3, omega, vertices_omegas) +
            _f(0, 2, omega, vertices_omegas) * f12_20 / g) /
           3;
}

static double _I_21(const double omega, const double vertices_omegas[4]) {
    double f13_21, g;
    f13_21 =
        _f(1, 3, omega, vertices_omegas) * _f(2, 1, omega, vertices_omegas);
    g = _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) +
        f13_21;

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(1, 2, omega, vertices_omegas) +
            _f(1, 3, omega, vertices_omegas) * f13_21 / g) /
           3;
}

static double _I_22(const double omega, const double vertices_omegas[4]) {
    double f12_20, g;
    f12_20 =
        _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas);
    g = f12_20 +
        _f(2, 1, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas);

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(2, 1, omega, vertices_omegas) +
            _f(2, 0, omega, vertices_omegas) * f12_20 / g) /
           3;
}

static double _I_23(const double omega, const double vertices_omegas[4]) {
    double f13_21, g;
    f13_21 =
        _f(1, 3, omega, vertices_omegas) * _f(2, 1, omega, vertices_omegas);
    g = _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) +
        f13_21;

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 0, omega, vertices_omegas) +
            _f(3, 1, omega, vertices_omegas) * f13_21 / g) /
           3;
}

static double _I_30(const double omega, const double vertices_omegas[4]) {
    return _f(0, 3, omega, vertices_omegas) / 3;
}

static double _I_31(const double omega, const double vertices_omegas[4]) {
    return _f(1, 3, omega, vertices_omegas) / 3;
}

static double _I_32(const double omega, const double vertices_omegas[4]) {
    return _f(2, 3, omega, vertices_omegas) / 3;
}

static double _I_33(const double omega, const double vertices_omegas[4]) {
    return (_f(3, 0, omega, vertices_omegas) +
            _f(3, 1, omega, vertices_omegas) +
            _f(3, 2, omega, vertices_omegas)) /
           3;
}

static double _I_4(void) { return 0.0; }
