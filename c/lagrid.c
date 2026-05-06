/* Copyright (C) 2021 Atsushi Togo */
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

#include "lagrid.h"

#include <stdint.h>

int64_t lagmat_get_determinant_l3(const int64_t a[3][3]) {
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) +
           a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) +
           a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

double lagmat_get_determinant_d3(const double a[3][3]) {
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) +
           a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) +
           a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

void lagmat_cast_matrix_3l_to_3d(double m[3][3], const int64_t a[3][3]) {
    m[0][0] = a[0][0];
    m[0][1] = a[0][1];
    m[0][2] = a[0][2];
    m[1][0] = a[1][0];
    m[1][1] = a[1][1];
    m[1][2] = a[1][2];
    m[2][0] = a[2][0];
    m[2][1] = a[2][1];
    m[2][2] = a[2][2];
}

void lagmat_cast_matrix_3d_to_3l(int64_t m[3][3], const double a[3][3]) {
    m[0][0] = lagmat_Nint(a[0][0]);
    m[0][1] = lagmat_Nint(a[0][1]);
    m[0][2] = lagmat_Nint(a[0][2]);
    m[1][0] = lagmat_Nint(a[1][0]);
    m[1][1] = lagmat_Nint(a[1][1]);
    m[1][2] = lagmat_Nint(a[1][2]);
    m[2][0] = lagmat_Nint(a[2][0]);
    m[2][1] = lagmat_Nint(a[2][1]);
    m[2][2] = lagmat_Nint(a[2][2]);
}

int64_t lagmat_get_similar_matrix_ld3(double m[3][3], const int64_t a[3][3],
                                      const double b[3][3],
                                      const double precision) {
    double c[3][3];
    if (!lagmat_inverse_matrix_d3(c, b, precision)) {
        warning_print("No similar matrix due to 0 determinant.\n");
        return 0;
    }
    lagmat_multiply_matrix_ld3(m, a, b);
    lagmat_multiply_matrix_d3(m, c, m);
    return 1;
}

int64_t lagmat_check_identity_matrix_l3(const int64_t a[3][3],
                                        const int64_t b[3][3]) {
    if (a[0][0] - b[0][0] || a[0][1] - b[0][1] || a[0][2] - b[0][2] ||
        a[1][0] - b[1][0] || a[1][1] - b[1][1] || a[1][2] - b[1][2] ||
        a[2][0] - b[2][0] || a[2][1] - b[2][1] || a[2][2] - b[2][2]) {
        return 0;
    } else {
        return 1;
    }
}

int64_t lagmat_check_identity_matrix_ld3(const int64_t a[3][3],
                                         const double b[3][3],
                                         const double symprec) {
    if (lagmat_Dabs(a[0][0] - b[0][0]) > symprec ||
        lagmat_Dabs(a[0][1] - b[0][1]) > symprec ||
        lagmat_Dabs(a[0][2] - b[0][2]) > symprec ||
        lagmat_Dabs(a[1][0] - b[1][0]) > symprec ||
        lagmat_Dabs(a[1][1] - b[1][1]) > symprec ||
        lagmat_Dabs(a[1][2] - b[1][2]) > symprec ||
        lagmat_Dabs(a[2][0] - b[2][0]) > symprec ||
        lagmat_Dabs(a[2][1] - b[2][1]) > symprec ||
        lagmat_Dabs(a[2][2] - b[2][2]) > symprec) {
        return 0;
    } else {
        return 1;
    }
}

int64_t lagmat_inverse_matrix_d3(double m[3][3], const double a[3][3],
                                 const double precision) {
    double det;
    double c[3][3];
    det = lagmat_get_determinant_d3(a);
    if (lagmat_Dabs(det) < precision) {
        warning_print("No inverse matrix (det=%f)\n", det);
        return 0;
    }

    c[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
    c[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    c[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
    c[0][1] = (a[2][1] * a[0][2] - a[2][2] * a[0][1]) / det;
    c[1][1] = (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / det;
    c[2][1] = (a[2][0] * a[0][1] - a[2][1] * a[0][0]) / det;
    c[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    c[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
    c[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
    lagmat_copy_matrix_d3(m, c);
    return 1;
}

void lagmat_transpose_matrix_l3(int64_t a[3][3], const int64_t b[3][3]) {
    int64_t c[3][3];
    c[0][0] = b[0][0];
    c[0][1] = b[1][0];
    c[0][2] = b[2][0];
    c[1][0] = b[0][1];
    c[1][1] = b[1][1];
    c[1][2] = b[2][1];
    c[2][0] = b[0][2];
    c[2][1] = b[1][2];
    c[2][2] = b[2][2];
    lagmat_copy_matrix_l3(a, c);
}

void lagmat_multiply_matrix_vector_l3(int64_t v[3], const int64_t a[3][3],
                                      const int64_t b[3]) {
    int64_t i;
    int64_t c[3];
    for (i = 0; i < 3; i++) {
        c[i] = a[i][0] * b[0] + a[i][1] * b[1] + a[i][2] * b[2];
    }
    for (i = 0; i < 3; i++) {
        v[i] = c[i];
    }
}

void lagmat_multiply_matrix_l3(int64_t m[3][3], const int64_t a[3][3],
                               const int64_t b[3][3]) {
    int64_t i, j; /* a_ij */
    int64_t c[3][3];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    lagmat_copy_matrix_l3(m, c);
}

void lagmat_multiply_matrix_ld3(double m[3][3], const int64_t a[3][3],
                                const double b[3][3]) {
    int64_t i, j; /* a_ij */
    double c[3][3];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    lagmat_copy_matrix_d3(m, c);
}

void lagmat_multiply_matrix_d3(double m[3][3], const double a[3][3],
                               const double b[3][3]) {
    int64_t i, j; /* a_ij */
    double c[3][3];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    lagmat_copy_matrix_d3(m, c);
}

void lagmat_copy_matrix_l3(int64_t a[3][3], const int64_t b[3][3]) {
    a[0][0] = b[0][0];
    a[0][1] = b[0][1];
    a[0][2] = b[0][2];
    a[1][0] = b[1][0];
    a[1][1] = b[1][1];
    a[1][2] = b[1][2];
    a[2][0] = b[2][0];
    a[2][1] = b[2][1];
    a[2][2] = b[2][2];
}

void lagmat_copy_matrix_d3(double a[3][3], const double b[3][3]) {
    a[0][0] = b[0][0];
    a[0][1] = b[0][1];
    a[0][2] = b[0][2];
    a[1][0] = b[1][0];
    a[1][1] = b[1][1];
    a[1][2] = b[1][2];
    a[2][0] = b[2][0];
    a[2][1] = b[2][1];
    a[2][2] = b[2][2];
}

void lagmat_copy_vector_l3(int64_t a[3], const int64_t b[3]) {
    a[0] = b[0];
    a[1] = b[1];
    a[2] = b[2];
}

int64_t lagmat_modulo_l(const int64_t a, const int64_t b) {
    int64_t c;
    c = a % b;
    if (c < 0) {
        c += b;
    }
    return c;
}

int64_t lagmat_Nint(const double a) {
    if (a < 0.0)
        return (int64_t)(a - 0.5);
    else
        return (int64_t)(a + 0.5);
}

double lagmat_Dabs(const double a) {
    if (a < 0.0)
        return -a;
    else
        return a;
}
