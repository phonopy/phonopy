/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef __lagrid_H__
#define __lagrid_H__

#include <stdint.h>

#ifdef LAGWARNING
#define warning_print(...) fprintf(stderr, __VA_ARGS__)
#else
#define warning_print(...)
#endif

int64_t lagmat_get_determinant_l3(const int64_t a[3][3]);
double lagmat_get_determinant_d3(const double a[3][3]);
void lagmat_cast_matrix_3l_to_3d(double m[3][3], const int64_t a[3][3]);
void lagmat_cast_matrix_3d_to_3l(int64_t m[3][3], const double a[3][3]);
int64_t lagmat_get_similar_matrix_ld3(double m[3][3], const int64_t a[3][3],
                                      const double b[3][3],
                                      const double precision);
int64_t lagmat_check_identity_matrix_l3(const int64_t a[3][3],
                                        const int64_t b[3][3]);
int64_t lagmat_check_identity_matrix_ld3(const int64_t a[3][3],
                                         const double b[3][3],
                                         const double symprec);
int64_t lagmat_inverse_matrix_d3(double m[3][3], const double a[3][3],
                                 const double precision);
void lagmat_transpose_matrix_l3(int64_t a[3][3], const int64_t b[3][3]);
void lagmat_multiply_matrix_vector_l3(int64_t v[3], const int64_t a[3][3],
                                      const int64_t b[3]);
void lagmat_multiply_matrix_l3(int64_t m[3][3], const int64_t a[3][3],
                               const int64_t b[3][3]);
void lagmat_multiply_matrix_ld3(double m[3][3], const int64_t a[3][3],
                                const double b[3][3]);
void lagmat_multiply_matrix_d3(double m[3][3], const double a[3][3],
                               const double b[3][3]);
void lagmat_copy_matrix_l3(int64_t a[3][3], const int64_t b[3][3]);
void lagmat_copy_matrix_d3(double a[3][3], const double b[3][3]);
void lagmat_copy_vector_l3(int64_t a[3], const int64_t b[3]);
int64_t lagmat_modulo_l(const int64_t a, const int64_t b);
int64_t lagmat_Nint(const double a);
double lagmat_Dabs(const double a);

#endif
