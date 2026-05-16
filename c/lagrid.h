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
