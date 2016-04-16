/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
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

#include <lapacke.h>
#include <math.h>
#include <dynmat.h>
#include <phonoc_array.h>
#include <phonoc_const.h>
#include <phonoc_utils.h>
#include <lapack_wrapper.h>

#define THZTOEVPARKB 47.992398658977166
#define INVSQRT2PI 0.3989422804014327

lapack_complex_double get_phase_factor(const double q[],
				       const Darray *shortest_vectors,
				       const Iarray *multiplicity,
				       const int pi0,
				       const int si,
				       const int qi)
{
  int i, j, multi;
  double *svecs;
  double sum_real, sum_imag, phase;

  svecs = shortest_vectors->data + (si * shortest_vectors->dims[1] *
				    shortest_vectors->dims[2] * 3 +
				    pi0 * shortest_vectors->dims[2] * 3);
  multi = multiplicity->data[si * multiplicity->dims[1] + pi0];

  sum_real = 0;
  sum_imag = 0;
  for (i = 0; i < multi; i++) {
    phase = 0;
    for (j = 0; j < 3; j++) {
      phase += q[qi * 3 + j] * svecs[i * 3 + j];
    }
    sum_real += cos(M_2PI * phase);
    sum_imag += sin(M_2PI * phase);
  }
  sum_real /= multi;
  sum_imag /= multi;

  return lapack_make_complex_double(sum_real, sum_imag);
}

double bose_einstein(const double x, const double t)
{
  return 1.0 / (exp(THZTOEVPARKB * x / t) - 1);
}

double gaussian(const double x, const double sigma)
{
  return INVSQRT2PI / sigma * exp(-x * x / 2 / sigma / sigma);
}

double inv_sinh_occupation(const double x, const double t)
{
  return 1.0 / sinh(x * THZTOEVPARKB / 2 / t);
}

lapack_complex_double
phonoc_complex_prod(const lapack_complex_double a,
		    const lapack_complex_double b)
{
  lapack_complex_double c;
  c = lapack_make_complex_double
    (lapack_complex_double_real(a) * lapack_complex_double_real(b) -
     lapack_complex_double_imag(a) * lapack_complex_double_imag(b),
     lapack_complex_double_imag(a) * lapack_complex_double_real(b) +
     lapack_complex_double_real(a) * lapack_complex_double_imag(b));
  return c;
}
