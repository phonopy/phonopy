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

#include <stdlib.h>
#include <lapacke.h>
#include <math.h>
#include <phonoc_utils.h>
#include <phonoc_array.h>
#include <phonon3_h/reciprocal_to_normal.h>

#ifdef MEASURE_R2N
#include <time.h>
#endif

static lapack_complex_double fc3_sum_in_reciprocal_to_normal
(const int bi0,
 const int bi1,
 const int bi2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const int num_atom);

static double get_fc3_sum
(const int i,
 const int j,
 const int k,
 const int bi,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const int num_atom,
 const int num_band,
 const double cutoff_frequency);

void reciprocal_to_normal_squared
(double *fc3_normal_squared,
 const char *g_zero,
 const lapack_complex_double *fc3_reciprocal,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const double *masses,
 const int *band_indices,
 const int num_band0,
 const int num_band,
 const double cutoff_frequency)
{
  int i, j, k, bi, num_atom;

  num_atom = num_band / 3;

  for (i = 0; i < num_band0; i++) {
    bi = band_indices[i];
    if (freqs0[bi] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	for (k = 0; k < num_band; k++) {
	  if (g_zero[i * num_band * num_band + j * num_band + k]) {
	    fc3_normal_squared[i * num_band * num_band + j * num_band + k] = 0;
	    continue;
	  }
	  fc3_normal_squared[i * num_band * num_band + j * num_band + k] =
	    get_fc3_sum(i, j, k, bi,
			freqs0, freqs1, freqs2,
			eigvecs0, eigvecs1, eigvecs2,
			fc3_reciprocal,
			masses,
			num_atom,
			num_band,
			cutoff_frequency);
	}
      }
    } else {
      for (j = 0; j < num_band * num_band; j++) {
	fc3_normal_squared[i * num_band * num_band + j] = 0;
      }
    }
  }
}

void reciprocal_to_normal_squared_openmp
(double *fc3_normal_squared,
 const char *g_zero,
 const lapack_complex_double *fc3_reciprocal,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const double *masses,
 const int *band_indices,
 const int num_band0,
 const int num_band,
 const double cutoff_frequency)
{
  int i, j, k, l, ijk, jk, bi, num_atom, count;
  int *n;

#ifdef MEASURE_R2N
  double loopTotalCPUTime, loopTotalWallTime;
  time_t loopStartWallTime;
  clock_t loopStartCPUTime;
#endif

  num_atom = num_band / 3;

  n = (int*)malloc(sizeof(int) * num_band0 * num_band * num_band);

  count = 0;
  for (ijk = 0; ijk < num_band0 * num_band * num_band; ijk++) {
    if (g_zero[ijk]) {
      fc3_normal_squared[ijk] = 0;
    } else {
      n[count] = ijk;
      count++;
    }
  }

#ifdef MEASURE_R2N
  loopStartWallTime = time(NULL);
  loopStartCPUTime = clock();
#endif

#pragma omp parallel for private(ijk, i, j, k, jk, bi)
  for (l = 0; l < count; l++) {
    ijk = n[l];
    i = ijk / (num_band * num_band);
    jk = ijk % (num_band * num_band);
    j = jk / num_band;
    k = jk % num_band;
    bi = band_indices[i];
    if (freqs0[bi] > cutoff_frequency) {
      fc3_normal_squared[n[l]] = get_fc3_sum(i, j, k, bi,
					     freqs0, freqs1, freqs2,
					     eigvecs0, eigvecs1, eigvecs2,
					     fc3_reciprocal,
					     masses,
					     num_atom,
					     num_band,
					     cutoff_frequency);
    } else {
      fc3_normal_squared[n[l]] = 0;      
    }
  }

#ifdef MEASURE_R2N
      loopTotalCPUTime = (double)(clock() - loopStartCPUTime) / CLOCKS_PER_SEC;
      loopTotalWallTime = difftime(time(NULL), loopStartWallTime);
      printf("  %1.3fs (%1.3fs CPU)\n", loopTotalWallTime, loopTotalCPUTime);
#endif

  free(n);
}

static double get_fc3_sum
(const int i,
 const int j,
 const int k,
 const int bi,
 const double *freqs0,
 const double *freqs1,
 const double *freqs2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const int num_atom,
 const int num_band,
 const double cutoff_frequency)
{
  double fff, sum_real, sum_imag;
  lapack_complex_double fc3_sum;

  if (freqs1[j] > cutoff_frequency && freqs2[k] > cutoff_frequency) {
    fff = freqs0[bi] * freqs1[j] * freqs2[k];
    fc3_sum = fc3_sum_in_reciprocal_to_normal
      (bi, j, k,
       eigvecs0, eigvecs1, eigvecs2,
       fc3_reciprocal,
       masses,
       num_atom);
    sum_real = lapack_complex_double_real(fc3_sum);
    sum_imag = lapack_complex_double_imag(fc3_sum);
    return (sum_real * sum_real + sum_imag * sum_imag) / fff;
  } else {
    return 0;
  }
}

static lapack_complex_double fc3_sum_in_reciprocal_to_normal
(const int bi0,
 const int bi1,
 const int bi2,
 const lapack_complex_double *eigvecs0,
 const lapack_complex_double *eigvecs1,
 const lapack_complex_double *eigvecs2,
 const lapack_complex_double *fc3_reciprocal,
 const double *masses,
 const int num_atom)
{
  int i, j, k, l, m, n, index_l, index_lm, baseIndex;
  double sum_real, sum_imag, mmm, mass_l, mass_lm;
  lapack_complex_double eig_prod, eig_prod1;

  sum_real = 0;
  sum_imag = 0;

  for (l = 0; l < num_atom; l++) {
    mass_l = masses[l];
    index_l = l * num_atom * num_atom * 27;

    for (m = 0; m < num_atom; m++) {
      mass_lm = mass_l * masses[m];
      index_lm = index_l + m * num_atom * 27;

      for (i = 0; i < 3; i++) {
	for (j = 0; j < 3; j++) {
	  eig_prod1 = phonoc_complex_prod
	    (eigvecs0[(l * 3 + i) * num_atom * 3 + bi0],
	     eigvecs1[(m * 3 + j) * num_atom * 3 + bi1]);

	  for (n = 0; n < num_atom; n++) {
	    mmm = 1.0 / sqrt(mass_lm * masses[n]);
	    baseIndex = index_lm + n * 27 + i * 9 + j * 3;

	    for (k = 0; k < 3; k++) {
	      eig_prod = phonoc_complex_prod
		(eig_prod1, eigvecs2[(n * 3 + k) * num_atom * 3 + bi2]);
	      eig_prod = phonoc_complex_prod
		(eig_prod, fc3_reciprocal[baseIndex + k]);
	      sum_real += lapack_complex_double_real(eig_prod) * mmm;
	      sum_imag += lapack_complex_double_imag(eig_prod) * mmm;
	    }
	  }
	}
      }
    }
  }
  return lapack_make_complex_double(sum_real, sum_imag);
}
