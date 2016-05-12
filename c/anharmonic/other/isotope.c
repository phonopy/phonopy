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
#include <phonoc_const.h>
#include <phonoc_utils.h>

void get_isotope_scattering_strength(double *gamma,
				     const int grid_point,
				     const double *mass_variances,
				     const double *frequencies,
				     const lapack_complex_double *eigenvectors,
				     const int num_grid_points,
				     const int *band_indices,
				     const int num_band,
				     const int num_band0,
				     const double sigma,
				     const double cutoff_frequency)
{
  int i, j, k, l, m;
  double *e0_r, *e0_i, e1_r, e1_i, a, b, f, *f0, dist, sum_g, sum_g_k;

  e0_r = (double*)malloc(sizeof(double) * num_band * num_band0);
  e0_i = (double*)malloc(sizeof(double) * num_band * num_band0);
  f0 = (double*)malloc(sizeof(double) * num_band0);

  for (i = 0; i < num_band0; i++) {
    f0[i] = frequencies[grid_point * num_band + band_indices[i]];
    for (j = 0; j < num_band; j++) {
      e0_r[i * num_band + j] = lapack_complex_double_real
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
      e0_i[i * num_band + j] = lapack_complex_double_imag
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
    }
  }
  
  for (i = 0; i < num_band0; i++) {
    gamma[i] = 0;
  }

  for (i = 0; i < num_band0; i++) { /* band index0 */
    if (f0[i] < cutoff_frequency) {
      continue;
    }
    sum_g = 0;
#pragma omp parallel for private(k, l, m, f, e1_r, e1_i, a, b, dist, sum_g_k) reduction(+:sum_g)
    for (j = 0; j < num_grid_points; j++) {
      sum_g_k = 0;
      for (k = 0; k < num_band; k++) { /* band index */
	f = frequencies[j * num_band + k];
	if (f < cutoff_frequency) {
	  continue;
	}
	dist = gaussian(f - f0[i], sigma);
	for (l = 0; l < num_band / 3; l++) { /* elements */
	  a = 0;
	  b = 0;
	  for (m = 0; m < 3; m++) {
	    e1_r = lapack_complex_double_real
	      (eigenvectors[j * num_band * num_band +
			    (l * 3 + m) * num_band + k]);
	    e1_i = lapack_complex_double_imag
	      (eigenvectors[j * num_band * num_band +
			    (l * 3 + m) * num_band + k]);
	    a += (e0_r[i * num_band + l * 3 + m] * e1_r +
		  e0_i[i * num_band + l * 3 + m] * e1_i);
	    b += (e0_i[i * num_band + l * 3 + m] * e1_r -
		  e0_r[i * num_band + l * 3 + m] * e1_i);
	  }
	  sum_g_k += (a * a + b * b) * mass_variances[l] * dist;
	}
      }
      sum_g += sum_g_k;
    }
    gamma[i] = sum_g;
  }

  for (i = 0; i < num_band0; i++) {
    /* Frequency unit to ang-freq: *(2pi)**2/(2pi) */
    /* Ang-freq to freq unit (for lifetime): /2pi */
    /* gamma = 1/2t */
    gamma[i] *= M_2PI / 4 * f0[i] * f0[i] / 2;
  }
  
  free(f0);
  free(e0_r);
  free(e0_i);
}

void
get_thm_isotope_scattering_strength(double *gamma,
				    const int grid_point,
				    const int *ir_grid_points,
				    const int *weights,
				    const double *mass_variances,
				    const double *frequencies,
				    const lapack_complex_double *eigenvectors,
				    const int num_grid_points,
				    const int *band_indices,
				    const int num_band,
				    const int num_band0,
				    const double *integration_weights,
				    const double cutoff_frequency)
{
  int i, j, k, l, m, gp;
  double *e0_r, *e0_i, *f0, *gamma_ij;
  double e1_r, e1_i, a, b, f, dist, sum_g_k;

  e0_r = (double*)malloc(sizeof(double) * num_band * num_band0);
  e0_i = (double*)malloc(sizeof(double) * num_band * num_band0);
  f0 = (double*)malloc(sizeof(double) * num_band0);

  for (i = 0; i < num_band0; i++) {
    f0[i] = frequencies[grid_point * num_band + band_indices[i]];
    for (j = 0; j < num_band; j++) {
      e0_r[i * num_band + j] = lapack_complex_double_real
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
      e0_i[i * num_band + j] = lapack_complex_double_imag
	(eigenvectors[grid_point * num_band * num_band +
		      j * num_band + band_indices[i]]);
    }
  }
  
  gamma_ij = (double*)malloc(sizeof(double) * num_grid_points * num_band0);
#pragma omp parallel for
  for (i = 0; i < num_grid_points * num_band0; i++) {
    gamma_ij[i] = 0;
  }

#pragma omp parallel for private(j, k, l, m, f, gp, e1_r, e1_i, a, b, dist, sum_g_k)
  for (i = 0; i < num_grid_points; i++) {
    gp = ir_grid_points[i];
    for (j = 0; j < num_band0; j++) { /* band index0 */
      if (f0[j] < cutoff_frequency) {
	continue;
      }
      sum_g_k = 0;
      for (k = 0; k < num_band; k++) { /* band index */
	f = frequencies[gp * num_band + k];
	if (f < cutoff_frequency) {
	  continue;
	}
	dist = integration_weights[gp * num_band0 * num_band +
				   j * num_band + k];
	for (l = 0; l < num_band / 3; l++) { /* elements */
	  a = 0;
	  b = 0;
	  for (m = 0; m < 3; m++) {
	    e1_r = lapack_complex_double_real
	      (eigenvectors
	       [gp * num_band * num_band + (l * 3 + m) * num_band + k]);
	    e1_i = lapack_complex_double_imag
	      (eigenvectors
	       [gp * num_band * num_band + (l * 3 + m) * num_band + k]);
	    a += (e0_r[j * num_band + l * 3 + m] * e1_r +
		  e0_i[j * num_band + l * 3 + m] * e1_i);
	    b += (e0_i[j * num_band + l * 3 + m] * e1_r -
		  e0_r[j * num_band + l * 3 + m] * e1_i);
	  }
	  sum_g_k += (a * a + b * b) * mass_variances[l] * dist;
	}
      }
      gamma_ij[gp * num_band0 + j] = sum_g_k * weights[gp];
    }
  }

  for (i = 0; i < num_band0; i++) {
    gamma[i] = 0;
  }

  for (i = 0; i < num_grid_points; i++) {
    gp = ir_grid_points[i];
    for (j = 0; j < num_band0; j++) {
      gamma[j] += gamma_ij[gp * num_band0 + j];
    }
  }
  
  for (i = 0; i < num_band0; i++) {
    /* Frequency unit to ang-freq: *(2pi)**2/(2pi) */
    /* Ang-freq to freq unit (for lifetime): /2pi */
    /* gamma = 1/2t */
    gamma[i] *= M_2PI / 4 * f0[i] * f0[i] / 2;
  }
  
  free(gamma_ij);
  free(f0);
  free(e0_r);
  free(e0_i);
}


