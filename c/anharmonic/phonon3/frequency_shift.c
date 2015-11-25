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
#include <stdlib.h>
#include <math.h>
#include <phonoc_array.h>
#include <phonoc_utils.h>
#include <phonon3_h/frequency_shift.h>
#include <phonon3_h/real_to_reciprocal.h>

static double get_frequency_shift_at_band(const int band_index,
					  const Darray *fc3_normal_squared,
					  const double fpoint,
					  const double *frequencies,
					  const int *grid_point_triplets,
					  const int *triplet_weights,
					  const double epsilon,
					  const double temperature,
					  const double unit_conversion_factor,
					  const double cutoff_frequency);
static double sum_frequency_shift_at_band(const int num_band,
					  const double *fc3_normal_squared,
					  const double fpoint,
					  const double *freqs0,
					  const double *freqs1,
					  const double epsilon,
					  const double temperature,
					  const double cutoff_frequency);
static double sum_frequency_shift_at_band_0K(const int num_band,
					     const double *fc3_normal_squared,
					     const double fpoint,
					     const double *freqs0,
					     const double *freqs1,
					     const double epsilon,
					     const double cutoff_frequency);

void get_frequency_shift_at_bands(double *frequency_shift,
				  const Darray *fc3_normal_squared,
				  const int *band_indices,
				  const double *frequencies,
				  const int *grid_point_triplets,
				  const int *triplet_weights,
				  const double epsilon,
				  const double temperature,
				  const double unit_conversion_factor,
				  const double cutoff_frequency)
{
  int i, num_band0, num_band, gp0;
  double fpoint;
  
  num_band0 = fc3_normal_squared->dims[1];
  num_band = fc3_normal_squared->dims[2];
  gp0 = grid_point_triplets[0];

  /* num_band0 and num_band_indices have to be same. */
  for (i = 0; i < num_band0; i++) {
    fpoint = frequencies[gp0 * num_band + band_indices[i]];
    frequency_shift[i] =
      get_frequency_shift_at_band(i,
				  fc3_normal_squared,
				  fpoint,
				  frequencies,
				  grid_point_triplets,
				  triplet_weights,
				  epsilon,
				  temperature,
				  unit_conversion_factor,
				  cutoff_frequency);
  }
}

static double get_frequency_shift_at_band(const int band_index,
					  const Darray *fc3_normal_squared,
					  const double fpoint,
					  const double *frequencies,
					  const int *grid_point_triplets,
					  const int *triplet_weights,
					  const double epsilon,
					  const double temperature,
					  const double unit_conversion_factor,
					  const double cutoff_frequency)
{
  int i, num_triplets, num_band0, num_band, gp1, gp2;
  double shift;

  num_triplets = fc3_normal_squared->dims[0];
  num_band0 = fc3_normal_squared->dims[1];
  num_band = fc3_normal_squared->dims[2];

  shift = 0;
#pragma omp parallel for private(gp1, gp2) reduction(+:shift)
  for (i = 0; i < num_triplets; i++) {
    gp1 = grid_point_triplets[i * 3 + 1];
    gp2 = grid_point_triplets[i * 3 + 2];
    if (temperature > 0) {
      shift +=
	sum_frequency_shift_at_band(num_band,
				    fc3_normal_squared->data +
				    i * num_band0 * num_band * num_band +
				    band_index * num_band * num_band,
				    fpoint,
				    frequencies + gp1 * num_band,
				    frequencies + gp2 * num_band,
				    epsilon,
				    temperature,
				    cutoff_frequency) *
	triplet_weights[i] * unit_conversion_factor;
    } else {
      shift +=
	sum_frequency_shift_at_band_0K(num_band,
				       fc3_normal_squared->data +
				       i * num_band0 * num_band * num_band +
				       band_index * num_band * num_band,
				       fpoint,
				       frequencies + gp1 * num_band,
				       frequencies + gp2 * num_band,
				       epsilon,
				       cutoff_frequency) *
	triplet_weights[i] * unit_conversion_factor;
    }
  }
  return shift;
}

static double sum_frequency_shift_at_band(const int num_band,
					  const double *fc3_normal_squared,
					  const double fpoint,
					  const double *freqs0,
					  const double *freqs1,
					  const double epsilon,
					  const double temperature,
					  const double cutoff_frequency)
{
  int i, j;
  double n2, n3, f1, f2, f3, f4, shift;
  /* double sum; */

  shift = 0;
  for (i = 0; i < num_band; i++) {
    if (freqs0[i] > cutoff_frequency) {
      n2 = bose_einstein(freqs0[i], temperature);
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  n3 = bose_einstein(freqs1[j], temperature);
	  f1 = fpoint + freqs0[i] + freqs1[j];
	  f2 = fpoint - freqs0[i] - freqs1[j];
	  f3 = fpoint - freqs0[i] + freqs1[j];
	  f4 = fpoint + freqs0[i] - freqs1[j];

	  /* sum = 0; */
	  /* if (fabs(f1) > epsilon) { */
	  /*   sum -= (n2 + n3 + 1) / f1; */
	  /* } */
	  /* if (fabs(f2) > epsilon) { */
	  /*   sum += (n2 + n3 + 1) / f2; */
	  /* } */
	  /* if (fabs(f3) > epsilon) { */
	  /*   sum -= (n2 - n3) / f3; */
	  /* } */
	  /* if (fabs(f4) > epsilon) { */
	  /*   sum += (n2 - n3) / f4; */
	  /* } */
	  /* shift += sum * fc3_normal_squared[i * num_band + j]; */
	  shift += (- (n2 + n3 + 1) * f1 / (f1 * f1 + epsilon * epsilon)
	  	    + (n2 + n3 + 1) * f2 / (f2 * f2 + epsilon * epsilon)
	  	    - (n2 - n3) * f3 / (f3 * f3 + epsilon * epsilon)
	  	    + (n2 - n3) * f4 / (f4 * f4 + epsilon * epsilon)) *
	    fc3_normal_squared[i * num_band + j];
	}
      }
    }
  }
  return shift;
}

static double sum_frequency_shift_at_band_0K(const int num_band,
					     const double *fc3_normal_squared,
					     const double fpoint,
					     const double *freqs0,
					     const double *freqs1,
					     const double epsilon,
					     const double cutoff_frequency)
{
  int i, j;
  double f1, f2, shift;

  shift = 0;
  for (i = 0; i < num_band; i++) {
    if (freqs0[i] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  f1 = fpoint + freqs0[i] + freqs1[j];
	  f2 = fpoint - freqs0[i] - freqs1[j];
	  shift += (- 1 * f1 / (f1 * f1 + epsilon * epsilon)
		    + 1 * f2 / (f2 * f2 + epsilon * epsilon)) *
	    fc3_normal_squared[i * num_band + j];
	}
      }
    }
  }
  return shift;
}
