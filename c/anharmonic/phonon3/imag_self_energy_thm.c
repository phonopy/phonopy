#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/imag_self_energy_thm.h"

static double
sum_thm_imag_self_energy_at_band(const int num_band,
				 const double *fc3_normal_sqared,
				 const double *n1,
				 const double *n2,
				 const double *g);
static double
sum_thm_imag_self_energy_at_band_0K(const int num_band,
				    const double *fc3_normal_sqared,
				    const double *n1,
				    const double *n2,
				    const double *g);

void get_thm_imag_self_energy_at_bands(double *imag_self_energy,
				       const Darray *fc3_normal_sqared,
				       const double *frequencies,
				       const int *triplets,
				       const int *weights,
				       const double *g,
				       const double temperature,
				       const double unit_conversion_factor,
				       const double cutoff_frequency)
{
  int i, j, num_triplets, num_band0, num_band, gp1, gp2;
  double f1, f2;
  double *n1, *n2, *ise;

  num_triplets = fc3_normal_sqared->dims[0];
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];

  ise = (double*)malloc(sizeof(double) * num_triplets * num_band0);
  
#pragma omp parallel for private(j, gp1, gp2, n1, n2, f1, f2)
  for (i = 0; i < num_triplets; i++) {
    gp1 = triplets[i * 3 + 1];
    gp2 = triplets[i * 3 + 2];
    n1 = (double*)malloc(sizeof(double) * num_band);
    n2 = (double*)malloc(sizeof(double) * num_band);
    for (j = 0; j < num_band; j++) {
      f1 = frequencies[gp1 * num_band + j];
      f2 = frequencies[gp2 * num_band + j];
      if (f1 > cutoff_frequency) {
	n1[j] = bose_einstein(f1, temperature);
      } else {
	n1[j] = -1;
      }
      if (f2 > cutoff_frequency) {
	n2[j] = bose_einstein(f2, temperature);
      } else {
	n2[j] = -1;
      }
    }
    
    for (j = 0; j < num_band0; j++) {
      if (temperature > 0) {
	ise[i * num_band0 + j] =
	  sum_thm_imag_self_energy_at_band
	  (num_band,
	   fc3_normal_sqared->data +
	   i * num_band0 * num_band * num_band + j * num_band * num_band,
	   n1,
	   n2,
	   g + (i * num_band0 * num_band * num_band +
		j * num_band * num_band) * 2);
      } else {
	ise[i * num_band0 + j] =
	  sum_thm_imag_self_energy_at_band_0K
	  (num_band,
	   fc3_normal_sqared->data +
	   i * num_band0 * num_band * num_band + j * num_band * num_band,
	   n1,
	   n2,
	   g + (i * num_band0 * num_band * num_band +
		j * num_band * num_band) * 2);
      }
    }
    free(n1);
    free(n2);
  }

  for (i = 0; i < num_band0; i++) {
    imag_self_energy[i] = 0;
    for (j = 0; j < num_triplets; j++) {
      imag_self_energy[i] += ise[j * num_band0 + i] * weights[j];
    }
    imag_self_energy[i] *= unit_conversion_factor;
  }
  free(ise);
}

static double
sum_thm_imag_self_energy_at_band(const int num_band,
				 const double *fc3_normal_sqared,
				 const double *n1,
				 const double *n2,
				 const double *g)
{
  int i, j;
  double g1, g2_3, sum_g;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (n1[i] < 0) {continue;}
    for (j = 0; j < num_band; j++) {
      if (n2[j] < 0) {continue;}
      g1 = g[i * num_band * 2 + j * 2];
      g2_3 = g[i * num_band * 2 + j * 2 + 1];
      sum_g += ((n1[i] + n2[j] + 1) * g1 + (n1[i] - n2[j]) * (g2_3)) *
	fc3_normal_sqared[i * num_band + j];
    }
  }
  return sum_g;
}

static double
sum_thm_imag_self_energy_at_band_0K(const int num_band,
				    const double *fc3_normal_sqared,
				    const double *n1,
				    const double *n2,
				    const double *g)
{
  int i, j;
  double sum_g;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (n1[i] < 0) {continue;}
    for (j = 0; j < num_band; j++) {
      if (n2[j] < 0) {continue;}
      sum_g += g[i * num_band * 2 + j * 2] * fc3_normal_sqared[i * num_band + j];
    }
  }
  return sum_g;
}
