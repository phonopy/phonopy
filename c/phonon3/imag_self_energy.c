#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"
#include "imag_self_energy.h"

#define THZTOEVPARKB 47.992398658977166
#define INVSQRT2PI 0.3989422804014327

static double get_gamma_at_band(const int num_band,
				const double *fc3_normal_sqared,
				const double fpoint,
				const double *freqs0,
				const double *freqs1,
				const double sigma,
				const double temperature);
static double gaussian(const double x, const double sigma);
static double occupation(const double x, const double t);
    
/* gamma[num_fpoints, num_band0] */
/* fc3_normal_sqared[num_triplets, num_band0, num_band, num_band] */
void get_imag_self_energy(double *gamma,
			  const Darray *fc3_normal_sqared,
			  const Darray *freq_points,
			  const double *frequencies,
			  const int *grid_point_triplets,
			  const int *triplet_weights,
			  const double sigma,
			  const double temperature,
			  const double unit_conversion_factor)
{
  int i, j, k, num_triplets, num_band0, num_band, num_fpoints, gp1, gp2;

  num_triplets = fc3_normal_sqared->dims[0];
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];
  num_fpoints = freq_points->dims[0];

#pragma omp parallel for private(j, k, gp1, gp2)
  for (i = 0; i < num_triplets; i++) {
    gp1 = grid_point_triplets[i * 3 + 1];
    gp2 = grid_point_triplets[i * 3 + 2];
    printf("%d / %d\n", i + 1, num_triplets);
    for (j = 0; j < num_fpoints; j++) {
      for (k = 0; k < num_band0; k++) {
	gamma[j * num_band0 + k] +=
	  get_gamma_at_band(num_band,
			    fc3_normal_sqared->data +
			    i * num_band0 * num_band * num_band +
			    k * num_band * num_band,
			    freq_points->data[j],
			    frequencies + gp1 * num_band,
			    frequencies + gp2 * num_band,
			    sigma,
			    temperature) *
	  triplet_weights[i] * unit_conversion_factor;
      }
    }
  }
}

static double get_gamma_at_band(const int num_band,
				const double *fc3_normal_sqared,
				const double fpoint,
				const double *freqs0,
				const double *freqs1,
				const double sigma,
				const double temperature)
{
  int i, j;
  double n2, n3, g1, g2, g3, sum_g;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (freqs0[i] > 0) {
      n2 = occupation(freqs0[i], temperature);
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > 0) {
	  n3 = occupation(freqs1[j], temperature);
	  g1 = gaussian(fpoint - freqs0[i] - freqs1[j], sigma);
	  g2 = gaussian(fpoint + freqs0[i] - freqs1[j], sigma);
	  g3 = gaussian(fpoint - freqs0[i] + freqs1[j], sigma);
	  sum_g += ((n2 + n3 + 1) * g1 + (n2 - n3) * (g2 - g3)) *
	    fc3_normal_sqared[i * num_band + j];
	}
      }
    }
  }
  return sum_g;
}

static double gaussian(const double x, const double sigma)
{
  return INVSQRT2PI / sigma * exp(-x * x / 2 / sigma / sigma);
}  

static double occupation(const double x, const double t)
{
  return 1.0 / (exp(THZTOEVPARKB * x / t) - 1);
}
