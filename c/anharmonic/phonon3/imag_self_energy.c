#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/imag_self_energy.h"

static double get_imag_self_energy_at_band(double *imag_self_energy,
					   const int band_index,
					   const Darray *fc3_normal_sqared,
					   const double fpoint,
					   const double *frequencies,
					   const int *grid_point_triplets,
					   const int *triplet_weights,
					   const double sigma,
					   const double temperature,
					   const double unit_conversion_factor,
					   const double cutoff_frequency);
static double sum_imag_self_energy_at_band(const int num_band,
					   const double *fc3_normal_sqared,
					   const double fpoint,
					   const double *freqs0,
					   const double *freqs1,
					   const double sigma,
					   const double temperature,
					   const double cutoff_frequency);
static double sum_imag_self_energy_at_band_0K(const int num_band,
					      const double *fc3_normal_sqared,
					      const double fpoint,
					      const double *freqs0,
					      const double *freqs1,
					      const double sigma,
					      const double cutoff_frequency);
    
/* imag_self_energy[num_band0] */
/* fc3_normal_sqared[num_triplets, num_band0, num_band, num_band] */
void get_imag_self_energy(double *imag_self_energy,
			  const Darray *fc3_normal_sqared,
			  const double fpoint,
			  const double *frequencies,
			  const int *grid_point_triplets,
			  const int *triplet_weights,
			  const double sigma,
			  const double temperature,
			  const double unit_conversion_factor,
			  const double cutoff_frequency)
{
  int i, num_band0;
  num_band0 = fc3_normal_sqared->dims[1];
  
  for (i = 0; i < num_band0; i++) {
    imag_self_energy[i] =
      get_imag_self_energy_at_band(imag_self_energy,
				   i,
				   fc3_normal_sqared,
				   fpoint,
				   frequencies,
				   grid_point_triplets,
				   triplet_weights,
				   sigma,
				   temperature,
				   unit_conversion_factor,
				   cutoff_frequency);
  }
}

void get_imag_self_energy_at_bands(double *imag_self_energy,
				   const Darray *fc3_normal_sqared,
				   const int *band_indices,
				   const double *frequencies,
				   const int *grid_point_triplets,
				   const int *triplet_weights,
				   const double sigma,
				   const double temperature,
				   const double unit_conversion_factor,
				   const double cutoff_frequency)
{
  int i, num_band0, num_band, gp0;
  double fpoint;
  
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];
  gp0 = grid_point_triplets[0];

  /* num_band0 and num_band_indices have to be same. */
  for (i = 0; i < num_band0; i++) {
    fpoint = frequencies[gp0 * num_band + band_indices[i]];
    imag_self_energy[i] =
      get_imag_self_energy_at_band(imag_self_energy,
				   i,
				   fc3_normal_sqared,
				   fpoint,
				   frequencies,
				   grid_point_triplets,
				   triplet_weights,
				   sigma,
				   temperature,
				   unit_conversion_factor,
				   cutoff_frequency);
  }
}

static double get_imag_self_energy_at_band(double *imag_self_energy,
					   const int band_index,
					   const Darray *fc3_normal_sqared,
					   const double fpoint,
					   const double *frequencies,
					   const int *grid_point_triplets,
					   const int *triplet_weights,
					   const double sigma,
					   const double temperature,
					   const double unit_conversion_factor,
					   const double cutoff_frequency)
{
  int i, num_triplets, num_band0, num_band, gp1, gp2;
  double sum_g;

  num_triplets = fc3_normal_sqared->dims[0];
  num_band0 = fc3_normal_sqared->dims[1];
  num_band = fc3_normal_sqared->dims[2];

  sum_g = 0;
#pragma omp parallel for private(gp1, gp2) reduction(+:sum_g)
  for (i = 0; i < num_triplets; i++) {
    gp1 = grid_point_triplets[i * 3 + 1];
    gp2 = grid_point_triplets[i * 3 + 2];
    if (temperature > 0) {
      sum_g +=
	sum_imag_self_energy_at_band(num_band,
				     fc3_normal_sqared->data +
				     i * num_band0 * num_band * num_band +
				     band_index * num_band * num_band,
				     fpoint,
				     frequencies + gp1 * num_band,
				     frequencies + gp2 * num_band,
				     sigma,
				     temperature,
				     cutoff_frequency) *
	triplet_weights[i] * unit_conversion_factor;
    } else {
      sum_g +=
	sum_imag_self_energy_at_band_0K(num_band,
					fc3_normal_sqared->data +
					i * num_band0 * num_band * num_band +
					band_index * num_band * num_band,
					fpoint,
					frequencies + gp1 * num_band,
					frequencies + gp2 * num_band,
					sigma,
					cutoff_frequency) *
	triplet_weights[i] * unit_conversion_factor;
    }
  }
  return sum_g;
}

static double sum_imag_self_energy_at_band(const int num_band,
					   const double *fc3_normal_sqared,
					   const double fpoint,
					   const double *freqs0,
					   const double *freqs1,
					   const double sigma,
					   const double temperature,
					   const double cutoff_frequency)
{
  int i, j;
  double n2, n3, g1, g2, g3, sum_g;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (freqs0[i] > cutoff_frequency) {
      n2 = bose_einstein(freqs0[i], temperature);
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  n3 = bose_einstein(freqs1[j], temperature);
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

static double sum_imag_self_energy_at_band_0K(const int num_band,
					      const double *fc3_normal_sqared,
					      const double fpoint,
					      const double *freqs0,
					      const double *freqs1,
					      const double sigma,
					      const double cutoff_frequency)
{
  int i, j;
  double g1, sum_g;

  sum_g = 0;
  for (i = 0; i < num_band; i++) {
    if (freqs0[i] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  g1 = gaussian(fpoint - freqs0[i] - freqs1[j], sigma);
	  sum_g += g1 * fc3_normal_sqared[i * num_band + j];
	}
      }
    }
  }
  return sum_g;
}

