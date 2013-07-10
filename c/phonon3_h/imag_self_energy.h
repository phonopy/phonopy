#ifndef __imag_self_energy_H__
#define __imag_self_energy_H__

#include "phonoc_array.h"

void get_imag_self_energy(double *gamma,
			  const Darray *fc3_normal_sqared,
			  const double fpoint,
			  const double *frequencies,
			  const int *grid_point_triplets,
			  const int *triplet_weights,
			  const double sigma,
			  const double temperature,
			  const double unit_conversion_factor);
void get_imag_self_energy_at_bands(double *imag_self_energy,
				   const Darray *fc3_normal_sqared,
				   const int *band_indices,
				   const double *frequencies,
				   const int *grid_point_triplets,
				   const int *triplet_weights,
				   const double sigma,
				   const double temperature,
				   const double unit_conversion_factor);
int get_jointDOS(double *jdos,
		 const int num_omega,
		 const int num_triplet,
		 const int num_band,
		 const double *o,
		 const double *f,
		 const int *w,
		 const double sigma);
#endif
