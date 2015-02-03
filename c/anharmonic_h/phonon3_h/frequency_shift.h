#ifndef __frequency_shift3_H__
#define __frequency_shift3_H__

#include "phonoc_array.h"

#endif

void get_frequency_shift_at_bands(double *frequency_shift,
				  const Darray *fc3_normal_squared,
				  const int *band_indices,
				  const double *frequencies,
				  const int *grid_point_triplets,
				  const int *triplet_weights,
				  const double epsilon,
				  const double temperature,
				  const double unit_conversion_factor,
				  const double cutoff_frequency);
