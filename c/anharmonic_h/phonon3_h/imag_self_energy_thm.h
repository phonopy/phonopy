#ifndef __imag_self_energy_thm_H__
#define __imag_self_energy_thm_H__

#include "phonoc_array.h"

void get_thm_imag_self_energy_at_bands(double *imag_self_energy,
				       const Darray *fc3_normal_sqared,
				       const double *frequencies,
				       const int *grid_point_triplets,
				       const int *triplet_weights,
				       const double *g,
				       const double temperature,
				       const double unit_conversion_factor,
				       const double cutoff_frequency);
#endif
