#ifndef __frequency_shift_H__
#define __frequency_shift_H__

#include <lapacke.h>
#include "phonoc_array.h"

void reciprocal_to_normal4(lapack_complex_double *fc4_normal,
			   const lapack_complex_double *fc4_reciprocal,
			   const double *freqs0,
			   const double *freqs1,
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const double *masses,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const double cutoff_frequency);

#endif
