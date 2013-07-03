#ifndef __reciprocal_to_normal_H__
#define __reciprocal_to_normal_H__

#include <lapacke.h>
#include "phonoc_array.h"

void reciprocal_to_normal(double *fc3_normal_squared,
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
			  const int num_band);

#endif
