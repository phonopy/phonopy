#ifndef __reciprocal_to_normal_H__
#define __reciprocal_to_normal_H__

void reciprocal_to_normal(double *fc3_normal_squared,
			  const lapack_complex_double *fc3_reciprocal,
			  const Darray *freqs,
			  const Carray *eigvecs,
			  const double *masses);

#endif
