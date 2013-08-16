#ifndef __lapack_wrapper_H__
#define __lapack_wrapper_H__

#include <lapacke.h>
int phonopy_zheev(double *w,
		  lapack_complex_double *a,
		  const int n,
		  const char uplo);
int phonopy_pinv(double *data_out,
		 const double *data_in,
		 const int m,
		 const int n,
		 const double cutoff);

#endif
