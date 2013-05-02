#include "lapack_wrapper.h"

#include <lapacke.h>
int phonopy_zheev(double *w,
		  lapack_complex_double *a,
		  const int n,
		  const char uplo)
{
  lapack_int info;
  info = LAPACKE_zheev(LAPACK_ROW_MAJOR,'V', uplo,
		       (lapack_int)n, a, (lapack_int)n, w);
  return (int)info;
}
