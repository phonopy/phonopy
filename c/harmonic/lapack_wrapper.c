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

int phonopy_pinv(double *data_out,
		 const double *data_in,
		 const int m,
		 const int n)
{
  int i, j, k;
  lapack_int info;
  double *s, *a, *u, *vt, *superb;

  a = (double*)malloc(sizeof(double) * m * n * 10);
  s = (double*)malloc(sizeof(double) * m);
  u = (double*)malloc(sizeof(double) * m * m);
  vt = (double*)malloc(sizeof(double) * n * n);
  superb = (double*)malloc(sizeof(double) * 10 * (m + n));

  for (i = 0; i < m * n; i++) {
    a[i] = data_in[i];
  }
  
  info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR,
			'A',
			'A',
			(lapack_int)m,
			(lapack_int)n,
			a,
			(lapack_int)m,
			s,
			u,
			(lapack_int)m,
			vt,
			(lapack_int)n,
			superb);

  for (i = 0; i < n * m; i++) {
    data_out[i] = 0;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      vt[i * n + j] /= s[i];
    }
  }
  
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
	data_out[j * m + i] += u[i * m + k] * vt[k * n + j];
      }
    }
  }

  free(a);
  free(s);
  free(u);
  free(vt);
  free(superb);

  return (int)info;
}
