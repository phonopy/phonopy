#include "lapack_wrapper.h"
#include <lapacke.h>

#define min(a,b) ((a)>(b)?(b):(a))

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
		 const int n,
		 const double cutoff)
{
  int i, j, k;
  lapack_int info;
  double *s, *a, *u, *vt, *superb;

  a = (double*)malloc(sizeof(double) * m * n);
  s = (double*)malloc(sizeof(double) * min(m,n));
  u = (double*)malloc(sizeof(double) * m * m);
  vt = (double*)malloc(sizeof(double) * n * n);
  superb = (double*)malloc(sizeof(double) * (min(m,n) - 1));

  for (i = 0; i < m * n; i++) {
    a[i] = data_in[i];
  }
  
  info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR,
			'A',
			'A',
			(lapack_int)m,
			(lapack_int)n,
			a,
			(lapack_int)n,
			s,
			u,
			(lapack_int)m,
			vt,
			(lapack_int)n,
			superb);

  for (i = 0; i < n * m; i++) {
    data_out[i] = 0;
  }

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < min(m,n); k++) {
	if (s[k] > cutoff) {
	  data_out[j * m + i] += u[i * m + k] / s[k] * vt[k * n + j];
	}
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

void phonopy_pinv_mt(double *data_out,
		     int *info_out,
		     const double *data_in,
		     const int num_thread,
		     const int *row_nums,
		     const int max_row_num,
		     const int column_num,
		     const double cutoff)
{
  int i;

#pragma omp parallel for
  for (i = 0; i < num_thread; i++) {
    info_out[i] = phonopy_pinv(data_out + i * max_row_num * column_num,
			       data_in + i * max_row_num * column_num,
			       row_nums[i],
			       column_num,
			       cutoff);
  }
}
