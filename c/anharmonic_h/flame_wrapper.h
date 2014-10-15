#ifndef __flame_wrapper_H__
#define __flame_wrapper_H__

int phonopy_pinv_libflame(double *matrix,
			  double *eigvals,
			  const int size,
			  const double cutoff);

#endif
