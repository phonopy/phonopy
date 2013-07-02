/*---------------------------------------------------------*/
/* Transform fc3 in reciprocal space to normal coordinates */
/*---------------------------------------------------------*/

#include <lapacke.h>
#include "phonoc_array.h"

static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b);

void reciprocal_to_normal(Darray *fc3_normal_squared,
			  const lapack_complex_double *fc3_reciprocal,
			  const double *freqs0,
			  const double *freqs1,
			  const double *freqs2,
			  const lapack_complex_double *eigvecs0,
			  const lapack_complex_double *eigvecs1,
			  const lapack_complex_double *eigvecs2,
			  const double *masses,
			  const int *band_indices)
{
  
}

static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b)
{
  lapack_complex_double c;
  c = lapack_make_complex_double
    (lapack_complex_double_real(a) * lapack_complex_double_real(b) -
     lapack_complex_double_imag(a) * lapack_complex_double_imag(b),
     lapack_complex_double_imag(a) * lapack_complex_double_real(b) +
     lapack_complex_double_real(a) * lapack_complex_double_imag(b));
  return c;
}
