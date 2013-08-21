#ifndef __phonoc_math_H__
#define __phonoc_math_H__

#include <lapacke.h>

#define M_2PI 6.283185307179586

lapack_complex_double
phonoc_complex_prod(const lapack_complex_double a,
		    const lapack_complex_double b);

#endif
