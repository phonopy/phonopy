#ifndef __phonoc_math_H__
#define __phonoc_math_H__

#include <lapacke.h>

lapack_complex_double
phonoc_complex_prod(const lapack_complex_double a,
		    const lapack_complex_double b);

#endif
