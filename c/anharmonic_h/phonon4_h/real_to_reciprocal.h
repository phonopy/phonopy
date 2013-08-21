#ifndef __real_to_reciprocal4_H__
#define __real_to_reciprocal4_H__

#include <lapacke.h>
#include "phonoc_array.h"

void real_to_reciprocal4(lapack_complex_double *fc4_reciprocal,
			 const double q[12],
			 const Darray *fc4,
			 const Darray *shortest_vectors,
			 const Iarray *multiplicity,
			 const int *p2s_map,
			 const int *s2p_map);

#endif
