#ifndef __real_to_reciprocal_H__
#define __real_to_reciprocal_H__

#include <lapacke.h>
#include "phonoc_array.h"

void real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
			const double q[9],
			const Darray *fc3,
			const Darray *shortest_vectors,
			const Iarray *multiplicity,
			const int *p2s_map,
			const int *s2p_map);

#endif
