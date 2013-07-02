#ifndef __real_to_reciprocal_H__
#define __real_to_reciprocal_H__

void real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
			const double q[6],
			const double sum_q[3],
			const Darray *fc3,
			const Darray *shortest_vectors,
			const Iarray *multiplicity,
			const int *p2s_map,
			const int *s2p_map);

#endif
