#ifndef __real_to_normal_H__
#define __real_to_normal_H__

double real_to_normal(const Darray *freqs,
		      const Carray *eigvecs,
		      const Darray *fc3,
		      const double q[6],
		      const double sum_q[3],
		      const Darray *shortest_vectors,
		      const Iarray *multiplicity,
		      const double *masses,
		      const int *p2s_map,
		      const int *s2p_map);

#endif
