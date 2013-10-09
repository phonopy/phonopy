#ifndef __interaction_H__
#define __interaction_H__

#include "phonoc_array.h"

void get_interaction(Darray *fc3_normal_squared,
		     const Darray *frequencies,
		     const Carray *eigenvectors,
		     const Iarray *triplets,
		     const int *grid_address,
		     const int *mesh,
		     const Darray *fc3,
		     const Darray *shortest_vectors,
		     const Iarray *multiplicity,
		     const double *masses,
		     const int *p2s_map,
		     const int *s2p_map,
		     const int *band_indices,
		     const int is_sym_q,
		     const double cutoff_frequency);
#endif
