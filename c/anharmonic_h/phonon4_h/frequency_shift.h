#ifndef __frequency_shift_H__
#define __frequency_shift_H__

#include <lapacke.h>
#include "phonoc_array.h"

void
get_fc4_normal_for_frequency_shift(double *fc4_normal_real,
				   const double *frequencies,
				   const lapack_complex_double *eigenvectors,
				   const int grid_point0,
				   const Iarray *grid_points1,
				   const Iarray *grid_address,
				   const int *mesh,
				   const double *fc4,
				   const Darray *shortest_vectors,
				   const Iarray *multiplicity,
				   const double *masses,
				   const int *p2s_map,
				   const int *s2p_map,
				   const Iarray *band_indices,
				   const double cutoff_frequency);
void set_phonons_for_frequency_shift(Darray *frequencies,
				     Carray *eigenvectors,
				     char *phonon_done,
				     const Iarray *grid_points,
				     const Iarray *grid_address,
				     const int *mesh,
				     const Darray *fc2,
				     const Darray *svecs_fc2,
				     const Iarray *multi_fc2,
				     const double *masses_fc2,
				     const int *p2s_fc2,
				     const int *s2p_fc2,
				     const double unit_conversion_factor,
				     const double *born,
				     const double *dielectric,
				     const double *reciprocal_lattice,
				     const double *q_direction,
				     const double nac_factor,
				     const char uplo);
void reciprocal_to_normal4(lapack_complex_double *fc4_normal,
			   const lapack_complex_double *fc4_reciprocal,
			   const double *freqs0,
			   const double *freqs1,
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const double *masses,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const double cutoff_frequency);

#endif
