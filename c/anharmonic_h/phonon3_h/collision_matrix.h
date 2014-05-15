#ifndef __collision_matrix_H__
#define __collision_matrix_H__

#include "phonoc_array.h"

void get_collision_matrix(double *collision_matrix,
			  const Darray *fc3_normal_squared,
			  const double *frequencies,
			  const int *triplets,
			  const Iarray *triplets_map,
			  const int *stabilized_gp_map,
			  const int *ir_grid_points,
			  const Iarray *rotated_grid_points,
			  const double *rotations_cartesian,
			  const double *g,
			  const double temperature,
			  const double unit_conversion_factor,
			  const double cutoff_frequency);

void get_collision_matrix_full(double *collision_matrix,
			       const Darray *fc3_normal_squared,
			       const double *frequencies,
			       const int *triplets,
			       const Iarray *triplets_map,
			       const int *stabilized_gp_map,
			       const double *g,
			       const double temperature,
			       const double unit_conversion_factor,
			       const double cutoff_frequency);
#endif
