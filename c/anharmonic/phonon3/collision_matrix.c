#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/collision_matrix.h"

static int *create_gp2tp_map(const int num_triplets,
			     const int *triplets);
  
void get_collision_matrix(double *collision_matrix,
			  const Darray *fc3_normal_squared,
			  const double *frequencies,
			  const int *triplets,
			  const int *triplets_map,
			  const int *stabilized_gp_map,
			  const int *ir_grid_points,
			  const Iarray *rotated_grid_points,
			  const double *rotations_cartesian,
			  const double *g,
			  const double temperature,
			  const double unit_conversion_factor,
			  const double cutoff_frequency)
{
  int i, j, k, l, m, n, ti, gp2, r_gp, num_triplets, num_band, num_ir_gp, num_rot;
  int *gp2tp_map;
  double f, collision;
  double *inv_sinh;

  num_triplets = fc3_normal_squared->dims[0];
  num_band = fc3_normal_squared->dims[2];
  num_ir_gp = rotated_grid_points->dims[0];
  num_rot = rotated_grid_points->dims[1];

  gp2tp_map = create_gp2tp_map(num_triplets, triplets);

#pragma omp parallel for private(j, k, l, m, n, ti, gp2, r_gp, f, collision, inv_sinh)
  for (i = 0; i < num_ir_gp; i++) {
    inv_sinh = (double*)malloc(sizeof(double) * num_band);
    for (j = 0; j < num_rot; j++) {
      r_gp = rotated_grid_points->data[i * num_rot + j];
      ti = gp2tp_map[triplets_map[r_gp]];
      if (triplets_map[r_gp] == stabilized_gp_map[r_gp]) {
	gp2 = triplets[ti * 3 + 2];
      } else {
	gp2 = triplets[ti * 3 + 1];
      }
      for (k = 0; k < num_band; k++) {
	f = frequencies[gp2 * num_band + k];
	if (f > cutoff_frequency) {
	  inv_sinh[k] = inv_sinh_occupation(f, temperature);
	} else {
	  inv_sinh[k] = 0;
	}
      }

      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  collision = 0;
	  for (m = 0; m < num_band; m++) {
	    collision +=
	      fc3_normal_squared->data[ti * num_band * num_band * num_band +
				       k * num_band * num_band +
				       l * num_band + m] *
	      g[2 * num_triplets * num_band * num_band * num_band +
		ti * num_band * num_band * num_band +
		k * num_band * num_band +
		l * num_band + m] *
	      inv_sinh[m] * unit_conversion_factor;
	  }
	  for (m = 0; m < 3; m++) {
	    for (n = 0; n < 3; n++) {
	      collision_matrix[k * 3 * num_ir_gp * num_band * 3 +
			       m * num_ir_gp * num_band * 3 +
			       i * num_band * 3 + l * 3 + n] +=
		collision * rotations_cartesian[j * 9 + m * 3 + n];
	    }
	  }
	}
      }
    }
    free(inv_sinh);
    inv_sinh = NULL;
  }

  free(gp2tp_map);
  gp2tp_map = NULL;
}

static int *create_gp2tp_map(const int num_triplets,
			     const int *triplets)
{
  int i, max_i;
  int *gp2tp_map;
  
  max_i = 0;
  for (i = 0; i < num_triplets; i++) {
    if (max_i < triplets[3 * i + 1]) {
      max_i = triplets[3 * i + 1];
    }
  }

  gp2tp_map = (int*)malloc(sizeof(int) * (max_i + 1));

  for (i = 0; i < num_triplets; i++) {
    gp2tp_map[triplets[3 * i + 1]] = i;
  }
  
  return gp2tp_map;
}

