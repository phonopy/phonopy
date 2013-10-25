#include <lapacke.h>
#include <stdlib.h>
#include "phonoc_array.h"
#include "phonoc_math.h"
#include "phonoc_utils.h"
#include "phonon3_h/frequency_shift.h"
#include "phonon3_h/real_to_reciprocal.h"

/*
void get_fc4_frequency_shifts(double *frequency_shifts,
			      const double *fc3_normal_real,
			      const double *frequencies,
			      const Iarray *grid_points1,
			      const Darray *temperatures,
			      const int *band_indicies,
			      const int num_band0,
			      const int num_band,
			      const double unit_conversion_factor)
{
  int i, j, k, l;
  double shift, num_phonon;


  for (i = 0; i < temperatures->dims[0]; i++) {
    for (j = 0; j < num_band0; j++) {
      shift = 0;
#pragma omp parallel for reduction(+:shift) private(k, l, num_phonon)
      for (k = 0; k < grid_points1->dims[0]; k++) {
	for (l = 0; l < num_band; l++) {
	  if (temperatures->data[i] > 0) {
	    num_phonon = 2 *
	      bose_einstein(frequencies[grid_points1->data[k] * num_band + l],
			    temperatures->data[i]) + 1;
	  } else {
	    num_phonon = 1;
	  }
	  shift += unit_conversion_factor * fc4_normal_real
	    [k * num_band0 * num_band + j * num_band + l] * num_phonon;
	}
      }
      frequency_shifts[i * num_band0 + j] = shift;
    }    
  }
}

void
get_fc4_normal_for_frequency_shift(double *fc4_normal_real,
				   const double *frequencies,
				   const lapack_complex_double *eigenvectors,
				   const int grid_point0,
				   const Iarray *grid_points1,
				   const int *grid_address,
				   const int *mesh,
				   const double *fc4,
				   const Darray *shortest_vectors,
				   const Iarray *multiplicity,
				   const double *masses,
				   const int *p2s_map,
				   const int *s2p_map,
				   const Iarray *band_indicies,
				   const double cutoff_frequency)
{
  int i, num_atom, num_band, num_band0;

  num_atom = multiplicity->dims[1];
  num_band = num_atom * 3;
  num_band0 = band_indicies->dims[0];

#pragma omp parallel for private(i)
  for (i = 0; i < grid_points1->dims[0]; i++) {
    get_fc4_normal_for_frequency_shift_at_gp(fc4_normal_real +
					     i * num_band0 * num_band,
					     frequencies,
					     eigenvectors,
					     grid_point0,
					     grid_points1->data[i],
					     grid_address,
					     mesh,
					     fc4,
					     shortest_vectors,
					     multiplicity,
					     masses,
					     p2s_map,
					     s2p_map,
					     band_indicies,
					     cutoff_frequency);
  }
}
*/
