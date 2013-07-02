#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "phonoc_array.h"

#define M_2PI 6.283185307179586

static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
					const double q[9],
					const Darray *fc3,
					const Darray *shortest_vectors,
					const Iarray *multiplicity,
					const int *p2s,
					const int *s2p,
					const int pi0,
					const int pi1,
					const int pi2);
static double get_phase(const double q[9],		       
			const Darray *shortest_vectors,
			const Iarray *multiplicity,
			const int pi0,
			const int si1,
			const int si2);

/* fc3_reciprocal[num_patom, num_patom, num_patom, 3, 3, 3] */
void real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
			const double q[9],
			const Darray *fc3,
			const Darray *shortest_vectors,
			const Iarray *multiplicity,
			const int *p2s_map,
			const int *s2p_map)
{
  int i, j, k, num_patom;
  num_patom = multiplicity->dims[1];

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (k = 0; k < num_patom; k++) {
	real_to_reciprocal_elements(fc3_reciprocal +
				    i * 27 * num_patom * num_patom +
				    j * 27 * num_patom +
				    k * 27,
				    q,
				    fc3,
				    shortest_vectors,
				    multiplicity,
				    p2s_map,
				    s2p_map,
				    i, j, k);
      }
    }
  }
}		       

static void real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
					const double q[9],
					const Darray *fc3,
					const Darray *shortest_vectors,
					const Iarray *multiplicity,
					const int *p2s,
					const int *s2p,
					const int pi0,
					const int pi1,
					const int pi2)
{
  int i, j, k, l, num_satom;
  double phase, pre_phase, cos_phase, sin_phase;
  double *fc3_elem;

  num_satom = multiplicity->dims[0];

  i = p2s[pi0];

  pre_phase = 0;
  for (j = 0; j < 3; j++) {
    pre_phase += shortest_vectors->data
      [p2s[i] * shortest_vectors->dims[1] *
       shortest_vectors->dims[2] * 3 + j] * (q[j] + q[3 + j] + q[6 + j]);
  }
  
  for (j = 0; j < num_satom; j++) {
    if (s2p[j] != p2s[pi1]) {
      continue;
    }
    for (k = 0; k < num_satom; k++) {
      if (s2p[k] != p2s[pi2]) {
	continue;
      }
      phase = get_phase(q, shortest_vectors, multiplicity, pi0, j, k);
      phase += pre_phase;
      phase *= M_2PI;
      cos_phase = cos(phase);
      sin_phase = sin(phase);
      fc3_elem = fc3->data + (i * 27 * num_satom * num_satom +
			      j * 27 * num_satom +
			      k * 27);
      for (l = 0; l < 27; i++) {
	fc3_rec_elem[l] =
	  lapack_make_complex_double(cos_phase * fc3_elem[l],
				     sin_phase * fc3_elem[l]);
      }
    }
  }
}

/* return phase factor without 2pi */
static double get_phase(const double q[9],
			const Darray *shortest_vectors,
			const Iarray *multiplicity,
			const int pi0,
			const int si1,
			const int si2)
{
  int i, j, k;
  int multi[2];
  double *svecs[2];
  double phase[2];

  svecs[0] = shortest_vectors->data + (si1 * shortest_vectors->dims[1] *
				       shortest_vectors->dims[2] * 3 +
				       pi0 * shortest_vectors->dims[2] * 3);
  svecs[1] = shortest_vectors->data + (si2 * shortest_vectors->dims[1] *
				       shortest_vectors->dims[2] * 3 +
				 pi0 * shortest_vectors->dims[2] * 3);
  multi[0] = multiplicity->data[si1 * multiplicity->dims[1]];
  multi[1] = multiplicity->data[si2 * multiplicity->dims[1]];

  for (i = 0; i < 2; i++) {
    phase[i] = 0;
    for (j = 0; j < multi[i]; j++) {
      for (k = 0; k < 3; k++) {
	/* q0 is not used. */
	phase[i] += q[(i + 1) * 3 + k] * svecs[i][j * 3 + k];
      }
    }
    phase[i] /= multi[i];
  }

  return phase[0] + phase[1];
}
				       

