#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "phonoc_array.h"
#include "phonoc_math.h"

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
static lapack_complex_double get_phase_factor(const double q[9],
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
  double pre_phase;
  lapack_complex_double pre_phase_factor;
  
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

    pre_phase = 0;
    for (j = 0; j < 3; j++) {
      pre_phase += shortest_vectors->data
    	[p2s_map[i] * shortest_vectors->dims[1] *
    	 shortest_vectors->dims[2] * 3 + j] * (q[j] + q[3 + j] + q[6 + j]);
    }
    pre_phase_factor = lapack_make_complex_double(cos(M_2PI * pre_phase),
    						  sin(M_2PI * pre_phase));
    for (j = 0; j < num_patom * num_patom * 27; j++) {
      fc3_reciprocal[i * num_patom * num_patom * 27 + j] =
    	phonoc_complex_prod(fc3_reciprocal[i * num_patom * num_patom * 27 + j],
    			    pre_phase_factor);
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
  lapack_complex_double phase_factor;
  double fc3_rec_real[27], fc3_rec_imag[27];
  double *fc3_elem;

  for (i = 0; i < 27; i++) {
    fc3_rec_real[i] = 0;
    fc3_rec_imag[i] = 0;
  }
  
  num_satom = multiplicity->dims[0];

  i = p2s[pi0];

  for (j = 0; j < num_satom; j++) {
    if (s2p[j] != p2s[pi1]) {
      continue;
    }
    for (k = 0; k < num_satom; k++) {
      if (s2p[k] != p2s[pi2]) {
	continue;
      }
      phase_factor =
	get_phase_factor(q, shortest_vectors, multiplicity, pi0, j, k);
      fc3_elem = fc3->data + (i * 27 * num_satom * num_satom +
			      j * 27 * num_satom +
			      k * 27);
      for (l = 0; l < 27; l++) {
	fc3_rec_real[l] +=
	  lapack_complex_double_real(phase_factor) * fc3_elem[l];
	fc3_rec_imag[l] +=
	  lapack_complex_double_imag(phase_factor) * fc3_elem[l];
      }
    }
  }

  for (i = 0; i < 27; i++) {
    fc3_rec_elem[i] =
      lapack_make_complex_double(fc3_rec_real[i], fc3_rec_imag[i]);
  }
}

static lapack_complex_double get_phase_factor(const double q[9],
					      const Darray *shortest_vectors,
					      const Iarray *multiplicity,
					      const int pi0,
					      const int si1,
					      const int si2)
{
  int i, j, k;
  int multi[2];
  double *svecs[2];
  double phase_real[2], phase_imag[2];
  double sum_real, sum_imag, phase;
  lapack_complex_double phase1, phase2;

  svecs[0] = shortest_vectors->data + (si1 * shortest_vectors->dims[1] *
				       shortest_vectors->dims[2] * 3 +
				       pi0 * shortest_vectors->dims[2] * 3);
  svecs[1] = shortest_vectors->data + (si2 * shortest_vectors->dims[1] *
				       shortest_vectors->dims[2] * 3 +
				       pi0 * shortest_vectors->dims[2] * 3);
  multi[0] = multiplicity->data[si1 * multiplicity->dims[1] + pi0];
  multi[1] = multiplicity->data[si2 * multiplicity->dims[1] + pi0];

  for (i = 0; i < 2; i++) {
    sum_real = 0;
    sum_imag = 0;
    for (j = 0; j < multi[i]; j++) {
      phase = 0;
      for (k = 0; k < 3; k++) {
	/* q0 is not used. */
	phase += q[(i + 1) * 3 + k] * svecs[i][j * 3 + k];
      }
      sum_real += cos(M_2PI * phase);
      sum_imag += sin(M_2PI * phase);
    }
    phase_real[i] = sum_real / multi[i];
    phase_imag[i] = sum_imag / multi[i];
  }

  phase1 = lapack_make_complex_double(phase_real[0], phase_imag[0]);
  phase2 = lapack_make_complex_double(phase_real[1], phase_imag[1]);
  
  return phonoc_complex_prod(phase1, phase2);
}
