#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "phonoc_array.h"
#include "phonoc_math.h"
#include "phonoc_utils.h"
#include "phonon3_h/real_to_reciprocal.h"
#include "phonon4_h/real_to_reciprocal.h"

static void real_to_reciprocal_elements(lapack_complex_double *fc4_rec_elem,
					const double q[12],
					const Darray *fc4,
					const Darray *shortest_vectors,
					const Iarray *multiplicity,
					const int *p2s,
					const int *s2p,
					const int pi0,
					const int pi1,
					const int pi2,
					const int pi3);

/* fc4_reciprocal[num_patom, num_patom, num_patom, num_patom, 3, 3, 3, 3] */
void real_to_reciprocal4(lapack_complex_double *fc4_reciprocal,
			 const double q[12],
			 const Darray *fc4,
			 const Darray *shortest_vectors,
			 const Iarray *multiplicity,
			 const int *p2s_map,
			 const int *s2p_map)
{
  int i, j, k, l, num_patom;
  
  num_patom = multiplicity->dims[1];

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (k = 0; k < num_patom; k++) {
	for (l = 0; l < num_patom; l++) {
	  real_to_reciprocal_elements
	    (fc4_reciprocal +
	     i * 81 * num_patom * num_patom * num_patom +
	     j * 81 * num_patom * num_patom +
	     k * 81 * num_patom +
	     l * 81,
	     q,
	     fc4,
	     shortest_vectors,
	     multiplicity,
	     p2s_map,
	     s2p_map,
	     i, j, k, l);
	}
      }
    }
  }
}		       

static void real_to_reciprocal_elements(lapack_complex_double *fc4_rec_elem,
					const double q[12],
					const Darray *fc4,
					const Darray *shortest_vectors,
					const Iarray *multiplicity,
					const int *p2s,
					const int *s2p,
					const int pi0,
					const int pi1,
					const int pi2,
					const int pi3)
{
  int i, j, k, l, m, num_satom;
  lapack_complex_double phase_factor, phase_factors[3];
  double fc4_rec_real[81], fc4_rec_imag[81];
  double *fc4_elem;

  for (i = 0; i < 81; i++) {
    fc4_rec_real[i] = 0;
    fc4_rec_imag[i] = 0;
  }
  
  num_satom = multiplicity->dims[0];

  i = p2s[pi0];

  for (j = 0; j < num_satom; j++) {
    if (s2p[j] != p2s[pi1]) {
      continue;
    }
    phase_factors[0] =
      get_phase_factor(q, shortest_vectors, multiplicity, pi0, j, 1);

    for (k = 0; k < num_satom; k++) {
      if (s2p[k] != p2s[pi2]) {
	continue;
      }
      phase_factors[1] =
	get_phase_factor(q, shortest_vectors, multiplicity, pi0, k, 2);

      for (l = 0; l < num_satom; l++) {
	if (s2p[l] != p2s[pi3]) {
	  continue;
	}
	phase_factors[2] =
	  get_phase_factor(q, shortest_vectors, multiplicity, pi0, l, 3);
	
	fc4_elem = fc4->data + (i * 81 * num_satom * num_satom * num_satom +
				j * 81 * num_satom * num_satom +
				k * 81 * num_satom +
				l * 81);

	phase_factor = phonoc_complex_prod(phase_factors[0], phase_factors[1]);
	phase_factor = phonoc_complex_prod(phase_factor, phase_factors[2]);
	for (m = 0; m < 81; m++) {
	  fc4_rec_real[m] +=
	    lapack_complex_double_real(phase_factor) * fc4_elem[m];
	  fc4_rec_imag[m] +=
	    lapack_complex_double_imag(phase_factor) * fc4_elem[m];
	}
      }
    }
  }

  for (i = 0; i < 81; i++) {
    fc4_rec_elem[i] =
      lapack_make_complex_double(fc4_rec_real[i], fc4_rec_imag[i]);
  }
}
