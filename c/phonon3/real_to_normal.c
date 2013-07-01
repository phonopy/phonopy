#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "array.h"

static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b);

/* fc3_reciprocal[num_patom, num_patom, num_patom, 3, 3, 3] */
int real_to_reciprocal(lapack_complex_double *fc3_reciprocal,
		       const Iarray *triplets,
		       const Iarray *grid_address,
		       const int *mesh,
		       const Darray *fc3,
		       const Darray *shortest_vectors,
		       const Iarray *multiplicity,
		       const int *p2s_map,
		       const int *s2p_map)
{
  int i, j, k, num_patom;
  num_patom = multi->dims[1];

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (k = 0; k < num_patom; k++) {
	real_to_reciprocal_elements(fc3_reciprocal +
				    i * 27 * num_patom * num_patom +
				    j * 27 * num_patom +
				    k * 27,
				    triplets,
				    grid_address,
				    mesh,
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

static int real_to_reciprocal_elements(lapack_complex_double *fc3_rec_elem,
				       const Iarray *triplets,
				       const Iarray *grid_address,
				       const int *mesh,
				       const Darray *fc3,
				       const Darray *svecs,
				       const Iarray *multi,
				       const int *p2s,
				       const int *s2p,
				       const int pi0,
				       const int pi1,
				       const int pi2)
{
  int i, j, k, num_satom;
  lapack_complex_double phase;
  num_satom = multi->dims[0];
  
  i = p2s[pi0];
  for (j = 0; j < num_satom; j++) {
    if (s2p[j] != p2s[pi1]) {
      continue;
    }
    for (k = 0; k < num_satom; k++) {
      if (s2p[k] != p2s[pi2]) {
	continue;
      }
      phase = get_phase(triplets,
			grid_address,
			mesh,
			svecs,
			multi,
			pi0,
			i, j, k);
    }
  }

}


static lapack_complex_double get_phase(const Iarray *triplets,
				       const Iarray *grid_address,
				       const int *mesh,
				       const Darray *svecs,
				       const Iarray *multi,
				       const int pi0,
				       const int si0,
				       const int si1,
				       const int si2)
{
}
				       

static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b)
{
  lapack_complex_double c;
  c = lapack_make_complex_double
    (lapack_complex_double_real(a) * lapack_complex_double_real(b) -
     lapack_complex_double_imag(a) * lapack_complex_double_imag(b),
     lapack_complex_double_imag(a) * lapack_complex_double_real(b) +
     lapack_complex_double_real(a) * lapack_complex_double_imag(b));
  return c;
}
