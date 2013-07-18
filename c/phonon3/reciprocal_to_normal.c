#include <lapacke.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_math.h"

static double fc3_sum_squared(const int bi0,
			      const int bi1,
			      const int bi2,
			      const lapack_complex_double *eigvecs0,
			      const lapack_complex_double *eigvecs1,
			      const lapack_complex_double *eigvecs2,
			      const lapack_complex_double *fc3_reciprocal,
			      const double *masses,
			      const int num_atom);
void reciprocal_to_normal(double *fc3_normal_squared,
			  const lapack_complex_double *fc3_reciprocal,
			  const double *freqs0,
			  const double *freqs1,
			  const double *freqs2,
			  const lapack_complex_double *eigvecs0,
			  const lapack_complex_double *eigvecs1,
			  const lapack_complex_double *eigvecs2,
			  const double *masses,
			  const int *band_indices,
			  const int num_band0,
			  const int num_band,
			  const double cutoff_frequency)
{
  int i, j, k, bi, num_atom;
  double fff;

  num_atom = num_band / 3;

  for (i = 0; i < num_band0; i++) {
    bi = band_indices[i];
    if (freqs0[bi] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  for (k = 0; k < num_band; k++) {
	    if (freqs2[k] > cutoff_frequency) {
	      fff = freqs0[bi] * freqs1[j] * freqs2[k];
	      fc3_normal_squared[i * num_band * num_band +
				 j * num_band +
				 k] =
		fc3_sum_squared(bi, j, k,
				eigvecs0, eigvecs1, eigvecs2,
				fc3_reciprocal,
				masses,
				num_atom) / fff;
	    }
	  }
	}
      }
    }
  }    
}

static double fc3_sum_squared(const int bi0,
			      const int bi1,
			      const int bi2,
			      const lapack_complex_double *eigvecs0,
			      const lapack_complex_double *eigvecs1,
			      const lapack_complex_double *eigvecs2,
			      const lapack_complex_double *fc3_reciprocal,
			      const double *masses,
			      const int num_atom)
{
  int i, j, k, l, m, n;
  double sum_real, sum_imag, sum_real_cart, sum_imag_cart, mmm;
  lapack_complex_double eig_prod;

  sum_real = 0;
  sum_imag = 0;
  for (i = 0; i < num_atom; i++) {
    for (j = 0; j < num_atom; j++) {
      for (k = 0; k < num_atom; k++) {
	sum_real_cart = 0;
	sum_imag_cart = 0;
	mmm = sqrt(masses[i] * masses[j] * masses[k]);
	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    for (n = 0; n < 3; n++) {
	      eig_prod =
		phonoc_complex_prod(eigvecs0[(i * 3 + l) * num_atom * 3 + bi0],
                phonoc_complex_prod(eigvecs1[(j * 3 + m) * num_atom * 3 + bi1],
		phonoc_complex_prod(eigvecs2[(k * 3 + n) * num_atom * 3 + bi2],
                fc3_reciprocal[i * num_atom * num_atom * 27 +
			       j * num_atom * 27 +
			       k * 27 +
			       l * 9 +
			       m * 3 +
			       n])));
	      sum_real_cart += lapack_complex_double_real(eig_prod);
	      sum_imag_cart += lapack_complex_double_imag(eig_prod);
	    }
	  }
	}
	sum_real += sum_real_cart / mmm;
	sum_imag += sum_imag_cart / mmm;
      }
    }
  }
  return sum_real * sum_real + sum_imag * sum_imag;
}
