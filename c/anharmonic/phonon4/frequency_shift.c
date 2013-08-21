#include <lapacke.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_math.h"
#include "phonon4_h/frequency_shift.h"

static lapack_complex_double fc4_sum(const int bi0,
				     const int bi1,
				     const lapack_complex_double *eigvecs0,
				     const lapack_complex_double *eigvecs1,
				     const lapack_complex_double *fc4_reciprocal,
				     const double *masses,
				     const int num_atom);

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
			   const double cutoff_frequency)
{
  int i, j, bi, num_atom;
  lapack_complex_double fc4_sum_elem;

  num_atom = num_band / 3;

  for (i = 0; i < num_band0; i++) {
    bi = band_indices[i];
    if (freqs0[bi] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  fc4_sum_elem = fc4_sum(bi,
				 j,
				 eigvecs0,
				 eigvecs1,
				 fc4_reciprocal,
				 masses,
				 num_atom);
	  fc4_normal[i * num_band + j] = lapack_make_complex_double
	    (lapack_complex_double_real(fc4_sum_elem) / freqs0[bi] / freqs1[j],
	     lapack_complex_double_imag(fc4_sum_elem) / freqs0[bi] / freqs1[j]);
	}
      }
    }
  }    
}

static lapack_complex_double fc4_sum(const int bi0,
				     const int bi1,
				     const lapack_complex_double *eigvecs0,
				     const lapack_complex_double *eigvecs1,
				     const lapack_complex_double *fc4_reciprocal,
				     const double *masses,
				     const int num_atom)
{
  int i, j, k, l, m, n, p, q;
  double sum_real, sum_imag, sum_real_cart, sum_imag_cart, mmm;
  lapack_complex_double eig_prod, eigvec0conj, eigvec1conj;

  sum_real = 0;
  sum_imag = 0;
  for (i = 0; i < num_atom; i++) {
    for (j = 0; j < num_atom; j++) {
      for (k = 0; k < num_atom; k++) {
	for (l = 0; l < num_atom; l++) {
	  sum_real_cart = 0;
	  sum_imag_cart = 0;
	  mmm = sqrt(masses[i] * masses[j] * masses[k] * masses[l]);
	  for (m = 0; m < 3; m++) {
	    eigvec0conj = lapack_make_complex_double
	      (lapack_complex_double_real
	       (eigvecs0[(i * 3 + m) * num_atom * 3 + bi0]),
	       -lapack_complex_double_imag
	       (eigvecs0[(i * 3 + m) * num_atom * 3 + bi0]));
	    for (n = 0; n < 3; n++) {
	    for (p = 0; p < 3; p++) {
	    for (q = 0; q < 3; q++) {
	      eigvec1conj = lapack_make_complex_double
		(lapack_complex_double_real
		 (eigvecs1[(l * 3 + q) * num_atom * 3 + bi1]),
		 -lapack_complex_double_imag
		 (eigvecs1[(l * 3 + q) * num_atom * 3 + bi1]));
	      eig_prod =
		phonoc_complex_prod(eigvec0conj,
		phonoc_complex_prod(eigvecs0[(j * 3 + n) * num_atom * 3 + bi0],
                phonoc_complex_prod(eigvecs1[(k * 3 + p) * num_atom * 3 + bi1],
                phonoc_complex_prod(eigvec1conj,
                fc4_reciprocal[i * num_atom * num_atom * num_atom * 81 +
			       j * num_atom * num_atom * 81 +
			       k * num_atom * 81 +
			       l * 81 +
			       m * 27 +
			       n * 9 +
			       p * 3 +
			       q]))));
	      sum_real_cart += lapack_complex_double_real(eig_prod);
	      sum_imag_cart += lapack_complex_double_imag(eig_prod);
	    }
	    }
	    }
	  }
	  sum_real += sum_real_cart / mmm;
	  sum_imag += sum_imag_cart / mmm;
	}
      }
    }
  }
  return lapack_make_complex_double(sum_real, sum_imag);
}
