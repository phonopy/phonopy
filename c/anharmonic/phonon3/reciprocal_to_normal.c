#include <lapacke.h>
#include <math.h>
#include "phonoc_array.h"
#include "phonoc_math.h"
#include "phonon3_h/reciprocal_to_normal.h"

void reciprocal_to_normal_squared
(double *fc3_normal_squared,
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
  double fff, sum_real, sum_imag;
  lapack_complex_double fc3_sum;

  num_atom = num_band / 3;

  for (i = 0; i < num_band0; i++) {
    bi = band_indices[i];
    if (freqs0[bi] > cutoff_frequency) {
      for (j = 0; j < num_band; j++) {
	if (freqs1[j] > cutoff_frequency) {
	  for (k = 0; k < num_band; k++) {
	    if (freqs2[k] > cutoff_frequency) {
	      fff = freqs0[bi] * freqs1[j] * freqs2[k];
	      fc3_sum = fc3_sum_in_reciprocal_to_normal
		(bi, j, k,
		 eigvecs0, eigvecs1, eigvecs2,
		 fc3_reciprocal,
		 masses,
		 num_atom);
	      sum_real = lapack_complex_double_real(fc3_sum);
	      sum_imag = lapack_complex_double_imag(fc3_sum);
	      fc3_normal_squared[i * num_band * num_band + j * num_band + k] =
		(sum_real * sum_real + sum_imag * sum_imag) / fff;
	    } else {
	      fc3_normal_squared[i * num_band * num_band + j * num_band + k] = 0;
	    }
	  }
	} else {
	  for (k = 0; k < num_band; k++) {
	    fc3_normal_squared[i * num_band * num_band + j * num_band + k] = 0;
	  }
	}
      }
    } else {
      for (j = 0; j < num_band * num_band; j++) {
	fc3_normal_squared[i * num_band * num_band + j] = 0;
      }
    }
  }
}

lapack_complex_double fc3_sum_in_reciprocal_to_normal
(const int bi0,
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
  double sum_real, sum_imag, mmm;
  /* double sum_real_cart, sum_imag_cart; */
  lapack_complex_double eig_prod, eig_prod1;

  sum_real = 0;
  sum_imag = 0;

  /* A slight tune-up with respect to the second one for many atomic case */
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
  	for (l = 0; l < num_atom; l++) {
  	  for (m = 0; m < num_atom; m++) {
	    eig_prod1 = phonoc_complex_prod
	      (eigvecs0[(l * 3 + i) * num_atom * 3 + bi0],
	       eigvecs1[(m * 3 + j) * num_atom * 3 + bi1]);
  	    for (n = 0; n < num_atom; n++) {
	      mmm = 1.0 / sqrt(masses[l] * masses[m] * masses[n]);
	      eig_prod = phonoc_complex_prod
		(eig_prod1, eigvecs2[(n * 3 + k) * num_atom * 3 + bi2]);
	      eig_prod = phonoc_complex_prod
		(eig_prod,
		 fc3_reciprocal[l * num_atom * num_atom * 27 +
				m * num_atom * 27 +
				n * 27 +
				i * 9 +
				j * 3 +
				k]);
  	      sum_real += lapack_complex_double_real(eig_prod) * mmm;
  	      sum_imag += lapack_complex_double_imag(eig_prod) * mmm;
  	    }
  	  }
  	}
      }
    }
  }

  /* for (i = 0; i < num_atom; i++) { */
  /*   for (j = 0; j < num_atom; j++) { */
  /*     for (k = 0; k < num_atom; k++) { */
  /* 	sum_real_cart = 0; */
  /* 	sum_imag_cart = 0; */
  /* 	mmm = sqrt(masses[i] * masses[j] * masses[k]); */
  /* 	for (l = 0; l < 3; l++) { */
  /* 	  for (m = 0; m < 3; m++) { */
  /* 	    for (n = 0; n < 3; n++) { */
  /* 	      eig_prod = */
  /* 		phonoc_complex_prod(eigvecs0[(i * 3 + l) * num_atom * 3 + bi0], */
  /*               phonoc_complex_prod(eigvecs1[(j * 3 + m) * num_atom * 3 + bi1], */
  /* 		phonoc_complex_prod(eigvecs2[(k * 3 + n) * num_atom * 3 + bi2], */
  /*               fc3_reciprocal[i * num_atom * num_atom * 27 + */
  /* 			       j * num_atom * 27 + */
  /* 			       k * 27 + */
  /* 			       l * 9 + */
  /* 			       m * 3 + */
  /* 			       n]))); */
  /* 	      sum_real_cart += lapack_complex_double_real(eig_prod); */
  /* 	      sum_imag_cart += lapack_complex_double_imag(eig_prod); */
  /* 	    } */
  /* 	  } */
  /* 	} */
  /* 	sum_real += sum_real_cart / mmm; */
  /* 	sum_imag += sum_imag_cart / mmm; */
  /*     } */
  /*   } */
  /* } */
  return lapack_make_complex_double(sum_real, sum_imag);
}
