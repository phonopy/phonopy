/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include <lapacke.h>
#include <phonoc_array.h>
#include <phonoc_utils.h>
#include <phonon4_h/real_to_reciprocal.h>

static void real_to_reciprocal_elements(lapack_complex_double *fc4_rec_elem,
					const double q[12],
					const double *fc4,
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
			 const double *fc4,
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
					const double *fc4,
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
  int fc4_elem_address;

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
	
	fc4_elem_address = (i * 81 * num_satom * num_satom * num_satom +
			    j * 81 * num_satom * num_satom +
			    k * 81 * num_satom +
			    l * 81);

	phase_factor = phonoc_complex_prod(phase_factors[0], phase_factors[1]);
	phase_factor = phonoc_complex_prod(phase_factor, phase_factors[2]);
	for (m = 0; m < 81; m++) {
	  fc4_rec_real[m] +=
	    lapack_complex_double_real(phase_factor) * fc4[fc4_elem_address + m];
	  fc4_rec_imag[m] +=
	    lapack_complex_double_imag(phase_factor) * fc4[fc4_elem_address + m];
	}
      }
    }
  }

  for (i = 0; i < 81; i++) {
    fc4_rec_elem[i] =
      lapack_make_complex_double(fc4_rec_real[i], fc4_rec_imag[i]);
  }
}
