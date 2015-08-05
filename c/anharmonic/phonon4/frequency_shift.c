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
#include <stdlib.h>
#include <phonoc_array.h>
#include <phonoc_utils.h>
#include <phonon4_h/frequency_shift.h>
#include <phonon4_h/real_to_reciprocal.h>

static void get_fc4_normal_for_frequency_shift_at_gp
(double *fc4_normal_real,
 const double *frequencies,
 const lapack_complex_double *eigenvectors,
 const int grid_point0,
 const int grid_point1,
 const int *grid_address,
 const int *mesh,
 const double *fc4,
 const Darray *shortest_vectors,
 const Iarray *multiplicity,
 const double *masses,
 const int *p2s_map,
 const int *s2p_map,
 const Iarray *band_indices,
 const double cutoff_frequency);
static lapack_complex_double fc4_sum(const int bi0,
				     const int bi1,
				     const lapack_complex_double *eigvecs0,
				     const lapack_complex_double *eigvecs1,
				     const lapack_complex_double *fc4_reciprocal,
				     const double *masses,
				     const int num_atom);
static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const int num_grid_points,
				      const int *grid_points);

void get_fc4_frequency_shifts(double *frequency_shifts,
			      const double *fc4_normal_real,
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


void set_phonons_for_frequency_shift(Darray *frequencies,
				     Carray *eigenvectors,
				     char *phonon_done,
				     const Iarray *grid_points,
				     const int *grid_address,
				     const int *mesh,
				     const Darray *fc2,
				     const Darray *svecs_fc2,
				     const Iarray *multi_fc2,
				     const double *masses_fc2,
				     const int *p2s_fc2,
				     const int *s2p_fc2,
				     const double unit_conversion_factor,
				     const double *born,
				     const double *dielectric,
				     const double *reciprocal_lattice,
				     const double *q_direction,
				     const double nac_factor,
				     const char uplo)
{
  int num_undone;
  int *undone;

  undone = (int*)malloc(sizeof(int) * frequencies->dims[0]);
  num_undone = collect_undone_grid_points(undone,
					  phonon_done,
					  grid_points->dims[0],
					  grid_points->data);

  get_undone_phonons(frequencies,
		     eigenvectors,
		     undone,
		     num_undone,
		     grid_address,
		     mesh,
		     fc2,
		     svecs_fc2,
		     multi_fc2,
		     masses_fc2,
		     p2s_fc2,
		     s2p_fc2,
		     unit_conversion_factor,
		     born,
		     dielectric,
		     reciprocal_lattice,
		     q_direction,
		     nac_factor,
		     uplo);
  
  free(undone);
}

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
	} else {
	  fc4_normal[i * num_band + j] = lapack_make_complex_double(0, 0);
	}
      }
    } else {
      for (j = 0; j < num_band; j++) {
	fc4_normal[i * num_band + j] = lapack_make_complex_double(0, 0);
      }
    }
  }
}

static void get_fc4_normal_for_frequency_shift_at_gp
(double *fc4_normal_real,
 const double *frequencies,
 const lapack_complex_double *eigenvectors,
 const int grid_point0,
 const int grid_point1,
 const int *grid_address,
 const int *mesh,
 const double *fc4,
 const Darray *shortest_vectors,
 const Iarray *multiplicity,
 const double *masses,
 const int *p2s_map,
 const int *s2p_map,
 const Iarray *band_indices,
 const double cutoff_frequency)
{
  int i, num_atom, num_band, num_band0;
  lapack_complex_double *fc4_reciprocal, *fc4_normal;
  double q[12];

  num_atom = multiplicity->dims[1];
  num_band = num_atom * 3;
  num_band0 = band_indices->dims[0];

  fc4_reciprocal = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) *
	   num_atom * num_atom * num_atom * num_atom * 81);
  fc4_normal = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_band0 * num_band);

  for (i = 0; i < 3; i++) {
    q[i + 3] = (double)grid_address[grid_point0 * 3 + i] / mesh[i];
    q[i] = -q[i + 3];
    q[i + 6] = (double)grid_address[grid_point1 * 3 + i] / mesh[i];
    q[i + 9] = -q[i + 6];
  }
    
  real_to_reciprocal4(fc4_reciprocal,
		      q,
		      fc4,
		      shortest_vectors,
		      multiplicity,
		      p2s_map,
		      s2p_map);
  reciprocal_to_normal4(fc4_normal,
			fc4_reciprocal,
			frequencies + grid_point0 * num_band,
			frequencies + grid_point1 * num_band,
			eigenvectors + grid_point0 * num_band * num_band,
			eigenvectors + grid_point1 * num_band * num_band,
			masses,
			band_indices->data,
			num_band0,
			num_band,
			cutoff_frequency);
  
  for (i = 0; i < num_band0 * num_band; i++) {
    fc4_normal_real[i] = lapack_complex_double_real(fc4_normal[i]);
  }

  free(fc4_reciprocal);
  free(fc4_normal);
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

static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const int num_grid_points,
				      const int *grid_points)
{
  int i, gp, num_undone;

  num_undone = 0;

  for (i = 0; i < num_grid_points; i++) {
    gp = grid_points[i];
    if (phonon_done[gp] == 0) {
      undone[num_undone] = gp;
      num_undone++;
      phonon_done[gp] = 1;
    }
  }

  return num_undone;
}
