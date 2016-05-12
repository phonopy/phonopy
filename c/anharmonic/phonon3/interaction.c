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

#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include <phonoc_array.h>
#include <phonoc_const.h>
#include <phonoc_utils.h>
#include <phonon3_h/interaction.h>
#include <phonon3_h/real_to_reciprocal.h>
#include <phonon3_h/reciprocal_to_normal.h>

static const int index_exchange[6][3] = {{0, 1, 2},
					 {2, 0, 1},
					 {1, 2, 0},
					 {2, 1, 0},
					 {0, 2, 1},
					 {1, 0, 2}};
static void get_interaction_at_triplet(Darray *fc3_normal_squared,
				       const int i,
				       const char *g_zero,
				       const Darray *frequencies,
				       const Carray *eigenvectors,
				       const Iarray *triplets,
				       const int *grid_address,
				       const int *mesh,
				       const Darray *fc3,
				       const Darray *shortest_vectors,
				       const Iarray *multiplicity,
				       const double *masses,
				       const int *p2s_map,
				       const int *s2p_map,
				       const int *band_indices,
				       const int symmetrize_fc3_q,
				       const double cutoff_frequency,
				       const int num_triplets,
				       const int openmp_at_bands);
static void real_to_normal(double *fc3_normal_squared,
			   const char *g_zero,
			   const double *freqs0,
			   const double *freqs1,
			   const double *freqs2,		      
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const lapack_complex_double *eigvecs2,
			   const Darray *fc3,
			   const double q[9], /* q0, q1, q2 */
			   const Darray *shortest_vectors,
			   const Iarray *multiplicity,
			   const double *masses,
			   const int *p2s_map,
			   const int *s2p_map,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const double cutoff_frequency,
			   const int triplet_index,
			   const int num_triplets,
			   const int openmp_at_bands);
static void real_to_normal_sym_q(double *fc3_normal_squared,
				 const char *g_zero,
				 PHPYCONST double *freqs[3],
				 PHPYCONST lapack_complex_double *eigvecs[3],
				 const Darray *fc3,
				 const double q[9], /* q0, q1, q2 */
				 const Darray *shortest_vectors,
				 const Iarray *multiplicity,
				 const double *masses,
				 const int *p2s_map,
				 const int *s2p_map,
				 const int *band_indices,
				 const int num_band0,
				 const int num_band,
				 const double cutoff_frequency,
				 const int triplet_index,
				 const int num_triplets,
				 const int openmp_at_bands);

/* fc3_normal_squared[num_triplets, num_band0, num_band, num_band] */
void get_interaction(Darray *fc3_normal_squared,
		     const char *g_zero,
		     const Darray *frequencies,
		     const Carray *eigenvectors,
		     const Iarray *triplets,
		     const int *grid_address,
		     const int *mesh,
		     const Darray *fc3,
		     const Darray *shortest_vectors,
		     const Iarray *multiplicity,
		     const double *masses,
		     const int *p2s_map,
		     const int *s2p_map,
		     const int *band_indices,
		     const int symmetrize_fc3_q,
		     const double cutoff_frequency)
{
  int i, num_band;

  num_band = frequencies->dims[1];

  if (triplets->dims[0] > num_band * num_band) {
#pragma omp parallel for
    for (i = 0; i < triplets->dims[0]; i++) {
      get_interaction_at_triplet(fc3_normal_squared,
				 i,
				 g_zero,
				 frequencies,
				 eigenvectors,
				 triplets,
				 grid_address,
				 mesh,
				 fc3,
				 shortest_vectors,
				 multiplicity,
				 masses,
				 p2s_map,
				 s2p_map,
				 band_indices,
				 symmetrize_fc3_q,
				 cutoff_frequency,
				 triplets->dims[0],
				 0);
    }
  } else {
    for (i = 0; i < triplets->dims[0]; i++) {
      get_interaction_at_triplet(fc3_normal_squared,
				 i,
				 g_zero,
				 frequencies,
				 eigenvectors,
				 triplets,
				 grid_address,
				 mesh,
				 fc3,
				 shortest_vectors,
				 multiplicity,
				 masses,
				 p2s_map,
				 s2p_map,
				 band_indices,
				 symmetrize_fc3_q,
				 cutoff_frequency,
				 triplets->dims[0],
				 1);
    }
  }
}

static void get_interaction_at_triplet(Darray *fc3_normal_squared,
				       const int i,
				       const char *g_zero,
				       const Darray *frequencies,
				       const Carray *eigenvectors,
				       const Iarray *triplets,
				       const int *grid_address,
				       const int *mesh,
				       const Darray *fc3,
				       const Darray *shortest_vectors,
				       const Iarray *multiplicity,
				       const double *masses,
				       const int *p2s_map,
				       const int *s2p_map,
				       const int *band_indices,
				       const int symmetrize_fc3_q,
				       const double cutoff_frequency,
				       const int num_triplets,
				       const int openmp_at_bands)
{
  int j, k, gp, num_band, num_band0;
  double *freqs[3];
  lapack_complex_double *eigvecs[3];
  double q[9];

  num_band = frequencies->dims[1];
  num_band0 = fc3_normal_squared->dims[1];

  for (j = 0; j < 3; j++) {
    gp = triplets->data[i * 3 + j];
    for (k = 0; k < 3; k++) {
      q[j * 3 + k] = ((double)grid_address[gp * 3 + k]) / mesh[k];
    }
    freqs[j] = frequencies->data + gp * num_band;
    eigvecs[j] = eigenvectors->data + gp * num_band * num_band;
  }

  if (symmetrize_fc3_q) {
    real_to_normal_sym_q((fc3_normal_squared->data +
			  i * num_band0 * num_band * num_band),
			 g_zero + i * num_band0 * num_band * num_band,
			 freqs,
			 eigvecs,
			 fc3,
			 q, /* q0, q1, q2 */
			 shortest_vectors,
			 multiplicity,
			 masses,
			 p2s_map,
			 s2p_map,
			 band_indices,
			 num_band0,
			 num_band,
			 cutoff_frequency,
			 i,
			 num_triplets,
			 openmp_at_bands);
  } else {
    real_to_normal((fc3_normal_squared->data +
		    i * num_band0 * num_band * num_band),
		   g_zero + i * num_band0 * num_band * num_band,
		   freqs[0],
		   freqs[1],
		   freqs[2],
		   eigvecs[0],
		   eigvecs[1],
		   eigvecs[2],
		   fc3,
		   q, /* q0, q1, q2 */
		   shortest_vectors,
		   multiplicity,
		   masses,
		   p2s_map,
		   s2p_map,
		   band_indices,
		   num_band0,
		   num_band,
		   cutoff_frequency,
		   i,
		   num_triplets,
		   openmp_at_bands);
  }
}

static void real_to_normal(double *fc3_normal_squared,
			   const char* g_zero,
			   const double *freqs0,
			   const double *freqs1,
			   const double *freqs2,		      
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const lapack_complex_double *eigvecs2,
			   const Darray *fc3,
			   const double q[9], /* q0, q1, q2 */
			   const Darray *shortest_vectors,
			   const Iarray *multiplicity,
			   const double *masses,
			   const int *p2s_map,
			   const int *s2p_map,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const double cutoff_frequency,
			   const int triplet_index,
			   const int num_triplets,
			   const int openmp_at_bands)
{
  int num_patom;
  lapack_complex_double *fc3_reciprocal;

  num_patom = num_band / 3;

  fc3_reciprocal =
    (lapack_complex_double*)malloc(sizeof(lapack_complex_double) *
				   num_patom * num_patom * num_patom * 27);

  real_to_reciprocal(fc3_reciprocal,
		     q,
		     fc3,
		     shortest_vectors,
		     multiplicity,
		     p2s_map,
		     s2p_map,
		     openmp_at_bands);

  if (openmp_at_bands) {
#ifdef MEASURE_R2N
    printf("At triplet %d/%d (# of bands=%d):\n",
	   triplet_index, num_triplets, num_band0);
#endif
    reciprocal_to_normal_squared_openmp(fc3_normal_squared,
					g_zero,
					fc3_reciprocal,
					freqs0,
					freqs1,
					freqs2,
					eigvecs0,
					eigvecs1,
					eigvecs2,
					masses,
					band_indices,
					num_band0,
					num_band,
					cutoff_frequency);
  } else {
    reciprocal_to_normal_squared(fc3_normal_squared,
				 g_zero,
				 fc3_reciprocal,
				 freqs0,
				 freqs1,
				 freqs2,
				 eigvecs0,
				 eigvecs1,
				 eigvecs2,
				 masses,
				 band_indices,
				 num_band0,
				 num_band,
				 cutoff_frequency);
  }

  free(fc3_reciprocal);
}

static void real_to_normal_sym_q(double *fc3_normal_squared,
				 const char *g_zero,
				 double *freqs[3],
				 lapack_complex_double *eigvecs[3],
				 const Darray *fc3,
				 const double q[9], /* q0, q1, q2 */
				 const Darray *shortest_vectors,
				 const Iarray *multiplicity,
				 const double *masses,
				 const int *p2s_map,
				 const int *s2p_map,
				 const int *band_indices,
				 const int num_band0,
				 const int num_band,
				 const double cutoff_frequency,
				 const int triplet_index,
				 const int num_triplets,
				 const int openmp_at_bands)
{
  int i, j, k, l;
  int band_ex[3];
  double q_ex[9];
  double *fc3_normal_squared_ex;

  fc3_normal_squared_ex =
    (double*)malloc(sizeof(double) * num_band * num_band * num_band);

  for (i = 0; i < num_band0 * num_band * num_band; i++) {
    fc3_normal_squared[i] = 0;
  }

  for (i = 0; i < 6; i++) {
    for (j = 0; j < 3; j ++) {
      for (k = 0; k < 3; k ++) {
	q_ex[j * 3 + k] = q[index_exchange[i][j] * 3 + k];
      }
    }
    real_to_normal(fc3_normal_squared_ex,
		   g_zero,
		   freqs[index_exchange[i][0]],
		   freqs[index_exchange[i][1]],
		   freqs[index_exchange[i][2]],
		   eigvecs[index_exchange[i][0]],
		   eigvecs[index_exchange[i][1]],
		   eigvecs[index_exchange[i][2]],
		   fc3,
		   q_ex, /* q0, q1, q2 */
		   shortest_vectors,
		   multiplicity,
		   masses,
		   p2s_map,
		   s2p_map,
		   band_indices,
		   num_band,
		   num_band,
		   cutoff_frequency,
		   triplet_index,
		   num_triplets,
		   openmp_at_bands);
    for (j = 0; j < num_band0; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  band_ex[0] = band_indices[j];
	  band_ex[1] = k;
	  band_ex[2] = l;
	  fc3_normal_squared[j * num_band * num_band +
			     k * num_band +
			     l] +=
	    fc3_normal_squared_ex[band_ex[index_exchange[i][0]] *
				  num_band * num_band +
				  band_ex[index_exchange[i][1]] * num_band +
				  band_ex[index_exchange[i][2]]] / 6;
	}
      }
    }
  }

  free(fc3_normal_squared_ex);

}

