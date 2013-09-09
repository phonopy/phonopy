#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/interaction.h"
#include "phonon3_h/real_to_reciprocal.h"
#include "phonon3_h/reciprocal_to_normal.h"

static const int index_exchange[6][3] = {{0, 1, 2},
					 {2, 0, 1},
					 {1, 2, 0},
					 {2, 1, 0},
					 {0, 2, 1},
					 {1, 0, 2}};
static void real_to_normal(double *fc3_normal_squared,
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
			   const double cutoff_frequency);
static void real_to_normal_sym_q(double *fc3_normal_squared,
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
				 const double cutoff_frequency);
static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const Iarray *triplets);

void set_phonon_triplets(Darray *frequencies,
			 Carray *eigenvectors,
			 char *phonon_done,
			 const Iarray *triplets,
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
  num_undone = collect_undone_grid_points(undone, phonon_done, triplets);

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

/* fc3_normal_squared[num_triplets, num_band0, num_band, num_band] */
void get_interaction(Darray *fc3_normal_squared,
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
  int i, j, k, gp, num_band, num_band0;
  double *freqs[3];
  lapack_complex_double *eigvecs[3];
  double q[9];

  num_band = frequencies->dims[1];
  num_band0 = fc3_normal_squared->dims[1];

#pragma omp parallel for private(j, q, gp, freqs, eigvecs)
  for (i = 0; i < triplets->dims[0]; i++) {

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
			   cutoff_frequency);
    } else {
      real_to_normal((fc3_normal_squared->data +
		      i * num_band0 * num_band * num_band),
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
		     cutoff_frequency);
    }
  }
}

static void real_to_normal(double *fc3_normal_squared,
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
			   const double cutoff_frequency)
			   
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
		     s2p_map);

  reciprocal_to_normal(fc3_normal_squared,
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

  free(fc3_reciprocal);
}

static void real_to_normal_sym_q(double *fc3_normal_squared,
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
				 const double cutoff_frequency)
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
		   cutoff_frequency);
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

static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const Iarray *triplets)
{
  int i, j, gp, num_undone;

  num_undone = 0;
  for (i = 0; i < triplets->dims[0]; i++) {
    for (j = 0; j < 3; j++) {
      gp = triplets->data[i * 3 + j];
      if (phonon_done[gp] == 0) {
	undone[num_undone] = gp;
	num_undone++;
	phonon_done[gp] = 1;
      }
    }
  }

  return num_undone;
}
