#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "dynmat.h"
#include "array.h"
#include "lapack_wrapper.h"
#include "interaction.h"

static int collect_undone_grid_points(int *undone,
				      const Iarray *triplets,
				      const char *phonon_done);
static int get_phonons(lapack_complex_double *a,
		       double *w,
		       const double q[3],
		       const Darray *fc2,
		       const double *masses,
		       const int *p2s,
		       const int *s2p,
		       const Iarray *multi,
		       const Darray *svecs,
		       const double *born,
		       const double *dielectric,
		       const double *reciprocal_lattice,
		       const double *q_direction,
		       const double nac_factor,
		       const char uplo);
static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b);

int set_phonon_triplets(Darray *freqs,
			Carray *eigvecs,
			char *phonon_done,
			const Iarray *triplets,
			const Iarray *grid_address,
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
  int i, j, gp, num_patom, num_band, num_triplets, num_grid_points, num_undone;
  int *undone;
  double f;
  double q[3];

  num_patom = multi_fc2->dims[1];
  num_band = num_patom * 3;
  num_triplets = triplets->dims[0];
  num_grid_points = grid_address->dims[0];

  undone = (int*)malloc(sizeof(int) * num_grid_points);
  num_undone = collect_undone_grid_points(undone, triplets, phonon_done);

#pragma omp parallel for private(j, q, gp, f)
  for (i = 0; i < num_undone; i++) {
    gp = undone[i];
    for (j = 0; j < 3; j++) {
      q[j] = ((double)grid_address->data[gp * 3 + j]) / mesh[j];
    }

    if (gp == 0) {
      get_phonons(eigvecs->data + num_band * num_band * gp,
		  freqs->data + num_band * gp,
		  q,
		  fc2,
		  masses_fc2,
		  p2s_fc2,
		  s2p_fc2,
		  multi_fc2,
		  svecs_fc2,
		  born,
		  dielectric,
		  reciprocal_lattice,
		  q_direction,
		  nac_factor,
		  uplo);
    } else {
      get_phonons(eigvecs->data + num_band * num_band * gp,
		  freqs->data + num_band * gp,
		  q,
		  fc2,
		  masses_fc2,
		  p2s_fc2,
		  s2p_fc2,
		  multi_fc2,
		  svecs_fc2,
		  born,
		  dielectric,
		  reciprocal_lattice,
		  NULL,
		  nac_factor,
		  uplo);
    }
    for (j = 0; j < num_band; j++) {
      f = freqs->data[num_band * gp + j];
      freqs->data[num_band * gp + j] = 
	sqrt(fabs(f)) * ((f > 0) - (f < 0)) * unit_conversion_factor;
    }
    phonon_done[gp] = 1;
  }

  free(undone);

  return 0;
}

int get_interaction(Darray *amps,
		    Darray *freqs,
		    Carray *eigvecs,
		    char *phonon_done,
		    const Iarray *triplets,
		    const Iarray *grid_address,
		    const int *mesh,
		    const Darray *fc2,
		    const Darray *fc3,
		    const Darray *svecs_fc2,
		    const Iarray *multi_fc2,
		    const Darray *svecs_fc3,
		    const Iarray *multi_fc3,
		    const double *masses_fc2,
		    const double *masses_fc3,
		    const int *p2s_fc2,
		    const int *s2p_fc2,
		    const int *p2s_fc3,
		    const int *s2p_fc3,
		    const Iarray *band_indices,
		    const double *born,
		    const double *dielectric,
		    const double *reciprocal_lattice,
		    const double *q_direction,
		    const double nac_factor,
		    const double freq_unit_factor,
		    const char uplo)
{
  int i, j, gp, num_patom, num_band, num_triplets, num_grid_points, num_undone;
  int *undone;
  double f;
  double q[3];

  /* First collect all grid-points where phonon is not yet calculated. */
  /* Second calculate phonons distributing by OpenMP. */
  /* Third proceed to calculate interaction. */


/* #pragma omp parallel for private(j, w, a, q_vecs) */
/*   for (i = 0; i < num_triplets; i++) { */
/*     ; */
/*   } */

/* #pragma omp parallel for */
/*   for (i = 0; */
/*        i < num_triplets * band_indices->dims[0] * num_band * num_band; */
/*        i++) { */
/*     amps[i] /= num_grid_points; */
/*   } */

  return 1;
}

static int collect_undone_grid_points(int *undone,
				      const Iarray *triplets,
				      const char *phonon_done)
{
  int i, j, gp, num_undone;

  num_undone = 0;

  for (i = 0; i < 3; i++) {
    gp = triplets->data[i];
    if (phonon_done[gp] == 0) {
      undone[num_undone] = gp;
      num_undone++;
    }
  }

  for (i = 0; i < triplets->dims[0]; i++) {
    for (j = 0; j < 2; j++) {
      gp = triplets->data[i * 3 + j + 1];
      if (phonon_done[gp] == 0) {
	undone[num_undone] = gp;
	num_undone++;
      }
    }
  }

  return num_undone;
}

static int get_phonons(lapack_complex_double *a,
		       double *w,
		       const double q[3],
		       const Darray *fc2,
		       const double *masses,
		       const int *p2s,
		       const int *s2p,
		       const Iarray *multi,
		       const Darray *svecs,
		       const double *born,
		       const double *dielectric,
		       const double *reciprocal_lattice,
		       const double *q_direction,
		       const double nac_factor,
		       const char uplo)
{
  int i, j, num_patom, num_satom;
  double q_cart[3];
  double *dm_real, *dm_imag, *charge_sum;
  double inv_dielectric_factor, dielectric_factor, tmp_val;

  num_patom = multi->dims[1];
  num_satom = multi->dims[0];

  dm_real = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
  dm_imag = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
  for (i = 0; i < num_patom * num_patom * 9; i++) {
    dm_real[i] = 0.0;
    dm_imag[i] = 0.0;
  }

  if (born) {
    if (fabs(q[0]) < 1e-10 && fabs(q[1]) < 1e-10 && fabs(q[2]) < 1e-10 &&
	(!q_direction)) {
      charge_sum = NULL;
    } else {
      charge_sum = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
      if (q_direction) {
	for (i = 0; i < 3; i++) {
	  q_cart[i] = 0.0;
	  for (j = 0; j < 3; j++) {
	    q_cart[i] += reciprocal_lattice[i * 3 + j] * q_direction[j];
	  }
	}
      } else {
	for (i = 0; i < 3; i++) {
	  q_cart[i] = 0.0;
	  for (j = 0; j < 3; j++) {
	    q_cart[i] += reciprocal_lattice[i * 3 + j] * q[j];
	  }
	}
      }

      inv_dielectric_factor = 0.0;
      for (i = 0; i < 3; i++) {
	tmp_val = 0.0;
	for (j = 0; j < 3; j++) {
	  tmp_val += dielectric[i * 3 + j] * q_cart[j];
	}
	inv_dielectric_factor += tmp_val * q_cart[i];
      }
      /* N = num_satom / num_patom = number of prim-cell in supercell */
      /* N is used for Wang's method. */
      dielectric_factor = nac_factor /
	inv_dielectric_factor / num_satom * num_patom;
      get_charge_sum(charge_sum,
		     num_patom,
		     dielectric_factor,
		     q_cart,
		     born);
    }
  } else {
    charge_sum = NULL;
  }
  get_dynamical_matrix_at_q(dm_real,
  			    dm_imag,
  			    num_patom,
  			    num_satom,
  			    fc2->data,
  			    q,
  			    svecs->data,
  			    multi->data,
   			    masses,
  			    s2p,
  			    p2s,
  			    charge_sum);
  if (born) {
    free(charge_sum);
  }

  for (i = 0; i < num_patom * 3; i++) {
    for (j = 0; j < num_patom * 3; j++) {
      a[i * num_patom * 3 + j] = lapack_make_complex_double
	((dm_real[i * num_patom * 3 + j] + dm_real[j * num_patom * 3 + i]) / 2,
	 (dm_imag[i * num_patom * 3 + j] - dm_imag[j * num_patom * 3 + i]) / 2);
    }
  }

  free(dm_real);
  free(dm_imag);

  return phonopy_zheev(w, a, num_patom * 3, uplo);
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
