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
#include <math.h>
#include <dynmat.h>
#include <phonoc_array.h>
#include <phonon.h>
#include <lapack_wrapper.h>

static int collect_undone_grid_points(int *undone,
				      char *phonon_done,
				      const int num_grid_points,
				      const int *grid_points);

void set_phonons_at_gridpoints(Darray *frequencies,
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

void get_undone_phonons(Darray *frequencies,
			Carray *eigenvectors,
			const int *undone_grid_points,
			const int num_undone_grid_points,
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
  int i, j, gp, num_band;
  double q[3];

  num_band = frequencies->dims[1];

#pragma omp parallel for private(j, q, gp)
  for (i = 0; i < num_undone_grid_points; i++) {
    gp = undone_grid_points[i];
    for (j = 0; j < 3; j++) {
      q[j] = ((double)grid_address[gp * 3 + j]) / mesh[j];
    }

    if (gp == 0) {
      get_phonons(eigenvectors->data + num_band * num_band * gp,
		  frequencies->data + num_band * gp,
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
		  unit_conversion_factor,
		  uplo);
    } else {
      get_phonons(eigenvectors->data + num_band * num_band * gp,
		  frequencies->data + num_band * gp,
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
		  unit_conversion_factor,
		  uplo);
    }
  }
}

int get_phonons(lapack_complex_double *a,
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
		const double unit_conversion_factor,
		const char uplo)
{
  int i, j, num_patom, num_satom, info;
  double q_cart[3];
  double *charge_sum;
  double inv_dielectric_factor, dielectric_factor, tmp_val;

  num_patom = multi->dims[1];
  num_satom = multi->dims[0];

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

  get_dynamical_matrix_at_q((double*)a,
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

  info = phonopy_zheev(w, a, num_patom * 3, uplo);
  
  for (i = 0; i < num_patom * 3; i++) {
    w[i] =
      sqrt(fabs(w[i])) * ((w[i] > 0) - (w[i] < 0)) * unit_conversion_factor;
  }
  
  return info;
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
