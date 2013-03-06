#include <stdlib.h>
#include <math.h>
#include <lapacke.h>
#include "alloc_array.h"
#include "dynmat.h"
#include "lapack_wrapper.h"
#include "interaction_strength.h"

static const int index_exchange[6][3] = { { 0, 1, 2 },
					  { 2, 0, 1 },
					  { 1, 2, 0 },
					  { 2, 1, 0 },
					  { 0, 2, 1 },
					  { 1, 0, 2 } };

static int sum_interaction_strength(double *amps,
				    const lapack_complex_double* eigvecs,
				    const double *freqs,
				    const double *masses,
				    const Array1D *p2s,
				    const Array2D *multi,
				    const ShortestVecs *svecs,
				    const CArray2D *fc3_q,
				    const Array1D *band_indices,
				    const double cutoff_frequency);
static lapack_complex_double get_phase_factor(const double q[3],
				    const int s_atom_index,
				    const int p_atom_index,
				    const int sign,
				    const ShortestVecs * svecs,
				    const Array2D * multi);
static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b);
static int get_phonons(lapack_complex_double *a,
		       double *w,
		       const double q[3],
		       const double *fc2,
		       const double *masses,
		       const Array1D *p2s,
		       const Array1D *s2p,
		       const Array2D *multi,
		       const ShortestVecs *svecs,
		       const double *born,
		       const double nac_factor);

int get_interaction_strength(double *amps,
			     const double *q0,
			     const double *q1s,
			     const double *q2s,
			     const Array1D *weights,
			     const double *fc2,
			     const double *fc3,
			     const double *masses,
			     const Array1D *p2s,
			     const Array1D *s2p,
			     const Array2D *multi,
			     const ShortestVecs *svecs,
			     const double *born,
			     const double *dielectric,
			     const double nac_factor,
			     const double cutoff_frequency,
			     const int is_symmetrize_fc3_q,
			     const int r2q_TI_index,
			     const double symprec)
{
  int i, j, num_triplets, num_patom, info0, info1, info2;
  double q1[3], q2[3];
  double *w0 ,*w1, *w2;
  lapack_complex_double *a0, *a1, *a2;

  num_patom = p2s->d1;
  num_triplets = weights->d1;

  w0 = (double*)malloc(sizeof(double) * num_patom * 3);
  a0 = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_patom * num_patom * 9);
  info0 = get_phonons(a0,
		      w0,
		      q0,
		      fc2,
		      masses,
		      p2s,
		      s2p,
		      multi,
		      svecs,
		      born,
		      nac_factor);

  printf("freq0: ");
  for (i = 0; i < num_patom * 3; i++) {
    printf("%f ", w0[i]);
  }
  printf("\n");

  for (i = 0; i < num_triplets; i++) {
    w1 = (double*)malloc(sizeof(double) * num_patom * 3);
    w2 = (double*)malloc(sizeof(double) * num_patom * 3);
    a1 = (lapack_complex_double*)
      malloc(sizeof(lapack_complex_double) * num_patom * num_patom * 9);
    a2 = (lapack_complex_double*)
      malloc(sizeof(lapack_complex_double) * num_patom * num_patom * 9);
    
    for (j = 0; j < 3; j++) {
      q1[j] = q1s[i * 3 + j];
      q2[j] = q2s[i * 3 + j];
    }

    info1 = get_phonons(a1,
			w1,
			q1,
			fc2,
			masses,
			p2s,
			s2p,
			multi,
			svecs,
			born,
			nac_factor);

    info2 = get_phonons(a2,
			w2,
			q2,
			fc2,
			masses,
			p2s,
			s2p,
			multi,
			svecs,
			born,
			nac_factor);
    
    printf("freq1: ");
      for (j = 0; j < num_patom * 3; j++) {
	printf("%f ", w1[j]);
      }
    printf("\n");
    printf("freq2: ");
    for (j = 0; j < num_patom * 3; j++) {
      printf("%f ", w2[j]);
    }
    printf("\n");

    free(w1);
    free(w2);
    free(a1);
    free(a2);
  }

  free(w0);
  free(a0);
}

int get_triplet_interaction_strength(double *amps,
				     const double *fc3,
				     const double *q_vecs,
				     const lapack_complex_double* eigvecs,
				     const double *freqs,
				     const double *masses,
				     const Array1D *p2s,
				     const Array1D *s2p,
				     const Array2D *multi,
				     const ShortestVecs *svecs,
				     const Array1D *band_indices,
				     const double cutoff_frequency,
				     const int is_symmetrize_fc3_q,
				     const int r2q_TI_index,
				     const double symprec)
{
  int i, j, k, num_patom;
  DArray2D * q;
  CArray2D * fc3_q;

  num_patom = p2s->d1;
  
  if (is_symmetrize_fc3_q == 0) {
    fc3_q = alloc_CArray2D(1, num_patom * num_patom * num_patom * 27);
  } else {
    fc3_q = alloc_CArray2D(6, num_patom * num_patom * num_patom * 27);
  }

  q = alloc_DArray2D(3, 3);
  for (i = 0; i < fc3_q->d1; i++) {
    for (j = 0; j < 3; j ++) {
      for (k = 0; k < 3; k ++) {
	q->data[ index_exchange[i][j] ][k] = q_vecs[j * 3 + k];
      }
    }
    get_fc3_reciprocal(fc3_q->data[i],
		       svecs,
		       multi,
		       q,
		       s2p,
		       p2s,
		       fc3,
		       r2q_TI_index,
		       symprec);
  }

  sum_interaction_strength(amps,
			   eigvecs,
			   freqs,
			   masses,
			   p2s,
			   multi,
			   svecs,
			   fc3_q,
			   band_indices,
			   cutoff_frequency);

  free_DArray2D(q);
  free_CArray2D(fc3_q);

  return 1;
}

int get_fc3_realspace(lapack_complex_double* fc3_real,
		      const ShortestVecs* svecs,
		      const Array2D * multi,
		      const double* q_triplet,
		      const Array1D * s2p,
		      const lapack_complex_double* fc3_rec,
		      const double symprec)
{
  int i, j, k, l, m, n;
  lapack_complex_double phase2, phase3, fc3_elem;
  double q[3];
  int num_satom = multi->d1;
  int num_patom = multi->d2;

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_satom; j++) {
      q[0] = q_triplet[3];
      q[1] = q_triplet[4];
      q[2] = q_triplet[5];
      phase2 = get_phase_factor(q,
				j,
				i,
				-1,
				svecs,
				multi);
      for (k = 0; k < num_satom; k++) {
	q[0] = q_triplet[6];
	q[1] = q_triplet[7];
	q[2] = q_triplet[8];
	phase3 = get_phase_factor(q,
				  k,
				  i,
				  -1,
				  svecs,
				  multi);
	for (l = 0; l < 3; l++) { 
	  for (m = 0; m < 3; m++) {
	    for (n = 0; n < 3; n++) {
	      fc3_elem = prod(fc3_rec[i * num_patom * num_patom * 27 +
				      s2p->data[j] * num_patom * 27 +
				      s2p->data[k] * 27 + l * 9 + m * 3 + n],
			      prod(phase2, phase3));

	      fc3_real[i * num_satom * num_satom * 27 +
		       j * num_satom * 27 +
		       k * 27 + l * 9 + m * 3 + n] = fc3_elem;
	    }
	  }
	}
      }
    }
  }

  return 0;
}

int get_fc3_reciprocal(lapack_complex_double* fc3_q,
		       const ShortestVecs * svecs,
		       const Array2D * multi,
		       const DArray2D * q,
		       const Array1D * s2p,
		       const Array1D * p2s,
		       const double* fc3,
		       const int r2q_TI_index,
		       const double symprec)
{
  int i, j, k, l, m, n, p;
  lapack_complex_double fc3_q_local[3][3][3];
  int num_patom = p2s->d1;

#pragma omp parallel for private(i, j, k, l, m, n, fc3_q_local)
  for (p = 0; p < num_patom * num_patom * num_patom; p++) {
    i = p / (num_patom * num_patom);
    j = (p % (num_patom * num_patom)) / num_patom;
    k = p % num_patom;
    get_fc3_sum_in_supercell(fc3_q_local,
			     i,
			     j,
			     k,
			     svecs,
			     multi,
			     q,
			     s2p,
			     p2s,
			     fc3,
			     r2q_TI_index,
			     symprec);

    for (l = 0; l < 3; l++) { 
      for (m = 0; m < 3; m++) {
	for (n = 0; n < 3; n++) {
	  fc3_q[i * num_patom * num_patom * 27 +
		j * num_patom * 27 +
		k * 27 + l * 9 + m * 3 + n] = fc3_q_local[l][m][n];
	}
      }
    }
  }

  return 0;
}

int get_fc3_sum_in_supercell(lapack_complex_double fc3_q[3][3][3],
			     const int p1,
			     const int p2,
			     const int p3,
			     const ShortestVecs * svecs,
			     const Array2D * multi,
			     const DArray2D * q,
			     const Array1D * s2p,
			     const Array1D * p2s,
			     const double* fc3,
			     const int r2q_TI_index,
			     const double symprec)
{
  lapack_complex_double phase2, phase3, phase_prod, phase_prim;
  int i, j, k, s1, s2, s3, address;
  double phase;
  int num_satom = s2p->d1;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	fc3_q[i][j][k] = lapack_make_complex_double(0.0, 0.0);
      }
    }
  }
  
  /* Phi(s1, s2, s3) */
  /* Sum in terms of s1 is not taken due to translational invariance. */
  /* Phase factor with q_1 is not used. */
  s1 = 0;
  for (i = 0; i < num_satom; i++) {
    if (p2s->data[p1] == s2p->data[i]) {
      if (fabs(svecs->data[i][p1][0][0]) < symprec &&
	  fabs(svecs->data[i][p1][0][1]) < symprec &&
	  fabs(svecs->data[i][p1][0][2]) < symprec) {
	s1 = i;
      }
    }
  }

  /* Sum in terms of s2 */
  for (s2 = 0; s2 < num_satom; s2++) {
    if (s2p->data[s2] == p2s->data[p2]) {
      phase2 = lapack_make_complex_double(0.0, 0.0);
      /* Supercell boundary treatment */
      for (i = 0; i < multi->data[s2][p1]; i++) {
	phase = 0.0;
	/* phi' = q' * [r(N;nu) - r(M;mu)] */
	for (j = 0; j < 3; j++) {
	  phase += q->data[1][j] * svecs->data[s2][p1][i][j];
	}
	phase2 = lapack_make_complex_double
	  (lapack_complex_double_real(phase2) + cos(phase * 2 * M_PI),
	   lapack_complex_double_imag(phase2) + sin(phase * 2 * M_PI));
      }
      phase2 = lapack_make_complex_double
	(lapack_complex_double_real(phase2) / multi->data[s2][p1],
	 lapack_complex_double_imag(phase2) / multi->data[s2][p1]);

      /* Sum in terms of s3 */
      for (s3 = 0; s3 < num_satom; s3++) {
	if (s2p->data[s3] == p2s->data[p3]) {
	  phase3 = lapack_make_complex_double(0.0, 0.0);
	  for (i = 0; i < multi->data[s3][p1]; i++) {
	    phase = 0.0;
	    /* phi'' = q'' * [r(P;pi) - r(M;mu)] */
	    for (j = 0; j < 3; j++) {
	      phase += q->data[2][j] * svecs->data[s3][p1][i][j];
	    }
	    phase3 = lapack_make_complex_double
	      (lapack_complex_double_real(phase3) + cos(phase * 2 * M_PI),
	       lapack_complex_double_imag(phase3) + sin(phase * 2 * M_PI));
	  }
	  phase3 = lapack_make_complex_double
	    (lapack_complex_double_real(phase3) / multi->data[s3][p1],
	     lapack_complex_double_imag(phase3) / multi->data[s3][p1]);

	  /* Fourier transform */
	  phase_prod = prod(phase2, phase3);

	  switch (r2q_TI_index) {
	  case 1:
	    address = s3 * num_satom * num_satom * 27 + s1 * num_satom * 27 + s2 * 27;
	    break;
	  case 2:
	    address = s2 * num_satom * num_satom * 27 + s3 * num_satom * 27 + s1 * 27;
	    break;
	  default:
	    address = s1 * num_satom * num_satom * 27 + s2 * num_satom * 27 + s3 * 27;
	    break;
	  }
	  for (i = 0; i < 3; i++) {
	    for (j = 0; j < 3; j++) {
	      for (k = 0; k < 3; k++) {
		fc3_q[i][j][k] = lapack_make_complex_double
		  (lapack_complex_double_real(fc3_q[i][j][k]) +
		   fc3[address + i * 9 + j * 3 + k] *
		   lapack_complex_double_real(phase_prod),
		   lapack_complex_double_imag(fc3_q[i][j][k]) +
		   fc3[address + i * 9 + j * 3 + k] *
		   lapack_complex_double_imag(phase_prod));
	      }
	    }
	  }
	}
      }
    }
  }

  phase = 0.0;
  for (i = 0; i < 3; i++) {
    phase += (q->data[0][i] + q->data[1][i] + q->data[2][i]) * svecs->data[p2s->data[p1]][0][0][i];
  }
  phase_prim = lapack_make_complex_double
    (cos(phase * 2 * M_PI), sin(phase * 2 * M_PI));

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
  	fc3_q[i][j][k] = prod(fc3_q[i][j][k], phase_prim);
      }
    }
  }

  return 0;
}

static int sum_interaction_strength(double *amps,
				    const lapack_complex_double* eigvecs,
				    const double *freqs,
				    const double *masses,
				    const Array1D *p2s,
				    const Array2D *multi,
				    const ShortestVecs *svecs,
				    const CArray2D *fc3_q,
				    const Array1D *band_indices,
				    const double cutoff_frequency)
{
  int i, j, k, n, num_band0, num_patom;
  int band[3];
  lapack_complex_double * e[3];

  num_band0 = band_indices->d1;
  num_patom = p2s->d1;
  
#pragma omp parallel for private(i, j, k, band, e)
  for (n = 0; n < num_band0 * num_patom * num_patom * 9; n++) {
    band[0] = band_indices->data[n / (num_patom * num_patom * 9)];
    band[1] = (n % (num_patom * num_patom * 9)) / (num_patom * 3);
    band[2] = n % (num_patom * 3);

    if (freqs[band[0]] < cutoff_frequency ||
	freqs[num_patom * 3 + band[1]] < cutoff_frequency ||
	freqs[2 * num_patom * 3 + band[2]] < cutoff_frequency) {
      amps[n] = 0;
      continue;
    }

    for (i = 0; i < 3; i++) {
      e[i] = (lapack_complex_double*)
	malloc(num_patom * 3 * sizeof(lapack_complex_double));
    }

    /* If symmetrize fc3_q for index exchange, i = 0..5 else i = 0*/
    amps[n] = 0;
    for (i = 0; i < fc3_q->d1; i++) { 
      for (j = 0; j < num_patom * 3; j++) {
	for (k = 0; k < 3; k++) {
	  e[ index_exchange[i][k] ][j] =
	    eigvecs[k * num_patom * num_patom * 9 +
		    j * num_patom * 3 + band[k]];
	}
      }
      amps[n] += get_sum_in_primivie(fc3_q->data[i], e[0], e[1], e[2],
				     num_patom, masses);
    }
    amps[n] /= fc3_q->d1 *
      freqs[band[0]] *
      freqs[num_patom * 3 + band[1]] *
      freqs[2 * num_patom * 3 + band[2]];

    for (i = 0; i < 3; i++) {
      free(e[i]);
    }
  }

  return 1;
}


static lapack_complex_double
get_phase_factor(const double q[3],
		 const int s_atom_index,
		 const int p_atom_index,
		 const int sign,
		 const ShortestVecs * svecs,
		 const Array2D * multi)
{
  int i, j;
  double phase;
  lapack_complex_double exp_phase;
  int m = multi->data[s_atom_index][p_atom_index];
  
  exp_phase = lapack_make_complex_double(0.0, 0.0);
  for (i = 0; i < m; i++) {
    phase = 0.0;
    for (j = 0; j < 3; j++) {
      phase += q[j] * svecs->data[s_atom_index][p_atom_index][i][j];
    }
    exp_phase = lapack_make_complex_double
      (lapack_complex_double_real(exp_phase) + cos(phase * 2 * M_PI),
       lapack_complex_double_imag(exp_phase) + sin(phase * 2 * M_PI));
  }

  exp_phase = lapack_make_complex_double
    (lapack_complex_double_real(exp_phase) / m,
     lapack_complex_double_imag(exp_phase) / m / sign);

  return exp_phase;
}


double get_sum_in_primivie(const lapack_complex_double *fc3,
			   const lapack_complex_double *e1,
			   const lapack_complex_double *e2,
			   const lapack_complex_double *e3,
			   const int num_atom,
			   const double *mass)
{
  int i1, i2, i3, a, b, c, shift;
  double mass_sqrt;
  lapack_complex_double sum, local_sum, tmp_val;
  
  sum = lapack_make_complex_double(0.0, 0.0);

  for (i1 = 0; i1 < num_atom; i1++) {
    for (i2 = 0; i2 < num_atom; i2++) {
      for (i3 = 0; i3 < num_atom; i3++) {
	shift = i1 * num_atom * num_atom * 27 + i2 * num_atom * 27 + i3 * 27;
	local_sum = lapack_make_complex_double(0.0, 0.0);
	for (a = 0; a < 3; a++) {
	  for (b = 0; b < 3; b++) {
	    for (c = 0; c < 3; c++) {
	      tmp_val = prod(e1[i1 * 3 + a], e2[i2 * 3 + b]);
	      tmp_val = prod(tmp_val, e3[i3 * 3 + c]);
	      tmp_val = prod(fc3[shift + a * 9 + b * 3 + c], tmp_val);
	      local_sum = lapack_make_complex_double
		(lapack_complex_double_real(local_sum) +
		 lapack_complex_double_real(tmp_val),
		 lapack_complex_double_imag(local_sum) + 
		 lapack_complex_double_imag(tmp_val));
	    }
	  }
	}
	mass_sqrt = sqrt(mass[i1] * mass[i2] * mass[i3]);
	sum = lapack_make_complex_double
	  (lapack_complex_double_real(sum) +
	   lapack_complex_double_real(local_sum) / mass_sqrt,
	   lapack_complex_double_imag(sum) +
	   lapack_complex_double_imag(local_sum) / mass_sqrt);
      }
    }
  }

  return lapack_complex_double_real(sum) * lapack_complex_double_real(sum) +
    lapack_complex_double_imag(sum) * lapack_complex_double_imag(sum);
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

static int get_phonons(lapack_complex_double *a,
		       double *w,
		       const double q[3],
		       const double *fc2,
		       const double *masses,
		       const Array1D *p2s,
		       const Array1D *s2p,
		       const Array2D *multi,
		       const ShortestVecs *svecs,
		       const double *born,
		       const double nac_factor)
{
  int i, j, num_patom, num_satom;
  double *dm_real, *dm_imag, *charge_sum;

  num_patom = p2s->d1;
  num_satom = s2p->d1;

  dm_real = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
  dm_imag = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
  for (i = 0; i < num_patom * num_patom * 9; i++) {
    dm_real[i] = 0.0;
    dm_imag[i] = 0.0;
  }

  if (born) {
    charge_sum = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
    get_charge_sum(charge_sum,
		   num_patom,
		   nac_factor,
		   q,
		   born);
  } else {
    charge_sum = NULL;
  }
  get_dynamical_matrix_at_q(dm_real,
			    dm_imag,
			    num_patom,
			    num_satom,
			    fc2,
			    q,
			    svecs->data[0][0][0],
			    multi->data[0],
			    masses,
			    s2p->data,
			    p2s->data,
			    charge_sum);
  if (born) {
    free(charge_sum);
  }

  for (i = 0; i < num_patom * 3; i++) {
    for (j = 0; j < num_patom * 3; j++) {
      a[i * 3 + j] = lapack_make_complex_double
	((dm_real[i * 3 + j] + dm_real[j * 3 + i]) / 2,
	 (dm_imag[i * 3 + j] - dm_imag[j * 3 + i]) / 2);
    }
  }

  free(dm_real);
  free(dm_imag);

  return phonopy_zheev(w, a, num_patom * 3);
}
