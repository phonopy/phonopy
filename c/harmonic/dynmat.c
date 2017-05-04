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

#include <math.h>
#include <stdlib.h>
#include <dynmat.h>
#define PI 3.14159265358979323846

static void get_dynmat_ij(double *dynamical_matrix,
			  const int num_patom, 
			  const int num_satom,
			  const double *fc,
			  const double *q,
			  const double *r,
			  const int *multi,
			  const double *mass,
			  const int *s2p_map, 
			  const int *p2s_map,
			  const double *charge_sum,
			  const int i,
			  const int j);
static void get_dm(double dm_real[3][3],
		   double dm_imag[3][3],
		   const int num_patom, 
		   const int num_satom,
		   const double *fc,
		   const double *q,
		   const double *r,
		   const int *multi,
		   const double mass_sqrt,
		   const int *p2s_map,
		   const double *charge_sum,
		   const int i,
		   const int j,
		   const int k);
static double get_dielectric_part(const double q[3],
                                  const double *dielectric);

int get_dynamical_matrix_at_q(double *dynamical_matrix,
			      const int num_patom, 
			      const int num_satom,
			      const double *fc,
			      const double *q,
			      const double *r,
			      const int *multi,
			      const double *mass,
			      const int *s2p_map, 
			      const int *p2s_map,
			      const double *charge_sum,
			      const int with_openmp)
{
  int i, j, ij, adrs, adrsT;
  
  if (with_openmp) {
#pragma omp parallel for
    for (ij = 0; ij < num_patom * num_patom ; ij++) {
      get_dynmat_ij(dynamical_matrix,
		    num_patom, 
		    num_satom,
		    fc,
		    q,
		    r,
		    multi,
		    mass,
		    s2p_map, 
		    p2s_map,
		    charge_sum,
		    ij / num_patom,  /* i */
		    ij % num_patom); /* j */
    }
  } else {
    for (i = 0; i < num_patom; i++) {
      for (j = 0; j < num_patom; j++) {
	get_dynmat_ij(dynamical_matrix,
		      num_patom, 
		      num_satom,
		      fc,
		      q,
		      r,
		      multi,
		      mass,
		      s2p_map, 
		      p2s_map,
		      charge_sum,
		      i,
		      j);
      }
    }
  }

  /* Symmetrize to be a Hermitian matrix */
  for (i = 0; i < num_patom * 3; i++) {
    for (j = i; j < num_patom * 3; j++) {
      adrs = i * num_patom * 6 + j * 2;
      adrsT = j * num_patom * 6 + i * 2;
      dynamical_matrix[adrs] += dynamical_matrix[adrsT];
      dynamical_matrix[adrs] /= 2;
      dynamical_matrix[adrs + 1] -= dynamical_matrix[adrsT+ 1];
      dynamical_matrix[adrs + 1] /= 2;
      dynamical_matrix[adrsT] = dynamical_matrix[adrs];
      dynamical_matrix[adrsT + 1] = -dynamical_matrix[adrs + 1];
    }
  }

  return 0;
}

void get_dipole_dipole(double *dd, /* [natom, 3, natom, 3, (real, imag)] */
                       const double *K_list, /* [num_kvec, 3] */
                       const int num_K,
                       const int num_patom,
                       const double *q_vector,
                       const double *q_direction,
                       const double *born,
                       const double *dielectric,
                       const double factor, /* 4pi/V*unit-conv */
                       const double *pos, /* [natom, 3] */
                       const double tolerance)
{
  int i, j, k, l, g, adrs;
  double q_K[3], q_G[3];
  double norm, cos_phase, sin_phase, phase, z;
  double *charge_sum;

  charge_sum = NULL;

  for (i = 0; i < num_patom * num_patom * 18; i++) {
    dd[i] = 0;
  }
  charge_sum = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);

  for (g = 0; g < num_K; g++) {
    norm = 0;
    for (i = 0; i < 3; i++) {
      norm += K_list[g * 3 + i] * K_list[g * 3 + i];
    }

    if (sqrt(norm) < tolerance) {
      if (!q_direction) {
        continue;
      } else {
        for (i = 0; i < 3; i++) {q_K[i] = q_direction[i];} 
      }
    } else {
      for (i = 0; i < 3; i++) {q_K[i] = K_list[g * 3 + i];} 
    }

    get_charge_sum(charge_sum,
                   num_patom,
                   factor / get_dielectric_part(q_K, dielectric),
                   q_K,
                   born);

    for (i = 0; i < num_patom; i++) {
      for (j = 0; j < num_patom; j++) {
        phase = 0;
        for (k = 0; k < 3; k++) {
          phase += (pos[i * 3 + k] - pos[j * 3 + k]) * K_list[g * 3 + k];
        }
        phase *= 2 * PI;
        cos_phase = cos(phase);
        sin_phase = sin(phase);
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            adrs = i * num_patom * 18 + k * num_patom * 6 + j * 6 + l * 2;
            z = charge_sum[i * num_patom * 9 + j * 9 + k * 3 + l];
            dd[adrs] += z * cos_phase;
            dd[adrs + 1] += z * sin_phase;
          }
        }
      }
    }
  }

  for (g = 0; g < num_K; g++) {
    norm = 0;
    for (i = 0; i < 3; i++) {
      q_G[i] = K_list[g * 3 + i] - q_vector[i];
      norm += q_G[i] * q_G[i];
    }

    if (sqrt(norm) < tolerance) {
      continue;
    }

    get_charge_sum(charge_sum,
                   num_patom,
                   factor / get_dielectric_part(q_G, dielectric),
                   q_G,
                   born);

    for (i = 0; i < num_patom; i++) {
      for (j = 0; j < num_patom; j++) {
        phase = 0;
        for (k = 0; k < 3; k++) {
          phase += (pos[i * 3 + k] - pos[j * 3 + k]) * q_G[k];
        }
        phase *= 2 * PI;
        cos_phase = cos(phase);
        sin_phase = sin(phase);
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            /* i is correct about the third index, not j */
            adrs = i * num_patom * 18 + k * num_patom * 6 + i * 6 + l * 2;
            z = charge_sum[i * num_patom * 9 + j * 9 + k * 3 + l];
            dd[adrs] -= z * cos_phase;
            dd[adrs + 1] -= z * sin_phase;
          }
        }
      }
    }
  }

  free(charge_sum);
  charge_sum = NULL;
}

void get_charge_sum(double *charge_sum,
		    const int num_patom,
		    const double factor, /* 4pi/V*unit-conv and denominator */
		    const double q_vector[3],
		    const double *born)
{
  int i, j, k, a, b;
  double (*q_born)[3];
  
  q_born = (double (*)[3]) malloc(sizeof(double[3]) * num_patom);
  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < 3; j++) {
      q_born[i][j] = 0;
    }
  }

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	q_born[i][j] += q_vector[k] * born[i * 9 + k * 3 + j];
      }
    }
  }

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (a = 0; a < 3; a++) {
	for (b = 0; b < 3; b++) {
	  charge_sum[i * 9 * num_patom + j * 9 + a * 3 + b] =
	    q_born[i][a] * q_born[j][b] * factor;
	}
      }
    }
  }

  free(q_born);
  q_born = NULL;
}


static void get_dynmat_ij(double *dynamical_matrix,
			  const int num_patom, 
			  const int num_satom,
			  const double *fc,
			  const double *q,
			  const double *r,
			  const int *multi,
			  const double *mass,
			  const int *s2p_map, 
			  const int *p2s_map,
			  const double *charge_sum,
			  const int i,
			  const int j)
{
  int k, l, adrs;
  double mass_sqrt;
  double dm_real[3][3], dm_imag[3][3];
  
  mass_sqrt = sqrt(mass[i] * mass[j]);

  for (k = 0; k < 3; k++) {
    for (l = 0; l < 3; l++) {
      dm_real[k][l] = 0;
      dm_imag[k][l] = 0;
    }
  }

  for (k = 0; k < num_satom; k++) { /* Lattice points of right index of fc */
    if (s2p_map[k] != p2s_map[j]) {
      continue;
    }
    get_dm(dm_real,
	   dm_imag,
	   num_patom, 
	   num_satom,
	   fc,
	   q,
	   r,
	   multi,
	   mass_sqrt,
	   p2s_map,
	   charge_sum,
	   i,
	   j,
	   k);
  }
      
  for (k = 0; k < 3; k++) {
    for (l = 0; l < 3; l++) {
      adrs = (i * 3 + k) * num_patom * 6 + j * 6 + l * 2;
      dynamical_matrix[adrs] = dm_real[k][l];
      dynamical_matrix[adrs + 1] = dm_imag[k][l];
    }
  }
}

static void get_dm(double dm_real[3][3],
		   double dm_imag[3][3],
		   const int num_patom, 
		   const int num_satom,
		   const double *fc,
		   const double *q,
		   const double *r,
		   const int *multi,
		   const double mass_sqrt,
		   const int *p2s_map,
		   const double *charge_sum,
		   const int i,
		   const int j,
		   const int k)
{
  int l, m;
  double phase, cos_phase, sin_phase, fc_elem;

  cos_phase = 0;
  sin_phase = 0;

  for (l = 0; l < multi[k * num_patom + i]; l++) {
    phase = 0;
    for (m = 0; m < 3; m++) {
      phase += q[m] * r[k * num_patom * 81 + i * 81 + l * 3 + m];
    }
    cos_phase += cos(phase * 2 * PI) / multi[k * num_patom + i];
    sin_phase += sin(phase * 2 * PI) / multi[k * num_patom + i];
  }

  for (l = 0; l < 3; l++) {
    for (m = 0; m < 3; m++) {
      if (charge_sum) {
	fc_elem = (fc[p2s_map[i] * num_satom * 9 + k * 9 + l * 3 + m] +
		   charge_sum[i * num_patom * 9 +
			      j * 9 + l * 3 + m]) / mass_sqrt;
      } else {
	fc_elem = fc[p2s_map[i] * num_satom * 9 +
		     k * 9 + l * 3 + m] / mass_sqrt;
      }
      dm_real[l][m] += fc_elem * cos_phase;
      dm_imag[l][m] += fc_elem * sin_phase;
    }
  }
}

static double get_dielectric_part(const double q[3],
                                  const double *dielectric)
{
  int i, j;
  double x[3];
  double sum;

  for (i = 0; i < 3; i++) {
    x[i] = 0;
    for (j = 0; j < 3; j++) {
      x[i] += dielectric[i * 3 + j] * q[j];
    }
  }

  sum = 0;
  for (i = 0; i < 3; i++) {
    sum += q[i] * x[i];
  }

  return sum;
}
