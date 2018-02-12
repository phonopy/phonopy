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
                          const double q[3],
                          PHPYCONST double (*svecs)[27][3],
                          const int *multi,
                          const double *mass,
                          const int *s2p_map,
                          const int *p2s_map,
                          PHPYCONST double (*charge_sum)[3][3],
                          const int i,
                          const int j);
static void get_dm(double dm_real[3][3],
                   double dm_imag[3][3],
                   const int num_patom,
                   const int num_satom,
                   const double *fc,
                   const double q[3],
                   PHPYCONST double (*svecs)[27][3],
                   const int *multi,
                   const int *p2s_map,
                   PHPYCONST double (*charge_sum)[3][3],
                   const int i,
                   const int j,
                   const int k);
static double get_dielectric_part(const double q_cart[3],
                                  PHPYCONST double dielectric[3][3]);
static void get_KK(double *dd_part, /* [natom, 3, natom, 3, (real,imag)] */
                   PHPYCONST double (*G_list)[3], /* [num_G, 3] */
                   const int num_G,
                   const int num_patom,
                   const double q_cart[3],
                   const double *q_direction_cart,
                   PHPYCONST double dielectric[3][3],
                   PHPYCONST double (*pos)[3], /* [num_patom, 3] */
                   const double lambda,
                   const double tolerance);
static void make_Hermitian(double *mat, const int num_band);
static void multiply_borns(double *dd,
                           const double *dd_in,
                           const int num_patom,
                           PHPYCONST double (*born)[3][3]);

int dym_get_dynamical_matrix_at_q(double *dynamical_matrix,
                                  const int num_patom,
                                  const int num_satom,
                                  const double *fc,
                                  const double q[3],
                                  PHPYCONST double (*svecs)[27][3],
                                  const int *multi,
                                  const double *mass,
                                  const int *s2p_map,
                                  const int *p2s_map,
                                  PHPYCONST double (*charge_sum)[3][3],
                                  const int with_openmp)
{
  int i, j, ij;

  if (with_openmp) {
#pragma omp parallel for
    for (ij = 0; ij < num_patom * num_patom ; ij++) {
      get_dynmat_ij(dynamical_matrix,
                    num_patom,
                    num_satom,
                    fc,
                    q,
                    svecs,
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
                      svecs,
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

  make_Hermitian(dynamical_matrix, num_patom * 3);

  return 0;
}

void dym_get_dipole_dipole(double *dd, /* [natom, 3, natom, 3, (real,imag)] */
                           const double *dd_q0, /* [natom, 3, 3, (real,imag)] */
                           PHPYCONST double (*G_list)[3], /* [num_G, 3] */
                           const int num_G,
                           const int num_patom,
                           const double q_cart[3],
                           const double *q_direction_cart, /* must be pointer */
                           PHPYCONST double (*born)[3][3],
                           PHPYCONST double dielectric[3][3],
                           PHPYCONST double (*pos)[3], /* [num_patom, 3] */
                           const double factor, /* 4pi/V*unit-conv */
                           const double lambda,
                           const double tolerance)
{
  int i, k, l, adrs, adrs_sum;
  double *dd_tmp;

  dd_tmp = NULL;
  dd_tmp = (double*) malloc(sizeof(double) * num_patom * num_patom * 18);

  for (i = 0; i < num_patom * num_patom * 18; i++) {
    dd[i] = 0;
    dd_tmp[i] = 0;
  }

  get_KK(dd_tmp,
         G_list,
         num_G,
         num_patom,
         q_cart,
         q_direction_cart,
         dielectric,
         pos,
         lambda,
         tolerance);

  multiply_borns(dd, dd_tmp, num_patom, born);

  for (i = 0; i < num_patom; i++) {
    for (k = 0; k < 3; k++) {   /* alpha */
      for (l = 0; l < 3; l++) { /* beta */
        adrs = i * num_patom * 9 + k * num_patom * 3 + i * 3 + l;
        adrs_sum = i * 9 + k * 3 + l;
        dd[adrs * 2] -= dd_q0[adrs_sum * 2];
        dd[adrs * 2 + 1] -= dd_q0[adrs_sum * 2 + 1];
      }
    }
  }

  for (i = 0; i < num_patom * num_patom * 18; i++) {
    dd[i] *= factor;
  }

  /* This may not be necessary. */
  make_Hermitian(dd, num_patom * 3);

  free(dd_tmp);
  dd_tmp = NULL;
}

void dym_get_dipole_dipole_q0(double *dd_q0, /* [natom, 3, 3, (real,imag)] */
                              PHPYCONST double (*G_list)[3], /* [num_G, 3] */
                              const int num_G,
                              const int num_patom,
                              PHPYCONST double (*born)[3][3],
                              PHPYCONST double dielectric[3][3],
                              PHPYCONST double (*pos)[3], /* [num_patom, 3] */
                              const double lambda,
                              const double tolerance)
{
  int i, j, k, l, adrs_tmp, adrs, adrsT;
  double zero_vec[3];
  double *dd_tmp1, *dd_tmp2;

  dd_tmp1 = NULL;
  dd_tmp1 = (double*) malloc(sizeof(double) * num_patom * num_patom * 18);
  dd_tmp2 = NULL;
  dd_tmp2 = (double*) malloc(sizeof(double) * num_patom * num_patom * 18);

  for (i = 0; i < num_patom * num_patom * 18; i++) {
    dd_tmp1[i] = 0;
    dd_tmp2[i] = 0;
  }

  zero_vec[0] = 0;
  zero_vec[1] = 0;
  zero_vec[2] = 0;

  get_KK(dd_tmp1,
         G_list,
         num_G,
         num_patom,
         zero_vec,
         NULL,
         dielectric,
         pos,
         lambda,
         tolerance);

  multiply_borns(dd_tmp2, dd_tmp1, num_patom, born);

  for (i = 0; i < num_patom * 18; i++) {
    dd_q0[i] = 0;
  }

  for (i = 0; i < num_patom; i++) {
    for (k = 0; k < 3; k++) {   /* alpha */
      for (l = 0; l < 3; l++) { /* beta */
        adrs = i * 9 + k * 3 + l;
        for (j = 0; j < num_patom; j++) {
          adrs_tmp = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l ;
          dd_q0[adrs * 2] += dd_tmp2[adrs_tmp * 2];
          dd_q0[adrs * 2 + 1] += dd_tmp2[adrs_tmp * 2 + 1];
        }
      }
    }
  }

  for (i = 0; i < num_patom; i++) {
    for (k = 0; k < 3; k++) {   /* alpha */
      for (l = 0; l < 3; l++) { /* beta */
        adrs = i * 9 + k * 3 + l;
        adrsT = i * 9 + l * 3 + k;
        dd_q0[adrs * 2] += dd_q0[adrsT * 2];
        dd_q0[adrs * 2] /= 2;
        dd_q0[adrsT * 2] = dd_q0[adrs * 2];
        dd_q0[adrs * 2 + 1] -= dd_q0[adrsT * 2 + 1];
        dd_q0[adrs * 2 + 1] /= 2;
        dd_q0[adrsT * 2 + 1] = -dd_q0[adrs * 2 + 1];
      }
    }
  }

  free(dd_tmp1);
  dd_tmp1 = NULL;
  free(dd_tmp2);
  dd_tmp2 = NULL;
}

void dym_get_charge_sum(double (*charge_sum)[3][3],
                        const int num_patom,
                        const double factor, /* 4pi/V*unit-conv and denominator */
                        const double q_cart[3],
                        PHPYCONST double (*born)[3][3])
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
        q_born[i][j] += q_cart[k] * born[i][k][j];
      }
    }
  }

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (a = 0; a < 3; a++) {
        for (b = 0; b < 3; b++) {
          charge_sum[i * num_patom + j][a][b] =
            q_born[i][a] * q_born[j][b] * factor;
        }
      }
    }
  }

  free(q_born);
  q_born = NULL;
}

/* fc[num_patom, num_satom, 3, 3] */
/* dm[num_comm_points, num_patom * 3, num_patom *3] */
/* comm_points[num_satom, num_patom, 27, 3] */
/* shortest_vectors[num_satom, num_patom, 27, 3] */
/* multiplicities[num_satom, num_patom] */
void dym_transform_dynmat_to_fc(double *fc,
                                const double *dm,
                                PHPYCONST double (*comm_points)[3],
                                PHPYCONST double (*shortest_vectors)[27][3],
                                const int *multiplicities,
                                const double *masses,
                                const int *s2pp_map,
                                const int num_patom,
                                const int num_satom)
{
  int i, j, k, l, m, N, adrs, multi;
  double coef, phase, cos_phase, sin_phase;

  N = num_satom / num_patom;
  for (i = 0; i < num_patom * num_satom * 9; i++) {
    fc[i] = 0;
  }

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_satom; j++) {
      coef = sqrt(masses[i] * masses[s2pp_map[j]]) / N;
      for (k = 0; k < N; k++) {
        cos_phase = 0;
        sin_phase = 0;
        multi = multiplicities[j * num_patom + i];
        for (l = 0; l < multi; l++) {
          phase = 0;
          for (m = 0; m < 3; m++) {
            phase -= comm_points[k][m] *
              shortest_vectors[j * num_patom + i][l][m];
          }
          cos_phase += cos(phase * 2 * PI);
          sin_phase += sin(phase * 2 * PI);
        }
        cos_phase /=  multi;
        sin_phase /=  multi;
        for (l = 0; l < 3; l++) {
          for (m = 0; m < 3; m++) {
            adrs = k * num_patom * num_patom * 18 + i * num_patom * 18 +
              l * num_patom * 6 + s2pp_map[j] * 6 + m * 2;
            fc[i * num_satom * 9 + j * 9 + l * 3 + m] +=
              (dm[adrs] * cos_phase - dm[adrs + 1] * sin_phase) * coef;
          }
        }
      }
    }
  }
}


static void get_dynmat_ij(double *dynamical_matrix,
                          const int num_patom,
                          const int num_satom,
                          const double *fc,
                          const double q[3],
                          PHPYCONST double (*svecs)[27][3],
                          const int *multi,
                          const double *mass,
                          const int *s2p_map,
                          const int *p2s_map,
                          PHPYCONST double (*charge_sum)[3][3],
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
           svecs,
           multi,
           p2s_map,
           charge_sum,
           i,
           j,
           k);
  }

  for (k = 0; k < 3; k++) {
    for (l = 0; l < 3; l++) {
      adrs = (i * 3 + k) * num_patom * 3 + j * 3 + l;
      dynamical_matrix[adrs * 2] = dm_real[k][l] / mass_sqrt;
      dynamical_matrix[adrs * 2 + 1] = dm_imag[k][l] / mass_sqrt;
    }
  }
}

static void get_dm(double dm_real[3][3],
                   double dm_imag[3][3],
                   const int num_patom,
                   const int num_satom,
                   const double *fc,
                   const double q[3],
                   PHPYCONST double (*svecs)[27][3],
                   const int *multi,
                   const int *p2s_map,
                   PHPYCONST double (*charge_sum)[3][3],
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
      phase += q[m] * svecs[k * num_patom + i][l][m];
    }
    cos_phase += cos(phase * 2 * PI) / multi[k * num_patom + i];
    sin_phase += sin(phase * 2 * PI) / multi[k * num_patom + i];
  }

  for (l = 0; l < 3; l++) {
    for (m = 0; m < 3; m++) {
      if (charge_sum) {
        fc_elem = (fc[p2s_map[i] * num_satom * 9 + k * 9 + l * 3 + m] +
                   charge_sum[i * num_patom + j][l][m]);
      } else {
        fc_elem = fc[p2s_map[i] * num_satom * 9 + k * 9 + l * 3 + m];
      }
      dm_real[l][m] += fc_elem * cos_phase;
      dm_imag[l][m] += fc_elem * sin_phase;
    }
  }
}

static double get_dielectric_part(const double q_cart[3],
                                  PHPYCONST double dielectric[3][3])
{
  int i, j;
  double x[3];
  double sum;

  for (i = 0; i < 3; i++) {
    x[i] = 0;
    for (j = 0; j < 3; j++) {
      x[i] += dielectric[i][j] * q_cart[j];
    }
  }

  sum = 0;
  for (i = 0; i < 3; i++) {
    sum += q_cart[i] * x[i];
  }

  return sum;
}

static void get_KK(double *dd_part, /* [natom, 3, natom, 3, (real,imag)] */
                   PHPYCONST double (*G_list)[3], /* [num_G, 3] */
                   const int num_G,
                   const int num_patom,
                   const double q_cart[3],
                   const double *q_direction_cart,
                   PHPYCONST double dielectric[3][3],
                   PHPYCONST double (*pos)[3], /* [num_patom, 3] */
                   const double lambda,
                   const double tolerance)
{
  int i, j, k, l, g, adrs;
  double q_K[3];
  double norm, cos_phase, sin_phase, phase, dielectric_part, exp_damp, L2;
  double KK[3][3];

  L2 = 4 * lambda * lambda;

  /* sum over K = G + q and over G (i.e. q=0) */
  /* q_direction has values for summation over K at Gamma point. */
  /* q_direction is NULL for summation over G */
  for (g = 0; g < num_G; g++) {
    norm = 0;
    for (i = 0; i < 3; i++) {
      q_K[i] = G_list[g][i] + q_cart[i];
      norm += q_K[i] * q_K[i];
    }

    if (sqrt(norm) < tolerance) {
      if (!q_direction_cart) {
        continue;
      } else {
        dielectric_part = get_dielectric_part(q_direction_cart, dielectric);
        for (i = 0; i < 3; i++) {
          for (j = 0; j < 3; j++) {
            KK[i][j] =
              q_direction_cart[i] * q_direction_cart[j] / dielectric_part;
          }
        }
      }
    } else {
      dielectric_part = get_dielectric_part(q_K, dielectric);
      exp_damp = exp(-dielectric_part / L2);
      for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
          KK[i][j] = q_K[i] * q_K[j] / dielectric_part * exp_damp;
        }
      }
    }

    for (i = 0; i < num_patom; i++) {
      for (j = 0; j < num_patom; j++) {
        phase = 0;
        for (k = 0; k < 3; k++) {
          /* For D-type dynamical matrix */
          /* phase += (pos[i][k] - pos[j][k]) * q_K[k]; */
          /* For C-type dynamical matrix */
          phase += (pos[i][k] - pos[j][k]) * G_list[g][k];
        }
        phase *= 2 * PI;
        cos_phase = cos(phase);
        sin_phase = sin(phase);
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
            dd_part[adrs * 2] += KK[k][l] * cos_phase;
            dd_part[adrs * 2 + 1] += KK[k][l] * sin_phase;
          }
        }
      }
    }
  }
}

static void make_Hermitian(double *mat, const int num_band)
{
  int i, j, adrs, adrsT;

  for (i = 0; i < num_band; i++) {
    for (j = i; j < num_band; j++) {
      adrs = i * num_band + j * 1;
      adrs *= 2;
      adrsT = j * num_band + i * 1;
      adrsT *= 2;
      /* real part */
      mat[adrs] += mat[adrsT];
      mat[adrs] /= 2;
      /* imaginary part */
      mat[adrs + 1] -= mat[adrsT+ 1];
      mat[adrs + 1] /= 2;
      /* store */
      mat[adrsT] = mat[adrs];
      mat[adrsT + 1] = -mat[adrs + 1];
    }
  }
}

static void multiply_borns(double *dd,
                           const double *dd_in,
                           const int num_patom,
                           PHPYCONST double (*born)[3][3])
{
  int i, j, k, l, m, n, adrs, adrs_in;
  double zz;

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_patom; j++) {
      for (k = 0; k < 3; k++) {   /* alpha */
        for (l = 0; l < 3; l++) { /* beta */
          adrs = i * num_patom * 9 + k * num_patom * 3 + j * 3 + l;
          for (m = 0; m < 3; m++) { /* alpha' */
            for (n = 0; n < 3; n++) { /* beta' */
              adrs_in = i * num_patom * 9 + m * num_patom * 3 + j * 3 + n ;
              zz = born[i][m][k] * born[j][n][l];
              dd[adrs * 2] += dd_in[adrs_in * 2] * zz;
              dd[adrs * 2 + 1] += dd_in[adrs_in * 2 + 1] * zz;
            }
          }
        }
      }
    }
  }
}
