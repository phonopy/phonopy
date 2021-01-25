/* Copyright (C) 2021 Atsushi Togo */
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
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "dynmat.h"
#include "derivative_dynmat.h"
#include "phonopy.h"

static void set_index_permutation_symmetry_fc(double * fc, const int natom);
static void set_translational_symmetry_fc(double * fc, const int natom);
static void set_translational_symmetry_compact_fc(double * fc,
                                                  const int p2s[],
                                                  const int n_satom,
                                                  const int n_patom);
static int nint(const double a);

void phpy_transform_dynmat_to_fc(double *fc,
                                 const double *dm,
                                 PHPYCONST double (*comm_points)[3],
                                 PHPYCONST double (*shortest_vectors)[27][3],
                                 const int *multiplicities,
                                 const double *masses,
                                 const int *s2pp_map,
                                 const int *fc_index_map,
                                 const int num_patom,
                                 const int num_satom)
{
  dym_transform_dynmat_to_fc(fc,
                             dm,
                             comm_points,
                             shortest_vectors,
                             multiplicities,
                             masses,
                             s2pp_map,
                             fc_index_map,
                             num_patom,
                             num_satom);
}


int phpy_get_dynamical_matrix_at_q(double *dynamical_matrix,
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
  return dym_get_dynamical_matrix_at_q(dynamical_matrix,
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
                                       1);
}


void phpy_get_charge_sum(double (*charge_sum)[3][3],
                         const int num_patom,
                         const double factor, /* 4pi/V*unit-conv and denominator */
                         const double q_cart[3],
                         PHPYCONST double (*born)[3][3])
{
  dym_get_charge_sum(charge_sum, num_patom, factor, q_cart, born);
}


void phpy_get_recip_dipole_dipole(double *dd, /* [natom, 3, natom, 3, (real,imag)] */
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
  dym_get_recip_dipole_dipole(dd,
                              dd_q0,
                              G_list,
                              num_G,
                              num_patom,
                              q_cart,
                              q_direction_cart,
                              born,
                              dielectric,
                              pos,
                              factor,
                              lambda,
                              tolerance);
}


void phpy_get_recip_dipole_dipole_q0(double *dd_q0, /* [natom, 3, 3, (real,imag)] */
                                     PHPYCONST double (*G_list)[3], /* [num_G, 3] */
                                     const int num_G,
                                     const int num_patom,
                                     PHPYCONST double (*born)[3][3],
                                     PHPYCONST double dielectric[3][3],
                                     PHPYCONST double (*pos)[3], /* [num_patom, 3] */
                                     const double lambda,
                                     const double tolerance)
{
  dym_get_recip_dipole_dipole_q0(dd_q0,
                                 G_list,
                                 num_G,
                                 num_patom,
                                 born,
                                 dielectric,
                                 pos,
                                 lambda,
                                 tolerance);
}


void phpy_get_derivative_dynmat_at_q(double *derivative_dynmat,
                                     const int num_patom,
                                     const int num_satom,
                                     const double *fc,
                                     const double *q,
                                     const double *lattice, /* column vector */
                                     const double *r,
                                     const int *multi,
                                     const double *mass,
                                     const int *s2p_map,
                                     const int *p2s_map,
                                     const double nac_factor,
                                     const double *born,
                                     const double *dielectric,
                                     const double *q_direction)
{
  ddm_get_derivative_dynmat_at_q(derivative_dynmat,
                                 num_patom,
                                 num_satom,
                                 fc,
                                 q,
                                 lattice,
                                 r,
                                 multi,
                                 mass,
                                 s2p_map,
                                 p2s_map,
                                 nac_factor,
                                 born,
                                 dielectric,
                                 q_direction);
}


int phpy_compute_permutation(int * rot_atom,
                             PHPYCONST double lat[3][3],
                             PHPYCONST double (*pos)[3],
                             PHPYCONST double (*rot_pos)[3],
                             const int num_pos,
                             const double symprec)
{
  int i,j,k,l;
  int search_start;
  double distance2, diff_cart;
  double diff[3];

  for (i = 0; i < num_pos; i++) {
    rot_atom[i] = -1;
  }

  /* optimization: Iterate primarily by pos instead of rot_pos. */
  /*  (find where 0 belongs in rot_atom, then where 1 belongs, etc.) */
  /*  Then track the first unassigned index. */
  /* */
  /* This works best if the permutation is close to the identity. */
  /* (more specifically, if the max value of 'rot_atom[i] - i' is small) */
  search_start = 0;
  for (i = 0; i < num_pos; i++) {
    while (rot_atom[search_start] >= 0) {
      search_start++;
    }
    for (j = search_start; j < num_pos; j++) {
      if (rot_atom[j] >= 0) {
        continue;
      }

      for (k = 0; k < 3; k++) {
        diff[k] = pos[i][k] - rot_pos[j][k];
        diff[k] -= nint(diff[k]);
      }
      distance2 = 0;
      for (k = 0; k < 3; k++) {
        diff_cart = 0;
        for (l = 0; l < 3; l++) {
          diff_cart += lat[k][l] * diff[l];
        }
        distance2 += diff_cart * diff_cart;
      }

      if (sqrt(distance2) < symprec) {
        rot_atom[j] = i;
        break;
      }
    }
  }

  for (i = 0; i < num_pos; i++) {
    if (rot_atom[i] < 0) {
      return 0;
    }
  }
  return 1;
}


/* Implementation detail of get_smallest_vectors. */
/* Finds the smallest vectors within each list and copies them to the output. */
void phpy_copy_smallest_vectors(double (*shortest_vectors)[27][3],
                                int * multiplicity,
                                PHPYCONST double (*vector_lists)[27][3],
                                PHPYCONST double (*length_lists)[27],
                                const int num_lists,
                                const double symprec)
{
  int i,j,k;
  int count;
  double minimum;
  double (*vectors)[3];
  double *lengths;

  for (i = 0; i < num_lists; i++) {
    /* Look at a single list of 27 vectors. */
    lengths = length_lists[i];
    vectors = vector_lists[i];

    /* Compute the minimum length. */
    minimum = DBL_MAX;
    for (j = 0; j < 27; j++) {
      if (lengths[j] < minimum) {
        minimum = lengths[j];
      }
    }

    /* Copy vectors whose length is within tolerance. */
    count = 0;
    for (j = 0; j < 27; j++) {
      if (lengths[j] - minimum <= symprec) {
        for (k = 0; k < 3; k++) {
          shortest_vectors[i][count][k] = vectors[j][k];
        }
        count++;
      }
    }

    multiplicity[i] = count;
  }
}


void phpy_set_smallest_vectors(double (*smallest_vectors)[27][3],
                               int *multiplicity,
                               PHPYCONST double (*pos_to)[3],
                               const int num_pos_to,
                               PHPYCONST double (*pos_from)[3],
                               const int num_pos_from,
                               PHPYCONST int (*lattice_points)[3],
                               const int num_lattice_points,
                               PHPYCONST double reduced_basis[3][3],
                               PHPYCONST int trans_mat[3][3],
                               const double symprec)
{
  int i, j, k, l, count;
  double length_tmp, minimum, vec_xyz;
  double *length;
  double (*vec)[3];

  length = (double*)malloc(sizeof(double) * num_lattice_points);
  vec = (double(*)[3])malloc(sizeof(double[3]) * num_lattice_points);

  for (i = 0; i < num_pos_to; i++) {
    for (j = 0; j < num_pos_from; j++) {
      for (k = 0; k < num_lattice_points; k++) {
        length[k] = 0;
        for (l = 0; l < 3; l++) {
          vec[k][l] = pos_to[i][l] - pos_from[j][l] + lattice_points[k][l];
        }
        for (l = 0; l < 3; l++) {
          length_tmp = (reduced_basis[l][0] * vec[k][0] +
                        reduced_basis[l][1] * vec[k][1] +
                        reduced_basis[l][2] * vec[k][2]);
          length[k] += length_tmp * length_tmp;
        }
        length[k] = sqrt(length[k]);
      }

      minimum = DBL_MAX;
      for (k = 0; k < num_lattice_points; k++) {
        if (length[k] < minimum) {
          minimum = length[k];
        }
      }

      count = 0;
      for (k = 0; k < num_lattice_points; k++) {
        if (length[k] - minimum < symprec) {
          for (l = 0; l < 3; l++) {
            /* Transform to supercell coordinates */
            vec_xyz = (trans_mat[l][0] * vec[k][0] +
                       trans_mat[l][1] * vec[k][1] +
                       trans_mat[l][2] * vec[k][2]);
            smallest_vectors[i * num_pos_from + j][count][l] = vec_xyz;
          }
          count++;
        }
      }
      if (count > 27) { /* should not be greater than 27 */
        printf("Warning (gsv_set_smallest_vectors): ");
        printf("number of shortest vectors is out of range,\n");
        break;
      } else {
        multiplicity[i * num_pos_from + j] = count;
      }
    }
  }

  free(length);
  length = NULL;
  free(vec);
  vec = NULL;
}


void phpy_perm_trans_symmetrize_fc(double *fc,
                                   const int n_satom,
                                   const int level)
{
  int i, j, k, l, iter;
  double sum;

  for (iter = 0; iter < level; iter++) {
    /* Subtract drift along column */
    for (j = 0; j < n_satom; j++) {
      for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
          sum = 0;
          for (i = 0; i < n_satom; i++) {
            sum += fc[i * n_satom * 9 + j * 9 + k * 3 + l];
          }
          sum /= n_satom;
          for (i = 0; i < n_satom; i++) {
            fc[i * n_satom * 9 + j * 9 + k * 3 + l] -= sum;
          }
        }
      }
    }
    /* Subtract drift along row */
    for (i = 0; i < n_satom; i++) {
      for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
          sum = 0;
          for (j = 0; j < n_satom; j++) {
            sum += fc[i * n_satom * 9 + j * 9 + k * 3 + l];
          }
          sum /= n_satom;
          for (j = 0; j < n_satom; j++) {
            fc[i * n_satom * 9 + j * 9 + k * 3 + l] -= sum;
          }
        }
      }
    }
    set_index_permutation_symmetry_fc(fc, n_satom);
  }
  set_translational_symmetry_fc(fc, n_satom);
}


void phpy_perm_trans_symmetrize_compact_fc(double *fc,
                                           const int p2s[],
                                           const int s2pp[],
                                           const int nsym_list[],
                                           const int perms[],
                                           const int n_satom,
                                           const int n_patom,
                                           const int level)
{
  int i, j, k, l, n, iter;
  double sum;

  for (iter=0; iter < level; iter++) {

    for (n = 0; n < 2; n++) {
      /* transpose only */
      phpy_set_index_permutation_symmetry_compact_fc(fc,
                                                     p2s,
                                                     s2pp,
                                                     nsym_list,
                                                     perms,
                                                     n_satom,
                                                     n_patom,
                                                     1);
      for (i = 0; i < n_patom; i++) {
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            sum = 0;
            for (j = 0; j < n_satom; j++) {
              sum += fc[i * n_satom * 9 + j * 9 + k * 3 + l];
            }
            sum /= n_satom;
            for (j = 0; j < n_satom; j++) {
              fc[i * n_satom * 9 + j * 9 + k * 3 + l] -= sum;
            }
          }
        }
      }
    }

    phpy_set_index_permutation_symmetry_compact_fc(fc,
                                                   p2s,
                                                   s2pp,
                                                   nsym_list,
                                                   perms,
                                                   n_satom,
                                                   n_patom,
                                                   0);
  }

  set_translational_symmetry_compact_fc(fc, p2s, n_satom, n_patom);

}


void phpy_set_index_permutation_symmetry_compact_fc(double * fc,
                                                    const int p2s[],
                                                    const int s2pp[],
                                                    const int nsym_list[],
                                                    const int perms[],
                                                    const int n_satom,
                                                    const int n_patom,
                                                    const int is_transpose)
{
  int i, j, k, l, m, n, i_p, j_p, i_trans;
  double fc_elem;
  char *done;

  done = NULL;
  done = (char*)malloc(sizeof(char) * n_satom * n_patom);
  for (i = 0; i < n_satom * n_patom; i++) {
    done[i] = 0;
  }

  for (j = 0; j < n_satom; j++) {
    j_p = s2pp[j];
    for (i_p = 0; i_p < n_patom; i_p++) {
      i = p2s[i_p];
      if (i == j) { /* diagnoal part */
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            if (l > k) {
              m = i_p * n_satom * 9 + i * 9 + k * 3 + l;
              n = i_p * n_satom * 9 + i * 9 + l * 3 + k;
              if (is_transpose) {
                fc_elem = fc[m];
                fc[m] = fc[n];
                fc[n] = fc_elem;
              } else {
                fc[m] = (fc[m] + fc[n]) / 2;
                fc[n] = fc[m];
              }
            }
          }
        }
      }
      if (!done[i_p * n_satom + j]) {
        /* (j, i) -- nsym_list[j] --> (j', i') */
        /* nsym_list[j] translates j to j' where j' is in */
        /* primitive cell. The same translation sends i to i' */
        /* where i' is not necessarily to be in primitive cell. */
        /* Thus, i' = perms[nsym_list[j] * n_satom + i] */
        i_trans = perms[nsym_list[j] * n_satom + i];
        done[i_p * n_satom + j] = 1;
        done[j_p * n_satom + i_trans] = 1;
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            m = i_p * n_satom * 9 + j * 9 + k * 3 + l;
            n = j_p * n_satom * 9 + i_trans * 9 + l * 3 + k;
            if (is_transpose) {
              fc_elem = fc[m];
              fc[m] = fc[n];
              fc[n] = fc_elem;
            } else {
              fc[m] = (fc[n] + fc[m]) / 2;
              fc[n] = fc[m];
            }
          }
        }
      }
    }
  }

  free(done);
  done = NULL;
}


static void set_index_permutation_symmetry_fc(double * fc,
                                              const int natom)
{
  int i, j, k, l, m, n;

  for (i = 0; i < natom; i++) {
    /* non diagonal part */
    for (j = i + 1; j < natom; j++) {
      for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
          m = i * natom * 9 + j * 9 + k * 3 + l;
          n = j * natom * 9 + i * 9 + l * 3 + k;
          fc[m] += fc[n];
          fc[m] /= 2;
          fc[n] = fc[m];
        }
      }
    }

    /* diagnoal part */
    for (k = 0; k < 2; k++) {
      for (l = k + 1; l < 3; l++) {
        m = i * natom * 9 + i * 9 + k * 3 + l;
        n = i * natom * 9 + i * 9 + l * 3 + k;
        fc[m] += fc[n];
        fc[m] /= 2;
        fc[n] = fc[m];
      }
    }
  }
}


static void set_translational_symmetry_fc(double * fc,
                                          const int natom)
{
  int i, j, k, l, m;
  double sums[3][3];

  for (i = 0; i < natom; i++) {
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        sums[k][l] = 0;
        m = i * natom * 9 + k * 3 + l;
        for (j = 0; j < natom; j++) {
          if (i != j) {
            sums[k][l] += fc[m];
          }
          m += 9;
        }
      }
    }
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        fc[i * natom * 9 + i * 9 + k * 3 + l] = -(sums[k][l] + sums[l][k]) / 2;
      }
    }
  }
}


static void set_translational_symmetry_compact_fc(double * fc,
                                                  const int p2s[],
                                                  const int n_satom,
                                                  const int n_patom)
{
  int j, k, l, m, i_p;
  double sums[3][3];

  for (i_p = 0; i_p < n_patom; i_p++) {
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        sums[k][l] = 0;
        m = i_p * n_satom * 9 + k * 3 + l;
        for (j = 0; j < n_satom; j++) {
          if (p2s[i_p] != j) {
            sums[k][l] += fc[m];
          }
          m += 9;
        }
      }
    }
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        fc[i_p * n_satom * 9 + p2s[i_p] * 9 + k * 3 + l] =
          -(sums[k][l] + sums[l][k]) / 2;
      }
    }
  }
}


static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}
