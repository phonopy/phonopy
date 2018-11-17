/* Copyright (C) 2011 Atsushi Togo */
/* All rights reserved. */

/* This file is part of spglib. */

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
#include "cell.h"
#include "mathfunc.h"
#include "symmetry.h"
#include "site_symmetry.h"
#include "sitesym_database.h"

#include "debug.h"

#define INCREASE_RATE 1.05
#define NUM_ATTEMPT 5

static VecDBL * get_exact_positions(int * equiv_atoms,
                                    const Cell * conv_prim,
                                    const Symmetry * conv_sym,
                                    const double symprec);
static void set_exact_location(double position[3],
                               const Symmetry * conv_sym,
                               SPGCONST double bravais_lattice[3][3],
                               const double symprec);
static int set_equivalent_atom(VecDBL *positions,
                               int * equiv_atoms,
                               const int i,
                               const int num_indep_atoms,
                               const int *indep_atoms,
                               const Cell * conv_prim,
                               const Symmetry * conv_sym,
                               const double symprec);
static int set_Wyckoffs_labels(int * wyckoffs,
                               const VecDBL *positions,
                               const int * equiv_atoms,
                               const Cell * conv_prim,
                               const Symmetry * conv_sym,
                               const int hall_number,
                               const double symprec);
static int get_Wyckoff_notation(const double position[3],
                                const Symmetry * conv_sym,
                                SPGCONST double bravais_lattice[3][3],
                                const int hall_number,
                                const double symprec);

/* Return NULL if failed */
VecDBL * ssm_get_exact_positions(int *wyckoffs,
                                 int *equiv_atoms,
                                 const Cell * conv_prim,
                                 const Symmetry * conv_sym,
                                 const int hall_number,
                                 const double symprec)
{
  int i;
  double tolerance;
  VecDBL *positions;

  positions = NULL;

  /* Attempt slightly loosen tolerance */
  tolerance = symprec;
  for (i = 0; i < NUM_ATTEMPT; i++) {
    if ((positions = get_exact_positions(equiv_atoms,
                                         conv_prim,
                                         conv_sym,
                                         tolerance)) == NULL) {
      return NULL;
    }

    if (set_Wyckoffs_labels(wyckoffs,
                            positions,
                            equiv_atoms,
                            conv_prim,
                            conv_sym,
                            hall_number,
                            symprec)) {
      break;
    } else {
      mat_free_VecDBL(positions);
      positions = NULL;
      tolerance *= INCREASE_RATE;
    }
  }

  return positions;
}

/* Return 0: succeeded, 1: tolerance too large, -1: tolerance too short */
static VecDBL * get_exact_positions(int * equiv_atoms,
                                    const Cell * conv_prim,
                                    const Symmetry * conv_sym,
                                    const double symprec)
{
  int i, num_indep_atoms;
  int *indep_atoms;
  VecDBL *positions;

  debug_print("get_exact_positions\n");

  positions = NULL;
  indep_atoms = NULL;

  if ((positions = mat_alloc_VecDBL(conv_prim->size)) == NULL) {
    return NULL;
  }

  if ((indep_atoms = (int*) malloc(sizeof(int) * conv_prim->size)) == NULL) {
    warning_print("spglib: Memory could not be allocated ");
    mat_free_VecDBL(positions);
    return NULL;
  }

  num_indep_atoms = 0;
  for (i = 0; i < conv_prim->size; i++) {
    /* Check if atom_i overlap to an atom already set at the exact position. */
    if (! set_equivalent_atom(positions,
                              equiv_atoms,
                              i,
                              num_indep_atoms,
                              indep_atoms,
                              conv_prim,
                              conv_sym,
                              symprec)) {
      /* No equivalent atom was found. */
      equiv_atoms[i] = i;
      indep_atoms[num_indep_atoms] = i;
      mat_copy_vector_d3(positions->vec[i], conv_prim->position[i]);
      set_exact_location(positions->vec[i],
                         conv_sym,
                         conv_prim->lattice,
                         symprec);
      num_indep_atoms++;
    }
  }

  free(indep_atoms);
  indep_atoms = NULL;

  return positions;
}

static int set_equivalent_atom(VecDBL *positions,
                               int * equiv_atoms,
                               const int i,
                               const int num_indep_atoms,
                               const int *indep_atoms,
                               const Cell * conv_prim,
                               const Symmetry * conv_sym,
                               const double symprec)
{
  int j, k, l;
  double pos[3];

  for (j = 0; j < num_indep_atoms; j++) {
    for (k = 0; k < conv_sym->size; k++) {
      mat_multiply_matrix_vector_id3(pos,
                                     conv_sym->rot[k],
                                     positions->vec[indep_atoms[j]]);
      for (l = 0; l < 3; l++) {
        pos[l] += conv_sym->trans[k][l];
      }
      if (cel_is_overlap_with_same_type(pos,
                                        conv_prim->position[i],
                                        conv_prim->types[indep_atoms[j]],
                                        conv_prim->types[i],
                                        conv_prim->lattice,
                                        symprec)) {
        for (l = 0; l < 3; l++) {
          positions->vec[i][l] = mat_Dmod1(pos[l]);
        }
        equiv_atoms[i] = indep_atoms[j];
        return 1;
      }
    }
  }

  return 0;
}

/* Site-symmetry is used to determine exact location of an atom */
/* R. W. Grosse-Kunstleve and P. D. Adams */
/* Acta Cryst. (2002). A58, 60-65 */
static void set_exact_location(double position[3],
                               const Symmetry * conv_sym,
                               SPGCONST double bravais_lattice[3][3],
                               const double symprec)
{
  int i, j, k, num_site_sym;
  double sum_rot[3][3];
  double pos[3], sum_trans[3];

  debug_print("set_exact_location\n");

  num_site_sym = 0;
  for (i = 0; i < 3; i++) {
    sum_trans[i] = 0.0;
    for (j = 0; j < 3; j++) {
      sum_rot[i][j] = 0;
    }
  }

  for (i = 0; i < conv_sym->size; i++) {
    mat_multiply_matrix_vector_id3(pos,
                                   conv_sym->rot[i],
                                   position);
    for (j = 0; j < 3; j++) {
      pos[j] += conv_sym->trans[i][j];
    }

    if (cel_is_overlap(pos,
                       position,
                       bravais_lattice,
                       symprec)) {
      for (j = 0; j < 3; j++) {
        for (k = 0; k < 3; k++) {
          sum_rot[j][k] += conv_sym->rot[i][j][k];
        }
        sum_trans[j] +=
          conv_sym->trans[i][j] - mat_Nint(pos[j] - position[j]);
      }
      num_site_sym++;
    }
  }

  for (i = 0; i < 3; i++) {
    sum_trans[i] /= num_site_sym;
    for (j = 0; j < 3; j++) {
      sum_rot[i][j] /= num_site_sym;
    }
  }

  /* (sum_rot|sum_trans) is the special-position operator. */
  /* Elements of sum_rot can be fractional values. */
  mat_multiply_matrix_vector_d3(position,
                                sum_rot,
                                position);

  for (i = 0; i < 3; i++) {
    position[i] += sum_trans[i];
  }
}

static int set_Wyckoffs_labels(int *wyckoffs,
                               const VecDBL *positions,
                               const int * equiv_atoms,
                               const Cell * conv_prim,
                               const Symmetry * conv_sym,
                               const int hall_number,
                               const double symprec)
{
  int i, w;

  for (i = 0; i < conv_prim->size; i++) {
    if (i == equiv_atoms[i]) {
      w = get_Wyckoff_notation(positions->vec[i],
                               conv_sym,
                               conv_prim->lattice,
                               hall_number,
                               symprec);
      if (w < 0) {
        goto err;
      } else {
        wyckoffs[i] = w;
      }
    }
  }

  for (i = 0; i < conv_prim->size; i++) {
    if (i != equiv_atoms[i]) {
      wyckoffs[i] = wyckoffs[equiv_atoms[i]];
    }
  }

  return 1;

 err:
  return 0;
}

/* Return -1 if failed */
static int get_Wyckoff_notation(const double position[3],
                                const Symmetry * conv_sym,
                                SPGCONST double bravais_lattice[3][3],
                                const int hall_number,
                                const double symprec)
{
  int i, j, k, l, num_sitesym, multiplicity, wyckoff_letter;
  int indices_wyc[2];
  int rot[3][3];
  double trans[3], orbit[3];
  VecDBL *pos_rot;

  debug_print("get_Wyckoff_notation\n");

  wyckoff_letter = -1;
  pos_rot = NULL;

  if ((pos_rot = mat_alloc_VecDBL(conv_sym->size)) == NULL) {
    return -1;
  }

  for (i = 0; i < conv_sym->size; i++) {
    mat_multiply_matrix_vector_id3(pos_rot->vec[i], conv_sym->rot[i], position);
    for (j = 0; j < 3; j++) {
      pos_rot->vec[i][j] += conv_sym->trans[i][j];
    }
  }

  ssmdb_get_wyckoff_indices(indices_wyc, hall_number);
  for (i = 0; i < indices_wyc[1]; i++) {
    /* (rot, trans) gives the first element of each Wyckoff position */
    /* of the 'Coordinates' in ITA */
    /* Example: (x,1/4,1/2)      */
    /* rot         trans         */
    /* [1, 0, 0]   [0, 1/4, 1/2] */
    /* [0, 0, 0]                 */
    /* [0, 0, 0]                 */
    multiplicity = ssmdb_get_coordinate(rot, trans, i + indices_wyc[0]);
    /* Effectively this iteration runs over all 'Coordinates' of each */
    /* Wyckoff position, i.e., works as looking for the first element. */
    for (j = 0; j < pos_rot->size; j++) {
      num_sitesym = 0;
      for (k = 0; k < pos_rot->size; k++) {
        if (cel_is_overlap(pos_rot->vec[j],
                           pos_rot->vec[k],
                           bravais_lattice,
                           symprec)) {
          mat_multiply_matrix_vector_id3(orbit, rot, pos_rot->vec[k]);
          for (l = 0; l < 3; l++) {
            orbit[l] += trans[l];
          }
          if (cel_is_overlap(pos_rot->vec[k],
                             orbit,
                             bravais_lattice,
                             symprec)) {
            num_sitesym++;
          }
        }
      }
      if (num_sitesym * multiplicity == conv_sym->size) {
        /* Database is made reversed order, e.g., gfedcba. */
        /* wyckoff is set 0 1 2 3 4... for a b c d e..., respectively. */
        wyckoff_letter = indices_wyc[1] - i - 1;
        goto end;
      }
    }
  }

 end:
  mat_free_VecDBL(pos_rot);
  pos_rot = NULL;
  return wyckoff_letter;
}
