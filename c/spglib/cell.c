/* Copyright (C) 2008 Atsushi Togo */
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

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "cell.h"
#include "mathfunc.h"

#include "debug.h"

#define INCREASE_RATE 2.0
#define REDUCE_RATE 0.95
#define NUM_ATTEMPT 100

static Cell * trim_cell(int * mapping_table,
                        SPGCONST double trimmed_lattice[3][3],
                        const Cell * cell,
                        const double symprec);
static void set_positions(Cell * trim_cell,
                          const VecDBL * position,
                          const int * mapping_table,
                          const int * overlap_table);
static VecDBL *
translate_atoms_in_trimmed_lattice(const Cell * cell,
                                   SPGCONST double prim_lat[3][3]);
static int * get_overlap_table(const VecDBL * position,
                               const int cell_size,
                               const int * cell_types,
                               const Cell * trimmed_cell,
                               const double symprec);

/* NULL is returned if faied */
Cell * cel_alloc_cell(const int size)
{
  Cell *cell;

  cell = NULL;

  if (size < 1) {
    return NULL;
  }

  cell = NULL;

  if ((cell = (Cell*) malloc(sizeof(Cell))) == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    return NULL;
  }

  if ((cell->lattice = (double (*)[3]) malloc(sizeof(double[3]) * 3)) == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    free(cell);
    cell = NULL;
    return NULL;
  }

  cell->size = size;

  if ((cell->types = (int *) malloc(sizeof(int) * size)) == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    free(cell->lattice);
    cell->lattice = NULL;
    free(cell);
    cell = NULL;
    return NULL;
  }
  if ((cell->position =
       (double (*)[3]) malloc(sizeof(double[3]) * size)) == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    free(cell->types);
    cell->types = NULL;
    free(cell->lattice);
    cell->lattice = NULL;
    free(cell);
    cell = NULL;
    return NULL;
  }

  return cell;
}

void cel_free_cell(Cell * cell)
{
  if (cell != NULL) {
    if (cell->lattice != NULL) {
      free(cell->lattice);
      cell->lattice = NULL;
    }
    if (cell->position != NULL) {
      free(cell->position);
      cell->position = NULL;
    }
    if (cell->types != NULL) {
      free(cell->types);
      cell->types = NULL;
    }
    free(cell);
  }
}

void cel_set_cell(Cell * cell,
                  SPGCONST double lattice[3][3],
                  SPGCONST double position[][3],
                  const int types[])
{
  int i, j;
  mat_copy_matrix_d3(cell->lattice, lattice);
  for (i = 0; i < cell->size; i++) {
    for (j = 0; j < 3; j++) {
      cell->position[i][j] = position[i][j] - mat_Nint(position[i][j]);
    }
    cell->types[i] = types[i];
  }
}

Cell * cel_copy_cell(const Cell * cell)
{
  Cell * cell_new;

  cell_new = NULL;

  if ((cell_new = cel_alloc_cell(cell->size)) == NULL) {
    return NULL;
  }

  cel_set_cell(cell_new,
               cell->lattice,
               cell->position,
               cell->types);

  return cell_new;
}

int cel_is_overlap(const double a[3],
                   const double b[3],
                   SPGCONST double lattice[3][3],
                   const double symprec)
{
  int i;
  double v_diff[3];

  for (i = 0; i < 3; i++) {
    v_diff[i] = a[i] - b[i];
    v_diff[i] -= mat_Nint(v_diff[i]);
  }

  mat_multiply_matrix_vector_d3(v_diff, lattice, v_diff);
  if (sqrt(mat_norm_squared_d3(v_diff)) < symprec) {
    return 1;
  } else {
    return 0;
  }
}

int cel_is_overlap_with_same_type(const double a[3],
                                  const double b[3],
                                  const int type_a,
                                  const int type_b,
                                  SPGCONST double lattice[3][3],
                                  const double symprec)
{
  if (type_a == type_b) {
    return cel_is_overlap(a, b, lattice, symprec);
  } else {
    return 0;
  }
}

/* 1: At least one overlap of a pair of atoms was found. */
/* 0: No overlap of atoms was found. */
int cel_any_overlap(const Cell * cell,
                    const double symprec) {
  int i, j;

  for (i = 0; i < cell->size; i++) {
    for (j = i + 1; j < cell->size; j++) {
      if (cel_is_overlap(cell->position[i],
                         cell->position[j],
                         cell->lattice,
                         symprec)) {
        return 1;
      }
    }
  }
  return 0;
}

/* 1: At least one overlap of a pair of atoms with same type was found. */
/* 0: No overlap of atoms was found. */
int cel_any_overlap_with_same_type(const Cell * cell,
                                   const double symprec) {
  int i, j;

  for (i = 0; i < cell->size; i++) {
    for (j = i + 1; j < cell->size; j++) {
      if (cel_is_overlap_with_same_type(cell->position[i],
                                        cell->position[j],
                                        cell->types[i],
                                        cell->types[j],
                                        cell->lattice,
                                        symprec)) {
        return 1;
      }
    }
  }
  return 0;
}

Cell * cel_trim_cell(int * mapping_table,
                     SPGCONST double trimmed_lattice[3][3],
                     const Cell * cell,
                     const double symprec)
{
  return trim_cell(mapping_table,
                   trimmed_lattice,
                   cell,
                   symprec);
}


/* Return NULL if failed */
static Cell * trim_cell(int * mapping_table,
                        SPGCONST double trimmed_lattice[3][3],
                        const Cell * cell,
                        const double symprec)
{
  int i, index_atom, ratio;
  Cell *trimmed_cell;
  VecDBL * position;
  int *overlap_table;
  int tmp_mat_int[3][3];
  double tmp_mat[3][3];

  debug_print("trim_cell\n");

  position = NULL;
  overlap_table = NULL;
  trimmed_cell = NULL;

  ratio = abs(mat_Nint(mat_get_determinant_d3(cell->lattice) /
                       mat_get_determinant_d3(trimmed_lattice)));

  mat_inverse_matrix_d3(tmp_mat, trimmed_lattice, symprec);
  mat_multiply_matrix_d3(tmp_mat, tmp_mat, cell->lattice);
  mat_cast_matrix_3d_to_3i(tmp_mat_int, tmp_mat);
  if (abs(mat_get_determinant_i3(tmp_mat_int)) != ratio) {
    warning_print("spglib: Determinant of change of basis matrix "
                  "has to be same as volume ratio (line %d, %s).\n",
                  __LINE__, __FILE__);
    goto err;
  }

  /* Check if cell->size is dividable by ratio */
  if ((cell->size / ratio) * ratio != cell->size) {
    warning_print("spglib: atom number ratio is inconsistent.\n");
    warning_print(" (line %d, %s).\n", __LINE__, __FILE__);
    goto err;
  }

  if ((trimmed_cell = cel_alloc_cell(cell->size / ratio)) == NULL) {
    goto err;
  }

  if ((position = translate_atoms_in_trimmed_lattice(cell,
                                                     trimmed_lattice))
      == NULL) {
    warning_print("spglib: translate_atoms_in_trimmed_lattice failed.\n");
    warning_print(" (line %d, %s).\n", __LINE__, __FILE__);
    cel_free_cell(trimmed_cell);
    trimmed_cell = NULL;
    goto err;
  }

  mat_copy_matrix_d3(trimmed_cell->lattice, trimmed_lattice);

  if ((overlap_table = get_overlap_table(position,
                                         cell->size,
                                         cell->types,
                                         trimmed_cell,
                                         symprec)) == NULL) {
    warning_print("spglib: get_overlap_table failed.\n");
    warning_print(" (line %d, %s).\n", __LINE__, __FILE__);
    mat_free_VecDBL(position);
    position = NULL;
    cel_free_cell(trimmed_cell);
    trimmed_cell = NULL;
    goto err;
  }

  index_atom = 0;
  for (i = 0; i < cell->size; i++) {
    if (overlap_table[i] == i) {
      mapping_table[i] = index_atom;
      trimmed_cell->types[index_atom] = cell->types[i];
      index_atom++;
    } else {
      mapping_table[i] = mapping_table[overlap_table[i]];
    }
  }

  set_positions(trimmed_cell,
                position,
                mapping_table,
                overlap_table);

  mat_free_VecDBL(position);
  position = NULL;
  free(overlap_table);

  return trimmed_cell;

 err:
  return NULL;
}

static void set_positions(Cell * trimmed_cell,
                          const VecDBL * position,
                          const int * mapping_table,
                          const int * overlap_table)
{
  int i, j, k, l, multi;

  for (i = 0; i < trimmed_cell->size; i++) {
    for (j = 0; j < 3; j++) {
      trimmed_cell->position[i][j] = 0;
    }
  }

  /* Positions of overlapped atoms are averaged. */
  for (i = 0; i < position->size; i++) {
    j = mapping_table[i];
    k = overlap_table[i];
    for (l = 0; l < 3; l++) {
      /* boundary treatment */
      /* One is at right and one is at left or vice versa. */
      if (mat_Dabs(position->vec[k][l] - position->vec[i][l]) > 0.5) {
        if (position->vec[i][l] < position->vec[k][l]) {
          trimmed_cell->position[j][l] += position->vec[i][l] + 1;
        } else {
          trimmed_cell->position[j][l] += position->vec[i][l] - 1;
        }
      } else {
        trimmed_cell->position[j][l] += position->vec[i][l];
      }
    }

  }

  multi = position->size / trimmed_cell->size;
  for (i = 0; i < trimmed_cell->size; i++) {
    for (j = 0; j < 3; j++) {
      trimmed_cell->position[i][j] /= multi;
      trimmed_cell->position[i][j] = mat_Dmod1(trimmed_cell->position[i][j]);
    }
  }
}

/* Return NULL if failed */
static VecDBL *
translate_atoms_in_trimmed_lattice(const Cell * cell,
                                   SPGCONST double trimmed_lattice[3][3])
{
  int i, j;
  double tmp_matrix[3][3], axis_inv[3][3];
  VecDBL * position;

  position = NULL;

  if ((position = mat_alloc_VecDBL(cell->size)) == NULL) {
    return NULL;
  }

  mat_inverse_matrix_d3(tmp_matrix, trimmed_lattice, 0);
  mat_multiply_matrix_d3(axis_inv, tmp_matrix, cell->lattice);

  /* Send atoms into the trimmed cell */
  for (i = 0; i < cell->size; i++) {
    mat_multiply_matrix_vector_d3(position->vec[i],
                                  axis_inv,
                                  cell->position[i]);
    for (j = 0; j < 3; j++) {
      position->vec[i][j] = mat_Dmod1(position->vec[i][j]);
    }
  }

  return position;
}


/* Return NULL if failed */
static int * get_overlap_table(const VecDBL * position,
                               const int cell_size,
                               const int * cell_types,
                               const Cell * trimmed_cell,
                               const double symprec)
{
  int i, j, attempt, num_overlap, ratio;
  double trim_tolerance;
  int *overlap_table;

  trim_tolerance = symprec;

  ratio = cell_size / trimmed_cell->size;

  if ((overlap_table = (int*)malloc(sizeof(int) * cell_size)) == NULL) {
    return NULL;
  }

  for (attempt = 0; attempt < NUM_ATTEMPT; attempt++) {
    for (i = 0; i < cell_size; i++) {
      overlap_table[i] = i;
      for (j = 0; j < cell_size; j++) {
        if (cell_types[i] == cell_types[j]) {
          if (cel_is_overlap(position->vec[i],
                             position->vec[j],
                             trimmed_cell->lattice,
                             trim_tolerance)) {
            if (overlap_table[j] == j) {
              overlap_table[i] = j;
              break;
            }
          }
        }
      }
    }

    for (i = 0; i < cell_size; i++) {
      if (overlap_table[i] != i) {
        continue;
      }

      num_overlap = 0;
      for (j = 0; j < cell_size; j++) {
        if (i == overlap_table[j]) {
          num_overlap++;
        }
      }

      if (num_overlap == ratio) {
        continue;
      }

      if (num_overlap < ratio) {
        trim_tolerance *= INCREASE_RATE;
        warning_print("spglib: Increase tolerance to %f ", trim_tolerance);
        warning_print("(line %d, %s).\n", __LINE__, __FILE__);
        goto cont;
      }
      if (num_overlap > ratio) {
        trim_tolerance *= REDUCE_RATE;
        warning_print("spglib: Reduce tolerance to %f ", trim_tolerance);
        warning_print("(%d) ", attempt);
        warning_print("(line %d, %s).\n", __LINE__, __FILE__);
        goto cont;
      }
    }

    goto found;

  cont:
    ;
  }

  warning_print("spglib: Could not trim cell well ");
  warning_print("(line %d, %s).\n", __LINE__, __FILE__);
  free(overlap_table);
  overlap_table = NULL;

found:
  return overlap_table;
}
