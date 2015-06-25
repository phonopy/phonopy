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

#include <stdlib.h>
#include <stdio.h>
#include "cell.h"
#include "mathfunc.h"

#include "debug.h"

/* NULL is returned if faied */
Cell * cel_alloc_cell(const int size)
{
  Cell *cell;
  int i, j;
  
  cell = NULL;

  if ((cell = (Cell*) malloc(sizeof(Cell))) == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    return NULL;
  }

  for ( i = 0; i < 3; i++ ) {
    for ( j = 0; j < 3; j++ ) {
      cell->lattice[i][j] = 0;
    }
  }
  cell->size = size;
  
  if (size > 0) {
    if ((cell->types = (int *) malloc(sizeof(int) * size)) == NULL) {
      warning_print("spglib: Memory could not be allocated.");
      free(cell);
      cell = NULL;
      return NULL;
    }
    if ((cell->position =
	 (double (*)[3]) malloc(sizeof(double[3]) * size)) == NULL) {
      warning_print("spglib: Memory could not be allocated.");
      free(cell->types);
      cell->types = NULL;
      free(cell);
      cell = NULL;
      return NULL;
    }
  }

  return cell;
}

void cel_free_cell(Cell * cell)
{
  if (cell->size > 0) {
    free(cell->position);
    cell->position = NULL;
    free(cell->types);
    cell->types = NULL;
  }
  free (cell);
  cell = NULL;
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
      cell->position[i][j] = position[i][j];
    }
    cell->types[i] = types[i];
  }
}

Cell * cel_copy_cell(SPGCONST Cell * cell)
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
  if ( mat_norm_squared_d3(v_diff) < symprec * symprec) {
    return 1;
  } else {
    return 0;
  }
}

