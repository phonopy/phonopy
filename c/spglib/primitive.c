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

#include <stdio.h>
#include <stdlib.h>
#include "cell.h"
#include "lattice.h"
#include "mathfunc.h"
#include "primitive.h"
#include "symmetry.h"

#include "debug.h"

#include <assert.h>

#define INCREASE_RATE 2.0
#define REDUCE_RATE 0.95

static double A_mat[3][3] = {{    1,    0,    0},
			     {    0, 1./2,-1./2},
			     {    0, 1./2, 1./2}};
static double C_mat[3][3] = {{ 1./2, 1./2,    0},
			     {-1./2, 1./2,    0},
			     {    0,    0,    1}};
static double R_mat[3][3] = {{ 2./3,-1./3,-1./3 },
			     { 1./3, 1./3,-2./3 },
			     { 1./3, 1./3, 1./3 }};
static double I_mat[3][3] = {{-1./2, 1./2, 1./2 },
			     { 1./2,-1./2, 1./2 },
			     { 1./2, 1./2,-1./2 }};
static double F_mat[3][3] = {{    0, 1./2, 1./2 },
			     { 1./2,    0, 1./2 },
			     { 1./2, 1./2,    0 }};

static Primitive * get_primitive(SPGCONST Cell * cell, const double symprec);
static void set_primitive_positions(Cell * primitive_cell,
				    const VecDBL * position,
				    const Cell * cell,
				    const int * mapping_table,
				    const int * overlap_table);
static VecDBL *
translate_atoms_in_primitive_lattice(SPGCONST Cell * cell,
				     SPGCONST double prim_lat[3][3]);
static int * get_overlap_table(SPGCONST Cell *primitive_cell,
			       const VecDBL * position,
			       const int *types,
			       const double symprec);
static Cell * get_cell_with_smallest_lattice(SPGCONST Cell * cell,
					     const double symprec);
static Cell * get_primitive_cell(int * mapping_table,
				 SPGCONST Cell * cell,
				 const VecDBL * pure_trans,
				 const double symprec);
static int trim_cell(Cell * primitive_cell,
		     int * mapping_table,
		     SPGCONST Cell * cell,
		     const double symprec);
static int get_primitive_lattice_vectors_iterative(double prim_lattice[3][3],
						   SPGCONST Cell * cell,
						   const VecDBL * pure_trans,
						   const double symprec);
static int get_primitive_lattice_vectors(double prim_lattice[3][3],
					 const VecDBL * vectors,
					 SPGCONST Cell * cell,
					 const double symprec);
static VecDBL * get_translation_candidates(const VecDBL * pure_trans);

/* return NULL if failed */
Primitive * prm_alloc_primitive(const int size)
{
  Primitive *primitive;

  primitive = NULL;

  if ((primitive = (Primitive*) malloc(sizeof(Primitive))) == NULL) {
    warning_print("spglib: Memory could not be allocated ");
    return NULL;
  }

  primitive->cell = NULL;
  primitive->mapping_table = NULL;
  primitive->size = size;
  primitive->tolerance = 0;

  if (size > 0) {
    if ((primitive->mapping_table = (int*) malloc(sizeof(int) * size)) == NULL) {
      warning_print("spglib: Memory could not be allocated ");
      warning_print("(Primitive, line %d, %s).\n", __LINE__, __FILE__);
      free(primitive);
      primitive = NULL;
      return NULL;
    }
  }

  return primitive;
}

void prm_free_primitive(Primitive * primitive)
{
  if (primitive != NULL) {
    if (primitive->mapping_table != NULL) {
      free(primitive->mapping_table);
      primitive->mapping_table = NULL;
    }

    if (primitive->cell != NULL) {
      cel_free_cell(primitive->cell);
    }
    free(primitive);
    primitive = NULL;
  }
}

/* Return NULL if failed */
Primitive * prm_get_primitive(SPGCONST Cell * cell, const double symprec)
{
  return get_primitive(cell, symprec);
}

/* Return NULL if failed */
Primitive * prm_transform_to_primitive(SPGCONST Cell * cell,
				       SPGCONST double trans_mat_Bravais[3][3],
				       const Centering centering,
				       const double symprec)
{
  int multi;
  double tmat[3][3];
  Primitive * primitive;

  if ((primitive = prm_alloc_primitive(cell->size)) == NULL) {
    return NULL;
  }

  switch (centering) {
  case NO_CENTER:
    mat_copy_matrix_d3(tmat, trans_mat_Bravais);
    break;
  case A_FACE:
    mat_multiply_matrix_d3(tmat, trans_mat_Bravais, A_mat);
    break;
  case C_FACE:
    mat_multiply_matrix_d3(tmat, trans_mat_Bravais, C_mat);
    break;
  case FACE:
    mat_multiply_matrix_d3(tmat, trans_mat_Bravais, F_mat);
    break;
  case BODY:
    mat_multiply_matrix_d3(tmat, trans_mat_Bravais, I_mat);
    break;
  case R_CENTER:
    mat_multiply_matrix_d3(tmat, trans_mat_Bravais, R_mat);
    break;
  default:
    goto err;
  }

  multi = mat_Nint(1.0 / mat_get_determinant_d3(tmat));
  if ((primitive->cell = cel_alloc_cell(cell->size / multi)) == NULL) {
    goto err;
  }

  mat_multiply_matrix_d3(primitive->cell->lattice,
			 cell->lattice,
			 tmat);

  if (trim_cell(primitive->cell,
		primitive->mapping_table,
		cell,
		symprec)) {
    return primitive;
  }

 err:
  prm_free_primitive(primitive);
  primitive = NULL;
  return NULL;
}

/* Return NULL if failed */
static Primitive * get_primitive(SPGCONST Cell * cell, const double symprec)
{
  int i, attempt;
  double tolerance;
  Primitive *primitive;
  VecDBL * pure_trans;

  debug_print("get_primitive (tolerance = %f):\n", symprec);

  primitive = NULL;
  pure_trans = NULL;

  if ((primitive = prm_alloc_primitive(cell->size)) == NULL) {
    return NULL;
  }

  tolerance = symprec;
  for (attempt = 0; attempt < 100; attempt++) {
    if ((pure_trans = sym_get_pure_translation(cell, tolerance)) == NULL) {
      goto cont;
    }

    if (pure_trans->size == 1) {
      if ((primitive->cell = get_cell_with_smallest_lattice(cell, tolerance))
	  != NULL) {
	for (i = 0; i < cell->size; i++) {
	  primitive->mapping_table[i] = i;
	}
	goto found;
      }
    } else {
      if ((primitive->cell = get_primitive_cell(primitive->mapping_table,
						cell,
						pure_trans,
						tolerance)) != NULL) {
	goto found;
      }
    }

    mat_free_VecDBL(pure_trans);

  cont:
    tolerance *= REDUCE_RATE;
    warning_print("spglib: Reduce tolerance to %f ", tolerance);
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);
  }

  prm_free_primitive(primitive);
  return NULL;

 found:
  primitive->tolerance = tolerance;
  mat_free_VecDBL(pure_trans);
  return primitive;
}

/* Return NULL if failed */
static Cell * get_cell_with_smallest_lattice(SPGCONST Cell * cell,
					     const double symprec)
{
  int i, j;
  double min_lat[3][3], trans_mat[3][3], inv_lat[3][3];
  Cell * smallest_cell;

  debug_print("get_cell_with_smallest_lattice:\n");
  
  smallest_cell = NULL;

  if (!lat_smallest_lattice_vector(min_lat,
				   cell->lattice,
				   symprec)) {
    goto err;
  }

  mat_inverse_matrix_d3(inv_lat, min_lat, 0);
  mat_multiply_matrix_d3(trans_mat, inv_lat, cell->lattice);

  if ((smallest_cell = cel_alloc_cell(cell->size)) == NULL) {
    goto err;
  }

  mat_copy_matrix_d3(smallest_cell->lattice, min_lat);
  for (i = 0; i < cell->size; i++) {
    smallest_cell->types[i] = cell->types[i];
    mat_multiply_matrix_vector_d3(smallest_cell->position[i],
				  trans_mat, cell->position[i]);
    for (j = 0; j < 3; j++) {
      cell->position[i][j] -= mat_Nint(cell->position[i][j]);
    }
  }

  return smallest_cell;

 err:
  return NULL;
}

/* Return NULL if failed */
static Cell * get_primitive_cell(int * mapping_table,
				 SPGCONST Cell * cell,
				 const VecDBL * pure_trans,
				 const double symprec)
{
  int multi;
  double prim_lattice[3][3];
  Cell * primitive_cell;

  debug_print("get_primitive:\n");

  primitive_cell = NULL;

  /* Primitive lattice vectors are searched. */
  /* To be consistent, sometimes tolerance is decreased iteratively. */
  /* The descreased tolerance is stored in 'static double tolerance'. */
  multi = get_primitive_lattice_vectors_iterative(prim_lattice,
						  cell,
						  pure_trans,
						  symprec);
  if (! multi) {
    goto not_found;
  }

  if ((primitive_cell = cel_alloc_cell(cell->size / multi)) == NULL) {
    goto not_found;
  }

  if (! lat_smallest_lattice_vector(primitive_cell->lattice,
				    prim_lattice,
				    symprec)) {
    cel_free_cell(primitive_cell);
    primitive_cell = NULL;
    goto not_found;
  }

  /* Fit atoms into new primitive cell */
  if (! trim_cell(primitive_cell, mapping_table, cell, symprec)) {
    cel_free_cell(primitive_cell);
    primitive_cell = NULL;
    goto not_found;
  }

  /* found */
  return primitive_cell;

 not_found:
  warning_print("spglib: Primitive cell could not be found ");
  warning_print("(line %d, %s).\n", __LINE__, __FILE__);
  return NULL;
}

/* Return 0 if failed */
static int trim_cell(Cell * primitive_cell,
		     int * mapping_table,
		     SPGCONST Cell * cell,
		     const double symprec)
{
  int i, index_prim_atom;
  VecDBL * position;
  int *overlap_table;

  position = NULL;
  overlap_table = NULL;

  if ((position =
       translate_atoms_in_primitive_lattice(cell, primitive_cell->lattice))
      == NULL) {
    goto err;
  }

  if ((overlap_table = get_overlap_table(primitive_cell,
					 position,
					 cell->types,
					 symprec)) == NULL) {
    mat_free_VecDBL(position);
    goto err;
  }


  index_prim_atom = 0;
  for (i = 0; i < cell->size; i++) {
    if (overlap_table[i] == i) {
      mapping_table[i] = index_prim_atom;
      primitive_cell->types[index_prim_atom] = cell->types[i];
      index_prim_atom++;
    } else {
      mapping_table[i] = mapping_table[overlap_table[i]];
    }
  }

  set_primitive_positions(primitive_cell,
			  position,
			  cell,
			  mapping_table,
			  overlap_table);

  mat_free_VecDBL(position);
  /* free_overlap_table(overlap_table, cell->size); */
  free(overlap_table);
  return 1;

 err:
  return 0;
}

static void set_primitive_positions(Cell * primitive_cell,
				    const VecDBL * position,
				    const Cell * cell,
				    const int * mapping_table,
				    const int * overlap_table)
{
  int i, j, k, l, multi;

  for (i = 0; i < primitive_cell->size; i++) {
    for (j = 0; j < 3; j++) {
      primitive_cell->position[i][j] = 0;
    }
  }

  /* Positions of overlapped atoms are averaged. */
  for (i = 0; i < cell->size; i++) {
    j = mapping_table[i];
    k = overlap_table[i];
    for (l = 0; l < 3; l++) {
      /* boundary treatment */
      /* One is at right and one is at left or vice versa. */
      if (mat_Dabs(position->vec[k][l] - position->vec[i][l]) > 0.5) {
	if (position->vec[i][l] < 0) {
	  primitive_cell->position[j][l] += position->vec[i][l] + 1;
	} else {
	  primitive_cell->position[j][l] += position->vec[i][l] - 1;
	}
      } else {
	primitive_cell->position[j][l] += position->vec[i][l];
      }
    }
	
  }

  multi = cell->size / primitive_cell->size;
  for (i = 0; i < primitive_cell->size; i++) {
    for (j = 0; j < 3; j++) {
      primitive_cell->position[i][j] /= multi;
      primitive_cell->position[i][j] -=	mat_Nint(primitive_cell->position[i][j]);
    }
  }
}

/* Return NULL if failed */
static VecDBL *
translate_atoms_in_primitive_lattice(SPGCONST Cell * cell,
				     SPGCONST double prim_lat[3][3])
{
  int i, j;
  double tmp_matrix[3][3], axis_inv[3][3];
  VecDBL * position;

  position = NULL;

  if ((position = mat_alloc_VecDBL(cell->size)) == NULL) {
    return NULL;
  }

  mat_inverse_matrix_d3(tmp_matrix, prim_lat, 0);
  mat_multiply_matrix_d3(axis_inv, tmp_matrix, cell->lattice);

  /* Send atoms into the primitive cell */
  for (i = 0; i < cell->size; i++) {
    mat_multiply_matrix_vector_d3(position->vec[i],
				  axis_inv,
				  cell->position[i]);
    for (j = 0; j < 3; j++) {
      position->vec[i][j] -= mat_Nint(position->vec[i][j]);
    }
  }

  return position;
}


/* If overlap_table is correctly obtained, */
/* shape of overlap_table will be (cell->size, cell->size / primitive->size). */
/* Return NULL if failed */
static int * get_overlap_table(SPGCONST Cell *primitive_cell,
			       const VecDBL * position,
			       const int *types,
			       const double symprec)
{
  int i, j, attempt, num_overlap, ratio, cell_size, count;
  double trim_tolerance;
  int *overlap_table;

  cell_size = position->size;
  ratio = cell_size / primitive_cell->size;
  trim_tolerance = symprec;

  if ((overlap_table = (int*)malloc(sizeof(int) * cell_size)) == NULL) {
    return NULL;
  }
  
  for (attempt = 0; attempt < 100; attempt++) {
    for (i = 0; i < cell_size; i++) {
      overlap_table[i] = -1;
      num_overlap = 0;
      for (j = 0; j < cell_size; j++) {
	if (types[i] == types[j]) {
	  if (cel_is_overlap(position->vec[i],
			     position->vec[j],
			     primitive_cell->lattice,
			     trim_tolerance)) {
	    num_overlap++;
	    if (overlap_table[i] == -1) {
	      overlap_table[i] = j;
	      assert(j <= i);
	    }
	  }
	}
      }

      if (num_overlap == ratio)	{
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
	warning_print("(line %d, %s).\n", __LINE__, __FILE__);
	goto cont;
      }
    }

    for (i = 0; i < cell_size; i++) {
      if (overlap_table[i] != i) {
	continue;
      }
      count = 0;
      for (j = 0; j < cell_size; j++) {
	if (i == overlap_table[j]) {
	  count++;
	}
      }
      assert(count == ratio);
    }

    goto found;

  cont:
    ;
  }

  warning_print("spglib: Could not trim cell into primitive ");
  warning_print("(line %d, %s).\n", __LINE__, __FILE__);
  return NULL;

found:
  return overlap_table;
}

/* Return 0 if failed */
static int get_primitive_lattice_vectors_iterative(double prim_lattice[3][3],
						   SPGCONST Cell * cell,
						   const VecDBL * pure_trans,
						   const double symprec)
{
  int i, multi, attempt;
  double tolerance;
  VecDBL * vectors, * pure_trans_reduced, *tmp_vec;

  vectors = NULL;
  pure_trans_reduced = NULL;
  tmp_vec = NULL;

  tolerance = symprec;

  if ((pure_trans_reduced = mat_alloc_VecDBL(pure_trans->size)) == NULL) {
    goto fail;
  }

  for (i = 0; i < pure_trans->size; i++) {
    mat_copy_vector_d3(pure_trans_reduced->vec[i], pure_trans->vec[i]);
  }
  
  for (attempt = 0; attempt < 100; attempt++) {
    multi = pure_trans_reduced->size;

    if ((vectors = get_translation_candidates(pure_trans_reduced)) == NULL) {
      mat_free_VecDBL(pure_trans_reduced);
      goto fail;
    }

    /* Lattice of primitive cell is found among pure translation vectors */
    if (get_primitive_lattice_vectors(prim_lattice,
				      vectors,
				      cell,
				      tolerance)) {

      mat_free_VecDBL(vectors);
      mat_free_VecDBL(pure_trans_reduced);

      goto found;

    } else {

      if ((tmp_vec = mat_alloc_VecDBL(multi)) == NULL) {
	mat_free_VecDBL(vectors);
	mat_free_VecDBL(pure_trans_reduced);
	goto fail;
      }

      for (i = 0; i < multi; i++) {
	mat_copy_vector_d3(tmp_vec->vec[i], pure_trans_reduced->vec[i]);
      }
      mat_free_VecDBL(pure_trans_reduced);

      pure_trans_reduced = sym_reduce_pure_translation(cell,
						       tmp_vec,
						       tolerance);

      mat_free_VecDBL(tmp_vec);
      mat_free_VecDBL(vectors);

      if (pure_trans_reduced == NULL) {
	goto fail;
      }

      warning_print("Tolerance is reduced to %f (%d), size = %d\n",
		    tolerance, attempt, pure_trans_reduced->size);

      tolerance *= REDUCE_RATE;
    }
  }

  mat_free_VecDBL(pure_trans_reduced);

 fail:
  return 0;

 found:
  return multi;
}

/* Return 0 if failed */
static int get_primitive_lattice_vectors(double prim_lattice[3][3],
					 const VecDBL * vectors,
					 SPGCONST Cell * cell,
					 const double symprec)
{
  int i, j, k, size;
  double initial_volume, volume;
  double relative_lattice[3][3], min_vectors[3][3], tmp_lattice[3][3];
  double inv_mat_dbl[3][3];
  int inv_mat_int[3][3];

  debug_print("get_primitive_lattice_vectors:\n");

  size = vectors->size;
  initial_volume = mat_Dabs(mat_get_determinant_d3(cell->lattice));

  /* check volumes of all possible lattices, find smallest volume */
  for (i = 0; i < size; i++) {
    for (j = i + 1; j < size; j++) {
      for (k = j + 1; k < size; k++) {
	mat_multiply_matrix_vector_d3(tmp_lattice[0],
				      cell->lattice,
				      vectors->vec[i]);
	mat_multiply_matrix_vector_d3(tmp_lattice[1],
				      cell->lattice,
				      vectors->vec[j]);
	mat_multiply_matrix_vector_d3(tmp_lattice[2],
				      cell->lattice,
				      vectors->vec[k]);
	volume = mat_Dabs(mat_get_determinant_d3(tmp_lattice));
	if (volume > symprec) {
	  if (mat_Nint(initial_volume / volume) == size-2) {
	    mat_copy_vector_d3(min_vectors[0], vectors->vec[i]);
	    mat_copy_vector_d3(min_vectors[1], vectors->vec[j]);
	    mat_copy_vector_d3(min_vectors[2], vectors->vec[k]);
	    goto ret;
	  }
	}
      }
    }
  }

  /* Not found */
  warning_print("spglib: Primitive lattice vectors cound not be found ");
  warning_print("(line %d, %s).\n", __LINE__, __FILE__);
  return 0;

  /* Found */
 ret:
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      relative_lattice[j][i] = min_vectors[i][j];
    }
  }

  mat_inverse_matrix_d3(inv_mat_dbl, relative_lattice, 0);
  mat_cast_matrix_3d_to_3i(inv_mat_int, inv_mat_dbl);
  if (abs(mat_get_determinant_i3(inv_mat_int)) == size-2) {
    mat_cast_matrix_3i_to_3d(inv_mat_dbl, inv_mat_int);
    mat_inverse_matrix_d3(relative_lattice, inv_mat_dbl, 0);
  } else {
    warning_print("spglib: Primitive lattice cleaning is incomplete ");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);
  }
  mat_multiply_matrix_d3(prim_lattice, cell->lattice, relative_lattice);

  return 1;  
}

static VecDBL * get_translation_candidates(const VecDBL * pure_trans)
{
  int i, j, multi;
  VecDBL * vectors;

  vectors = NULL;
  multi = pure_trans->size;

  if ((vectors = mat_alloc_VecDBL(multi + 2)) == NULL) {
    return NULL;
  }

  /* store pure translations in original cell */ 
  /* as trial primitive lattice vectors */
  for (i = 0; i < multi - 1; i++) {
    mat_copy_vector_d3(vectors->vec[i], pure_trans->vec[i + 1]);
  }

  /* store lattice translations of original cell */
  /* as trial primitive lattice vectors */
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      if (i == j) {
	vectors->vec[i+multi-1][j] = 1;
      } else {
	vectors->vec[i+multi-1][j] = 0;
      }
    }
  }

  return vectors;
}

