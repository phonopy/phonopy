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
#include "sitesym_database.h"

#include "debug.h"

#define INCREASE_RATE 1.05
#define REDUCE_RATE 0.95
#define NUM_ATTEMPT 100

static int get_exact_positions(VecDBL *positions,
			       int * equiv_atoms,
			       SPGCONST Cell * conv_prim,
			       SPGCONST Symmetry * conv_sym,
			       const double symprec);
static int set_exact_location(double position[3],
			      SPGCONST Symmetry * conv_sym,
			      SPGCONST double bravais_lattice[3][3],
			      const double symprec);
static int set_equivalent_atom(VecDBL *positions,
			       int * equiv_atoms,
			       const int i,
			       const int num_indep_atoms,
			       const int *indep_atoms,
			       SPGCONST Cell * conv_prim,
			       SPGCONST Symmetry * conv_sym,
			       const double symprec);
static int set_Wyckoffs_labels(int * wyckoffs,
			       const VecDBL *positions,
			       const int * equiv_atoms,
			       SPGCONST Cell * conv_prim,
			       SPGCONST Symmetry * conv_sym,
			       const int hall_number,
			       const double symprec);
static int get_Wyckoff_notation(double position[3],
				SPGCONST Symmetry * conv_sym,
				SPGCONST double bravais_lattice[3][3],
				const int hall_number,
				const double symprec);
static SPGCONST int identity[3][3] = {
  { 1, 0, 0},
  { 0, 1, 0},
  { 0, 0, 1},
};

/* Return if failed */
VecDBL * ssm_get_exact_positions(int *wyckoffs,
				 int *equiv_atoms,
				 SPGCONST Cell * conv_prim,
				 SPGCONST Symmetry * conv_sym,
				 const int hall_number,
				 const double symprec)
{
  int attempt, num_atoms;
  double tolerance;
  VecDBL *positions;

  positions = NULL;

  if ((positions = mat_alloc_VecDBL(conv_prim->size)) == NULL) {
    return NULL;
  }

  tolerance = symprec;
  for (attempt = 0; attempt < NUM_ATTEMPT; attempt++) {
    num_atoms = get_exact_positions(positions,
				    equiv_atoms,
				    conv_prim,
				    conv_sym,
				    tolerance);
    if (num_atoms == conv_prim->size) {
      goto succeeded;
    }

    if (num_atoms > conv_prim->size || num_atoms == 0) {
      tolerance *= INCREASE_RATE;
      warning_print("spglib: Some site-symmetry is found broken. ");
      warning_print("(%d != %d)\n", num_atoms, conv_prim->size);
      warning_print("        Increase tolerance to %f", tolerance);
      warning_print(" (%d)", attempt);
      warning_print(" (line %d, %s).\n", __LINE__, __FILE__);
    }

    if (num_atoms < conv_prim->size && num_atoms) {
      tolerance *= REDUCE_RATE;
      warning_print("spglib: Some site-symmetry is found broken. ");
      warning_print("(%d != %d)\n", num_atoms, conv_prim->size);
      warning_print("        Reduce tolerance to %f", tolerance);
      warning_print(" (%d)", attempt);
      warning_print(" (line %d, %s).\n", __LINE__, __FILE__);
    }
  }

  mat_free_VecDBL(positions);
  positions = NULL;
  return NULL;

 succeeded:
  if (! set_Wyckoffs_labels(wyckoffs,
			    positions,
			    equiv_atoms,
			    conv_prim,
			    conv_sym,
			    hall_number,
			    symprec)) {
    mat_free_VecDBL(positions);
    positions = NULL;
  }

  return positions;
}

/* Return 0 if failed */
static int get_exact_positions(VecDBL *positions,
			       int * equiv_atoms,
			       SPGCONST Cell * conv_prim,
			       SPGCONST Symmetry * conv_sym,
			       const double symprec)
{
  int i, num_indep_atoms, sum_num_atoms_in_orbits, num_atoms_in_orbits;
  int *indep_atoms;

  debug_print("get_exact_positions\n");

  indep_atoms = NULL;

  if ((indep_atoms = (int*) malloc(sizeof(int) * conv_prim->size)) == NULL) {
    warning_print("spglib: Memory could not be allocated ");
    return 0;
  }

  num_indep_atoms = 0;
  sum_num_atoms_in_orbits = 0;
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
      indep_atoms[num_indep_atoms] = i;
      num_indep_atoms++;
      mat_copy_vector_d3(positions->vec[i], conv_prim->position[i]);
      num_atoms_in_orbits = set_exact_location(positions->vec[i],
					       conv_sym,
					       conv_prim->lattice,
					       symprec);
      if (num_atoms_in_orbits) {
	sum_num_atoms_in_orbits += num_atoms_in_orbits;
	equiv_atoms[i] = i;
      } else {
	sum_num_atoms_in_orbits = 0;
	break;
      }
    }
  }

  free(indep_atoms);
  indep_atoms = NULL;

  return sum_num_atoms_in_orbits;
}

static int set_equivalent_atom(VecDBL *positions,
			       int * equiv_atoms,
			       const int i,
			       const int num_indep_atoms,
			       const int *indep_atoms,
			       SPGCONST Cell * conv_prim,
			       SPGCONST Symmetry * conv_sym,
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
      if (cel_is_overlap(pos,
			 conv_prim->position[i],
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
static int set_exact_location(double position[3],
			      SPGCONST Symmetry * conv_sym,
			      SPGCONST double bravais_lattice[3][3],
			      const double symprec)
{
  int i, j, k, num_sum, multi, num_pure_trans;
  double sum_rot[3][3];
  double pos[3], sum_trans[3];

  debug_print("get_exact_location\n");

  num_sum = 0;
  for (i = 0; i < 3; i++) {
    sum_trans[i] = 0.0;
    for (j = 0; j < 3; j++) {
      sum_rot[i][j] = 0;
    }
  }

  num_pure_trans = 0;
  for (i = 0; i < conv_sym->size; i++) {
    if (mat_check_identity_matrix_i3(identity, conv_sym->rot[i])) {
      num_pure_trans++;
      for (j = 0; j < 3; j++) {
	pos[j] = position[j];
      }
    } else {
      mat_multiply_matrix_vector_id3(pos,
				     conv_sym->rot[i],
				     position);
    }
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
      num_sum++;
    }
  }

  for (i = 0; i < 3; i++) {
    sum_trans[i] /= num_sum;
    for (j = 0; j < 3; j++) {
      sum_rot[i][j] /= num_sum;
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

  multi = conv_sym->size / num_pure_trans / num_sum;
  if (multi * num_sum * num_pure_trans == conv_sym->size) {
    return multi;
  } else {
    return 0;
  }
}

static int set_Wyckoffs_labels(int *wyckoffs,
			       const VecDBL *positions,
			       const int * equiv_atoms,
			       SPGCONST Cell * conv_prim,
			       SPGCONST Symmetry * conv_sym,
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
static int get_Wyckoff_notation(double position[3],
				SPGCONST Symmetry * conv_sym,
				SPGCONST double bravais_lattice[3][3],
				const int hall_number,
				const double symprec)
{
  int i, j, k, l, at_orbit, num_sitesym, wyckoff_letter;
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
    num_sitesym = ssmdb_get_coordinate(rot, trans, i + indices_wyc[0]);
    for (j = 0; j < pos_rot->size; j++) {
      at_orbit = 0;
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
	    at_orbit++;
	  }
	}
      }
      if (at_orbit == conv_sym->size / num_sitesym) {
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
