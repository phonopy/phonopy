/* Copyright (C) 2017 Atsushi Togo */
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
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "overlap.h"
#include "mathfunc.h"
#include "debug.h"

/* 'a++' generalized to an arbitrary increment. */
/* Performs 'a += b' and returns the old value of a. */
#ifndef SPG_POST_INCREMENT
#define SPG_POST_INCREMENT(a, b) (a += (b), a - (b))
#endif

/* Note: data_out and data_in MUST NOT ALIAS. */
static void permute(void *data_out,
                    const void *data_in,
                    const int *perm,
                    int value_size,
                    int n);

static void permute_int(int *data_out,
                        const int *data_in,
                        const int *perm,
                        const int n);

static void permute_double_3(double (*data_out)[3],
                             SPGCONST double (*data_in)[3],
                             const int *perm,
                             const int n);

static int ValueWithIndex_comparator(const void *pa, const void *pb);

static void* perm_argsort_work_malloc(int n);

static void perm_argsort_work_free(void *work);

static int perm_argsort(int *perm,
                        const int *types,
                        const double *values,
                        void *provided_work,
                        const int n);

static int check_possible_overlap(OverlapChecker *checker,
                                  const double test_trans[3],
                                  SPGCONST int rot[3][3],
                                  const double symprec);

static int argsort_by_lattice_point_distance(int * perm,
                                             SPGCONST double lattice[3][3],
                                             SPGCONST double (* positions)[3],
                                             const int * types,
                                             double * distance_temp,
                                             void *argsort_work,
                                             const int size);

static OverlapChecker* overlap_checker_alloc(int size);

static int check_total_overlap_for_sorted(SPGCONST double lattice[3][3],
                                          SPGCONST double (*pos_original)[3],
                                          SPGCONST double (*pos_rotated)[3],
                                          const int types_original[],
                                          const int types_rotated[],
                                          const int num_pos,
                                          const double symprec);

/* ------------------------------------- */
/*          arg-sorting                  */

/* Helper type used to get sorted indices of values. */
typedef struct {
  double value;
  int type;
  int index;
} ValueWithIndex;


void ovl_overlap_checker_free(OverlapChecker *checker)
{
  if (checker != NULL) {
    if (checker->argsort_work != NULL) {
      free(checker->argsort_work);
      checker->argsort_work = NULL;
    }
    if (checker->blob != NULL) {
      free(checker->blob);
      checker->blob = NULL;
    }
    free(checker);
  }
}

OverlapChecker* ovl_overlap_checker_init(const Cell *cell)
{
  OverlapChecker * checker;
  checker = NULL;

  /* Allocate */
  if ((checker = overlap_checker_alloc(cell->size)) == NULL) {
    return NULL;
  }

  mat_copy_matrix_d3(checker->lattice, cell->lattice);

  /* Get the permutation that sorts the original cell. */
  if (!argsort_by_lattice_point_distance(checker->perm_temp,
                                         cell->lattice,
                                         cell->position,
                                         cell->types,
                                         checker->distance_temp,
                                         checker->argsort_work,
                                         checker->size)) {
    ovl_overlap_checker_free(checker);
    return NULL;
  }

  /* Use the perm to sort the cell. */
  /* The sorted cell is saved for as long as the OverlapChecker lives. */
  permute_double_3(checker->pos_sorted,
                   cell->position,
                   checker->perm_temp,
                   cell->size);

  permute_int(checker->types_sorted,
              cell->types,
              checker->perm_temp,
              cell->size);

  return checker;
}

/* Uses a OverlapChecker to efficiently--but thoroughly--confirm that a given symmetry operator */
/* is a symmetry of the cell. If you need to test many symmetry operators on the same cell, */
/* you can create one OverlapChecker from the Cell and call this function many times. */
/* -1: Error.  0:  Not a symmetry.   1. Is a symmetry. */
int ovl_check_total_overlap(OverlapChecker *checker,
                            const double test_trans[3],
                            int rot[3][3],
                            const double symprec,
                            const int is_identity)
{
  int i, k, check;

  /* Check a few atoms by brute force before continuing. */
  /* For bad translations, this can be much cheaper than sorting. */
  if (!check_possible_overlap(checker,
                              test_trans,
                              rot,
                              symprec)) {
    return 0;
  }

  /* Write rotated positions to 'pos_temp_1' */
  for (i = 0; i < checker->size; i++) {
    if (is_identity) {
      for (k = 0; k < 3; k++) {
        checker->pos_temp_1[i][k] = checker->pos_sorted[i][k];
      }
    } else {
      mat_multiply_matrix_vector_id3(checker->pos_temp_1[i],
                                     rot,
                                     checker->pos_sorted[i]);
    }

    for (k = 0; k < 3; k++) {
      checker->pos_temp_1[i][k] += test_trans[k];
    }
  }

  /* Get permutation that sorts these positions. */
  if (!argsort_by_lattice_point_distance(checker->perm_temp,
                                         checker->lattice,
                                         checker->pos_temp_1,
                                         checker->types_sorted,
                                         checker->distance_temp,
                                         checker->argsort_work,
                                         checker->size)) {
    return -1;
  }

  /* Use the permutation to sort them. Write to 'pos_temp_2'. */
  permute_double_3(checker->pos_temp_2,
                   checker->pos_temp_1,
                   checker->perm_temp,
                   checker->size);

  /* Do optimized check for overlap between sorted coordinates. */
  check = check_total_overlap_for_sorted(checker->lattice,
                                         checker->pos_sorted, /* pos_original */
                                         checker->pos_temp_2, /* pos_rotated */
                                         checker->types_sorted, /* types_original */
                                         checker->types_sorted, /* types_original */
                                         checker->size,
                                         symprec);
  if (check == -1) {
    /* Error! */
    return -1;
  }

  return check;
}

static int ValueWithIndex_comparator(const void *pa, const void *pb)
{
  int cmp;
  ValueWithIndex a, b;

  a = *((ValueWithIndex*) pa);
  b = *((ValueWithIndex*) pb);

  /* order by atom type, then by value */
  cmp = (b.type < a.type) - (a.type < b.type);
  if (!cmp) {
    cmp = (b.value < a.value) - (a.value < b.value);
  }

  return cmp;
}

static void* perm_argsort_work_malloc(int n)
{
  ValueWithIndex *work;

  work = NULL;

  if ((work = (ValueWithIndex*)(malloc(sizeof(ValueWithIndex) * n))) == NULL) {
    warning_print("spglib: Memory could not be allocated for argsort workspace.");
    return NULL;
  }
  return work;
}

static void perm_argsort_work_free(void *work)
{
  free(work);
}

/* Compute a permutation that sorts values first by atom type, */
/* and then by value.  If types is NULL, all atoms are assumed */
/* to have the same type. */
/* */
/* Returns 0 on failure. */
static int perm_argsort(int *perm,
                        const int *types,
                        const double *values,
                        void *provided_work,
                        const int n)
{
  int i;
  ValueWithIndex *work;

  work = NULL;

  if (provided_work) {
    work = (ValueWithIndex *) provided_work;
  } else if ((work = perm_argsort_work_malloc(n)) == NULL) {
    return 0;
  }

  /* Make array of all data for each value. */
  for (i = 0; i < n; i++) {
    work[i].value = values[i];
    work[i].index = i;
    work[i].type = types ? types[i] : 0;
  }

  /* Sort by type and value. */
  qsort(work, n, sizeof(ValueWithIndex), &ValueWithIndex_comparator);

  /* Retrieve indices.  This is the permutation. */
  for (i = 0; i < n; i++) {
    perm[i] = work[i].index;
  }

  if (!provided_work) {
    perm_argsort_work_free(work);
    work = NULL;
  }

  return 1;
}

/* Permute an array. */
/* data_out and data_in MUST NOT ALIAS. */
static void permute(void *data_out,
                    const void *data_in,
                    const int *perm,
                    int value_size,
                    int n)
{
  int i;
  const void *read;
  void *write;

  for (i = 0; i < n; i++) {
    read = (void *)((char *)data_in + perm[i] * value_size);
    write = (void *)((char *)data_out + i * value_size);
    memcpy(write, read, value_size);
  }
}

/* ***************************************** */
/*             OverlapChecker                */

static void permute_int(int *data_out,
                        const int *data_in,
                        const int *perm,
                        const int n) {
  permute(data_out, data_in, perm, sizeof(int), n);
}

static void permute_double_3(double (*data_out)[3],
                             SPGCONST double (*data_in)[3],
                             const int *perm,
                             const int n) {
  permute(data_out, data_in, perm, sizeof(double[3]), n);
}


static OverlapChecker* overlap_checker_alloc(int size)
{
  int offset_pos_temp_1, offset_pos_temp_2, offset_distance_temp;
  int offset_perm_temp, offset_pos_sorted, offset_types_sorted, offset_lattice;
  int offset, blob_size;
  char * chr_blob;
  OverlapChecker * checker;

  chr_blob = NULL;
  checker = NULL;

  /* checker->blob is going to contain lots of things. */
  /* Compute its total size and the number of bytes before each thing. */
  offset = 0;
  offset_pos_temp_1 = SPG_POST_INCREMENT(offset, size * sizeof(double[3]));
  offset_pos_temp_2 = SPG_POST_INCREMENT(offset, size * sizeof(double[3]));
  offset_distance_temp = SPG_POST_INCREMENT(offset, size * sizeof(double));
  offset_perm_temp = SPG_POST_INCREMENT(offset, size * sizeof(int));
  offset_lattice = SPG_POST_INCREMENT(offset, 9 * sizeof(double));
  offset_pos_sorted = SPG_POST_INCREMENT(offset, size * sizeof(double[3]));
  offset_types_sorted =  SPG_POST_INCREMENT(offset, size * sizeof(int));
  blob_size = offset;

  if ((checker = (OverlapChecker*)malloc(sizeof(OverlapChecker))) == NULL) {
    warning_print("spglib: Memory could not be allocated for checker.");
    return NULL;
  }

  if ((checker->blob = malloc(blob_size)) == NULL) {
    warning_print("spglib: Memory could not be allocated for checker.");
    free(checker);
    checker = NULL;
    return NULL;
  }

  if ((checker->argsort_work = perm_argsort_work_malloc(size)) == NULL) {
    free(checker->blob);
    checker->blob = NULL;
    free(checker);
    checker = NULL;
    return NULL;
  }

  checker->size = size;

  /* Create the pointers to the things contained in checker->blob. */
  /* The C spec doesn't allow arithmetic directly on 'void *', */
  /* so a 'char *' is used. */
  chr_blob = (char *)checker->blob;
  checker->pos_temp_1 = (double (*)[3])(chr_blob + offset_pos_temp_1);
  checker->pos_temp_2 = (double (*)[3])(chr_blob + offset_pos_temp_2);
  checker->distance_temp = (double *)(chr_blob + offset_distance_temp);
  checker->perm_temp = (int *)(chr_blob + offset_perm_temp);
  checker->lattice = (double (*)[3])(chr_blob + offset_lattice);
  checker->pos_sorted  = (double (*)[3])(chr_blob + offset_pos_sorted);
  checker->types_sorted = (int *)(chr_blob + offset_types_sorted);

  return checker;
}

static int argsort_by_lattice_point_distance(int * perm,
                                             SPGCONST double lattice[3][3],
                                             SPGCONST double (* positions)[3],
                                             const int * types,
                                             double * distance_temp,
                                             void *argsort_work,
                                             const int size)
{
  double diff[3];
  int i, k;
  double x;

  /* Fill distance_temp with distances. */
  for (i = 0; i < size; i++) {
    /* Fractional vector to nearest lattice point. */
    for (k = 0; k < 3; k++) {
      x = positions[i][k];
      diff[k] = x - mat_Nint(x);
    }

    /* Squared distance to lattice point. */
    mat_multiply_matrix_vector_d3(diff, lattice, diff);
    distance_temp[i] = mat_norm_squared_d3(diff);
  }

  return perm_argsort(perm,
                      types,
                      distance_temp,
                      argsort_work,
                      size);
}

/* Tests if an operator COULD be a symmetry of the cell, */
/* without the cost of sorting the rotated positions. */
/* It only inspects a few atoms. */
/* 0:  Not a symmetry.   1. Possible symmetry. */
static int check_possible_overlap(OverlapChecker *checker,
                                  const double test_trans[3],
                                  SPGCONST int rot[3][3],
                                  const double symprec)
{
  double pos_rot[3];
  int i, i_test, k, max_search_num, search_num;
  int type_rot, is_found;

  max_search_num = 3;
  search_num = checker->size <= max_search_num ? checker->size
                                                : max_search_num;

  /* Check a few rotated positions. */
  /* (this could be optimized by focusing on the min_atom_type) */
  for (i_test = 0; i_test < search_num; i_test++) {

    type_rot = checker->types_sorted[i_test];
    mat_multiply_matrix_vector_id3(pos_rot, rot, checker->pos_sorted[i_test]);
    for (k = 0; k < 3; k++) {
      pos_rot[k] += test_trans[k];
    }

    /* Brute-force search for the rotated position. */
    /* (this could be optimized by saving the sorted ValueWithIndex data */
    /*  for the original Cell and using it to binary search for lower and */
    /*  upper bounds on 'i'. For now though, brute force is good enough) */
    is_found = 0;
    for (i = 0; i < checker->size; i++) {
      if (cel_is_overlap_with_same_type(pos_rot,
                                        checker->pos_sorted[i],
                                        type_rot,
                                        checker->types_sorted[i],
                                        checker->lattice,
                                        symprec)) {
        is_found = 1;
        break;
      }
    }

    /* The rotated position is not in the structure! */
    /* This symmetry operator is therefore clearly invalid. */
    if (!is_found) {
      return 0;
    }
  }

  return 1;
}

/* Optimized for the case where the max difference in index */
/* between pos_original and pos_rotated is small. */
/* -1: Error.  0: False.  1:  True. */
static int check_total_overlap_for_sorted(SPGCONST double lattice[3][3],
                                          SPGCONST double (*pos_original)[3],
                                          SPGCONST double (*pos_rotated)[3],
                                          const int types_original[],
                                          const int types_rotated[],
                                          const int num_pos,
                                          const double symprec)
{
  int * found;
  int i, i_orig, i_rot;
  int search_start;

  found = NULL;

  if ((found = (int *)malloc(num_pos * sizeof(int))) == NULL) {
    warning_print("spglib: Memory could not be allocated");
    return -1;
  }

  /* found[i] = 1 if pos_rotated[i] has been found in pos_original */
  for (i = 0; i < num_pos; i++) {
    found[i] = 0;
  }

  search_start = 0;
  for (i_orig = 0; i_orig < num_pos; i_orig++) {

    /* Permanently skip positions filled near the beginning. */
    while (found[search_start]) {
      search_start++;
    }

    for (i_rot = search_start; i_rot < num_pos; i_rot++) {

      /* Skip any filled positions that aren't near the beginning. */
      if (found[i_rot]) {
        continue;
      }

      if (cel_is_overlap_with_same_type(pos_original[i_orig],
                                        pos_rotated[i_rot],
                                        types_original[i_orig],
                                        types_rotated[i_rot],
                                        lattice,
                                        symprec)) {
        found[i_rot] = 1;
        break;
      }
    }

    if (i_rot == num_pos) {
      /* We never hit the 'break'. */
      /* Failure; a position in pos_original does not */
      /* overlap with any position in pos_rotated. */
      return 0;
    }
  }

  free(found);
  found = NULL;

  /* Success */
  return 1;
}
