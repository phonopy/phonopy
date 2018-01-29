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

#include "mathfunc.h"
#include "cell.h"

/* Contains pre-allocated memory and precomputed data for check_total_overlap. */
typedef struct {
  /* Number of atoms. */
  int size;

  /* Pre-allocated memory for various things. */
  void * argsort_work;
  void * blob;

  /* Temp areas for writing stuff. (points into blob) */
  double (*pos_temp_1)[3];
  double (*pos_temp_2)[3];

  /* Temp area for writing lattice point distances. (points into blob) */
  double * distance_temp; /* for lattice point distances */
  int * perm_temp; /* for permutations during sort */

  /* Sorted data of original cell. (points into blob)*/
  double (*lattice)[3];
  double (*pos_sorted)[3];
  int * types_sorted;
} OverlapChecker;

OverlapChecker* ovl_overlap_checker_init(const Cell *cell);

int ovl_check_total_overlap(OverlapChecker *checker,
                            const double test_trans[3],
                            int rot[3][3],
                            const double symprec,
                            const int is_identity);

void ovl_overlap_checker_free(OverlapChecker *checker);
