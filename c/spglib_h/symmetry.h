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

#ifndef __symmetry_H__
#define __symmetry_H__

#include "cell.h"
#include "mathfunc.h"

typedef struct {
  int size;
  int (*rot)[3][3];
  double (*trans)[3];
} Symmetry;

typedef struct {
  int rot[48][3][3];
  int size;
} PointSymmetry;

Symmetry * sym_alloc_symmetry(const int size);
void sym_free_symmetry(Symmetry * symmetry);
Symmetry * sym_get_operation(const Cell * primitive,
                             const double symprec,
                             const double angle_tolerance);
Symmetry * sym_reduce_operation(const Cell * primitive,
                                const Symmetry * symmetry,
                                const double symprec,
                                const double angle_tolerance);
VecDBL * sym_get_pure_translation(const Cell *cell,
                                  const double symprec);
VecDBL * sym_reduce_pure_translation(const Cell * cell,
                                     const VecDBL * pure_trans,
                                     const double symprec,
                                     const double angle_tolerance);

#endif
