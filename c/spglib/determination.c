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
#include "cell.h"
#include "determination.h"
#include "primitive.h"
#include "spacegroup.h"

#include "debug.h"

#define REDUCE_RATE_OUTER 0.9
#define NUM_ATTEMPT_OUTER 10
#define REDUCE_RATE 0.95
#define NUM_ATTEMPT 20

static int get_spacegroup_and_primitive(DataContainer * container,
                                        const Cell * cell,
                                        const int hall_number,
                                        const double symprec,
                                        const double angle_tolerance);
DataContainer * det_determine_all(const Cell * cell,
                                  const int hall_number,
                                  const double symprec,
                                  const double angle_tolerance)
{
  int attempt;
  double tolerance;
  DataContainer *container;

  container = NULL;

  if ((container = (DataContainer*) malloc(sizeof(DataContainer))) == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    return NULL;
  }

  container->primitive = NULL;
  container->spacegroup = NULL;
  container->exact_structure = NULL;

  if ((container->spacegroup = (Spacegroup*) malloc(sizeof(Spacegroup)))
      == NULL) {
    warning_print("spglib: Memory could not be allocated.");
    det_free_container(container);
    container = NULL;
    return NULL;
  }

  tolerance = symprec;
  for (attempt = 0; attempt < NUM_ATTEMPT_OUTER; attempt++) {
    if (get_spacegroup_and_primitive(container,
                                     cell,
                                     hall_number,
                                     tolerance,
                                     angle_tolerance)) {
      if (container->spacegroup->number > 0) {
        if ((container->exact_structure = ref_get_exact_structure_and_symmetry(
               container->primitive->cell,
               cell,
               container->spacegroup,
               container->primitive->mapping_table,
               container->primitive->tolerance)) == NULL) {
          warning_print("spglib: ref_get_exact_structure_and_symmetry failed.");
          warning_print(" (line %d, %s).\n", __LINE__, __FILE__);
        } else {
          goto found;
        }
      }
      ref_free_exact_structure(container->exact_structure);
      container->exact_structure = NULL;
    }
    tolerance *= REDUCE_RATE_OUTER;
    prm_free_primitive(container->primitive);
    container->primitive = NULL;
  }

  det_free_container(container);
  return NULL;

found:
  return container;
}

/* NULL is returned if failed */
static int get_spacegroup_and_primitive(DataContainer * container,
                                        const Cell * cell,
                                        const int hall_number,
                                        const double symprec,
                                        const double angle_tolerance)
{
  int attempt;
  double tolerance;

  debug_print("get_spacegroup_and_primitive (tolerance = %f):\n", symprec);

  if (hall_number < 0 || hall_number > 530) {
    return 0;
  }

  tolerance = symprec;

  for (attempt = 0; attempt < NUM_ATTEMPT; attempt++) {
    if ((container->primitive = prm_get_primitive(cell,
                                                  tolerance,
                                                  angle_tolerance)) != NULL) {
      *(container->spacegroup) = spa_search_spacegroup(
        container->primitive->cell,
        hall_number,
        container->primitive->tolerance,
        container->primitive->angle_tolerance);

      if (container->spacegroup->number > 0) {
        goto found;
      }

      prm_free_primitive(container->primitive);
      container->primitive = NULL;
    }

    warning_print("spglib: Attempt %d tolerance = %f failed.",
                  attempt, tolerance);
    warning_print(" (line %d, %s).\n", __LINE__, __FILE__);

    tolerance *= REDUCE_RATE;
  }

  return 0;

found:
  return 1;
}

void det_free_container(DataContainer * container)
{
  if (container != NULL) {
    if (container->spacegroup != NULL) {
      free(container->spacegroup);
      container->spacegroup = NULL;
    }
    if (container->primitive != NULL) {
      prm_free_primitive(container->primitive);
      container->primitive = NULL;
    }
    if (container->exact_structure != NULL) {
      ref_free_exact_structure(container->exact_structure);
      container->exact_structure = NULL;
    }
    free(container);
  }
}
