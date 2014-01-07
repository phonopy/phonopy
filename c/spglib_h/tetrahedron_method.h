/* tetrahedron_method.h */
/* Copyright (C) 2014 Atsushi Togo */

#ifndef __tetrahedron_method_H__
#define __tetrahedron_method_H__

#include "mathfunc.h"

void thm_get_relative_grid_address(int relative_grid_address[24][4][3],
				   SPGCONST double rec_lattice[3][3]);
double thm_get_integration_weight(const double omega,
				  SPGCONST double tetrahedra_omegas[24][4],
				  const char function);

#endif
