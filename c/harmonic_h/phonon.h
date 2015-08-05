/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

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

#ifndef __phonon_H__
#define __phonon_H__

#include <lapacke.h>
#include <phonoc_array.h>

void set_phonons_at_gridpoints(Darray *frequencies,
			       Carray *eigenvectors,
			       char *phonon_done,
			       const Iarray *grid_points,
			       const int *grid_address,
			       const int *mesh,
			       const Darray *fc2,
			       const Darray *svecs_fc2,
			       const Iarray *multi_fc2,
			       const double *masses_fc2,
			       const int *p2s_fc2,
			       const int *s2p_fc2,
			       const double unit_conversion_factor,
			       const double *born,
			       const double *dielectric,
			       const double *reciprocal_lattice,
			       const double *q_direction,
			       const double nac_factor,
			       const char uplo);
void get_undone_phonons(Darray *frequencies,
			Carray *eigenvectors,
			const int *undone_grid_points,
			const int num_undone_grid_points,
			const int *grid_address,
			const int *mesh,
			const Darray *fc2,
			const Darray *svecs_fc2,
			const Iarray *multi_fc2,
			const double *masses_fc2,
			const int *p2s_fc2,
			const int *s2p_fc2,
			const double unit_conversion_factor,
			const double *born,
			const double *dielectric,
			const double *reciprocal_lattice,
			const double *q_direction,
			const double nac_factor,
			const char uplo);
int get_phonons(lapack_complex_double *a,
		double *w,
		const double q[3],
		const Darray *fc2,
		const double *masses,
		const int *p2s,
		const int *s2p,
		const Iarray *multi,
		const Darray *svecs,
		const double *born,
		const double *dielectric,
		const double *reciprocal_lattice,
		const double *q_direction,
		const double nac_factor,
		const double unit_conversion_factor,
		const char uplo);

#endif
