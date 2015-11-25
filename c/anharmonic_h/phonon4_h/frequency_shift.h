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

#ifndef __frequency_shift4_H__
#define __frequency_shift4_H__

#include <lapacke.h>
#include <phonoc_array.h>

void get_fc4_frequency_shifts(double *frequency_shifts,
			      const double *fc4_normal_real,
			      const double *frequencies,
			      const Iarray *grid_points1,
			      const Darray *temperatures,
			      const int *band_indicies,
			      const int num_band0,
			      const int num_band,
			      const double unit_conversion_factor);
void
get_fc4_normal_for_frequency_shift(double *fc4_normal_real,
				   const double *frequencies,
				   const lapack_complex_double *eigenvectors,
				   const int grid_point0,
				   const Iarray *grid_points1,
				   const int *grid_address,
				   const int *mesh,
				   const double *fc4,
				   const Darray *shortest_vectors,
				   const Iarray *multiplicity,
				   const double *masses,
				   const int *p2s_map,
				   const int *s2p_map,
				   const Iarray *band_indicies,
				   const double cutoff_frequency);
void reciprocal_to_normal4(lapack_complex_double *fc4_normal,
			   const lapack_complex_double *fc4_reciprocal,
			   const double *freqs0,
			   const double *freqs1,
			   const lapack_complex_double *eigvecs0,
			   const lapack_complex_double *eigvecs1,
			   const double *masses,
			   const int *band_indices,
			   const int num_band0,
			   const int num_band,
			   const double cutoff_frequency);
void set_phonons_for_frequency_shift(Darray *frequencies,
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

#endif
