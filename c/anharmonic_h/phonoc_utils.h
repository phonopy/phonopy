#ifndef __phonoc_utils_H__
#define __phonoc_utils_H__

#include <lapacke.h>
#include "phonoc_array.h"

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
lapack_complex_double get_phase_factor(const double q[],
				       const Darray *shortest_vectors,
				       const Iarray *multiplicity,
				       const int pi0,
				       const int si,
				       const int qi);
double bose_einstein(const double x, const double t);
double gaussian(const double x, const double sigma);

#endif
