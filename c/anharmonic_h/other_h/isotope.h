#ifndef __isotope_H__
#define __isotope_H__

void get_isotope_scattering_strength(double *gamma,
				     const int grid_point,
				     const double *mass_variances,
				     const double *frequencies,
				     const lapack_complex_double *eigenvectors,
				     const int num_grid_points,
				     const int *band_indices,
				     const int num_band,
				     const int num_band0,
				     const double sigma,
				     const double cutoff_frequency);
#endif
