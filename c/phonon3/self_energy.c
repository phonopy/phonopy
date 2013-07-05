#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "phonoc_array.h"

#define THZTOEVPARKB 47.992398658977166

static double gaussian(const double x, const double sigma);


static double occupation(const double x, const double t);
    
/* gamma[num_band, num_fpoints] */
void get_imag_self_energy(double *gamma,
			  const int *band_indices,
			  const double *freq_points,
			  const Darray *fc3_normal_sqared,
			  const double *frequencies,
			  const int *grid_point_triplets,
			  const int *triplet_weights,
			  const double sigma,
			  const double unit_conversion_factor)
{
}

static double gaussian(const double x, const double sigma)
{
  return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2);
}  

static double occupation(const double x, const double t)
{
  return 1.0 / (exp(THZTOEVPARKB * x / t) - 1);
}
