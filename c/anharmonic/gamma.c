#include <stdio.h>
#include <math.h>
#include "gamma.h"

#define DISTFUNC gaussian

/* Boltzmann constant eV/K */
#define KB 8.6173382568083159E-05
/* Planck Constant for THz to Ev */
#define PlanckConstant 4.13566733E-3

static double gaussian(const double x, const double sigma);
static double lorentzian(const double x, const double sigma);
static double bs(const double x, const double t);

int get_gamma(double *gamma,
	      const int num_omega,
	      const int num_triplet,
	      const int band_index,
	      const int num_band0,
	      const int num_band,
	      const int *w,
	      const double *o,
	      const double *f,
	      const double *amp,
	      const double sigma,
	      const double t,
	      const double cutoff_frequency,
	      const double freq_factor,
	      const int option)
{
  int i, j, k, l;
  double f2, f3, n2, n3, a, factor2eV, sum;

  factor2eV = PlanckConstant / freq_factor;

  for (i = 0; i < num_omega; i++) {
    sum = 0.0;
#pragma omp parallel for private(k, l, a, f2, f3, n2, n3) reduction(+:sum)
    for (j = 0; j < num_triplet; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  f2 = f[j * 3 * num_band + num_band + k];
	  f3 = f[j * 3 * num_band + 2 * num_band + l];
	  if (f2 < cutoff_frequency || f3 < cutoff_frequency) {
	    continue;
	  }
	  a = amp[j * num_band0 * num_band * num_band +
		  band_index * num_band * num_band + k * num_band + l];

	  if (t > 0.0) {
	    n2 = bs(f2 * factor2eV, t);
	    n3 = bs(f3 * factor2eV, t);
	    switch (option) {
	    case 1:
	      sum += ((1.0 + n2 + n3) * DISTFUNC(f2 + f3 - o[i], sigma) +
	    	      (n3 - n2) * 2 * DISTFUNC(f2 - f3 - o[i], sigma)
	    	      ) * a * w[j];
	      break;
	    case 2:
	      sum += (1.0 + n2 + n3) * DISTFUNC(f2 + f3 - o[i], sigma) * a * w[j];
	      break;
	    case 3:
	      sum += (n3 - n2) * 2 * DISTFUNC(f2 - f3 - o[i], sigma) * a * w[j];
	      break;
	    case 4:
	      sum += (n3 - n2) * (DISTFUNC(f2 - f3 - o[i], sigma) -
	    			  DISTFUNC(f3 - f2 - o[i], sigma)) * a * w[j];
	      break;
	    case 5:
	      sum += (n3 - n2) * DISTFUNC(f2 - f3 - o[i], sigma) * a * w[j];
	      break;
	    case 6:
	      sum += (n2 - n3) * DISTFUNC(f3 - f2 - o[i], sigma) * a * w[j];
	      break;
	    case 0:
	    default:
	      sum += ((1.0 + n2 + n3) * DISTFUNC(f2 + f3 - o[i], sigma) +
	    	      (n3 - n2) * (DISTFUNC(f2 - f3 - o[i], sigma) -
	    			   DISTFUNC(f3 - f2 - o[i], sigma))
	    	      ) * a * w[j];
	      break;
	    }
	  } else {
	    sum += DISTFUNC(f2 + f3 - o[i], sigma) * a * w[j];
	  }
	}
      }
    }
    gamma[i] = sum;
  }

  return 1;
}

int get_decay_channels(double *decay,
		       const int num_omega,
		       const int num_triplet,
		       const int num_band,
		       const double *o,
		       const double *f,
		       const double *amp,
		       const double sigma,
		       const double t,
		       const double freq_factor)
{
  int i, j, k, l, address_a, address_d;
  double f2, f3, n2, n3, factor2eV;

  factor2eV = PlanckConstant / freq_factor;

#pragma omp parallel for private(j, k, l, address_a, address_d, f2, f3, n2, n3)
  for (i = 0; i < num_triplet; i++) {
    for (j = 0; j < num_band; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_omega; l++) {
	  address_a = i * num_omega * num_band * num_band + l * num_band * num_band + j * num_band + k;
	  address_d = i * num_band * num_band + j * num_band + k;
	  f2 = f[i * 3 * num_band + num_band + j];
	  f3 = f[i * 3 * num_band + 2 * num_band + k];
	  if (t > 0) {
	    n2 = bs(f2 * factor2eV, t);
	    n3 = bs(f3 * factor2eV, t);
	    decay[address_d] += ((1.0 + n2 + n3) * DISTFUNC(f2 + f3 - o[l], sigma) +
				 (n3 - n2) * (DISTFUNC(f2 - f3 - o[l], sigma) -
					      DISTFUNC(f3 - f2 - o[l], sigma))
				 ) * amp[address_a];
	  } else {
	    decay[address_d] += DISTFUNC(f2 + f3 - o[l], sigma) * amp[address_a];
	  }
	}
      }
    }
  }

  return 1;
}

int get_jointDOS(double *jdos,
		 const int num_omega,
		 const int num_triplet,
		 const int num_band,
		 const double *o,
		 const double *f,
		 const int *w,
		 const double sigma)
{
  int i, j, k, l;
  double f2, f3;

#pragma omp parallel for private(j, k, l, f2, f3)
  for (i = 0; i < num_omega; i++) {
    jdos[i] = 0.0;
    for (j = 0; j < num_triplet; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  f2 = f[j * 3 * num_band + num_band + k];
	  f3 = f[j * 3 * num_band + 2 * num_band + l];
	  jdos[i] += DISTFUNC(f2 + f3 - o[i], sigma) * w[j];
	}
      }
    }
  }

  return 1;
}

static double gaussian(const double x, const double sigma)
{
  return 1.0 / sqrt(2 * M_PI) / sigma * exp(-x*x / 2.0 / sigma / sigma);
}

static double lorentzian(const double x, const double gamma)
{
  return gamma / (x * x + gamma * gamma) / M_PI;
}

static double bs(const double x, const double t)
{
  return 1.0 / (exp(x / (KB * t)) - 1);
}

