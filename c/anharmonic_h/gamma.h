#ifndef __gamma_H__
#define __gamma_H__

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
	      const int option);
int get_decay_channels(double *decay,
		       const int num_omega,
		       const int num_triplet,
		       const int num_band,
		       const double *o,
		       const double *f,
		       const double *amp,
		       const double sigma,
		       const double t,
		       const double freq_factor);
int get_jointDOS(double *jdos,
		 const int num_omega,
		 const int num_triplet,
		 const int num_band,
		 const double *o,
		 const double *f,
		 const int *w,
		 const double sigma);
#endif
