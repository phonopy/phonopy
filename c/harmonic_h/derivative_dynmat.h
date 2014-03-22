#ifndef __derivative_dynmat_H__
#define __derivative_dynmat_H__

void get_derivative_dynmat_at_q(double *derivative_dynmat_real,
				double *derivative_dynmat_imag,
				const int num_patom, 
				const int num_satom,
				const double *fc,
				const double *q,
				const double *lattice, /* column vector */
				const double *r,
				const int *multi,
				const double *mass,
				const int *s2p_map, 
				const int *p2s_map,
				const double nac_factor,
				const double *born,
				const double *dielectric,
				const double *q_direction);

#endif
