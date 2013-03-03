#ifndef __dynmat_H__
#define __dynmat_H__

int get_dynamical_matrix_at_q(double *dynamical_matrix_real,
			      double *dynamical_matrix_image,
			      const int num_patom, 
			      const int num_satom,
			      const double *fc,
			      const double *q,
			      const double *r,
			      const int *multi,
			      const double *mass,
			      const int *s2p_map, 
			      const int *p2s_map,
			      const int is_nac,
			      const double *charge_sum);
void get_charge_sum(double *charge_sum,
		    const int num_patom,
		    const double factor,
		    const double q_vector[3],
		    const double *born);
#endif
