#ifndef __fc3_H__
#define __fc3_H__

void distribute_fc3(double *fc3_copy,
		    const double *fc3,
		    const int third_atom,
		    const int *atom_mapping,
		    const int num_atom,
		    const double *rot_cart);
void tensor3_rotation(double *rot_tensor,
		      const double *tensor,
		      const double *rot_cartesian);
void set_permutation_symmetry_fc3(double *fc3, const int num_atom);

#endif
