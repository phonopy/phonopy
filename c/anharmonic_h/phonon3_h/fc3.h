#ifndef __fc3_H__
#define __fc3_H__

int distribute_fc3(double *fc3,
		   const int third_atom,
		   const int *atom_mapping,
		   const int num_atom,
		   const double *rot_cart);
void tensor3_roation(double *rot_tensor,
		     const double *fc3,
		     const int atom_i,
		     const int atom_j,
		     const int atom_k,
		     const int atom_rot_i,
		     const int atom_rot_j,
		     const int atom_rot_k,
		     const int num_atom,
		     const double *rot_cartesian);

#endif
