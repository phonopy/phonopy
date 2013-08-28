#ifndef __fc4_H__
#define __fc4_H__

int rotate_delta_fc3s(double *rotated_delta_fc3s,
		      const double *delta_fc3s,
		      const int *rot_map_syms,
		      const double *site_sym_cart,
		      const int num_rot,
		      const int num_delta_fc3s,
		      const int atom1,
		      const int atom2,
		      const int atom3,
		      const int num_atom);
int distribute_fc4(double *fc4,
		   const int fourth_atom,
		   const int *atom_mapping,
		   const int num_atom,
		   const double *rot_cart);
void set_translational_invariance_fc4(double *fc4,
				      const int num_atom);
void set_translational_invariance_fc4_per_index(double *fc4,
						const int num_atom,
						const int index);

#endif
