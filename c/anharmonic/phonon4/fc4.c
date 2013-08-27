#include "phonon3_h/fc3.h"
#include "phonon4_h/fc4.h"

static void tensor4_roation(double *rot_tensor,
			    const double *fc4,
			    const int atom_i,
			    const int atom_j,
			    const int atom_k,
			    const int atom_l,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int atom_rot_k,
			    const int atom_rot_l,
			    const int num_atom,
			    const double *rot_cartesian);
static double tensor4_rotation_elem(const double tensor[81],
				    const double *r,
				    const int m,
				    const int n,
				    const int p,
				    const int q);

int rotate_delta_fc3s(double *rotated_delta_fc3s,
		      const double *delta_fc3s,
		      const int *rot_map_syms,
		      const double *site_sym_cart,
		      const int num_rot,
		      const int num_delta_fc3s,
		      const int atom1,
		      const int atom2,
		      const int atom3,
		      const int num_atom)
{
  int i, j;
  double *rot_tensor;
  for (i = 0; i < num_delta_fc3s; i++) {
    for (j = 0; j < num_rot; j++) {
      rot_tensor = rotated_delta_fc3s + i * num_rot * 27 + j * 27;
      tensor3_roation(rot_tensor,
		      delta_fc3s +
		      i * num_atom * num_atom * num_atom * 27,
		      atom1,
		      atom2,
		      atom3,
		      rot_map_syms[num_atom * j + atom1],
		      rot_map_syms[num_atom * j + atom2],
		      rot_map_syms[num_atom * j + atom3],
		      num_atom,
		      site_sym_cart + j * 9);
    }
  }
  return 0;
}


int distribute_fc4(double *fc4,
		   const int fourth_atom,
		   const int *atom_mapping,
		   const int num_atom,
		   const double *rot_cart)
{
  int i, j, k, atom_rot_i, atom_rot_j, atom_rot_k, fourth_atom_rot;
  double *tensor;

  fourth_atom_rot = atom_mapping[fourth_atom];
  
  for (i = 0; i < num_atom; i++) {
    atom_rot_i = atom_mapping[i];

    for (j = 0; j < num_atom; j++) {
      atom_rot_j = atom_mapping[j];

      for (k = 0; k < num_atom; k++) {
	atom_rot_k = atom_mapping[k];

	tensor = (fc4 +
		  81 * num_atom * num_atom * num_atom * fourth_atom +
		  81 * num_atom * num_atom * i +
		  81 * num_atom * j +
		  81 * k);
	tensor4_roation(tensor,
			fc4,
			fourth_atom,
			i,
			j,
			k,
			fourth_atom_rot,
			atom_rot_i,
			atom_rot_j,
			atom_rot_k,
			num_atom,
			rot_cart);
      }
    }
  }
  return 1;
}

static void tensor4_roation(double *rot_tensor,
			    const double *fc4,
			    const int atom_i,
			    const int atom_j,
			    const int atom_k,
			    const int atom_l,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int atom_rot_k,
			    const int atom_rot_l,
			    const int num_atom,
			    const double *rot_cartesian)
{
  int i, j, k, l;
  double tensor[81];

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  tensor[i * 27 + j * 9 + k * 3 + l] =
	    fc4[81 * num_atom * num_atom * num_atom * atom_rot_i +
		81 * num_atom * num_atom * atom_rot_j +
		81 * num_atom * atom_rot_k +
		81 * atom_rot_l +
		27 * i + 9 * j + 3 * k + l];
	}
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  rot_tensor[i * 27 + j * 9 + k * 3 + l] = 
	    tensor4_rotation_elem(tensor, rot_cartesian, i, j, k, l);
	}
      }
    }
  }
}

static double tensor4_rotation_elem(const double tensor[81],
				    const double *r,
				    const int m,
				    const int n,
				    const int p,
				    const int q)
{
  int i, j, k, l;
  double sum;

  sum = 0.0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	sum += r[m * 3 + i] * r[n * 3 + j] * r[p * 3 + k] * r[q * 3 + l] *
	  tensor[i * 27 + j * 9 + k * 3 + l];
	}
      }
    }
  }
  return sum;
}

