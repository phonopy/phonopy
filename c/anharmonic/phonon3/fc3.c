#include "phonon3_h/fc3.h"

static double tensor3_rotation_elem(const double tensor[27],
				    const double *r,
				    const int l,
				    const int m,
				    const int n);

int distribute_fc3(double *fc3,
		   const int third_atom,
		   const int *atom_mapping,
		   const int num_atom,
		   const double *rot_cart)
{
  int i, j, atom_rot_i, atom_rot_j, third_atom_rot;
  double *tensor;

  third_atom_rot = atom_mapping[third_atom];
  
  for (i = 0; i < num_atom; i++) {
    atom_rot_i = atom_mapping[i];

    for (j = 0; j < num_atom; j++) {
      atom_rot_j = atom_mapping[j];
      tensor = (fc3 +
		27 * num_atom * num_atom * third_atom +
		27 * num_atom * i +
		27 * j);
      tensor3_roation(tensor,
		      fc3,
		      third_atom,
		      i,
		      j,
		      third_atom_rot,
		      atom_rot_i,
		      atom_rot_j,
		      num_atom,
		      rot_cart);
    }
  }
  return 1;
}

void tensor3_roation(double *rot_tensor,
		     const double *fc3,
		     const int atom_i,
		     const int atom_j,
		     const int atom_k,
		     const int atom_rot_i,
		     const int atom_rot_j,
		     const int atom_rot_k,
		     const int num_atom,
		     const double *rot_cartesian)
{
  int i, j, k;
  double tensor[27];

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	tensor[i * 9 + j * 3 + k] = fc3[27 * num_atom * num_atom * atom_rot_i +
					27 * num_atom * atom_rot_j +
					27 * atom_rot_k +
					9 * i + 3 * j + k];
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot_tensor[i * 9 + j * 3 + k] = 
	  tensor3_rotation_elem(tensor, rot_cartesian, i, j, k);
      }
    }
  }
}

static double tensor3_rotation_elem(const double tensor[27],
				    const double *r,
				    const int l,
				    const int m,
				    const int n) 
{
  int i, j, k;
  double sum;

  sum = 0.0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	sum += r[l * 3 + i] * r[m * 3 + j] * r[n * 3 + k] *
	  tensor[i * 9 + j * 3 + k];
      }
    }
  }
  return sum;
}

