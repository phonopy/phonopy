#include "phonon3_h/fc3.h"

static double tensor3_rotation_elem(const double tensor[27],
				    const double *r,
				    const int l,
				    const int m,
				    const int n);
static void copy_permutation_symmetry_fc3_elem(double *fc3,
					       const double fc3_elem[27],
					       const int a,
					       const int b,
					       const int c,
					       const int num_atom);
static void set_permutation_symmetry_fc3_elem(double *fc3_elem,
					      const double *fc3,
					      const int a,
					      const int b,
					      const int c,
					      const int num_atom);

int distribute_fc3(double *fc3,
		   const int third_atom,
		   const int *atom_mapping,
		   const int num_atom,
		   const double *rot_cart)
{
  int i, j, atom_rot_i, atom_rot_j, third_atom_rot;
  double *tensor;

  third_atom_rot = atom_mapping[third_atom];
  
#pragma omp parallel for private(j, atom_rot_i, atom_rot_j, tensor)
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

void set_permutation_symmetry_fc3(double *fc3, const int num_atom)
{
  double fc3_elem[27];
  int i, j, k;

#pragma omp parallel for private(j, k, fc3_elem)
  for (i = 0; i < num_atom; i++) {
    for (j = i; j < num_atom; j++) {
      for (k = j; k < num_atom; k++) {
	set_permutation_symmetry_fc3_elem(fc3_elem, fc3, i, j, k, num_atom);
	copy_permutation_symmetry_fc3_elem(fc3, fc3_elem,
					   i, j, k, num_atom);
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


static void copy_permutation_symmetry_fc3_elem(double *fc3,
					       const double fc3_elem[27],
					       const int a,
					       const int b,
					       const int c,
					       const int num_atom)
{
  int i, j, k;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	fc3[a * num_atom * num_atom * 27 +
	    b * num_atom * 27 +
	    c * 27 + i * 9 + j * 3 + k] =
	  fc3_elem[i * 9 + j * 3 + k];
	fc3[a * num_atom * num_atom * 27 +
	    c * num_atom * 27 +
	    b * 27 + i * 9 + k * 3 + j] =
	  fc3_elem[i * 9 + j * 3 + k];
	fc3[b * num_atom * num_atom * 27 +
	    a * num_atom * 27 +
	    c * 27 + j * 9 + i * 3 + k] =
	  fc3_elem[i * 9 + j * 3 + k];
	fc3[b * num_atom * num_atom * 27 +
	    c * num_atom * 27 +
	    a * 27 + j * 9 + k * 3 + i] =
	  fc3_elem[i * 9 + j * 3 + k];
	fc3[c * num_atom * num_atom * 27 +
	    a * num_atom * 27 +
	    b * 27 + k * 9 + i * 3 + j] =
	  fc3_elem[i * 9 + j * 3 + k];
	fc3[c * num_atom * num_atom * 27 +
	    b * num_atom * 27 +
	    a * 27 + k * 9 + j * 3 + i] =
	  fc3_elem[i * 9 + j * 3 + k];
      }
    }
  }
}

static void set_permutation_symmetry_fc3_elem(double *fc3_elem,
					      const double *fc3,
					      const int a,
					      const int b,
					      const int c,
					      const int num_atom)
{
  int i, j, k;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	fc3_elem[i * 9 + j * 3 + k] =
	  (fc3[a * num_atom * num_atom * 27 +
	       b * num_atom * 27 +
	       c * 27 + i * 9 + j * 3 + k] +
	   fc3[a * num_atom * num_atom * 27 +
	       c * num_atom * 27 +
	       b * 27 + i * 9 + k * 3 + j] +
	   fc3[b * num_atom * num_atom * 27 +
	       a * num_atom * 27 +
	       c * 27 + j * 9 + i * 3 + k] +
	   fc3[b * num_atom * num_atom * 27 +
	       c * num_atom * 27 +
	       a * 27 + j * 9 + k * 3 + i] +
	   fc3[c * num_atom * num_atom * 27 +
	       a * num_atom * 27 +
	       b * 27 + k * 9 + i * 3 + j] +
	   fc3[c * num_atom * num_atom * 27 +
	       b * num_atom * 27 +
	       a * 27 + k * 9 + j * 3 + i]) / 6;
      }
    }
  }
}
