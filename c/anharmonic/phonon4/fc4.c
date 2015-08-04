/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include <math.h>
#include <stdlib.h>
#include <phonon3_h/fc3.h>
#include <phonon4_h/fc4.h>

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
static double get_drift_fc4_elem(const double *fc4,
				 const int num_atom,
				 const int i,
				 const int j,
				 const int k,
				 const int l,
				 const int index);
static void copy_permutation_symmetry_fc4_elem(double *fc4,
					       const double fc4_elem[81],
					       const int a,
					       const int b,
					       const int c,
					       const int d,
					       const int num_atom);
static void set_permutation_symmetry_fc4_elem(double *fc4_elem,
					      const double *fc4,
					      const int a,
					      const int b,
					      const int c,
					      const int d,
					      const int num_atom);

int rotate_delta_fc3s_elem(double *rotated_delta_fc3s,
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
      tensor3_rotation(rot_tensor,
		       delta_fc3s +
		       i * num_atom * num_atom * num_atom * 27 +
		       27 * num_atom * num_atom *
		       rot_map_syms[num_atom * j + atom1] +
		       27 * num_atom * rot_map_syms[num_atom * j + atom2] +
		       27 * rot_map_syms[num_atom * j + atom3],
		       site_sym_cart + j * 9);
    }
  }
  return 0;
}


int distribute_fc4(double *fc4_copy,
		   const double *fc4,
		   const int fourth_atom,
		   const int *atom_mapping,
		   const int num_atom,
		   const double *rot_cart)
{
  int i, j, k, atom_rot_i, atom_rot_j, atom_rot_k, fourth_atom_rot;
  double *tensor;

  fourth_atom_rot = atom_mapping[fourth_atom];
  
#pragma omp parallel for private(j, k, atom_rot_i, atom_rot_j, atom_rot_k, tensor)
  for (i = 0; i < num_atom; i++) {
    atom_rot_i = atom_mapping[i];

    for (j = 0; j < num_atom; j++) {
      atom_rot_j = atom_mapping[j];

      for (k = 0; k < num_atom; k++) {
	atom_rot_k = atom_mapping[k];

	tensor = (fc4_copy +
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

void set_translational_invariance_fc4(double *fc4,
				      const int num_atom)
{
  int i;

  for (i = 0; i < 4; i++) {
    set_translational_invariance_fc4_per_index(fc4, num_atom, i);
  }
}
  
void set_translational_invariance_fc4_per_index(double *fc4,
						const int num_atom,
						const int index)
{
  int i, j, k, l, m;
  double drift;

#pragma omp parallel for private(j, k, l, m, drift)
  for (i = 0; i < 81; i++) {
    for (j = 0; j < num_atom; j++) {
      for (k = 0; k < num_atom; k++) {
	for (l = 0; l < num_atom; l++) {
	  drift =
	    get_drift_fc4_elem(fc4, num_atom, i, j, k, l, index) / num_atom;
	  for (m = 0; m < num_atom; m++) {
	    switch (index) {
	    case 0:
	      fc4[m * num_atom * num_atom * num_atom * 81 +
		  j * num_atom * num_atom * 81 +
		  k * num_atom * 81 +
		  l * 81 + i] -= drift;
	      break;
	    case 1:
	      fc4[j * num_atom * num_atom * num_atom * 81 +
		  m * num_atom * num_atom * 81 +
		  k * num_atom * 81 +
		  l * 81 + i] -= drift;
	      break;
	    case 2:
	      fc4[j * num_atom * num_atom * num_atom * 81 +
		  k * num_atom * num_atom * 81 +
		  m * num_atom * 81 +			 
		  l * 81 + i] -= drift;
	      break;
	    case 3:
	      fc4[j * num_atom * num_atom * num_atom * 81 +
		  k * num_atom * num_atom * 81 +
		  l * num_atom * 81 +
		  m * 81 + i] -= drift;
	      break;
	    }
	  }
	}
      }
    }
  }
}

void set_permutation_symmetry_fc4(double *fc4, const int num_atom)
{
  double fc4_elem[81];
  int i, j, k, l;

#pragma omp parallel for private(j, k, l, fc4_elem)
  for (i = 0; i < num_atom; i++) {
    for (j = i; j < num_atom; j++) {
      for (k = j; k < num_atom; k++) {
	for (l = k; l < num_atom; l++) {
	  set_permutation_symmetry_fc4_elem(fc4_elem, fc4, i, j, k, l, num_atom);
	  copy_permutation_symmetry_fc4_elem(fc4, fc4_elem,
					     i, j, k, l, num_atom);
	}
      }
    }
  }
}

static void copy_permutation_symmetry_fc4_elem(double *fc4,
					       const double fc4_elem[81],
					       const int a,
					       const int b,
					       const int c,
					       const int d,
					       const int num_atom)
{
  int i, j, k, l;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  fc4[a * num_atom * num_atom * num_atom * 81 +
	      b * num_atom * num_atom * 81 +
	      c * num_atom * 81+
	      d * 81 + i * 27 + j * 9 + k * 3 + l] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[a * num_atom * num_atom * num_atom * 81 +
	      b * num_atom * num_atom * 81 +
	      d * num_atom * 81+
	      c * 81 + i * 27 + j * 9 + l * 3 + k] = 
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[a * num_atom * num_atom * num_atom * 81 +
	      c * num_atom * num_atom * 81 +
	      b * num_atom * 81+
	      d * 81 + i * 27 + k * 9 + j * 3 + l] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[a * num_atom * num_atom * num_atom * 81 +
	      c * num_atom * num_atom * 81 +
	      d * num_atom * 81+
	      b * 81 + i * 27 + k * 9 + l * 3 + j] = 
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[a * num_atom * num_atom * num_atom * 81 +
	      d * num_atom * num_atom * 81 +
	      b * num_atom * 81+
	      c * 81 + i * 27 + l * 9 + j * 3 + k] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[a * num_atom * num_atom * num_atom * 81 +
	      d * num_atom * num_atom * 81 +
	      c * num_atom * 81+
	      b * 81 + i * 27 + l * 9 + k * 3 + j] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[b * num_atom * num_atom * num_atom * 81 +
	      a * num_atom * num_atom * 81 +
	      c * num_atom * 81+
	      d * 81 + j * 27 + i * 9 + k * 3 + l] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[b * num_atom * num_atom * num_atom * 81 +
	      a * num_atom * num_atom * 81 +
	      d * num_atom * 81+
	      c * 81 + j * 27 + i * 9 + l * 3 + k] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[b * num_atom * num_atom * num_atom * 81 +
	      c * num_atom * num_atom * 81 +
	      a * num_atom * 81+
	      d * 81 + j * 27 + k * 9 + i * 3 + l] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[b * num_atom * num_atom * num_atom * 81 +
	      c * num_atom * num_atom * 81 +
	      d * num_atom * 81+
	      a * 81 + j * 27 + k * 9 + l * 3 + i] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[b * num_atom * num_atom * num_atom * 81 +
	      d * num_atom * num_atom * 81 +
	      a * num_atom * 81+
	      c * 81 + j * 27 + l * 9 + i * 3 + k] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[b * num_atom * num_atom * num_atom * 81 +
	      d * num_atom * num_atom * 81 +
	      c * num_atom * 81+
	      a * 81 + j * 27 + l * 9 + k * 3 + i] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[c * num_atom * num_atom * num_atom * 81 +
	      a * num_atom * num_atom * 81 +
	      b * num_atom * 81+
	      d * 81 + k * 27 + i * 9 + j * 3 + l] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[c * num_atom * num_atom * num_atom * 81 +
	      a * num_atom * num_atom * 81 +
	      d * num_atom * 81+
	      b * 81 + k * 27 + i * 9 + l * 3 + j] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[c * num_atom * num_atom * num_atom * 81 +
	      b * num_atom * num_atom * 81 +
	      a * num_atom * 81+
	      d * 81 + k * 27 + j * 9 + i * 3 + l] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[c * num_atom * num_atom * num_atom * 81 +
	      b * num_atom * num_atom * 81 +
	      d * num_atom * 81+
	      a * 81 + k * 27 + j * 9 + l * 3 + i] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[c * num_atom * num_atom * num_atom * 81 +
	      d * num_atom * num_atom * 81 +
	      a * num_atom * 81+
	      b * 81 + k * 27 + l * 9 + i * 3 + j] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[c * num_atom * num_atom * num_atom * 81 +
	      d * num_atom * num_atom * 81 +
	      b * num_atom * 81+
	      a * 81 + k * 27 + l * 9 + j * 3 + i] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[d * num_atom * num_atom * num_atom * 81 +
	      a * num_atom * num_atom * 81 +
	      b * num_atom * 81+
	      c * 81 + l * 27 + i * 9 + j * 3 + k] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[d * num_atom * num_atom * num_atom * 81 +
	      a * num_atom * num_atom * 81 +
	      c * num_atom * 81+
	      b * 81 + l * 27 + i * 9 + k * 3 + j] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[d * num_atom * num_atom * num_atom * 81 +
	      b * num_atom * num_atom * 81 +
	      a * num_atom * 81+
	      c * 81 + l * 27 + j * 9 + i * 3 + k] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[d * num_atom * num_atom * num_atom * 81 +
	      b * num_atom * num_atom * 81 +
	      c * num_atom * 81+
	      a * 81 + l * 27 + j * 9 + k * 3 + i] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[d * num_atom * num_atom * num_atom * 81 +
	      c * num_atom * num_atom * 81 +
	      a * num_atom * 81+
	      b * 81 + l * 27 + k * 9 + i * 3 + j] =
	    fc4_elem[i * 27 + j * 9 + k * 3 + l];
	  fc4[d * num_atom * num_atom * num_atom * 81 +
	      c * num_atom * num_atom * 81 +
	      b * num_atom * 81+
	      a * 81 + l * 27 + k * 9 + j * 3 + i] =
	   fc4_elem[i * 27 + j * 9 + k * 3 + l];
	}
      }
    }
  }
}

static void set_permutation_symmetry_fc4_elem(double *fc4_elem,
					      const double *fc4,
					      const int a,
					      const int b,
					      const int c,
					      const int d,
					      const int num_atom)
{
  int i, j, k, l;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  fc4_elem[i * 27 + j * 9 + k * 3 + l] =
	    (fc4[a * num_atom * num_atom * num_atom * 81 +
		 b * num_atom * num_atom * 81 +
		 c * num_atom * 81+
		 d * 81 + i * 27 + j * 9 + k * 3 + l] +
	     fc4[a * num_atom * num_atom * num_atom * 81 +
		 b * num_atom * num_atom * 81 +
		 d * num_atom * 81+
		 c * 81 + i * 27 + j * 9 + l * 3 + k] +
	     fc4[a * num_atom * num_atom * num_atom * 81 +
		 c * num_atom * num_atom * 81 +
		 b * num_atom * 81+
		 d * 81 + i * 27 + k * 9 + j * 3 + l] +
	     fc4[a * num_atom * num_atom * num_atom * 81 +
		 c * num_atom * num_atom * 81 +
		 d * num_atom * 81+
		 b * 81 + i * 27 + k * 9 + l * 3 + j] +
	     fc4[a * num_atom * num_atom * num_atom * 81 +
		 d * num_atom * num_atom * 81 +
		 b * num_atom * 81+
		 c * 81 + i * 27 + l * 9 + j * 3 + k] +
	     fc4[a * num_atom * num_atom * num_atom * 81 +
		 d * num_atom * num_atom * 81 +
		 c * num_atom * 81+
		 b * 81 + i * 27 + l * 9 + k * 3 + j] +
	     fc4[b * num_atom * num_atom * num_atom * 81 +
		 a * num_atom * num_atom * 81 +
		 c * num_atom * 81+
		 d * 81 + j * 27 + i * 9 + k * 3 + l] +
	     fc4[b * num_atom * num_atom * num_atom * 81 +
		 a * num_atom * num_atom * 81 +
		 d * num_atom * 81+
		 c * 81 + j * 27 + i * 9 + l * 3 + k] +
	     fc4[b * num_atom * num_atom * num_atom * 81 +
		 c * num_atom * num_atom * 81 +
		 a * num_atom * 81+
		 d * 81 + j * 27 + k * 9 + i * 3 + l] +
	     fc4[b * num_atom * num_atom * num_atom * 81 +
		 c * num_atom * num_atom * 81 +
		 d * num_atom * 81+
		 a * 81 + j * 27 + k * 9 + l * 3 + i] +
	     fc4[b * num_atom * num_atom * num_atom * 81 +
		 d * num_atom * num_atom * 81 +
		 a * num_atom * 81+
		 c * 81 + j * 27 + l * 9 + i * 3 + k] +
	     fc4[b * num_atom * num_atom * num_atom * 81 +
		 d * num_atom * num_atom * 81 +
		 c * num_atom * 81+
		 a * 81 + j * 27 + l * 9 + k * 3 + i] +
	     fc4[c * num_atom * num_atom * num_atom * 81 +
		 a * num_atom * num_atom * 81 +
		 b * num_atom * 81+
		 d * 81 + k * 27 + i * 9 + j * 3 + l] +
	     fc4[c * num_atom * num_atom * num_atom * 81 +
		 a * num_atom * num_atom * 81 +
		 d * num_atom * 81+
		 b * 81 + k * 27 + i * 9 + l * 3 + j] +
	     fc4[c * num_atom * num_atom * num_atom * 81 +
		 b * num_atom * num_atom * 81 +
		 a * num_atom * 81+
		 d * 81 + k * 27 + j * 9 + i * 3 + l] +
	     fc4[c * num_atom * num_atom * num_atom * 81 +
		 b * num_atom * num_atom * 81 +
		 d * num_atom * 81+
		 a * 81 + k * 27 + j * 9 + l * 3 + i] +
	     fc4[c * num_atom * num_atom * num_atom * 81 +
		 d * num_atom * num_atom * 81 +
		 a * num_atom * 81+
		 b * 81 + k * 27 + l * 9 + i * 3 + j] +
	     fc4[c * num_atom * num_atom * num_atom * 81 +
		 d * num_atom * num_atom * 81 +
		 b * num_atom * 81+
		 a * 81 + k * 27 + l * 9 + j * 3 + i] +
	     fc4[d * num_atom * num_atom * num_atom * 81 +
		 a * num_atom * num_atom * 81 +
		 b * num_atom * 81+
		 c * 81 + l * 27 + i * 9 + j * 3 + k] +
	     fc4[d * num_atom * num_atom * num_atom * 81 +
		 a * num_atom * num_atom * 81 +
		 c * num_atom * 81+
		 b * 81 + l * 27 + i * 9 + k * 3 + j] +
	     fc4[d * num_atom * num_atom * num_atom * 81 +
		 b * num_atom * num_atom * 81 +
		 a * num_atom * 81+
		 c * 81 + l * 27 + j * 9 + i * 3 + k] +
	     fc4[d * num_atom * num_atom * num_atom * 81 +
		 b * num_atom * num_atom * 81 +
		 c * num_atom * 81+
		 a * 81 + l * 27 + j * 9 + k * 3 + i] +
	     fc4[d * num_atom * num_atom * num_atom * 81 +
		 c * num_atom * num_atom * 81 +
		 a * num_atom * 81+
		 b * 81 + l * 27 + k * 9 + i * 3 + j] +
	     fc4[d * num_atom * num_atom * num_atom * 81 +
		 c * num_atom * num_atom * 81 +
		 b * num_atom * 81+
		 a * 81 + l * 27 + k * 9 + j * 3 + i]) / 24;
	}
      }
    }
  }
}
				  

void get_drift_fc4(double *drifts_out, const double *fc4, const int num_atom)
{
  int i, j, k, l, index;
  double drift;
  double max_drift[81];
  

  for (index = 0; index < 4; index++) {
    for (i = 0; i < 81; i++) {
      max_drift[i] = 0;
    }

#pragma omp parallel for private(j, k, l, drift)
    for (i = 0; i < 81; i++) {
      for (j = 0; j < num_atom; j++) {
	for (k = 0; k < num_atom; k++) {
	  for (l = 0; l < num_atom; l++) {
	    drift = get_drift_fc4_elem(fc4, num_atom, i, j, k, l, index);
	    if (fabs(max_drift[i]) < fabs(drift)) {
	      max_drift[i] = drift;
	    }
	  }
	}
      }
    }

    drift = 0;
    for (i = 0; i < 81; i++) {
      if (fabs(drift) < fabs(max_drift[i])) {
	drift = max_drift[i];
      }
    }
    drifts_out[index] = drift;
  }
}

static double get_drift_fc4_elem(const double *fc4,
				 const int num_atom,
				 const int i,
				 const int j,
				 const int k,
				 const int l,
				 const int index)
{
  double sum;
  int m;
  
  sum = 0;
  for (m = 0; m < num_atom; m++) {
    switch (index) {
    case 0:
      sum += fc4[m * num_atom * num_atom * num_atom * 81 +
		 j * num_atom * num_atom * 81 +
		 k * num_atom * 81 +
		 l * 81 + i];
      break;
    case 1:
      sum += fc4[j * num_atom * num_atom * num_atom * 81 +
		 m * num_atom * num_atom * 81 +
		 k * num_atom * 81 +
		 l * 81 + i];
      break;
    case 2:
      sum += fc4[j * num_atom * num_atom * num_atom * 81 +
		 k * num_atom * num_atom * 81 +
		 m * num_atom * 81 +			 
		 l * 81 + i];
      break;
    case 3:
      sum += fc4[j * num_atom * num_atom * num_atom * 81 +
		 k * num_atom * num_atom * 81 +
		 l * num_atom * 81 +
		 m * 81 + i];
      break;
    }
  }

  return sum;
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

