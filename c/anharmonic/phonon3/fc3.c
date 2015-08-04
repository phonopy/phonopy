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

#include <phonon3_h/fc3.h>

static double tensor3_rotation_elem(const double *tensor,
				    const double *r,
				    const int pos);
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

void distribute_fc3(double *fc3_copy,
		    const double *fc3,
		    const int third_atom,
		    const int *atom_mapping,
		    const int num_atom,
		    const double *rot_cart)
{
  int i, j;

  for (i = 0; i < num_atom; i++) {
    for (j = 0; j < num_atom; j++) {
      tensor3_rotation(fc3_copy +
		       27 * num_atom * num_atom * third_atom +
		       27 * num_atom * i +
		       27 * j,
		       fc3 +
		       27 * num_atom * num_atom * atom_mapping[third_atom] +
		       27 * num_atom * atom_mapping[i] +
		       27 * atom_mapping[j],
		       rot_cart);
    }
  }
}

void tensor3_rotation(double *rot_tensor,
		      const double *tensor,
		      const double *rot_cartesian)
{
  int l;

  for (l = 0; l < 27; l++) {
    rot_tensor[l] = tensor3_rotation_elem(tensor, rot_cartesian, l);
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

static double tensor3_rotation_elem(const double *tensor,
				    const double *r,
				    const int pos)
{
  int i, j, k, l, m, n;
  double sum;

  l = pos / 9;
  m = (pos % 9) / 3;
  n = pos % 3;

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
