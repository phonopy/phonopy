/* Copyright (C) 2011 Atsushi Togo */
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

#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <numpy/arrayobject.h>

#define KB 8.6173382568083159E-05

/* Build dynamical matrix */
static PyObject * py_get_dynamical_matrix(PyObject *self, PyObject *args);
static PyObject * py_get_thermal_properties(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc2(PyObject *self, PyObject *args);
static PyObject * py_get_rotated_forces(PyObject *self, PyObject *args);

static int get_dynamical_matrix_at_q(double *dynamical_matrix_real,
				     double *dynamical_matrix_image,
				     const int d_prim, const int d_super,
				     const double *force_constants, 
				     const double *q,
				     const double *r,
				     const int *multi,
				     const double *mass,
				     const int *s2p_map, 
				     const int *p2s_map);
static double get_free_energy_omega(const double temperature,
				    const double omega);
static double get_entropy_omega(const double temperature,
				const double omega);
static double get_heat_capacity_omega(const double temperature,
				      const double omega);
/* static double get_energy_omega(double temperature, double omega); */
static int distribute_fc2(double * fc2,
			  const double * pos,
			  const int num_pos,
			  const int atom_disp,
			  const int map_atom_disp,
			  const double * r_cart,
			  int r[3][3],
			  const double * t,
			  const double symprec);
static int get_rotated_forces(double * rotated_forces,
			      const double * pos,
			      const int num_pos,
			      const int atom_number,
			      const double * f,
			      int (*r)[3][3],
			      const int num_rot,
			      const double symprec);
static int nint(const double a);

static PyMethodDef functions[] = {
  {"dynamical_matrix", py_get_dynamical_matrix, METH_VARARGS, "Dynamical matrix"},
  {"thermal_properties", py_get_thermal_properties, METH_VARARGS, "Thermal properties"},
  {"distribute_fc2", py_distribute_fc2, METH_VARARGS, "Distribute force constants"},
  {"rotated_forces", py_get_rotated_forces, METH_VARARGS, "Rotate forces following site-symmetry"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_phonopy(void)
{
  Py_InitModule3("_phonopy", functions, "C-extension for phonopy\n\n...\n");
  return;
}

static PyObject * py_get_dynamical_matrix(PyObject *self, PyObject *args)
{
  PyArrayObject* dynamical_matrix_real;
  PyArrayObject* dynamical_matrix_image;
  PyArrayObject* force_constants;
  PyArrayObject* r_vector;
  PyArrayObject* q_vector;
  PyArrayObject* multiplicity;
  PyArrayObject* mass;
  PyArrayObject* s2p_map;
  PyArrayObject* p2s_map;

  if (!PyArg_ParseTuple(args, "OOOOOOOOO", &dynamical_matrix_real,
			&dynamical_matrix_image,
			&force_constants, &q_vector,
			&r_vector,
			&multiplicity,
			&mass,
			&s2p_map,
			&p2s_map))
    return NULL;

  int i;
  double* dm_r = (double*)dynamical_matrix_real->data;
  double* dm_i = (double*)dynamical_matrix_image->data;
  const double* fc = (double*)force_constants->data;
  const double* q = (double*)q_vector->data;
  const double* r = (double*)r_vector->data;
  const double* m = (double*)mass->data;
  const long* multi_long = (long*)multiplicity->data;
  const long* s2p_map_long = (long*)s2p_map->data;
  const long* p2s_map_long = (long*)p2s_map->data;
  const int d_prim = p2s_map->dimensions[0];
  const int d_super = s2p_map->dimensions[0];

  int *multi, *s2p_map_int, *p2s_map_int;

  multi = (int*) malloc(d_prim * d_super * sizeof(int));
  for (i = 0; i < d_prim*d_super; i++) {
    multi[i] = (int)multi_long[i];
  }

  s2p_map_int = (int*) malloc(d_super * sizeof(int));
  for (i = 0; i < d_super; i++) {
    s2p_map_int[i] = (int)s2p_map_long[i];
  }

  p2s_map_int = (int*) malloc(d_prim * sizeof(int));
  for (i = 0; i < d_prim; i++) {
    p2s_map_int[i] = (int)p2s_map_long[i];
  }

  get_dynamical_matrix_at_q(dm_r,
			    dm_i,
			    d_prim,
			    d_super,
			    fc,
			    q,
			    r,
			    multi,
			    m,
			    s2p_map_int,
			    p2s_map_int);

  free(multi);
  free(s2p_map_int);
  free(p2s_map_int);

  Py_RETURN_NONE;
}

static int get_dynamical_matrix_at_q(double *dynamical_matrix_real,
				     double *dynamical_matrix_image,
				     const int d_prim, 
				     const int d_super,
				     const double *force_constants,
				     const double *q,
				     const double *r,
				     const int *multi,
				     const double *mass,
				     const int *s2p_map, 
				     const int *p2s_map)
{
  int i, j, k, l, m;
  double phase, cos_phase, sin_phase, mass_sqrt;
  double dm_real[3][3], dm_imag[3][3];
 

#pragma omp parallel for private(j, k, l, m, phase, cos_phase, sin_phase, mass_sqrt, dm_real, dm_imag) 
  for (i = 0; i < d_prim; i++) { /* left index of dm */

    for (j = 0; j < d_prim; j++) { /* right index of dm */
      mass_sqrt = sqrt(mass[i] * mass[j]);

      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  dm_real[k][l] = 0;
	  dm_imag[k][l] = 0;
	}
      }

      for (k = 0; k < d_super; k++) { /* Lattice points of right index of fc */
	if (! (s2p_map[k] == p2s_map[j])) {
	  continue;
	}

	cos_phase = 0;
	sin_phase = 0;
	for (l = 0; l < multi[ k * d_prim + i ]; l++) {
	  phase = 0;
	  for (m = 0; m < 3; m++) {
	    phase += q[m] * r[  k * d_prim*81 + i*81 + l*3 + m ];
	  }
	  cos_phase += cos(phase * 2 * M_PI) / multi[ k * d_prim + i ];
	  sin_phase += sin(phase * 2 * M_PI) / multi[ k * d_prim + i ];
	}

	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    dm_real[l][m] += force_constants[ p2s_map[i] * d_super*9 + k*9 + l*3 + m ] * cos_phase / mass_sqrt;
	    dm_imag[l][m] += force_constants[ p2s_map[i] * d_super*9 + k*9 + l*3 + m ] * sin_phase / mass_sqrt;
	  }
	}
      }
      
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  dynamical_matrix_real[ (i*3 + k) * d_prim * 3 + j*3 + l ] += dm_real[k][l];
	  dynamical_matrix_image[ (i*3 + k) * d_prim * 3 + j*3 + l ] += dm_imag[k][l];
	}
      }
    }
  }

  return 0;
}


/* Thermal properties */
static PyObject * py_get_thermal_properties(PyObject *self, PyObject *args)
{
  double temperature;
  double cutoff; /* Cutoff eigenvalue not to be calculated */
  PyArrayObject* frequencies;
  PyArrayObject* weights;

  if (!PyArg_ParseTuple(args, "dOO",
			&temperature,
			&frequencies,
			&weights)) {
    return NULL;
  }

  const double* freqs = (double*)frequencies->data;
  const long* w = (long*)weights->data;
  const int num_qpoints = frequencies->dimensions[0];
  const int num_bands = frequencies->dimensions[1];

  int i, j;
  long sum_weights = 0;
  double free_energy = 0;
  double entropy = 0;
  double heat_capacity = 0;
  double omega = 0;

  for (i = 0; i < num_qpoints; i++){
    sum_weights += w[i];
    for (j = 0; j < num_bands; j++){
      omega = freqs[i * num_bands + j];
      if (omega > 0.0) {
	free_energy += get_free_energy_omega(temperature, omega) * w[i];
	entropy += get_entropy_omega(temperature, omega) * w[i];
	heat_capacity += get_heat_capacity_omega(temperature, omega)* w[i];
      }
    }
  }

  return PyTuple_Pack(3,
		      PyFloat_FromDouble(free_energy / sum_weights), 
		      PyFloat_FromDouble(entropy / sum_weights),
		      PyFloat_FromDouble(heat_capacity / sum_weights));
}

static double get_free_energy_omega(const double temperature,
				    const double omega)
{
  /* temperature is defined by T (K) */
  /* omega must be normalized to eV. */
  return KB * temperature * log(1 - exp(- omega / (KB * temperature)));
}

static double get_entropy_omega(const double temperature,
				const double omega)
{
  /* temperature is defined by T (K) */
  /* omega must be normalized to eV. */
  double val;

  val = omega / (2 * KB * temperature);
  return 1 / (2 * temperature) * omega * cosh(val) / sinh(val) - KB * log(2 * sinh(val));
}

static double get_heat_capacity_omega(const double temperature,
				      const double omega)
{
  /* temperature is defined by T (K) */
  /* omega must be normalized to eV. */
  /* If val is close to 1. Then expansion is used. */
  double val, val1, val2;

  val = omega / (KB * temperature);
  val1 = exp(val);
  val2 = (val) / (val1 - 1);
  return KB * val1 * val2 * val2;
}

/* static double get_energy_omega(double temperature, double omega){ */
/*   /\* temperature is defined by T (K) *\/ */
/*   /\* omega must be normalized to eV. *\/ */
/*   return omega / (exp(omega / (KB * temperature)) - 1); */
/* } */


static PyObject * py_distribute_fc2(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants;
  PyArrayObject* positions;
  PyArrayObject* rotation;
  PyArrayObject* rotation_cart;
  PyArrayObject* translation;
  int atom_disp, map_atom_disp;
  double symprec;

  if (!PyArg_ParseTuple(args, "OOiiOOOd",
			&force_constants,
			&positions,
			&atom_disp,
			&map_atom_disp,
			&rotation_cart,
			&rotation,
			&translation,
			&symprec)) {
    return NULL;
  }

  const long* r_long = (long*)rotation->data;
  const double* r_cart = (double*)rotation_cart->data;
  double* fc2 = (double*)force_constants->data;
  const double* t = (double*)translation->data;
  const double* pos = (double*)positions->data;
  const int num_pos = positions->dimensions[0];


  int i, j;
  int r[3][3];

  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      r[i][j] = (int) r_long[ i * 3 + j ];
    }
  }
  distribute_fc2(fc2,
		 pos,
		 num_pos,
		 atom_disp,
		 map_atom_disp,
		 r_cart,
		 r,
		 t,
		 symprec);

  Py_RETURN_NONE;
}

static int distribute_fc2(double * fc2,
			  const double * pos,
			  const int num_pos,
			  const int atom_disp,
			  const int map_atom_disp,
			  const double * r_cart,
			  int r[3][3],
			  const double * t,
			  const double symprec)
{
  int i, j, k, l, m, address_new, address;
  int is_found, rot_atom;
  double rot_pos[3], diff[3];

#pragma omp parallel for private(j, k, l, m, rot_pos, diff, is_found, rot_atom, address_new, address)
  for (i = 0; i < num_pos; i++) {
    for (j = 0; j < 3; j++) {
      rot_pos[ j ] = t[j];
      for (k = 0; k < 3; k++) {
	rot_pos[ j ] += r[ j ][ k ] * pos[ i * 3 + k ];
      }
    }

    for (j = 0; j < num_pos; j++) {
      is_found = 1;
      for (k = 0; k < 3; k++) {
	diff[ k ] = pos[ j * 3 + k ] - rot_pos[ k ];
	diff[ k ] -= nint(diff[k]);
	if (fabs(diff[ k ]) > symprec) {
	  is_found = 0;
	  break;
	}
      }
      if (is_found) {
	rot_atom = j;
	break;
      }
    }

    if (! is_found) {
      printf("Encounter some problem in distribute_fc2.\n");
      goto end;
    }

    /* R^-1 P R */
    address = map_atom_disp * num_pos * 9 + rot_atom * 9;
    address_new = atom_disp * num_pos * 9 + i * 9;
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  for (m = 0; m < 3; m++) {
	    fc2[ address_new + j * 3 + k ] +=
	      r_cart[ l * 3 + j ] * r_cart[ m * 3 + k ] *
	      fc2[ address + l * 3 + m ];
	  }
	}
      }
    }
  end:
    ;
  }

  return is_found;
}

static PyObject * py_get_rotated_forces(PyObject *self, PyObject *args)
{
  PyArrayObject* rotated_forces;
  PyArrayObject* positions;
  PyArrayObject* rotations;
  PyArrayObject* forces;
  int atom_number;
  double symprec;

  if (!PyArg_ParseTuple(args, "OOiOOd",
			&rotated_forces,
			&positions,
			&atom_number,
			&forces,
			&rotations,
			&symprec)) {
    return NULL;
  }

  const long* rots_long = (long*)rotations->data;
  const int num_rot = rotations->dimensions[0];
  const double* pos = (double*)positions->data;
  const int num_pos = positions->dimensions[0];
  const double* f = (double*)forces->data;
  double* rot_forces = (double*)rotated_forces->data;

  int i, j, k;
  int (*r)[3][3];
  if (! ((r = (int (*)[3][3]) malloc(sizeof(int[3][3]) * num_rot))  == NULL)) {
    for (i = 0; i < num_rot; i++){
      for (j = 0; j < 3; j++){
	for (k = 0; k < 3; k++){
	  r[i][j][k] = (int) rots_long[ i * 9 + j * 3 + k ];
	}
      }
    }
    get_rotated_forces(rot_forces,
		       pos,
		       num_pos,
		       atom_number,
		       f,
		       r,
		       num_rot,
		       symprec);

    free(r);
  }

  Py_RETURN_NONE;
}


static int get_rotated_forces(double * rotated_forces,
			      const double * pos,
			      const int num_pos,
			      const int atom_number,
			      const double * f,
			      int (*r)[3][3],
			      const int num_rot,
			      const double symprec)
{
  int i, j, k, is_found;
  double rot_pos[3], diff[3];

#pragma omp parallel for private(j, k, rot_pos, diff, is_found)
  for (i = 0; i < num_rot; i++) {
    for (j = 0; j < 3; j++) {
      rot_pos[ j ] = 0;
      for (k = 0; k < 3; k++) {
	rot_pos[ j ] += r[ i ][ j ][ k ] * pos[ atom_number * 3 + k ];
      }
    }
    
    for (j = 0; j < num_pos; j++) {
      is_found = 1;
      for (k = 0; k < 3; k++) {
	diff[ k ] = pos[ j * 3 + k ] - rot_pos[ k ];
	diff[ k ] -= nint(diff[k]);
	if (fabs(diff[ k ]) > symprec) {
	  is_found = 0;
	  break;
	}
      }
      if (is_found) {
	for (k = 0; k < 3; k++) {
	  rotated_forces[ i * 3 + k ] = f[ j * 3 + k ];
	}
	break;
      }
    }
    if (! is_found) {
      printf("Phonopy encounted symmetry problem (c)\n");
    }
  }

  return 1;
}
				 

static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}
