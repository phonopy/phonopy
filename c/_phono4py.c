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

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "phonoc_array.h"
#include "phonon4_h/fc4.h"
#include "phonon4_h/real_to_reciprocal.h"
#include "phonon4_h/frequency_shift.h"

static PyObject * py_get_fc4_normal_for_frequency_shift(PyObject *self, PyObject *args);
static PyObject * py_get_fc4_frequency_shifts(PyObject *self, PyObject *args);
static PyObject * py_real_to_reciprocal4(PyObject *self, PyObject *args);
static PyObject * py_reciprocal_to_normal4(PyObject *self, PyObject *args);
static PyObject * py_set_phonons_grid_points(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc4(PyObject *self, PyObject *args);
static PyObject * py_rotate_delta_fc3s_elem(PyObject *self, PyObject *args);
static PyObject * py_set_translational_invariance_fc4(PyObject *self,
						      PyObject *args);
static PyObject * py_set_permutation_symmetry_fc4(PyObject *self,
						  PyObject *args);
static PyObject * py_get_drift_fc4(PyObject *self, PyObject *args);

static PyMethodDef functions[] = {
  {"fc4_normal_for_frequency_shift", py_get_fc4_normal_for_frequency_shift, METH_VARARGS, "Calculate fc4 normal for frequency shift"},
  {"fc4_frequency_shifts", py_get_fc4_frequency_shifts, METH_VARARGS, "Calculate fc4 frequency shift"},
  {"real_to_reciprocal4", py_real_to_reciprocal4, METH_VARARGS, "Transform fc4 of real space to reciprocal space"},
  {"reciprocal_to_normal4", py_reciprocal_to_normal4, METH_VARARGS, "Transform fc4 of reciprocal space to normal coordinate in special case for frequency shift"},
  {"phonons_grid_points", py_set_phonons_grid_points, METH_VARARGS, "Set phonons on grid points"},
  {"distribute_fc4", py_distribute_fc4, METH_VARARGS, "Distribute least fc4 to full fc4"},
  {"rotate_delta_fc3s_elem", py_rotate_delta_fc3s_elem, METH_VARARGS, "Rotate delta fc3s for a set of atomic indices"},
  {"translational_invariance_fc4", py_set_translational_invariance_fc4, METH_VARARGS, "Set translational invariance for fc4"},
  {"permutation_symmetry_fc4", py_set_permutation_symmetry_fc4, METH_VARARGS, "Set permutation symmetry for fc4"},
  {"drift_fc4", py_get_drift_fc4, METH_VARARGS, "Get drifts of fc4"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_phono4py(void)
{
  Py_InitModule3("_phono4py", functions, "C-extension for phono4py\n\n...\n");
  return;
}

static PyObject * py_get_fc4_normal_for_frequency_shift(PyObject *self,
							PyObject *args)
{
  PyArrayObject* fc4_normal_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* grid_points1_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* fc4_py;
  PyArrayObject* shortest_vectors_py;
  PyArrayObject* multiplicity_py;
  PyArrayObject* masses_py;
  PyArrayObject* p2s_map_py;
  PyArrayObject* s2p_map_py;
  PyArrayObject* band_indicies_py;
  double cutoff_frequency;
  int grid_point0;

  if (!PyArg_ParseTuple(args, "OOOiOOOOOOOOOOd",
			&fc4_normal_py,
			&frequencies_py,
			&eigenvectors_py,
			&grid_point0,
			&grid_points1_py,
			&grid_address_py,
			&mesh_py,
			&fc4_py,
			&shortest_vectors_py,
			&multiplicity_py,
			&masses_py,
			&p2s_map_py,
			&s2p_map_py,
			&band_indicies_py,
			&cutoff_frequency)) {
    return NULL;
  }

  double* fc4_normal = (double*)fc4_normal_py->data;
  double* freqs = (double*)frequencies_py->data;
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  lapack_complex_double* eigvecs =
    (lapack_complex_double*)eigenvectors_py->data;
  Iarray* grid_points1 = convert_to_iarray(grid_points1_py);
  const int* grid_address = (int*)grid_address_py->data;
  const int* mesh = (int*)mesh_py->data;
  double* fc4 = (double*)fc4_py->data;
  Darray* svecs = convert_to_darray(shortest_vectors_py);
  Iarray* multi = convert_to_iarray(multiplicity_py);
  const double* masses = (double*)masses_py->data;
  const int* p2s = (int*)p2s_map_py->data;
  const int* s2p = (int*)s2p_map_py->data;
  Iarray* band_indicies = convert_to_iarray(band_indicies_py);

  get_fc4_normal_for_frequency_shift(fc4_normal,
				     freqs,
				     eigvecs,
				     grid_point0,
				     grid_points1,
				     grid_address,
				     mesh,
				     fc4,
				     svecs,
				     multi,
				     masses,
				     p2s,
				     s2p,
				     band_indicies,
				     cutoff_frequency);

  free(grid_points1);
  free(svecs);
  free(multi);
  free(band_indicies);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_fc4_frequency_shifts(PyObject *self, PyObject *args)
{
  PyArrayObject* frequency_shifts_py;
  PyArrayObject* fc4_normal_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_points1_py;
  PyArrayObject* temperatures_py;
  PyArrayObject* band_indicies_py;
  double unit_conversion_factor;

  if (!PyArg_ParseTuple(args, "OOOOOOd",
			&frequency_shifts_py,
			&fc4_normal_py,
			&frequencies_py,
			&grid_points1_py,
			&temperatures_py,
			&band_indicies_py,
			&unit_conversion_factor)) {
    return NULL;
  }

  double* freq_shifts = (double*)frequency_shifts_py->data;
  double* fc4_normal = (double*)fc4_normal_py->data;
  double* freqs = (double*)frequencies_py->data;
  Iarray* grid_points1 = convert_to_iarray(grid_points1_py);
  Darray* temperatures = convert_to_darray(temperatures_py);
  int* band_indicies = (int*)band_indicies_py->data;
  const int num_band0 = (int)band_indicies_py->dimensions[0];
  const int num_band = (int)frequencies_py->dimensions[1];

  get_fc4_frequency_shifts(freq_shifts,
			   fc4_normal,
			   freqs,
			   grid_points1,
			   temperatures,
			   band_indicies,
			   num_band0,
			   num_band,
			   unit_conversion_factor);

  free(grid_points1);
  free(temperatures);
  
  Py_RETURN_NONE;
}

static PyObject * py_real_to_reciprocal4(PyObject *self, PyObject *args)
{
  PyArrayObject* fc4_py;
  PyArrayObject* fc4_reciprocal_py;
  PyArrayObject* q_py;
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;

  if (!PyArg_ParseTuple(args, "OOOOOOO",
			&fc4_reciprocal_py,
			&fc4_py,
			&q_py,
			&shortest_vectors,
			&multiplicity,
			&p2s_map,
			&s2p_map)) {
    return NULL;
  }

  double* fc4 = (double*)fc4_py->data;
  lapack_complex_double* fc4_reciprocal =
    (lapack_complex_double*)fc4_reciprocal_py->data;
  Darray* svecs = convert_to_darray(shortest_vectors);
  Iarray* multi = convert_to_iarray(multiplicity);
  const int* p2s = (int*)p2s_map->data;
  const int* s2p = (int*)s2p_map->data;
  const double* q = (double*)q_py->data;

  real_to_reciprocal4(fc4_reciprocal,
		      q,
		      fc4,
		      svecs,
		      multi,
		      p2s,
		      s2p);

  free(svecs);
  free(multi);
  
  Py_RETURN_NONE;
}

static PyObject * py_reciprocal_to_normal4(PyObject *self, PyObject *args)
{
  PyArrayObject* fc4_normal_py;
  PyArrayObject* fc4_reciprocal_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* masses_py;
  PyArrayObject* band_indicies_py;
  double cutoff_frequency;

  if (!PyArg_ParseTuple(args, "OOOOOOOd",
			&fc4_normal_py,
			&fc4_reciprocal_py,
			&frequencies_py,
			&eigenvectors_py,
			&grid_points_py,
			&masses_py,
			&band_indicies_py,
			&cutoff_frequency)) {
    return NULL;
  }

  lapack_complex_double* fc4_normal =
    (lapack_complex_double*)fc4_normal_py->data;
  const lapack_complex_double* fc4_reciprocal =
    (lapack_complex_double*)fc4_reciprocal_py->data;
  const lapack_complex_double* eigenvectors =
    (lapack_complex_double*)eigenvectors_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* grid_points = (int*)grid_points_py->data;
  const double* masses = (double*)masses_py->data;
  const int* band_indices = (int*)band_indicies_py->data;
  const int num_band0 = (int)band_indicies_py->dimensions[0];
  const int num_band = (int)frequencies_py->dimensions[1];

  reciprocal_to_normal4(fc4_normal,
			fc4_reciprocal,
			frequencies + grid_points[0] * num_band,
			frequencies + grid_points[1] * num_band,
			eigenvectors + grid_points[0] * num_band * num_band,
			eigenvectors + grid_points[1] * num_band * num_band,
			masses,
			band_indices,
			num_band0,
			num_band,
			cutoff_frequency);

  Py_RETURN_NONE;
}

static PyObject * py_set_phonons_grid_points(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* phonon_done_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors_fc2;
  PyArrayObject* multiplicity_fc2;
  PyArrayObject* fc2_py;
  PyArrayObject* atomic_masses_fc2;
  PyArrayObject* p2s_map_fc2;
  PyArrayObject* s2p_map_fc2;
  PyArrayObject* reciprocal_lattice;
  PyArrayObject* born_effective_charge;
  PyArrayObject* q_direction;
  PyArrayObject* dielectric_constant;
  double nac_factor, unit_conversion_factor;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOdOOOOdc",
			&frequencies,
			&eigenvectors,
			&phonon_done_py,
			&grid_points_py,
			&grid_address_py,
			&mesh_py,
			&fc2_py,
			&shortest_vectors_fc2,
			&multiplicity_fc2,
			&atomic_masses_fc2,
			&p2s_map_fc2,
			&s2p_map_fc2,
			&unit_conversion_factor,
			&born_effective_charge,
			&dielectric_constant,
			&reciprocal_lattice,
			&q_direction,
			&nac_factor,
			&uplo)) {
    return NULL;
  }

  double* born;
  double* dielectric;
  double *q_dir;
  Darray* freqs = convert_to_darray(frequencies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  Carray* eigvecs = convert_to_carray(eigenvectors);
  char* phonon_done = (char*)phonon_done_py->data;
  Iarray* grid_points = convert_to_iarray(grid_points_py);
  const int* grid_address = (int*)grid_address_py->data;
  const int* mesh = (int*)mesh_py->data;
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs_fc2 = convert_to_darray(shortest_vectors_fc2);
  Iarray* multi_fc2 = convert_to_iarray(multiplicity_fc2);
  const double* masses_fc2 = (double*)atomic_masses_fc2->data;
  const int* p2s_fc2 = (int*)p2s_map_fc2->data;
  const int* s2p_fc2 = (int*)s2p_map_fc2->data;
  const double* rec_lat = (double*)reciprocal_lattice->data;
  if ((PyObject*)born_effective_charge == Py_None) {
    born = NULL;
  } else {
    born = (double*)born_effective_charge->data;
  }
  if ((PyObject*)dielectric_constant == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)dielectric_constant->data;
  }
  if ((PyObject*)q_direction == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)q_direction->data;
  }

  set_phonons_for_frequency_shift(freqs,
				  eigvecs,
				  phonon_done,
				  grid_points,
				  grid_address,
				  mesh,
				  fc2,
				  svecs_fc2,
				  multi_fc2,
				  masses_fc2,
				  p2s_fc2,
				  s2p_fc2,
				  unit_conversion_factor,
				  born,
				  dielectric,
				  rec_lat,
				  q_dir,
				  nac_factor,
				  uplo);

  free(freqs);
  free(eigvecs);
  free(grid_points);
  free(fc2);
  free(svecs_fc2);
  free(multi_fc2);
  
  Py_RETURN_NONE;
}

static PyObject * py_distribute_fc4(PyObject *self, PyObject *args)
{
  PyArrayObject* fc4_copy_py;
  PyArrayObject* fc4_py;
  int fourth_atom;
  PyArrayObject* rotation_cart_inv;
  PyArrayObject* atom_mapping_py;

  if (!PyArg_ParseTuple(args, "OOiOO",
			&fc4_copy_py,
			&fc4_py,
			&fourth_atom,
			&atom_mapping_py,
			&rotation_cart_inv)) {
    return NULL;
  }

  double* fc4_copy = (double*)fc4_copy_py->data;
  const double* fc4 = (double*)fc4_py->data;
  const double* rot_cart_inv = (double*)rotation_cart_inv->data;
  const int* atom_mapping = (int*)atom_mapping_py->data;
  const int num_atom = (int)atom_mapping_py->dimensions[0];

  return PyInt_FromLong((long) distribute_fc4(fc4_copy,
					      fc4,
					      fourth_atom,
					      atom_mapping,
					      num_atom,
					      rot_cart_inv));
}

static PyObject * py_rotate_delta_fc3s_elem(PyObject *self, PyObject *args)
{
  PyArrayObject* rotated_delta_fc3s_py;
  PyArrayObject* delta_fc3s_py;
  PyArrayObject* atom_mappings_of_rotations_py;
  PyArrayObject* site_symmetries_cartesian_py;
  int atom1, atom2, atom3;

  if (!PyArg_ParseTuple(args, "OOOOiii",
			&rotated_delta_fc3s_py,
			&delta_fc3s_py,
			&atom_mappings_of_rotations_py,
			&site_symmetries_cartesian_py,
			&atom1,
			&atom2,
			&atom3)) {
    return NULL;
  }

  double* rotated_delta_fc3s = (double*)rotated_delta_fc3s_py->data;
  const double* delta_fc3s = (double*)delta_fc3s_py->data;
  const int* rot_map_syms = (int*)atom_mappings_of_rotations_py->data;
  const double* site_syms_cart = (double*)site_symmetries_cartesian_py->data;
  const int num_rot = (int)site_symmetries_cartesian_py->dimensions[0];
  const int num_delta_fc3s = (int)delta_fc3s_py->dimensions[0];
  const int num_atom = (int)delta_fc3s_py->dimensions[1];

  return PyInt_FromLong((long) rotate_delta_fc3s_elem(rotated_delta_fc3s,
						      delta_fc3s,
						      rot_map_syms,
						      site_syms_cart,
						      num_rot,
						      num_delta_fc3s,
						      atom1,
						      atom2,
						      atom3,
						      num_atom));
}

static PyObject * py_set_translational_invariance_fc4(PyObject *self,
						      PyObject *args)
{
  PyArrayObject* fc4_py;
  int index;

  if (!PyArg_ParseTuple(args, "Oi",
			&fc4_py,
			&index)) {
    return NULL;
  }

  double* fc4 = (double*)fc4_py->data;
  const int num_atom = (int)fc4_py->dimensions[0];

  set_translational_invariance_fc4_per_index(fc4, num_atom, index);

  Py_RETURN_NONE;
}

static PyObject * py_set_permutation_symmetry_fc4(PyObject *self, PyObject *args)
{
  PyArrayObject* fc4_py;

  if (!PyArg_ParseTuple(args, "O",
			&fc4_py)) {
    return NULL;
  }

  double* fc4 = (double*)fc4_py->data;
  const int num_atom = (int)fc4_py->dimensions[0];

  set_permutation_symmetry_fc4(fc4, num_atom);

  Py_RETURN_NONE;
}

static PyObject * py_get_drift_fc4(PyObject *self, PyObject *args)
{
  PyArrayObject* fc4_py;

  if (!PyArg_ParseTuple(args, "O",
			&fc4_py)) {
    return NULL;
  }

  double* fc4 = (double*)fc4_py->data;
  const int num_atom = (int)fc4_py->dimensions[0];

  int i;
  double drift[4];
  PyObject* drift_py;

  get_drift_fc4(drift, fc4, num_atom);
  drift_py = PyList_New(4);

  for (i = 0; i < 4; i++) {
    PyList_SetItem(drift_py, i, PyFloat_FromDouble(drift[i]));
  }

  return drift_py;
}
