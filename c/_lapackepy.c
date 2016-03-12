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
#include <assert.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include <lapack_wrapper.h>
#include <phonon.h>
#include <phonoc_array.h>

static PyObject * py_set_phonons_at_gridpoints(PyObject *self, PyObject *args);
static PyObject * py_get_phonons_at_qpoints(PyObject *self, PyObject *args);
static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);

struct module_state {
  PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
  struct module_state *st = GETSTATE(m);
  PyErr_SetString(st->error, "something bad happened");
  return NULL;
}

static PyMethodDef _lapackepy_methods[] = {
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {"phonons_at_gridpoints", py_set_phonons_at_gridpoints, METH_VARARGS,
   "Set phonons at grid points"},
  {"phonons_at_qpoints", py_get_phonons_at_qpoints, METH_VARARGS,
   "Get phonons at a q-point"},
  {"pinv", py_phonopy_pinv, METH_VARARGS, "Pseudo-inverse using Lapack dgesvd"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int _lapackepy_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int _lapackepy_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_lapackepy",
  NULL,
  sizeof(struct module_state),
  _lapackepy_methods,
  NULL,
  _lapackepy_traverse,
  _lapackepy_clear,
  NULL
};

#define INITERROR return NULL

PyObject *
PyInit__lapackepy(void)

#else
#define INITERROR return

  void
  init_lapackepy(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("_lapackepy", _lapackepy_methods);
#endif

  if (module == NULL)
    INITERROR;
  struct module_state *st = GETSTATE(module);

  st->error = PyErr_NewException("_lapackepy.Error", NULL, NULL);
  if (st->error == NULL) {
    Py_DECREF(module);
    INITERROR;
  }

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}

static PyObject * py_set_phonons_at_gridpoints(PyObject *self, PyObject *args)
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
  char* uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOdOOOOds",
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
  char* phonon_done = (char*)PyArray_DATA(phonon_done_py);
  Iarray* grid_points = convert_to_iarray(grid_points_py);
  const int* grid_address = (int*)PyArray_DATA(grid_address_py);
  const int* mesh = (int*)PyArray_DATA(mesh_py);
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs_fc2 = convert_to_darray(shortest_vectors_fc2);
  Iarray* multi_fc2 = convert_to_iarray(multiplicity_fc2);
  const double* masses_fc2 = (double*)PyArray_DATA(atomic_masses_fc2);
  const int* p2s_fc2 = (int*)PyArray_DATA(p2s_map_fc2);
  const int* s2p_fc2 = (int*)PyArray_DATA(s2p_map_fc2);
  const double* rec_lat = (double*)PyArray_DATA(reciprocal_lattice);
  if ((PyObject*)born_effective_charge == Py_None) {
    born = NULL;
  } else {
    born = (double*)PyArray_DATA(born_effective_charge);
  }
  if ((PyObject*)dielectric_constant == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)PyArray_DATA(dielectric_constant);
  }
  if ((PyObject*)q_direction == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)PyArray_DATA(q_direction);
  }

  set_phonons_at_gridpoints(freqs,
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
			    uplo[0]);

  free(freqs);
  free(eigvecs);
  free(grid_points);
  free(fc2);
  free(svecs_fc2);
  free(multi_fc2);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_phonons_at_qpoints(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* qpoints_py;
  PyArrayObject* shortest_vectors_py;
  PyArrayObject* multiplicity_py;
  PyArrayObject* fc2_py;
  PyArrayObject* atomic_masses_py;
  PyArrayObject* p2s_map_py;
  PyArrayObject* s2p_map_py;
  PyArrayObject* reciprocal_lattice_py;
  PyArrayObject* born_effective_charge_py;
  PyArrayObject* q_direction_py;
  PyArrayObject* dielectric_constant_py;
  double nac_factor, unit_conversion_factor;
  char* uplo;

  if (sizeof(lapack_complex_double) != sizeof(npy_complex128)) {
    printf("***********************************************************\n");
    printf("* sizeof(lapack_complex_double) != sizeof(npy_complex128) *\n");
    printf("* Please report this problem to atz.togo@gmail.com        *\n");
    printf("***********************************************************\n");
    return NULL;
  }

  if (!PyArg_ParseTuple(args, "OOOOOOOOOdOOOOds",
			&frequencies_py,
			&eigenvectors_py,
			&qpoints_py,
			&fc2_py,
			&shortest_vectors_py,
			&multiplicity_py,
			&atomic_masses_py,
			&p2s_map_py,
			&s2p_map_py,
			&unit_conversion_factor,
			&born_effective_charge_py,
			&dielectric_constant_py,
			&reciprocal_lattice_py,
			&q_direction_py,
			&nac_factor,
			&uplo)) {
    return NULL;
  }

  int i;
  double* born;
  double* dielectric;
  double *q_dir;
  double* freqs = (double*)PyArray_DATA(frequencies_py);
  const int num_band = PyArray_DIMS(frequencies_py)[1];
  lapack_complex_double* eigvecs =
    (lapack_complex_double*)PyArray_DATA(eigenvectors_py);
  double (*qpoints)[3] = (double(*)[3])PyArray_DATA(qpoints_py);
  const int num_q = PyArray_DIMS(qpoints_py)[0];
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs = convert_to_darray(shortest_vectors_py);
  Iarray* multi = convert_to_iarray(multiplicity_py);
  const double* masses = (double*)PyArray_DATA(atomic_masses_py);
  const int* p2s = (int*)PyArray_DATA(p2s_map_py);
  const int* s2p = (int*)PyArray_DATA(s2p_map_py);
  const double* rec_lat = (double*)PyArray_DATA(reciprocal_lattice_py);

  if ((PyObject*)born_effective_charge_py == Py_None) {
    born = NULL;
  } else {
    born = (double*)PyArray_DATA(born_effective_charge_py);
  }
  if ((PyObject*)dielectric_constant_py == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)PyArray_DATA(dielectric_constant_py);
  }
  if ((PyObject*)q_direction_py == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)PyArray_DATA(q_direction_py);
  }

#pragma omp parallel for
  for (i = 0; i < num_q; i++) {
    get_phonons(eigvecs + num_band * num_band * i,
		freqs + num_band * i,
		qpoints[i],
		fc2,
		masses,
		p2s,
		s2p,
		multi,
		svecs,
		born,
		dielectric,
		rec_lat,
		q_dir,
		nac_factor,
		unit_conversion_factor,
		uplo[0]);
  }

  free(fc2);
  free(svecs);
  free(multi);
  
  Py_RETURN_NONE;
}

static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args)
{
  PyArrayObject* dynamical_matrix;
  PyArrayObject* eigenvalues;

  if (!PyArg_ParseTuple(args, "OO",
			&dynamical_matrix,
			&eigenvalues)) {
    return NULL;
  }

  const int dimension = (int)PyArray_DIMS(dynamical_matrix)[0];
  npy_cdouble *dynmat = (npy_cdouble*)PyArray_DATA(dynamical_matrix);
  double *eigvals = (double*)PyArray_DATA(eigenvalues);

  lapack_complex_double *a;
  int i, info;

  a = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) *
				      dimension * dimension);
  for (i = 0; i < dimension * dimension; i++) {
    a[i] = lapack_make_complex_double(dynmat[i].real, dynmat[i].imag);
  }

  info = phonopy_zheev(eigvals, a, dimension, 'L');

  for (i = 0; i < dimension * dimension; i++) {
    dynmat[i].real = lapack_complex_double_real(a[i]);
    dynmat[i].imag = lapack_complex_double_imag(a[i]);
  }

  free(a);
  
  return PyLong_FromLong((long) info);
}

static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args)
{
  PyArrayObject* data_in_py;
  PyArrayObject* data_out_py;
  double cutoff;

  if (!PyArg_ParseTuple(args, "OOd",
			&data_out_py,
			&data_in_py,
			&cutoff)) {
    return NULL;
  }

  const int m = (int)PyArray_DIMS(data_in_py)[0];
  const int n = (int)PyArray_DIMS(data_in_py)[1];
  const double *data_in = (double*)PyArray_DATA(data_in_py);
  double *data_out = (double*)PyArray_DATA(data_out_py);
  int info;
  
  info = phonopy_pinv(data_out, data_in, m, n, cutoff);

  return PyLong_FromLong((long) info);
}
