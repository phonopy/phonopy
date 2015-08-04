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
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include <lapack_wrapper.h>
#include <phonoc_array.h>
#include <phonoc_utils.h>
#include <dynmat.h>
#include <derivative_dynmat.h>
#include <tetrahedron_method.h>

#define KB 8.6173382568083159E-05

/* Build dynamical matrix */
static PyObject * py_get_dynamical_matrix(PyObject *self, PyObject *args);
static PyObject * py_get_nac_dynamical_matrix(PyObject *self, PyObject *args);
static PyObject * py_get_derivative_dynmat(PyObject *self, PyObject *args);
static PyObject * py_get_thermal_properties(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc2(PyObject *self, PyObject *args);

static int distribute_fc2(double * fc2,
			  const double * pos,
			  const int num_pos,
			  const int atom_disp,
			  const int map_atom_disp,
			  const double * r_cart,
			  const int * r,
			  const double * t,
			  const double symprec);
static PyObject * py_get_neighboring_grid_points(PyObject *self, PyObject *args);
static PyObject *
py_get_tetrahedra_relative_grid_address(PyObject *self, PyObject *args);
static PyObject *
py_get_all_tetrahedra_relative_grid_address(PyObject *self, PyObject *args);
static PyObject *
py_get_tetrahedra_integration_weight(PyObject *self, PyObject *args);
static PyObject *
py_get_tetrahedra_integration_weight_at_omegas(PyObject *self, PyObject *args);
static PyObject * py_set_phonons_at_gridpoints(PyObject *self, PyObject *args);
static PyObject * py_get_phonon(PyObject *self, PyObject *args);
static PyObject * py_phonopy_pinv(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);

static double get_free_energy_omega(const double temperature,
				    const double omega);
static double get_entropy_omega(const double temperature,
				const double omega);
static double get_heat_capacity_omega(const double temperature,
				      const double omega);
/* static double get_energy_omega(double temperature, double omega); */
static int nint(const double a);

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

static PyMethodDef _phonopy_methods[] = {
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {"dynamical_matrix", py_get_dynamical_matrix, METH_VARARGS, "Dynamical matrix"},
  {"nac_dynamical_matrix", py_get_nac_dynamical_matrix, METH_VARARGS, "NAC dynamical matrix"},
  {"derivative_dynmat", py_get_derivative_dynmat, METH_VARARGS, "Q derivative of dynamical matrix"},
  {"thermal_properties", py_get_thermal_properties, METH_VARARGS, "Thermal properties"},
  {"distribute_fc2", py_distribute_fc2, METH_VARARGS, "Distribute force constants"},
  {"neighboring_grid_points", py_get_neighboring_grid_points,
   METH_VARARGS, "Neighboring grid points by relative grid addresses"},
  {"tetrahedra_relative_grid_address", py_get_tetrahedra_relative_grid_address,
   METH_VARARGS, "Relative grid addresses of vertices of 24 tetrahedra"},
  {"all_tetrahedra_relative_grid_address",
   py_get_all_tetrahedra_relative_grid_address,
   METH_VARARGS,
   "4 (all) sets of relative grid addresses of vertices of 24 tetrahedra"},
  {"tetrahedra_integration_weight", py_get_tetrahedra_integration_weight,
   METH_VARARGS, "Integration weight for tetrahedron method"},
  {"tetrahedra_integration_weight_at_omegas",
   py_get_tetrahedra_integration_weight_at_omegas,
   METH_VARARGS, "Integration weight for tetrahedron method at omegas"},
  {"phonons_at_gridpoints", py_set_phonons_at_gridpoints, METH_VARARGS,
   "Set phonons at grid points"},
  {"phonon", py_get_phonon, METH_VARARGS, "Get phonon"},
  {"pinv", py_phonopy_pinv, METH_VARARGS, "Pseudo-inverse using Lapack dgesvd"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int _phonopy_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int _phonopy_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_phonopy",
  NULL,
  sizeof(struct module_state),
  _phonopy_methods,
  NULL,
  _phonopy_traverse,
  _phonopy_clear,
  NULL
};

#define INITERROR return NULL

PyObject *
PyInit__phonopy(void)

#else
#define INITERROR return

  void
  init_phonopy(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("_phonopy", _phonopy_methods);
#endif

  if (module == NULL)
    INITERROR;
  struct module_state *st = GETSTATE(module);

  st->error = PyErr_NewException("_phonopy.Error", NULL, NULL);
  if (st->error == NULL) {
    Py_DECREF(module);
    INITERROR;
  }

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}


static PyObject * py_get_dynamical_matrix(PyObject *self, PyObject *args)
{
  PyArrayObject* dynamical_matrix_real;
  PyArrayObject* dynamical_matrix_imag;
  PyArrayObject* force_constants;
  PyArrayObject* r_vector;
  PyArrayObject* q_vector;
  PyArrayObject* multiplicity;
  PyArrayObject* mass;
  PyArrayObject* super2prim_map;
  PyArrayObject* prim2super_map;

  if (!PyArg_ParseTuple(args, "OOOOOOOOO",
			&dynamical_matrix_real,
			&dynamical_matrix_imag,
			&force_constants,
			&q_vector,
			&r_vector,
			&multiplicity,
			&mass,
			&super2prim_map,
			&prim2super_map))
    return NULL;

  double* dm_r = (double*)PyArray_DATA(dynamical_matrix_real);
  double* dm_i = (double*)PyArray_DATA(dynamical_matrix_imag);
  const double* fc = (double*)PyArray_DATA(force_constants);
  const double* q = (double*)PyArray_DATA(q_vector);
  const double* r = (double*)PyArray_DATA(r_vector);
  const double* m = (double*)PyArray_DATA(mass);
  const int* multi = (int*)PyArray_DATA(multiplicity);
  const int* s2p_map = (int*)PyArray_DATA(super2prim_map);
  const int* p2s_map = (int*)PyArray_DATA(prim2super_map);
  const int num_patom = PyArray_DIMS(prim2super_map)[0];
  const int num_satom = PyArray_DIMS(super2prim_map)[0];

  get_dynamical_matrix_at_q(dm_r,
			    dm_i,
			    num_patom,
			    num_satom,
			    fc,
			    q,
			    r,
			    multi,
			    m,
			    s2p_map,
			    p2s_map,
			    NULL);

  Py_RETURN_NONE;
}


static PyObject * py_get_nac_dynamical_matrix(PyObject *self, PyObject *args)
{
  PyArrayObject* dynamical_matrix_real;
  PyArrayObject* dynamical_matrix_imag;
  PyArrayObject* force_constants;
  PyArrayObject* r_vector;
  PyArrayObject* q_cart_vector;
  PyArrayObject* q_vector;
  PyArrayObject* multiplicity;
  PyArrayObject* mass;
  PyArrayObject* super2prim_map;
  PyArrayObject* prim2super_map;
  PyArrayObject* born;
  double factor;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOd",
			&dynamical_matrix_real,
			&dynamical_matrix_imag,
			&force_constants,
			&q_vector,
			&r_vector,
			&multiplicity,
			&mass,
			&super2prim_map,
			&prim2super_map,
			&q_cart_vector,
			&born,
			&factor))
    return NULL;

  double* dm_r = (double*)PyArray_DATA(dynamical_matrix_real);
  double* dm_i = (double*)PyArray_DATA(dynamical_matrix_imag);
  const double* fc = (double*)PyArray_DATA(force_constants);
  const double* q_cart = (double*)PyArray_DATA(q_cart_vector);
  const double* q = (double*)PyArray_DATA(q_vector);
  const double* r = (double*)PyArray_DATA(r_vector);
  const double* m = (double*)PyArray_DATA(mass);
  const double* z = (double*)PyArray_DATA(born);
  const int* multi = (int*)PyArray_DATA(multiplicity);
  const int* s2p_map = (int*)PyArray_DATA(super2prim_map);
  const int* p2s_map = (int*)PyArray_DATA(prim2super_map);
  const int num_patom = PyArray_DIMS(prim2super_map)[0];
  const int num_satom = PyArray_DIMS(super2prim_map)[0];

  int n;
  double *charge_sum;

  charge_sum = (double*) malloc(sizeof(double) * num_patom * num_patom * 9);
  n = num_satom / num_patom;

  get_charge_sum(charge_sum, num_patom, factor / n, q_cart, z);
  get_dynamical_matrix_at_q(dm_r,
			    dm_i,
			    num_patom,
			    num_satom,
			    fc,
			    q,
			    r,
			    multi,
			    m,
			    s2p_map,
			    p2s_map,
			    charge_sum);

  free(charge_sum);

  Py_RETURN_NONE;
}

static PyObject * py_get_derivative_dynmat(PyObject *self, PyObject *args)
{
  PyArrayObject* derivative_dynmat_real;
  PyArrayObject* derivative_dynmat_imag;
  PyArrayObject* force_constants;
  PyArrayObject* r_vector;
  PyArrayObject* lattice;
  PyArrayObject* q_vector;
  PyArrayObject* multiplicity;
  PyArrayObject* mass;
  PyArrayObject* super2prim_map;
  PyArrayObject* prim2super_map;
  PyArrayObject* born;
  PyArrayObject* dielectric;
  PyArrayObject* q_direction;
  double nac_factor;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOdOOO",
			&derivative_dynmat_real,
			&derivative_dynmat_imag,
			&force_constants,
			&q_vector,
			&lattice, /* column vectors */
			&r_vector,
			&multiplicity,
			&mass,
			&super2prim_map,
			&prim2super_map,
			&nac_factor,
			&born,
			&dielectric,
			&q_direction)) {
    return NULL;
  }

  double* ddm_r = (double*)PyArray_DATA(derivative_dynmat_real);
  double* ddm_i = (double*)PyArray_DATA(derivative_dynmat_imag);
  const double* fc = (double*)PyArray_DATA(force_constants);
  const double* q = (double*)PyArray_DATA(q_vector);
  const double* lat = (double*)PyArray_DATA(lattice);
  const double* r = (double*)PyArray_DATA(r_vector);
  const double* m = (double*)PyArray_DATA(mass);
  const int* multi = (int*)PyArray_DATA(multiplicity);
  const int* s2p_map = (int*)PyArray_DATA(super2prim_map);
  const int* p2s_map = (int*)PyArray_DATA(prim2super_map);
  const int num_patom = PyArray_DIMS(prim2super_map)[0];
  const int num_satom = PyArray_DIMS(super2prim_map)[0];
  double *z;
  double *epsilon;
  double *q_dir;
  if ((PyObject*)born == Py_None) {
    z = NULL;
  } else {
    z = (double*)PyArray_DATA(born);
  }
  if ((PyObject*)dielectric == Py_None) {
    epsilon = NULL;
  } else {
    epsilon = (double*)PyArray_DATA(dielectric);
  }
  if ((PyObject*)q_direction == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)PyArray_DATA(q_direction);
  }

  get_derivative_dynmat_at_q(ddm_r,
			     ddm_i,
			     num_patom,
			     num_satom,
			     fc,
			     q,
			     lat,
			     r,
			     multi,
			     m,
			     s2p_map,
			     p2s_map,
			     nac_factor,
			     z,
			     epsilon,
			     q_dir);

  Py_RETURN_NONE;
}

/* Thermal properties */
static PyObject * py_get_thermal_properties(PyObject *self, PyObject *args)
{
  double temperature;
  PyArrayObject* frequencies;
  PyArrayObject* weights;

  if (!PyArg_ParseTuple(args, "dOO",
			&temperature,
			&frequencies,
			&weights)) {
    return NULL;
  }

  const double* freqs = (double*)PyArray_DATA(frequencies);
  const int* w = (int*)PyArray_DATA(weights);
  const int num_qpoints = PyArray_DIMS(frequencies)[0];
  const int num_bands = PyArray_DIMS(frequencies)[1];

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

  const int* r = (int*)PyArray_DATA(rotation);
  const double* r_cart = (double*)PyArray_DATA(rotation_cart);
  double* fc2 = (double*)PyArray_DATA(force_constants);
  const double* t = (double*)PyArray_DATA(translation);
  const double* pos = (double*)PyArray_DATA(positions);
  const int num_pos = PyArray_DIMS(positions)[0];

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
			  const int * r,
			  const double * t,
			  const double symprec)
{
  int i, j, k, l, m, address_new, address;
  int is_found, rot_atom;
  double rot_pos[3], diff[3];

#pragma omp parallel for private(j, k, l, m, rot_pos, diff, is_found, rot_atom, address_new, address)
  for (i = 0; i < num_pos; i++) {
    for (j = 0; j < 3; j++) {
      rot_pos[j] = t[j];
      for (k = 0; k < 3; k++) {
	rot_pos[j] += r[j * 3 + k] * pos[i * 3 + k];
      }
    }

    for (j = 0; j < num_pos; j++) {
      is_found = 1;
      for (k = 0; k < 3; k++) {
	diff[k] = pos[j * 3 + k] - rot_pos[k];
	diff[k] -= nint(diff[k]);
	if (fabs(diff[k]) > symprec) {
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
	    fc2[address_new + j * 3 + k] +=
	      r_cart[l * 3 + j] * r_cart[m * 3 + k] *
	      fc2[address + l * 3 + m];
	  }
	}
      }
    }
  end:
    ;
  }

  return is_found;
}

static PyObject *py_get_neighboring_grid_points(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  int grid_point;
  if (!PyArg_ParseTuple(args, "OiOOOO",
			&relative_grid_points_py,
			&grid_point,
			&relative_grid_address_py,
			&mesh_py,
			&bz_grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  int* relative_grid_points = (int*)PyArray_DATA(relative_grid_points_py);
  THMCONST int (*relative_grid_address)[3] =
    (int(*)[3])PyArray_DATA(relative_grid_address_py);
  const int num_relative_grid_address = PyArray_DIMS(relative_grid_address_py)[0];
  const int *mesh = (int*)PyArray_DATA(mesh_py);
  THMCONST int (*bz_grid_address)[3] = (int(*)[3])PyArray_DATA(bz_grid_address_py);
  const int *bz_map = (int*)PyArray_DATA(bz_map_py);

  thm_get_neighboring_grid_points(relative_grid_points,
				  grid_point,
				  relative_grid_address,
				  num_relative_grid_address,
				  mesh,
				  bz_grid_address,
				  bz_map);
  Py_RETURN_NONE;
}

static PyObject *
py_get_tetrahedra_relative_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* reciprocal_lattice_py;

  if (!PyArg_ParseTuple(args, "OO",
			&relative_grid_address_py,
			&reciprocal_lattice_py)) {
    return NULL;
  }

  int (*relative_grid_address)[4][3] =
    (int(*)[4][3])PyArray_DATA(relative_grid_address_py);
  THMCONST double (*reciprocal_lattice)[3] =
    (double(*)[3])PyArray_DATA(reciprocal_lattice_py);

  thm_get_relative_grid_address(relative_grid_address, reciprocal_lattice);

  Py_RETURN_NONE;
}

static PyObject *
py_get_all_tetrahedra_relative_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_address_py;

  if (!PyArg_ParseTuple(args, "O",
			&relative_grid_address_py)) {
    return NULL;
  }

  int (*relative_grid_address)[24][4][3] =
    (int(*)[24][4][3])PyArray_DATA(relative_grid_address_py);

  thm_get_all_relative_grid_address(relative_grid_address);

  Py_RETURN_NONE;
}

static PyObject *
py_get_tetrahedra_integration_weight(PyObject *self, PyObject *args)
{
  double omega;
  PyArrayObject* tetrahedra_omegas_py;
  char function;
  if (!PyArg_ParseTuple(args, "dOc",
			&omega,
			&tetrahedra_omegas_py,
			&function)) {
    return NULL;
  }

  THMCONST double (*tetrahedra_omegas)[4] =
    (double(*)[4])PyArray_DATA(tetrahedra_omegas_py);

  double iw = thm_get_integration_weight(omega,
					 tetrahedra_omegas,
					 function);

  return PyFloat_FromDouble(iw);
}

static PyObject *
py_get_tetrahedra_integration_weight_at_omegas(PyObject *self, PyObject *args)
{
  PyArrayObject* integration_weights_py;
  PyArrayObject* omegas_py;
  PyArrayObject* tetrahedra_omegas_py;
  char function;
  if (!PyArg_ParseTuple(args, "OOOc",
			&integration_weights_py,
			&omegas_py,
			&tetrahedra_omegas_py,
			&function)) {
    return NULL;
  }

  const double *omegas = (double*)PyArray_DATA(omegas_py);
  double *iw = (double*)PyArray_DATA(integration_weights_py);
  const int num_omegas = (int)PyArray_DIMS(omegas_py)[0];
  THMCONST double (*tetrahedra_omegas)[4] =
    (double(*)[4])PyArray_DATA(tetrahedra_omegas_py);

  thm_get_integration_weight_at_omegas(iw,
				       num_omegas,
				       omegas,
				       tetrahedra_omegas,
				       function);

  Py_RETURN_NONE;
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
			    uplo);

  free(freqs);
  free(eigvecs);
  free(grid_points);
  free(fc2);
  free(svecs_fc2);
  free(multi_fc2);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_phonon(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* q_py;
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
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOdOOOOdc",
			&frequencies_py,
			&eigenvectors_py,
			&q_py,
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

  double* born;
  double* dielectric;
  double *q_dir;
  double* freqs = (double*)frequencies_py->data;
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  lapack_complex_double* eigvecs =
    (lapack_complex_double*)eigenvectors_py->data;
  const double* q = (double*) q_py->data;
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* svecs = convert_to_darray(shortest_vectors_py);
  Iarray* multi = convert_to_iarray(multiplicity_py);
  const double* masses = (double*)atomic_masses_py->data;
  const int* p2s = (int*)p2s_map_py->data;
  const int* s2p = (int*)s2p_map_py->data;
  const double* rec_lat = (double*)reciprocal_lattice_py->data;

  if ((PyObject*)born_effective_charge_py == Py_None) {
    born = NULL;
  } else {
    born = (double*)born_effective_charge_py->data;
  }
  if ((PyObject*)dielectric_constant_py == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)dielectric_constant_py->data;
  }
  if ((PyObject*)q_direction_py == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)q_direction_py->data;
  }

  get_phonons(eigvecs,
	      freqs,
	      q,
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
	      uplo);

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

static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}
