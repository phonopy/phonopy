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
#include <dynmat.h>
#include <derivative_dynmat.h>
#include <kgrid.h>
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
static PyObject * py_thm_neighboring_grid_points(PyObject *self, PyObject *args);
static PyObject *
py_thm_relative_grid_address(PyObject *self, PyObject *args);
static PyObject *
py_thm_all_relative_grid_address(PyObject *self, PyObject *args);
static PyObject *
py_thm_integration_weight(PyObject *self, PyObject *args);
static PyObject *
py_thm_integration_weight_at_omegas(PyObject *self, PyObject *args);
static PyObject * py_get_tetrahedra_frequenies(PyObject *self, PyObject *args);
static PyObject * py_run_tetrahedron_method(PyObject *self, PyObject *args);

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
  {"neighboring_grid_points", py_thm_neighboring_grid_points,
   METH_VARARGS, "Neighboring grid points by relative grid addresses"},
  {"tetrahedra_relative_grid_address", py_thm_relative_grid_address,
   METH_VARARGS, "Relative grid addresses of vertices of 24 tetrahedra"},
  {"all_tetrahedra_relative_grid_address",
   py_thm_all_relative_grid_address, METH_VARARGS,
   "4 (all) sets of relative grid addresses of vertices of 24 tetrahedra"},
  {"tetrahedra_integration_weight", py_thm_integration_weight,
   METH_VARARGS, "Integration weight for tetrahedron method"},
  {"tetrahedra_integration_weight_at_omegas",
   py_thm_integration_weight_at_omegas,
   METH_VARARGS, "Integration weight for tetrahedron method at omegas"},
  {"get_tetrahedra_frequencies", py_get_tetrahedra_frequenies,
   METH_VARARGS, "Run tetrahedron method"},
  {"run_tetrahedron_method", py_run_tetrahedron_method,
   METH_VARARGS, "Run tetrahedron method"},
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
  PyArrayObject* dynamical_matrix;
  PyArrayObject* force_constants;
  PyArrayObject* r_vector;
  PyArrayObject* q_vector;
  PyArrayObject* multiplicity;
  PyArrayObject* mass;
  PyArrayObject* super2prim_map;
  PyArrayObject* prim2super_map;

  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&dynamical_matrix,
			&force_constants,
			&q_vector,
			&r_vector,
			&multiplicity,
			&mass,
			&super2prim_map,
			&prim2super_map))
    return NULL;

  double* dm = (double*)PyArray_DATA(dynamical_matrix);
  const double* fc = (double*)PyArray_DATA(force_constants);
  const double* q = (double*)PyArray_DATA(q_vector);
  const double* r = (double*)PyArray_DATA(r_vector);
  const double* m = (double*)PyArray_DATA(mass);
  const int* multi = (int*)PyArray_DATA(multiplicity);
  const int* s2p_map = (int*)PyArray_DATA(super2prim_map);
  const int* p2s_map = (int*)PyArray_DATA(prim2super_map);
  const int num_patom = PyArray_DIMS(prim2super_map)[0];
  const int num_satom = PyArray_DIMS(super2prim_map)[0];

  get_dynamical_matrix_at_q(dm,
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
  PyArrayObject* dynamical_matrix;
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

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOd",
			&dynamical_matrix,
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

  double* dm = (double*)PyArray_DATA(dynamical_matrix);
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
  get_dynamical_matrix_at_q(dm,
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
  PyArrayObject* derivative_dynmat;
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

  if (!PyArg_ParseTuple(args, "OOOOOOOOOdOOO",
			&derivative_dynmat,
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

  double* ddm = (double*)PyArray_DATA(derivative_dynmat);
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

  get_derivative_dynmat_at_q(ddm,
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

#pragma omp parallel for private(j, omega) reduction(+:free_energy, entropy, heat_capacity)
  for (i = 0; i < num_qpoints; i++){
    for (j = 0; j < num_bands; j++){
      omega = freqs[i * num_bands + j];
      if (omega > 0.0) {
	free_energy += get_free_energy_omega(temperature, omega) * w[i];
	entropy += get_entropy_omega(temperature, omega) * w[i];
	heat_capacity += get_heat_capacity_omega(temperature, omega)* w[i];
      }
    }
  }

#pragma omp parallel for reduction(+:sum_weights)
  for (i = 0; i < num_qpoints; i++){
    sum_weights += w[i];
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

static PyObject *py_thm_neighboring_grid_points(PyObject *self, PyObject *args)
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
py_thm_relative_grid_address(PyObject *self, PyObject *args)
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
py_thm_all_relative_grid_address(PyObject *self, PyObject *args)
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
py_thm_integration_weight(PyObject *self, PyObject *args)
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
py_thm_integration_weight_at_omegas(PyObject *self, PyObject *args)
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

static PyObject * py_get_tetrahedra_frequenies(PyObject *self, PyObject *args)
{
  PyArrayObject* freq_tetras_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* mesh_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* gp_ir_index_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* frequencies_py;

  if (!PyArg_ParseTuple(args, "OOOOOOO",
			&freq_tetras_py,
			&grid_points_py,
			&mesh_py,
			&grid_address_py,
			&gp_ir_index_py,
			&relative_grid_address_py,
			&frequencies_py)) {
    return NULL;
  }

  double* freq_tetras = (double*)PyArray_DATA(freq_tetras_py);
  const int* grid_points = (int*)PyArray_DATA(grid_points_py);
  const int num_gp_in = (int)PyArray_DIMS(grid_points_py)[0];
  const int* mesh = (int*)PyArray_DATA(mesh_py);
  THMCONST int (*grid_address)[3] = (int(*)[3])PyArray_DATA(grid_address_py);
  const int* gp_ir_index = (int*)PyArray_DATA(gp_ir_index_py);
  THMCONST int (*relative_grid_address)[3] =
    (int(*)[3])PyArray_DATA(relative_grid_address_py);
  const double* frequencies = (double*)PyArray_DATA(frequencies_py);
  const int num_band = (int)PyArray_DIMS(frequencies_py)[1];
  const int is_shift[3] = {0, 0, 0};

  int i, j, k, gp;
  int g_addr[3];
  int address_double[3];

  for (i = 0; i < num_gp_in;  i++) {
#pragma omp parallel for private(k, g_addr, gp, address_double)
    for (j = 0; j < num_band * 96; j++) {
      for (k = 0; k < 3; k++) {
	g_addr[k] = grid_address[grid_points[i]][k] +
	  relative_grid_address[j % 96][k];
      }
      kgd_get_grid_address_double_mesh(address_double,
				       g_addr,
				       mesh,
				       is_shift);
      gp = kgd_get_grid_point_double_mesh(address_double, mesh);
      freq_tetras[i * num_band * 96 + j] =
	frequencies[gp_ir_index[gp] * num_band + j / 96];
    }
  }

  Py_RETURN_NONE;
}

static PyObject * py_run_tetrahedron_method(PyObject *self, PyObject *args)
{
  PyArrayObject* data_out_py;
  PyArrayObject* data_in_py;
  PyArrayObject* mesh_py;
  PyArrayObject* freq_points_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* weights_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* grid_mapping_table_py;
  PyArrayObject* ir_grid_points_py;
  PyArrayObject* relative_grid_address_py;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOO",
			&data_out_py,
			&data_in_py,
			&mesh_py,
			&freq_points_py,
			&frequencies_py,
			&weights_py,
			&grid_address_py,
			&grid_mapping_table_py,
			&ir_grid_points_py,
			&relative_grid_address_py)) {
    return NULL;
  }

  /* data_out[num_freq_points][num_kind][2] */
  double (*data_out)[2] = (double(*)[2])PyArray_DATA(data_out_py);
  /* data_in[num_ir_gp][num_band][num_kind] */
  const double* data_in = (double*)PyArray_DATA(data_in_py);
  const int* mesh = (int*)PyArray_DATA(mesh_py);
  const int num_kind = (int)PyArray_DIMS(data_in_py)[2];
  const double* freq_points = (double*)PyArray_DATA(freq_points_py);
  const int num_freq_points = (int)PyArray_DIMS(frequencies_py)[0];
  const double* frequencies = (double*)PyArray_DATA(frequencies_py);
  const int num_band = (int)PyArray_DIMS(frequencies_py)[1];
  const int* weights = (int*)PyArray_DATA(weights_py);
  THMCONST int (*grid_address)[3] = (int(*)[3])PyArray_DATA(grid_address_py);
  const int num_gp = (int)PyArray_DIMS(grid_address_py)[0];
  const int* grid_mapping_table = (int*)PyArray_DATA(grid_mapping_table_py);
  const int* ir_gp = (int*)PyArray_DATA(ir_grid_points_py);
  const int num_ir_gp = (int)PyArray_DIMS(ir_grid_points_py)[0];
  THMCONST int (*relative_grid_address)[4][3] =
    (int(*)[4][3])PyArray_DATA(relative_grid_address_py);
  const int is_shift[3] = {0, 0, 0};

  int i, j, k, l, q, r, gp;
  int g_addr[3];
  double iw;
  double tetrahedra[24][4];
  int *gp_ir_index;
  int address_double[3];
  int data_adrs;

  gp_ir_index = (int*)malloc(sizeof(int) * num_gp);

  j = 0;
  for (i = 0; i < num_gp; i++) {
    if (grid_mapping_table[i] == i) {
      gp_ir_index[i] = j;
      j++;
    } else {
      gp_ir_index[i] = gp_ir_index[grid_mapping_table[i]];
    }
  }

  for (i = 0; i < num_freq_points * num_kind; i++) {
    data_out[i][0] = 0;
    data_out[i][1] = 0;
  }

#pragma omp parallel for private(j, k, l, q, r, g_addr, gp, tetrahedra, iw, address_double)
  for (i = 0; i < num_freq_points; i++) {
    for (j = 0; j < num_ir_gp;  j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < 24; l++) {
	  for (q = 0; q < 4; q++) {
	    for (r = 0; r < 3; r++) {
	      g_addr[r] = grid_address[ir_gp[j]][r] +
		relative_grid_address[l][q][r];
	    }
	    kgd_get_grid_address_double_mesh(address_double,
					     g_addr,
					     mesh,
					     is_shift);
	    gp = kgd_get_grid_point_double_mesh(address_double, mesh);
	    tetrahedra[l][q] = frequencies[gp_ir_index[gp] * num_band + k];
	  }
	}
	data_adrs = j * num_band * num_kind + k * num_kind;
	for (l = 0; l < num_kind; i++) {
	  iw = thm_get_integration_weight(freq_points[i], tetrahedra, 'J');
	  data_out[i][0] += iw * weights[j] * data_in[data_adrs + l] / num_gp;
	  iw = thm_get_integration_weight(freq_points[i], tetrahedra, 'I');
	  data_out[i][1] += iw * weights[j] * data_in[data_adrs + l] / num_gp;
	}
      }
    }
  }

  free(gp_ir_index);

  Py_RETURN_NONE;
}

static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}
