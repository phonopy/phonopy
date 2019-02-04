/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of spglib. */

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
#include <assert.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <spglib.h>

#if (PY_MAJOR_VERSION < 3) && (PY_MINOR_VERSION < 6)
#define PYUNICODE_FROMSTRING PyString_FromString
#else
#define PYUNICODE_FROMSTRING PyUnicode_FromString
#endif

static PyObject * py_get_version(PyObject *self, PyObject *args);
static PyObject * py_get_dataset(PyObject *self, PyObject *args);
static PyObject * py_get_spacegroup_type(PyObject *self, PyObject *args);
static PyObject * py_get_pointgroup(PyObject *self, PyObject *args);
static PyObject * py_standardize_cell(PyObject *self, PyObject *args);
static PyObject * py_refine_cell(PyObject *self, PyObject *args);
static PyObject * py_get_symmetry(PyObject *self, PyObject *args);
static PyObject *
py_get_symmetry_with_collinear_spin(PyObject *self, PyObject *args);
static PyObject *
py_get_hall_number_from_symmetry(PyObject *self, PyObject *args);
static PyObject * py_find_primitive(PyObject *self, PyObject *args);
static PyObject *
py_get_grid_point_from_address(PyObject *self, PyObject *args);
static PyObject * py_get_ir_reciprocal_mesh(PyObject *self, PyObject *args);
static PyObject *
py_get_stabilized_reciprocal_mesh(PyObject *self, PyObject *args);
static PyObject *
py_get_grid_points_by_rotations(PyObject *self, PyObject *args);
static PyObject *
py_get_BZ_grid_points_by_rotations(PyObject *self, PyObject *args);
static PyObject * py_relocate_BZ_grid_address(PyObject *self, PyObject *args);
static PyObject * py_get_symmetry_from_database(PyObject *self, PyObject *args);
static PyObject * py_delaunay_reduce(PyObject *self, PyObject *args);
static PyObject * py_niggli_reduce(PyObject *self, PyObject *args);
static PyObject * py_get_error_message(PyObject *self, PyObject *args);

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

static PyMethodDef _spglib_methods[] = {
  {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
  {"version", py_get_version, METH_VARARGS, "Spglib version"},
  {"dataset", py_get_dataset, METH_VARARGS, "Dataset for crystal symmetry"},
  {"spacegroup_type", py_get_spacegroup_type, METH_VARARGS, "Space-group type symbols"},
  {"symmetry_from_database", py_get_symmetry_from_database, METH_VARARGS,
   "Get symmetry operations from database"},
  {"pointgroup", py_get_pointgroup, METH_VARARGS,
   "International symbol of pointgroup"},
  {"standardize_cell", py_standardize_cell, METH_VARARGS, "Standardize cell"},
  {"refine_cell", py_refine_cell, METH_VARARGS, "Refine cell"},
  {"symmetry", py_get_symmetry, METH_VARARGS, "Symmetry operations"},
  {"symmetry_with_collinear_spin", py_get_symmetry_with_collinear_spin,
   METH_VARARGS, "Symmetry operations with collinear spin magnetic moments"},
  {"hall_number_from_symmetry", py_get_hall_number_from_symmetry,
   METH_VARARGS, "Space group type is searched from symmetry operations."},
  {"primitive", py_find_primitive, METH_VARARGS,
   "Find primitive cell in the input cell"},
  {"grid_point_from_address", py_get_grid_point_from_address, METH_VARARGS,
   "Translate grid address to grid point index"},
  {"ir_reciprocal_mesh", py_get_ir_reciprocal_mesh, METH_VARARGS,
   "Reciprocal mesh points with map"},
  {"stabilized_reciprocal_mesh", py_get_stabilized_reciprocal_mesh, METH_VARARGS,
   "Reciprocal mesh points with map"},
  {"grid_points_by_rotations", py_get_grid_points_by_rotations, METH_VARARGS,
   "Rotated grid points are returned"},
  {"BZ_grid_points_by_rotations", py_get_BZ_grid_points_by_rotations, METH_VARARGS,
   "Rotated grid points in BZ are returned"},
  {"BZ_grid_address", py_relocate_BZ_grid_address, METH_VARARGS,
   "Relocate grid addresses inside Brillouin zone"},
  {"delaunay_reduce", py_delaunay_reduce, METH_VARARGS, "Delaunay reduction"},
  {"niggli_reduce", py_niggli_reduce, METH_VARARGS, "Niggli reduction"},
  {"error_message", py_get_error_message, METH_VARARGS, "Error message"},

  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int _spglib_traverse(PyObject *m, visitproc visit, void *arg) {
  Py_VISIT(GETSTATE(m)->error);
  return 0;
}

static int _spglib_clear(PyObject *m) {
  Py_CLEAR(GETSTATE(m)->error);
  return 0;
}

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_spglib",
  NULL,
  sizeof(struct module_state),
  _spglib_methods,
  NULL,
  _spglib_traverse,
  _spglib_clear,
  NULL
};

#define INITERROR return NULL

PyObject *
PyInit__spglib(void)

#else
#define INITERROR return

  void
  init_spglib(void)
#endif
{
  struct module_state *st;
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&moduledef);
#else
  PyObject *module = Py_InitModule("_spglib", _spglib_methods);
#endif

  if (module == NULL)
    INITERROR;

  st = GETSTATE(module);

  st->error = PyErr_NewException("_spglib.Error", NULL, NULL);
  if (st->error == NULL) {
    Py_DECREF(module);
    INITERROR;
  }

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}

static PyObject * py_get_version(PyObject *self, PyObject *args)
{
  PyObject *array;
  int i;
  int version[3];

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  version[0] = spg_get_major_version();
  version[1] = spg_get_minor_version();
  version[2] = spg_get_micro_version();

  array = PyList_New(3);
  for (i = 0; i < 3; i++) {
    PyList_SetItem(array, i, PyLong_FromLong((long)version[i]));
  }

  return array;
}

static PyObject * py_get_dataset(PyObject *self, PyObject *args)
{
  int hall_number;
  double symprec, angle_tolerance;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_atom_types;

  PyObject *array, *vec, *mat, *rot, *trans, *wyckoffs, *equiv_atoms;
  PyObject *site_symmetry_symbols, *mapping_to_primitive;
  PyObject *std_lattice, *std_types, *std_positions, *std_mapping_to_primitive;
  PyObject *std_rotation;

  int i, j, k, n;
  double (*lat)[3];
  double (*pos)[3];
  int num_atom, len_list;
  int* typat;
  SpglibDataset *dataset;

  if (!PyArg_ParseTuple(args, "OOOidd",
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &hall_number,
                        &symprec,
                        &angle_tolerance)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  num_atom = PyArray_DIMS(py_positions)[0];
  typat = (int*)PyArray_DATA(py_atom_types);

  if ((dataset = spgat_get_dataset_with_hall_number(lat,
                                                    pos,
                                                    typat,
                                                    num_atom,
                                                    hall_number,
                                                    symprec,
                                                    angle_tolerance)) == NULL) {
    Py_RETURN_NONE;
  }

  len_list = 19;
  array = PyList_New(len_list);
  n = 0;

  /* Space group number, international symbol, hall symbol */
  PyList_SetItem(array, n, PyLong_FromLong((long) dataset->spacegroup_number));
  n++;
  PyList_SetItem(array, n, PyLong_FromLong((long) dataset->hall_number));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(dataset->international_symbol));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(dataset->hall_symbol));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(dataset->choice));
  n++;

  /* Transformation matrix */
  mat = PyList_New(3);
  for (i = 0; i < 3; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyFloat_FromDouble(dataset->transformation_matrix[i][j]));
    }
    PyList_SetItem(mat, i, vec);
  }
  PyList_SetItem(array, n, mat);
  n++;

  /* Origin shift */
  vec = PyList_New(3);
  for (i = 0; i < 3; i++) {
    PyList_SetItem(vec, i, PyFloat_FromDouble(dataset->origin_shift[i]));
  }
  PyList_SetItem(array, n, vec);
  n++;

  /* Rotation matrices */
  rot = PyList_New(dataset->n_operations);
  for (i = 0; i < dataset->n_operations; i++) {
    mat = PyList_New(3);
    for (j = 0; j < 3; j++) {
      vec = PyList_New(3);
      for (k = 0; k < 3; k++) {
        PyList_SetItem(vec, k, PyLong_FromLong((long) dataset->rotations[i][j][k]));
      }
      PyList_SetItem(mat, j, vec);
    }
    PyList_SetItem(rot, i, mat);
  }
  PyList_SetItem(array, n, rot);
  n++;

  /* Translation vectors */
  trans = PyList_New(dataset->n_operations);
  for (i = 0; i < dataset->n_operations; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyFloat_FromDouble(dataset->translations[i][j]));
    }
    PyList_SetItem(trans, i, vec);
  }
  PyList_SetItem(array, n, trans);
  n++;

  /* Wyckoff letters, Equivalent atoms */
  wyckoffs = PyList_New(dataset->n_atoms);
  site_symmetry_symbols = PyList_New(dataset->n_atoms);
  equiv_atoms = PyList_New(dataset->n_atoms);
  mapping_to_primitive = PyList_New(dataset->n_atoms);
  for (i = 0; i < dataset->n_atoms; i++) {
    PyList_SetItem(wyckoffs, i,
                   PyLong_FromLong((long) dataset->wyckoffs[i]));
    PyList_SetItem(site_symmetry_symbols, i,
                   PYUNICODE_FROMSTRING(dataset->site_symmetry_symbols[i]));
    PyList_SetItem(equiv_atoms, i,
                   PyLong_FromLong((long) dataset->equivalent_atoms[i]));
    PyList_SetItem(mapping_to_primitive, i,
                   PyLong_FromLong((long) dataset->mapping_to_primitive[i]));
  }
  PyList_SetItem(array, n, wyckoffs);
  n++;
  PyList_SetItem(array, n, site_symmetry_symbols);
  n++;
  PyList_SetItem(array, n, equiv_atoms);
  n++;
  PyList_SetItem(array, n, mapping_to_primitive);
  n++;

  std_lattice = PyList_New(3);
  for (i = 0; i < 3; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyFloat_FromDouble(dataset->std_lattice[i][j]));
    }
    PyList_SetItem(std_lattice, i, vec);
  }
  PyList_SetItem(array, n, std_lattice);
  n++;

  /* Standardized unit cell */
  std_types = PyList_New(dataset->n_std_atoms);
  std_positions = PyList_New(dataset->n_std_atoms);
  std_mapping_to_primitive = PyList_New(dataset->n_std_atoms);
  for (i = 0; i < dataset->n_std_atoms; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyFloat_FromDouble(dataset->std_positions[i][j]));
    }
    PyList_SetItem(std_types, i, PyLong_FromLong((long) dataset->std_types[i]));
    PyList_SetItem(std_positions, i, vec);
    PyList_SetItem(std_mapping_to_primitive, i,
                   PyLong_FromLong((long) dataset->std_mapping_to_primitive[i]));
  }
  PyList_SetItem(array, n, std_types);
  n++;
  PyList_SetItem(array, n, std_positions);
  n++;

  std_rotation = PyList_New(3);
  for (i = 0; i < 3; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j,
                     PyFloat_FromDouble(dataset->std_rotation_matrix[i][j]));
    }
    PyList_SetItem(std_rotation, i, vec);
  }
  PyList_SetItem(array, n, std_rotation);
  n++;

  PyList_SetItem(array, n, std_mapping_to_primitive);
  n++;

  /* Point group */
  /* PyList_SetItem(array, n, PyLong_FromLong((long) dataset->pointgroup_number)); */
  /* n++; */
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(dataset->pointgroup_symbol));
  n++;

  assert(n != len_list);

  spg_free_dataset(dataset);

  return array;
}

static PyObject * py_get_symmetry_from_database(PyObject *self, PyObject *args)
{
  int hall_number;
  PyArrayObject* py_rotations;
  PyArrayObject* py_translations;

  int (*rot)[3][3];
  double (*trans)[3];
  int num_sym;

  if (!PyArg_ParseTuple(args, "OOi",
                        &py_rotations,
                        &py_translations,
                        &hall_number)) {
    return NULL;
  }

  if (PyArray_DIMS(py_rotations)[0] < 192 || PyArray_DIMS(py_translations)[0] < 192) {
    Py_RETURN_NONE;
  }

  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  trans = (double(*)[3])PyArray_DATA(py_translations);


  num_sym = spg_get_symmetry_from_database(rot, trans, hall_number);

  return PyLong_FromLong((long) num_sym);
}

static PyObject * py_get_spacegroup_type(PyObject *self, PyObject *args)
{
  int n, hall_number;
  PyObject *array;
  SpglibSpacegroupType spg_type;

  if (!PyArg_ParseTuple(args, "i", &hall_number)) {
    return NULL;
  }

  spg_type = spg_get_spacegroup_type(hall_number);
  if (spg_type.number == 0) {
    Py_RETURN_NONE;
  }

  array = PyList_New(11);
  n = 0;
  PyList_SetItem(array, n, PyLong_FromLong((long) spg_type.number));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.international_short));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.international_full));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.international));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.schoenflies));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.hall_symbol));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.choice));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.pointgroup_international));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.pointgroup_schoenflies));
  n++;
  PyList_SetItem(array, n, PyLong_FromLong((long) spg_type.arithmetic_crystal_class_number));
  n++;
  PyList_SetItem(array, n, PYUNICODE_FROMSTRING(spg_type.arithmetic_crystal_class_symbol));
  n++;

  return array;
}

static PyObject * py_get_pointgroup(PyObject *self, PyObject *args)
{
  PyArrayObject* py_rotations;

  int i, j;
  int trans_mat[3][3];
  char symbol[6];
  PyObject* array, * mat, * vec;
  int(*rot)[3][3];
  int num_rot;
  int ptg_num;

  if (! PyArg_ParseTuple(args, "O", &py_rotations)) {
    return NULL;
  }

  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  num_rot = PyArray_DIMS(py_rotations)[0];
  ptg_num = spg_get_pointgroup(symbol, trans_mat, rot, num_rot);

  /* Transformation matrix */
  mat = PyList_New(3);
  for (i = 0; i < 3; i++) {
    vec = PyList_New(3);
    for (j = 0; j < 3; j++) {
      PyList_SetItem(vec, j, PyLong_FromLong((long)trans_mat[i][j]));
    }
    PyList_SetItem(mat, i, vec);
  }

  array = PyList_New(3);
  PyList_SetItem(array, 0, PYUNICODE_FROMSTRING(symbol));
  PyList_SetItem(array, 1, PyLong_FromLong((long) ptg_num));
  PyList_SetItem(array, 2, mat);

  return array;
}

static PyObject * py_standardize_cell(PyObject *self, PyObject *args)
{
  int num_atom, to_primitive, no_idealize;
  double symprec, angle_tolerance;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_atom_types;

  double (*lat)[3];
  double (*pos)[3];
  int* typat;
  int num_atom_std;

  if (!PyArg_ParseTuple(args, "OOOiiidd",
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &num_atom,
                        &to_primitive,
                        &no_idealize,
                        &symprec,
                        &angle_tolerance)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  typat = (int*)PyArray_DATA(py_atom_types);

  num_atom_std = spgat_standardize_cell(lat,
                                        pos,
                                        typat,
                                        num_atom,
                                        to_primitive,
                                        no_idealize,
                                        symprec,
                                        angle_tolerance);

  return PyLong_FromLong((long) num_atom_std);
}

static PyObject * py_refine_cell(PyObject *self, PyObject *args)
{
  int num_atom;
  double symprec, angle_tolerance;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_atom_types;

  double (*lat)[3];
  double (*pos)[3];
  int* typat;
  int num_atom_std;

  if (!PyArg_ParseTuple(args, "OOOidd",
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &num_atom,
                        &symprec,
                        &angle_tolerance)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  typat = (int*)PyArray_DATA(py_atom_types);

  num_atom_std = spgat_refine_cell(lat,
                                   pos,
                                   typat,
                                   num_atom,
                                   symprec,
                                   angle_tolerance);

  return PyLong_FromLong((long) num_atom_std);
}


static PyObject * py_find_primitive(PyObject *self, PyObject *args)
{
  double symprec, angle_tolerance;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_atom_types;

  double (*lat)[3];
  double (*pos)[3];
  int num_atom;
  int* types;
  int num_atom_prim;

  if (!PyArg_ParseTuple(args, "OOOdd",
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &symprec,
                        &angle_tolerance)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  num_atom = PyArray_DIMS(py_positions)[0];
  types = (int*)PyArray_DATA(py_atom_types);

  num_atom_prim = spgat_find_primitive(lat,
                                       pos,
                                       types,
                                       num_atom,
                                       symprec,
                                       angle_tolerance);

  return PyLong_FromLong((long) num_atom_prim);
}

static PyObject * py_get_symmetry(PyObject *self, PyObject *args)
{
  double symprec, angle_tolerance;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_rotations;
  PyArrayObject* py_translations;
  PyArrayObject* py_atom_types;

  double (*lat)[3];
  double (*pos)[3];
  int* types;
  int num_atom;
  int (*rot)[3][3];
  double (*trans)[3];
  int num_sym_from_array_size;
  int num_sym;

  if (!PyArg_ParseTuple(args, "OOOOOdd",
                        &py_rotations,
                        &py_translations,
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &symprec,
                        &angle_tolerance)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  types = (int*)PyArray_DATA(py_atom_types);
  num_atom = PyArray_DIMS(py_positions)[0];
  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  trans = (double(*)[3])PyArray_DATA(py_translations);
  num_sym_from_array_size = PyArray_DIMS(py_rotations)[0];

  /* num_sym has to be larger than num_sym_from_array_size. */
  num_sym = spgat_get_symmetry(rot,
                               trans,
                               num_sym_from_array_size,
                               lat,
                               pos,
                               types,
                               num_atom,
                               symprec,
                               angle_tolerance);
  return PyLong_FromLong((long) num_sym);
}

static PyObject * py_get_symmetry_with_collinear_spin(PyObject *self,
                                                      PyObject *args)
{
  double symprec, angle_tolerance;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_rotations;
  PyArrayObject* py_translations;
  PyArrayObject* py_atom_types;
  PyArrayObject* py_magmoms;
  PyArrayObject* py_equiv_atoms;

  double (*lat)[3];
  double (*pos)[3];
  double *spins;
  int *types;
  int num_atom;
  int (*rot)[3][3];
  double (*trans)[3];
  int *equiv_atoms;
  int num_sym_from_array_size;
  int num_sym;

  if (!PyArg_ParseTuple(args, "OOOOOOOdd",
                        &py_rotations,
                        &py_translations,
                        &py_equiv_atoms,
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &py_magmoms,
                        &symprec,
                        &angle_tolerance)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  spins = (double*)PyArray_DATA(py_magmoms);
  types = (int*)PyArray_DATA(py_atom_types);
  num_atom = PyArray_DIMS(py_positions)[0];
  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  trans = (double(*)[3])PyArray_DATA(py_translations);
  equiv_atoms = (int*)PyArray_DATA(py_equiv_atoms);
  num_sym_from_array_size = PyArray_DIMS(py_rotations)[0];

  /* num_sym has to be larger than num_sym_from_array_size. */
  num_sym = spgat_get_symmetry_with_collinear_spin(rot,
                                                   trans,
                                                   equiv_atoms,
                                                   num_sym_from_array_size,
                                                   lat,
                                                   pos,
                                                   types,
                                                   spins,
                                                   num_atom,
                                                   symprec,
                                                   angle_tolerance);
  return PyLong_FromLong((long) num_sym);
}

static PyObject *
py_get_hall_number_from_symmetry(PyObject *self, PyObject *args)
{
  double symprec;
  PyArrayObject* py_rotations;
  PyArrayObject* py_translations;

  int (*rot)[3][3];
  double (*trans)[3];
  int num_sym;
  int hall_number;

  if (!PyArg_ParseTuple(args, "OOd",
                        &py_rotations,
                        &py_translations,
                        &symprec)) {
    return NULL;
  }

  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  trans = (double(*)[3])PyArray_DATA(py_translations);
  num_sym = PyArray_DIMS(py_rotations)[0];

  hall_number = spg_get_hall_number_from_symmetry(rot,
                                              trans,
                                              num_sym,
                                              symprec);

  return PyLong_FromLong((long) hall_number);
}

static PyObject * py_get_grid_point_from_address(PyObject *self, PyObject *args)
{
  PyArrayObject* py_grid_address;
  PyArrayObject* py_mesh;

  int* grid_address;
  int* mesh;
  size_t gp;

  if (!PyArg_ParseTuple(args, "OO",
                        &py_grid_address,
                        &py_mesh)) {
    return NULL;
  }

  grid_address = (int*)PyArray_DATA(py_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);

  gp = spg_get_dense_grid_point_from_address(grid_address, mesh);

  return PyLong_FromSize_t(gp);
}

static PyObject * py_get_ir_reciprocal_mesh(PyObject *self, PyObject *args)
{
  double symprec;
  PyArrayObject* py_grid_address;
  PyArrayObject* py_grid_mapping_table;
  PyArrayObject* py_mesh;
  PyArrayObject* py_is_shift;
  int is_time_reversal;
  PyArrayObject* py_lattice;
  PyArrayObject* py_positions;
  PyArrayObject* py_atom_types;

  double (*lat)[3];
  double (*pos)[3];
  int* types;
  int* mesh;
  int* is_shift;
  int num_atom;
  int (*grid_address)[3];
  int *grid_mapping_table_int;
  size_t *grid_mapping_table_size_t;
  int num_ir_int;
  size_t num_ir_size_t;

  if (!PyArg_ParseTuple(args, "OOOOiOOOd",
                        &py_grid_address,
                        &py_grid_mapping_table,
                        &py_mesh,
                        &py_is_shift,
                        &is_time_reversal,
                        &py_lattice,
                        &py_positions,
                        &py_atom_types,
                        &symprec)) {
    return NULL;
  }

  lat = (double(*)[3])PyArray_DATA(py_lattice);
  pos = (double(*)[3])PyArray_DATA(py_positions);
  types = (int*)PyArray_DATA(py_atom_types);
  mesh = (int*)PyArray_DATA(py_mesh);
  is_shift = (int*)PyArray_DATA(py_is_shift);
  num_atom = PyArray_DIMS(py_positions)[0];
  grid_address = (int(*)[3])PyArray_DATA(py_grid_address);
  if (PyArray_TYPE(py_grid_mapping_table) == NPY_UINTP) {
    grid_mapping_table_size_t = (size_t*)PyArray_DATA(py_grid_mapping_table);
    num_ir_size_t = spg_get_dense_ir_reciprocal_mesh(grid_address,
                                                     grid_mapping_table_size_t,
                                                     mesh,
                                                     is_shift,
                                                     is_time_reversal,
                                                     lat,
                                                     pos,
                                                     types,
                                                     num_atom,
                                                     symprec);
    return PyLong_FromSize_t(num_ir_size_t);
  }
  if (PyArray_TYPE(py_grid_mapping_table) == NPY_INT) {
    grid_mapping_table_int = (int*)PyArray_DATA(py_grid_mapping_table);
    /* num_sym has to be larger than num_sym_from_array_size. */
    num_ir_int = spg_get_ir_reciprocal_mesh(grid_address,
                                            grid_mapping_table_int,
                                            mesh,
                                            is_shift,
                                            is_time_reversal,
                                            lat,
                                            pos,
                                            types,
                                            num_atom,
                                            symprec);
    return PyLong_FromLong((long) num_ir_int);
  }

  Py_RETURN_NONE;
}

static PyObject *
py_get_stabilized_reciprocal_mesh(PyObject *self, PyObject *args)
{
  PyArrayObject* py_grid_address;
  PyArrayObject* py_grid_mapping_table;
  PyArrayObject* py_mesh;
  PyArrayObject* py_is_shift;
  int is_time_reversal;
  PyArrayObject* py_rotations;
  PyArrayObject* py_qpoints;

  int (*grid_address)[3];
  int* mesh;
  int* is_shift;
  int (*rot)[3][3];
  int num_rot;
  double (*q)[3];
  int num_q;

  int *grid_mapping_table_int;
  size_t *grid_mapping_table_size_t;
  int num_ir_int;
  size_t num_ir_size_t;

  if (!PyArg_ParseTuple(args, "OOOOiOO",
                        &py_grid_address,
                        &py_grid_mapping_table,
                        &py_mesh,
                        &py_is_shift,
                        &is_time_reversal,
                        &py_rotations,
                        &py_qpoints)) {
    return NULL;
  }

  grid_address = (int(*)[3])PyArray_DATA(py_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);
  is_shift = (int*)PyArray_DATA(py_is_shift);
  rot = (int(*)[3][3])PyArray_DATA(py_rotations);
  num_rot = PyArray_DIMS(py_rotations)[0];
  q = (double(*)[3])PyArray_DATA(py_qpoints);
  num_q = PyArray_DIMS(py_qpoints)[0];

  if (PyArray_TYPE(py_grid_mapping_table) == NPY_UINTP) {
    grid_mapping_table_size_t = (size_t*)PyArray_DATA(py_grid_mapping_table);
    num_ir_size_t =
      spg_get_dense_stabilized_reciprocal_mesh(grid_address,
                                               grid_mapping_table_size_t,
                                               mesh,
                                               is_shift,
                                               is_time_reversal,
                                               num_rot,
                                               rot,
                                               num_q,
                                               q);
    return PyLong_FromSize_t(num_ir_size_t);
  }
  if (PyArray_TYPE(py_grid_mapping_table) == NPY_INT) {
    grid_mapping_table_int = (int*)PyArray_DATA(py_grid_mapping_table);
    num_ir_int = spg_get_stabilized_reciprocal_mesh(grid_address,
                                                    grid_mapping_table_int,
                                                    mesh,
                                                    is_shift,
                                                    is_time_reversal,
                                                    num_rot,
                                                    rot,
                                                    num_q,
                                                    q);
    return PyLong_FromLong((long) num_ir_int);
  }

  Py_RETURN_NONE;
}

static PyObject *
py_get_grid_points_by_rotations(PyObject *self, PyObject *args)
{
  PyArrayObject* py_rot_grid_points;
  PyArrayObject* py_address_orig;
  PyArrayObject* py_rot_reciprocal;
  PyArrayObject* py_mesh;
  PyArrayObject* py_is_shift;

  size_t *rot_grid_points;
  int *address_orig;
  int (*rot_reciprocal)[3][3];
  int num_rot;
  int* mesh;
  int* is_shift;

  if (!PyArg_ParseTuple(args, "OOOOO",
                        &py_rot_grid_points,
                        &py_address_orig,
                        &py_rot_reciprocal,
                        &py_mesh,
                        &py_is_shift)) {
    return NULL;
  }

  rot_grid_points = (size_t*)PyArray_DATA(py_rot_grid_points);
  address_orig = (int*)PyArray_DATA(py_address_orig);
  rot_reciprocal = (int(*)[3][3])PyArray_DATA(py_rot_reciprocal);
  num_rot = PyArray_DIMS(py_rot_reciprocal)[0];
  mesh = (int*)PyArray_DATA(py_mesh);
  is_shift = (int*)PyArray_DATA(py_is_shift);

  spg_get_dense_grid_points_by_rotations(rot_grid_points,
                                         address_orig,
                                         num_rot,
                                         rot_reciprocal,
                                         mesh,
                                         is_shift);
  Py_RETURN_NONE;
}

static PyObject *
py_get_BZ_grid_points_by_rotations(PyObject *self, PyObject *args)
{
  PyArrayObject* py_rot_grid_points;
  PyArrayObject* py_address_orig;
  PyArrayObject* py_rot_reciprocal;
  PyArrayObject* py_mesh;
  PyArrayObject* py_is_shift;
  PyArrayObject* py_bz_map;

  size_t *rot_grid_points;
  int *address_orig;
  int (*rot_reciprocal)[3][3];
  int num_rot;
  int* mesh;
  int* is_shift;
  size_t* bz_map;

  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &py_rot_grid_points,
                        &py_address_orig,
                        &py_rot_reciprocal,
                        &py_mesh,
                        &py_is_shift,
                        &py_bz_map)) {
    return NULL;
  }

  rot_grid_points = (size_t*)PyArray_DATA(py_rot_grid_points);
  address_orig = (int*)PyArray_DATA(py_address_orig);
  rot_reciprocal = (int(*)[3][3])PyArray_DATA(py_rot_reciprocal);
  num_rot = PyArray_DIMS(py_rot_reciprocal)[0];
  mesh = (int*)PyArray_DATA(py_mesh);
  is_shift = (int*)PyArray_DATA(py_is_shift);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);

  spg_get_dense_BZ_grid_points_by_rotations(rot_grid_points,
                                            address_orig,
                                            num_rot,
                                            rot_reciprocal,
                                            mesh,
                                            is_shift,
                                            bz_map);
  Py_RETURN_NONE;
}

static PyObject * py_relocate_BZ_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* py_bz_grid_address;
  PyArrayObject* py_bz_map;
  PyArrayObject* py_grid_address;
  PyArrayObject* py_mesh;
  PyArrayObject* py_is_shift;
  PyArrayObject* py_reciprocal_lattice;

  int (*bz_grid_address)[3];
  size_t *bz_map;
  int (*grid_address)[3];
  int* mesh;
  int* is_shift;
  double (*reciprocal_lattice)[3];
  size_t num_ir_gp;

  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &py_bz_grid_address,
                        &py_bz_map,
                        &py_grid_address,
                        &py_mesh,
                        &py_reciprocal_lattice,
                        &py_is_shift)) {
    return NULL;
  }

  bz_grid_address = (int(*)[3])PyArray_DATA(py_bz_grid_address);
  bz_map = (size_t*)PyArray_DATA(py_bz_map);
  grid_address = (int(*)[3])PyArray_DATA(py_grid_address);
  mesh = (int*)PyArray_DATA(py_mesh);
  is_shift = (int*)PyArray_DATA(py_is_shift);
  reciprocal_lattice = (double(*)[3])PyArray_DATA(py_reciprocal_lattice);

  num_ir_gp = spg_relocate_dense_BZ_grid_address(bz_grid_address,
                                                 bz_map,
                                                 grid_address,
                                                 mesh,
                                                 reciprocal_lattice,
                                                 is_shift);

  return PyLong_FromSize_t(num_ir_gp);
}

static PyObject * py_delaunay_reduce(PyObject *self, PyObject *args)
{
  PyArrayObject* py_lattice;
  double symprec;

  double (*lattice)[3];
  int result;

  if (!PyArg_ParseTuple(args, "Od", &py_lattice, &symprec)) {
    return NULL;
  }

  lattice = (double(*)[3])PyArray_DATA(py_lattice);

  result = spg_delaunay_reduce(lattice, symprec);

  return PyLong_FromLong((long) result);
}

static PyObject * py_niggli_reduce(PyObject *self, PyObject *args)
{
  PyArrayObject* py_lattice;
  double eps;

  double (*lattice)[3];
  int result;

  if (!PyArg_ParseTuple(args, "Od", &py_lattice, &eps)) {
    return NULL;
  }

  lattice = (double(*)[3])PyArray_DATA(py_lattice);

  result = spg_niggli_reduce(lattice, eps);

  return PyLong_FromLong((long) result);
}

static PyObject * py_get_error_message(PyObject *self, PyObject *args)
{
  SpglibError error;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  error = spg_get_error_code();

  return PYUNICODE_FROMSTRING(spg_get_error_message(error));
}
