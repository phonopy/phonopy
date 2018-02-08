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
#include <float.h>
#include <numpy/arrayobject.h>
#include <dynmat.h>
#include <derivative_dynmat.h>
#include <kgrid.h>
#include <tetrahedron_method.h>

#define KB 8.6173382568083159E-05
#define PHPYCONST

/* Build dynamical matrix */
static PyObject * py_perm_trans_symmetrize_fc(PyObject *self, PyObject *args);
static PyObject *
py_perm_trans_symmetrize_compact_fc(PyObject *self, PyObject *args);
static PyObject * py_get_dynamical_matrix(PyObject *self, PyObject *args);
static PyObject * py_get_nac_dynamical_matrix(PyObject *self, PyObject *args);
static PyObject * py_get_dipole_dipole(PyObject *self, PyObject *args);
static PyObject * py_get_dipole_dipole_q0(PyObject *self, PyObject *args);
static PyObject * py_get_derivative_dynmat(PyObject *self, PyObject *args);
static PyObject * py_get_thermal_properties(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc2_with_mappings(PyObject *self,
                                                  PyObject *args);
static PyObject * py_compute_permutation(PyObject *self, PyObject *args);
static PyObject * py_gsv_copy_smallest_vectors(PyObject *self, PyObject *args);

static void distribute_fc2_with_mappings(double (*fc2)[3][3],
                                         const int * atom_list,
                                         const int len_atom_list,
                                         PHPYCONST double (*r_carts)[3][3],
                                         const int * permutations,
                                         const int * map_atoms,
                                         const int * map_syms,
                                         const int num_rot,
                                         const int num_pos);

static int compute_permutation(int * rot_atom,
                                  PHPYCONST double lat[3][3],
                                  PHPYCONST double (*pos)[3],
                                  PHPYCONST double (*rot_pos)[3],
                                  const int num_pos,
                                  const double symprec);

static void gsv_copy_smallest_vectors(double (*shortest_vectors)[27][3],
                                      int * multiplicity,
                                      PHPYCONST double (*vector_lists)[27][3],
                                      PHPYCONST double (*length_lists)[27],
                                      const int num_lists,
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
static PyObject * py_tetrahedron_method_dos(PyObject *self, PyObject *args);

static double get_free_energy_omega(const double temperature,
                                    const double omega);
static double get_entropy_omega(const double temperature,
                                const double omega);
static double get_heat_capacity_omega(const double temperature,
                                      const double omega);
static void set_index_permutation_symmetry_fc(double * fc,
                                              const int natom);
static void set_translational_symmetry_fc(double * fc,
                                          const int natom);
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
  {"perm_trans_symmetrize_fc", py_perm_trans_symmetrize_fc, METH_VARARGS,
   "Enforce permutation and translational symmetry of force constants"},
  {"perm_trans_symmetrize_compact_fc", py_perm_trans_symmetrize_compact_fc,
   METH_VARARGS,
   "Enforce permutation and translational symmetry of compact force constants"},
  {"dynamical_matrix", py_get_dynamical_matrix, METH_VARARGS,
   "Dynamical matrix"},
  {"nac_dynamical_matrix", py_get_nac_dynamical_matrix, METH_VARARGS,
   "NAC dynamical matrix"},
  {"dipole_dipole", py_get_dipole_dipole, METH_VARARGS,
   "Dipole-dipole interaction"},
  {"dipole_dipole_q0", py_get_dipole_dipole_q0, METH_VARARGS,
   "q=0 terms of Dipole-dipole interaction"},
  {"derivative_dynmat", py_get_derivative_dynmat, METH_VARARGS,
   "Q derivative of dynamical matrix"},
  {"thermal_properties", py_get_thermal_properties, METH_VARARGS,
   "Thermal properties"},
  {"distribute_fc2_with_mappings", py_distribute_fc2_with_mappings,
   METH_VARARGS,
   "Distribute force constants for all atoms in atom_list using precomputed symmetry mappings."},
  {"compute_permutation", py_compute_permutation, METH_VARARGS,
   "Compute indices of original points in a set of rotated points."},
  {"gsv_copy_smallest_vectors", py_gsv_copy_smallest_vectors, METH_VARARGS,
   "Implementation detail of get_smallest_vectors."},
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
  {"tetrahedron_method_dos", py_tetrahedron_method_dos,
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
  struct module_state *st;
  if (module == NULL)
    INITERROR;
  st = GETSTATE(module);

  st->error = PyErr_NewException("_phonopy.Error", NULL, NULL);
  if (st->error == NULL) {
    Py_DECREF(module);
    INITERROR;
  }

#if PY_MAJOR_VERSION >= 3
  return module;
#endif
}

static PyObject * py_compute_permutation(PyObject *self, PyObject *args)
{
  PyArrayObject* permutation;
  PyArrayObject* lattice;
  PyArrayObject* positions;
  PyArrayObject* permuted_positions;
  double symprec;

  int* rot_atoms;
  double (*lat)[3];
  double (*pos)[3];
  double (*rot_pos)[3];
  int num_pos;

  int is_found;

  if (!PyArg_ParseTuple(args, "OOOOd",
                        &permutation,
                        &lattice,
                        &positions,
                        &permuted_positions,
                        &symprec)) {
    return NULL;
  }

  rot_atoms = (int*)PyArray_DATA(permutation);
  lat = (double(*)[3])PyArray_DATA(lattice);
  pos = (double(*)[3])PyArray_DATA(positions);
  rot_pos = (double(*)[3])PyArray_DATA(permuted_positions);
  num_pos = PyArray_DIMS(positions)[0];

  is_found = compute_permutation(rot_atoms,
                                 lat,
                                 pos,
                                 rot_pos,
                                 num_pos,
                                 symprec);

  return Py_BuildValue("i", is_found);
}

static PyObject * py_gsv_copy_smallest_vectors(PyObject *self, PyObject *args)
{
  PyArrayObject* py_shortest_vectors;
  PyArrayObject* py_multiplicity;
  PyArrayObject* py_vectors;
  PyArrayObject* py_lengths;
  double symprec;

  double (*shortest_vectors)[27][3];
  double (*vectors)[27][3];
  double (*lengths)[27];
  int * multiplicity;
  int size_super, size_prim;

  if (!PyArg_ParseTuple(args, "OOOOd",
                        &py_shortest_vectors,
                        &py_multiplicity,
                        &py_vectors,
                        &py_lengths,
                        &symprec)) {
    return NULL;
  }

  shortest_vectors = (double(*)[27][3])PyArray_DATA(py_shortest_vectors);
  multiplicity = (int*)PyArray_DATA(py_multiplicity);
  vectors = (double(*)[27][3])PyArray_DATA(py_vectors);
  lengths = (double(*)[27])PyArray_DATA(py_lengths);
  size_super = PyArray_DIMS(py_vectors)[0];
  size_prim = PyArray_DIMS(py_vectors)[1];

  gsv_copy_smallest_vectors(shortest_vectors,
                            multiplicity,
                            vectors,
                            lengths,
                            size_super * size_prim,
                            symprec);

  Py_RETURN_NONE;
}

static PyObject * py_perm_trans_symmetrize_fc(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants;
  double *fc;
  int natom;

  if (!PyArg_ParseTuple(args, "O", &force_constants)) {
    return NULL;
  }

  fc = (double*)PyArray_DATA(force_constants);
  natom = PyArray_DIMS(force_constants)[0];

  set_index_permutation_symmetry_fc(fc, natom);
  set_translational_symmetry_fc(fc, natom);

  Py_RETURN_NONE;
}

static PyObject *
py_perm_trans_symmetrize_compact_fc(PyObject *self, PyObject *args)
{
  PyArrayObject* fc_py;
  PyArrayObject* permutations_py;
  PyArrayObject* s2p_map_py;
  PyArrayObject* p2s_map_py;
  double sums[3][3];
  double *fc;
  int *s2p;
  int *p2s;
  int *perms;
  int n_patom, n_satom, nsym, i_p, i, j, k, l, m, n, target;
  int *nsym_list;
  int *s2pp;
  double *fc_tmp;

  nsym_list = NULL;
  s2pp = NULL;
  fc_tmp = NULL;

  if (!PyArg_ParseTuple(args, "OOOO",
                        &fc_py,
                        &permutations_py,
                        &s2p_map_py,
                        &p2s_map_py)) {
    return NULL;
  }

  fc = (double*)PyArray_DATA(fc_py);
  perms = (int*)PyArray_DATA(permutations_py);
  s2p = (int*)PyArray_DATA(s2p_map_py);
  p2s = (int*)PyArray_DATA(p2s_map_py);
  n_patom = PyArray_DIMS(fc_py)[0];
  n_satom = PyArray_DIMS(fc_py)[1];
  nsym = PyArray_DIMS(permutations_py)[0];

  nsym_list = (int*) malloc(sizeof(int) * n_satom);
  for (i = 0; i < n_satom; i++) {
    nsym_list[i] = -1;
  }

  s2pp = (int*) malloc(sizeof(int) * n_satom);
  for (i = 0; i < n_satom; i++) {
    s2pp[i] = -1;
    for (j = 0; j < n_patom; j++) {
      if (p2s[j] == s2p[i]) {
        s2pp[i] = j;
        break;
      }
    }
  }

  fc_tmp = (double*) malloc(sizeof(double) * n_patom * n_satom * 3 * 3);
  for (i = 0; i < n_patom * n_satom * 3 * 3; i++) {
    fc_tmp[i] = 0;
  }

  for (i = 0; i < n_satom; i++) {
    target = s2p[i];
    for (j = 0; j < nsym; j++) {
      if (perms[j * n_satom + i] == target) {
        nsym_list[i] = j;
        break;
      }
    }
  }

#pragma omp parallel for private(i_p, i, k, l, m, n)
  for (j = 0; j < n_satom; j++) {
    for (i_p = 0; i_p < n_patom; i_p++) {
      i = p2s[i_p];
      if (i == j) { /* diagnoal part */
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            m = i_p * n_satom * 9 + i * 9 + k * 3 + l;
            n = i_p * n_satom * 9 + i * 9 + l * 3 + k;
            fc_tmp[m] = (fc[m] + fc[n]) / 2;
          }
        }
      } else { /* non diagonal part */
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            m = i_p * n_satom * 9 + j * 9 + k * 3 + l;
            n = s2pp[j] * n_satom * 9 + perms[nsym_list[j] * n_satom + i] * 9 + l * 3 + k;
            fc_tmp[m] = (fc[n] + fc[m]) / 2;
          }
        }
      }
    }
  }

  for (i_p = 0; i_p < n_patom; i_p++) {
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        sums[k][l] = 0;
        m = i_p * n_satom * 9 + k * 3 + l;
        for (j = 0; j < n_satom; j++) {
          if (p2s[i_p] != j) {
            sums[k][l] += fc_tmp[m];
          }
          m += 9;
        }
      }
    }
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        fc_tmp[i_p * n_satom * 9 + p2s[i_p] * 9 + k * 3 + l] =
          -(sums[k][l] + sums[l][k]) / 2;
      }
    }
  }

  for (i = 0; i < n_patom * n_satom * 3 * 3; i++) {
    fc[i] = fc_tmp[i];
  }

  free(nsym_list);
  nsym_list = NULL;
  free(s2pp);
  s2pp = NULL;
  free(fc_tmp);
  fc_tmp = NULL;

  Py_RETURN_NONE;
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

  double* dm;
  double* fc;
  double* q;
  double* r;
  double* m;
  int* multi;
  int* s2p_map;
  int* p2s_map;
  int num_patom;
  int num_satom;

  if (!PyArg_ParseTuple(args, "OOOOOOOO",
                        &dynamical_matrix,
                        &force_constants,
                        &q_vector,
                        &r_vector,
                        &multiplicity,
                        &mass,
                        &super2prim_map,
                        &prim2super_map)) {
    return NULL;
  }

  dm = (double*)PyArray_DATA(dynamical_matrix);
  fc = (double*)PyArray_DATA(force_constants);
  q = (double*)PyArray_DATA(q_vector);
  r = (double*)PyArray_DATA(r_vector);
  m = (double*)PyArray_DATA(mass);
  multi = (int*)PyArray_DATA(multiplicity);
  s2p_map = (int*)PyArray_DATA(super2prim_map);
  p2s_map = (int*)PyArray_DATA(prim2super_map);
  num_patom = PyArray_DIMS(prim2super_map)[0];
  num_satom = PyArray_DIMS(super2prim_map)[0];

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
                            NULL,
                            1);

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

  double* dm;
  double* fc;
  double* q_cart;
  double* q;
  double* r;
  double* m;
  double* z;
  int* multi;
  int* s2p_map;
  int* p2s_map;
  int num_patom;
  int num_satom;

  int n;
  double *charge_sum;

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

  dm = (double*)PyArray_DATA(dynamical_matrix);
  fc = (double*)PyArray_DATA(force_constants);
  q_cart = (double*)PyArray_DATA(q_cart_vector);
  q = (double*)PyArray_DATA(q_vector);
  r = (double*)PyArray_DATA(r_vector);
  m = (double*)PyArray_DATA(mass);
  z = (double*)PyArray_DATA(born);
  multi = (int*)PyArray_DATA(multiplicity);
  s2p_map = (int*)PyArray_DATA(super2prim_map);
  p2s_map = (int*)PyArray_DATA(prim2super_map);
  num_patom = PyArray_DIMS(prim2super_map)[0];
  num_satom = PyArray_DIMS(super2prim_map)[0];

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
                            charge_sum,
                            1);

  free(charge_sum);

  Py_RETURN_NONE;
}

static PyObject * py_get_dipole_dipole(PyObject *self, PyObject *args)
{
  PyArrayObject* dd_py;
  PyArrayObject* dd_q0_py;
  PyArrayObject* G_list_py;
  PyArrayObject* q_vector_py;
  PyArrayObject* q_direction_py;
  PyArrayObject* born_py;
  PyArrayObject* dielectric_py;
  PyArrayObject* pos_py;
  double factor;
  double lambda;
  double tolerance;

  double* dd;
  double* dd_q0;
  double* G_list;
  double* q_vector;
  double* q_direction;
  double* born;
  double* dielectric;
  double *pos;
  int num_patom, num_G;

  if (!PyArg_ParseTuple(args, "OOOOOOOOddd",
                        &dd_py,
                        &dd_q0_py,
                        &G_list_py,
                        &q_vector_py,
                        &q_direction_py,
                        &born_py,
                        &dielectric_py,
                        &pos_py,
                        &factor,
                        &lambda,
                        &tolerance))
    return NULL;


  dd = (double*)PyArray_DATA(dd_py);
  dd_q0 = (double*)PyArray_DATA(dd_q0_py);
  G_list = (double*)PyArray_DATA(G_list_py);
  if ((PyObject*)q_direction_py == Py_None) {
    q_direction = NULL;
  } else {
    q_direction = (double*)PyArray_DATA(q_direction_py);
  }
  q_vector = (double*)PyArray_DATA(q_vector_py);
  born = (double*)PyArray_DATA(born_py);
  dielectric = (double*)PyArray_DATA(dielectric_py);
  pos = (double*)PyArray_DATA(pos_py);
  num_G = PyArray_DIMS(G_list_py)[0];
  num_patom = PyArray_DIMS(pos_py)[0];

  get_dipole_dipole(dd, /* [natom, 3, natom, 3, (real, imag)] */
                    dd_q0, /* [natom, 3, 3, (real, imag)] */
                    G_list, /* [num_kvec, 3] */
                    num_G,
                    num_patom,
                    q_vector,
                    q_direction,
                    born,
                    dielectric,
                    pos, /* [natom, 3] */
                    factor, /* 4pi/V*unit-conv */
                    lambda, /* 4 * Lambda^2 */
                    tolerance);

  Py_RETURN_NONE;
}

static PyObject * py_get_dipole_dipole_q0(PyObject *self, PyObject *args)
{
  PyArrayObject* dd_q0_py;
  PyArrayObject* G_list_py;
  PyArrayObject* dielectric_py;
  PyArrayObject* pos_py;
  double lambda;
  double tolerance;

  double* dd_q0;
  double* G_list;
  double* dielectric;
  double *pos;
  int num_patom, num_G;

  if (!PyArg_ParseTuple(args, "OOOOdd",
                        &dd_q0_py,
                        &G_list_py,
                        &dielectric_py,
                        &pos_py,
                        &lambda,
                        &tolerance))
    return NULL;


  dd_q0 = (double*)PyArray_DATA(dd_q0_py);
  G_list = (double*)PyArray_DATA(G_list_py);
  dielectric = (double*)PyArray_DATA(dielectric_py);
  pos = (double*)PyArray_DATA(pos_py);
  num_G = PyArray_DIMS(G_list_py)[0];
  num_patom = PyArray_DIMS(pos_py)[0];

  get_dipole_dipole_q0(dd_q0, /* [natom, 3, 3, (real, imag)] */
                       G_list, /* [num_kvec, 3] */
                       num_G,
                       num_patom,
                       dielectric,
                       pos, /* [natom, 3] */
                       lambda, /* 4 * Lambda^2 */
                       tolerance);

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

  double* ddm;
  double* fc;
  double* q;
  double* lat;
  double* r;
  double* m;
  int* multi;
  int* s2p_map;
  int* p2s_map;
  int num_patom;
  int num_satom;

  double *z;
  double *epsilon;
  double *q_dir;

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

  ddm = (double*)PyArray_DATA(derivative_dynmat);
  fc = (double*)PyArray_DATA(force_constants);
  q = (double*)PyArray_DATA(q_vector);
  lat = (double*)PyArray_DATA(lattice);
  r = (double*)PyArray_DATA(r_vector);
  m = (double*)PyArray_DATA(mass);
  multi = (int*)PyArray_DATA(multiplicity);
  s2p_map = (int*)PyArray_DATA(super2prim_map);
  p2s_map = (int*)PyArray_DATA(prim2super_map);
  num_patom = PyArray_DIMS(prim2super_map)[0];
  num_satom = PyArray_DIMS(super2prim_map)[0];

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
  PyArrayObject* thermal_props_py;
  PyArrayObject* temperatures_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* weights_py;

  double *temperatures;
  double* freqs;
  double *thermal_props;
  int* w;
  int num_qpoints;
  int num_bands;
  int num_temp;

  int i, j, k;
  long sum_weights;
  double omega;
  double *tp;

  if (!PyArg_ParseTuple(args, "OOOO",
                        &thermal_props_py,
                        &temperatures_py,
                        &frequencies_py,
                        &weights_py)) {
    return NULL;
  }

  thermal_props = (double*)PyArray_DATA(thermal_props_py);
  temperatures = (double*)PyArray_DATA(temperatures_py);
  num_temp = PyArray_DIMS(temperatures_py)[0];
  freqs = (double*)PyArray_DATA(frequencies_py);
  num_qpoints = PyArray_DIMS(frequencies_py)[0];
  w = (int*)PyArray_DATA(weights_py);
  num_bands = PyArray_DIMS(frequencies_py)[1];

  for (i = 0; i < num_temp * 3; i++) {
    thermal_props[i] = 0;
  }

  tp = (double*)malloc(sizeof(double) * num_qpoints * num_temp * 3);
  for (i = 0; i < num_qpoints * num_temp * 3; i++) {
    tp[i] = 0;
  }

#pragma omp parallel for private(j, k, omega)
  for (i = 0; i < num_qpoints; i++){
    for (j = 0; j < num_temp; j++) {
      for (k = 0; k < num_bands; k++){
        omega = freqs[i * num_bands + k];
        if (temperatures[j] > 0 && omega > 0.0) {
          tp[i * num_temp * 3 + j * 3] +=
            get_free_energy_omega(temperatures[j], omega) * w[i];
          tp[i * num_temp * 3 + j * 3 + 1] +=
            get_entropy_omega(temperatures[j], omega) * w[i];
          tp[i * num_temp * 3 + j * 3 + 2] +=
            get_heat_capacity_omega(temperatures[j], omega)* w[i];
        }
      }
    }
  }

  for (i = 0; i < num_temp * 3; i++) {
    for (j = 0; j < num_qpoints; j++) {
      thermal_props[i] += tp[j * num_temp * 3 + i];
    }
  }

  free(tp);

  sum_weights = 0;
#pragma omp parallel for reduction(+:sum_weights)
  for (i = 0; i < num_qpoints; i++){
    sum_weights += w[i];
  }

  for (i = 0; i < num_temp * 3; i++) {
    thermal_props[i] /= sum_weights;
  }


  Py_RETURN_NONE;
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

static int compute_permutation(int * rot_atom,
                               PHPYCONST double lat[3][3],
                               PHPYCONST double (*pos)[3],
                               PHPYCONST double (*rot_pos)[3],
                               const int num_pos,
                               const double symprec)
{
  int i,j,k,l;
  int search_start;
  double distance2, diff_cart;
  double diff[3];

  for (i = 0; i < num_pos; i++) {
    rot_atom[i] = -1;
  }

  /* optimization: Iterate primarily by pos instead of rot_pos. */
  /*  (find where 0 belongs in rot_atom, then where 1 belongs, etc.) */
  /*  Then track the first unassigned index. */
  /* */
  /* This works best if the permutation is close to the identity. */
  /* (more specifically, if the max value of 'rot_atom[i] - i' is small) */
  search_start = 0;
  for (i = 0; i < num_pos; i++) {
    while (rot_atom[search_start] >= 0) {
      search_start++;
    }
    for (j = search_start; j < num_pos; j++) {
      if (rot_atom[j] >= 0) {
        continue;
      }

      for (k = 0; k < 3; k++) {
        diff[k] = pos[i][k] - rot_pos[j][k];
        diff[k] -= nint(diff[k]);
      }
      distance2 = 0;
      for (k = 0; k < 3; k++) {
        diff_cart = 0;
        for (l = 0; l < 3; l++) {
          diff_cart += lat[k][l] * diff[l];
        }
        distance2 += diff_cart * diff_cart;
      }

      if (sqrt(distance2) < symprec) {
        rot_atom[j] = i;
        break;
      }
    }
  }

  for (i = 0; i < num_pos; i++) {
    if (rot_atom[i] < 0) {
      printf("Encounter some problem in compute_permutation.\n");
      return 0;
    }
  }
  return 1;
}

/* Implementation detail of get_smallest_vectors. */
/* Finds the smallest vectors within each list and copies them to the output. */
static void gsv_copy_smallest_vectors(double (*shortest_vectors)[27][3],
                                      int * multiplicity,
                                      PHPYCONST double (*vector_lists)[27][3],
                                      PHPYCONST double (*length_lists)[27],
                                      const int num_lists,
                                      const double symprec)
{
  int i,j,k;
  int count;
  double minimum;
  double (*vectors)[3];
  double * lengths;

  for (i = 0; i < num_lists; i++) {
    /* Look at a single list of 27 vectors. */
    lengths = length_lists[i];
    vectors = vector_lists[i];

    /* Compute the minimum length. */
    minimum = DBL_MAX;
    for (j = 0; j < 27; j++) {
      if (lengths[j] < minimum) {
        minimum = lengths[j];
      }
    }

    /* Copy vectors whose length is within tolerance. */
    count = 0;
    for (j = 0; j < 27; j++) {
      if (lengths[j] - minimum <= symprec) {
        for (k = 0; k < 3; k++) {
          shortest_vectors[i][count][k] = vectors[j][k];
        }
        count++;
      }
    }

    multiplicity[i] = count;
  }
}

static PyObject * py_distribute_fc2_with_mappings(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants_py;
  PyArrayObject* permutations_py;
  PyArrayObject* map_atoms_py;
  PyArrayObject* map_syms_py;
  PyArrayObject* atom_list_py;
  PyArrayObject* rotations_cart_py;

  double (*r_carts)[3][3];
  double (*fc2)[3][3];
  int *permutations;
  int *map_atoms;
  int *map_syms;
  int *atom_list;
  int num_pos, num_rot, len_atom_list;

  if (!PyArg_ParseTuple(args, "OOOOOO",
                        &force_constants_py,
                        &atom_list_py,
                        &rotations_cart_py,
                        &permutations_py,
                        &map_atoms_py,
                        &map_syms_py)) {
    return NULL;
  }

  fc2 = (double(*)[3][3])PyArray_DATA(force_constants_py);
  atom_list = (int*)PyArray_DATA(atom_list_py);
  len_atom_list = PyArray_DIMS(atom_list_py)[0];
  permutations = (int*)PyArray_DATA(permutations_py);
  map_atoms = (int*)PyArray_DATA(map_atoms_py);
  map_syms = (int*)PyArray_DATA(map_syms_py);
  r_carts = (double(*)[3][3])PyArray_DATA(rotations_cart_py);
  num_rot = PyArray_DIMS(permutations_py)[0];
  num_pos = PyArray_DIMS(permutations_py)[1];

  if (PyArray_NDIM(map_atoms_py) != 1 || PyArray_DIMS(map_atoms_py)[0] != num_pos)
  {
    PyErr_SetString(PyExc_ValueError, "wrong shape for map_atoms");
    return NULL;
  }

  if (PyArray_NDIM(map_syms_py) != 1 || PyArray_DIMS(map_syms_py)[0] != num_pos)
  {
    PyErr_SetString(PyExc_ValueError, "wrong shape for map_syms");
    return NULL;
  }

  if (PyArray_DIMS(rotations_cart_py)[0] != num_rot)
  {
    PyErr_SetString(PyExc_ValueError, "permutations and rotations are different length");
    return NULL;
  }

  distribute_fc2_with_mappings(fc2,
                               atom_list,
                               len_atom_list,
                               r_carts,
                               permutations,
                               map_atoms,
                               map_syms,
                               num_rot,
                               num_pos);
  Py_RETURN_NONE;
}

static PyObject *py_thm_neighboring_grid_points(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  int grid_point;

  int* relative_grid_points;
  int (*relative_grid_address)[3];
  int num_relative_grid_address;
  int *mesh;
  int (*bz_grid_address)[3];
  int *bz_map;

  if (!PyArg_ParseTuple(args, "OiOOOO",
                        &relative_grid_points_py,
                        &grid_point,
                        &relative_grid_address_py,
                        &mesh_py,
                        &bz_grid_address_py,
                        &bz_map_py)) {
    return NULL;
  }

  relative_grid_points = (int*)PyArray_DATA(relative_grid_points_py);
  relative_grid_address = (int(*)[3])PyArray_DATA(relative_grid_address_py);
  num_relative_grid_address = PyArray_DIMS(relative_grid_address_py)[0];
  mesh = (int*)PyArray_DATA(mesh_py);
  bz_grid_address = (int(*)[3])PyArray_DATA(bz_grid_address_py);
  bz_map = (int*)PyArray_DATA(bz_map_py);

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

  int (*relative_grid_address)[4][3];
  double (*reciprocal_lattice)[3];

  if (!PyArg_ParseTuple(args, "OO",
                        &relative_grid_address_py,
                        &reciprocal_lattice_py)) {
    return NULL;
  }

  relative_grid_address = (int(*)[4][3])PyArray_DATA(relative_grid_address_py);
  reciprocal_lattice = (double(*)[3])PyArray_DATA(reciprocal_lattice_py);

  thm_get_relative_grid_address(relative_grid_address, reciprocal_lattice);

  Py_RETURN_NONE;
}

static PyObject *
py_thm_all_relative_grid_address(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_address_py;

  int (*relative_grid_address)[24][4][3];

  if (!PyArg_ParseTuple(args, "O",
                        &relative_grid_address_py)) {
    return NULL;
  }

  relative_grid_address =
    (int(*)[24][4][3])PyArray_DATA(relative_grid_address_py);

  thm_get_all_relative_grid_address(relative_grid_address);

  Py_RETURN_NONE;
}

static PyObject *
py_thm_integration_weight(PyObject *self, PyObject *args)
{
  double omega;
  PyArrayObject* tetrahedra_omegas_py;
  char* function;

  double (*tetrahedra_omegas)[4];
  double iw;

  if (!PyArg_ParseTuple(args, "dOs",
                        &omega,
                        &tetrahedra_omegas_py,
                        &function)) {
    return NULL;
  }

  tetrahedra_omegas = (double(*)[4])PyArray_DATA(tetrahedra_omegas_py);

  iw = thm_get_integration_weight(omega,
                                  tetrahedra_omegas,
                                  function[0]);

  return PyFloat_FromDouble(iw);
}

static PyObject *
py_thm_integration_weight_at_omegas(PyObject *self, PyObject *args)
{
  PyArrayObject* integration_weights_py;
  PyArrayObject* omegas_py;
  PyArrayObject* tetrahedra_omegas_py;
  char* function;

  double *omegas;
  double *iw;
  int num_omegas;
  double (*tetrahedra_omegas)[4];

  if (!PyArg_ParseTuple(args, "OOOs",
                        &integration_weights_py,
                        &omegas_py,
                        &tetrahedra_omegas_py,
                        &function)) {
    return NULL;
  }

  omegas = (double*)PyArray_DATA(omegas_py);
  iw = (double*)PyArray_DATA(integration_weights_py);
  num_omegas = (int)PyArray_DIMS(omegas_py)[0];
  tetrahedra_omegas = (double(*)[4])PyArray_DATA(tetrahedra_omegas_py);

  thm_get_integration_weight_at_omegas(iw,
                                       num_omegas,
                                       omegas,
                                       tetrahedra_omegas,
                                       function[0]);

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

  double* freq_tetras;
  int* grid_points;
  int num_gp_in;
  int* mesh;
  int (*grid_address)[3];
  int* gp_ir_index;
  int (*relative_grid_address)[3];
  double* frequencies;
  int num_band;

  int is_shift[3] = {0, 0, 0};
  int i, j, k, gp;
  int g_addr[3];
  int address_double[3];

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

  freq_tetras = (double*)PyArray_DATA(freq_tetras_py);
  grid_points = (int*)PyArray_DATA(grid_points_py);
  num_gp_in = (int)PyArray_DIMS(grid_points_py)[0];
  mesh = (int*)PyArray_DATA(mesh_py);
  grid_address = (int(*)[3])PyArray_DATA(grid_address_py);
  gp_ir_index = (int*)PyArray_DATA(gp_ir_index_py);
  relative_grid_address = (int(*)[3])PyArray_DATA(relative_grid_address_py);
  frequencies = (double*)PyArray_DATA(frequencies_py);
  num_band = (int)PyArray_DIMS(frequencies_py)[1];

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

static PyObject * py_tetrahedron_method_dos(PyObject *self, PyObject *args)
{
  PyArrayObject* dos_py;
  PyArrayObject* mesh_py;
  PyArrayObject* freq_points_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* coef_py;
  PyArrayObject* grid_address_py;
  PyArrayObject* grid_mapping_table_py;
  PyArrayObject* relative_grid_address_py;

  double *dos;
  int* mesh;
  double* freq_points;
  int num_freq_points;
  double* frequencies;
  double* coef;
  int (*grid_address)[3];
  int num_gp;
  int num_ir_gp;
  int num_coef;
  int num_band;
  int* grid_mapping_table;
  int (*relative_grid_address)[4][3];

  int is_shift[3] = {0, 0, 0};
  int i, j, k, l, m, q, r, count;
  int g_addr[3];
  int ir_gps[24][4];
  double tetrahedra[24][4];
  int address_double[3];
  int *gp2ir, *ir_grid_points, *weights;
  double iw;

  gp2ir = NULL;
  ir_grid_points = NULL;
  weights = NULL;

  if (!PyArg_ParseTuple(args, "OOOOOOOO",
                        &dos_py,
                        &mesh_py,
                        &freq_points_py,
                        &frequencies_py,
                        &coef_py,
                        &grid_address_py,
                        &grid_mapping_table_py,
                        &relative_grid_address_py)) {
    return NULL;
  }

  /* dos[num_ir_gp][num_band][num_freq_points][num_coef] */
  dos = (double*)PyArray_DATA(dos_py);
  mesh = (int*)PyArray_DATA(mesh_py);
  freq_points = (double*)PyArray_DATA(freq_points_py);
  num_freq_points = (int)PyArray_DIMS(freq_points_py)[0];
  frequencies = (double*)PyArray_DATA(frequencies_py);
  num_ir_gp = (int)PyArray_DIMS(frequencies_py)[0];
  num_band = (int)PyArray_DIMS(frequencies_py)[1];
  coef = (double*)PyArray_DATA(coef_py);
  num_coef = (int)PyArray_DIMS(coef_py)[1];
  grid_address = (int(*)[3])PyArray_DATA(grid_address_py);
  num_gp = (int)PyArray_DIMS(grid_address_py)[0];
  grid_mapping_table = (int*)PyArray_DATA(grid_mapping_table_py);
  relative_grid_address = (int(*)[4][3])PyArray_DATA(relative_grid_address_py);

  gp2ir = (int*)malloc(sizeof(int) * num_gp);
  ir_grid_points = (int*)malloc(sizeof(int) * num_ir_gp);
  weights = (int*)malloc(sizeof(int) * num_ir_gp);

  count = 0;
  for (i = 0; i < num_gp; i++) {
    if (grid_mapping_table[i] == i) {
      gp2ir[i] = count;
      ir_grid_points[count] = i;
      weights[count] = 1;
      count++;
    } else {
      gp2ir[i] = gp2ir[grid_mapping_table[i]];
      weights[gp2ir[i]]++;
    }
  }

  if (num_ir_gp != count) {
    printf("Something is wrong!\n");
  }

#pragma omp parallel for private(j, k, l, m, q, r, iw, ir_gps, g_addr, tetrahedra, address_double)
  for (i = 0; i < num_ir_gp; i++) {
    /* set 24 tetrahedra */
    for (l = 0; l < 24; l++) {
      for (q = 0; q < 4; q++) {
        for (r = 0; r < 3; r++) {
          g_addr[r] = grid_address[ir_grid_points[i]][r] +
            relative_grid_address[l][q][r];
        }
        kgd_get_grid_address_double_mesh(address_double,
                                         g_addr,
                                         mesh,
                                         is_shift);
        ir_gps[l][q] = gp2ir[kgd_get_grid_point_double_mesh(address_double, mesh)];
      }
    }

    for (k = 0; k < num_band; k++) {
      for (l = 0; l < 24; l++) {
        for (q = 0; q < 4; q++) {
          tetrahedra[l][q] = frequencies[ir_gps[l][q] * num_band + k];
        }
      }
      for (j = 0; j < num_freq_points; j++) {
        iw = thm_get_integration_weight(freq_points[j], tetrahedra, 'I') * weights[i];
        for (m = 0; m < num_coef; m++) {
          dos[i * num_band * num_freq_points * num_coef +
              k * num_coef * num_freq_points + j * num_coef + m] +=
            iw * coef[i * num_coef * num_band + m * num_band + k];
        }
      }
    }
  }

  free(gp2ir);
  gp2ir = NULL;
  free(ir_grid_points);
  ir_grid_points = NULL;
  free(weights);
  weights = NULL;

  Py_RETURN_NONE;
}

/* Distributes all force constants using precomputed data about symmetry mappings. */
static void
distribute_fc2_with_mappings(double (*fc2)[3][3], /* shape[num_pos][num_pos] */
                             const int * atom_list,
                             const int len_atom_list,
                             PHPYCONST double (*r_carts)[3][3], /* shape[num_rot] */
                             const int * permutations, /* shape[num_rot][num_pos] */
                             const int * map_atoms, /* shape [num_pos] */
                             const int * map_syms, /* shape [num_pos] */
                             const int num_rot,
                             const int num_pos)
{
  int i, j, k, l, m;
  int atom_todo, atom_done, atom_other;
  int sym_index;
  double (*fc2_done)[3];
  double (*fc2_todo)[3];
  double (*r_cart)[3];
  const int * permutation;

  for (i = 0; i < len_atom_list; i++) {
    /* look up how this atom maps into the done list. */
    atom_todo = atom_list[i];
    atom_done = map_atoms[atom_todo];
    sym_index = map_syms[atom_todo];

    /* skip the atoms in the done list, */
    /* which are easily identified because they map to themselves. */
    if (atom_todo == atom_done) {
      continue;
    }

    /* look up information about the rotation */
    r_cart = r_carts[sym_index];
    permutation = &permutations[sym_index * num_pos]; /* shape[num_pos] */

    /* distribute terms from atom_done to atom_todo */
    for (atom_other = 0; atom_other < num_pos; atom_other++) {
      fc2_done = fc2[atom_done * num_pos + permutation[atom_other]];
      fc2_todo = fc2[atom_todo * num_pos + atom_other];
      for (j = 0; j < 3; j++) {
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            for (m = 0; m < 3; m++) {
              /* P' = R^-1 P R */
              fc2_todo[j][k] += r_cart[l][j] * r_cart[m][k] * fc2_done[l][m];
            }
          }
        }
      }
    }
  }
}

static void set_index_permutation_symmetry_fc(double * fc,
                                              const int natom)
{
  int i, j, k, l, m, n;

#pragma omp parallel for private(j, k, l, m, n)
  for (i = 0; i < natom; i++) {
    /* non diagonal part */
    for (j = i + 1; j < natom; j++) {
      for (k = 0; k < 3; k++) {
        for (l = 0; l < 3; l++) {
          m = i * natom * 9 + j * 9 + k * 3 + l;
          n = j * natom * 9 + i * 9 + l * 3 + k;
          fc[m] += fc[n];
          fc[m] /= 2;
          fc[n] = fc[m];
        }
      }
    }

    /* diagnoal part */
    for (k = 0; k < 2; k++) {
      for (l = k + 1; l < 3; l++) {
        m = i * natom * 9 + i * 9 + k * 3 + l;
        n = i * natom * 9 + i * 9 + l * 3 + k;
        fc[m] += fc[n];
        fc[m] /= 2;
        fc[n] = fc[m];
      }
    }
  }
}

static void set_translational_symmetry_fc(double * fc,
                                          const int natom)
{
  int i, j, k, l, m;
  double sums[3][3];

  for (i = 0; i < natom; i++) {
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        sums[k][l] = 0;
        m = i * natom * 9 + k * 3 + l;
        for (j = 0; j < natom; j++) {
          if (i != j) {
            sums[k][l] += fc[m];
          }
          m += 9;
        }
      }
    }
    for (k = 0; k < 3; k++) {
      for (l = 0; l < 3; l++) {
        fc[i * natom * 9 + i * 9 + k * 3 + l] = -(sums[k][l] + sums[l][k]) / 2;
      }
    }
  }
}

static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}
