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
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdio.h>

#include "phonopy.h"

/* Build dynamical matrix */
static PyObject* py_transform_dynmat_to_fc(PyObject* self, PyObject* args);
static PyObject* py_perm_trans_symmetrize_fc(PyObject* self, PyObject* args);
static PyObject* py_perm_trans_symmetrize_compact_fc(PyObject* self,
                                                     PyObject* args);
static PyObject* py_transpose_compact_fc(PyObject* self, PyObject* args);
static PyObject* py_get_dynamical_matrices_with_dd_openmp_over_qpoints(
    PyObject* self, PyObject* args);
static PyObject* py_get_dynamical_matrix(PyObject* self, PyObject* args);
static PyObject* py_get_nac_dynamical_matrix(PyObject* self, PyObject* args);
static PyObject* py_get_recip_dipole_dipole(PyObject* self, PyObject* args);
static PyObject* py_get_recip_dipole_dipole_q0(PyObject* self, PyObject* args);
static PyObject* py_get_derivative_dynmat(PyObject* self, PyObject* args);
static PyObject* py_get_thermal_properties(PyObject* self, PyObject* args);
static PyObject* py_distribute_fc2(PyObject* self, PyObject* args);
static PyObject* py_compute_permutation(PyObject* self, PyObject* args);
static PyObject* py_gsv_set_smallest_vectors_sparse(PyObject* self,
                                                    PyObject* args);
static PyObject* py_gsv_set_smallest_vectors_dense(PyObject* self,
                                                   PyObject* args);
static PyObject* py_thm_relative_grid_address(PyObject* self, PyObject* args);
static PyObject* py_thm_all_relative_grid_address(PyObject* self,
                                                  PyObject* args);
static PyObject* py_thm_integration_weight(PyObject* self, PyObject* args);
static PyObject* py_thm_integration_weight_at_omegas(PyObject* self,
                                                     PyObject* args);
static PyObject* py_get_tetrahedra_frequenies(PyObject* self, PyObject* args);
static PyObject* py_tetrahedron_method_dos(PyObject* self, PyObject* args);
static PyObject* py_use_openmp(PyObject* self, PyObject* args);
static PyObject* py_get_omp_max_threads(PyObject* self, PyObject* args);
struct module_state {
    PyObject* error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject* error_out(PyObject* m) {
    struct module_state* st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}

static PyMethodDef _phonopy_methods[] = {
    {"error_out", (PyCFunction)error_out, METH_NOARGS, NULL},
    {"transform_dynmat_to_fc", py_transform_dynmat_to_fc, METH_VARARGS,
     "Transform a set of dynmat to force constants"},
    {"perm_trans_symmetrize_fc", py_perm_trans_symmetrize_fc, METH_VARARGS,
     "Enforce permutation and translational symmetry of force constants"},
    {"perm_trans_symmetrize_compact_fc", py_perm_trans_symmetrize_compact_fc,
     METH_VARARGS,
     "Enforce permutation and translational symmetry of compact force "
     "constants"},
    {"transpose_compact_fc", py_transpose_compact_fc, METH_VARARGS,
     "Transpose compact force constants"},
    {"dynamical_matrices_with_dd_openmp_over_qpoints",
     py_get_dynamical_matrices_with_dd_openmp_over_qpoints, METH_VARARGS,
     "Get dynamical matrices plus dd with openmp over qpoints"},
    {"dynamical_matrix", py_get_dynamical_matrix, METH_VARARGS,
     "Dynamical matrix"},
    {"nac_dynamical_matrix", py_get_nac_dynamical_matrix, METH_VARARGS,
     "NAC dynamical matrix"},
    {"recip_dipole_dipole", py_get_recip_dipole_dipole, METH_VARARGS,
     "Reciprocal part of dipole-dipole interaction"},
    {"recip_dipole_dipole_q0", py_get_recip_dipole_dipole_q0, METH_VARARGS,
     "q=0 terms of reciprocal part of dipole-dipole interaction"},
    {"derivative_dynmat", py_get_derivative_dynmat, METH_VARARGS,
     "Q derivative of dynamical matrix"},
    {"thermal_properties", py_get_thermal_properties, METH_VARARGS,
     "Thermal properties"},
    {"distribute_fc2", py_distribute_fc2, METH_VARARGS,
     "Distribute force constants for all atoms in atom_list using precomputed "
     "symmetry mappings."},
    {"compute_permutation", py_compute_permutation, METH_VARARGS,
     "Compute indices of original points in a set of rotated points."},
    {"gsv_set_smallest_vectors_sparse", py_gsv_set_smallest_vectors_sparse,
     METH_VARARGS, "Set shortest vectors in sparse array."},
    {"gsv_set_smallest_vectors_dense", py_gsv_set_smallest_vectors_dense,
     METH_VARARGS, "Set shortest vectors in dense array."},
    {"tetrahedra_relative_grid_address", py_thm_relative_grid_address,
     METH_VARARGS, "Relative grid addresses of vertices of 24 tetrahedra"},
    {"all_tetrahedra_relative_grid_address", py_thm_all_relative_grid_address,
     METH_VARARGS,
     "4 (all) sets of relative grid addresses of vertices of 24 tetrahedra"},
    {"tetrahedra_integration_weight", py_thm_integration_weight, METH_VARARGS,
     "Integration weight for tetrahedron method"},
    {"tetrahedra_integration_weight_at_omegas",
     py_thm_integration_weight_at_omegas, METH_VARARGS,
     "Integration weight for tetrahedron method at omegas"},
    {"tetrahedra_frequencies", py_get_tetrahedra_frequenies, METH_VARARGS,
     "Run tetrahedron method"},
    {"tetrahedron_method_dos", py_tetrahedron_method_dos, METH_VARARGS,
     "Run tetrahedron method"},
    {"use_openmp", py_use_openmp, METH_VARARGS, "Use OpenMP or not"},
    {"omp_max_threads", py_get_omp_max_threads, METH_VARARGS,
     "Return openmp max number of threads. Return 0 unless openmp is "
     "activated. "},
    {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3

static int _phonopy_traverse(PyObject* m, visitproc visit, void* arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int _phonopy_clear(PyObject* m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,       "_phonopy",       NULL,
    sizeof(struct module_state), _phonopy_methods, NULL,
    _phonopy_traverse,           _phonopy_clear,   NULL};

#define INITERROR return NULL

PyObject* PyInit__phonopy(void)
#else
#define INITERROR return

void init_phonopy(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject* module = PyModule_Create(&moduledef);
#else
    PyObject* module = Py_InitModule("_phonopy", _phonopy_methods);
#endif
    struct module_state* st;
    if (module == NULL) INITERROR;
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

static PyObject* py_transform_dynmat_to_fc(PyObject* self, PyObject* args) {
    PyArrayObject* py_force_constants;
    PyArrayObject* py_dynamical_matrices;
    PyArrayObject* py_commensurate_points;
    PyArrayObject* py_svecs;
    PyArrayObject* py_multi;
    PyArrayObject* py_masses;
    PyArrayObject* py_s2pp_map;
    PyArrayObject* py_fc_index_map;
    long use_openmp;

    double* fc;
    double(*dm)[2];
    double(*comm_points)[3];
    double(*svecs)[3];
    double* masses;
    long(*multi)[2];
    long* s2pp_map;
    long* fc_index_map;
    long num_patom;
    long num_satom;

    if (!PyArg_ParseTuple(args, "OOOOOOOOl", &py_force_constants,
                          &py_dynamical_matrices, &py_commensurate_points,
                          &py_svecs, &py_multi, &py_masses, &py_s2pp_map,
                          &py_fc_index_map, &use_openmp)) {
        return NULL;
    }

    fc = (double*)PyArray_DATA(py_force_constants);
    dm = (double(*)[2])PyArray_DATA(py_dynamical_matrices);
    comm_points = (double(*)[3])PyArray_DATA(py_commensurate_points);
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    masses = (double*)PyArray_DATA(py_masses);
    multi = (long(*)[2])PyArray_DATA(py_multi);
    s2pp_map = (long*)PyArray_DATA(py_s2pp_map);
    fc_index_map = (long*)PyArray_DATA(py_fc_index_map);
    num_patom = PyArray_DIMS(py_multi)[1];
    num_satom = PyArray_DIMS(py_multi)[0];

    phpy_transform_dynmat_to_fc(fc, dm, comm_points, svecs, multi, masses,
                                s2pp_map, fc_index_map, num_patom, num_satom,
                                use_openmp);

    Py_RETURN_NONE;
}

static PyObject* py_compute_permutation(PyObject* self, PyObject* args) {
    PyArrayObject* permutation;
    PyArrayObject* lattice;
    PyArrayObject* positions;
    PyArrayObject* permuted_positions;
    double symprec;

    int* rot_atoms;
    double(*lat)[3];
    double(*pos)[3];
    double(*rot_pos)[3];
    int num_pos;

    int is_found;

    if (!PyArg_ParseTuple(args, "OOOOd", &permutation, &lattice, &positions,
                          &permuted_positions, &symprec)) {
        return NULL;
    }

    rot_atoms = (int*)PyArray_DATA(permutation);
    lat = (double(*)[3])PyArray_DATA(lattice);
    pos = (double(*)[3])PyArray_DATA(positions);
    rot_pos = (double(*)[3])PyArray_DATA(permuted_positions);
    num_pos = PyArray_DIMS(positions)[0];

    is_found = phpy_compute_permutation(rot_atoms, lat, pos, rot_pos, num_pos,
                                        symprec);

    if (is_found) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject* py_gsv_set_smallest_vectors_sparse(PyObject* self,
                                                    PyObject* args) {
    PyArrayObject* py_smallest_vectors;
    PyArrayObject* py_multiplicity;
    PyArrayObject* py_pos_to;
    PyArrayObject* py_pos_from;
    PyArrayObject* py_lattice_points;
    PyArrayObject* py_reduced_basis;
    PyArrayObject* py_trans_mat;
    double symprec;

    double(*smallest_vectors)[27][3];
    int* multiplicity;
    double(*pos_to)[3];
    double(*pos_from)[3];
    int(*lattice_points)[3];
    double(*reduced_basis)[3];
    int(*trans_mat)[3];
    int num_pos_to, num_pos_from, num_lattice_points;

    if (!PyArg_ParseTuple(args, "OOOOOOOd", &py_smallest_vectors,
                          &py_multiplicity, &py_pos_to, &py_pos_from,
                          &py_lattice_points, &py_reduced_basis, &py_trans_mat,
                          &symprec)) {
        return NULL;
    }

    smallest_vectors = (double(*)[27][3])PyArray_DATA(py_smallest_vectors);
    multiplicity = (int*)PyArray_DATA(py_multiplicity);
    pos_to = (double(*)[3])PyArray_DATA(py_pos_to);
    pos_from = (double(*)[3])PyArray_DATA(py_pos_from);
    num_pos_to = PyArray_DIMS(py_pos_to)[0];
    num_pos_from = PyArray_DIMS(py_pos_from)[0];
    lattice_points = (int(*)[3])PyArray_DATA(py_lattice_points);
    num_lattice_points = PyArray_DIMS(py_lattice_points)[0];
    reduced_basis = (double(*)[3])PyArray_DATA(py_reduced_basis);
    trans_mat = (int(*)[3])PyArray_DATA(py_trans_mat);

    phpy_set_smallest_vectors_sparse(smallest_vectors, multiplicity, pos_to,
                                     num_pos_to, pos_from, num_pos_from,
                                     lattice_points, num_lattice_points,
                                     reduced_basis, trans_mat, symprec);

    Py_RETURN_NONE;
}

static PyObject* py_gsv_set_smallest_vectors_dense(PyObject* self,
                                                   PyObject* args) {
    PyArrayObject* py_smallest_vectors;
    PyArrayObject* py_multiplicity;
    PyArrayObject* py_pos_to;
    PyArrayObject* py_pos_from;
    PyArrayObject* py_lattice_points;
    PyArrayObject* py_reduced_basis;
    PyArrayObject* py_trans_mat;
    long initialize;
    double symprec;

    double(*smallest_vectors)[3];
    long(*multiplicity)[2];
    double(*pos_to)[3];
    double(*pos_from)[3];
    long(*lattice_points)[3];
    double(*reduced_basis)[3];
    long(*trans_mat)[3];
    long num_pos_to, num_pos_from, num_lattice_points;

    if (!PyArg_ParseTuple(args, "OOOOOOOld", &py_smallest_vectors,
                          &py_multiplicity, &py_pos_to, &py_pos_from,
                          &py_lattice_points, &py_reduced_basis, &py_trans_mat,
                          &initialize, &symprec)) {
        return NULL;
    }

    smallest_vectors = (double(*)[3])PyArray_DATA(py_smallest_vectors);
    multiplicity = (long(*)[2])PyArray_DATA(py_multiplicity);
    pos_to = (double(*)[3])PyArray_DATA(py_pos_to);
    pos_from = (double(*)[3])PyArray_DATA(py_pos_from);
    num_pos_to = PyArray_DIMS(py_pos_to)[0];
    num_pos_from = PyArray_DIMS(py_pos_from)[0];
    lattice_points = (long(*)[3])PyArray_DATA(py_lattice_points);
    num_lattice_points = PyArray_DIMS(py_lattice_points)[0];
    reduced_basis = (double(*)[3])PyArray_DATA(py_reduced_basis);
    trans_mat = (long(*)[3])PyArray_DATA(py_trans_mat);

    phpy_set_smallest_vectors_dense(
        smallest_vectors, multiplicity, pos_to, num_pos_to, pos_from,
        num_pos_from, lattice_points, num_lattice_points, reduced_basis,
        trans_mat, initialize, symprec);

    Py_RETURN_NONE;
}

static PyObject* py_perm_trans_symmetrize_fc(PyObject* self, PyObject* args) {
    PyArrayObject* py_fc;
    double* fc;
    int level;

    int n_satom;

    if (!PyArg_ParseTuple(args, "Oi", &py_fc, &level)) {
        return NULL;
    }

    fc = (double*)PyArray_DATA(py_fc);
    n_satom = PyArray_DIMS(py_fc)[0];

    phpy_perm_trans_symmetrize_fc(fc, n_satom, level);

    Py_RETURN_NONE;
}

static PyObject* py_perm_trans_symmetrize_compact_fc(PyObject* self,
                                                     PyObject* args) {
    PyArrayObject* py_fc;
    PyArrayObject* py_permutations;
    PyArrayObject* py_s2pp_map;
    PyArrayObject* py_p2s_map;
    PyArrayObject* py_nsym_list;
    int level;
    double* fc;
    int* perms;
    int* s2pp;
    int* p2s;
    int* nsym_list;

    int n_patom, n_satom;

    if (!PyArg_ParseTuple(args, "OOOOOi", &py_fc, &py_permutations,
                          &py_s2pp_map, &py_p2s_map, &py_nsym_list, &level)) {
        return NULL;
    }

    fc = (double*)PyArray_DATA(py_fc);
    perms = (int*)PyArray_DATA(py_permutations);
    s2pp = (int*)PyArray_DATA(py_s2pp_map);
    p2s = (int*)PyArray_DATA(py_p2s_map);
    nsym_list = (int*)PyArray_DATA(py_nsym_list);
    n_patom = PyArray_DIMS(py_fc)[0];
    n_satom = PyArray_DIMS(py_fc)[1];

    phpy_perm_trans_symmetrize_compact_fc(fc, p2s, s2pp, nsym_list, perms,
                                          n_satom, n_patom, level);

    Py_RETURN_NONE;
}

static PyObject* py_transpose_compact_fc(PyObject* self, PyObject* args) {
    PyArrayObject* py_fc;
    PyArrayObject* py_permutations;
    PyArrayObject* py_s2pp_map;
    PyArrayObject* py_p2s_map;
    PyArrayObject* py_nsym_list;
    double* fc;
    int* s2pp;
    int* p2s;
    int* nsym_list;
    int* perms;
    int n_patom, n_satom;

    if (!PyArg_ParseTuple(args, "OOOOO", &py_fc, &py_permutations, &py_s2pp_map,
                          &py_p2s_map, &py_nsym_list)) {
        return NULL;
    }

    fc = (double*)PyArray_DATA(py_fc);
    perms = (int*)PyArray_DATA(py_permutations);
    s2pp = (int*)PyArray_DATA(py_s2pp_map);
    p2s = (int*)PyArray_DATA(py_p2s_map);
    nsym_list = (int*)PyArray_DATA(py_nsym_list);
    n_patom = PyArray_DIMS(py_fc)[0];
    n_satom = PyArray_DIMS(py_fc)[1];

    phpy_set_index_permutation_symmetry_compact_fc(fc, p2s, s2pp, nsym_list,
                                                   perms, n_satom, n_patom, 1);

    Py_RETURN_NONE;
}

static PyObject* py_get_dynamical_matrix(PyObject* self, PyObject* args) {
    PyArrayObject* py_dynamical_matrix;
    PyArrayObject* py_force_constants;
    PyArrayObject* py_svecs;
    PyArrayObject* py_q;
    PyArrayObject* py_multi;
    PyArrayObject* py_masses;
    PyArrayObject* py_s2p_map;
    PyArrayObject* py_p2s_map;
    long use_openmp;

    double(*dm)[2];
    double* fc;
    double* q;
    double(*svecs)[3];
    double* m;
    long(*multi)[2];
    long* s2p_map;
    long* p2s_map;
    long num_patom;
    long num_satom;

    if (!PyArg_ParseTuple(args, "OOOOOOOOl", &py_dynamical_matrix,
                          &py_force_constants, &py_q, &py_svecs, &py_multi,
                          &py_masses, &py_s2p_map, &py_p2s_map, &use_openmp)) {
        return NULL;
    }

    dm = (double(*)[2])PyArray_DATA(py_dynamical_matrix);
    fc = (double*)PyArray_DATA(py_force_constants);
    q = (double*)PyArray_DATA(py_q);
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    m = (double*)PyArray_DATA(py_masses);
    multi = (long(*)[2])PyArray_DATA(py_multi);
    s2p_map = (long*)PyArray_DATA(py_s2p_map);
    p2s_map = (long*)PyArray_DATA(py_p2s_map);
    num_patom = PyArray_DIMS(py_p2s_map)[0];
    num_satom = PyArray_DIMS(py_s2p_map)[0];

    if (PyArray_NDIM(py_q) == 1) {
        phpy_get_dynamical_matrix_at_q(dm, num_patom, num_satom, fc, q, svecs,
                                       multi, m, s2p_map, p2s_map, NULL,
                                       use_openmp);
    } else {
        phpy_get_dynamical_matrices_openmp_over_qpoints(
            dm, num_patom, num_satom, fc, (double(*)[3])q,
            PyArray_DIMS(py_q)[0], svecs, multi, m, s2p_map, p2s_map, NULL);
    }

    Py_RETURN_NONE;
}

static PyObject* py_get_nac_dynamical_matrix(PyObject* self, PyObject* args) {
    PyArrayObject* py_dynamical_matrix;
    PyArrayObject* py_force_constants;
    PyArrayObject* py_svecs;
    PyArrayObject* py_q_cart;
    PyArrayObject* py_q;
    PyArrayObject* py_multi;
    PyArrayObject* py_masses;
    PyArrayObject* py_s2p_map;
    PyArrayObject* py_p2s_map;
    PyArrayObject* py_born;
    double factor;
    long use_openmp;

    double(*dm)[2];
    double* fc;
    double* q_cart;
    double* q;
    double(*svecs)[3];
    double* m;
    double(*born)[3][3];
    long(*multi)[2];
    long* s2p_map;
    long* p2s_map;
    long num_patom;
    long num_satom;

    long n;
    double(*charge_sum)[3][3];

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOdl", &py_dynamical_matrix,
                          &py_force_constants, &py_q, &py_svecs, &py_multi,
                          &py_masses, &py_s2p_map, &py_p2s_map, &py_q_cart,
                          &py_born, &factor, &use_openmp))
        return NULL;

    dm = (double(*)[2])PyArray_DATA(py_dynamical_matrix);
    fc = (double*)PyArray_DATA(py_force_constants);
    q_cart = (double*)PyArray_DATA(py_q_cart);
    q = (double*)PyArray_DATA(py_q);
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    m = (double*)PyArray_DATA(py_masses);
    born = (double(*)[3][3])PyArray_DATA(py_born);
    multi = (long(*)[2])PyArray_DATA(py_multi);
    s2p_map = (long*)PyArray_DATA(py_s2p_map);
    p2s_map = (long*)PyArray_DATA(py_p2s_map);
    num_patom = PyArray_DIMS(py_p2s_map)[0];
    num_satom = PyArray_DIMS(py_s2p_map)[0];

    charge_sum =
        (double(*)[3][3])malloc(sizeof(double[3][3]) * num_patom * num_patom);
    n = num_satom / num_patom;

    phpy_get_charge_sum(charge_sum, num_patom, factor / n, q_cart, born);

    if (PyArray_NDIM(py_q) == 1) {
        phpy_get_dynamical_matrix_at_q(dm, num_patom, num_satom, fc, q, svecs,
                                       multi, m, s2p_map, p2s_map, charge_sum,
                                       use_openmp);
    } else {
        phpy_get_dynamical_matrices_openmp_over_qpoints(
            dm, num_patom, num_satom, fc, (double(*)[3])q,
            PyArray_DIMS(py_q)[0], svecs, multi, m, s2p_map, p2s_map,
            charge_sum);
    }

    free(charge_sum);
    charge_sum = NULL;

    Py_RETURN_NONE;
}

static PyObject* py_get_dynamical_matrices_with_dd_openmp_over_qpoints(
    PyObject* self, PyObject* args) {
    PyArrayObject* py_dynamical_matrix;
    PyArrayObject* py_qpoints;
    PyArrayObject* py_force_constants;
    PyArrayObject* py_svecs;
    PyArrayObject* py_multi;
    PyArrayObject* py_positions;
    PyArrayObject* py_q_direction;
    PyArrayObject* py_masses;
    PyArrayObject* py_s2p_map;
    PyArrayObject* py_p2s_map;
    PyArrayObject* py_born;
    PyArrayObject* py_dielectric;
    PyArrayObject* py_reciprocal_lattice;
    PyArrayObject* py_dd_q0;
    PyArrayObject* py_G_list;
    double nac_factor;
    double lambda;
    long use_Wang_NAC;

    double(*dm)[2];
    double* fc;
    double* q_direction;
    double(*qpoints)[3];
    double(*svecs)[3];
    long(*multi)[2];
    double(*positions)[3];
    double* masses;
    double(*born)[3][3];
    double(*dielectric)[3];
    double(*reciprocal_lattice)[3];
    double(*dd_q0)[2];
    double(*G_list)[3];

    long* s2p_map;
    long* p2s_map;
    long num_patom;
    long num_satom;
    long n_qpoints;
    long n_Gpoints;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOdOOdl", &py_dynamical_matrix,
                          &py_qpoints, &py_force_constants, &py_svecs,
                          &py_multi, &py_positions, &py_masses, &py_s2p_map,
                          &py_p2s_map, &py_q_direction, &py_born,
                          &py_dielectric, &py_reciprocal_lattice, &nac_factor,
                          &py_dd_q0, &py_G_list, &lambda, &use_Wang_NAC))
        return NULL;

    dm = (double(*)[2])PyArray_DATA(py_dynamical_matrix);
    qpoints = (double(*)[3])PyArray_DATA(py_qpoints);
    n_qpoints = PyArray_DIMS(py_qpoints)[0];
    fc = (double*)PyArray_DATA(py_force_constants);
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    multi = (long(*)[2])PyArray_DATA(py_multi);
    if ((PyObject*)py_positions == Py_None) {
        positions = NULL;
    } else {
        positions = (double(*)[3])PyArray_DATA(py_positions);
    }
    masses = (double*)PyArray_DATA(py_masses);
    s2p_map = (long*)PyArray_DATA(py_s2p_map);
    p2s_map = (long*)PyArray_DATA(py_p2s_map);
    if ((PyObject*)py_q_direction == Py_None) {
        q_direction = NULL;
    } else {
        q_direction = (double*)PyArray_DATA(py_q_direction);
    }
    if ((PyObject*)py_born == Py_None) {
        born = NULL;
    } else {
        born = (double(*)[3][3])PyArray_DATA(py_born);
    }
    if ((PyObject*)py_dielectric == Py_None) {
        dielectric = NULL;
    } else {
        dielectric = (double(*)[3])PyArray_DATA(py_dielectric);
    }
    if ((PyObject*)py_reciprocal_lattice == Py_None) {
        reciprocal_lattice = NULL;
    } else {
        reciprocal_lattice = (double(*)[3])PyArray_DATA(py_reciprocal_lattice);
    }
    if ((PyObject*)py_dd_q0 == Py_None) {
        dd_q0 = NULL;
    } else {
        dd_q0 = (double(*)[2])PyArray_DATA(py_dd_q0);
    }
    if ((PyObject*)py_G_list == Py_None) {
        G_list = NULL;
        n_Gpoints = 0;
    } else {
        G_list = (double(*)[3])PyArray_DATA(py_G_list);
        n_Gpoints = PyArray_DIMS(py_G_list)[0];
    }
    num_patom = PyArray_DIMS(py_p2s_map)[0];
    num_satom = PyArray_DIMS(py_s2p_map)[0];

    phpy_dynamical_matrices_with_dd_openmp_over_qpoints(
        dm, qpoints, n_qpoints, fc, svecs, multi, positions, num_patom,
        num_satom, masses, p2s_map, s2p_map, born, dielectric,
        reciprocal_lattice, q_direction, nac_factor, dd_q0, G_list, n_Gpoints,
        lambda, use_Wang_NAC);

    Py_RETURN_NONE;
}

static PyObject* py_get_recip_dipole_dipole(PyObject* self, PyObject* args) {
    PyArrayObject* py_dd;
    PyArrayObject* py_dd_q0;
    PyArrayObject* py_G_list;
    PyArrayObject* py_q_cart;
    PyArrayObject* py_q_direction;
    PyArrayObject* py_born;
    PyArrayObject* py_dielectric;
    PyArrayObject* py_positions;
    double factor;
    double lambda;
    double tolerance;
    long use_openmp;

    double(*dd)[2];
    double(*dd_q0)[2];
    double(*G_list)[3];
    double* q_vector;
    double* q_direction;
    double(*born)[3][3];
    double(*dielectric)[3];
    double(*pos)[3];
    long num_patom, num_G;

    if (!PyArg_ParseTuple(args, "OOOOOOOOdddl", &py_dd, &py_dd_q0, &py_G_list,
                          &py_q_cart, &py_q_direction, &py_born, &py_dielectric,
                          &py_positions, &factor, &lambda, &tolerance,
                          &use_openmp))
        return NULL;

    dd = (double(*)[2])PyArray_DATA(py_dd);
    dd_q0 = (double(*)[2])PyArray_DATA(py_dd_q0);
    G_list = (double(*)[3])PyArray_DATA(py_G_list);
    if ((PyObject*)py_q_direction == Py_None) {
        q_direction = NULL;
    } else {
        q_direction = (double*)PyArray_DATA(py_q_direction);
    }
    q_vector = (double*)PyArray_DATA(py_q_cart);
    born = (double(*)[3][3])PyArray_DATA(py_born);
    dielectric = (double(*)[3])PyArray_DATA(py_dielectric);
    pos = (double(*)[3])PyArray_DATA(py_positions);
    num_G = PyArray_DIMS(py_G_list)[0];
    num_patom = PyArray_DIMS(py_positions)[0];

    phpy_get_recip_dipole_dipole(dd,    /* [natom, 3, natom, 3, (real, imag)] */
                                 dd_q0, /* [natom, 3, 3, (real, imag)] */
                                 G_list, /* [num_kvec, 3] */
                                 num_G, num_patom, q_vector, q_direction, born,
                                 dielectric, pos, /* [natom, 3] */
                                 factor,          /* 4pi/V*unit-conv */
                                 lambda,          /* 4 * Lambda^2 */
                                 tolerance, use_openmp);

    Py_RETURN_NONE;
}

static PyObject* py_get_recip_dipole_dipole_q0(PyObject* self, PyObject* args) {
    PyArrayObject* py_dd_q0;
    PyArrayObject* py_G_list;
    PyArrayObject* py_born;
    PyArrayObject* py_dielectric;
    PyArrayObject* py_positions;
    double lambda;
    double tolerance;
    long use_openmp;

    double(*dd_q0)[2];
    double(*G_list)[3];
    double(*born)[3][3];
    double(*dielectric)[3];
    double(*pos)[3];
    long num_patom, num_G;

    if (!PyArg_ParseTuple(args, "OOOOOddl", &py_dd_q0, &py_G_list, &py_born,
                          &py_dielectric, &py_positions, &lambda, &tolerance,
                          &use_openmp))
        return NULL;

    dd_q0 = (double(*)[2])PyArray_DATA(py_dd_q0);
    G_list = (double(*)[3])PyArray_DATA(py_G_list);
    born = (double(*)[3][3])PyArray_DATA(py_born);
    dielectric = (double(*)[3])PyArray_DATA(py_dielectric);
    pos = (double(*)[3])PyArray_DATA(py_positions);
    num_G = PyArray_DIMS(py_G_list)[0];
    num_patom = PyArray_DIMS(py_positions)[0];

    phpy_get_recip_dipole_dipole_q0(dd_q0,  /* [natom, 3, 3, (real, imag)] */
                                    G_list, /* [num_kvec, 3] */
                                    num_G, num_patom, born, dielectric,
                                    pos,    /* [natom, 3] */
                                    lambda, /* 4 * Lambda^2 */
                                    tolerance, use_openmp);

    Py_RETURN_NONE;
}

static PyObject* py_get_derivative_dynmat(PyObject* self, PyObject* args) {
    PyArrayObject* py_derivative_dynmat;
    PyArrayObject* py_force_constants;
    PyArrayObject* py_svecs;
    PyArrayObject* py_lattice;
    PyArrayObject* py_q_vector;
    PyArrayObject* py_multi;
    PyArrayObject* py_masses;
    PyArrayObject* py_s2p_map;
    PyArrayObject* py_p2s_map;
    PyArrayObject* py_born;
    PyArrayObject* py_dielectric;
    PyArrayObject* py_q_direction;
    double nac_factor;
    long use_openmp;

    double(*ddm)[2];
    double* fc;
    double* q_vector;
    double* lat;
    double(*svecs)[3];
    double* masses;
    long(*multi)[2];
    long* s2p_map;
    long* p2s_map;
    long num_patom;
    long num_satom;

    double* born;
    double* epsilon;
    double* q_dir;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOdOOOl", &py_derivative_dynmat,
                          &py_force_constants, &py_q_vector,
                          &py_lattice, /* column vectors */
                          &py_svecs, &py_multi, &py_masses, &py_s2p_map,
                          &py_p2s_map, &nac_factor, &py_born, &py_dielectric,
                          &py_q_direction, &use_openmp)) {
        return NULL;
    }

    ddm = (double(*)[2])PyArray_DATA(py_derivative_dynmat);
    fc = (double*)PyArray_DATA(py_force_constants);
    q_vector = (double*)PyArray_DATA(py_q_vector);
    lat = (double*)PyArray_DATA(py_lattice);
    svecs = (double(*)[3])PyArray_DATA(py_svecs);
    masses = (double*)PyArray_DATA(py_masses);
    multi = (long(*)[2])PyArray_DATA(py_multi);
    s2p_map = (long*)PyArray_DATA(py_s2p_map);
    p2s_map = (long*)PyArray_DATA(py_p2s_map);
    num_patom = PyArray_DIMS(py_p2s_map)[0];
    num_satom = PyArray_DIMS(py_s2p_map)[0];

    if ((PyObject*)py_born == Py_None) {
        born = NULL;
    } else {
        born = (double*)PyArray_DATA(py_born);
    }
    if ((PyObject*)py_dielectric == Py_None) {
        epsilon = NULL;
    } else {
        epsilon = (double*)PyArray_DATA(py_dielectric);
    }
    if ((PyObject*)py_q_direction == Py_None) {
        q_dir = NULL;
    } else {
        q_dir = (double*)PyArray_DATA(py_q_direction);
    }

    phpy_get_derivative_dynmat_at_q(
        ddm, num_patom, num_satom, fc, q_vector, lat, svecs, multi, masses,
        s2p_map, p2s_map, nac_factor, born, epsilon, q_dir, use_openmp);

    Py_RETURN_NONE;
}

/* Thermal properties */
static PyObject* py_get_thermal_properties(PyObject* self, PyObject* args) {
    PyArrayObject* py_thermal_props;
    PyArrayObject* py_temperatures;
    PyArrayObject* py_frequencies;
    PyArrayObject* py_weights;

    double cutoff_frequency;
    int classical;

    double* temperatures;
    double* freqs;
    double* thermal_props;
    long* weights;
    long num_qpoints;
    long num_bands;
    long num_temp;

    if (!PyArg_ParseTuple(args, "OOOOdi", &py_thermal_props, &py_temperatures,
                          &py_frequencies, &py_weights, &cutoff_frequency, &classical)) {
        return NULL;
    }

    thermal_props = (double*)PyArray_DATA(py_thermal_props);
    temperatures = (double*)PyArray_DATA(py_temperatures);
    num_temp = (long)PyArray_DIMS(py_temperatures)[0];
    freqs = (double*)PyArray_DATA(py_frequencies);
    num_qpoints = (long)PyArray_DIMS(py_frequencies)[0];
    weights = (long*)PyArray_DATA(py_weights);
    num_bands = (long)PyArray_DIMS(py_frequencies)[1];

    phpy_get_thermal_properties(thermal_props, temperatures, freqs, weights,
                                num_temp, num_qpoints, num_bands,
                                cutoff_frequency, classical);

    Py_RETURN_NONE;
}

static PyObject* py_distribute_fc2(PyObject* self, PyObject* args) {
    PyArrayObject* py_force_constants;
    PyArrayObject* py_permutations;
    PyArrayObject* py_map_atoms;
    PyArrayObject* py_map_syms;
    PyArrayObject* py_atom_list;
    PyArrayObject* py_fc_indices_of_atom_list;
    PyArrayObject* py_rotations_cart;

    double(*r_carts)[3][3];
    double(*fc2)[3][3];
    int* permutations;
    int* map_atoms;
    int* map_syms;
    int* atom_list;
    int* fc_indices_of_atom_list;
    npy_intp num_pos, num_rot, len_atom_list;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &py_force_constants, &py_atom_list,
                          &py_fc_indices_of_atom_list, &py_rotations_cart,
                          &py_permutations, &py_map_atoms, &py_map_syms)) {
        return NULL;
    }

    fc2 = (double(*)[3][3])PyArray_DATA(py_force_constants);
    atom_list = (int*)PyArray_DATA(py_atom_list);
    len_atom_list = PyArray_DIMS(py_atom_list)[0];
    fc_indices_of_atom_list = (int*)PyArray_DATA(py_fc_indices_of_atom_list);
    permutations = (int*)PyArray_DATA(py_permutations);
    map_atoms = (int*)PyArray_DATA(py_map_atoms);
    map_syms = (int*)PyArray_DATA(py_map_syms);
    r_carts = (double(*)[3][3])PyArray_DATA(py_rotations_cart);
    num_rot = PyArray_DIMS(py_permutations)[0];
    num_pos = PyArray_DIMS(py_permutations)[1];

    if (PyArray_NDIM(py_map_atoms) != 1 ||
        PyArray_DIMS(py_map_atoms)[0] != num_pos) {
        PyErr_SetString(PyExc_ValueError, "wrong shape for map_atoms");
        return NULL;
    }

    if (PyArray_NDIM(py_map_syms) != 1 ||
        PyArray_DIMS(py_map_syms)[0] != num_pos) {
        PyErr_SetString(PyExc_ValueError, "wrong shape for map_syms");
        return NULL;
    }

    if (PyArray_DIMS(py_rotations_cart)[0] != num_rot) {
        PyErr_SetString(PyExc_ValueError,
                        "permutations and rotations are different length");
        return NULL;
    }

    phpy_distribute_fc2(fc2, atom_list, len_atom_list, fc_indices_of_atom_list,
                        r_carts, permutations, map_atoms, map_syms, num_rot,
                        num_pos);
    Py_RETURN_NONE;
}

static PyObject* py_thm_relative_grid_address(PyObject* self, PyObject* args) {
    PyArrayObject* py_relative_grid_address;
    PyArrayObject* py_reciprocal_lattice_py;

    long(*relative_grid_address)[4][3];
    double(*reciprocal_lattice)[3];

    if (!PyArg_ParseTuple(args, "OO", &py_relative_grid_address,
                          &py_reciprocal_lattice_py)) {
        return NULL;
    }

    relative_grid_address =
        (long(*)[4][3])PyArray_DATA(py_relative_grid_address);
    reciprocal_lattice = (double(*)[3])PyArray_DATA(py_reciprocal_lattice_py);

    phpy_get_relative_grid_address(relative_grid_address, reciprocal_lattice);

    Py_RETURN_NONE;
}

static PyObject* py_thm_all_relative_grid_address(PyObject* self,
                                                  PyObject* args) {
    PyArrayObject* py_relative_grid_address;

    long(*relative_grid_address)[24][4][3];

    if (!PyArg_ParseTuple(args, "O", &py_relative_grid_address)) {
        return NULL;
    }

    relative_grid_address =
        (long(*)[24][4][3])PyArray_DATA(py_relative_grid_address);

    phpy_get_all_relative_grid_address(relative_grid_address);

    Py_RETURN_NONE;
}

static PyObject* py_thm_integration_weight(PyObject* self, PyObject* args) {
    double omega;
    PyArrayObject* py_tetrahedra_omegas;
    char* function;

    double(*tetrahedra_omegas)[4];
    double iw;

    if (!PyArg_ParseTuple(args, "dOs", &omega, &py_tetrahedra_omegas,
                          &function)) {
        return NULL;
    }

    tetrahedra_omegas = (double(*)[4])PyArray_DATA(py_tetrahedra_omegas);

    iw = phpy_get_integration_weight(omega, tetrahedra_omegas, function[0]);

    return PyFloat_FromDouble(iw);
}

static PyObject* py_thm_integration_weight_at_omegas(PyObject* self,
                                                     PyObject* args) {
    PyArrayObject* py_integration_weights;
    PyArrayObject* py_omegas;
    PyArrayObject* py_tetrahedra_omegas;
    char* function;

    double* omegas;
    double* iw;
    long num_omegas;
    double(*tetrahedra_omegas)[4];

    long i;

    if (!PyArg_ParseTuple(args, "OOOs", &py_integration_weights, &py_omegas,
                          &py_tetrahedra_omegas, &function)) {
        return NULL;
    }

    omegas = (double*)PyArray_DATA(py_omegas);
    iw = (double*)PyArray_DATA(py_integration_weights);
    num_omegas = (long)PyArray_DIMS(py_omegas)[0];
    tetrahedra_omegas = (double(*)[4])PyArray_DATA(py_tetrahedra_omegas);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < num_omegas; i++) {
        iw[i] = phpy_get_integration_weight(omegas[i], tetrahedra_omegas,
                                            function[0]);
    }

    Py_RETURN_NONE;
}

static PyObject* py_get_tetrahedra_frequenies(PyObject* self, PyObject* args) {
    PyArrayObject* py_freq_tetras;
    PyArrayObject* py_grid_points;
    PyArrayObject* py_mesh;
    PyArrayObject* py_grid_address;
    PyArrayObject* py_gp_ir_index;
    PyArrayObject* py_relative_grid_address;
    PyArrayObject* py_frequencies;

    double* freq_tetras;
    long* grid_points;
    long* mesh;
    long(*grid_address)[3];
    long* gp_ir_index;
    long(*relative_grid_address)[3];
    double* frequencies;

    long num_gp_in, num_band;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &py_freq_tetras, &py_grid_points,
                          &py_mesh, &py_grid_address, &py_gp_ir_index,
                          &py_relative_grid_address, &py_frequencies)) {
        return NULL;
    }

    freq_tetras = (double*)PyArray_DATA(py_freq_tetras);
    grid_points = (long*)PyArray_DATA(py_grid_points);
    num_gp_in = PyArray_DIMS(py_grid_points)[0];
    mesh = (long*)PyArray_DATA(py_mesh);
    grid_address = (long(*)[3])PyArray_DATA(py_grid_address);
    gp_ir_index = (long*)PyArray_DATA(py_gp_ir_index);
    relative_grid_address = (long(*)[3])PyArray_DATA(py_relative_grid_address);
    frequencies = (double*)PyArray_DATA(py_frequencies);
    num_band = PyArray_DIMS(py_frequencies)[1];

    phpy_get_tetrahedra_frequenies(freq_tetras, mesh, grid_points, grid_address,
                                   relative_grid_address, gp_ir_index,
                                   frequencies, num_band, num_gp_in);

    Py_RETURN_NONE;
}

static PyObject* py_tetrahedron_method_dos(PyObject* self, PyObject* args) {
    PyArrayObject* py_dos;
    PyArrayObject* py_mesh;
    PyArrayObject* py_freq_points;
    PyArrayObject* py_frequencies;
    PyArrayObject* py_coef;
    PyArrayObject* py_grid_address;
    PyArrayObject* py_grid_mapping_table;
    PyArrayObject* py_relative_grid_address;

    double* dos;
    long* mesh;
    double* freq_points;
    double* frequencies;
    double* coef;
    long(*grid_address)[3];
    long num_gp, num_ir_gp, num_band, num_freq_points, num_coef;
    long* grid_mapping_table;
    long(*relative_grid_address)[4][3];

    if (!PyArg_ParseTuple(args, "OOOOOOOO", &py_dos, &py_mesh, &py_freq_points,
                          &py_frequencies, &py_coef, &py_grid_address,
                          &py_grid_mapping_table, &py_relative_grid_address)) {
        return NULL;
    }

    /* dos[num_ir_gp][num_band][num_freq_points][num_coef] */
    dos = (double*)PyArray_DATA(py_dos);
    mesh = (long*)PyArray_DATA(py_mesh);
    freq_points = (double*)PyArray_DATA(py_freq_points);
    num_freq_points = (long)PyArray_DIMS(py_freq_points)[0];
    frequencies = (double*)PyArray_DATA(py_frequencies);
    num_ir_gp = (long)PyArray_DIMS(py_frequencies)[0];
    num_band = (long)PyArray_DIMS(py_frequencies)[1];
    coef = (double*)PyArray_DATA(py_coef);
    num_coef = (long)PyArray_DIMS(py_coef)[1];
    grid_address = (long(*)[3])PyArray_DATA(py_grid_address);
    num_gp = (long)PyArray_DIMS(py_grid_address)[0];
    grid_mapping_table = (long*)PyArray_DATA(py_grid_mapping_table);
    relative_grid_address =
        (long(*)[4][3])PyArray_DATA(py_relative_grid_address);

    phpy_tetrahedron_method_dos(dos, mesh, grid_address, relative_grid_address,
                                grid_mapping_table, freq_points, frequencies,
                                coef, num_freq_points, num_ir_gp, num_band,
                                num_coef, num_gp);

    Py_RETURN_NONE;
}

static PyObject* py_use_openmp(PyObject* self, PyObject* args) {
    if (phpy_use_openmp()) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject* py_get_omp_max_threads(PyObject* self, PyObject* args) {
    return PyLong_FromLong(phpy_get_max_threads());
}
