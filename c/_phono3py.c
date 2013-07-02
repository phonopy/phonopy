#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "dynmat.h"
#include "lapack_wrapper.h"
#include "interaction_strength.h"
#include "gamma.h"
#include "alloc_array.h"
#include "interaction.h"
#include "phonoc_array.h"

static PyObject * py_get_interaction_strength(PyObject *self, PyObject *args);
static PyObject * py_get_triplet_interaction_strength(PyObject *self,
						      PyObject *args);
static PyObject * py_get_sum_in_primitive(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_reciprocal(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_realspace(PyObject *self, PyObject *args);
static PyObject * py_get_gamma(PyObject *self, PyObject *args);
static PyObject * py_get_decay_channel(PyObject *self, PyObject *args);
static PyObject * py_get_jointDOS(PyObject *self, PyObject *args);


static PyObject * py_get_interaction(PyObject *self, PyObject *args);
static PyObject * py_set_phonon_triplets(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);

static int distribute_fc3(double *fc3,
			  const int third_atom,
			  const int third_atom_rot,
			  const double *positions,
			  const int num_atom,
			  const int * rot,
			  const double *rot_cartesian,
			  const double *trans,
			  const double symprec);
static int get_atom_by_symmetry(const int atom_number,
				const int num_atom,
				const double *positions,
				const int *rot,
				const double *trans,
				const double symprec);
static void tensor3_roation(double *fc3,
			    const int third_atom,
			    const int third_atom_rot,
			    const int atom_i,
			    const int atom_j,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int num_atom,
			    const double *rot_cartesian);
static double tensor3_rotation_elem(double tensor[3][3][3],
				    const double *r,
				    const int l,
				    const int m,
				    const int n);
static int nint(const double a);

static PyMethodDef functions[] = {
  {"interaction_strength", py_get_interaction_strength, METH_VARARGS, "Interaction strength of triplets"},
  {"triplet_interaction_strength", py_get_triplet_interaction_strength, METH_VARARGS, "One triplet interaction strength"},
  {"sum_in_primitive", py_get_sum_in_primitive, METH_VARARGS, "Summation in primitive cell"},
  {"fc3_reciprocal", py_get_fc3_reciprocal, METH_VARARGS, "Transformation of fc3 from real to reciprocal space"},
  {"fc3_realspace", py_get_fc3_realspace, METH_VARARGS, "Transformation of fc3 from reciprocal to real space"},
  {"gamma", py_get_gamma, METH_VARARGS, "Calculate damping function with gaussian smearing"},
  {"joint_dos", py_get_jointDOS, METH_VARARGS, "Calculate joint density of states"},
  {"decay_channel", py_get_decay_channel, METH_VARARGS, "Calculate decay of phonons"},
  {"interaction", py_get_interaction, METH_VARARGS, "Interaction of triplets"},
  {"phonon_triplets", py_set_phonon_triplets, METH_VARARGS, "Set phonon triplets"},
  {"distribute_fc3", py_distribute_fc3, METH_VARARGS, "Distribute least fc3 to full fc3"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_phono3py(void)
{
  Py_InitModule3("_phono3py", functions, "C-extension for phono3py\n\n...\n");
  return;
}

static PyObject * py_set_phonon_triplets(PyObject *self, PyObject *args)
{
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* phonon_done_py;
  PyArrayObject* gridpoint_triplets;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors_fc2;
  PyArrayObject* multiplicity_fc2;
  PyArrayObject* fc2_py;
  PyArrayObject* atomic_masses_fc2;
  PyArrayObject* p2s_map_fc2;
  PyArrayObject* s2p_map_fc2;
  PyArrayObject* reciprocal_lattice;
  PyArrayObject* band_indicies_py;
  PyArrayObject* born_effective_charge;
  PyArrayObject* q_direction;
  PyArrayObject* dielectric_constant;
  double nac_factor, unit_conversion_factor;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOdOOOOdc",
			&frequencies,
			&eigenvectors,
			&phonon_done_py,
			&gridpoint_triplets,
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
  Iarray* triplets = convert_to_iarray(gridpoint_triplets);
  Iarray* grid_address = convert_to_iarray(grid_address_py);
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

  set_phonon_triplets(freqs,
		      eigvecs,
		      phonon_done,
		      triplets,
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
  free(triplets);
  free(grid_address);
  free(fc2);
  free(svecs_fc2);
  free(multi_fc2);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_interaction(PyObject *self, PyObject *args)
{
  PyArrayObject* amplitude;
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* phonon_done_py;
  PyArrayObject* gridpoint_triplets;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors_fc2;
  PyArrayObject* multiplicity_fc2;
  PyArrayObject* shortest_vectors_fc3;
  PyArrayObject* multiplicity_fc3;
  PyArrayObject* fc2_py;
  PyArrayObject* fc3_py;
  PyArrayObject* atomic_masses_fc2;
  PyArrayObject* atomic_masses_fc3;
  PyArrayObject* p2s_map_fc2;
  PyArrayObject* s2p_map_fc2;
  PyArrayObject* p2s_map_fc3;
  PyArrayObject* s2p_map_fc3;
  PyArrayObject* reciprocal_lattice;
  PyArrayObject* band_indicies_py;
  PyArrayObject* born_effective_charge;
  PyArrayObject* q_direction;
  PyArrayObject* dielectric_constant;
  double nac_factor, freq_unit_conversion_factor;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOOOOOOOOddc",
			&amplitude,
			&frequencies,
			&eigenvectors,
			&phonon_done_py,
			&gridpoint_triplets,
			&grid_address_py,
			&mesh_py,
			&fc2_py,
			&fc3_py,
			&shortest_vectors_fc2,
			&multiplicity_fc2,
			&shortest_vectors_fc3,
			&multiplicity_fc3,
			&atomic_masses_fc2,
			&atomic_masses_fc3,
			&p2s_map_fc2,
			&s2p_map_fc2,
			&p2s_map_fc3,
			&s2p_map_fc3,
			&band_indicies_py,
			&born_effective_charge,
			&dielectric_constant,
			&reciprocal_lattice,
			&q_direction,
			&nac_factor,
			&freq_unit_conversion_factor,
			&uplo)) {
    return NULL;
  }

  double* born;
  double* dielectric;
  double *q_dir;
  Darray* amps = convert_to_darray(amplitude);
  Darray* freqs = convert_to_darray(frequencies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  Carray* eigvecs = convert_to_carray(eigenvectors);
  char* phonon_done = (char*)phonon_done_py->data;
  Iarray* triplets = convert_to_iarray(gridpoint_triplets);
  Iarray* grid_address = convert_to_iarray(grid_address_py);
  const int* mesh = (int*)mesh_py->data;
  Darray* fc2 = convert_to_darray(fc2_py);
  Darray* fc3 = convert_to_darray(fc3_py);
  Darray* svecs_fc2 = convert_to_darray(shortest_vectors_fc2);
  Iarray* multi_fc2 = convert_to_iarray(multiplicity_fc2);
  Darray* svecs_fc3 = convert_to_darray(shortest_vectors_fc3);
  Iarray* multi_fc3 = convert_to_iarray(multiplicity_fc3);
  const double* masses_fc2 = (double*)atomic_masses_fc2->data;
  const double* masses_fc3 = (double*)atomic_masses_fc3->data;
  const int* p2s_fc2 = (int*)p2s_map_fc2->data;
  const int* s2p_fc2 = (int*)s2p_map_fc2->data;
  const int* p2s_fc3 = (int*)p2s_map_fc3->data;
  const int* s2p_fc3 = (int*)s2p_map_fc3->data;
  Iarray* band_indicies = convert_to_iarray(band_indicies_py);
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

  get_interaction(amps,
		  freqs,
		  eigvecs,
		  phonon_done,
		  triplets,
		  grid_address,
		  mesh,
		  fc2,
		  fc3,
		  svecs_fc2,
		  multi_fc2,
		  svecs_fc3,
		  multi_fc3,
		  masses_fc2,
		  masses_fc3,
		  p2s_fc2,
		  s2p_fc2,
		  p2s_fc3,
		  s2p_fc3,
		  band_indicies,
		  born,
		  dielectric,
		  rec_lat,
		  q_dir,
		  nac_factor,
		  freq_unit_conversion_factor,
		  uplo);

  free(amps);
  free(freqs);
  free(eigvecs);
  free(triplets);
  free(grid_address);
  free(fc2);
  free(fc3);
  free(svecs_fc2);
  free(multi_fc2);
  free(svecs_fc3);
  free(multi_fc3);
  free(band_indicies);
  
  Py_RETURN_NONE;
}


static PyObject * py_get_interaction_strength(PyObject *self, PyObject *args)
{
  PyArrayObject* amplitude;
  PyArrayObject* frequencies;
  PyArrayObject* qvec0;
  PyArrayObject* qvec1s;
  PyArrayObject* qvec2s;
  PyArrayObject* shortest_vectors_fc2;
  PyArrayObject* multiplicity_fc2;
  PyArrayObject* shortest_vectors_fc3;
  PyArrayObject* multiplicity_fc3;
  PyArrayObject* p2s_map_fc2;
  PyArrayObject* s2p_map_fc2;
  PyArrayObject* p2s_map_fc3;
  PyArrayObject* s2p_map_fc3;
  PyArrayObject* force_constants_second;
  PyArrayObject* force_constants_third;
  PyArrayObject* atomic_masses_fc2;
  PyArrayObject* atomic_masses_fc3;
  PyArrayObject* reciprocal_lattice;
  PyArrayObject* band_indicies;
  PyArrayObject* born_effective_charge;
  PyArrayObject* q_direction;
  PyArrayObject* dielectric_constant;
  double cutoff_frequency, nac_factor, freq_unit_conversion_factor;
  int num_grid_points, r2q_TI_index, is_symmetrize_fc3_q;
  char uplo;

  if (!PyArg_ParseTuple(args, "OOiOOOOOOOOOOOOOOOOOOOOdddiic",
			&amplitude,
			&frequencies,
			&num_grid_points,
			&qvec0,
			&qvec1s,
			&qvec2s,
			&shortest_vectors_fc2,
			&multiplicity_fc2,
			&shortest_vectors_fc3,
			&multiplicity_fc3,
			&p2s_map_fc2,
			&s2p_map_fc2,
			&p2s_map_fc3,
			&s2p_map_fc3,
			&force_constants_second,
			&force_constants_third,
			&atomic_masses_fc2,
			&atomic_masses_fc3,
			&band_indicies,
			&born_effective_charge,
			&dielectric_constant,
			&reciprocal_lattice,
			&q_direction,
			&nac_factor,
			&freq_unit_conversion_factor,
			&cutoff_frequency,
			&is_symmetrize_fc3_q,
			&r2q_TI_index,
			&uplo)) {
    return NULL;
  }

  int i, j;
  int svecs_dims[4];
  Array1D * s2p_fc2, * p2s_fc2, * s2p_fc3, * p2s_fc3, *bands;
  Array2D * multi_fc2, * multi_fc3;
  ShortestVecs * svecs_fc2, * svecs_fc3;

  double* amps = (double*)amplitude->data;
  double* freqs = (double*)frequencies->data;
  const double *q0 = (double*)qvec0->data;
  const double *q1s = (double*)qvec1s->data;
  const double *q2s = (double*)qvec2s->data;
  const double* masses_fc2 = (double*)atomic_masses_fc2->data;
  const double* masses_fc3 = (double*)atomic_masses_fc3->data;
  const int* multiplicity_fc2_int = (int*)multiplicity_fc2->data;
  const int* multiplicity_fc3_int = (int*)multiplicity_fc3->data;
  const int* p2s_map_fc2_int = (int*)p2s_map_fc2->data;
  const int* s2p_map_fc2_int = (int*)s2p_map_fc2->data;
  const int* p2s_map_fc3_int = (int*)p2s_map_fc3->data;
  const int* s2p_map_fc3_int = (int*)s2p_map_fc3->data;
  const double* fc2 = (double*)force_constants_second->data;
  const double* fc3 = (double*)force_constants_third->data;
  const int* bands_int = (int*)band_indicies->data;
  const double* rec_lat = (double*)reciprocal_lattice->data;
  double* born;
  if ((PyObject*)born_effective_charge == Py_None) {
    born = NULL;
  } else {
    born = (double*)born_effective_charge->data;
  }
  double* dielectric;
  if ((PyObject*)dielectric_constant == Py_None) {
    dielectric = NULL;
  } else {
    dielectric = (double*)dielectric_constant->data;
  }
  double *q_dir;
  if ((PyObject*)q_direction == Py_None) {
    q_dir = NULL;
  } else {
    q_dir = (double*)q_direction->data;
  }

  const int num_satom_fc2 = (int)multiplicity_fc2->dimensions[0];
  const int num_satom_fc3 = (int)multiplicity_fc3->dimensions[0];
  const int num_patom = (int)multiplicity_fc3->dimensions[1];
  const int num_triplets = (int)amplitude->dimensions[0];

  bands = alloc_Array1D(band_indicies->dimensions[0]);
  for (i = 0; i < bands->d1; i++) {
    bands->data[i] = (int)bands_int[i];
  }
  
  for (i = 0; i < 4; i++) {
    svecs_dims[i] = (int)shortest_vectors_fc2->dimensions[i];
  }
  svecs_fc2 = get_shortest_vecs((double*)shortest_vectors_fc2->data,
				svecs_dims);
  for (i = 0; i < 4; i++) {
    svecs_dims[i] = (int)shortest_vectors_fc3->dimensions[i];
  }
  svecs_fc3 = get_shortest_vecs((double*)shortest_vectors_fc3->data,
				svecs_dims);

  multi_fc2 = alloc_Array2D(num_satom_fc2, num_patom);
  for (i = 0; i < num_satom_fc2; i++) {
    for (j = 0; j < num_patom; j++) {
      multi_fc2->data[i][j] = multiplicity_fc2_int[i * num_patom + j];
    }
  }
  multi_fc3 = alloc_Array2D(num_satom_fc3, num_patom);
  for (i = 0; i < num_satom_fc3; i++) {
    for (j = 0; j < num_patom; j++) {
      multi_fc3->data[i][j] = multiplicity_fc3_int[i * num_patom + j];
    }
  }

  s2p_fc2 = alloc_Array1D(num_satom_fc2);
  for (i = 0; i < num_satom_fc2; i++) {
    s2p_fc2->data[i] = s2p_map_fc2_int[i];
  }
  p2s_fc2 = alloc_Array1D(num_patom);
  for (i = 0; i < num_patom; i++) {
    p2s_fc2->data[i] = p2s_map_fc2_int[i];
  }
  s2p_fc3 = alloc_Array1D(num_satom_fc3);
  for (i = 0; i < num_satom_fc3; i++) {
    s2p_fc3->data[i] = s2p_map_fc3_int[i];
  }
  p2s_fc3 = alloc_Array1D(num_patom);
  for (i = 0; i < num_patom; i++) {
    p2s_fc3->data[i] = p2s_map_fc3_int[i];
  }

  get_interaction_strength(amps,
			   freqs,
			   num_triplets,
			   num_grid_points,
			   q0,
			   q1s,
			   q2s,
			   fc2,
			   fc3,
			   masses_fc2,
			   masses_fc3,
			   p2s_fc2,
			   s2p_fc2,
			   p2s_fc3,
			   s2p_fc3,
			   multi_fc2,
			   svecs_fc2,
			   multi_fc3,
			   svecs_fc3,
			   bands,
			   born,
			   dielectric,
			   rec_lat,
			   q_dir,
			   nac_factor,
			   freq_unit_conversion_factor,
			   cutoff_frequency,
			   is_symmetrize_fc3_q,
			   r2q_TI_index,
			   uplo);

  free_Array1D(p2s_fc2);
  free_Array1D(s2p_fc2);
  free_Array1D(p2s_fc3);
  free_Array1D(s2p_fc3);
  free_Array2D(multi_fc2);
  free_Array2D(multi_fc3);
  free_Array1D(bands);
  free_shortest_vecs(svecs_fc2);
  free_shortest_vecs(svecs_fc3);

  Py_RETURN_NONE;
}

static PyObject * py_get_triplet_interaction_strength(PyObject *self,
						      PyObject *args)
{
  PyArrayObject* amplitude;
  PyArrayObject* omegas;
  PyArrayObject* eigenvectors;
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* q_triplet;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;
  PyArrayObject* force_constants_third;
  PyArrayObject* atomic_masses;
  PyArrayObject* set_of_band_indices;
  double cutoff_frequency;
  int r2q_TI_index, is_symmetrize_fc3_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOdii",
			&amplitude,
			&omegas,
			&eigenvectors,
			&shortest_vectors,
			&multiplicity,
			&q_triplet,
			&p2s_map,
			&s2p_map,
			&force_constants_third,
			&atomic_masses,
			&set_of_band_indices,
			&cutoff_frequency,
			&is_symmetrize_fc3_q,
			&r2q_TI_index)) {
    return NULL;
  }

  int i, j;
  int * svecs_dimensions;
  Array1D * s2p, * p2s, * band_indices;
  Array2D * multi;
  ShortestVecs * svecs;
  lapack_complex_double * lpk_eigvecs;

  double* amps = (double*)amplitude->data;
  const double* freqs = (double*)omegas->data;
  const npy_cdouble* eigvecs = (npy_cdouble*)eigenvectors->data;
  const double* masses = (double*)atomic_masses->data;
  const int* multiplicity_int = (int*)multiplicity->data;
  const int* p2s_map_int = (int*)p2s_map->data;
  const int* s2p_map_int = (int*)s2p_map->data;
  const int* band_indices_int = (int*)set_of_band_indices->data;
  const double* fc3 = (double*)force_constants_third->data;
  const double* q_vecs = (double*)q_triplet->data;

  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];
  const int num_band0 = (int)set_of_band_indices->dimensions[0];

  svecs_dimensions = (int*)malloc(sizeof(int) * 4);
  for (i = 0; i < 4; i++) {
    svecs_dimensions[i] = (int)shortest_vectors->dimensions[i];
  }
  svecs = get_shortest_vecs((double*)shortest_vectors->data, svecs_dimensions);
  free(svecs_dimensions);
  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multiplicity_int[i * num_patom + j];
    }
  }

  s2p = alloc_Array1D(num_satom);
  for (i = 0; i < num_satom; i++) {
    s2p->data[i] = s2p_map_int[i];
  }
  p2s = alloc_Array1D(num_patom);
  for (i = 0; i < num_patom; i++) {
    p2s->data[i] = p2s_map_int[i];
  }
  band_indices = alloc_Array1D(num_band0);
  for (i = 0; i < num_band0; i++) {
    band_indices->data[i] = band_indices_int[i];
  }

  lpk_eigvecs = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_patom * num_patom * 27);
  for (i = 0; i < num_patom * num_patom * 27; i++) {
    lpk_eigvecs[i] = lapack_make_complex_double
      (eigvecs[i].real, eigvecs[i].imag);
  }

  get_triplet_interaction_strength(amps,
				   fc3,
				   q_vecs,
				   lpk_eigvecs,
				   freqs,
				   masses,
				   p2s,
				   s2p,
				   multi,
				   svecs,
				   band_indices,
				   cutoff_frequency,
				   is_symmetrize_fc3_q,
				   r2q_TI_index);

  free(lpk_eigvecs);
  free_Array1D(p2s);
  free_Array1D(s2p);
  free_Array2D(multi);
  free_shortest_vecs(svecs);

  Py_RETURN_NONE;
}

static PyObject * py_get_sum_in_primitive(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants_third;
  PyArrayObject* eigvec1;
  PyArrayObject* eigvec2;
  PyArrayObject* eigvec3;
  PyArrayObject* masses;

  if (!PyArg_ParseTuple(args, "OOOOO",
			&force_constants_third,
			&eigvec1,
			&eigvec2,
			&eigvec3,
			&masses))
    return NULL;

  int i;
  double sum;
  lapack_complex_double * lpk_fc3, * lpk_e1, * lpk_e2, * lpk_e3;
  
  const npy_cdouble* fc3 = (npy_cdouble*)force_constants_third->data;
  const npy_cdouble* e1 = (npy_cdouble*)eigvec1->data;
  const npy_cdouble* e2 = (npy_cdouble*)eigvec2->data;
  const npy_cdouble* e3 = (npy_cdouble*)eigvec3->data;
  const double* m = (double*)masses->data;
  const int num_atom = (int) masses->dimensions[0];

  lpk_fc3 = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_atom * num_atom * num_atom * 27);
  for (i = 0; i < num_atom * num_atom * num_atom * 27; i++) {
    lpk_fc3[i] = lapack_make_complex_double(fc3[i].real, fc3[i].imag);
  }
  lpk_e1 = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_atom * 3);  
  for (i = 0; i < num_atom * 3; i++) {
    lpk_e1[i] = lapack_make_complex_double(e1[i].real, e1[i].imag);
  }
  lpk_e2 = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_atom * 3);  
  for (i = 0; i < num_atom * 3; i++) {
    lpk_e2[i] = lapack_make_complex_double(e2[i].real, e2[i].imag);
  }
  lpk_e3 = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * num_atom * 3);
  for (i = 0; i < num_atom * 3; i++) {
    lpk_e3[i] = lapack_make_complex_double(e3[i].real, e3[i].imag);
  }

  sum = get_sum_in_primivie(lpk_fc3,
			    lpk_e1,
			    lpk_e2,
			    lpk_e3,
			    num_atom,
			    m);

  free(lpk_fc3);
  free(lpk_e1);
  free(lpk_e2);
  free(lpk_e3);
  
  return PyFloat_FromDouble(sum);
}

static PyObject * py_get_fc3_reciprocal(PyObject *self, PyObject *args)
{
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* q_triplet;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;
  PyArrayObject* force_constants_third;
  PyArrayObject* force_constants_third_reciprocal;
  const int r2q_TI_index;
  

  if (!PyArg_ParseTuple(args, "OOOOOOOi",
			&force_constants_third_reciprocal,
			&shortest_vectors,
			&multiplicity,
			&q_triplet,
			&p2s_map,
			&s2p_map,
			&force_constants_third,
			&r2q_TI_index)) {
    return NULL;
  }

  int i, j;
  int * svecs_dimensions;
  Array1D * s2p, * p2s;
  Array2D * multi;
  DArray2D * q;
  ShortestVecs * svecs;
  lapack_complex_double * lpk_fc3_q;
  
  npy_cdouble* fc3_q = (npy_cdouble*)force_constants_third_reciprocal->data;
  
  const int* multiplicity_int = (int*)multiplicity->data;
  const int* p2s_map_int = (int*)p2s_map->data;
  const int* s2p_map_int = (int*)s2p_map->data;
  const double* fc3 = (double*)force_constants_third->data;

  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];

  svecs_dimensions = (int*)malloc(sizeof(int) * 4);
  for (i = 0; i < 4; i++) {
    svecs_dimensions[i] = (int)shortest_vectors->dimensions[i];
  }
  svecs = get_shortest_vecs((double*)shortest_vectors->data, svecs_dimensions);
  free(svecs_dimensions);
  
  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multiplicity_int[i * num_patom + j];
    }
  }

  s2p = alloc_Array1D(num_satom);
  p2s = alloc_Array1D(num_patom);
  for (i = 0; i < num_satom; i++) {
    s2p->data[i] = s2p_map_int[i];
  }
  for (i = 0; i < num_patom; i++) {
    p2s->data[i] = p2s_map_int[i];
  }

  q = alloc_DArray2D(3, 3);
  for (i = 0; i < 3; i ++) {
    for (j = 0; j < 3; j ++) {
      q->data[i][j] = ((double*)q_triplet->data)[i * 3 + j];
    }
  }

  lpk_fc3_q = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) *
	   num_patom * num_patom * num_patom * 27);

  get_fc3_reciprocal(lpk_fc3_q,
		     svecs,
		     multi,
		     q,
		     s2p,
		     p2s,
		     fc3,
		     r2q_TI_index);
  for (i = 0; i < num_patom * num_patom * num_patom * 27; i++) {
    fc3_q[i].real = lapack_complex_double_real(lpk_fc3_q[i]);
    fc3_q[i].imag = lapack_complex_double_imag(lpk_fc3_q[i]);
  }

  free(lpk_fc3_q);
  free_DArray2D(q);
  free_Array1D(p2s);
  free_Array1D(s2p);
  free_Array2D(multi);
  free_shortest_vecs(svecs);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_fc3_realspace(PyObject *self, PyObject *args)
{
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* qpoints_triplet;
  PyArrayObject* s2p_map;
  PyArrayObject* fc3_realspace;
  PyArrayObject* fc3_reciprocal;

  if (!PyArg_ParseTuple(args, "OOOOOO",
			&fc3_realspace,
			&shortest_vectors,
			&multiplicity,
			&qpoints_triplet,
			&s2p_map,
			&fc3_reciprocal)) {
    return NULL;
  }

  int i, j;
  int * svecs_dimensions;
  Array2D * multi;
  ShortestVecs * svecs;
  lapack_complex_double *lpk_fc3_real, *lpk_fc3_rec;

  npy_cdouble* fc3_real = (npy_cdouble*)fc3_realspace->data;
  const npy_cdouble* fc3_rec = (npy_cdouble*)fc3_reciprocal->data;
  const int* multi_int = (int*)multiplicity->data;
  const double* q_triplet = (double*)qpoints_triplet->data;
  const int* s2p_map_int = (int*)s2p_map->data;
 
  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];

  svecs_dimensions = (int*)malloc(sizeof(int) * 4);
  for (i = 0; i < 4; i++) {
    svecs_dimensions[i] = (int)shortest_vectors->dimensions[i];
  }
  svecs = get_shortest_vecs((double*)shortest_vectors->data, svecs_dimensions);
  free(svecs_dimensions);

  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multi_int[i * num_patom + j];
    }
  }

  lpk_fc3_real = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) *
	   num_satom * num_satom * num_satom * 27);
  lpk_fc3_rec = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) *
	   num_patom * num_patom * num_patom * 27);
  for (i = 0; i < num_patom * num_patom * num_patom * 27; i++) {
    lpk_fc3_rec[i] = lapack_make_complex_double(fc3_rec[i].real,
						fc3_rec[i].imag);
  }
  get_fc3_realspace(lpk_fc3_real,
		    svecs,
		    multi,
		    q_triplet,
		    s2p_map_int,
		    lpk_fc3_rec);
  for (i = 0; i < num_satom * num_satom * num_satom * 27; i++) {
    fc3_real[i].real = lapack_complex_double_real(lpk_fc3_real[i]);
    fc3_real[i].imag = lapack_complex_double_imag(lpk_fc3_real[i]);
  }
  free(lpk_fc3_real);
  free(lpk_fc3_rec);
  free_Array2D(multi);
  free_shortest_vecs(svecs);

  Py_RETURN_NONE;
}

static PyObject * py_get_gamma(PyObject *self, PyObject *args)
{
  PyArrayObject* gammas;
  PyArrayObject* omegas;
  PyArrayObject* amplitudes;
  PyArrayObject* weights;
  PyArrayObject* frequencies;
  double sigma, t, freq_factor, cutoff_frequency;
  int band_index, option;

  if (!PyArg_ParseTuple(args, "OOOOOiddddi",
			&gammas,
			&omegas,
			&amplitudes,
			&weights,
			&frequencies,
			&band_index,
			&sigma,
			&freq_factor,
			&t,
			&cutoff_frequency,
			&option)) {
    return NULL;
  }
  
  /* options */
  /* 0: Second part is a delta function multiplied by 2 */
  /* 1: Second part is sum of two delta function. */
  /* 2: No second part */
  /* 3: No first part with the option 0 */
  /* 4: No first part with the option 1 */
  /* 5: Only the first part of second part of option 1 */
  /* 6: Only the second part of second part of option 1 */


  double* dfun = (double*)gammas->data;
  const double* o = (double*)omegas->data;
  const double* amp = (double*)amplitudes->data;
  const int* w = (int*)weights->data;
  const double* f = (double*)frequencies->data;
  const int num_band0 = (int)amplitudes->dimensions[1];
  const int num_band = (int)amplitudes->dimensions[2];
  const int num_omega = (int)omegas->dimensions[0];
  const int num_triplet = (int)weights->dimensions[0];

  get_gamma(dfun,
	    num_omega,
	    num_triplet,
	    band_index,
	    num_band0,
	    num_band,
	    w,
	    o,
	    f,
	    amp,
	    sigma,
	    t,
	    cutoff_frequency,
	    freq_factor,
	    option);

  Py_RETURN_NONE;
}

static PyObject * py_get_jointDOS(PyObject *self, PyObject *args)
{
  PyArrayObject* jointdos;
  PyArrayObject* omegas;
  PyArrayObject* weights;
  PyArrayObject* frequencies;
  double sigma;

  if (!PyArg_ParseTuple(args, "OOOOd",
			&jointdos,
			&omegas,
			&weights,
			&frequencies,
			&sigma)) {
    return NULL;
  }
  
  double* jdos = (double*)jointdos->data;
  const double* o = (double*)omegas->data;
  const int* w = (int*)weights->data;
  const double* f = (double*)frequencies->data;
  const int num_band = (int)frequencies->dimensions[2];
  const int num_omega = (int)omegas->dimensions[0];
  const int num_triplet = (int)weights->dimensions[0];

  get_jointDOS(jdos,
	       num_omega,
	       num_triplet,
	       num_band,
	       o,
	       f,
	       w,
	       sigma);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_decay_channel(PyObject *self, PyObject *args)
{
  PyArrayObject* decay_values;
  PyArrayObject* amplitudes;
  PyArrayObject* omegas;
  PyArrayObject* frequencies;
  double sigma, t, freq_factor;

  if (!PyArg_ParseTuple(args, "OOOOddd",
			&decay_values,
			&amplitudes,
			&frequencies,
			&omegas,
			&freq_factor,
			&t,
			&sigma)) {
    return NULL;
  }
  

  double* decay = (double*)decay_values->data;
  const double* amp = (double*)amplitudes->data;
  const double* f = (double*)frequencies->data;
  const double* o = (double*)omegas->data;
  
  const int num_band = (int)amplitudes->dimensions[2];
  const int num_triplet = (int)amplitudes->dimensions[0];
  const int num_omega = (int)omegas->dimensions[0];

  get_decay_channels(decay,
		     num_omega,
		     num_triplet,
		     num_band,
		     o,
		     f,
		     amp,
		     sigma,
		     t,
		     freq_factor);
  
  Py_RETURN_NONE;
}

static PyObject * py_distribute_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants_third;
  int third_atom, third_atom_rot;
  PyArrayObject* positions;
  PyArrayObject* rotation;
  PyArrayObject* rotation_cart_inv;
  PyArrayObject* translation;
  double symprec;

  if (!PyArg_ParseTuple(args, "OiiOOOOd",
			&force_constants_third,
			&third_atom,
			&third_atom_rot,
			&positions,
			&rotation,
			&rotation_cart_inv,
			&translation,
			&symprec))
    return NULL;

  double* fc3 = (double*)force_constants_third->data;
  const double* pos = (double*)positions->data;
  const int* rot = (int*)rotation->data;
  const double* rot_cart_inv = (double*)rotation_cart_inv->data;
  const double* trans = (double*)translation->data;
  const int num_atom = (int)positions->dimensions[0];

  return PyInt_FromLong((long) distribute_fc3(fc3,
					      third_atom,
					      third_atom_rot,
					      pos,
					      num_atom,
					      rot,
					      rot_cart_inv,
					      trans,
					      symprec));
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

  const int dimension = (int)dynamical_matrix->dimensions[0];
  npy_cdouble *dynmat = (npy_cdouble*)dynamical_matrix->data;
  double *eigvals = (double*)eigenvalues->data;

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
  
  return PyInt_FromLong((long) info);
}


static int distribute_fc3(double *fc3,
			  const int third_atom,
			  const int third_atom_rot,
			  const double *positions,
			  const int num_atom,
			  const int *rot,
			  const double *rot_cart_inv,
			  const double *trans,
			  const double symprec)
{
  int i, j, atom_rot_i, atom_rot_j;
  
  for (i = 0; i < num_atom; i++) {
    atom_rot_i =
      get_atom_by_symmetry(i, num_atom, positions, rot, trans, symprec);

    if (atom_rot_i < 0) {
      fprintf(stderr, "phono3c: Unexpected behavior in distribute_fc3.\n");
      fprintf(stderr, "phono3c: atom_i %d\n", i);
      return 0;
    }

    for (j = 0; j < num_atom; j++) {
      atom_rot_j =
	get_atom_by_symmetry(j, num_atom, positions, rot, trans, symprec);

      if (atom_rot_j < 0) {
	fprintf(stderr, "phono3c: Unexpected behavior in distribute_fc3.\n");
	return 0;
      }

      tensor3_roation(fc3,
		      third_atom,
		      third_atom_rot,
		      i,
		      j,
		      atom_rot_i,
		      atom_rot_j,
		      num_atom,
		      rot_cart_inv);
    }
  }
  return 1;
}

static int get_atom_by_symmetry(const int atom_number,
				const int num_atom,
				const double *positions,
				const int *rot,
				const double *trans,
				const double symprec)
{
  int i, j, found;
  double rot_pos[3], diff[3];
  
  for (i = 0; i < 3; i++) {
    rot_pos[i] = trans[i];
    for (j = 0; j < 3; j++) {
      rot_pos[i] += rot[i * 3 + j] * positions[atom_number * 3 + j];
    }
  }

  for (i = 0; i < num_atom; i++) {
    found = 1;
    for (j = 0; j < 3; j++) {
      diff[j] = positions[i * 3 + j] - rot_pos[j];
      diff[j] -= nint(diff[j]);
      if (fabs(diff[j]) > symprec) {
	found = 0;
	break;
      }
    }
    if (found) {
      return i;
    }
  }
  /* Not found */
  return -1;
}

static void tensor3_roation(double *fc3,
			    const int third_atom,
			    const int third_atom_rot,
			    const int atom_i,
			    const int atom_j,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int num_atom,
			    const double *rot_cart_inv)
{
  int i, j, k;
  double tensor[3][3][3];

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	tensor[i][j][k] = fc3[27 * num_atom * num_atom * third_atom_rot +
			      27 * num_atom * atom_rot_i +
			      27 * atom_rot_j +
			      9 * i + 3 * j + k];
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	fc3[27 * num_atom * num_atom * third_atom +
	    27 * num_atom * atom_i +
	    27 * atom_j +
	    9 * i + 3 * j + k] = 
	  tensor3_rotation_elem(tensor, rot_cart_inv, i, j, k);
      }
    }
  }
}

static double tensor3_rotation_elem(double tensor[3][3][3],
				    const double *r,
				    const int l,
				    const int m,
				    const int n) 
{
  int i, j, k;
  double sum;

  sum = 0.0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	sum += r[l * 3 + i] * r[m * 3 + j] * r[n * 3 + k] * tensor[i][j][k];
      }
    }
  }
  return sum;
}


static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}

