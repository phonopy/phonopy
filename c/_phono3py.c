#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/interaction.h"
#include "phonon3_h/imag_self_energy.h"
#include "phonon4_h/real_to_reciprocal.h"
#include "phonon4_h/frequency_shift.h"

static PyObject * py_get_jointDOS(PyObject *self, PyObject *args);

static PyObject * py_get_interaction(PyObject *self, PyObject *args);
static PyObject * py_get_fc4_normal_for_frequency_shift(PyObject *self, PyObject *args);
static PyObject * py_get_fc4_frequency_shifts(PyObject *self, PyObject *args);
static PyObject * py_real_to_reciprocal4(PyObject *self, PyObject *args);
static PyObject * py_reciprocal_to_normal4(PyObject *self, PyObject *args);
static PyObject * py_get_imag_self_energy(PyObject *self, PyObject *args);
static PyObject * py_get_imag_self_energy_at_bands(PyObject *self,
						   PyObject *args);
static PyObject * py_set_phonon_triplets(PyObject *self, PyObject *args);
static PyObject * py_set_phonons_grid_points(PyObject *self, PyObject *args);
static PyObject * py_get_phonon(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc4(PyObject *self, PyObject *args);
static PyObject * py_rotate_delta_fc3s(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);

static int distribute_fc3(double *fc3,
			  const int third_atom,
			  const int *atom_mapping,
			  const int num_atom,
			  const double *rot_cart);
static int distribute_fc4(double *fc4,
			  const int fourth_atom,
			  const int *atom_mapping,
			  const int num_atom,
			  const double *rot_cart);
static int rotate_delta_fc3s(double *rotated_delta_fc3s,
			     const double *delta_fc3s,
			     const int *rot_map_syms,
			     const double *site_sym_cart,
			     const int num_rot,
			     const int num_delta_fc3s,
			     const int atom1,
			     const int atom2,
			     const int atom3,
			     const int num_atom);
static void tensor3_roation(double *rot_tensor,
			    const double *fc3,
			    const int atom_i,
			    const int atom_j,
			    const int atom_k,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int atom_rot_k,
			    const int num_atom,
			    const double *rot_cartesian);
static double tensor3_rotation_elem(const double tensor[27],
				    const double *r,
				    const int l,
				    const int m,
				    const int n);
static void tensor4_roation(double *rot_tensor,
			    const double *fc4,
			    const int atom_i,
			    const int atom_j,
			    const int atom_k,
			    const int atom_l,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int atom_rot_k,
			    const int atom_rot_l,
			    const int num_atom,
			    const double *rot_cartesian);
static double tensor4_rotation_elem(const double tensor[81],
				    const double *r,
				    const int m,
				    const int n,
				    const int p,
				    const int q);
static PyMethodDef functions[] = {
  {"joint_dos", py_get_jointDOS, METH_VARARGS, "Calculate joint density of states"},
  {"interaction", py_get_interaction, METH_VARARGS, "Interaction of triplets"},
  {"fc4_normal_for_frequency_shift", py_get_fc4_normal_for_frequency_shift, METH_VARARGS, "Calculate fc4 normal for frequency shift"},
  {"fc4_frequency_shifts", py_get_fc4_frequency_shifts, METH_VARARGS, "Calculate fc4 frequency shift"},
  {"imag_self_energy", py_get_imag_self_energy, METH_VARARGS, "Imaginary part of self energy"},
  {"imag_self_energy_at_bands", py_get_imag_self_energy_at_bands, METH_VARARGS, "Imaginary part of self energy at phonon frequencies of bands"},
  {"real_to_reciprocal4", py_real_to_reciprocal4, METH_VARARGS, "Transform fc4 of real space to reciprocal space"},
  {"reciprocal_to_normal4", py_reciprocal_to_normal4, METH_VARARGS, "Transform fc4 of reciprocal space to normal coordinate in special case for frequency shift"},
  {"phonon_triplets", py_set_phonon_triplets, METH_VARARGS, "Set phonon triplets"},
  {"phonons_grid_points", py_set_phonons_grid_points, METH_VARARGS, "Set phonons on grid points"},
  {"phonon", py_get_phonon, METH_VARARGS, "Get phonon"},
  {"distribute_fc3", py_distribute_fc3, METH_VARARGS, "Distribute least fc3 to full fc3"},
  {"distribute_fc4", py_distribute_fc4, METH_VARARGS, "Distribute least fc4 to full fc4"},
  {"rotate_delta_fc3s", py_rotate_delta_fc3s, METH_VARARGS, "Rotate delta fc3s"},
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
  PyArrayObject* grid_point_triplets;
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
			&grid_point_triplets,
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
  Iarray* triplets = convert_to_iarray(grid_point_triplets);
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
  free(grid_address);
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


static PyObject * py_get_interaction(PyObject *self, PyObject *args)
{
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies;
  PyArrayObject* eigenvectors;
  PyArrayObject* grid_point_triplets;
  PyArrayObject* grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* fc3_py;
  PyArrayObject* atomic_masses;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;
  PyArrayObject* band_indicies_py;
  double cutoff_frequency;
  int symmetrize_fc3_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOid",
			&fc3_normal_squared_py,
			&frequencies,
			&eigenvectors,
			&grid_point_triplets,
			&grid_address_py,
			&mesh_py,
			&fc3_py,
			&shortest_vectors,
			&multiplicity,
			&atomic_masses,
			&p2s_map,
			&s2p_map,
			&band_indicies_py,
			&symmetrize_fc3_q,
			&cutoff_frequency)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  Darray* freqs = convert_to_darray(frequencies);
  /* npy_cdouble and lapack_complex_double may not be compatible. */
  /* So eigenvectors should not be used in Python side */
  Carray* eigvecs = convert_to_carray(eigenvectors);
  Iarray* triplets = convert_to_iarray(grid_point_triplets);
  Iarray* grid_address = convert_to_iarray(grid_address_py);
  const int* mesh = (int*)mesh_py->data;
  Darray* fc3 = convert_to_darray(fc3_py);
  Darray* svecs = convert_to_darray(shortest_vectors);
  Iarray* multi = convert_to_iarray(multiplicity);
  const double* masses = (double*)atomic_masses->data;
  const int* p2s = (int*)p2s_map->data;
  const int* s2p = (int*)s2p_map->data;
  const int* band_indicies = (int*)band_indicies_py->data;

  get_interaction(fc3_normal_squared,
		  freqs,
		  eigvecs,
		  triplets,
		  grid_address,
		  mesh,
		  fc3,
		  svecs,
		  multi,
		  masses,
		  p2s,
		  s2p,
		  band_indicies,
		  symmetrize_fc3_q,
		  cutoff_frequency);

  free(fc3_normal_squared);
  free(freqs);
  free(eigvecs);
  free(triplets);
  free(grid_address);
  free(fc3);
  free(svecs);
  free(multi);
  
  Py_RETURN_NONE;
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

static PyObject * py_get_imag_self_energy(PyObject *self, PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_point_triplets_py;
  PyArrayObject* triplet_weights_py;
  double sigma, unit_conversion_factor, cutoff_frequency, temperature, fpoint;

  if (!PyArg_ParseTuple(args, "OOOOOddddd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&fpoint,
			&temperature,
			&sigma,
			&unit_conversion_factor,
			&cutoff_frequency)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  double* gamma = (double*)gamma_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* grid_point_triplets = (int*)grid_point_triplets_py->data;
  const int* triplet_weights = (int*)triplet_weights_py->data;

  get_imag_self_energy(gamma,
		       fc3_normal_squared,
		       fpoint,
		       frequencies,
		       grid_point_triplets,
		       triplet_weights,
		       sigma,
		       temperature,
		       unit_conversion_factor,
		       cutoff_frequency);

  free(fc3_normal_squared);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_imag_self_energy_at_bands(PyObject *self,
						   PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_point_triplets_py;
  PyArrayObject* triplet_weights_py;
  PyArrayObject* band_indices_py;
  double sigma, unit_conversion_factor, cutoff_frequency, temperature;

  if (!PyArg_ParseTuple(args, "OOOOOOdddd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&band_indices_py,
			&temperature,
			&sigma,
			&unit_conversion_factor,
			&cutoff_frequency)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  double* gamma = (double*)gamma_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* band_indices = (int*)band_indices_py->data;
  const int* grid_point_triplets = (int*)grid_point_triplets_py->data;
  const int* triplet_weights = (int*)triplet_weights_py->data;

  get_imag_self_energy_at_bands(gamma,
				fc3_normal_squared,
				band_indices,
				frequencies,
				grid_point_triplets,
				triplet_weights,
				sigma,
				temperature,
				unit_conversion_factor,
				cutoff_frequency);

  free(fc3_normal_squared);
  
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

static PyObject * py_distribute_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants_third;
  int third_atom;
  PyArrayObject* rotation_cart_inv;
  PyArrayObject* atom_mapping_py;

  if (!PyArg_ParseTuple(args, "OiOO",
			&force_constants_third,
			&third_atom,
			&atom_mapping_py,
			&rotation_cart_inv)) {
    return NULL;
  }

  double* fc3 = (double*)force_constants_third->data;
  const double* rot_cart_inv = (double*)rotation_cart_inv->data;
  const int* atom_mapping = (int*)atom_mapping_py->data;
  const int num_atom = (int)atom_mapping_py->dimensions[0];

  return PyInt_FromLong((long) distribute_fc3(fc3,
					      third_atom,
					      atom_mapping,
					      num_atom,
					      rot_cart_inv));
}

static PyObject * py_distribute_fc4(PyObject *self, PyObject *args)
{
  PyArrayObject* force_constants_fourth;
  int fourth_atom;
  PyArrayObject* rotation_cart_inv;
  PyArrayObject* atom_mapping_py;

  if (!PyArg_ParseTuple(args, "OiOO",
			&force_constants_fourth,
			&fourth_atom,
			&atom_mapping_py,
			&rotation_cart_inv)) {
    return NULL;
  }

  double* fc4 = (double*)force_constants_fourth->data;
  const double* rot_cart_inv = (double*)rotation_cart_inv->data;
  const int* atom_mapping = (int*)atom_mapping_py->data;
  const int num_atom = (int)atom_mapping_py->dimensions[0];

  return PyInt_FromLong((long) distribute_fc4(fc4,
					      fourth_atom,
					      atom_mapping,
					      num_atom,
					      rot_cart_inv));
}

static PyObject * py_rotate_delta_fc3s(PyObject *self, PyObject *args)
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

  return PyInt_FromLong((long) rotate_delta_fc3s(rotated_delta_fc3s,
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
			  const int *atom_mapping,
			  const int num_atom,
			  const double *rot_cart)
{
  int i, j, atom_rot_i, atom_rot_j, third_atom_rot;
  double *tensor;

  third_atom_rot = atom_mapping[third_atom];
  
  for (i = 0; i < num_atom; i++) {
    atom_rot_i = atom_mapping[i];

    for (j = 0; j < num_atom; j++) {
      atom_rot_j = atom_mapping[j];
      tensor = (fc3 +
		27 * num_atom * num_atom * third_atom +
		27 * num_atom * i +
		27 * j);
      tensor3_roation(tensor,
		      fc3,
		      third_atom,
		      i,
		      j,
		      third_atom_rot,
		      atom_rot_i,
		      atom_rot_j,
		      num_atom,
		      rot_cart);
    }
  }
  return 1;
}

static int rotate_delta_fc3s(double *rotated_delta_fc3s,
			     const double *delta_fc3s,
			     const int *rot_map_syms,
			     const double *site_sym_cart,
			     const int num_rot,
			     const int num_delta_fc3s,
			     const int atom1,
			     const int atom2,
			     const int atom3,
			     const int num_atom)
{
  int i, j;
  double *rot_tensor;
  for (i = 0; i < num_delta_fc3s; i++) {
    for (j = 0; j < num_rot; j++) {
      rot_tensor = rotated_delta_fc3s + i * num_rot * 27 + j * 27;
      tensor3_roation(rot_tensor,
		      delta_fc3s +
		      i * num_atom * num_atom * num_atom * 27,
		      atom1,
		      atom2,
		      atom3,
		      rot_map_syms[num_atom * j + atom1],
		      rot_map_syms[num_atom * j + atom2],
		      rot_map_syms[num_atom * j + atom3],
		      num_atom,
		      site_sym_cart + j * 9);
    }
  }
  return 0;
}

static void tensor3_roation(double *rot_tensor,
			    const double *fc3,
			    const int atom_i,
			    const int atom_j,
			    const int atom_k,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int atom_rot_k,
			    const int num_atom,
			    const double *rot_cartesian)
{
  int i, j, k;
  double tensor[27];

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	tensor[i * 9 + j * 3 + k] = fc3[27 * num_atom * num_atom * atom_rot_i +
					27 * num_atom * atom_rot_j +
					27 * atom_rot_k +
					9 * i + 3 * j + k];
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	rot_tensor[i * 9 + j * 3 + k] = 
	  tensor3_rotation_elem(tensor, rot_cartesian, i, j, k);
      }
    }
  }
}

static double tensor3_rotation_elem(const double tensor[27],
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
	sum += r[l * 3 + i] * r[m * 3 + j] * r[n * 3 + k] *
	  tensor[i * 9 + j * 3 + k];
      }
    }
  }
  return sum;
}


static int distribute_fc4(double *fc4,
			  const int fourth_atom,
			  const int *atom_mapping,
			  const int num_atom,
			  const double *rot_cart)
{
  int i, j, k, atom_rot_i, atom_rot_j, atom_rot_k, fourth_atom_rot;
  double *tensor;

  fourth_atom_rot = atom_mapping[fourth_atom];
  
  for (i = 0; i < num_atom; i++) {
    atom_rot_i = atom_mapping[i];

    for (j = 0; j < num_atom; j++) {
      atom_rot_j = atom_mapping[j];

      for (k = 0; k < num_atom; k++) {
	atom_rot_k = atom_mapping[k];

	tensor = (fc4 +
		  81 * num_atom * num_atom * num_atom * fourth_atom +
		  81 * num_atom * num_atom * i +
		  81 * num_atom * j +
		  81 * k);
	tensor4_roation(tensor,
			fc4,
			fourth_atom,
			i,
			j,
			k,
			fourth_atom_rot,
			atom_rot_i,
			atom_rot_j,
			atom_rot_k,
			num_atom,
			rot_cart);
      }
    }
  }
  return 1;
}

static void tensor4_roation(double *rot_tensor,
			    const double *fc4,
			    const int atom_i,
			    const int atom_j,
			    const int atom_k,
			    const int atom_l,
			    const int atom_rot_i,
			    const int atom_rot_j,
			    const int atom_rot_k,
			    const int atom_rot_l,
			    const int num_atom,
			    const double *rot_cartesian)
{
  int i, j, k, l;
  double tensor[81];

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  tensor[i * 27 + j * 9 + k * 3 + l] =
	    fc4[81 * num_atom * num_atom * num_atom * atom_rot_i +
		81 * num_atom * num_atom * atom_rot_j +
		81 * num_atom * atom_rot_k +
		81 * atom_rot_l +
		27 * i + 9 * j + 3 * k + l];
	}
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	  rot_tensor[i * 27 + j * 9 + k * 3 + l] = 
	    tensor4_rotation_elem(tensor, rot_cartesian, i, j, k, l);
	}
      }
    }
  }
}

static double tensor4_rotation_elem(const double tensor[81],
				    const double *r,
				    const int m,
				    const int n,
				    const int p,
				    const int q)
{
  int i, j, k, l;
  double sum;

  sum = 0.0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	for (l = 0; l < 3; l++) {
	sum += r[m * 3 + i] * r[n * 3 + j] * r[p * 3 + k] * r[q * 3 + l] *
	  tensor[i * 27 + j * 9 + k * 3 + l];
	}
      }
    }
  }
  return sum;
}

