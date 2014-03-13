#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "lapack_wrapper.h"
#include "phonoc_array.h"
#include "phonoc_utils.h"
#include "phonon3_h/fc3.h"
#include "phonon3_h/interaction.h"
#include "phonon3_h/imag_self_energy.h"
#include "phonon3_h/imag_self_energy_thm.h"
#include "other_h/isotope.h"
#include "spglib_h/kpoint.h"
#include "spglib_h/tetrahedron_method.h"


static PyObject * py_get_jointDOS(PyObject *self, PyObject *args);

static PyObject * py_get_interaction(PyObject *self, PyObject *args);
static PyObject * py_get_imag_self_energy(PyObject *self, PyObject *args);
static PyObject * py_get_imag_self_energy_at_bands(PyObject *self,
						   PyObject *args);
static PyObject * py_get_thm_imag_self_energy_at_bands(PyObject *self,
						       PyObject *args);
static PyObject * py_set_phonons_at_gridpoints(PyObject *self, PyObject *args);
static PyObject * py_get_phonon(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);
static PyObject * py_get_isotope_strength(PyObject *self, PyObject *args);
static PyObject * py_get_thm_isotope_strength(PyObject *self, PyObject *args);
static PyObject * py_set_permutation_symmetry_fc3(PyObject *self,
						  PyObject *args);
static PyObject *py_get_neighboring_gird_points(PyObject *self, PyObject *args);
static PyObject * py_set_integration_weights(PyObject *self, PyObject *args);
static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args);
static void get_triplet_tetrahedra_vertices
  (int vertices[2][24][4],
   SPGCONST int relative_grid_address[2][24][4][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int bz_grid_address[][3],
   const int bz_map[]);

static PyMethodDef functions[] = {
  {"joint_dos", py_get_jointDOS, METH_VARARGS, "Calculate joint density of states"},
  {"interaction", py_get_interaction, METH_VARARGS, "Interaction of triplets"},
  {"imag_self_energy", py_get_imag_self_energy, METH_VARARGS, "Imaginary part of self energy"},
  {"imag_self_energy_at_bands", py_get_imag_self_energy_at_bands, METH_VARARGS, "Imaginary part of self energy at phonon frequencies of bands"},
  {"thm_imag_self_energy_at_bands", py_get_thm_imag_self_energy_at_bands, METH_VARARGS, "Imaginary part of self energy at phonon frequencies of bands for tetrahedron method"},
  {"phonons_at_gridpoints", py_set_phonons_at_gridpoints, METH_VARARGS, "Set phonons at grid points"},
  {"phonon", py_get_phonon, METH_VARARGS, "Get phonon"},
  {"distribute_fc3", py_distribute_fc3, METH_VARARGS, "Distribute least fc3 to full fc3"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {"isotope_strength", py_get_isotope_strength, METH_VARARGS, "Isotope scattering strength"},
  {"thm_isotope_strength", py_get_thm_isotope_strength, METH_VARARGS, "Isotope scattering strength for tetrahedron_method"},
  {"permutation_symmetry_fc3", py_set_permutation_symmetry_fc3, METH_VARARGS, "Set permutation symmetry for fc3"},
  {"neighboring_grid_points", py_get_neighboring_gird_points, METH_VARARGS, "Neighboring grid points by relative grid addresses"},
  {"integration_weights", py_set_integration_weights, METH_VARARGS, "Integration weights of tetrahedron method"},
  {"triplets_integration_weights", py_set_triplets_integration_weights, METH_VARARGS, "Integration weights of tetrahedron metod for triplets"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_phono3py(void)
{
  Py_InitModule3("_phono3py", functions, "C-extension for phono3py\n\n...\n");
  return;
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
  const int* grid_address = (int*)grid_address_py->data;
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
  free(fc3);
  free(svecs);
  free(multi);
  
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

static PyObject * py_get_thm_imag_self_energy_at_bands(PyObject *self,
						       PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* fc3_normal_squared_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* grid_point_triplets_py;
  PyArrayObject* triplet_weights_py;
  PyArrayObject* g_py;
  double unit_conversion_factor, cutoff_frequency, temperature;

  if (!PyArg_ParseTuple(args, "OOOOOdOdd",
			&gamma_py,
			&fc3_normal_squared_py,
			&grid_point_triplets_py,
			&triplet_weights_py,
			&frequencies_py,
			&temperature,
			&g_py,
			&unit_conversion_factor,
			&cutoff_frequency)) {
    return NULL;
  }


  Darray* fc3_normal_squared = convert_to_darray(fc3_normal_squared_py);
  double* gamma = (double*)gamma_py->data;
  double* g = (double*)g_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* grid_point_triplets = (int*)grid_point_triplets_py->data;
  const int* triplet_weights = (int*)triplet_weights_py->data;

  get_thm_imag_self_energy_at_bands(gamma,
				    fc3_normal_squared,
				    frequencies,
				    grid_point_triplets,
				    triplet_weights,
				    g,
				    temperature,
				    unit_conversion_factor,
				    cutoff_frequency);

  free(fc3_normal_squared);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_isotope_strength(PyObject *self, PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* band_indices_py;
  PyArrayObject* mass_variances_py;
  int grid_point;
  int num_grid_points;
  double cutoff_frequency;
  double sigma;

  if (!PyArg_ParseTuple(args, "OiOOOOidd",
			&gamma_py,
			&grid_point,
			&mass_variances_py,
			&frequencies_py,
			&eigenvectors_py,
			&band_indices_py,
			&num_grid_points,
			&sigma,
			&cutoff_frequency)) {
    return NULL;
  }


  double* gamma = (double*)gamma_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const lapack_complex_double* eigenvectors =
    (lapack_complex_double*)eigenvectors_py->data;
  const int* band_indices = (int*)band_indices_py->data;
  const double* mass_variances = (double*)mass_variances_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_band0 = (int)band_indices_py->dimensions[0];

  int i, j, k;
  double f, f0;
  int *weights, *ir_grid_points;
  double *integration_weights;

  ir_grid_points = (int*)malloc(sizeof(int) * num_grid_points);
  weights = (int*)malloc(sizeof(int) * num_grid_points);
  integration_weights = (double*)malloc(sizeof(double) *
  					num_grid_points * num_band0 * num_band);

  for (i = 0; i < num_grid_points; i++) {
    ir_grid_points[i] = i;
    weights[i] = 1;
    for (j = 0; j < num_band0; j++) {
      f0 = frequencies[grid_point * num_band + band_indices[j]];
      for (k = 0; k < num_band; k++) {
  	f = frequencies[i * num_band + k];
  	integration_weights[i * num_band0 * num_band +
  			    j * num_band + k] = gaussian(f - f0, sigma);
      }
    }
  }

  get_thm_isotope_scattering_strength(gamma,
  				      grid_point,
  				      ir_grid_points,
  				      weights,
  				      mass_variances,
  				      frequencies,
  				      eigenvectors,
  				      num_grid_points,
  				      band_indices,
  				      num_band,
  				      num_band0,
  				      integration_weights,
  				      cutoff_frequency);
      
  free(ir_grid_points);
  free(weights);
  free(integration_weights);
  
  /* get_isotope_scattering_strength(gamma, */
  /* 				  grid_point, */
  /* 				  mass_variances, */
  /* 				  frequencies, */
  /* 				  eigenvectors, */
  /* 				  num_grid_points, */
  /* 				  band_indices, */
  /* 				  num_band, */
  /* 				  num_band0, */
  /* 				  sigma, */
  /* 				  cutoff_frequency); */
  
  Py_RETURN_NONE;
}

static PyObject * py_get_thm_isotope_strength(PyObject *self, PyObject *args)
{
  PyArrayObject* gamma_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* eigenvectors_py;
  PyArrayObject* band_indices_py;
  PyArrayObject* mass_variances_py;
  PyArrayObject* ir_grid_points_py;
  PyArrayObject* weights_py;
  int grid_point;
  double cutoff_frequency;
  PyArrayObject* integration_weights_py;


  if (!PyArg_ParseTuple(args, "OiOOOOOOOd",
			&gamma_py,
			&grid_point,
			&ir_grid_points_py,
			&weights_py,
			&mass_variances_py,
			&frequencies_py,
			&eigenvectors_py,
			&band_indices_py,
			&integration_weights_py,
			&cutoff_frequency)) {
    return NULL;
  }


  double* gamma = (double*)gamma_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int* ir_grid_points = (int*)ir_grid_points_py->data;
  const int* weights = (int*)weights_py->data;
  const lapack_complex_double* eigenvectors =
    (lapack_complex_double*)eigenvectors_py->data;
  const int* band_indices = (int*)band_indices_py->data;
  const double* mass_variances = (double*)mass_variances_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_band0 = (int)band_indices_py->dimensions[0];
  const double* integration_weights = (double*)integration_weights_py->data;
  const int num_ir_grid_points = (int)ir_grid_points_py->dimensions[0];
    
  get_thm_isotope_scattering_strength(gamma,
				      grid_point,
				      ir_grid_points,
				      weights,
				      mass_variances,
				      frequencies,
				      eigenvectors,
				      num_ir_grid_points,
				      band_indices,
				      num_band,
				      num_band0,
				      integration_weights,
				      cutoff_frequency);
  
  Py_RETURN_NONE;
}

static PyObject * py_get_jointDOS(PyObject *self, PyObject *args)
{
  PyArrayObject* jointdos;
  PyArrayObject* frequency_points_py;
  PyArrayObject* triplets_py;
  PyArrayObject* triplets_weights_py;
  PyArrayObject* frequencies_py;
  double sigma;

  if (!PyArg_ParseTuple(args, "OOOOOd",
			&jointdos,
			&frequency_points_py,
			&triplets_py,
			&triplets_weights_py,
			&frequencies_py,
			&sigma)) {
    return NULL;
  }
  
  double* jdos = (double*)jointdos->data;
  const double* freq_points = (double*)frequency_points_py->data;
  const int* triplets = (int*)triplets_py->data;
  const int* weights = (int*)triplets_weights_py->data;
  const double* frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_fpoints = (int)frequency_points_py->dimensions[0];
  const int num_triplet = (int)triplets_weights_py->dimensions[0];

  get_jointDOS(jdos,
	       num_fpoints,
	       num_triplet,
	       num_band,
	       freq_points,
	       frequencies,
	       triplets,
	       weights,
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

static PyObject * py_set_permutation_symmetry_fc3(PyObject *self, PyObject *args)
{
  PyArrayObject* fc3_py;

  if (!PyArg_ParseTuple(args, "O",
			&fc3_py)) {
    return NULL;
  }

  double* fc3 = (double*)fc3_py->data;
  const int num_atom = (int)fc3_py->dimensions[0];

  set_permutation_symmetry_fc3(fc3, num_atom);

  Py_RETURN_NONE;
}

static PyObject * py_get_neighboring_gird_points(PyObject *self, PyObject *args)
{
  PyArrayObject* relative_grid_points_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOO",
			&relative_grid_points_py,
			&grid_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&bz_grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  int* relative_grid_points = (int*)relative_grid_points_py->data;
  const int *grid_points = (int*)grid_points_py->data;
  const int num_grid_points = (int)grid_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[3] =
    (int(*)[3])relative_grid_address_py->data;
  const int num_relative_grid_address = relative_grid_address_py->dimensions[0];
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*bz_grid_address)[3] = (int(*)[3])bz_grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;

  int i;
#pragma omp parallel for
  for (i = 0; i < num_grid_points; i++) {
    kpt_get_neighboring_grid_points
      (relative_grid_points + i * num_relative_grid_address,
       grid_points[i],
       relative_grid_address,
       num_relative_grid_address,
       mesh,
       bz_grid_address,
       bz_map);
  }
  
  Py_RETURN_NONE;
}

static PyObject * py_set_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* grid_points_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&grid_points_py,
			&frequencies_py,
			&bz_grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[4][3] =
    (int(*)[4][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int *grid_points = (int*)grid_points_py->data;
  const int num_gp = (int)grid_points_py->dimensions[0];
  SPGCONST int (*bz_grid_address)[3] = (int(*)[3])bz_grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];

  int i, j, k, bi;
  int vertices[24][4];
  double freq_vertices[24][4];
    
#pragma omp parallel for private(j, k, bi, vertices, freq_vertices)
  for (i = 0; i < num_gp; i++) {
    for (j = 0; j < 24; j++) {
      kpt_get_neighboring_grid_points(vertices[j],
				      grid_points[i],
				      relative_grid_address[j],
				      4,
				      mesh,
				      bz_grid_address,
				      bz_map);
    }
    for (bi = 0; bi < num_band; bi++) {
      for (j = 0; j < 24; j++) {
	for (k = 0; k < 4; k++) {
	  freq_vertices[j][k] = frequencies[vertices[j][k] * num_band + bi];
	}
      }
      for (j = 0; j < num_band0; j++) {
	iw[i * num_band0 * num_band + j * num_band + bi] =
	  thm_get_integration_weight(frequency_points[j], freq_vertices, 'I');
      }
    }
  }
	    
  Py_RETURN_NONE;
}

static PyObject *
py_set_triplets_integration_weights(PyObject *self, PyObject *args)
{
  PyArrayObject* iw_py;
  PyArrayObject* frequency_points_py;
  PyArrayObject* relative_grid_address_py;
  PyArrayObject* mesh_py;
  PyArrayObject* triplets_py;
  PyArrayObject* frequencies_py;
  PyArrayObject* bz_grid_address_py;
  PyArrayObject* bz_map_py;
  if (!PyArg_ParseTuple(args, "OOOOOOOO",
			&iw_py,
			&frequency_points_py,
			&relative_grid_address_py,
			&mesh_py,
			&triplets_py,
			&frequencies_py,
			&bz_grid_address_py,
			&bz_map_py)) {
    return NULL;
  }

  double *iw = (double*)iw_py->data;
  const double *frequency_points = (double*)frequency_points_py->data;
  const int num_band0 = frequency_points_py->dimensions[0];
  SPGCONST int (*relative_grid_address)[4][3] =
    (int(*)[4][3])relative_grid_address_py->data;
  const int *mesh = (int*)mesh_py->data;
  SPGCONST int (*triplets)[3] = (int(*)[3])triplets_py->data;
  const int num_triplets = (int)triplets_py->dimensions[0];
  SPGCONST int (*bz_grid_address)[3] = (int(*)[3])bz_grid_address_py->data;
  const int *bz_map = (int*)bz_map_py->data;
  const double *frequencies = (double*)frequencies_py->data;
  const int num_band = (int)frequencies_py->dimensions[1];
  const int num_iw = (int)iw_py->dimensions[0];

  int i, j, k, l, b1, b2, sign;
  int tp_relative_grid_address[2][24][4][3];
  int vertices[2][24][4];
  int adrs_shift;
  double f0, f1, f2, g0, g1, g2;
  double freq_vertices[3][24][4];
    
  for (i = 0; i < 2; i++) {
    sign = 1 - i * 2;
    for (j = 0; j < 24; j++) {
      for (k = 0; k < 4; k++) {
	for (l = 0; l < 3; l++) {
	  tp_relative_grid_address[i][j][k][l] = 
	    relative_grid_address[j][k][l] * sign;
	}
      }
    }
  }

#pragma omp parallel for private(j, k, b1, b2, sign, vertices, adrs_shift, f0, f1, f2, g0, g1, g2, freq_vertices)
  for (i = 0; i < num_triplets; i++) {
    get_triplet_tetrahedra_vertices(vertices,
				    tp_relative_grid_address,
				    mesh,
				    triplets[i],
				    bz_grid_address,
				    bz_map);
    for (b1 = 0; b1 < num_band; b1++) {
      for (b2 = 0; b2 < num_band; b2++) {
	for (j = 0; j < 24; j++) {
	  for (k = 0; k < 4; k++) {
	    f1 = frequencies[vertices[0][j][k] * num_band + b1];
	    f2 = frequencies[vertices[1][j][k] * num_band + b2];
	    freq_vertices[0][j][k] = f1 + f2;
	    freq_vertices[1][j][k] = -f1 + f2;
	    freq_vertices[2][j][k] = f1 - f2;
	  }
	}
	for (j = 0; j < num_band0; j++) {
	  f0 = frequency_points[j];
	  g0 = thm_get_integration_weight(f0, freq_vertices[0], 'I');
	  g1 = thm_get_integration_weight(f0, freq_vertices[1], 'I');
	  g2 = thm_get_integration_weight(f0, freq_vertices[2], 'I');
	  adrs_shift = i * num_band0 * num_band * num_band +
	    j * num_band * num_band + b1 * num_band + b2;
	  iw[adrs_shift] = g0;
	  adrs_shift += num_triplets * num_band0 * num_band * num_band;
	  iw[adrs_shift] = g1 - g2;
	  if (num_iw == 3) {
	    adrs_shift += num_triplets * num_band0 * num_band * num_band;
	    iw[adrs_shift] = g0 + g1 + g2;
	  }
	}
      }	
    }
  }
	    
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




static void get_triplet_tetrahedra_vertices
  (int vertices[2][24][4],
   SPGCONST int relative_grid_address[2][24][4][3],
   const int mesh[3],
   const int triplet[3],
   SPGCONST int bz_grid_address[][3],
   const int bz_map[])
{
  int i, j;

  for (i = 0; i < 2; i++) {
    for (j = 0; j < 24; j++) {
      kpt_get_neighboring_grid_points(vertices[i][j],
				      triplet[i + 1],
				      relative_grid_address[i][j],
				      4,
				      mesh,
				      bz_grid_address,
				      bz_map);
    }
  }
}

