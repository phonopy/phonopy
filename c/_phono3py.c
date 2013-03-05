#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "dynmat.h"

/* Boltzmann constant eV/K */
#define KB 8.6173382568083159E-05
/* Planck Constant for THz to Ev */
#define PlanckConstant 4.13566733E-3

typedef struct {
  int d1;
  int d2;
  int **data;
  int *_data;
} Array2D;

typedef struct {
  int d1;
  int d2;
  double **data;
  double *_data;
} DArray2D;

typedef struct {
  int d1;
  int d2;
  npy_cdouble **data;
  npy_cdouble *_data;
} CArray2D;

typedef struct {
  int d1;
  int *data;
} Array1D;

typedef struct {
  int d[4];
  double ****data;
} ShortestVecs;

static const int index_exchange[6][3] = { { 0, 1, 2 },
					  { 2, 0, 1 },
					  { 1, 2, 0 },
					  { 2, 1, 0 },
					  { 0, 2, 1 },
					  { 1, 0, 2 } };

static PyObject * py_get_interaction_strength(PyObject *self, PyObject *args);
static PyObject * py_get_triplet_interaction_strength(PyObject *self,
						      PyObject *args);
static PyObject * py_get_sum_in_primitive(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_reciprocal(PyObject *self, PyObject *args);
static PyObject * py_get_fc3_realspace(PyObject *self, PyObject *args);
static PyObject * py_get_gamma(PyObject *self, PyObject *args);
static PyObject * py_get_decay_channel(PyObject *self, PyObject *args);
static PyObject * py_get_jointDOS(PyObject *self, PyObject *args);
static PyObject * py_distribute_fc3(PyObject *self, PyObject *args);
static PyObject * py_phonopy_zheev(PyObject *self, PyObject *args);
static int get_interaction_strength(double *amps,
				    const double *q0,
				    const double *q1s,
				    const double *q2s,
				    const Array1D *weights,
				    const double *fc2,
				    const double *fc3,
				    const double *masses,
				    const Array1D *p2s,
				    const Array1D *s2p,
				    const Array2D *multi,
				    const ShortestVecs *svecs,
				    const double cutoff_frequency,
				    const int is_symmetrize_fc3_q,
				    const int r2q_TI_index,
				    const double symprec);
static int get_triplet_interaction_strength(double *amps,
					    const double *fc3,
					    const double *q_vecs,
					    const npy_cdouble* eigvecs,
					    const double *freqs,
					    const double *masses,
					    const Array1D *p2s,
					    const Array1D *s2p,
					    const Array2D *multi,
					    const ShortestVecs *svecs,
					    const Array1D *band_indices,
					    const double cutoff_frequency,
					    const int is_symmetrize_fc3_q,
					    const int r2q_TI_index,
					    const double symprec);
static int sum_interaction_strength(double *amps,
				    const npy_cdouble* eigvecs,
				    const double *freqs,
				    const double *masses,
				    const Array1D *p2s,
				    const Array2D *multi,
				    const ShortestVecs *svecs,
				    const CArray2D *fc3_q,
				    const Array1D *band_indices,
				    const double cutoff_frequency);
static double get_sum_in_primivie(const npy_cdouble *fc3,
				  const npy_cdouble *e1,
				  const npy_cdouble *e2,
				  const npy_cdouble *e3,
				  const int num_atom,
				  const double *m);
static int get_fc3_realspace(npy_cdouble* fc3_real,
			     const ShortestVecs * svecs,
			     const Array2D * multi,
			     const double* q_triplet,
			     const Array1D * s2p,
			     const npy_cdouble* fc3_rec,
			     const double symprec);
static npy_cdouble get_phase_factor(const double q[3],
				    const int s_atom_index,
				    const int p_atom_index,
				    const int sign,
				    const ShortestVecs * svecs,
				    const Array2D * multi);
static int get_fc3_reciprocal(npy_cdouble* fc3_q,
			      const ShortestVecs * svecs,
			      const Array2D * multi,
			      const DArray2D * q,
			      const Array1D * p2s,
			      const Array1D * s2p,
			      const double* fc3,
			      const int r2q_TI_index,
			      const double symprec);
static int get_fc3_sum_in_supercell(npy_cdouble fc3_q[3][3][3],
				    const int i1,
				    const int i2,
				    const int i3,
				    const ShortestVecs * svecs,
				    const Array2D * multi,
				    const DArray2D * q,
				    const Array1D * s2p,
				    const Array1D * p2s,
				    const double* fc3,
				    const int r2q_TI_index,
				    const double symprec);
static int distribute_fc3(double *fc3,
			  const int third_atom,
			  const int third_atom_rot,
			  const double *positions,
			  const int num_atom,
			  const int rot[3][3],
			  const double *rot_cartesian,
			  const double *trans,
			  const double symprec);
static int get_atom_by_symmetry(const int atom_number,
				const int num_atom,
				const double *positions,
				const int rot[3][3],
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
static npy_cdouble prod(const npy_cdouble a, const npy_cdouble b);
static int nint(const double a);
static double gauss(const double x, const double sigma);
static double bs(const double x, const double t);
static ShortestVecs * get_shortest_vecs(PyArrayObject* shortest_vectors);
static void free_shortest_vecs(ShortestVecs * svecs);
static Array2D * alloc_Array2D(const int index1, const int index2);
static void free_Array2D(Array2D * array);
static DArray2D * alloc_DArray2D(const int index1, const int index2);
static void free_DArray2D(DArray2D * array);
static CArray2D * alloc_CArray2D(const int index1, const int index2);
static void free_CArray2D(CArray2D * array);
static Array1D * alloc_Array1D(const int index1);
static void free_Array1D(Array1D * array);
static int phonopy_zheev(double *w,
			 lapack_complex_double *a,
			 const int n);


static PyMethodDef functions[] = {
  {"interaction_strength", py_get_interaction_strength, METH_VARARGS, "Interaction strength of triplets"},
  {"triplet_interaction_strength", py_get_triplet_interaction_strength, METH_VARARGS, "One triplet interaction strength"},
  {"sum_in_primitive", py_get_sum_in_primitive, METH_VARARGS, "Summation in primitive cell"},
  {"fc3_reciprocal", py_get_fc3_reciprocal, METH_VARARGS, "Transformation of fc3 from real to reciprocal space"},
  {"fc3_realspace", py_get_fc3_realspace, METH_VARARGS, "Transformation of fc3 from reciprocal to real space"},
  {"gamma", py_get_gamma, METH_VARARGS, "Calculate damping function with gaussian smearing"},
  {"joint_dos", py_get_jointDOS, METH_VARARGS, "Calculate joint density of states"},
  {"decay_channel", py_get_decay_channel, METH_VARARGS, "Calculate decay of phonons"},
  {"distribute_fc3", py_distribute_fc3, METH_VARARGS, "Distribute least fc3 to full fc3"},
  {"zheev", py_phonopy_zheev, METH_VARARGS, "Lapack zheev wrapper"},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_phono3py(void)
{
  Py_InitModule3("_phono3py", functions, "C-extension for phono3py\n\n...\n");
  return;
}

static PyObject * py_get_interaction_strength(PyObject *self, PyObject *args)
{
  int i, j;
  Array1D * s2p, * p2s, *w;
  Array2D * multi;
  ShortestVecs * svecs;

  PyArrayObject* amplitude;
  PyArrayObject* qvec0;
  PyArrayObject* qvec1s;
  PyArrayObject* qvec2s;
  PyArrayObject* weights;
  PyArrayObject* shortest_vectors;
  PyArrayObject* multiplicity;
  PyArrayObject* p2s_map;
  PyArrayObject* s2p_map;
  PyArrayObject* force_constants_second;
  PyArrayObject* force_constants_third;
  PyArrayObject* atomic_masses;
  double symprec, cutoff_frequency;
  int r2q_TI_index, is_symmetrize_fc3_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOdiid",
			&amplitude,
			&qvec0,
			&qvec1s,
			&qvec2s,
			&weights,
			&shortest_vectors,
			&multiplicity,
			&p2s_map,
			&s2p_map,
			&force_constants_second,
			&force_constants_third,
			&atomic_masses,
			&cutoff_frequency,
			&is_symmetrize_fc3_q,
			&r2q_TI_index,
			&symprec)) {
    return NULL;
  }

  double* amps = (double*)amplitude->data;
  const double *q0 = (double*)qvec0->data;
  const double *q1s = (double*)qvec1s->data;
  const double *q2s = (double*)qvec2s->data;
  const long *weights_long = (long*)weights->data;
  const double* masses = (double*)atomic_masses->data;
  const long* multiplicity_long = (long*)multiplicity->data;
  const long* p2s_map_long = (long*)p2s_map->data;
  const long* s2p_map_long = (long*)s2p_map->data;
  const double* fc2 = (double*)force_constants_second->data;
  const double* fc3 = (double*)force_constants_third->data;

  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];
  const int num_triplets = (int)weights->dimensions[0];
  

  svecs = get_shortest_vecs(shortest_vectors);
  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multiplicity_long[i * num_patom + j];
    }
  }

  s2p = alloc_Array1D(num_satom);
  for (i = 0; i < num_satom; i++) {
    s2p->data[i] = s2p_map_long[i];
  }
  p2s = alloc_Array1D(num_patom);
  for (i = 0; i < num_patom; i++) {
    p2s->data[i] = p2s_map_long[i];
  }
  w = alloc_Array1D(num_triplets);
  for (i = 0; i < num_triplets; i++) {
    w->data[i] = weights_long[i];
  }

  get_interaction_strength(amps,
			   q0,
			   q1s,
			   q2s,
			   w,
			   fc2,
			   fc3,
			   masses,
			   p2s,
			   s2p,
			   multi,
			   svecs,
			   cutoff_frequency,
			   is_symmetrize_fc3_q,
			   r2q_TI_index,
			   symprec);

  free_Array1D(p2s);
  free_Array1D(s2p);
  free_Array2D(multi);
  free_Array1D(w);
  free_shortest_vecs(svecs);

  Py_RETURN_NONE;
}

static PyObject * py_get_triplet_interaction_strength(PyObject *self,
						      PyObject *args)
{
  int i, j;
  Array1D * s2p, * p2s, * band_indices;
  Array2D * multi;
  ShortestVecs * svecs;

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
  double symprec, cutoff_frequency;
  int r2q_TI_index, is_symmetrize_fc3_q;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOdiid",
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
			&r2q_TI_index,
			&symprec)) {
    return NULL;
  }

  double* amps = (double*)amplitude->data;
  const double* freqs = (double*)omegas->data;
  const npy_cdouble* eigvecs = (npy_cdouble*)eigenvectors->data;
  const double* masses = (double*)atomic_masses->data;
  const long* multiplicity_long = (long*)multiplicity->data;
  const long* p2s_map_long = (long*)p2s_map->data;
  const long* s2p_map_long = (long*)s2p_map->data;
  const long* band_indices_long = (long*)set_of_band_indices->data;
  const double* fc3 = (double*)force_constants_third->data;
  const double* q_vecs = (double*)q_triplet->data;

  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];
  const int num_band0 = (int)set_of_band_indices->dimensions[0];

  svecs = get_shortest_vecs(shortest_vectors);
  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multiplicity_long[i * num_patom + j];
    }
  }

  s2p = alloc_Array1D(num_satom);
  for (i = 0; i < num_satom; i++) {
    s2p->data[i] = s2p_map_long[i];
  }
  p2s = alloc_Array1D(num_patom);
  for (i = 0; i < num_patom; i++) {
    p2s->data[i] = p2s_map_long[i];
  }
  band_indices = alloc_Array1D(num_band0);
  for (i = 0; i < num_band0; i++) {
    band_indices->data[i] = band_indices_long[i];
  }

  get_triplet_interaction_strength(amps,
				   fc3,
				   q_vecs,
				   eigvecs,
				   freqs,
				   masses,
				   p2s,
				   s2p,
				   multi,
				   svecs,
				   band_indices,
				   cutoff_frequency,
				   is_symmetrize_fc3_q,
				   r2q_TI_index,
				   symprec);

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

  const npy_cdouble* fc3 = (npy_cdouble*)force_constants_third->data;
  const npy_cdouble* e1 = (npy_cdouble*)eigvec1->data;
  const npy_cdouble* e2 = (npy_cdouble*)eigvec2->data;
  const npy_cdouble* e3 = (npy_cdouble*)eigvec3->data;
  const double* m = (double*)masses->data;
  const int num_atom = (int) masses->dimensions[0];


  return PyFloat_FromDouble(get_sum_in_primivie(fc3, e1, e2, e3, num_atom, m));
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
  const double symprec;
  const int r2q_TI_index;
  

  if (!PyArg_ParseTuple(args, "OOOOOOOid",
			&force_constants_third_reciprocal,
			&shortest_vectors,
			&multiplicity,
			&q_triplet,
			&p2s_map,
			&s2p_map,
			&force_constants_third,
			&r2q_TI_index,
			&symprec))
    return NULL;

  int i, j;
  Array1D * s2p, * p2s;
  Array2D * multi;
  DArray2D * q;
  ShortestVecs * svecs;
  npy_cdouble* fc3_q = (npy_cdouble*)force_constants_third_reciprocal->data;
  
  const long* multiplicity_long = (long*)multiplicity->data;
  const long* p2s_map_long = (long*)p2s_map->data;
  const long* s2p_map_long = (long*)s2p_map->data;
  const double* fc3 = (double*)force_constants_third->data;

  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];

  svecs = get_shortest_vecs(shortest_vectors);
  
  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multiplicity_long[i * num_patom + j];
    }
  }

  s2p = alloc_Array1D(num_satom);
  p2s = alloc_Array1D(num_patom);
  for (i = 0; i < num_satom; i++) {
    s2p->data[i] = s2p_map_long[i];
  }
  for (i = 0; i < num_patom; i++) {
    p2s->data[i] = p2s_map_long[i];
  }

  q = alloc_DArray2D(3, 3);
  for (i = 0; i < 3; i ++) {
    for (j = 0; j < 3; j ++) {
      q->data[i][j] = ((double*)q_triplet->data)[i * 3 + j];
    }
  }

  get_fc3_reciprocal(fc3_q,
		     svecs,
		     multi,
		     q,
		     s2p,
		     p2s,
		     fc3,
		     r2q_TI_index,
		     symprec);

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
  const double symprec;
  

  if (!PyArg_ParseTuple(args, "OOOOOOd",
			&fc3_realspace,
			&shortest_vectors,
			&multiplicity,
			&qpoints_triplet,
			&s2p_map,
			&fc3_reciprocal,
			&symprec))
    return NULL;

  int i, j;
  Array1D * s2p;
  Array2D * multi;
  ShortestVecs * svecs;
  npy_cdouble* fc3_real = (npy_cdouble*)fc3_realspace->data;
  const npy_cdouble* fc3_rec = (npy_cdouble*)fc3_reciprocal->data;
  const long* multi_long = (long*)multiplicity->data;
  const double* q_triplet = (double*)qpoints_triplet->data;
  const long* s2p_map_long = (long*)s2p_map->data;
 
  const int num_satom = (int)multiplicity->dimensions[0];
  const int num_patom = (int)multiplicity->dimensions[1];

  svecs = get_shortest_vecs(shortest_vectors);

  multi = alloc_Array2D(num_satom, num_patom);
  for (i = 0; i < num_satom; i++) {
    for (j = 0; j < num_patom; j++) {
      multi->data[i][j] = multi_long[i * num_patom + j];
    }
  }

  s2p = alloc_Array1D(num_satom);
  for (i = 0; i < num_satom; i++) {
    s2p->data[i] = s2p_map_long[i];
  }

  get_fc3_realspace(fc3_real,
		    svecs,
		    multi,
		    q_triplet,
		    s2p,
		    fc3_rec,
		    symprec);

  free_Array1D(s2p);
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


  int i, j, k, l;
  double f2, f3, n2, n3, a, factor2eV;
  double* dfun = (double*)gammas->data;
  double sum;
  const double* o = (double*)omegas->data;
  const double* amp = (double*)amplitudes->data;
  const long* w_long = (long*)weights->data;
  const double* f = (double*)frequencies->data;
  const int num_band0 = (int)amplitudes->dimensions[1];
  const int num_band = (int)amplitudes->dimensions[2];
  const int num_omega = (int)omegas->dimensions[0];
  const int num_triplet = (int)weights->dimensions[0];
  int w[num_triplet];
  for (i = 0; i < num_triplet; i++) {
    w[i] = w_long[i];
  }

  factor2eV = PlanckConstant / freq_factor;

  for (i = 0; i < num_omega; i++) {
    sum = 0.0;
#pragma omp parallel for private(k, l, a, f2, f3, n2, n3) reduction(+:sum)
    for (j = 0; j < num_triplet; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  f2 = f[j * 3 * num_band + num_band + k];
	  f3 = f[j * 3 * num_band + 2 * num_band + l];
	  if (f2 < cutoff_frequency || f3 < cutoff_frequency) {
	    continue;
	  }
	  a = amp[j * num_band0 * num_band * num_band +
		  band_index * num_band * num_band + k * num_band + l];

	  if (t > 0.0) {
	    n2 = bs(f2 * factor2eV, t);
	    n3 = bs(f3 * factor2eV, t);
	    switch (option) {
	    case 1:
	      sum += ((1.0 + n2 + n3) * gauss(f2 + f3 - o[i], sigma) +
	    	      (n3 - n2) * 2 *gauss(f2 - f3 - o[i], sigma)
	    	      ) * a * w[j];
	      break;
	    case 2:
	      sum += (1.0 + n2 + n3) * gauss(f2 + f3 - o[i], sigma) * a * w[j];
	      break;
	    case 3:
	      sum += (n3 - n2) * 2 *gauss(f2 - f3 - o[i], sigma) * a * w[j];
	      break;
	    case 4:
	      sum += (n3 - n2) * (gauss(f2 - f3 - o[i], sigma) -
	    			  gauss(f3 - f2 - o[i], sigma)) * a * w[j];
	      break;
	    case 5:
	      sum += (n3 - n2) * gauss(f2 - f3 - o[i], sigma) * a * w[j];
	      break;
	    case 6:
	      sum += (n2 - n3) * gauss(f3 - f2 - o[i], sigma) * a * w[j];
	      break;
	    case 0:
	    default:
	      sum += ((1.0 + n2 + n3) * gauss(f2 + f3 - o[i], sigma) +
	    	      (n3 - n2) * (gauss(f2 - f3 - o[i], sigma) -
	    			   gauss(f3 - f2 - o[i], sigma))
	    	      ) * a * w[j];
	      break;
	    }
	  } else {
	    sum += gauss(f2 + f3 - o[i], sigma) * a * w[j];
	  }
	}
      }
    }
    dfun[i] = sum;
  }

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
  

  int i, j, k, l;
  double f2, f3;
  double* jdos = (double*)jointdos->data;
  const double* o = (double*)omegas->data;
  const long* w_long = (long*)weights->data;
  const double* f = (double*)frequencies->data;
  const int num_band = (int)frequencies->dimensions[2];
  const int num_omega = (int)omegas->dimensions[0];
  const int num_triplet = (int)weights->dimensions[0];
  int w[num_triplet];
  for (i = 0; i < num_triplet; i++) {
    w[i] = w_long[i];
  }
  
#pragma omp parallel for private(j, k, l, f2, f3)
  for (i = 0; i < num_omega; i++) {
    jdos[i] = 0.0;
    for (j = 0; j < num_triplet; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_band; l++) {
	  f2 = f[j * 3 * num_band + num_band + k];
	  f3 = f[j * 3 * num_band + 2 * num_band + l];
	  jdos[i] += gauss(f2 + f3 - o[i], sigma) * w[j];
	}
      }
    }
  }
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
  

  int i, j, k, l, address_a, address_d;
  double f2, f3, n2, n3, factor2eV;
  double* decay = (double*)decay_values->data;
  const double* amp = (double*)amplitudes->data;
  const double* f = (double*)frequencies->data;
  const double* o = (double*)omegas->data;
  
  const int num_band = (int)amplitudes->dimensions[2];
  const int num_triplet = (int)amplitudes->dimensions[0];
  const int num_omega = (int)omegas->dimensions[0];

  factor2eV = PlanckConstant / freq_factor;

#pragma omp parallel for private(j, k, l, address_a, address_d, f2, f3, n2, n3)
  for (i = 0; i < num_triplet; i++) {
    for (j = 0; j < num_band; j++) {
      for (k = 0; k < num_band; k++) {
	for (l = 0; l < num_omega; l++) {
	  address_a = i * num_omega * num_band * num_band + l * num_band * num_band + j * num_band + k;
	  address_d = i * num_band * num_band + j * num_band + k;
	  f2 = f[i * 3 * num_band + num_band + j];
	  f3 = f[i * 3 * num_band + 2 * num_band + k];
	  if (t > 0) {
	    n2 = bs(f2 * factor2eV, t);
	    n3 = bs(f3 * factor2eV, t);
	    decay[address_d] += ((1.0 + n2 + n3) * gauss(f2 + f3 - o[l], sigma) +
				 (n3 - n2) * (gauss(f2 - f3 - o[l], sigma) -
					      gauss(f3 - f2 - o[l], sigma))
				 ) * amp[address_a];
	  } else {
	    decay[address_d] += gauss(f2 + f3 - o[l], sigma) * amp[address_a];
	  }
	}
      }
    }
  }
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

  int i, j;
  double* fc3 = (double*)force_constants_third->data;
  const double* pos = (double*)positions->data;
  const long* rot_long = (long*)rotation->data;
  int rot_int[3][3];
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      rot_int[i][j] = rot_long[i*3 + j];
    }
  }
  const double* rot_cart_inv = (double*)rotation_cart_inv->data;
  const double* trans = (double*)translation->data;
  const int num_atom = (int)positions->dimensions[0];

  return PyInt_FromLong((long) distribute_fc3(fc3,
					      third_atom,
					      third_atom_rot,
					      pos,
					      num_atom,
					      rot_int,
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

  info = phonopy_zheev(eigvals, a, dimension);

  for (i = 0; i < dimension * dimension; i++) {
    dynmat[i].real = lapack_complex_double_real(a[i]);
    dynmat[i].imag = lapack_complex_double_imag(a[i]);
  }

  return PyInt_FromLong((long) info);
}

static int get_interaction_strength(double *amps,
				    const double *q0,
				    const double *q1s,
				    const double *q2s,
				    const Array1D *weights,
				    const double *fc2,
				    const double *fc3,
				    const double *masses,
				    const Array1D *p2s,
				    const Array1D *s2p,
				    const Array2D *multi,
				    const ShortestVecs *svecs,
				    const double cutoff_frequency,
				    const int is_symmetrize_fc3_q,
				    const int r2q_TI_index,
				    const double symprec)
{
  int i, num_triplets;
  double q1[3], q2[3];

  num_triplets = weights->d1;
  for (i = 0; i < num_triplets; i++) {
    for (j = 0; j < 3; j++) {
      q1[3] = q1s[i * 3 + j];
      q2[3] = q2s[i * 3 + j];
    }
    
  }
}

static int get_triplet_interaction_strength(double *amps,
					    const double *fc3,
					    const double *q_vecs,
					    const npy_cdouble* eigvecs,
					    const double *freqs,
					    const double *masses,
					    const Array1D *p2s,
					    const Array1D *s2p,
					    const Array2D *multi,
					    const ShortestVecs *svecs,
					    const Array1D *band_indices,
					    const double cutoff_frequency,
					    const int is_symmetrize_fc3_q,
					    const int r2q_TI_index,
					    const double symprec)
{
  int i, j, k, num_patom;
  DArray2D * q;
  CArray2D * fc3_q;

  num_patom = p2s->d1;
  
  if (is_symmetrize_fc3_q == 0) {
    fc3_q = alloc_CArray2D(1, num_patom * num_patom * num_patom * 27);
  } else {
    fc3_q = alloc_CArray2D(6, num_patom * num_patom * num_patom * 27);
  }

  q = alloc_DArray2D(3, 3);
  for (i = 0; i < fc3_q->d1; i++) {
    for (j = 0; j < 3; j ++) {
      for (k = 0; k < 3; k ++) {
	q->data[ index_exchange[i][j] ][k] = q_vecs[j * 3 + k];
      }
    }
    get_fc3_reciprocal(fc3_q->data[i],
		       svecs,
		       multi,
		       q,
		       s2p,
		       p2s,
		       fc3,
		       r2q_TI_index,
		       symprec);
  }

  sum_interaction_strength(amps,
			   eigvecs,
			   freqs,
			   masses,
			   p2s,
			   multi,
			   svecs,
			   fc3_q,
			   band_indices,
			   cutoff_frequency);

  free_DArray2D(q);
  free_CArray2D(fc3_q);

  return 1;
}

static int sum_interaction_strength(double *amps,
				    const npy_cdouble* eigvecs,
				    const double *freqs,
				    const double *masses,
				    const Array1D *p2s,
				    const Array2D *multi,
				    const ShortestVecs *svecs,
				    const CArray2D *fc3_q,
				    const Array1D *band_indices,
				    const double cutoff_frequency)
{
  int i, j, k, n, num_band0, num_patom;
  int band[3];
  npy_cdouble * e[3];

  num_band0 = band_indices->d1;
  num_patom = p2s->d1;
  
#pragma omp parallel for private(i, j, k, band, e)
  for (n = 0; n < num_band0 * num_patom * num_patom * 9; n++) {
    band[0] = band_indices->data[n / (num_patom * num_patom * 9)];
    band[1] = (n % (num_patom * num_patom * 9)) / (num_patom * 3);
    band[2] = n % (num_patom * 3);

    if (freqs[band[0]] < cutoff_frequency ||
	freqs[num_patom * 3 + band[1]] < cutoff_frequency ||
	freqs[2 * num_patom * 3 + band[2]] < cutoff_frequency) {
      amps[n] = 0;
      continue;
    }

    for (i = 0; i < 3; i++) {
      e[i] = (npy_cdouble*) malloc(num_patom * 3 * sizeof(npy_cdouble));
    }

    /* If symmetrize fc3_q for index exchange, i = 0..5 else i = 0*/
    amps[n] = 0;
    for (i = 0; i < fc3_q->d1; i++) { 
      for (j = 0; j < num_patom * 3; j++) {
	for (k = 0; k < 3; k++) {
	  e[ index_exchange[i][k] ][j] =
	    eigvecs[k * num_patom * num_patom * 9 +
		    j * num_patom * 3 + band[k]];
	}
      }
      amps[n] += get_sum_in_primivie(fc3_q->data[i], e[0], e[1], e[2],
				     num_patom, masses);
    }
    amps[n] /= fc3_q->d1 *
      freqs[band[0]] *
      freqs[num_patom * 3 + band[1]] *
      freqs[2 * num_patom * 3 + band[2]];

    for (i = 0; i < 3; i++) {
      free(e[i]);
    }
  }

  return 1;
}

static int get_fc3_realspace(npy_cdouble* fc3_real,
			     const ShortestVecs* svecs,
			     const Array2D * multi,
			     const double* q_triplet,
			     const Array1D * s2p,
			     const npy_cdouble* fc3_rec,
			     const double symprec)
{
  int i, j, k, l, m, n;
  npy_cdouble phase2, phase3, fc3_elem;
  double q[3];
  int num_satom = multi->d1;
  int num_patom = multi->d2;

  for (i = 0; i < num_patom; i++) {
    for (j = 0; j < num_satom; j++) {
      q[0] = q_triplet[3];
      q[1] = q_triplet[4];
      q[2] = q_triplet[5];
      phase2 = get_phase_factor(q,
				j,
				i,
				-1,
				svecs,
				multi);
      for (k = 0; k < num_satom; k++) {
	q[0] = q_triplet[6];
	q[1] = q_triplet[7];
	q[2] = q_triplet[8];
	phase3 = get_phase_factor(q,
				  k,
				  i,
				  -1,
				  svecs,
				  multi);
	for (l = 0; l < 3; l++) { 
	  for (m = 0; m < 3; m++) {
	    for (n = 0; n < 3; n++) {
	      fc3_elem = prod(fc3_rec[i * num_patom * num_patom * 27 +
				      s2p->data[j] * num_patom * 27 +
				      s2p->data[k] * 27 + l * 9 + m * 3 + n],
			      prod(phase2, phase3));

	      fc3_real[i * num_satom * num_satom * 27 +
		       j * num_satom * 27 +
		       k * 27 + l * 9 + m * 3 + n].real += fc3_elem.real;
	      fc3_real[i * num_satom * num_satom * 27 +
		       j * num_satom * 27 +
		       k * 27 + l * 9 + m * 3 + n].imag += fc3_elem.imag;
		
	    }
	  }
	}
      }
    }
  }

  return 0;
}

static npy_cdouble get_phase_factor(const double q[3],
				    const int s_atom_index,
				    const int p_atom_index,
				    const int sign,
				    const ShortestVecs * svecs,
				    const Array2D * multi)
{
  int i, j;
  double phase;
  npy_cdouble exp_phase;
  int m = multi->data[s_atom_index][p_atom_index];
  
  exp_phase.real = 0.0;
  exp_phase.imag = 0.0;
  for (i = 0; i < m; i++) {
    phase = 0.0;
    for (j = 0; j < 3; j++) {
      phase += q[j] * svecs->data[s_atom_index][p_atom_index][i][j];
    }
    exp_phase.real += cos(phase * 2 * M_PI);
    exp_phase.imag += sin(phase * 2 * M_PI);
  }
  exp_phase.real /= m;
  exp_phase.imag /= m * sign;

  return exp_phase;
}

static int get_fc3_reciprocal(npy_cdouble* fc3_q,
			      const ShortestVecs * svecs,
			      const Array2D * multi,
			      const DArray2D * q,
			      const Array1D * s2p,
			      const Array1D * p2s,
			      const double* fc3,
			      const int r2q_TI_index,
			      const double symprec)
{
  int i, j, k, l, m, n, p;
  npy_cdouble fc3_q_local[3][3][3];
  int num_patom = p2s->d1;

#pragma omp parallel for private(i, j, k, l, m, n, fc3_q_local)
  for (p = 0; p < num_patom * num_patom * num_patom; p++) {
    i = p / (num_patom * num_patom);
    j = (p % (num_patom * num_patom)) / num_patom;
    k = p % num_patom;
    get_fc3_sum_in_supercell(fc3_q_local,
			     i,
			     j,
			     k,
			     svecs,
			     multi,
			     q,
			     s2p,
			     p2s,
			     fc3,
			     r2q_TI_index,
			     symprec);

    for (l = 0; l < 3; l++) { 
      for (m = 0; m < 3; m++) {
	for (n = 0; n < 3; n++) {
	  fc3_q[i * num_patom * num_patom * 27 +
		j * num_patom * 27 +
		k * 27 + l * 9 + m * 3 + n] = fc3_q_local[l][m][n];
	}
      }
    }
  }

  return 0;
}

static int get_fc3_sum_in_supercell(npy_cdouble fc3_q[3][3][3],
				    const int p1,
				    const int p2,
				    const int p3,
				    const ShortestVecs * svecs,
				    const Array2D * multi,
				    const DArray2D * q,
				    const Array1D * s2p,
				    const Array1D * p2s,
				    const double* fc3,
				    const int r2q_TI_index,
				    const double symprec)
{
  npy_cdouble phase2, phase3, phase_prod, phase_prim;
  int i, j, k, s1, s2, s3, address;
  double phase;
  int num_satom = s2p->d1;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	fc3_q[i][j][k].real = 0.0;
	fc3_q[i][j][k].imag = 0.0;
      }
    }
  }

  /* Phi(s1, s2, s3) */
  /* Sum in terms of s1 is not taken due to translational invariance. */
  /* Phase factor with q_1 is not used. */
  s1 = 0;
  for (i = 0; i < num_satom; i++) {
    if (p2s->data[p1] == s2p->data[i]) {
      if (fabs(svecs->data[i][p1][0][0]) < symprec &&
	  fabs(svecs->data[i][p1][0][1]) < symprec &&
	  fabs(svecs->data[i][p1][0][2]) < symprec) {
	s1 = i;
      }
    }
  }

  /* Sum in terms of s2 */
  for (s2 = 0; s2 < num_satom; s2++) {
    if (s2p->data[s2] == p2s->data[p2]) {
      phase2.real = 0.0;
      phase2.imag = 0.0;
      /* Supercell boundary treatment */
      for (i = 0; i < multi->data[s2][p1]; i++) {
	phase = 0.0;
	/* phi' = q' * [r(N;nu) - r(M;mu)] */
	for (j = 0; j < 3; j++) {
	  phase += q->data[1][j] * svecs->data[s2][p1][i][j];
	}
	phase2.real += cos(phase * 2 * M_PI);
	phase2.imag += sin(phase * 2 * M_PI);
      }
      phase2.real /= multi->data[s2][p1];
      phase2.imag /= multi->data[s2][p1];

      /* Sum in terms of s3 */
      for (s3 = 0; s3 < num_satom; s3++) {
	if (s2p->data[s3] == p2s->data[p3]) {
	  phase3.real = 0.0;
	  phase3.imag = 0.0;
	  for (i = 0; i < multi->data[s3][p1]; i++) {
	    phase = 0.0;
	    /* phi'' = q'' * [r(P;pi) - r(M;mu)] */
	    for (j = 0; j < 3; j++) {
	      phase += q->data[2][j] * svecs->data[s3][p1][i][j];
	    }
	    phase3.real += cos(phase * 2 * M_PI);
	    phase3.imag += sin(phase * 2 * M_PI);
	  }
	  phase3.real /= multi->data[s3][p1];
	  phase3.imag /= multi->data[s3][p1];

	  /* Fourier transform */
	  phase_prod = prod(phase2, phase3);

	  switch (r2q_TI_index) {
	  case 1:
	    address = s3 * num_satom * num_satom * 27 + s1 * num_satom * 27 + s2 * 27;
	    break;
	  case 2:
	    address = s2 * num_satom * num_satom * 27 + s3 * num_satom * 27 + s1 * 27;
	    break;
	  default:
	    address = s1 * num_satom * num_satom * 27 + s2 * num_satom * 27 + s3 * 27;
	    break;
	  }
	  for (i = 0; i < 3; i++) {
	    for (j = 0; j < 3; j++) {
	      for (k = 0; k < 3; k++) {
		fc3_q[i][j][k].real += fc3[address + i * 9 + j * 3 + k] * phase_prod.real;
		fc3_q[i][j][k].imag += fc3[address + i * 9 + j * 3 + k] * phase_prod.imag;
	      }
	    }
	  }
	}
      }
    }
  }

  phase = 0.0;
  for (i = 0; i < 3; i++) {
    phase += (q->data[0][i] + q->data[1][i] + q->data[2][i]) * svecs->data[p2s->data[p1]][0][0][i];
  }
  phase_prim.real = cos(phase * 2 * M_PI);
  phase_prim.imag = sin(phase * 2 * M_PI);

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
  	fc3_q[i][j][k] = prod(fc3_q[i][j][k], phase_prim);
      }
    }
  }

  return 0;
}

static double get_sum_in_primivie(const npy_cdouble *fc3,
				  const npy_cdouble *e1,
				  const npy_cdouble *e2,
				  const npy_cdouble *e3,
				  const int num_atom,
				  const double *mass)
{
  int i1, i2, i3, a, b, c, shift;
  npy_cdouble sum, local_sum, tmp_val;
  
  sum.real = 0.0;
  sum.imag = 0.0;

  for (i1 = 0; i1 < num_atom; i1++) {
    for (i2 = 0; i2 < num_atom; i2++) {
      for (i3 = 0; i3 < num_atom; i3++) {
	shift = i1 * num_atom * num_atom * 27 + i2 * num_atom * 27 + i3 * 27;
	local_sum.real = 0.0;
	local_sum.imag = 0.0;
	for (a = 0; a < 3; a++) {
	  for (b = 0; b < 3; b++) {
	    for (c = 0; c < 3; c++) {
	      tmp_val = prod(e1[i1 * 3 + a], e2[i2 * 3 + b]);
	      tmp_val = prod(tmp_val, e3[i3 * 3 + c]);
	      tmp_val = prod(fc3[shift + a * 9 + b * 3 + c], tmp_val);
	      local_sum.real += tmp_val.real;
	      local_sum.imag += tmp_val.imag;
	    }
	  }
	}
	sum.real += local_sum.real / sqrt(mass[i1] * mass[i2] * mass[i3]);
	sum.imag += local_sum.imag / sqrt(mass[i1] * mass[i2] * mass[i3]);
      }
    }
  }

  return sum.real * sum.real + sum.imag * sum.imag;
}

static int distribute_fc3(double *fc3,
			  const int third_atom,
			  const int third_atom_rot,
			  const double *positions,
			  const int num_atom,
			  const int rot[3][3],
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
				const int rot[3][3],
				const double *trans,
				const double symprec)
{
  int i, j, found;
  double rot_pos[3], diff[3];
  
  for (i = 0; i < 3; i++) {
    rot_pos[i] = trans[i];
    for (j = 0; j < 3; j++) {
      rot_pos[i] += rot[i][j] * positions[atom_number * 3 + j];
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


static npy_cdouble prod(const npy_cdouble a, const npy_cdouble b)
{
  npy_cdouble c;
  c.real = a.real * b.real - a.imag * b.imag;
  c.imag = a.imag * b.real + a.real * b.imag;
  return c;
}

static int nint(const double a)
{
  if (a < 0.0)
    return (int) (a - 0.5);
  else
    return (int) (a + 0.5);
}

static double gauss(const double x, const double sigma)
{
  return 1.0 / sqrt(2 * M_PI) / sigma * exp(-x*x / 2.0 / sigma / sigma);
}

static double bs(const double x, const double t)
{
  return 1.0 / (exp(x / (KB * t)) - 1);
}

static ShortestVecs * get_shortest_vecs(PyArrayObject* shortest_vectors)
{
  int i, j, k;
  ShortestVecs * svecs;

  svecs = (ShortestVecs*) malloc(sizeof(ShortestVecs));
  for (i = 0; i < 4; i++) {
    svecs->d[i] = shortest_vectors->dimensions[i];
  }

  svecs->data = (double****) malloc(sizeof(double***) * svecs->d[0]);
  for (i = 0; i < svecs->d[0]; i++) {
    svecs->data[i] = (double***) malloc(sizeof(double**) * svecs->d[1]);
    for (j = 0; j < svecs->d[1]; j++) {
      svecs->data[i][j] = (double**) malloc(sizeof(double*) * svecs->d[2]);
      for (k = 0; k < svecs->d[2]; k++) {
  	svecs->data[i][j][k] =((double*)shortest_vectors->data +
			       svecs->d[1] * svecs->d[2] * svecs->d[3] * i +
			       svecs->d[2] * svecs->d[3] * j +
			       svecs->d[3] * k);

      }
    }
  }

  return svecs;
}

static void free_shortest_vecs(ShortestVecs * svecs)
{
  int i, j;
  
  for (i = 0; i < svecs->d[0]; i++) {
    for (j = 0; j < svecs->d[1]; j++) {
      free(svecs->data[i][j]);
      svecs->data[i][j] = NULL;
    }
    free(svecs->data[i]);
    svecs->data[i] = NULL;
  }
  free(svecs->data);
  svecs->data = NULL;
  free(svecs);
  svecs = NULL;
}

static Array2D * alloc_Array2D(const int index1, const int index2)
{
  int i;
  Array2D * array;
  
  array = (Array2D*) malloc(sizeof(Array2D));
  array->d1 = index1;
  array->d2 = index2;
  array->_data = (int*) malloc(sizeof(int) * index1 * index2);
  array->data = (int**) malloc(sizeof(int*) * index1);
  for (i = 0; i < index1; i++) {
    array->data[i] = array->_data + i * index2;
  }
  
  return array;
}

static void free_Array2D(Array2D * array)
{
  free(array->data);
  free(array->_data);
  free(array);
  array = NULL;
}

static DArray2D * alloc_DArray2D(const int index1, const int index2)
{
  int i;
  DArray2D * array;
  
  array = (DArray2D*) malloc(sizeof(DArray2D));
  array->d1 = index1;
  array->d2 = index2;
  array->_data = (double*) malloc(sizeof(double) * index1 * index2);
  array->data = (double**) malloc(sizeof(double*) * index1);
  for (i = 0; i < index1; i++) {
    array->data[i] = array->_data + i * index2;
  }
  
  return array;
}

static void free_DArray2D(DArray2D * array)
{
  free(array->data);
  free(array->_data);
  free(array);
  array = NULL;
}

static CArray2D * alloc_CArray2D(const int index1, const int index2)
{
  int i;
  CArray2D * array;
  
  array = (CArray2D*) malloc(sizeof(CArray2D));
  array->d1 = index1;
  array->d2 = index2;
  array->_data = (npy_cdouble*) malloc(sizeof(npy_cdouble) * index1 * index2);
  array->data = (npy_cdouble**) malloc(sizeof(npy_cdouble*) * index1);
  for (i = 0; i < index1; i++) {
    array->data[i] = array->_data + i * index2;
  }
  
  return array;
}

static void free_CArray2D(CArray2D * array)
{
  free(array->data);
  free(array->_data);
  free(array);
  array = NULL;
}

static Array1D * alloc_Array1D(const int index1)
{
  Array1D * array;
  
  array = (Array1D*) malloc(sizeof(Array1D));
  array->d1 = index1;
  array->data = (int*) malloc(sizeof(int) * index1);
  
  return array;
}

static void free_Array1D(Array1D * array)
{
  free(array->data);
  array->data = NULL;
  free(array);
  array = NULL;
}

static int phonopy_zheev(double *w,
			 lapack_complex_double *a,
			 const int n)
{
  lapack_int info;
  info = LAPACKE_zheev(LAPACK_ROW_MAJOR,'V', 'U',
		       (lapack_int)n, a, (lapack_int)n, w);
  return (int)info;
}
