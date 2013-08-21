#ifndef __phonoc_array_H__
#define __phonoc_array_H__

#include <Python.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#define MAX_NUM_DIM 5

/* It is assumed that number of dimensions is known for each array. */
typedef struct {
  int dims[MAX_NUM_DIM];
  int *data;
} Iarray;

typedef struct {
  int dims[MAX_NUM_DIM];
  double *data;
} Darray;

typedef struct {
  int dims[MAX_NUM_DIM];
  lapack_complex_double *data;
} Carray;

Iarray* convert_to_iarray(const PyArrayObject* npyary);
Darray* convert_to_darray(const PyArrayObject* npyary);
Carray* convert_to_carray(const PyArrayObject* npyary);

#endif
