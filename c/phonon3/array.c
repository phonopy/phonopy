#include <Python.h>
#include <numpy/arrayobject.h>
#include <lapacke.h>
#include "array.h"

Iarray* convert_to_iarray(const PyArrayObject* npyary)
{
  int i;
  Iarray *ary;

  ary = (Iarray*) malloc(sizeof(Iarray));
  for (i = 0; i < npyary->nd; i++) {
    ary->dims[i] = npyary->dimensions[i];
  }
  ary->data = (int*)npyary->data;
  return ary;
}

Darray* convert_to_darray(const PyArrayObject* npyary)
{
  int i;
  Darray *ary;

  ary = (Darray*) malloc(sizeof(Darray));
  for (i = 0; i < npyary->nd; i++) {
    ary->dims[i] = npyary->dimensions[i];
  }
  ary->data = (double*)npyary->data;
  return ary;
}

Carray* convert_to_carray(const PyArrayObject* npyary)
{
  int i;
  Carray *ary;

  ary = (Carray*) malloc(sizeof(Carray));
  for (i = 0; i < npyary->nd; i++) {
    ary->dims[i] = npyary->dimensions[i];
  }
  ary->data = (lapack_complex_double*)npyary->data;
  return ary;
}
