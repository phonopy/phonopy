#include <stdlib.h>
#include <lapacke.h>
#include "alloc_array.h"

ShortestVecs * get_shortest_vecs(const double* shortest_vectors,
				 const int* dimensions)
{
  int i, j, k;
  ShortestVecs * svecs;

  svecs = (ShortestVecs*) malloc(sizeof(ShortestVecs));
  for (i = 0; i < 4; i++) {
    svecs->d[i] = dimensions[i];
  }

  svecs->data = (double****) malloc(sizeof(double***) * svecs->d[0]);
  for (i = 0; i < svecs->d[0]; i++) {
    svecs->data[i] = (double***) malloc(sizeof(double**) * svecs->d[1]);
    for (j = 0; j < svecs->d[1]; j++) {
      svecs->data[i][j] = (double**) malloc(sizeof(double*) * svecs->d[2]);
      for (k = 0; k < svecs->d[2]; k++) {
  	svecs->data[i][j][k] =((double*)shortest_vectors +
			       svecs->d[1] * svecs->d[2] * svecs->d[3] * i +
			       svecs->d[2] * svecs->d[3] * j +
			       svecs->d[3] * k);

      }
    }
  }

  return svecs;
}

void free_shortest_vecs(ShortestVecs * svecs)
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

Array2D * alloc_Array2D(const int index1, const int index2)
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

void free_Array2D(Array2D * array)
{
  free(array->data);
  free(array->_data);
  free(array);
  array = NULL;
}

DArray2D * alloc_DArray2D(const int index1, const int index2)
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

void free_DArray2D(DArray2D * array)
{
  free(array->data);
  free(array->_data);
  free(array);
  array = NULL;
}

CArray2D * alloc_CArray2D(const int index1, const int index2)
{
  int i;
  CArray2D * array;
  
  array = (CArray2D*) malloc(sizeof(CArray2D));
  array->d1 = index1;
  array->d2 = index2;
  array->_data = (lapack_complex_double*)
    malloc(sizeof(lapack_complex_double) * index1 * index2);
  array->data = (lapack_complex_double**)
    malloc(sizeof(lapack_complex_double*) * index1);
  for (i = 0; i < index1; i++) {
    array->data[i] = array->_data + i * index2;
  }
  
  return array;
}

void free_CArray2D(CArray2D * array)
{
  free(array->data);
  free(array->_data);
  free(array);
  array = NULL;
}

Array1D * alloc_Array1D(const int index1)
{
  Array1D * array;
  
  array = (Array1D*) malloc(sizeof(Array1D));
  array->d1 = index1;
  array->data = (int*) malloc(sizeof(int) * index1);
  
  return array;
}

void free_Array1D(Array1D * array)
{
  free(array->data);
  array->data = NULL;
  free(array);
  array = NULL;
}

