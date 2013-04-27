#ifndef __alloc_array_H__
#define __alloc_array_H__

#include <lapacke.h>

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
  lapack_complex_double **data;
  lapack_complex_double *_data;
} CArray2D;

typedef struct {
  int d1;
  int *data;
} Array1D;

typedef struct {
  int d[4];
  double ****data;
} ShortestVecs;

ShortestVecs * get_shortest_vecs(const double* shortest_vectors,
				 const int* dimensions);
void free_shortest_vecs(ShortestVecs * svecs);
Array2D * alloc_Array2D(const int index1, const int index2);
void free_Array2D(Array2D * array);
DArray2D * alloc_DArray2D(const int index1, const int index2);
void free_DArray2D(DArray2D * array);
CArray2D * alloc_CArray2D(const int index1, const int index2);
void free_CArray2D(CArray2D * array);
Array1D * alloc_Array1D(const int index1);
void free_Array1D(Array1D * array);

#endif
