#include "flame_wrapper.h"
#include "FLAME.h"
#include "math.h"

int flame_Hevd(double *matrix,
	       double *eigvals,
	       const int size,
	       const double cutoff)
{
  FLA_Obj A, B, C, l;
  /* FLA_Obj C; */
  int i;
  
  FLA_Init();
  FLA_Obj_create_without_buffer(FLA_DOUBLE, size, size, &A);
  FLA_Obj_attach_buffer(matrix, 0, 0, &A);
  
  FLA_Obj_create_without_buffer(FLA_DOUBLE, size, 1, &l);
  FLA_Obj_attach_buffer(eigvals, 0, 0, &l);

  /* Eigensolver */
  FLA_Obj_create_copy_of(FLA_NO_TRANSPOSE, A, &B);
  FLA_Hevd(FLA_EVD_WITH_VECTORS, FLA_LOWER_TRIANGULAR, B, l);

  /* SVD */
  /* FLA_Obj_create(FLA_DOUBLE, size, size, 0, 0, &B); */
  /* use U */
  /* FLA_Svd(FLA_SVD_VECTORS_ALL, FLA_SVD_VECTORS_NONE, A, l, B, C); */
  /* use V */
  /* FLA_Svd(FLA_SVD_VECTORS_NONE, FLA_SVD_VECTORS_ALL, A, l, C, B); */
  
  FLA_Obj_free_without_buffer(&l);
  
  for (i = 0; i < size; i++) {
    if (eigvals[i] < cutoff) {
      eigvals[i] = 0;
    } else {
      eigvals[i] = 1.0 / sqrt(eigvals[i]);
    }
  }
  
  FLA_Obj_create_without_buffer(FLA_DOUBLE, size, 1, &l);
  FLA_Obj_attach_buffer(eigvals, 0, 0, &l);
  
  FLA_Apply_diag_matrix(FLA_RIGHT, FLA_NO_CONJUGATE, l, B);
  FLA_Syrk(FLA_LOWER_TRIANGULAR, FLA_NO_TRANSPOSE, FLA_ONE, B, FLA_ZERO, A);
  FLA_Symmetrize(FLA_LOWER_TRIANGULAR, A);
  
  FLA_Obj_free_without_buffer(&A);
  FLA_Obj_free_without_buffer(&l);
  FLA_Obj_free(&B);
  
  FLA_Finalize();
  
  return 0;
}
