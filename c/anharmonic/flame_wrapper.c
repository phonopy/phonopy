#include "flame_wrapper.h"
#include "FLAME.h"

int flame_Hevd(double *matrix,
	       double *eigvals,
	       const int size)
{
  FLA_Obj A, l;
  
  FLA_Init();
  FLA_Obj_create_without_buffer(FLA_DOUBLE, size, size, &A);
  FLA_Obj_attach_buffer(matrix, 0, 0, &A);
  FLA_Obj_create_without_buffer(FLA_DOUBLE, 1, size, &l);
  FLA_Obj_attach_buffer(eigvals, 0, 0, &l);
  printf("start Hevd\n");
  FLA_Hevd(FLA_EVD_WITH_VECTORS, FLA_LOWER_TRIANGULAR, A, l);
  printf("end Hevd\n");
  FLA_Obj_free_without_buffer(&A);
  FLA_Obj_free_without_buffer(&l);
  FLA_Finalize();
  
  return 0;
}
