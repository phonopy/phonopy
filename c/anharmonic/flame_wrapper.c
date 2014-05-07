#include "flame_wrapper.h"
#include "FLAME.h"

int flame_Hevd(void)
{
  double* buffer;
  int     m;
  FLA_Obj A, l;
  
  FLA_Init();
  FLA_Obj_create(FLA_DOUBLE, m, m, 0, 0, &A);
  FLA_Obj_create(FLA_DOUBLE, m, 1, 0, 0, &l);
  FLA_Copy_buffer_to_object(FLA_TRANSPOSE, m, m, buffer, 0, 0, 0, 0, A);
  FLA_Hevd(FLA_EVD_WITH_VECTORS, FLA_LOWER_TRIANGULAR, A, l);
  FLA_Obj_free(&A);
  FLA_Obj_free(&l);
  FLA_Finalize();
  return 0;
}
