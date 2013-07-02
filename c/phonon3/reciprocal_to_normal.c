/*---------------------------------------------------------*/
/* Transform fc3 in reciprocal space to normal coordinates */
/*---------------------------------------------------------*/

static void reciprocal_to_normal(double *fc3_normal_squared,
				 const lapack_complex_double *fc3_reciprocal,
				 const Darray *freqs,
				 const Carray *eigvecs,
				 const double *masses);
static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b);

static void reciprocal_to_normal(double *fc3_normal_squared,
				 const lapack_complex_double *fc3_reciprocal,
				 const Darray *freqs,
				 const Carray *eigvecs,
				 const double *masses)
{
  
}

static lapack_complex_double
prod(const lapack_complex_double a, const lapack_complex_double b)
{
  lapack_complex_double c;
  c = lapack_make_complex_double
    (lapack_complex_double_real(a) * lapack_complex_double_real(b) -
     lapack_complex_double_imag(a) * lapack_complex_double_imag(b),
     lapack_complex_double_imag(a) * lapack_complex_double_real(b) +
     lapack_complex_double_real(a) * lapack_complex_double_imag(b));
  return c;
}
