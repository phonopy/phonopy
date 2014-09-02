from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_lapacke = ['../lapack-3.5.0/lapacke/include']
# include_dirs_libflame = ['../libflame-bin/include']
# libs_libflame = ['-L/opt/local/lib/gcc48/gcc/x86_64-apple-darwin13/4.8.2 -L/opt/local/lib/gcc48/gcc/x86_64-apple-darwin13/4.8.2/../../.. -lgfortran -lquadmath -lm']

extension = Extension(
    'anharmonic._phono3py',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h',
                   'c/spglib_h'] +
                  include_dirs_numpy +
                  # include_dirs_libflame +
                  include_dirs_lapacke),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['../lapack-3.5.0/liblapacke.a',
                     # '../libflame-bin/lib/libflame.a',
                     '-lgomp'],
                     # '/opt/local/lib/libopenblas.a',
                     # '-llapack',
                     # '-lblas'],
                     # ] + libs_libflame,
    sources=['c/_phono3py.c',
             'c/harmonic/dynmat.c',
             # 'c/anharmonic/flame_wrapper.c',
             'c/anharmonic/lapack_wrapper.c',
             'c/anharmonic/phonoc_array.c',
             'c/anharmonic/phonoc_math.c',
             'c/anharmonic/phonoc_utils.c',
             'c/anharmonic/phonon3/fc3.c',
             'c/anharmonic/phonon3/frequency_shift.c',
             'c/anharmonic/phonon3/interaction.c',
             'c/anharmonic/phonon3/real_to_reciprocal.c',
             'c/anharmonic/phonon3/reciprocal_to_normal.c',
             'c/anharmonic/phonon3/imag_self_energy.c',
             'c/anharmonic/phonon3/imag_self_energy_with_g.c',
             'c/anharmonic/phonon3/collision_matrix.c',
             'c/anharmonic/other/isotope.c',
             'c/spglib/debug.c',
             'c/spglib/kpoint.c',
             'c/spglib/mathfunc.c',
             'c/spglib/tetrahedron_method.c'])

setup(name='phono3py',
      version='0.9.0',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic',
                'anharmonic.other',
                'anharmonic.phonon3'],
      scripts=['scripts/phono3py'],
      ext_modules=[extension])
