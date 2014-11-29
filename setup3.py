from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_lapacke = ['../lapacke/include']

use_libflame = False

sources = ['c/_phono3py.c',
           'c/harmonic/dynmat.c',
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
           'c/spglib/tetrahedron_method.c']
extra_link_args=['-lgomp',
                 '../lapacke/liblapacke.a',
                 '-llapack',
                 '-lblas']
include_dirs = (['c/harmonic_h',
                 'c/anharmonic_h',
                 'c/spglib_h'] +
                include_dirs_numpy +
                include_dirs_lapacke)

if use_libflame:
    sources.append('c/anharmonic/flame_wrapper.c')
    extra_link_args.append('../libflame-bin/lib/libflame.a')
    include_dirs_libflame = ['../libflame-bin/include']
    include_dirs += include_dirs_libflame
    
extension = Extension(
    'anharmonic._phono3py',
    include_dirs=include_dirs,
    extra_compile_args=['-fopenmp'],
    extra_link_args=extra_link_args,
    sources=sources)

setup(name='phono3py',
      version='0.9.3',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic',
                'anharmonic.other',
                'anharmonic.phonon3'],
      scripts=['scripts/phono3py'],
      ext_modules=[extension])
