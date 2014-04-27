from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_lapacke = ['../lapack-3.5.0/lapacke/include']

extension_phono4py = Extension(
    'anharmonic._phono4py',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h'] +
                  include_dirs_numpy +
                  include_dirs_lapacke),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp',
                     '../lapack-3.5.0/liblapacke.a'],
                     #'-llapack',
                     #'-lblas'],
    sources=['c/_phono4py.c',
             'c/harmonic/dynmat.c',
             'c/harmonic/lapack_wrapper.c',
             'c/anharmonic/phonoc_array.c',
             'c/anharmonic/phonoc_math.c',
             'c/anharmonic/phonoc_utils.c',
             'c/anharmonic/phonon3/fc3.c',
             'c/anharmonic/phonon4/fc4.c',
             'c/anharmonic/phonon4/real_to_reciprocal.c',
             'c/anharmonic/phonon4/frequency_shift.c'])

extension_forcefit = Extension(
    'anharmonic._forcefit',
    include_dirs=(['c/harmonic_h'] +
                  include_dirs_numpy +
                  include_dirs_lapacke),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp',
                     '../lapack-3.5.0/liblapacke.a'],
                     #'-llapack',
                     #'-lblas'],
    sources=['c/_forcefit.c',
             'c/harmonic/lapack_wrapper.c'])

setup(name='phono4py',
      version='0.8.0',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic',
                'anharmonic.force_fit',
                'anharmonic.phonon4'],
      scripts=['scripts/force-fit',
               'scripts/phono4py'],
      ext_modules=[extension_phono4py,
                   extension_forcefit])
