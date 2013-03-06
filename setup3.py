from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_lapacke = ['../lapacke/include']

extension = Extension(
    'anharmonic._phono3py',
    include_dirs=(['c/harmonic_include',
                   'c/anharmonic_include'] +
                  include_dirs_numpy +
                  include_dirs_lapacke),
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp',
                     '../lapacke/liblapacke.a',
                     '-llapack',
                     '-lblas'],
    sources=['c/_phono3py.c',
             'c/harmonic/dynmat.c',
             'c/harmonic/lapack_wrapper.c',
             'c/anharmonic/interaction_strength.c',
             'c/anharmonic/alloc_array.c'])

setup(name='phono3py',
      version='0.2.0',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic'],
      scripts=['scripts/phono3py'],
      ext_modules=[extension])
