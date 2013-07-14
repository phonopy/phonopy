from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_lapacke = ['../lapacke/include']

extension = Extension(
    'anharmonic._phono3py',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h',
                   'c/phonon3_h'] +
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
             'c/phonon3/interaction.c',
             'c/phonon3/real_to_reciprocal.c',
             'c/phonon3/reciprocal_to_normal.c',
             'c/phonon3/phonoc_array.c',
             'c/phonon3/phonoc_math.c',
             'c/phonon3/imag_self_energy.c'])

setup(name='phono3py',
      version='0.4.1',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic'],
      scripts=['scripts/phono3py'],
      ext_modules=[extension])
