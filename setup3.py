from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_lapacke = ['../lapacke/include']

extension = Extension(
    'anharmonic._phono3py',
    include_dirs=(['c/harmonic_h',
                   'c/anharmonic_h'] +
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
             'c/anharmonic/phonoc_array.c',
             'c/anharmonic/phonoc_math.c',
             'c/anharmonic/phonoc_utils.c',
             'c/anharmonic/phonon3/interaction.c',
             'c/anharmonic/phonon3/real_to_reciprocal.c',
             'c/anharmonic/phonon3/reciprocal_to_normal.c',
             'c/anharmonic/phonon3/imag_self_energy.c',
             'c/anharmonic/phonon4/real_to_reciprocal.c',
             'c/anharmonic/phonon4/frequency_shift.c'])

setup(name='phono3py',
      version='0.5.1',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic',
                'anharmonic.force_fit',
                'anharmonic.phonon3',
                'anharmonic.phonon4'],
      scripts=['scripts/force-fit',
               'scripts/phono3py',
               'scripts/phono4py'],
      ext_modules=[extension])
