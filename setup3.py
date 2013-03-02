from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]

extension = Extension('anharmonic._phono3py',
                      include_dirs=['c'] + include_dirs_numpy,
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-lgomp', '-llapacke'],
                      sources=['c/_phono3py.c'])

setup(name='phono3py',
      version='0.2.0',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['anharmonic'],
      scripts=['scripts/phono3py'],
      ext_modules=[extension])
