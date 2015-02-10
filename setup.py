from distutils.core import setup, Extension
#from setuptools import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]

extension = Extension('phonopy._phonopy',
                      # extra_compile_args=['-fopenmp'],
                      # extra_link_args=['-lgomp'],
                      include_dirs=['c/harmonic_h'] + include_dirs_numpy,
                      sources=['c/_phonopy.c',
                               'c/harmonic/dynmat.c',
                               'c/harmonic/derivative_dynmat.c'])

extension_spglib = Extension(
    'phonopy._spglib',
    include_dirs=['c/spglib_h'] + include_dirs_numpy,
    # extra_compile_args=['-fopenmp'],
    # extra_link_args=['-lgomp'],
    sources=['c/_spglib.c',
             'c/spglib/cell.c',
             'c/spglib/debug.c',
             'c/spglib/hall_symbol.c',
             'c/spglib/kpoint.c',
             'c/spglib/lattice.c',
             'c/spglib/mathfunc.c',
             'c/spglib/pointgroup.c',
             'c/spglib/primitive.c',
             'c/spglib/refinement.c',
             'c/spglib/sitesym_database.c',
             'c/spglib/site_symmetry.c',
             'c/spglib/spacegroup.c',
             'c/spglib/spg_database.c',
             'c/spglib/spglib.c',
             'c/spglib/spin.c',
             'c/spglib/symmetry.c',
             'c/spglib/tetrahedron_method.c'])


setup(name='phonopy',
      version='1.9.3',
      description='This is the phonopy module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=['phonopy',
                'phonopy.cui',
                'phonopy.gruneisen',
                'phonopy.harmonic',
                'phonopy.interface',
                'phonopy.phonon',
                'phonopy.qha',
                'phonopy.structure'],
      scripts=['scripts/phonopy',
               'scripts/phonopy-qha',
               'scripts/phonopy-FHI-aims',
               'scripts/bandplot',
               'scripts/outcar-born',
               'scripts/propplot',
               'scripts/tdplot',
               'scripts/dispmanager',
               'scripts/gruneisen',
               'scripts/pdosplot'],
      ext_modules=[extension, extension_spglib])
