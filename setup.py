from distutils.core import setup, Extension
import numpy
include_dirs_numpy = [numpy.get_include()]
include_dirs_phonopy = ['c/harmonic_h','c/kspclib_h'] + include_dirs_numpy

sources_phonopy = ['c/_phonopy.c',
                   'c/harmonic/dynmat.c',
                   'c/harmonic/derivative_dynmat.c',
                   'c/kspclib/kgrid.c',
                   'c/kspclib/tetrahedron_method.c']
extra_compile_args_phonopy = []
extra_link_args_phonopy = []

## Uncomment below if openmp multithreading is to be used.
# extra_compile_args_phonopy += ['-fopenmp',]
# extra_link_args_phonopy += ['-lgomp',]

## Uncomment below if lapack zheev is used instead of numpy.linalg.eigh.
sources_phonopy += ['c/harmonic/phonoc_array.c',
                    'c/harmonic/phonoc_math.c',
                    'c/harmonic/phonoc_utils.c',
                    'c/harmonic/lapack_wrapper.c']
## With lapacke installed in the system library path
# extra_link_args_phonopy += ['-llapacke', '-llapack', '-lblas']
## Without lapacke installed in the system library path
extra_link_args_phonopy += ['../lapack-3.5.0/liblapacke.a',]
include_dirs_phonopy += ['../lapack-3.5.0/lapacke/include',]

extension_phonopy = Extension(
    'phonopy._phonopy',
    extra_compile_args=extra_compile_args_phonopy,
    extra_link_args=extra_link_args_phonopy,
    include_dirs=include_dirs_phonopy,
    sources=sources_phonopy)

if __name__ == '__main__':
    extra_compile_args_spglib=[]
    extra_link_args_spglib=[]
else:
    extra_compile_args_spglib=['-fopenmp']
    extra_link_args_spglib=['-lgomp']

extension_spglib = Extension(
    'phonopy._spglib',
    include_dirs=['c/spglib_h'] + include_dirs_numpy,
    extra_compile_args=extra_compile_args_spglib,
    extra_link_args=extra_link_args_spglib,
    sources=['c/_spglib.c',
             'c/spglib/cell.c',
             'c/spglib/hall_symbol.c',
             'c/spglib/kgrid.c',
             'c/spglib/kpoint.c',
             'c/spglib/lattice.c',
             'c/spglib/mathfunc.c',
             'c/spglib/niggli.c',
             'c/spglib/pointgroup.c',
             'c/spglib/primitive.c',
             'c/spglib/refinement.c',
             'c/spglib/sitesym_database.c',
             'c/spglib/site_symmetry.c',
             'c/spglib/spacegroup.c',
             'c/spglib/spg_database.c',
             'c/spglib/spglib.c',
             'c/spglib/spin.c',
             'c/spglib/symmetry.c'])

packages_phonopy = ['phonopy',
                    'phonopy.cui',
                    'phonopy.gruneisen',
                    'phonopy.harmonic',
                    'phonopy.interface',
                    'phonopy.phonon',
                    'phonopy.qha',
                    'phonopy.structure']
scripts_phonopy = ['scripts/phonopy',
                   'scripts/phonopy-qha',
                   'scripts/phonopy-FHI-aims',
                   'scripts/bandplot',
                   'scripts/outcar-born',
                   'scripts/propplot',
                   'scripts/tdplot',
                   'scripts/dispmanager',
                   'scripts/gruneisen',
                   'scripts/pdosplot']

if __name__ == '__main__':
    setup(name='phonopy',
          version='1.9.5',
          description='This is the phonopy module.',
          author='Atsushi Togo',
          author_email='atz.togo@gmail.com',
          url='http://phonopy.sourceforge.net/',
          packages=packages_phonopy,
          scripts=scripts_phonopy,
          ext_modules=[extension_phonopy, extension_spglib])
