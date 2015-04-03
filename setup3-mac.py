from distutils.core import setup, Extension
import numpy
from setup import extension_spglib, extension_phonopy, packages_phonopy, scripts_phonopy

include_dirs_numpy = [numpy.get_include()]

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
                 '-llapacke', # this is when lapacke is installed on system
                 '-llapack',
                 '-lblas']
include_dirs = (['c/harmonic_h',
                 'c/anharmonic_h',
                 'c/spglib_h'] +
                include_dirs_numpy)
##
## If lapacke is prepared manually,
##
include_dirs += ['../lapack-3.5.0/lapacke/include']
extra_link_args=['-lgomp',
                 '../lapack-3.5.0/liblapacke.a']

##
## Libflame test
##
# use_libflame = False
# if use_libflame:
#     sources.append('c/anharmonic/flame_wrapper.c')
#     extra_link_args.append('../libflame-bin/lib/libflame.a')
#     include_dirs_libflame = ['../libflame-bin/include']
#     include_dirs += include_dirs_libflame
    
extension_phono3py = Extension(
    'anharmonic._phono3py',
    include_dirs=include_dirs,
    extra_compile_args=['-fopenmp'],
    extra_link_args=extra_link_args,
    sources=sources)

packages_phono3py = ['anharmonic',
                     'anharmonic.other',
                     'anharmonic.phonon3']
scripts_phono3py = ['scripts/phono3py',
                    'scripts/kaccum']

setup(name='phono3py',
      version='0.9.9',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=(packages_phonopy + packages_phono3py),
      scripts=(scripts_phonopy + scripts_phono3py),
      ext_modules=[extension_spglib, extension_phonopy, extension_phono3py])
