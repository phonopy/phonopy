from distutils.core import setup, Extension
import numpy
from setup import extension_spglib, extension_phonopy, packages_phonopy, scripts_phonopy

include_dirs_numpy = [numpy.get_include()]

sources = ['c/_phono3py.c',
           'c/harmonic/dynmat.c',
           'c/harmonic/lapack_wrapper.c',
           'c/harmonic/phonoc_array.c',
           'c/harmonic/phonoc_utils.c',
           'c/anharmonic/phonon3/fc3.c',
           'c/anharmonic/phonon3/frequency_shift.c',
           'c/anharmonic/phonon3/interaction.c',
           'c/anharmonic/phonon3/real_to_reciprocal.c',
           'c/anharmonic/phonon3/reciprocal_to_normal.c',
           'c/anharmonic/phonon3/imag_self_energy.c',
           'c/anharmonic/phonon3/imag_self_energy_with_g.c',
           'c/anharmonic/phonon3/collision_matrix.c',
           'c/anharmonic/other/isotope.c',
           'c/anharmonic/triplet/triplet.c',
           'c/anharmonic/triplet/triplet_kpoint.c',
           'c/spglib/mathfunc.c',
           'c/spglib/kpoint.c',
           'c/kspclib/kgrid.c',
           'c/kspclib/tetrahedron_method.c']
extra_link_args = ['-lgomp',
                   '-llapacke', # this is when lapacke is installed on system
                   '-llapack',
                   '-lblas']
extra_compile_args = ['-fopenmp',]
include_dirs = (['c/harmonic_h',
                 'c/anharmonic_h',
                 'c/spglib_h',
                 'c/kspclib_h'] +
                include_dirs_numpy)
##
## Uncomment and modify below if lapacke is prepared in a special location
#
# include_dirs += ['../lapack-3.5.0/lapacke/include']
# extra_link_args = ['-lgomp', '../lapack-3.5.0/liblapacke.a']

##
## This is for the test of libflame
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
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    sources=sources)

packages_phono3py = ['anharmonic',
                     'anharmonic.other',
                     'anharmonic.phonon3']
scripts_phono3py = ['scripts/phono3py',
                    'scripts/kaccum',
                    'scripts/gaccum']

########################
# _lapackepy extension #
########################
include_dirs_lapackepy = ['c/harmonic_h',] + include_dirs_numpy
sources_lapackepy = ['c/_lapackepy.c',
                     'c/harmonic/dynmat.c',
                     'c/harmonic/phonon.c',
                     'c/harmonic/phonoc_array.c',
                     'c/harmonic/phonoc_utils.c',
                     'c/harmonic/lapack_wrapper.c']
extension_lapackepy = Extension(
    'phonopy._lapackepy',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    sources=sources_lapackepy)

setup(name='phono3py',
      version='0.9.12.1',
      description='This is the phono3py module.',
      author='Atsushi Togo',
      author_email='atz.togo@gmail.com',
      url='http://phonopy.sourceforge.net/',
      packages=(packages_phonopy + packages_phono3py),
      scripts=(scripts_phonopy + scripts_phono3py),
      ext_modules=[extension_spglib,
                   extension_lapackepy,
                   extension_phonopy,
                   extension_phono3py])
