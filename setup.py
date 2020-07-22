import os
import sys
import numpy
import sysconfig

with_openmp = False

try:
    from setuptools import setup, Extension
    use_setuptools = True
    print("setuptools is used.")
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print("distutils is used.")

try:
    from setuptools_scm import get_version
except ImportError:
    git_num = None

if 'setuptools_scm' in sys.modules.keys():
    try:
        git_ver = get_version()
        git_num = int(git_ver.split('.')[3].split('+')[0].replace("dev", ""))
    except:
        git_num = None

include_dirs_numpy = [numpy.get_include()]

cc = None
if 'CC' in os.environ:
    if 'clang' in os.environ['CC']:
        cc = 'clang'
    if 'gcc' in os.environ['CC']:
        cc = 'gcc'

# Workaround Python issue 21121
config_var = sysconfig.get_config_var("CFLAGS")
if (config_var is not None and
    "-Werror=declaration-after-statement" in config_var):
    os.environ['CFLAGS'] = config_var.replace(
        "-Werror=declaration-after-statement", "")

######################
# _phonopy extension #
######################
include_dirs_phonopy = (['c/harmonic_h', 'c/kspclib_h', 'c/spglib_h']
                        + include_dirs_numpy)
sources_phonopy = ['c/_phonopy.c',
                   'c/harmonic/dynmat.c',
                   'c/harmonic/derivative_dynmat.c',
                   'c/spglib/kgrid.c',
                   'c/kspclib/tetrahedron_method.c']

if with_openmp:
    extra_compile_args_phonopy = ['-fopenmp', ]
    if cc == 'gcc':
        extra_link_args_phonopy = ['-lgomp', ]
    elif cc == 'clang':
        extra_link_args_phonopy = ['-lomp']
    else:
        extra_link_args_phonopy = ['-lgomp', ]
else:
    extra_compile_args_phonopy = []
    extra_link_args_phonopy = []

extension_phonopy = Extension(
    'phonopy._phonopy',
    extra_compile_args=extra_compile_args_phonopy,
    extra_link_args=extra_link_args_phonopy,
    include_dirs=include_dirs_phonopy,
    sources=sources_phonopy)


ext_modules_phonopy = [extension_phonopy, ]
packages_phonopy = ['phonopy',
                    'phonopy.cui',
                    'phonopy.gruneisen',
                    'phonopy.harmonic',
                    'phonopy.interface',
                    'phonopy.phonon',
                    'phonopy.qha',
                    'phonopy.spectrum',
                    'phonopy.structure',
                    'phonopy.unfolding']
scripts_phonopy = ['scripts/phonopy',
                   'scripts/phonopy-load',
                   'scripts/phonopy-qha',
                   'scripts/phonopy-bandplot',
                   'scripts/phonopy-vasp-born',
                   'scripts/phonopy-vasp-efe',
                   'scripts/phonopy-crystal-born',
                   'scripts/phonopy-propplot',
                   'scripts/phonopy-tdplot',
                   'scripts/phonopy-gruneisen',
                   'scripts/phonopy-gruneisenplot',
                   'scripts/phonopy-pdosplot']

if __name__ == '__main__':

    version_nums = [None, None, None]
    with open("phonopy/version.py") as f:
        for line in f:
            if "__version__" in line:
                for i, num in enumerate(
                        line.split()[2].strip('\"').split('.')):
                    version_nums[i] = num
                break

    # # To deploy to pypi/conda by travis-CI
    if os.path.isfile("__nanoversion__.txt"):
        nanoversion = 0
        with open('__nanoversion__.txt') as nv:
            try:
                for line in nv:
                    nanoversion = int(line.strip())
                    break
            except ValueError:
                pass
        if nanoversion != 0:
            version_nums.append(nanoversion)
    elif git_num:
        version_nums.append(git_num)

    if None in version_nums:
        print("Failed to get version number in setup.py.")
        raise

    version = ".".join(["%s" % n for n in version_nums[:3]])
    if len(version_nums) > 3:
        version += "-%d" % version_nums[3]

    if use_setuptools:
        setup(name='phonopy',
              version=version,
              description='This is the phonopy module.',
              author='Atsushi Togo',
              author_email='atz.togo@gmail.com',
              url='http://phonopy.github.io/phonopy/',
              packages=packages_phonopy,
              install_requires=['numpy', 'PyYAML', 'matplotlib', 'h5py', 'spglib'],
              extras_require={'cp2k': ['cp2k-input-tools']},
              provides=['phonopy'],
              scripts=scripts_phonopy,
              ext_modules=ext_modules_phonopy)
    else:
        setup(name='phonopy',
              version=version,
              description='This is the phonopy module.',
              author='Atsushi Togo',
              author_email='atz.togo@gmail.com',
              url='http://phonopy.github.io/phonopy/',
              packages=packages_phonopy,
              requires=['numpy', 'PyYAML', 'matplotlib', 'h5py', 'spglib'],
              provides=['phonopy'],
              scripts=scripts_phonopy,
              ext_modules=ext_modules_phonopy)
