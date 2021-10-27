"""Setup script of phonopy."""
import os
import sysconfig

import numpy

with_openmp = False

try:
    from setuptools import Extension, setup

    use_setuptools = True
    print("setuptools is used.")
except ImportError:
    from distutils.core import Extension, setup

    use_setuptools = False
    print("distutils is used.")

include_dirs_numpy = [numpy.get_include()]

cc = None
if "CC" in os.environ:
    if "clang" in os.environ["CC"]:
        cc = "clang"
    if "gcc" in os.environ["CC"]:
        cc = "gcc"

# Workaround Python issue 21121
config_var = sysconfig.get_config_var("CFLAGS")
if config_var is not None and "-Werror=declaration-after-statement" in config_var:
    os.environ["CFLAGS"] = config_var.replace("-Werror=declaration-after-statement", "")

######################
# _phonopy extension #
######################
include_dirs_phonopy = [
    "c",
] + include_dirs_numpy
sources_phonopy = [
    "c/_phonopy.c",
    "c/phonopy.c",
    "c/dynmat.c",
    "c/derivative_dynmat.c",
    "c/rgrid.c",
    "c/tetrahedron_method.c",
]

if with_openmp:
    extra_compile_args_phonopy = [
        "-fopenmp",
    ]
    if cc == "gcc":
        extra_link_args_phonopy = [
            "-lgomp",
        ]
    elif cc == "clang":
        extra_link_args_phonopy = ["-lomp"]
    else:
        extra_link_args_phonopy = [
            "-lgomp",
        ]
else:
    extra_compile_args_phonopy = []
    extra_link_args_phonopy = []

extension_phonopy = Extension(
    "phonopy._phonopy",
    extra_compile_args=extra_compile_args_phonopy,
    extra_link_args=extra_link_args_phonopy,
    include_dirs=include_dirs_phonopy,
    sources=sources_phonopy,
)


ext_modules_phonopy = [
    extension_phonopy,
]
packages_phonopy = [
    "phonopy",
    "phonopy.cui",
    "phonopy.gruneisen",
    "phonopy.harmonic",
    "phonopy.interface",
    "phonopy.phonon",
    "phonopy.qha",
    "phonopy.spectrum",
    "phonopy.structure",
    "phonopy.unfolding",
]
scripts_phonopy = [
    "scripts/phonopy",
    "scripts/phonopy-load",
    "scripts/phonopy-qha",
    "scripts/phonopy-bandplot",
    "scripts/phonopy-vasp-born",
    "scripts/phonopy-vasp-efe",
    "scripts/phonopy-crystal-born",
    "scripts/phonopy-calc-convert",
    "scripts/phonopy-propplot",
    "scripts/phonopy-tdplot",
    "scripts/phonopy-gruneisen",
    "scripts/phonopy-gruneisenplot",
    "scripts/phonopy-pdosplot",
]

if __name__ == "__main__":
    version_nums = []
    with open("phonopy/version.py") as f:
        for line in f:
            if "__version__" in line:
                for i, num in enumerate(line.split()[2].strip('"').split(".")):
                    version_nums.append(int(num))
                break

    # # To deploy to pypi/conda by travis-CI
    if os.path.isfile("__nanoversion__.txt"):
        nanoversion = 0
        with open("__nanoversion__.txt") as nv:
            try:
                for line in nv:
                    nanoversion = int(line.strip())
                    break
            except ValueError:
                pass
        if nanoversion != 0:
            version_nums.append(nanoversion)

    if len(version_nums) < 3:
        print("Failed to get version number in setup.py.")
        raise

    version = ".".join(["%s" % n for n in version_nums[:3]])
    if len(version_nums) > 3:
        version += "-%d" % version_nums[3]
    print(version)

    if use_setuptools:
        setup(
            name="phonopy",
            version=version,
            description="This is the phonopy module.",
            author="Atsushi Togo",
            author_email="atz.togo@gmail.com",
            url="https://phonopy.github.io/phonopy/",
            packages=packages_phonopy,
            python_requires=">=3.6",
            install_requires=[
                "numpy>=1.11.0",
                "PyYAML",
                "matplotlib>2.0.0",
                "h5py",
                "spglib",
            ],
            extras_require={"cp2k": ["cp2k-input-tools"]},
            provides=["phonopy"],
            scripts=scripts_phonopy,
            ext_modules=ext_modules_phonopy,
        )
    else:
        setup(
            name="phonopy",
            version=version,
            description="This is the phonopy module.",
            author="Atsushi Togo",
            author_email="atz.togo@gmail.com",
            url="https://phonopy.github.io/phonopy/",
            packages=packages_phonopy,
            requires=["numpy", "PyYAML", "matplotlib", "h5py", "spglib"],
            provides=["phonopy"],
            scripts=scripts_phonopy,
            ext_modules=ext_modules_phonopy,
        )
