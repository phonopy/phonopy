"""Setup script of phonopy."""
import os
import pathlib
import shutil
import subprocess

import numpy
import setuptools

if (
    "PHONOPY_USE_OPENMP" in os.environ
    and os.environ["PHONOPY_USE_OPENMP"].lower() == "true"
):
    use_openmp = True
else:
    use_openmp = False


def _run_cmake(build_dir):
    build_dir.mkdir()
    args = [
        "cmake",
        "-S",
        ".",
        "-B",
        "_build",
        "-DCMAKE_INSTALL_PREFIX=.",
        "-DPHONOPY=on",
    ]
    # if "CONDA_PREFIX" in os.environ:
    #     args.append("-DUSE_CONDA_PATH=on")
    # if "CC" in os.environ:
    #     args.append(f'-DCMAKE_C_COMPILER={os.environ["CC"]}')

    cmake_output = subprocess.check_output(args)
    print(cmake_output.decode("utf-8"))
    subprocess.check_call(["cmake", "--build", "_build", "-v"])
    return cmake_output


def _clean_cmake(build_dir):
    if build_dir.exists():
        shutil.rmtree(build_dir)


def _get_extensions(build_dir):
    # Initialization of parameters
    define_macros = []
    extra_link_args = []
    extra_compile_args = []
    extra_objects = []
    include_dirs = []

    # Libraray search
    found_extra_link_args = []
    found_extra_compile_args = []
    if not use_openmp or not shutil.which("cmake") or build_dir.exists():
        sources = [
            "c/_phonopy.c",
            "c/phonopy.c",
            "c/dynmat.c",
            "c/derivative_dynmat.c",
            "c/rgrid.c",
            "c/tetrahedron_method.c",
        ]
    else:
        sources = ["c/_phonopy.c"]
        cmake_output = _run_cmake(build_dir)
        found_flags = {}
        found_libs = {}
        for line in cmake_output.decode("utf-8").split("\n"):
            for key in ["OpenMP"]:
                if f"{key} libs" in line and len(line.split()) > 3:
                    found_libs[key] = line.split()[3].split(";")
                if f"{key} flags" in line and len(line.split()) > 3:
                    found_flags[key] = line.split()[3].split(";")
                    # if key == "OpenMP":
                    #     define_macros.append(("_OPENMP", None))
        for key, value in found_libs.items():
            found_extra_link_args += value
        for key, value in found_flags.items():
            found_extra_compile_args += value
        print("=============================================")
        print("extra_compile_args: ", found_extra_compile_args)
        print("extra_link_args: ", found_extra_link_args)
        print("define_macros: ", define_macros)
        print("=============================================")
        print()

    # Build ext_modules
    extensions = []
    extra_link_args += found_extra_link_args
    extra_compile_args += found_extra_compile_args
    define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
    include_dirs += ["c", numpy.get_include()]

    libphpy = list((pathlib.Path.cwd() / "_build").glob("*phpy.*"))
    if libphpy:
        print("=============================================")
        print(f"Phonopy library: {libphpy[0]}")
        print("=============================================")
        extra_objects += [str(libphpy[0])]

    extensions.append(
        setuptools.Extension(
            "phonopy._phonopy",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            extra_objects=extra_objects,
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
    )

    return extensions


def _get_version() -> str:
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
    return version


def main(build_dir):
    """Run setuptools."""
    version = _get_version()

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

    setuptools.setup(
        name="phonopy",
        version=version,
        description="This is the phonopy module.",
        author="Atsushi Togo",
        author_email="atz.togo@gmail.com",
        url="https://phonopy.github.io/phonopy/",
        packages=packages_phonopy,
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.15.0",
            "PyYAML",
            "matplotlib>=2.2.2",
            "h5py",
            "spglib<2.0",
        ],
        extras_require={"cp2k": ["cp2k-input-tools"]},
        provides=["phonopy"],
        scripts=scripts_phonopy,
        ext_modules=_get_extensions(build_dir),
    )

    _clean_cmake(build_dir)


if __name__ == "__main__":
    build_dir = pathlib.Path.cwd() / "_build"
    main(build_dir)
