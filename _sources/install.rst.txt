.. _install:

Installation
=============

**From phonopy v2.7.0, spglib has to be installed separately.**

.. contents::
   :depth: 3
   :local:

.. _install_conda:

Installation via conda
----------------------

Conda is an open source package management system. Once the conda
system is set-up (see `details about conda setting up
<https://conda.io/docs/user-guide/install/index.html>`_), the installation
of phonopy is super easy for any of Linux, MacOSX, and Windows.
To install::

   % conda install -c conda-forge phonopy

This phonopy's conda package is prepared and maintained by
Paweł T. Jochym at conda-forge channel (please be aware that this is
not a trivial job).

Minimum steps to install and use phonopy via conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following procedure, conda's environment (see `details at conda
web site
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_)
is used not to interfere existing environment (mainly python
environment).

::

   % conda create -n phonopy -c conda-forge python=3
   % conda activate phonopy
   % conda install -c conda-forge phonopy

To exit from this conda's environment::

   % conda deactivate

To use this phonopy, entering this environment is necessary like below.

::

   % conda activate phonopy
   (phonopy) % phonopy
     _ __ | |__   ___  _ __   ___   _ __  _   _
    | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
    | |_) | | | | (_) | | | | (_) || |_) | |_| |
    | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
    |_|                            |_|    |___/
                                          2.7.0

   Python version 3.7.6
   Spglib version 1.14.1


   Supercell matrix (DIM or --dim) was not explicitly specified.
   By this reason, phonopy_yaml mode was invoked.
   But "phonopy_params.yaml", "phonopy_disp.yaml" and "phonopy.yaml" could not be found.
     ___ _ __ _ __ ___  _ __
    / _ \ '__| '__/ _ \| '__|
   |  __/ |  | | | (_) | |
    \___|_|  |_|  \___/|_|

.. _install_from_source:

Using HDF5 on NFS mounted file system
-------------------------------------

Recent hdf5 versions just as installed may not work on NFS mounted
file systems. In this case, setting the following environment variable
may solve the problem::

   export HDF5_USE_FILE_LOCKING=FALSE


Installation from source code
-----------------------------

System requirement
~~~~~~~~~~~~~~~~~~

The procedure to setup phonopy is explained in this section. It is
supposed that phonopy is installed on the recent linux distribution
like Ubuntu or Fedora with Python version 2.7 or later. Python version
3.4 or later is expected to work. Mac OS X users may use conda
(conda-forge channel) packages.  Windows users should use conda
(conda-forge channel) packages as well.

Prepare the following Python libraries:

* Python and its header files
* numpy
* matplotlib
* python-yaml (pyyaml)
* python-h5py (h5py)

For the CP2K interface, the following package will be needed to install:

* cp2k-input-tools


Installing required packages by conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The python libraries can be installed using conda. Conda packages are
distributed in binary. Minimum setup of conda envrironment is done by
miniconda, which is downloaded at https://conda.io/miniconda.html. It
is strongly recommended to create conda's virtual environment by
``conda create -n <venvname>`` as written above. The installation of
necessary libraries is done as follows::

   % conda install -c conda-forge numpy scipy h5py pyyaml matplotlib-base spglib

A libblas library installed can be chosen among ``[openblas, mkl, blis,
netlib]``. If specific one is expected, it is installed by (e.g. ``openblas``)

::

   % conda install -c conda-forge "libblas=*=*openblas"

If you need a compiler, for usual 64-bit linux system::

   % conda install -c conda-forge gcc_linux-64

For macOS::

   % conda install -c conda-forge clang_osx-64

.. _install_setup_py:

Building using setup.py
~~~~~~~~~~~~~~~~~~~~~~~~

If package installation is not possible or you want to compile with
special compiler or special options, phonopy is built using
setup.py. In this case, manual modification of ``setup.py`` may be
needed.

1. Get the source code from github

   ::

      % git clone https://github.com/phonopy/phonopy.git
      % cd phonopy
      % git checkout master

2. Run ``setup.py`` script

   ::

      % python setup.py build
      % pip install -e .

.. _install_trouble_shooting:

Trouble shooting
-----------------

Remove previous phonopy installations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes previous installations of phonopy prevent from loading newly
installed phonopy. In this case, it is recommended to uninstall all
the older phonopy packages by

1. Running ``pip uninstall phonopy`` as many times as no phonopy
   packages will be found. Error message may be shown, but don't mind
   it. Similarly do ``conda uninstall phonopy``.

2. There may still exist litter of phonopy packages. So it is also
   recommend to remove them if it is found, e.g.::

     % rm -fr ~/.local/lib/python*/site-packages/phonopy*

Set correct environment variables ``PATH`` and ``PYTHONPATH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using conda environment, this information is not applicable.

In phonopy, ``PATH`` and ``PYTHONPATH`` play important roles. Of
course the information about them can be easily found in internet
(e.g. https://en.wikipedia.org/wiki/PATH_(variable)), so you really
have to find information by yourself and read them. Even if you can't
understand them, first you must ask to your colleagues or people
before sending this unnecessary question (as a researcher using
computer simulation) to the mailing list.

The problem appears when phonopy execution and library paths are set
multiple times in those environment variable. It is easy to check
current environment variables by::

   % echo $PATH

::

   % echo $PYTHONPATH

When multiple different phonopy paths are found, remove all except for
what you really need. Then logout from the current shell (terminal)
and open new shell (terminal) to confirm that the modification is activated.

Error in plotting on display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``texlive-fonts-recommended`` and ``dviping`` packages may be required
to install on your system, if you see something like the following
messages when ploting::

   ! I can't find file `pncr7t'.

or::

   ! LaTeX Error: File `type1cm.sty' not found.


Missing Intel libraries when building from source using icc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LDSHARED="icc -shared"`` may be of help. See this github issues,
https://github.com/phonopy/phonopy/issues/123.
