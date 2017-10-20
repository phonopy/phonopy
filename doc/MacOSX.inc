.. _install_MacOSX:

Using phonopy on Mac OS X
--------------------------

This is one way to install phonopy on Mac OS X. This procedure was
tested with MacPorts version 2.4.1 and Homebrew 1.3.6. In the
following case, gcc-7 is used as the default C-compiler.

1) Install necessary python libraries by conda (see the detail about
   conda, e.g., https://conda.io/miniconda.html)::

   % conda install numpy scipy h5py pyyaml matplotlib

2) Install gcc

   Using Homebrew. This is normally easier than using MacPorts. See
   how to install homebrew https://brew.sh/ and follow the
   installation instruction. Then::

      % brew install gcc

   Using MacPorts, download MacPorts from http://www.macports.org/
   and follow the installation instruction. Then::

      % sudo port install gcc7
      % sudo port select --set gcc mp-gcc7

4) Install phonopy following :ref:`install_setup_py`.

   Before running setup.py, the environment variable of ``CC=gcc-7``
   (Homebrew) or ``CC=gcc`` (MacPorts) must be set so that installed
   gcc is used as the C-compiler instead of build-in clang for compiling
   the phonopy C-extension module.

   For Homebrew::

      % export CC=gcc-7

   For MacPorts::

      % export CC=gcc (MacPorts)

   Then::

      % python setup.py install --user
