.. _install_MacOSX:

Using phonopy on Mac OS X
==========================

Installation using MacPorts
----------------------------

This is one way to install phonopy on Mac OS X. This procedure was
tested on Sierra with MacPorts version 2.4.1. In the following case,
gcc-7 is used as the default C-compiler.

1) Install MacPorts. Download MacPorts from http://www.macports.org/
   and follow the installation instruction.

2) Install ``gcc`` by

   ::
   
      % sudo port install gcc7
      % sudo port select --set gcc mp-gcc7

3) Install necessary python libraries by conda::

   % conda install numpy scipy h5py pyyaml matplotlib

4) Install phonopy following :ref:`install_setup_py`.

   Before running setup.py, the environment variable of ``CC=gcc`` is
   set so that gcc can be used as the C-compiler instead
   of clang for compiling phonopy C-extension as follows::

      % export CC=gcc
      % python setup.py install --user
