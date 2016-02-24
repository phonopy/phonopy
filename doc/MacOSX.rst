.. _install_MacOSX:

Using phonopy on Mac OS X
==========================

Installation using MacPorts
----------------------------

This is one way to install phonopy on Mac OS X. This procedure was
tested on Yosemite with MacPorts version 2.3.3. In the following case,
gcc-4.9 and python-2.7 are used as the default C-compiler and python
and using these numpy and scipy are made.

1) Install MacPorts. Download MacPorts from http://www.macports.org/
   and follow the installation instruction.

2) Install the following packages

   ::
   
      % sudo port install gcc49
      % sudo port select --set gcc mp-gcc49
      % sudo port install python27
      % sudo port select --set python python27
      % sudo port install atlas +gcc49
      % sudo port install py27-numpy +gcc49 +atlas
      % sudo port install py27-scipy +gcc49 +atlas
      % sudo port install py27-matplotlib py27-yaml py27-h5py
   
3) Install phonopy following :ref:`install` (step 1 can be omitted.)

   Before running setup.py, gcc can be used as the C-compiler instead
   of clang for compiling phonopy C-extension as follows,

   ::
   
      % export CC=gcc
      % python setup.py install --home=.
