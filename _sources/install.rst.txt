.. _install:

Download and install
=====================

.. contents::
   :depth: 3
   :local:

From source code
-----------------

System requirement
~~~~~~~~~~~~~~~~~~

The procedure to setup phonopy is explained in this section. It is
supposed that phonopy is installed on the recent linux distribution
like Ubuntu or Fedora with Python version 2.6 or later. Python version
3.4 or later is expected to work. Mac OS X users may use conda packages 
and also find some more information at :ref:`install_MacOSX`. 
Windows users should use conda packages as well. 

Prepare the following Python libraries:

* Python and its header files
* numpy
* matplotlib
* python-yaml (pyyaml)
* python-h5py (h5py)

By Ubuntu package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^

The most recommended system is Ubuntu linux version 14.04 or later. If
you have any installation problem that you may feel difficult to
solve, please use a ubuntu virtual machine (see
:ref:`virtualmachine`).  

The python libraries are installed by::
   
   % sudo apt-get install python-dev python-numpy  python-matplotlib python-yaml python-h5py
    
``python-scipy`` is also required to use ``phonopy-qha`` or
``DEBYE_MODEL`` tag.

The ``texlive-fonts-recommended`` package may be required, if you
see the following message in ploting results::
   
   ! I can't find file `pncr7t'.

By conda
^^^^^^^^^

The python libraries may be also installed using pip or
conda. Conda packages are distributed in binary and recommended often
more than pip. The installation of necessary libraries is done as
follows::

   % conda install numpy scipy h5py pyyaml matplotlib

Building using setup.py
~~~~~~~~~~~~~~~~~~~~~~~~

If package installation is not possible or you want to compile with
special compiler or special options, phonopy is built using
setup.py. In this case, manual modification of ``setup.py`` may be
needed.

1. Download the source code at

   https://pypi.python.org/pypi/phonopy

   and extract it::

   % tar xvfz phonopy-1.11.2.tar.gz

2. Set up C-libraries for python C-API and python codes. This can be
   done as follows:

   Run ``setup.py`` script::

      % python setup.py install --home=<my-directory>

   where :file:`{<my-directory>}` may be your current directory, :file:`.`.
   Another choice may be to use the user scheme (see the python document)::

      % python setup.py install --user

   The executable command ``phonopy`` is located in the ``bin`` directory.

3. Put ``lib/python`` path into :envvar:`$PYTHONPATH`, e.g., in your
   .bashrc, .zshenv, etc. If it is installed under your current
   directory, the path to be added to :envvar:`$PYTHONPATH` is such as below::

      export PYTHONPATH=~/phonopy-1.11.2/lib/python

Tips on setup.py installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
.. toctree::
   :maxdepth: 1

   MacOSX   
   virtualmachine

conda
------

Conda is a good choice for all users (Linux/MacOSX/Windows). 
The Linux (64bit) conda packages are prepared by the author and 
can be installed using::

   % conda install numpy scipy h5py pyyaml matplotlib
   % conda install -c atztogo phonopy

Currently conda packages for the other OS: (MacOSX, Windows and other flavours
of Linux), are prepared by conda-forge project and may be installed using::

   % conda install -c conda-forge phonopy

The conda-forge packages are usually available within few days after 
the release. 


pip
----

Phonopy is installed using pip by::

   % pip install phonopy

If you see the error message like below in the installation process::

   _phonopy.c:35:20: fatal error: Python.h: No such file or directory

development tools for building python module are additionally
necessary and are installed using OS's package management system,
e.g.,::

   sudo apt-get install python-dev


