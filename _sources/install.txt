.. _install:

Download and install
=====================

.. contents::
   :depth: 2
   :local:

System requirement
-------------------

The procedure to setup phonopy is explained in this section. It is
supposed that phonopy is installed on the recent linux distribution
like Ubuntu or Fedora with Python version 2.6 or later. Python version
3.4 or later is expected to work. Mac OS X users
may find some more information at :ref:`install_MacOSX`.
The most recommended system is Ubuntu linux version 14.04 or later. If
you have any installation problem that you may feel difficult to
solve, please use a ubuntu virtual machine (see
:ref:`virtualmachine`).

Prepare the following Python libraries:

* Python and its header files
* numpy
* matplotlib
* python-yaml (pyyaml)
* python-h5py (h5py)
    
In Ubuntu linux, they are installed by::
   
   % sudo apt-get install python-dev python-numpy \
     python-matplotlib python-yaml python-h5py
    
``python-scipy`` is also required to use ``phonopy-qha`` or
``DEBYE_MODEL`` tag. The python libraries are also possible to be
installed using pip or conda.

The ``texlive-fonts-recommended`` package may be required, if you
see the following message in ploting results::
   
   ! I can't find file `pncr7t'.

Install using pip/conda
------------------------

Occasionally PyPI and conda packages are prepared at phonopy
releases. Using these packages, the phonopy installtion is expected to
be easily done.

Using pip
~~~~~~~~~

Numpy is required before the python-spglib installation. The command to
install spglib is::

   % pip install phonopy

If you see the error message like below in the installation process::

   _phonopy.c:35:20: fatal error: Python.h: No such file or directory

development tools for building python module are additionally
necessary and are installed using OS's package management system,
e.g.,::

   sudo apt-get install python-dev

Using conda
~~~~~~~~~~~~

Conda is another choice for Linux (64bit) users::

   % conda install -c atztogo phonopy

Currently conda packages for the other OS, e.g., Mac and windows, are
not prepared by the main developers of phonopy.

Building using setup.py
------------------------

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

