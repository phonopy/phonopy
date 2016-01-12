.. _install:

Download and install
=====================

The procedure of setup phonopy is explained in this section. It is
supposed that phonopy is installed on the recent linux distribution
like Ubuntu or Fedora with Python version 2.6 or later. Mac OS X users
may find some more information on :ref:`install_MacOSX`. If you met
installation problems, it is recommended to prepare a system with
Ubuntu linux as a virtual machine. See :ref:`virtualmachine`

1. Prepare the following Python libraries:

   * Python and its header files
   * numpy
   * matplotlib
   * python-yaml
    
   In Ubuntu linux, they are installed by::
   
      % sudo apt-get install python-dev python-numpy \
        python-matplotlib python-yaml
    
   ``python-scipy`` is also required to use ``phonopy-qha`` or
   ``DEBYE_MODEL`` tag..

   The ``texlive-fonts-recommended`` package may be required, if you
   see the following message in ploting results::
   
      ! I can't find file `pncr7t'.


2. Download the source code from:

   https://sourceforge.net/projects/phonopy/files/phonopy/phonopy-1.9/ .

   and extract it::

   % tar xvfz phonopy-1.10.0.tar.gz

3. Set up C-libraries for python C-API and python codes. This can be
   done as follows:

   Run ``setup.py`` script::

      % python setup.py install --home=<my-directory>

   where :file:`{<my-directory>}` may be your current directory, :file:`.`.
   Another choice may be to use the user scheme (see the python document)::

      % python setup.py install --user

   The executable command ``phonopy`` is located in the ``bin`` directory.

4. Put ``lib/python`` path into :envvar:`$PYTHONPATH`, e.g., in your
   .bashrc, .zshenv, etc. If it is installed under your current
   directory, the path to be added to :envvar:`$PYTHONPATH` is such as below::

      export PYTHONPATH=~/phonopy-1.10.0/lib/python


Special cases on installation
------------------------------
   
.. toctree::
   :maxdepth: 1

   MacOSX   
   virtualmachine
