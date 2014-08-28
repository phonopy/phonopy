.. _install:

Download and install
=====================

The procedure of setup phonopy is explained in this section. It is
supposed that phonopy is installed on the recent linux distribution
like Ubuntu or Fedora with Python version 2.5 or later. Mac OS X users
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

   https://sourceforge.net/projects/phonopy/ .

   and extract it::

   % tar xvfz phonopy-1.8.5.1.tar.gz

3. Put your phonopy directory into your ``PYTHONPATH`` in .bashrc etc,
   e.g.::

      export PYTHONPATH=~/phonopy-1.8.5.1/lib/python

4. Set up C-libraries for python C-API and python codes. This can be
   done as follows:

   Run ``setup.py`` script::

   % python setup.py install --home=.

   The command ``phonopy`` is located in the ``bin`` directory. The
   install location can be specified by the option ``--home``.


Special cases on installation
==============================
   
.. toctree::
   :maxdepth: 1

   MacOSX   
   virtualmachine

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|
