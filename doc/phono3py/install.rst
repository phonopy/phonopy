.. _install:

Installation
=============

System requirement
-------------------

The following python libraries are required.

::

   python-dev python-numpy python-yaml python-h5py python-matplotlib 

``python-matplotlib`` is optional, but it is strongly recommended to
install it.  The OpenMP library is necessary for multithreding
support. The GNU OpenMP library is ``libgomp1``.  In the case of
ubuntu linux, these are installed using the package manager::

   % sudo apt-get install python-dev python-numpy python-matplotlib \
     python-yaml python-h5py libgomp1 liblapacke-dev

After the versions of Ubuntu-12.10, lapacke
(http://www.netlib.org/lapack/lapacke.html) can be installed from the
package manager (``liblapacke`` and ``liblapacke-dev``), but in older
versions of ubuntu, or in other environments, you may have to compile
lapacke by yourself. The compilation procedure is found at the lapacke
web site. After creating the lapacke library, ``liblapacke.a`` (or the
dynamic link library) ``setup3.py`` must be properly modified to link
it. As an example, the procedure of compiling lapacke is shown below.

::

   % tar xvfz lapack-3.5.0.tgz
   % cd lapack-3.5.0
   % cp make.inc.example make.inc
   % make lapackelib

Multithreading support
------------------------

Phono3py supports OpenMP multithreading and most users will need it,
otherwise the calculation may take long time. However, without special
OpenMP environment variables (``-lgomp`` and ``-fopenmp`` in
``setup3.py``), phono3py will be compiled without the OpenMP
multithreding support.

Installation procedure
------------------------

Download the latest version from
http://sourceforge.net/projects/phonopy/files/phono3py/ and extract it
somewhere. The version number here is not related to the version
number of harmonic (usual) phonopy. The harmonic phonopy included in
this package is a development version and can be different from that
distributed at sourceforge.net.

In the directory, open ``setup3.py`` and set the location of
lapacke. If you installed lapacke from the package manager, you can
remove the line related to lapacke. If you compiled it by yourself,
set the location of it. Then run ``setup3.py`` (for anharmonic
phonopy)::

   % python setup3.py install --home=.

In this way to setup, ``PYTHONPATH`` has to be set so that python can
find harmonic and anharmonic phonopy libraries. If you have been
already a user of phonopy, the original phonopy version distributed at
sourceforge.net will be removed from the list of the ``PYTHONPATH``.
The ``PYTHONPATH`` setting depends on shells that you use. For example
in bash or zsh::

   export PYTHONPATH=~/phonopy-0.9.9/lib/python

or::

   export PYTHONPATH=$PYTHONPATH:~/phonopy-0.9.9/lib/python



|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net
   
