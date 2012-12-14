.. _install_MacOSX:

Using phonopy on Mac OS X
==========================

Installation using MacPorts
----------------------------

1) Install MacPorts. Download MacPorts from http://www.macports.org/
   and follow the installation instruction.

2) Install the following packages

   ::

      py27-matplotlib
      py27-lxml
      py27-yaml
      py27-scipy
   
   MacPorts command can be used as follows::
   
      % sudo port install py27-matplotlib

   At the same time, many dependent packages are also installed.

3) Add the following line to ``~/.matplotlib/matplotlibrc``

   ::

      backend : MacOSX

4) Set ``/opt/local/bin/python`` to be prior than the Mac OS X default
   ``python``, e.g.,::

   export PATH=/opt/local/bin:$PATH

   in ``.bashrc`` or ``.zshrc``.

5) Set ``python27`` as the default ``python`` command by::

   % port select python python27

6) Add the path below to ``PYTHONPATH``.

   ::

      /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

   This path can be system dependent. ``PYTHONPATH`` setting in step 3
   of :ref:`install` is also necessary.

7) Install phonopy following :ref:`install` (step 1 can be omitted.)

Make sure that step 6 is done after step 5.

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|
