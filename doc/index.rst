.. phonopy documentation master file, created by
   sphinx-quickstart on Mon Apr 13 15:11:21 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================================
Welcome to phonopy
=======================================================

.. |i0| image:: band.png
        :width: 15%

.. |i1| image:: pdos.png
        :width: 15%

.. |i2| image:: thermalprop.png
        :width: 15%

.. |i3| image:: QHA.png
        :width: 15%

**Phonopy** is an open source package for phonon calculations at
harmonic and quasi-harmonic levels.

**Phono3py** for phonon-phonon interaction calculations has been 
released as an open beta
version. See the document at http://atztogo.github.io/phono3py/
.

**Phonon database**: A collection of phonon and mode-Gruneisen
parameter calculations is available at
http://phonondb.mtl.kyoto-u.ac.jp/ . The raw data of phonopy & VASP
results can be downloaed.


Selected features
==================

- Phonon dispersion relation (band structure)
- Phonon DOS and partial-DOS
- Phonon thermal properties, free energy, heat capacity (Cv), and
  entropy
- Phonon group velocity
- Thermal ellipsoids / mean square displacements
- Thermal expansion and heat capacity at constant pressure (Cp) within
  quasi-harmonic approximation (:ref:`phonopy-qha <phonopy_qha>`)
- Mode Grüneisen parameters (:ref:`gruneisen <phonopy_gruneisen>`)
- Non-analytical-term correction, LO-TO splitting (Born effective
  charges and dielectric constant are required.)
- Irreducible representations of normal modes
- Interfaces to calculators:
  :ref:`VASP <tutorial>`,
  :ref:`VASP DFPT <vasp_dfpt_interface>`,
  :ref:`Abinit <abinit_interface>`,
  :ref:`Pwscf <pwscf_interface>`,
  :ref:`Siesta <siesta_interface>`,
  :ref:`Elk <elk_interface>`,
  :ref:`FHI-aims <FHI_aims_interface>`,
  :ref:`Wien2k <wien2k_interface>`
- :ref:`APIs <phonopy_module>`
  
|i0| |i1| |i2| |i3|

Documentation
=============

.. toctree::
   :maxdepth: 1

   examples
   Tutorial <procedure>
   workflow
   install
   features
   input-files
   output-files
   setting-tags
   command-options
   qha
   Mode Grüneisen parameters <gruneisen>
   interfaces
   auxiliary-tools
   external-tools
   theory
   citation
   reference
   changelog

.. _mailinglist:

Mailing list
============

For questions, bug reports, and comments, please visit following
mailing list:

https://lists.sourceforge.net/lists/listinfo/phonopy-users

Message body including attached files has to be smaller than 300 KB.

License
=======

New BSD from version 1.3.

(LGPL from ver. 0.9.3 to version 1.2.1., GPL to version 0.9.2.)

Contact
=======

* Author: `Atsushi Togo <http://atztogo.github.io/>`_

