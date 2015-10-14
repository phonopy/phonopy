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
version. See the document at http://phonopy.sourceforge.net/phono3py/
.

**Phonon database**: A collection of phonon and mode-Gruneisen
parameter calculations is available at
http://phonondb.mtl.kyoto-u.ac.jp/ . The raw data of phonopy & VASP
results can be downloaed.


Selected features
=======================

Selected features of harmonic and quasi-harmonic phonons are:

- Phonon dispersion relation (band structure)
- Phonon DOS and partial-DOS
- Phonon thermal properties, free energy, heat capacity (Cv), and entropy
- Phonon group velocity
- Thermal expansion and heat capacity at constant pressure (Cp) within
  quasi-harmonic approximation (:ref:`phonopy-qha <phonopy_qha>`)
- Mode Grüneisen parameters (:ref:`gruneisen <phonopy_gruneisen>`)
- Non-analytical-term correction, LO-TO splitting (Born effective
  charges and dielectric constant are required.)
- Creation of atomic modulations of normal modes
- Characters of irreducible representations of normal modes
- Crystal symmetry tools
- :ref:`Python module <phonopy_module>` (Phonopy can be used in python script.)
- Graphical plots
  
|i0| |i1| |i2| |i3|


Interfaces to calculators
===========================

- :ref:`VASP interface <tutorial>` (:ref:`finite displacement method <reference_plk>`)
- :ref:`Wien2k interface <wien2k_interface>` (:ref:`finite displacement method <reference_plk>`)
- :ref:`Abinit interface <abinit_interface>` (:ref:`finite displacement method <reference_plk>`) 
- :ref:`Pwscf interface <pwscf_interface>` (:ref:`finite displacement method <reference_plk>`) 
- :ref:`Siesta interface <siesta_interface>` (:ref:`finite displacement method <reference_plk>`) 
- :ref:`Elk interface <elk_interface>` (:ref:`finite displacement method <reference_plk>`) 
- :ref:`VASP DFPT interface <vasp_dfpt_interface>` (force constants)
- :ref:`FHI-aims interface <FHI_aims_interface>` (:ref:`finite displacement method <reference_plk>`)


For the other calculators, input files under certain formats
(:ref:`force sets <file_forces>` or :ref:`force constants
<file_force_constants>`) are required. But the interfaces for popular
calculators may be implemented if requested.

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


- `Manual in PDF <https://sourceforge.net/projects/phonopy/files/phonopy%20documentation/phonopy-manual.pdf/download>`_

- `Presentation: Introduction to phonopy <https://sourceforge.net/projects/phonopy/files/phonopy%20documentation/phonopy-workshop.pdf/download>`_

- `Presentation: Introduction to phonons <http://sourceforge.net/projects/phonopy/files/phonopy%20documentation/introduction-phonon-calc.pdf/download>`_

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

|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net


