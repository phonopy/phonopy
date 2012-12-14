How to cite phonopy
====================

Citation of phonopy
--------------------

If you have used phonopy, please cite the following article:

- "First-principles calculations of the ferroelastic transition
  between rutile-type and CaCl2-type SiO2 at high pressures",
  Atsushi Togo, Fumiyasu Oba, and Isao Tanaka, Phys. Rev. B, 78, 134106 (2008)

::

  @article {phonopy,
       Journal = {Phys. rev. B},
       Year = {2008},
       Title = {First-principles calculations of the ferroelastic transition between rutile-type and CaCl2-type SiO2 at high pressures},
       Author = {Togo, A and Oba, F and Tanaka, I},
       Pages = {134106},
       Volume = {78},
       Issue = {13},
       Month = {Oct}
  }

A short history of phonopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Phonopy development started to replace and extend fropho
(http://fropho.sourceforge.net/). The implementation of fropho is also
based on :ref:`reference_plk`. Although fropho was implemented from
scratch except for the symmetry finder and input file parser, to start
the development, it was motivated by the existence of PHON code. The
important part of the implementation is the symmetry handling. In
fropho, at first the symmetry finder in Abinit code was employed, but
later the symmetry finder was replaced by spglib
(http://spglib.sourceforge.net/).


|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net
