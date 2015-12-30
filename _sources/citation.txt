How to cite phonopy
====================

Citation of phonopy
--------------------

If you have used phonopy, please cite the following article:

* "First principles phonon calculations in materials science",
    Atsushi Togo and Isao Tanaka, Scr. Mater., **108**, 1-5 (2015)

    http://dx.doi.org/10.1016/j.scriptamat.2015.07.021

  ::
  
     @article {phonopy,
          Journal = {Scr. Mater.},
          Year = {2015},
          Title = {First principles phonon calculations in materials science},
          Author = {Togo, A and Tanaka, I},
          Pages = {1--5},
          Volume = {108},
          Month = {Nov}
     }

The following article is not necessarily cited, but the citation
encourages running phonopy project.

* "First-principles calculations of the ferroelastic transition between rutile-type and CaCl2-type SiO2 at high pressures",
    Atsushi Togo, Fumiyasu Oba, and Isao Tanaka, Phys. Rev. B, **78**, 134106 (2008)

    http://dx.doi.org/10.1103/PhysRevB.78.134106

  ::
  
     @article {phonopy,
  	   Journal = {Phys. rev. B},
  	   Year = {2008},
  	   Title = {First-principles calculations of the ferroelastic transition between rutile-type and CaCl$_2$-type SiO$_2$ at high pressures},
  	   Author = {Togo, A and Oba, F and Tanaka, I},
  	   Pages = {134106--134114},
  	   Volume = {78},
  	   Issue = {13},
  	   Month = {Oct}
     }

Some papers where phonopy was used
-----------------------------------

``phonopy-qha``: Thermal properties, quasi-harmonic approximation used for thermal expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  "First-principles phonon calculations of thermal expansion in Ti3SiC2, Ti3AlC2, and Ti3GeC2"
     Atsushi Togo, Laurent Chaput, Isao Tanaka, Gilles Hug,
     Phys. Rev. B, **81**, 174301-1-6 (2010)

``MODULATION`` tag: Collective atomic modulation, symmetry breaking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  "Evolution of crystal structures in metallic elements"
     Atsushi Togo and Isao Tanaka,
     Phys. Rev. B, **87**, 184104-1-6 (2013)
 
*  "Transition pathway of CO2 crystals under high pressures"
     Atsushi Togo, Fumiyasu Oba, and Isao Tanaka,
     Phys. Rev. B, **77**, 184101-1-5 (2008)

*  "Inversion Symmetry Breaking by Oxygen Octahedral Rotations in the Ruddlesden-Popper NaRTiO4 Family"
     Hirofumi Akamatsu, Koji Fujita, Toshihiro Kuge, Arnab Sen Gupta, Atsushi Togo, Shiming Lei, Fei Xue, Greg Stone, James M. Rondinelli, Long-Qing Chen, Isao Tanaka, Venkatraman Gopalan, and Katsuhisa Tanaka
     Phys. Rev. Lett. **112**, 187602-1-5 (2014)   

``TDISP``, ``TDISPMAT`` tags: Root mean square thermal atomic displacement 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  "Neutron diffraction measurements and first-principles study of thermal motion of atoms in select Mn+1AXn and binary MX transition-metal carbide phases"
     Nina J. Lane, Sven C. Vogel, Gilles Hug, Atsushi Togo, Laurent Chaput, Lars Hultman, and Michel W. Barsoum,
     Phys. Rev. B, **86**, 214301-1-9 (2012)

*  "Ab initio ORTEP drawings: a case study of N-based molecular crystals with different chemical nature"
     Volker L. Deringer, Ralf P. Stoffel, Atsushi Togo, Bernhard Eck, Martin Mevencd and Richard Dronskowski
     Cryst. Eng. Comm., (2014)

  
A short history of phonopy
---------------------------

Phonopy development started to replace and extend fropho
(http://fropho.sourceforge.net/). The implementation of fropho is also
based on :ref:`reference_plk`. Although fropho was implemented from
scratch except for the symmetry finder and input file parser, to start
the development, it was motivated by the existence of PHON code. The
important part of the implementation is the symmetry handling. In
fropho, at first the symmetry finder in Abinit code was employed, but
later the symmetry finder was replaced by spglib
(http://spglib.sourceforge.net/).

