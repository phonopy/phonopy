# How to cite phonopy

## Citation of phonopy

If you have used phonopy, please cite the following articles, which
indeed helps the phonopy project to continue:

- "Implementation strategies in phonopy and phono3py",

  Atsushi Togo, Laurent Chaput, Terumasa Tadano, and Isao Tanaka, J. Phys. Condens. Matter **35**, 353001-1-22 (2023)

  https://dx.doi.org/10.1088/1361-648X/acd831 (Open access)

  ```
  @article{phonopy-phono3py-JPCM,
    author  = {Togo, Atsushi and Chaput, Laurent and Tadano, Terumasa and Tanaka, Isao},
    title   = {Implementation strategies in phonopy and phono3py},
    journal = {J. Phys. Condens. Matter},
    volume  = {35},
    number  = {35},
    pages   = {353001},
    year    = {2023},
    doi     = {10.1088/1361-648X/acd831}
  }
  ```

- "First-principles Phonon Calculations with Phonopy and Phono3py",

  Atsushi Togo, J. Phys. Soc. Jpn., **92**, 012001-1-21 (2023)

  https://doi.org/10.7566/JPSJ.92.012001 (Open access)

  ```
  @article{phonopy-phono3py-JPSJ,
    author  = {Togo, Atsushi},
    title   = {First-principles Phonon Calculations with Phonopy and Phono3py},
    journal = {J. Phys. Soc. Jpn.},
    volume  = {92},
    number  = {1},
    pages   = {012001},
    year    = {2023},
    doi     = {10.7566/JPSJ.92.012001}
  }
  ```

## Some papers where phonopy was used

### `phonopy-qha`: Thermal properties, quasi-harmonic approximation used for thermal expansion

- "First-principles phonon calculations of thermal expansion in Ti3SiC2,
  Ti3AlC2, and Ti3GeC2",

  Atsushi Togo, Laurent Chaput, Isao Tanaka, Gilles Hug, Phys. Rev. B, **81**,
  174301-1-6 (2010)

### `MODULATION` tag: Collective atomic modulation, symmetry breaking

- "Evolution of crystal structures in metallic elements",

  Atsushi Togo and Isao Tanaka, Phys. Rev. B, **87**, 184104-1-6 (2013)

- "Transition pathway of CO2 crystals under high pressures",

  Atsushi Togo, Fumiyasu Oba, and Isao Tanaka, Phys. Rev. B, **77**, 184101-1-5
  (2008)

- "Inversion Symmetry Breaking by Oxygen Octahedral Rotations in the
  Ruddlesden-Popper NaRTiO4 Family",

  Hirofumi Akamatsu, Koji Fujita, Toshihiro Kuge, Arnab Sen Gupta, Atsushi Togo,
  Shiming Lei, Fei Xue, Greg Stone, James M. Rondinelli, Long-Qing Chen, Isao
  Tanaka, Venkatraman Gopalan, and Katsuhisa Tanaka, Phys. Rev. Lett. **112**,
  187602-1-5 (2014)

- "First-principles calculations of the ferroelastic transition between
  rutile-type and CaCl2-type SiO2 at high pressures",

  Atsushi Togo, Fumiyasu Oba, and Isao Tanaka, Phys. Rev. B, **78**, 134106
  (2008)

### `TDISP`, `TDISPMAT` tags: Root mean square thermal atomic displacement

- "Neutron diffraction measurements and first-principles study of thermal motion
  of atoms in select Mn+1AXn and binary MX transition-metal carbide phases",

  Nina J. Lane, Sven C. Vogel, Gilles Hug, Atsushi Togo, Laurent Chaput, Lars
  Hultman, and Michel W. Barsoum, Phys. Rev. B, **86**, 214301-1-9 (2012)

- "Ab initio ORTEP drawings: a case study of N-based molecular crystals with
  different chemical nature",

  Volker L. Deringer, Ralf P. Stoffel, Atsushi Togo, Bernhard Eck, Martin
  Mevencd and Richard Dronskowski, Cryst. Eng. Comm., (2014)

## A short history of phonopy

Phonopy development started to replace and extend fropho
(http://fropho.sourceforge.net/). The implementation of fropho is also based on
{ref}`reference_plk`. Although fropho was implemented from scratch except for
the symmetry finder and input file parser, to start the development, it was
motivated by the existence of PHON code. The important part of the
implementation is the symmetry handling. In fropho, at first the symmetry finder
in Abinit code was employed, but later the symmetry finder was replaced by
spglib (https://spglib.github.io/spglib/).
