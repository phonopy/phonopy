# References

## Methods used in phonopy

(reference_force_constants)=

### Generation of supercell force constants

In phonopy, force constants are generated using the supercell method with finite
displacements. Several approaches can be employed to calculate supercell force
constants. Technical details regarding supercell method can be found in the
following paper:

- A. Togo, L. Chaput, T. Tadano, I. Tanaka,  J. Phys.: Condens. Matter 35 353001
  (2023)

(reference_systematic_displacement)=
#### Systematic displacement method

This is the traditional method that phonopy has employed for many years. Crystal
symmetry is utilized to reduce both the computational cost and numerical noise
in supercell force constant calculations. First, a symmetry-reduced set of
atomic displacements is systematically generated. After calculating the atomic
forces, the displacements are expanded using symmetry operations. The force
constants between atoms in the primitive cell and the supercell are then fitted
to the symmetry-expanded forces of atoms in the supercells using the
Moore–Penrose pseudoinverse.

This procedure can be considered a variant of {ref}`reference_plk` (see below).
Unlike the Parlinski–Li–Kawazoe method, supercell force constants are initially
computed without imposing the translational invariance constraint. The
constraint is applied a posteriori. Additional implementation details in phonopy
can be found in the appendix of the following paper:

- L. Chaput, A. Togo, I. Tanaka, and G. Hug, Phys. Rev. B, **84**, 094302 (2011)

(reference_random_displacement)=
#### Projector-based method

This approach is implemented in the symfc code, allowing for the displacement of
any number of atoms in the supercell. Typically, all atoms are displaced either
in random directions with a fixed displacement magnitude or with both random
directions and magnitudes. The former approach, using a small displacement
(e.g., 0.01 to 0.03 Angstrom), is recommended. However, for estimating supercell
force constants at finite temperatures, the latter approach may be used.

- A. Seko and A. Togo, Phys. Rev. B **110**, 214302 (2024)

#### Tadano-Tsuneyuki method

This approach is implemented in the ALM code, allowing for the displacement of
any number of atoms in the supercell.

- T. Tadano and S. Tsuneyuki, J. Phys. Soc. Jpn. **87**, 041015 (2018).

(reference_plk)=
### Parlinski-Li-Kawazoe method

Supercell force constants are calculated using the Moore–Penrose pseudoinverse
by fitting the symmetry-reduced elements of supercell force constants to the
linear relationships between atomic forces and atomic displacements. When
constructing the dynamical matrix, supercell boundary conditions are treated to
preserve crystal symmetry by averaging the phase factors of atomic pairs that
are equivalent under supercell lattice translations.

- K. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. **78**, 4063 (1997)

(reference_thermal_expansion)=

### Thermal expansion using quasi-harmonic approximation

In {ref}`phonopy-qha <phonopy_qha>`, thermal properties at constant pressure is
obtained from the thermodynamic definition. To achieve Legendre transformation,
volume-energy function is generated from a set of Helmholtz free energies and
_pV_ terms at volumes by fitting to a smooth function for which equations of
states are prepared in phonopy-qha.

The volume dependence of the Helmholtz free energy is included from
quasi-harmonicity. When using DFT-GGA (-LDA), often we should have some amount
of error in the absolute value since phonon frequencies are underestimated
(overestimated). However the value of some ratio like thermal expansion
coefficient is often very well estimated. An example is shown in the following
paper:

- A. Togo, L. Chaput, I. Tanaka, G. Hug, Phys. Rev. B, 81, 174301-1-6 (2010)

(reference_NAC)=

### Non-analytical term correction

Non-metallic crystals are polarized by atomic displacements and the generated
macroscopic field changes force constants near {math}`\Gamma` point. This
contribution is included through non-analytical term correction.

- R. M. Pick, M. H. Cohen, and R. M. Martin, Phys. Rev. B 1, 910, (1970)

(reference_dp_dp_NAC)=

### Correction by dipole-dipole interaction

1. P. Giannozzi, S. Degironcoli, P. Pavone, and S. Baroni, Phys. Rev. B 43, 7231
   (1991)
2. X. Gonze, J.-C. Charlier, D.C. Allan, and M.P. Teter Phys. Rev. B 50,
   13035(R) (1994)
3. X. Gonze, and C. Lee, Phys. Rev. B 55, 10355 (1997)

Currently phonopy implements the method by Gonze _et al._ written in the above
two papers (2 and 3) as the default method.

(reference_wang_NAC)=

### Interpolation scheme at general _q_-points with non-analytical term correction

This is an interpolation scheme using phonons at
{math}`\mathbf{q}\rightarrow \mathbf{0}` with the correction by Pick _et al._
and other commensurate points.

- Y. Wang , J. J. Wang , W. Y. Wang , Z. G. Mei , S. L. Shang , L. Q. Chen and Z
  K Liu, J. Phys.: Condens. Matter. 22, 202201 (2010)

The first derivative of this expression, which is for example used for group
velocity calculation, is described in the following paper:

- Atsushi Togo, Laurent Chaput, and Isao Tanaka, Phys. Rev. B, 91, 094306-1-31
  (2015)

## Other methods and software for calculating force constants

(reference_small_displacement)=

### Parlinsk-Li-Kawazoe method

[PHONON](http://wolf.ifj.edu.pl/phonon/) is the original implementation of the
Parlinsk-Li-Kawazoe method.

### Small displacement method

- Dario Alfè, Computer Physics Communications, 180, 2622 (2009)

[PHON](http://www.homepages.ucl.ac.uk/~ucfbdxa/phon/) is based on the small
displacement method.

(reference_dfpt)=

### DFPT

- Paolo Giannozzi, Stefano de Gironcoli, Pasquale Pavone, and Stefano Baroni,
  Phys. Rev. B, **43**, 7231 (1991)
- Xavier Gonze and Changyol Lee, Phys. Rev. B **55**, 10355 (1997)

Currently there are several many implementations such as
[Abinit](http://www.abinit.org/)
[Quantum espresso](http://www.quantum-espresso.org/)
[Elk](http://elk.sourceforge.net/), etc. VASP can calculate force constants
using DFPT however only at Gamma-point.

(reference_sscha)=

### SSCHA

Selected papers of SSCHA:
- Ion Errea, Matteo Calandra, and Francesco Mauri, Phys. Rev. Lett. **111**,
  177002 (2013)
- Lorenzo Monacelli, Raffaello Bianco, Marco Cherubini, Matteo Calandra, Ion
  Errea, and Francesco Mauri. J. Phys. Condens. Matter **33**, 363001 (2021).

A kind of SSCHA calculation performed using phonopy and ALM is presented in the
following paper:
- Atsushi Togo, Hiroyuki Hayashi, Terumasa Tadano, Satoshi Tsutsui, Isao Tanaka,
  J. Phys.: Condens. Matter **34**, 365401 (2022)

## For the study of basics

### Phonons

- Introduction to Lattice Dynamics, Martin. T. Dove, Cambridge university press
- Thermodynamics of Crystals, Duane C. Wallace, Dover Publications
- Electrons and Phonons by J. M. Ziman, Oxford University Press
- The Physics of Phonons by G. P. Srivastava, CRC Press

### Symmetry

- International Tables for Crystallography - IUCr
- Symmetry Relationships between Crystal Structures by Ulrich Müller, Oxford
  University Press
- Bilbao crystallographic server, https://www.cryst.ehu.es/
- Supplementary Material for the Lekeitio School,
  https://www.cryst.ehu.es/html/lekeitio.html, the presentation by B. Mihailova
  (phonons) is considered nice for beginners.
