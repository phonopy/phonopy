# References

## Method used in phonopy

(reference_force_constants)=

### Generation of force constants

In phonopy, force constants are generated based on finite displacement method.
Crystal symmetry is used to reduce the calculation cost and numerical noise of
the force constants. Firstly a symmetry reduced set of atomic displacements is
generated. After the atomic force calculations, the set of atomic displacements
are expanded using the symmetry and then all the elements of force constans
between atoms in a primitive cell and the supercell are fit to the symmetry
expanded forces of atoms in supercells using Moore–Penrose pseudoinverse. This
procedure may considered as a variant of {ref}`reference_plk`. Some of the
details are found in the appendix of the following paper:

- L. Chaput, A. Togo, I. Tanaka, and G. Hug, Phys. Rev. B, 84, 094302 (2011)

(reference_plk)=

### Parlinski-Li-Kawazoe method

Parlinski-Li-Kawazoe method is based on the supercell approach with the finite
displacement method.

Force constants are calculated using Moore–Penrose pseudoinverse by fitting
symmetry reduced elements of force constans to the linear relations between
atomic forces and atomic displacements. The pseudoinverse is easy to handle
arbitrary number of displacements amplitudes and directions, and can rely on the
existing library, e.g., LAPACK.

- K. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. 78, 4063 (1997)

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
