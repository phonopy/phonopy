.. _analyze_phonon:

Features
============

Band structure
--------------

Phonon band structure is calculated for the specified band paths.
See :ref:`band_structure_related_tags`.

Density of states
-----------------

Total and partial density of states are calculated based on the
*q*-point sampling mesh (:ref:`dos_related_tags`). Smearing parameter
is set by :ref:`sigma_tag` tag or ``--sigma`` option.

Group velocity
---------------

Phonon group velocity is calculated from first derivative of dynamical
matrix. See :ref:`group_velocity_tag`.

Thermal properties
------------------

Helmholtz free energy, heat capacity at constant volume, and entropy
at temperatures are calculated from the phonon frequencies on the
*q*-point sampling mesh. See :ref:`thermal_properties_tag`.

Thermal properties at constant pressure and thermal expansion
---------------------------------------------------------------

Gibbs free energy, heat capacity at constant pressure, and thermal
expansion are calcualted with quasi-harmonic approximation. See
:ref:`phonopy_qha`.

Measure of how far atoms move at finite temperature
----------------------------------------------------

How far atoms are displaced from their equilibrium positions at
temperatures is calculated as mean square displacements. See
:ref:`thermal_atomic_displacements_tags`.

Calculation of mode Grüneisen parameters
-----------------------------------------

A script ``gruneisen`` is used for calculating mode Grüneisen
parameters in band structure style and mesh sampling style. See the
details at :ref:`phonopy_gruneisen`. 

Normal mode analysis
---------------------

Irreducible representations are assigned using polarization vectors of
phonon normal modes
(:ref:`irreducible_representation_related_tags`). Atomic displacements
corresponding to the real part of the polarization vectors are
obtained (:ref:`modulation_tag`). This may be
applied for research of the second-order like structural phase
transition.

Animation
----------

Phonon mode is visualized by animation. See :ref:`animation_tag`.

Plot and output
---------------

The results of DOS, PDOS, band structure, and thermal properties are
immediately plotted by specifying ``-p`` option
(:ref:`graph_option`). When ``-s`` option is set together with the
``-p`` option, the plot is stored in the PDF file
(:ref:`graph_save_option`). In addition those results are saved
in output text files (:ref:`output_files`), too.


