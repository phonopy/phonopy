.. _analyze_phonon:

Features
============

Band structure
--------------

Phonon band structure is calculated for the specified band paths
(:ref:`band_structure_related_tags`).

Density of states
-----------------

Total and partial density of states are calculated based on the
*q*-point sampling mesh (:ref:`dos_related_tags`). Smearing parameter
is set by :ref:`sigma_tag` tag or ``--sigma`` option.

Thermal properties
------------------

Helmholtz free energy, heat capacity at constant volume, and entropy
at temperatures are calculated from the phonon frequencies on the
*q*-point sampling mesh (:ref:`thermal_properties_tag`).

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

Calculation of mode Grüneisen parameters
-----------------------------------------

A script ``gruneisen`` is used for calculating mode Grüneisen
parameters in band structure style and mesh sampling style. See the
details at :ref:`phonopy_gruneisen`.


.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|

