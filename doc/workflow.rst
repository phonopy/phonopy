.. _workflow:

Work flow
==========

Phonon calculations at constant volume
--------------------------------------

Work flow of phonopy is shown schematically. There are two ways to
calculate, (1) atomic forces from finite displacements and (2) given
force constants. You can choose one of them. Forces on atoms or force
constants are calculated by your favorite calculator (shown by the
octagons in the work flow). The boxes are jobs being done by phonopy,
and the circles are input and intermediate output data structures.

.. figure:: procedure.png
   :scale: 80
   :align: center

   Work flow of phonon calculation

Combinations of phonon calculations at different volumes
---------------------------------------------------------

Mode Grüneisen parameters can be calculated from two or three phonon
calculation results obtained at slightly different volume points. See
the details at :ref:`phonopy_gruneisen`.

With more volume points and fitting the thermal properties, thermal
properties at constant pressure are obtained under the (so-called)
quasi-harmonic approximation. See more details at :ref:`phonopy_qha`.
