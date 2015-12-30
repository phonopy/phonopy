.. _phonopy_qha:

Quasi harmonic approximation
=============================================

Usage of ``phonopy-qha``
------------------------

Using phonopy results of thermal properties, thermal expansion and
heat capacity at constant pressure can be calculated under the
quasi-harmonic approximation. ``phonopy-qha`` is the script to
calculate them. An example of the usage is as follows:

::

   phonopy-qha e-v.dat thermal_properties-{1..10}.yaml


1st argument is the filename of volume-energy data (in the above
expample, ``e-v.dat``). The volume and energy of the cell (default
units are in :math:`\mathrm{\AA}^3` and eV, respectively). An example of the
volume-energy file is::

   #   cell volume        energy of cell other than phonon
      156.7387309525      -104.5290025375
      154.4138492700      -104.6868148175
      152.2544070150      -104.8064238800
      150.2790355600      -104.8911768625
      148.4469296725      -104.9470385875
      146.7037426750      -104.9783724075
      145.1182305450      -104.9871878600
      143.5676103350      -104.9765270775
      142.1282086200      -104.9485225225
      139.4989658225      -104.8492814250

Lines starting with ``#`` are ignored. The other arguments are the
filenames of ``thermal_properties.yaml`` calculated at the respective
volumes given in the 1st argument. The ``thermal_properties.yaml`` at
volume points have to be calculated with the same temperature ranges
and same temperature steps. ``thermal_properties.yaml`` can be
calculated by following :ref:`thermal_properties_tag`, where the
physical unit of the Helmholtz free energy is kJ/mol as the default,
i.e., no need to convert the physical unit in usual cases.

The example for Aluminum is found in the ``example`` directory.

If the condition under puressure is expected, :math:`PV` terms may be
included in the energies.

.. _phonopy_qha_options:

Options
^^^^^^^

``-h`` 
~~~~~~~

Show help. The available options are shown. Without any option, the
results are saved into text files in simple data format.

``--tmax`` 
~~~~~~~~~~~~

The maximum temperature calculated is specified. This temperature has
to be lower than the maximum temperature calculated in
``thermal_properties.yaml`` to let at least two temperature points
fewer. The default value is ``--tmax=1000``.

``-p`` 
~~~~~~~

The fitting results, volume-temperature relation, and thermal expansion
coefficient are plotted on the display.

``--sparse`` 
~~~~~~~~~~~~~~

This is used with ``-s`` or ``-p`` to thin out the number of plots of
the fitting results at temperatures. When ``--sparse=10``, 1/10 is
only plotted.

``-s`` 
~~~~~~~

The calculated values are written into files.

``--pressure`` 
~~~~~~~~~~~~~~~~

**This option is not yet well tested. Please report to the mailing list when you get wrong results.**

Pressure is specified in GPa. This corresponds to the :math:`pV` term
described in the following section :ref:`theory_of_qha`. 

``-b`` 
~~~~~~~

Fitting volume-energy data to an EOS, and show bulk
modulus (without considering phonons). This is made by::

   phonopy-qha -b e-v.dat

``--eos``
~~~~~~~~~~~

EOS is chosen among ``vinet``, ``birch_murnaghan``, and
``murnaghan``. The default EOS is ``vinet``.

::

   phonopy-qha --eos='birch_murnaghan' -b e-v.dat

.. _phonopy_qha_output_files:

Output files
^^^^^^^^^^^^^

- Bulk modulus vs T (``bulk_modulus-temperature.*``)
- Gibbs free energy vs T (``gibbs-temperature.*``)
- Volume change with respect to the volume at 300 K vs T (``volume_expansion.*``)
- Heat capacity at constant pressure vs T derived by
  :math:`-T\frac{\partial^2 G}{\partial T^2}`  (``Cp-temperature.*``)
- Heat capacity at constant puressure vs T by polynomial fittings of
  Cv and S (``Cp-temperature_polyfit.*``)
- Helmholtz free energy vs volume (``helmholtz-volume.*``). When
  ``--pressure`` option is specified, energy offset of :math:`pV` is
  added. See also the following section (:ref:`theory_of_qha`).
- Volume vs T (``volume-temperature.*``)
- Thermal expansion coefficient vs T (``thermal_expansion.*``)

.. _theory_of_qha:

Theory of quasi-harmonic approximation
--------------------------------------

Here the word 'quasi-harmonic approximation' is used for an
approximation that introduces volume dependence of phonon frequencies
as a part of anharmonic effect.

A part of temperature effect can be included into total energy of
electronic structure through phonon (Helmholtz) free energy at
constant volume. But what we want to know is thermal properties at
constant pressure. We need some transformation from function of *V* to
function of *p*. Gibbs free energy is defined at a constant pressure by
the transformation:

.. math::

    G(T, p) = \min_V \left[ U(V) + F_\mathrm{phonon}(T;\,V) + pV \right],

where

.. math::
   \min_V[ \text{function of } V ]

means to find unique minimum value in the brackets by changing
volume. Since volume dependencies of energies in electronic and phonon
structures are different, volume giving the minimum value of the
energy function in the square brackets shifts from the value
calculated only from electronic structure even at 0 K. By increasing
temperature, the volume dependence of phonon free energy changes, then
the equilibrium volume at temperatures changes. This is considered as
thermal expansion under this approximation.

``phonopy-qha`` collects the values at volumes and transforms into the
thermal properties at constant pressure.

