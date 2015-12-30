.. _create_animation:

How to watch animation
-----------------------

To watch each phonon mode, v_sim is recommended. The file
``anime.ascii`` is supposed to work with v_sim version 3.51 or later.
An example how to watch phonon modes at a *q*-point is shown as follows.

First, you need to create a phonopy input file with, e.g., ``ANIME =
0.5 0.5 0``. After running phonopy with this input file, you get
``anime.ascii`` that contains all phonon modes at the *q*-point. Then
start v_sim

::

   v_sim anime.ascii

After opening the graphical user interface, you can find a tab called
**Phonons**. There you can see the phonon modes at the *q*-point that
you specified in the phonopy input file. Then select one of the phonon
modes and watch by pushing the play button. Because only the unit cell
shows up at the start of v_sim, if you want to watch a phonon
modulation with a longer period, then change the values of **Expand
nodes** in the **Box and symmetry** tab
(http://inac.cea.fr/L_Sim/V_Sim/user_guide.html#trans).  This is
especially important when you choose a *q*-point other than the
:math:`\Gamma`-point.

V_sim has a good graphical user interface and also a lot of command
line options. To read the manual well and to check the command line
options help you to use v_sim comfortably, e.g.,

::

   v_sim -w oneWindow anime.ascii -x 1:1:0 -t 0.5:0.5:0.5
