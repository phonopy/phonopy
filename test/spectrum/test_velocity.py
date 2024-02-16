"""Tests for velocity calculation from MD data."""

import os

import numpy as np

from phonopy.interface.vasp import read_XDATCAR
from phonopy.spectrum.velocity import Velocity

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_Velocity():
    """Test Velocity class."""
    lattice, positions = read_XDATCAR(os.path.join(data_dir, "XDATCAR"))
    v = Velocity(positions=positions, lattice=lattice, timestep=2)
    v.run()
    velocity = v.get_velocities()
    velocity_cmp = np.loadtxt(os.path.join(data_dir, "velocities.dat"))
    assert (np.abs(velocity.ravel() - velocity_cmp.ravel()) < 1e-1).all()


def _show(velocity):
    print(velocity)


def _write(velocity):
    np.savetxt(os.path.join(data_dir, "velocities.dat"), velocity.reshape((-1, 3)))
