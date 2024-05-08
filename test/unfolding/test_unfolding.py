"""Tests for band unfolding calculations."""

import os

import numpy as np

from phonopy import Phonopy
from phonopy.unfolding.core import Unfolding

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_Unfolding_NaCl(ph_nacl: Phonopy):
    """Test to reproduce proper band structure of primitive cell.

    Results are written to "bin-unfolding-test.dat".
    This data can be plotted by

    % plot_band.py bin-unfolding-test.dat

    Increase nd to get better plot.
    The test is done with nd=10.

    """
    # ph = _get_phonon(ph_nacl)
    ph = ph_nacl
    nd = 10
    qpoints = (
        np.array(
            [
                [
                    x,
                ]
                * 3
                for x in range(nd)
            ]
        )
        / float(nd)
        - 0.5
    )
    unfolding_supercell_matrix = [[-2, 2, 2], [2, -2, 2], [2, 2, -2]]
    mapping = np.arange(len(ph.supercell), dtype=int)
    unfolding = Unfolding(
        ph, unfolding_supercell_matrix, ph.supercell.scaled_positions, mapping, qpoints
    )
    unfolding.run()
    weights = _get_weights(unfolding, qpoints)
    # _write_weights(weights, "unfolding.dat")
    # filename_out = os.path.join(data_dir, "bin-unfolding-test.dat")
    _compare(weights, os.path.join(data_dir, "bin-unfolding.dat"), filename_out=None)


def test_Unfolding_SC(ph_nacl: Phonopy):
    """Test to reproduce unfoled band structure.

    Atomic positions are considered as the lattice ponts.

    Results are written to "bin-unfolding_to_atoms-test.dat".
    This data can be plotted by

    % plot_band.py bin-unfolding_to_atoms-test.dat

    Increase nd to get better plot.
    The test is done with nd=10.

    """
    # ph = _get_phonon(ph_nacl)
    ph = ph_nacl
    nd = 10
    qpoints = (
        np.array(
            [
                [
                    x,
                ]
                * 3
                for x in range(nd)
            ]
        )
        / float(nd)
        - 0.5
    )
    unfolding_supercell_matrix = np.diag([4, 4, 4])
    mapping = np.arange(len(ph.supercell), dtype=int)
    unfolding = Unfolding(
        ph, unfolding_supercell_matrix, ph.supercell.scaled_positions, mapping, qpoints
    )
    unfolding.run()
    weights = _get_weights(unfolding, qpoints)
    # _write_weights(weights, "unfolding_to_atoms.dat")
    # filename_out = os.path.join(data_dir, "bin-unfolding_to_atoms-test.dat")
    _compare(
        weights, os.path.join(data_dir, "bin-unfolding_to_atoms.dat"), filename_out=None
    )


def _compare(weights, filename, filename_out=None):
    bin_data = _binning(weights)
    if filename_out:
        _write_bin_data(bin_data, filename_out)
    with open(filename) as f:
        bin_data_in_file = np.loadtxt(f)
        np.testing.assert_allclose(bin_data, bin_data_in_file, atol=1e-2)


def _get_weights(unfolding, qpoints):
    weights = unfolding.unfolding_weights
    freqs = unfolding.frequencies

    out_vals = []
    for i, q in enumerate(qpoints):
        for f, w in zip(freqs[i], weights[i]):
            out_vals.append([q[0], q[1], q[2], f, w])

    return out_vals


def _write_weights(weights, filename):
    with open(filename, "w") as w:
        lines = ["%10.7f %10.7f %10.7f  %12.7f  %10.7f" % tuple(x) for x in weights]
        w.write("\n".join(lines))


def _write_bin_data(bin_data, filename):
    with open(filename, "w") as w:
        lines = ["%8.5f %8.5f %8.5f" % tuple(v) for v in bin_data]
        w.write("\n".join(lines))


def _binning(data):
    x = []
    y = []
    w = []
    for vals in data:
        if vals[4] > 1e-3:
            x.append(vals[0])
            y.append(vals[3])
            w.append(vals[4])
    x = np.around(x, decimals=5)
    y = np.around(y, decimals=5)
    w = np.array(w)

    points = {}
    for e_x, e_y, e_z in zip(x, y, w):
        if (e_x, e_y) in points:
            points[(e_x, e_y)] += e_z
        else:
            points[(e_x, e_y)] = e_z

    x = []
    y = []
    w = []
    for key in points:
        x.append(key[0])
        y.append(key[1])
        w.append(points[key])

    data = np.transpose([x, y, w])
    data = sorted(data, key=lambda data: data[1])
    data = sorted(data, key=lambda data: data[0])

    return np.array(data)


def _get_phonon(ph_in):
    ph = Phonopy(ph_in.supercell, supercell_matrix=[1, 1, 1])
    ph.force_constants = ph_in.force_constants
    born_elems = {
        s: ph_in.nac_params["born"][i] for i, s in enumerate(ph_in.primitive.symbols)
    }
    born = [born_elems[s] for s in ph_in.supercell.symbols]
    epsilon = ph_in.nac_params["dielectric"]
    factors = ph_in.nac_params["factor"]
    ph.nac_params = {"born": born, "factor": factors, "dielectric": epsilon}
    return ph
