"""Tests for pypolymlp calculater interface."""

from pathlib import Path

import pytest

from phonopy import Phonopy

cwd = Path(__file__).parent

atom_energies = {
    "Ac": -0.31859431,
    "Ag": -0.23270167,
    "Al": -0.28160067,
    "Ar": -0.03075282,
    "As": -0.98913829,
    "Au": -0.20332349,
    "B": -0.35900115,
    "Ba": -0.03514655,
    "Be": -0.02262445,
    "Bi": -0.76094876,
    "Br": -0.22521092,
    "C": -1.34047293,
    "Ca": -0.00975715,
    "Cd": -0.02058397,
    "Ce": 0.59933812,
    "Cl": -0.31144759,
    "Co": -1.84197165,
    "Cr": -5.53573206,
    "Cs": -0.16571995,
    "Cu": -0.27335372,
    "Dy": -0.28929836,
    "Er": -0.28193519,
    "Eu": -8.30324208,
    "F": -0.55560950,
    "Fe": -3.21756673,
    "Ga": -0.28636180,
    "Gd": -9.13005268,
    "Ge": -0.46078332,
    "H": -0.94561287,
    "Hf": -3.12504358,
    "Hg": -0.01574795,
    "Ho": -0.28263653,
    "I": -0.18152719,
    "In": -0.26352899,
    "Ir": -1.47509000,
    "K": -0.18236475,
    "Kr": -0.03632558,
    "La": -0.49815357,
    "Li": -0.28639524,
    "Lu": -0.27476515,
    "Mg": -0.00896717,
    "Mn": -5.16159814,
    "Mo": -2.14186043,
    "N": -1.90527586,
    "Na": -0.24580545,
    "Nb": -1.97405930,
    "Nd": -0.42445094,
    "Ne": -0.01878024,
    "Ni": -0.51162090,
    "Np": -6.82474030,
    "O": -0.95743902,
    "Os": -1.56551008,
    "P": -1.13999499,
    "Pa": -2.20245429,
    "Pb": -0.37387443,
    "Pd": -1.52069522,
    "Pm": -0.38617776,
    "Pr": -0.46104333,
    "Pt": -0.54871119,
    "Rb": -0.16762686,
    "Re": -2.54514167,
    "Rh": -1.38847510,
    "Ru": -1.43756529,
    "S": -0.57846242,
    "Sb": -0.82792086,
    "Sc": -2.05581313,
    "Se": -0.43790631,
    "Si": -0.52218874,
    "Sm": -0.35864636,
    "Sn": -0.40929915,
    "Sr": -0.03232469,
    "Ta": -3.14076900,
    "Tb": -0.29992122,
    "Tc": -3.39960587,
    "Te": -0.35871276,
    "Th": -1.08192032,
    "Ti": -2.48142180,
    "Tl": -0.25080265,
    "Tm": -0.27940877,
    "U": -4.39733623,
    "V": -3.58743790,
    "W": -2.99479772,
    "Xe": -0.01891425,
    "Yb": -0.00227841,
    "Y": -2.21597360,
    "Zn": -0.01564280,
    "Zr": -2.22799885,
}


def test_pypolymlp_develop(ph_nacl_rd: Phonopy):
    """Test of pypolymlp-develop using NaCl 2x2x2 with RD results."""
    pytest.importorskip("pypolymlp")
    pytest.importorskip("symfc")
    from phonopy.interface.pypolymlp import (
        PypolymlpParams,
        develop_polymlp,
        evalulate_polymlp,
    )

    params = PypolymlpParams(gtinv_maxl=(4, 4))

    disps = ph_nacl_rd.displacements
    forces = ph_nacl_rd.forces
    energies = ph_nacl_rd.supercell_energies
    ph_nacl_rd.produce_force_constants(fc_calculator="symfc")
    # ph_nacl_rd.auto_band_structure(write_yaml=True, filename="band_orig.yaml")
    ndata = 4
    polymlp = develop_polymlp(
        ph_nacl_rd.supercell,
        atom_energies,
        disps[:ndata].transpose((0, 2, 1)),
        forces[:ndata].transpose((0, 2, 1)),
        energies[:ndata],
        disps[8:].transpose((0, 2, 1)),
        forces[8:].transpose((0, 2, 1)),
        energies[8:],
        params=params,
    )

    ph = Phonopy(
        ph_nacl_rd.unitcell, ph_nacl_rd.supercell_matrix, ph_nacl_rd.primitive_matrix
    )
    ph.nac_params = ph_nacl_rd.nac_params
    ph.generate_displacements(distance=0.001, number_of_snapshots=2, random_seed=1)
    energies, forces, _ = evalulate_polymlp(polymlp, ph.supercells_with_displacements)
    ph.supercell_energies = energies
    ph.forces = forces
    ph.produce_force_constants(fc_calculator="symfc")
    # ph.auto_band_structure(write_yaml=True, filename="band_pypolymlp.yaml")
