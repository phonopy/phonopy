"""Tests for pypolymlp calculater interface."""

import pathlib

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.interface.pypolymlp import (
    PypolymlpData,
    PypolymlpParams,
    develop_pypolymlp,
    evalulate_pypolymlp,
)

cwd_called = pathlib.Path.cwd()

# POSCAR for atom energy calculation
#    1.0
# 15 0 0
# 0 15.1 0
# 0 0 15.2
# element
# 1
# Direct
# 0 0 0
#       PREC = Accurate
#     IBRION = -1
#     NELMIN = 5
#      ENCUT = 520
#      EDIFF = 1.000000e-08
#     ISMEAR = 0
#      SIGMA = 0.002
#      IALGO = 38
#      LREAL = .FALSE.
#      LWAVE = .FALSE.
#     LCHARG = .FALSE.
#       NPAR = 4
#       ISYM = 0
#        GGA = PS
#      ISPIN = 2
#       NELM = 500
#       ALGO = D
#    LSUBROT = .FALSE.
#       TIME = 0.2
#    ADDGRID = .TRUE.
# ps_map = {
#     "Ac": "Ac",
#     "Ag": "Ag_pv",
#     "Al": "Al",
#     "Ar": "Ar",
#     "As": "As_d",
#     "Au": "Au",
#     "B": "B",
#     "Ba": "Ba_sv",
#     "Be": "Be",
#     "Bi": "Bi",
#     "Br": "Br",
#     "C": "C",
#     "Ca": "Ca_pv",
#     "Cd": "Cd",
#     "Ce": "Ce",
#     "Cl": "Cl",
#     "Co": "Co",
#     "Cr": "Cr_pv",
#     "Cs": "Cs_sv",
#     "Cu": "Cu_pv",
#     "Dy": "Dy_3",
#     "Er": "Er_3",
#     "Eu": "Eu",
#     "F": "F",
#     "Fe": "Fe_pv",
#     "Ga": "Ga_d",
#     "Gd": "Gd",
#     "Ge": "Ge_d",
#     "H": "H",
#     "Hf": "Hf_pv",
#     "Hg": "Hg",
#     "Ho": "Ho_3",
#     "I": "I",
#     "In": "In_d",
#     "Ir": "Ir",
#     "K": "K_pv",
#     "Kr": "Kr",
#     "La": "La",
#     "Li": "Li_sv",
#     "Lu": "Lu_3",
#     "Mg": "Mg_pv",
#     "Mn": "Mn_pv",
#     "Mo": "Mo_pv",
#     "N": "N",
#     "Na": "Na_pv",
#     "Nb": "Nb_pv",
#     "Nd": "Nd_3",
#     "Ne": "Ne",
#     "Ni": "Ni_pv",
#     "Np": "Np",
#     "O": "O",
#     "Os": "Os_pv",
#     "P": "P",
#     "Pa": "Pa",
#     "Pb": "Pb_d",
#     "Pd": "Pd_pv",
#     "Pm": "Pm_3",
#     "Pr": "Pr_3",
#     "Pt": "Pt",
#     "Rb": "Rb_pv",
#     "Re": "Re_pv",
#     "Rh": "Rh_pv",
#     "Ru": "Ru_pv",
#     "S": "S",
#     "Sb": "Sb",
#     "Sc": "Sc_sv",
#     "Se": "Se",
#     "Si": "Si",
#     "Sm": "Sm_3",
#     "Sn": "Sn_d",
#     "Sr": "Sr_sv",
#     "Ta": "Ta_pv",
#     "Tb": "Tb_3",
#     "Tc": "Tc_pv",
#     "Te": "Te",
#     "Th": "Th",
#     "Ti": "Ti_pv",
#     "Tl": "Tl_d",
#     "Tm": "Tm_3",
#     "U": "U",
#     "V": "V_pv",
#     "W": "W_sv",
#     "Xe": "Xe",
#     "Y": "Y_sv",
#     "Yb": "Yb_2",
#     "Zn": "Zn",
#     "Zr": "Zr_sv",
# }
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
    pytest.importorskip("pypolymlp", minversion="0.10.0")
    pytest.importorskip("symfc")
    params = PypolymlpParams(gtinv_maxl=(4, 4), atom_energies=atom_energies)
    disps = ph_nacl_rd.displacements
    forces = ph_nacl_rd.forces
    energies = ph_nacl_rd.supercell_energies
    ph_nacl_rd.produce_force_constants(fc_calculator="symfc")
    # ph_nacl_rd.auto_band_structure(write_yaml=True, filename="band-orig.yaml")

    ndata = 4
    polymlp = develop_pypolymlp(
        ph_nacl_rd.supercell,
        PypolymlpData(
            displacements=disps[:ndata],
            forces=forces[:ndata],
            supercell_energies=energies[:ndata],
        ),
        PypolymlpData(
            displacements=disps[8:],
            forces=forces[8:],
            supercell_energies=energies[8:],
        ),
        params=params,
        verbose=True,
    )

    ph = Phonopy(
        ph_nacl_rd.unitcell,
        ph_nacl_rd.supercell_matrix,
        ph_nacl_rd.primitive_matrix,
        log_level=2,
    )
    ph.nac_params = ph_nacl_rd.nac_params
    # ph.generate_displacements(distance=0.001, number_of_snapshots=2, random_seed=1)
    # for v in ph.displacements.reshape(-1, 3):
    #     print(f"[{v[0]:.12f}, {v[1]:.12f}, {v[2]:.12f}],")
    displacements = [
        [0.000180268321, -0.000453658023, 0.000872752961],
        [0.000536302753, 0.000050282117, -0.000842526596],
        [0.000330618905, 0.000445286170, -0.000832112593],
        [-0.000977576278, -0.000171846132, -0.000121710833],
        [0.000607898748, -0.000579136579, 0.000543194196],
        [0.000555067463, 0.000770704299, 0.000312913718],
        [-0.000270591925, -0.000887099147, 0.000373945335],
        [0.000364662102, -0.000646885856, -0.000669746400],
        [0.000359734499, 0.000038998419, 0.000932239354],
        [0.000195533352, -0.000904804839, 0.000378278881],
        [0.000017818877, 0.000017550564, -0.000999687184],
        [0.000334379657, -0.000033555286, 0.000941840904],
        [-0.000286551811, 0.000349696633, 0.000891964307],
        [-0.000135098844, -0.000758622299, -0.000637369995],
        [-0.000608721346, -0.000790266333, 0.000070266958],
        [0.000384760855, 0.000214070436, 0.000897849059],
        [0.000013841826, -0.000856378711, -0.000516162676],
        [-0.000079142622, 0.000838911994, -0.000538482229],
        [-0.000468826199, -0.000418905515, -0.000777637554],
        [-0.000268087282, -0.000760751939, -0.000591088569],
        [0.000007848629, 0.000830081071, -0.000557587494],
        [-0.000412491332, -0.000059615806, 0.000909008722],
        [0.000583819791, -0.000802793210, 0.000121150786],
        [0.000585221724, 0.000364440907, -0.000724360656],
        [-0.000935217657, 0.000295063265, 0.000195718173],
        [-0.000699970508, -0.000166726794, 0.000694437517],
        [-0.000140708283, -0.000226715275, 0.000963743412],
        [-0.000357234776, 0.000411213978, 0.000838621714],
        [0.000228795432, -0.000973234658, 0.000021609030],
        [0.000197815347, 0.000399039275, 0.000895341692],
        [0.000906486344, 0.000085304595, -0.000413528275],
        [-0.000740336533, -0.000449341675, 0.000499993877],
        [-0.000261310807, -0.000963364513, -0.000060377788],
        [0.000870822149, -0.000096174508, 0.000482098795],
        [0.000546221585, -0.000739403615, 0.000393604210],
        [0.000408838031, 0.000617459207, -0.000672008625],
        [-0.000936345792, 0.000262474817, 0.000233159879],
        [-0.000751311446, 0.000356530493, 0.000555353148],
        [0.000145685606, 0.000117114073, -0.000982374673],
        [0.000173753618, 0.000419028597, -0.000891192861],
        [-0.000745516709, -0.000475608261, -0.000466906435],
        [-0.000384951997, 0.000376399525, -0.000842695293],
        [-0.000035517628, 0.000879858577, 0.000473906513],
        [-0.000574194999, -0.000188219889, 0.000796789418],
        [-0.000098215583, -0.000592446474, 0.000799600447],
        [0.000313320598, -0.000517929638, 0.000795976817],
        [0.000073314590, -0.000991531263, -0.000107194801],
        [-0.000564042854, -0.000781494973, 0.000266685705],
        [0.000623511496, 0.000145121020, -0.000768227378],
        [0.000688006698, -0.000224596761, 0.000690074690],
        [0.000177528488, 0.000796142755, 0.000578481072],
        [-0.000659533276, 0.000000162534, 0.000751675350],
        [0.000761914593, 0.000337309495, -0.000552909086],
        [-0.000464787706, 0.000882434676, 0.000072673446],
        [0.000931458075, -0.000318646515, -0.000175642402],
        [-0.000423266534, 0.000567390964, 0.000706337692],
        [0.000811828321, -0.000561681248, 0.000159527280],
        [-0.000009794644, -0.000394612095, -0.000918795603],
        [-0.000917520322, -0.000269115301, 0.000292802689],
        [-0.000164198626, -0.000060007868, 0.000984600359],
        [0.000034663251, -0.000897822527, 0.000438991082],
        [0.000296035060, -0.000038085104, 0.000954417502],
        [-0.000507436256, -0.000861487867, 0.000018630641],
        [-0.000417040087, 0.000524283593, -0.000742431330],
        [0.000109479743, -0.000044429923, -0.000992995553],
        [-0.000315952423, -0.000434554398, -0.000843407696],
        [0.000248711151, -0.000959271491, -0.000133943908],
        [0.000838362745, -0.000424336937, 0.000342178420],
        [-0.000915490715, -0.000123865082, 0.000382797848],
        [0.000225588394, -0.000926260058, -0.000301914194],
        [0.000678113565, -0.000509310476, 0.000529872468],
        [-0.000662290126, -0.000416648648, -0.000622716383],
        [-0.000679239662, -0.000436273575, -0.000590168492],
        [0.000510270225, 0.000637097261, 0.000577694883],
        [0.000171060563, 0.000767991408, -0.000617193229],
        [0.000311891939, 0.000005577608, -0.000950101210],
        [-0.000285022166, 0.000391020374, -0.000875137379],
        [-0.000742080749, -0.000668649865, 0.000047154221],
        [-0.000034953305, 0.000202524320, -0.000978653241],
        [-0.000779346397, -0.000053462565, -0.000624308375],
        [0.000797012992, 0.000498226591, -0.000341380366],
        [0.000089955695, 0.000743474829, -0.000662686315],
        [-0.000514184744, -0.000719123655, -0.000467413326],
        [-0.000913862571, 0.000199529903, -0.000353614224],
        [0.000584220150, -0.000726561971, -0.000361655248],
        [0.000438770705, 0.000382247620, 0.000813244751],
        [-0.000424456131, -0.000870675074, 0.000248519432],
        [-0.000000633079, -0.000299034961, -0.000954241946],
        [0.000452396031, 0.000205837949, -0.000867737616],
        [0.000448029552, 0.000586822782, 0.000674469082],
        [0.000419253562, 0.000035786554, 0.000907163587],
        [0.000279226575, -0.000863126212, 0.000420767944],
        [-0.000147749171, -0.000862441833, 0.000484111833],
        [-0.000124657458, 0.000425963710, 0.000896111285],
        [0.000999755946, -0.000005207967, -0.000021469160],
        [-0.000795940478, -0.000595425502, -0.000109303373],
        [-0.000083944712, 0.000510770307, -0.000855609128],
        [0.000049889341, 0.000629281721, -0.000775574348],
        [-0.000519037986, 0.000318052532, 0.000793373907],
        [0.000222101337, -0.000224646491, -0.000948791310],
        [-0.000618130873, 0.000786003433, 0.000010622050],
        [0.000439368229, -0.000540554267, -0.000717465431],
        [-0.000210145069, 0.000953743624, 0.000214970116],
        [0.000550756193, -0.000403614562, 0.000730590789],
        [0.000861918314, 0.000476875033, -0.000172299227],
        [0.000294442568, 0.000773260802, 0.000561579297],
        [-0.000647672986, -0.000544329599, 0.000533127556],
        [-0.000958579939, -0.000032428810, 0.000282971504],
        [0.000714544090, 0.000015874900, 0.000699410274],
        [-0.000078029664, 0.000834102940, 0.000546061953],
        [-0.000665035019, 0.000686276604, -0.000294538361],
        [0.000102753664, -0.000868485814, -0.000484947498],
        [-0.000195298729, 0.000466901888, -0.000862473787],
        [0.000693979808, 0.000606796409, 0.000387543732],
        [0.000015790804, 0.000988458582, -0.000150666125],
        [0.000004290865, -0.000524016021, 0.000851697598],
        [-0.000348973828, -0.000261934267, 0.000899782033],
        [0.000328409977, 0.000932542909, -0.000150035366],
        [-0.000595572294, -0.000780605767, -0.000189600315],
        [0.000305891510, -0.000550999592, 0.000776421171],
        [0.000615171274, 0.000208731729, -0.000760260067],
        [-0.000807433341, 0.000539322252, -0.000239129481],
        [-0.000904567307, -0.000245259646, 0.000348720080],
        [0.000502975931, 0.000440396051, -0.000743684430],
        [0.000981936098, 0.000045073472, -0.000183765833],
        [-0.000540310857, 0.000819838365, -0.000189550079],
        [-0.000884174313, -0.000001073455, 0.000467155897],
        [0.000507583278, 0.000853411695, 0.000118522969],
    ]
    ph.displacements = np.reshape(displacements, (2, -1, 3))
    energies, forces, _ = evalulate_pypolymlp(polymlp, ph.supercells_with_displacements)
    ph.supercell_energies = energies
    ph.forces = forces
    ph.produce_force_constants(fc_calculator="symfc")
    # ph.auto_band_structure(write_yaml=True, filename="band-pypolymlp.yaml")

    # ph_nacl_rd.run_mesh([2, 2, 2])
    # ph_nacl_rd.get_mesh_dict()["frequencies"]
    ph.run_mesh([2, 2, 2])
    freqs = ph.get_mesh_dict()["frequencies"]
    print(freqs.ravel().tolist())
    freqs_ref = [
        1.941547832280018,
        1.9415478322800197,
        3.193725481896553,
        4.3156898153907415,
        4.315689815390742,
        6.956947216285988,
        2.7371782984779234,
        3.551580148629576,
        3.91783431236152,
        4.480847025290504,
        4.888973568501199,
        5.4520357131954515,
    ]
    np.testing.assert_allclose(freqs.ravel(), freqs_ref, atol=1e-6)
