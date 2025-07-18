"""Tests of Phonopy API."""

import copy
from pathlib import Path

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.interface.pypolymlp import PypolymlpParams
from phonopy.structure.dataset import get_displacements_and_forces

cwd = Path(__file__).parent


def test_displacements_setter_NaCl(ph_nacl: Phonopy):
    """Test Phonopy.displacements setter and getter."""
    ph_in = ph_nacl
    displacements, _ = get_displacements_and_forces(ph_in.dataset)
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.displacements = displacements
    np.testing.assert_allclose(displacements, ph.displacements)


def test_forces_setter_NaCl_type1(ph_nacl: Phonopy):
    """Test Phonopy.forces setter and getter (type1 dataset)."""
    ph_in = ph_nacl
    forces = ph_in.forces
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    ph.forces = forces
    np.testing.assert_allclose(ph.forces, forces)


def test_forces_setter_NaCl_type2(ph_nacl: Phonopy):
    """Test Phonopy.forces setter and getter (type2 dataset)."""
    ph_in = ph_nacl
    displacements, forces = get_displacements_and_forces(ph_in.dataset)
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.displacements = displacements
    ph.forces = forces
    np.testing.assert_allclose(ph.forces, forces)


def test_energies_setter_NaCl_type1(ph_nacl_fd: Phonopy):
    """Test Phonopy.energies setter and getter (type1 dataset)."""
    ph_in = ph_nacl_fd
    energies = ph_in.supercell_energies
    ref_supercell_energies = [-216.82820693, -216.82817843]
    np.testing.assert_allclose(energies, ref_supercell_energies)

    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    np.testing.assert_allclose(energies, ph.supercell_energies)
    ph.supercell_energies = ph_in.supercell_energies + 1
    np.testing.assert_allclose(ph.supercell_energies, ph_in.supercell_energies + 1)


def test_energies_setter_NaCl_type2(ph_nacl_rd: Phonopy):
    """Test Phonopy.energies setter and getter (type2 dataset)."""
    ph_in = ph_nacl_rd
    energies = ph_in.supercell_energies
    ref_supercell_energies = [
        -216.84472784,
        -216.84225045,
        -216.84473283,
        -216.84431749,
        -216.84276813,
        -216.84204234,
        -216.84150156,
        -216.84438529,
        -216.84339285,
        -216.84005085,
    ]
    np.testing.assert_allclose(energies, ref_supercell_energies)

    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
    )
    ph.dataset = ph_in.dataset
    np.testing.assert_allclose(energies, ph.supercell_energies)
    ph.supercell_energies = ph_in.supercell_energies + 1
    np.testing.assert_allclose(ph.supercell_energies, ph_in.supercell_energies + 1)


def test_mlp_NaCl_type2(ph_nacl_rd: Phonopy):
    """Test MLP features in Phonopy."""
    pytest.importorskip("pypolymlp", minversion="0.10.0")
    pytest.importorskip("symfc")

    atom_energies = {"Cl": -0.31144759, "Na": -0.24580545}
    params = PypolymlpParams(gtinv_maxl=(4, 4), atom_energies=atom_energies)

    ph_in = ph_nacl_rd
    ph = Phonopy(
        ph_in.unitcell,
        supercell_matrix=ph_in.supercell_matrix,
        primitive_matrix=ph_in.primitive_matrix,
        log_level=2,
    )
    ph.nac_params = ph_in.nac_params
    ph.mlp_dataset = copy.copy(ph_in.dataset)
    ph.develop_mlp(params=params)
    # ph.generate_displacements(distance=0.001, number_of_snapshots=2, random_seed=1)
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
    ph.evaluate_mlp()
    ph.produce_force_constants(fc_calculator="symfc")
    ph.run_mesh([2, 2, 2])
    freqs = ph.get_mesh_dict()["frequencies"]
    print(freqs.ravel().tolist())

    freqs_ref = [
        1.9479582873751362,
        1.9479582873751387,
        3.201448818645571,
        4.3184184471176446,
        4.318418447117647,
        6.962692062451476,
        2.7407481459662795,
        3.55742972698832,
        3.921702373852099,
        4.487456193539974,
        4.894982489029413,
        5.46152086052448,
    ]
    np.testing.assert_allclose(freqs.ravel(), freqs_ref, atol=1e-5)


def test_load_mlp_pypolymlp(ph_kcl: Phonopy):
    """Test load_mlp."""
    pytest.importorskip("pypolymlp", minversion="0.9.2")
    ph_kcl.load_mlp(cwd / ".." / "polymlp_KCL-120.yaml")
    ph_kcl.load_mlp(cwd / ".." / "polymlp_KCL-120.yaml.xz")


def test_Phonopy_calculator():
    """Test phonopy_load with phonopy_params.yaml."""
    ph_orig = phonopy.load(
        cwd / ".." / "phonopy_params_NaCl-fd.yaml.xz", produce_fc=False, log_level=2
    )
    ph = Phonopy(
        unitcell=ph_orig.unitcell,
        supercell_matrix=ph_orig.supercell_matrix,
        primitive_matrix=ph_orig.primitive_matrix,
    )
    assert ph.calculator is None
    assert ph.unit_conversion_factor == pytest.approx(15.6333023)

    with pytest.warns(DeprecationWarning):
        ph = Phonopy(
            unitcell=ph_orig.unitcell,
            supercell_matrix=ph_orig.supercell_matrix,
            primitive_matrix=ph_orig.primitive_matrix,
            factor=100,
        )
        assert ph.unit_conversion_factor == pytest.approx(100)


def test_Phonopy_calculator_QE():
    """Test phonopy_load with phonopy_params.yaml for QE."""
    ph_orig = phonopy.load(
        cwd / ".." / "phonopy_params_NaCl-QE.yaml.xz", produce_fc=False, log_level=2
    )
    ph = Phonopy(
        unitcell=ph_orig.unitcell,
        supercell_matrix=ph_orig.supercell_matrix,
        primitive_matrix=ph_orig.primitive_matrix,
        calculator="qe",
        set_factor_by_calculator=False,
    )
    assert ph.calculator == "qe"
    assert ph.unit_conversion_factor == pytest.approx(15.6333023)

    ph = Phonopy(
        unitcell=ph_orig.unitcell,
        supercell_matrix=ph_orig.supercell_matrix,
        primitive_matrix=ph_orig.primitive_matrix,
        calculator="qe",
        set_factor_by_calculator=True,
    )
    assert ph.unit_conversion_factor == pytest.approx(108.9707718)
