# SPDX-License-Identifier: BSD-3-Clause
"""Tests for generation of random displacements at finite temperatures."""

import os

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.random_displacements import RandomDisplacements
from phonopy.structure.atoms import PhonopyAtoms

current_dir = os.path.dirname(os.path.abspath(__file__))

randn_ii_TiPN3 = [
    -0.75205998,
    0.27277617,
    0.83138473,
    0.34503635,
    -0.97548523,
    0.78267442,
    -0.60242826,
    -0.25536472,
    0.90575698,
    0.20660722,
    0.37338047,
    -0.23586957,
    -0.79314925,
    -0.05509304,
    -0.72695366,
    -0.43095519,
    -0.22766689,
    -0.30966152,
    1.58896690,
    -1.38804088,
    1.24741741,
    -0.91095314,
    1.59402303,
    0.34787412,
    -0.35344703,
    -0.30253557,
    1.59663041,
    0.24723264,
    -0.95605159,
    0.93430119,
    0.03101086,
    0.82053802,
    -0.11366465,
    1.89223040,
    -0.07915884,
    0.16342100,
    0.33778021,
    0.71913179,
    0.20305406,
    1.16636417,
    -1.05671394,
    0.64985711,
    1.13683096,
    -0.09042930,
    -0.69500094,
    -0.17883372,
    1.23122338,
    0.18391431,
    -2.68475189,
    0.47078780,
    -0.92502772,
    0.27108019,
    -0.93544559,
    0.48289366,
    0.99005989,
    -0.60780100,
    -0.17010333,
    -0.21495777,
    -0.42092361,
    -0.30136512,
]
randn_ij_TiPN3 = [
    -0.06170102,
    -0.61992360,
    -0.15385570,
    0.61595751,
    1.74995758,
    -0.01277410,
    1.57894094,
    0.02803090,
    0.18967601,
    0.58580242,
    0.94803810,
    -0.53533249,
    -0.55944620,
    -0.58923952,
    1.00185600,
    0.10782162,
    0.74801460,
    -0.93669240,
    0.56579240,
    -0.44337235,
    0.40251768,
    0.44409582,
    0.27938228,
    0.54620289,
    0.15704113,
    -1.15980182,
    0.60455184,
    1.52619073,
    -1.40402257,
    0.12923401,
    0.15311305,
    1.78486834,
    0.94891279,
    0.76584774,
    0.31839150,
    0.96939954,
    -1.03083869,
    -1.51490811,
    -0.79035140,
    -1.43092023,
    -1.01136020,
    1.17697257,
    2.19987314,
    1.10584541,
    -0.79716970,
    -0.33445035,
    1.66966013,
    -0.29773735,
    -0.81565641,
    0.06708510,
    -0.21093234,
    0.63910228,
    -0.33365209,
    -0.17910169,
    0.84486735,
    0.73227838,
    -0.57086059,
    0.04706162,
    -2.15314754,
    -1.12680838,
    -0.52941257,
    -0.60067451,
    -1.57028855,
    0.15149039,
    2.03872260,
    0.36690557,
    1.42465078,
    2.12936953,
    1.13876513,
    2.63876826,
    0.77853690,
    -0.69479458,
    0.17494312,
    -1.21956080,
    0.24734145,
    -0.70420130,
    -1.01762657,
    1.69735535,
    1.02293971,
    0.33068771,
    -1.18381693,
    -0.73911188,
    0.88437257,
    -0.06829583,
    -0.45273303,
    -1.17146102,
    -0.66878199,
    0.21220457,
    -0.58618574,
    -2.17105050,
    -0.48690853,
    1.07327023,
    -1.76610782,
    -0.76953833,
    1.42340199,
    -0.65425765,
    1.82247852,
    -1.01847797,
    0.21697186,
    -1.41241488,
    -0.02819111,
    0.43165528,
    1.56489757,
    -0.75862053,
    -0.89771539,
    1.08170070,
    -1.40722818,
    1.10563205,
    0.29128808,
    -2.34011892,
    0.03318174,
    1.22442957,
    -0.70733216,
    -1.66160820,
    0.46448168,
    -0.08978977,
    -0.49194569,
    -1.06799459,
    -0.42687841,
    -1.15033246,
    0.39172712,
    -0.02453619,
    -0.71092205,
    -0.27827295,
    -0.74001233,
    -1.42921413,
    -0.66637440,
    0.36445006,
    0.09197221,
    -0.26774866,
    0.19212148,
    -0.07196675,
    -0.50970485,
    -0.06048711,
    -0.43353398,
    0.38235511,
    0.97993911,
    1.17115844,
    -0.25580411,
    0.69102351,
    0.31149217,
    0.22494472,
    -0.31922476,
    1.10494808,
    0.92246173,
    0.92187124,
    -0.03082644,
    0.19061205,
    0.21873457,
    -0.23614598,
    -0.57197880,
    -0.55070213,
    1.05419500,
    -0.70516847,
    -0.73828818,
    -2.17953225,
    0.87760855,
    -0.41778020,
    2.17132000,
    -0.66981628,
    0.94582619,
    0.30568614,
    -0.00290777,
    -0.39489324,
    -0.53653039,
    -0.29528960,
    -1.98492973,
    -0.42386292,
    -0.87808285,
    0.84105616,
    -0.51821701,
    0.46532344,
    -1.00646306,
    1.26105193,
    0.35928298,
    0.15480106,
    0.81593661,
    0.18777615,
    -0.17189708,
    0.11404128,
]
disp_ref_TiPN3 = [
    -0.0139921,
    -0.1467103,
    0.0499177,
    -0.0184652,
    -0.0457580,
    0.0311567,
    -0.0093055,
    -0.1038517,
    0.0025469,
    -0.0092331,
    0.0181665,
    0.0145632,
    -0.0044955,
    -0.1846546,
    -0.0269976,
    0.0024339,
    -0.1514289,
    -0.0735135,
    0.0236752,
    -0.1397663,
    -0.0021655,
    0.0191045,
    -0.1234406,
    0.0311259,
    -0.0133368,
    0.0703198,
    0.0771007,
    0.0024880,
    0.2986681,
    -0.0650089,
    -0.0070736,
    0.0826114,
    -0.0178861,
    -0.0481727,
    -0.0353204,
    -0.0300294,
    -0.0143637,
    0.1736886,
    -0.1347469,
    -0.0012858,
    0.1810298,
    0.0218296,
    -0.0221461,
    0.2600267,
    0.0703658,
    0.0366836,
    -0.0289361,
    0.0932440,
    0.0146079,
    0.0374171,
    0.0321373,
    -0.0629188,
    0.1846235,
    0.0955936,
    0.0439165,
    0.0893446,
    0.1069014,
    -0.0496009,
    0.0136494,
    -0.0928596,
    -0.0561736,
    0.1209682,
    -0.0429780,
    -0.0640210,
    0.1804612,
    -0.0984785,
    0.0283641,
    -0.0154683,
    0.0020529,
    -0.0975423,
    0.0227738,
    -0.0292111,
    0.0320202,
    -0.1137538,
    0.0222184,
    0.0690526,
    -0.1187491,
    -0.0167023,
    0.0737557,
    -0.0132887,
    0.0235017,
    0.0832655,
    -0.1574012,
    0.0370525,
    0.0533234,
    -0.1668163,
    -0.0075261,
    0.0372498,
    -0.0733970,
    -0.0512159,
    0.0162566,
    -0.1622803,
    0.0138448,
    0.0621854,
    -0.0602974,
    0.0401007,
    0.0810000,
    -0.0901605,
    -0.0445714,
    0.0574960,
    -0.0487999,
    0.0401881,
    0.1112084,
    -0.1593127,
    -0.0181409,
    0.0661456,
    -0.1572080,
    -0.0183305,
    -0.2571804,
    -0.1585012,
    -0.1714262,
    0.0174708,
    -0.2248008,
    -0.0946571,
    -0.0312199,
    -0.0175085,
    0.1391121,
    -0.0738157,
    -0.2307013,
    -0.1231036,
    -0.0260191,
    -0.0217891,
    0.0071777,
    -0.0149250,
    -0.1100150,
    0.0169939,
    -0.1318810,
    0.0022363,
    0.1161809,
    -0.0212155,
    0.0759783,
    -0.0128350,
    -0.0406390,
    0.0436901,
    -0.0580798,
    -0.0365576,
    0.0760328,
    -0.0837137,
    0.0525299,
    0.0915990,
    -0.0268440,
    -0.0035426,
    0.0502476,
    -0.0786706,
    0.1089748,
    -0.1040078,
    0.0820544,
    0.0008493,
    -0.1051912,
    -0.0401635,
    0.1453380,
    -0.1253290,
    0.0380263,
    0.0743620,
    -0.1016348,
    -0.0273094,
    -0.1378251,
    -0.1261827,
    -0.0154314,
    -0.1426802,
    -0.0669976,
    -0.0170440,
    -0.1604904,
    -0.1858899,
    0.1594560,
    -0.0662879,
    -0.1829222,
    0.1334110,
    -0.0583901,
    0.2756146,
    0.3073976,
    0.0150459,
    0.2327394,
    0.2643992,
    0.0025823,
    0.0880147,
    0.0072265,
    -0.0167039,
    0.1695736,
    0.1377619,
    -0.0424297,
    0.2198620,
    0.2125829,
    0.0638527,
    0.2284500,
    0.0274021,
    0.0390822,
    0.0831478,
    0.0531077,
    -0.0961539,
    0.0289731,
    -0.0319525,
    0.0900563,
    -0.0280840,
    0.0171844,
    0.0060835,
    -0.1595669,
    -0.0070878,
    0.0491180,
    -0.1390511,
    0.0201374,
    0.0879246,
    0.0395648,
    -0.0279471,
    0.0249554,
    -0.0753852,
    0.0114907,
    0.1018765,
    -0.1199073,
    -0.0265889,
    0.0608649,
    -0.1327127,
    0.0210577,
    -0.0375886,
    -0.0450452,
    0.0191296,
    0.0425393,
    0.3003195,
    -0.2376978,
    0.0071464,
    0.3127125,
    -0.1029075,
    0.0289461,
    0.0842798,
    0.0955425,
    0.1030041,
    0.1486068,
    -0.2807427,
    -0.0238278,
    0.1882260,
    -0.2696367,
    -0.1205425,
    0.2373104,
    -0.2575442,
    -0.0374932,
    -0.0249045,
    -0.0460490,
    -0.0025609,
    0.0519746,
    -0.0265196,
]

# The same draw under sampling_matrix="eigenvector", which needs the
# precalculated eigenvectors to be reproducible across eigensolvers.
disp_ref_TiPN3_eigenvector = [
    0.1618134,
    -0.0614143,
    0.0210274,
    0.0726223,
    0.0325882,
    0.0478187,
    0.0988512,
    -0.0519708,
    -0.0759353,
    0.0134498,
    0.1019672,
    -0.0161726,
    -0.0349990,
    -0.0032618,
    0.0049472,
    -0.0016998,
    0.0877175,
    -0.0671919,
    -0.1189687,
    -0.0306470,
    0.0901558,
    -0.1230101,
    -0.1056568,
    0.0658698,
    0.0167282,
    0.0168195,
    0.0637658,
    -0.0189956,
    0.0396138,
    -0.0490587,
    0.0554470,
    0.0679658,
    0.0418343,
    -0.0208550,
    -0.0245474,
    -0.0442203,
    0.0377164,
    0.0558982,
    -0.1068723,
    0.0262384,
    -0.0866143,
    0.0618299,
    -0.0069328,
    -0.0077085,
    0.0423974,
    0.0966033,
    0.0304852,
    -0.0193072,
    0.0454410,
    -0.0000874,
    -0.0822773,
    0.0889690,
    -0.0092456,
    -0.0505241,
    0.0164528,
    0.0367526,
    0.1032214,
    -0.0139953,
    0.0332637,
    -0.0355744,
    -0.0285934,
    0.0361651,
    0.0580788,
    -0.0279109,
    0.0788766,
    -0.0482856,
    -0.0005008,
    -0.0205448,
    0.0894570,
    -0.0080147,
    0.0389441,
    -0.0480408,
    -0.0545595,
    -0.0311157,
    0.0272869,
    -0.0283823,
    0.0497238,
    -0.0098941,
    0.0170338,
    -0.0176504,
    0.0233931,
    -0.0922070,
    -0.0214924,
    0.0530348,
    -0.0810012,
    -0.0022118,
    -0.0893545,
    -0.0182495,
    -0.0738738,
    0.0765532,
    -0.0383897,
    -0.0492228,
    -0.0508230,
    -0.0285300,
    -0.0257752,
    0.0496289,
    -0.0181072,
    0.0033779,
    -0.0564677,
    -0.0142723,
    -0.0523550,
    0.0444132,
    -0.0885254,
    -0.0405047,
    -0.0240664,
    -0.0796836,
    0.0497994,
    0.1500404,
    0.0702281,
    0.0268927,
    0.0532923,
    -0.0016054,
    -0.0102284,
    -0.0699401,
    -0.0513170,
    -0.0902195,
    -0.0688610,
    0.0505787,
    -0.0527843,
    -0.0368956,
    0.0320662,
    0.0272937,
    -0.1042032,
    0.0768927,
    0.0596951,
    -0.0163141,
    0.0844153,
    -0.0244247,
    -0.0549370,
    -0.0614501,
    -0.0891216,
    -0.0114040,
    -0.0077435,
    0.1430504,
    0.0008030,
    -0.0478228,
    0.1327940,
    -0.0120928,
    0.0138161,
    0.0325583,
    0.0726303,
    -0.0577982,
    -0.0178575,
    0.0590799,
    -0.0167005,
    -0.0581823,
    0.0490973,
    -0.1114379,
    -0.0711706,
    -0.0192567,
    -0.1040579,
    -0.0987688,
    -0.0074678,
    -0.1291350,
    0.0664635,
    -0.0007583,
    0.0136866,
    0.0127900,
    -0.1447172,
    0.0159992,
    0.0742667,
    -0.0827642,
    -0.0619132,
    -0.0640262,
    -0.0518244,
    0.1624519,
    0.0225123,
    -0.0042195,
    0.0598908,
    -0.0417625,
    -0.1999325,
    0.0663072,
    0.0085402,
    -0.0700757,
    0.0991456,
    0.0539809,
    0.1545221,
    0.1336583,
    -0.0235658,
    -0.2064598,
    -0.1142594,
    -0.0542608,
    -0.1011531,
    -0.1937924,
    -0.0129030,
    -0.1066291,
    0.0762996,
    0.0448192,
    0.1575854,
    -0.0130099,
    -0.0213280,
    -0.1098245,
    -0.0442808,
    0.0761112,
    0.0873370,
    -0.1255166,
    -0.0327616,
    0.0631406,
    0.0244151,
    0.0933865,
    -0.0103704,
    -0.0500082,
    -0.1108138,
    0.0471695,
    -0.0640202,
    0.0431027,
    -0.1355471,
    0.0317649,
    -0.1556919,
    -0.0200289,
    -0.0630049,
    -0.0157067,
    -0.0346294,
    0.0962192,
    0.0111355,
    -0.0167414,
    -0.0262084,
    -0.0974471,
    0.1875614,
    0.0402915,
    0.0088297,
    -0.0648625,
    -0.0475399,
    -0.0384461,
    0.2446050,
    -0.0515721,
    0.0348452,
    -0.0456382,
    0.0253257,
    -0.0505731,
    0.0246412,
    -0.0471122,
    -0.0212583,
    0.0771720,
    0.1058798,
    0.0065750,
    0.0318078,
    0.1027323,
    0.0543087,
    0.0294196,
]

# This is just a record of the histgram of NaCl 2x2x2 calculation.
# ph_nacl, number_of_snapshots=100000, temperature=500, nbins=101
disp_bins = np.linspace(0, 1, 101)
h_Na = [
    131,
    884,
    2439,
    4680,
    7670,
    11344,
    15492,
    20375,
    25447,
    31250,
    36869,
    42855,
    49056,
    54910,
    60854,
    66522,
    72224,
    77071,
    81986,
    86454,
    90295,
    93479,
    96001,
    97900,
    99001,
    99978,
    100033,
    99898,
    99001,
    97994,
    95998,
    93555,
    90986,
    87959,
    85552,
    81545,
    77635,
    74093,
    69226,
    65031,
    61404,
    57073,
    53036,
    49600,
    45360,
    41963,
    38282,
    34505,
    31644,
    28668,
    25958,
    23186,
    20969,
    18672,
    16734,
    14703,
    12958,
    11162,
    10009,
    8643,
    7596,
    6604,
    5821,
    4872,
    4199,
    3644,
    3089,
    2707,
    2200,
    1932,
    1553,
    1292,
    1101,
    929,
    751,
    623,
    543,
    424,
    384,
    303,
    244,
    197,
    146,
    132,
    92,
    85,
    65,
    43,
    50,
    35,
    34,
    18,
    23,
    13,
    11,
    12,
    5,
    11,
    4,
    3,
]
h_Cl = [
    178,
    1347,
    3693,
    7469,
    11766,
    17497,
    23524,
    30447,
    38050,
    45919,
    53889,
    62269,
    70103,
    77359,
    84780,
    92006,
    97850,
    103442,
    108053,
    111876,
    113442,
    115291,
    116398,
    116577,
    114803,
    112641,
    110651,
    108045,
    103658,
    99194,
    95266,
    89463,
    85213,
    78880,
    73102,
    68533,
    62582,
    57701,
    52733,
    47538,
    42725,
    38592,
    34362,
    30343,
    26928,
    23748,
    20681,
    18026,
    15670,
    13572,
    11624,
    10202,
    8609,
    7016,
    6147,
    5131,
    4403,
    3454,
    3007,
    2426,
    1927,
    1661,
    1284,
    1031,
    890,
    702,
    577,
    448,
    342,
    264,
    222,
    189,
    147,
    105,
    65,
    57,
    38,
    42,
    31,
    20,
    15,
    18,
    8,
    5,
    6,
    2,
    4,
    2,
    1,
    0,
    2,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


def _test_random_displacements_NaCl(ph_nacl):
    """Compare histgram of NaCl 2x2x2 displacements.

    answer gives the highest histgram bins of two atoms.

    """
    answer = (16, 14)
    argmaxs = _test_random_displacements(ph_nacl, answer, ntimes=20, d_max=0.8)
    assert argmaxs == answer


def _test_random_displacements_SnO2(ph_sno2):
    """Compare histgram of SnO2 2x2x3 displacements.

    answer gives the highest histgram bins of two atoms.

    """
    answer = (19, 14)
    argmaxs = _test_random_displacements(ph_sno2, answer, ntimes=30, d_max=0.3)
    assert argmaxs == answer


def test_random_displacements_Zr3N4(ph_zr3n4):
    """Test init and run for Zr3N4 1x1x1.

    This tests q-points only in A part.

    """
    ph = ph_zr3n4
    ph.init_random_displacements()
    ph.get_random_displacements_at_temperature(300, 10)


def _test_random_displacements(ph, answer, ntimes=30, d_max=1, nbins=51):
    hist = np.zeros((2, nbins - 1), dtype=int)
    for i in range(100):
        ph.generate_displacements(number_of_snapshots=5000, temperature=500)
        h_1, h_2 = _generate_random_displacements(ph, d_max, nbins)
        hist[0] += h_1
        hist[1] += h_2
        argmaxs = (np.argmax(hist[0]), np.argmax(hist[1]))
        print(hist[0])
        print(hist[1])
        print(i, argmaxs)
        if i >= ntimes and argmaxs == answer:
            break
    return argmaxs


def _generate_random_displacements(ph, d_max, nbins):
    _disp_bins = np.linspace(0, d_max, nbins)
    natom1 = ph.supercell.symbols.count(ph.supercell.symbols[0])
    d_1 = np.linalg.norm(ph.displacements[:, :natom1].reshape(-1, 3), axis=1)
    d_2 = np.linalg.norm(ph.displacements[:, natom1:].reshape(-1, 3), axis=1)
    h_1, bin_d = np.histogram(d_1, bins=_disp_bins)
    h_2, bin_d = np.histogram(d_2, bins=_disp_bins)
    # for line in h_1.reshape(-1, 10):
    #     print("".join(["%d, " % n for n in line]))
    # for line in h_2.reshape(-1, 10):
    #     print("".join(["%d, " % n for n in line]))
    return h_1, h_2


def test_random_displacements_all_atoms_TiPN3(ph_tipn3: Phonopy):
    """Test by fixed random numbers of np.random.normal.

    randn_ii and randn_ij were created by

        np.random.seed(seed=100)
        randn_ii = np.random.normal(size=(N_ii, 1, num_band))
        randn_ij = np.random.normal(size=(N_ij, 2, 1, num_band)).

    The eigenvectors are computed on the fly. Under the default
    sampling_matrix="symmetric" they enter only through
    E.diag(sigma).E^dagger, so the displacements do not depend on which
    eigenvectors the solver returned; see the eigenvector variant below,
    which does and therefore has to pin them.

    """
    ph = ph_tipn3
    rd = _get_random_displacements_all_atoms_TiPN3(ph)

    # for line in rd.u.ravel().reshape(-1, 6):
    #     print(("%.7f, " * 6) % tuple(line))

    data = np.array(disp_ref_TiPN3)
    np.testing.assert_allclose(data, rd.u.ravel(), atol=1e-5)

    rd.run_d2f()
    np.testing.assert_allclose(
        rd.force_constants, ph.force_constants, atol=1e-5, rtol=1e-5
    )

    rd.run_correlation_matrix(500)
    shape = (len(ph.supercell) * 3, len(ph.supercell) * 3)
    uu = np.transpose(rd.uu, axes=[0, 2, 1, 3]).reshape(shape)
    uu_inv = np.transpose(rd.uu_inv, axes=[0, 2, 1, 3]).reshape(shape)

    sqrt_masses = np.repeat(np.sqrt(ph.supercell.masses), 3)
    uu_bare = _mass_sand(uu, sqrt_masses)
    uu_inv_bare = np.linalg.pinv(uu_bare)
    _uu_inv = _mass_sand(uu_inv_bare, sqrt_masses)

    np.testing.assert_allclose(_uu_inv, uu_inv, atol=1e-5, rtol=1e-5)


def test_random_displacements_all_atoms_TiPN3_eigenvector(ph_tipn3: Phonopy):
    """Test sampling_matrix="eigenvector", the behaviour up to phonopy 4.4.

    Precalculated eigenvectors are used because this choice makes the
    displacements depend on them, and they depend in turn on the eigensolver
    (openblas, mkl, blis, netlib). Eigenvalues are equivalent over the
    eigensolvers, so those are calculated on the fly.

    """
    ph = ph_tipn3
    rd = _get_random_displacements_all_atoms_TiPN3(
        ph, sampling_matrix="eigenvector", pin_eigvecs=True
    )
    np.testing.assert_allclose(
        np.array(disp_ref_TiPN3_eigenvector), rd.u.ravel(), atol=1e-5
    )

    rd.run_d2f()
    np.testing.assert_allclose(
        rd.force_constants, ph.force_constants, atol=1e-5, rtol=1e-5
    )


def test_random_displacements_gauge_invariance_TiPN3(ph_tipn3: Phonopy):
    """The displacements must not depend on the eigenvector convention.

    Eigenvectors are fixed only up to a rotation inside a degenerate subspace
    and, at q != -q, a phase per band. Neither has physical content.
    sampling_matrix="symmetric" contracts them as E.diag(sigma).E^dagger and
    is therefore unaffected; "eigenvector" is not, which is what makes this a
    test rather than a tautology.

    """
    ph = ph_tipn3
    rng = np.random.default_rng(0)

    def _regauge(rd: RandomDisplacements) -> None:
        """Rotate degenerate eigenvectors and rephase the complex ones."""
        for eigvals, eigvecs in (
            (rd._eigvals_ii, rd._eigvecs_ii),
            (rd._eigvals_ij, rd._eigvecs_ij),
        ):
            for w, v in zip(eigvals, eigvecs, strict=True):
                order = np.argsort(w)
                start = 0
                for k in range(1, len(order) + 1):
                    if k == len(order) or abs(w[order[k]] - w[order[start]]) > 1e-8:
                        group = order[start:k]
                        if len(group) >= 2:
                            c, s = np.cos(0.7), np.sin(0.7)
                            a = v[:, group[0]].copy()
                            b = v[:, group[1]].copy()
                            v[:, group[0]] = c * a - s * b
                            v[:, group[1]] = s * a + c * b
                        start = k
                if np.iscomplexobj(v):
                    v *= np.exp(1j * rng.uniform(0, 2 * np.pi, v.shape[1]))

    for sampling_matrix, invariant in (("symmetric", True), ("eigenvector", False)):
        plain = _get_random_displacements_all_atoms_TiPN3(
            ph, sampling_matrix=sampling_matrix
        )
        regauged = _get_random_displacements_all_atoms_TiPN3(
            ph, sampling_matrix=sampling_matrix, regauge=_regauge
        )
        difference = np.abs(plain.u - regauged.u).max()
        if invariant:
            assert difference < 1e-10
        else:
            assert difference > 1e-3


def test_random_displacements_all_atoms_TiPN3_max_distance(ph_tipn3):
    """Test max_distance."""
    rd = _get_random_displacements_all_atoms_TiPN3(ph_tipn3)
    n_gt_max = (np.linalg.norm(rd.u.reshape(-1, 3), axis=1) > 0.2).sum()
    assert n_gt_max == 20
    rd = _get_random_displacements_all_atoms_TiPN3(ph_tipn3, max_distance=0.2)
    distances = np.linalg.norm(rd.u.reshape(-1, 3), axis=1)
    distances_gt_max = np.extract(distances > 0.2 - 1e-5, distances)
    assert len(distances_gt_max) == 20
    np.testing.assert_almost_equal(distances_gt_max, 0.2)


def _get_random_displacements_all_atoms_TiPN3(
    ph: Phonopy,
    max_distance=None,
    sampling_matrix=None,
    pin_eigvecs: bool = False,
    regauge=None,
) -> RandomDisplacements:
    """Return a RandomDisplacements driven by the fixed normals above.

    pin_eigvecs replaces the eigenvectors by precalculated ones. That is
    needed only for sampling_matrix="eigenvector", whose displacements depend
    on which eigenvectors the solver returned and so on the LAPACK
    implementation. The default "symmetric" does not, so it can use the
    eigenvectors computed on the fly.

    regauge, if given, is called on the instance to replace the eigenvectors
    by an equally valid set before the displacements are drawn.

    """
    rd = RandomDisplacements(
        ph.supercell,
        ph.primitive,
        ph.force_constants,
        max_distance=max_distance,
        sampling_matrix=sampling_matrix,
    )
    num_band = len(ph.primitive) * 3
    N = len(ph.supercell) // len(ph.primitive)
    # N = N_ii + N_ij * 2
    # len(rd.qpoints) = N_ii + N_ij
    N_ij = N - len(rd.qpoints)
    N_ii = N - N_ij * 2
    shape_ii = (N_ii, 1, num_band)
    shape_ij = (N_ij, 2, 1, num_band)
    randn_ii = np.reshape(randn_ii_TiPN3, shape_ii)
    randn_ij = np.reshape(randn_ij_TiPN3, shape_ij)

    if pin_eigvecs:
        rd._eigvecs_ii = np.reshape(
            np.loadtxt(os.path.join(current_dir, "eigvecs_ii_TiPN3.txt")),
            (N_ii, num_band, num_band),
        )
        rd._eigvecs_ij = np.reshape(
            np.loadtxt(
                os.path.join(current_dir, "eigvecs_ij_TiPN3.txt"), dtype=complex
            ),
            (N_ij, num_band, num_band),
        )
    if regauge is not None:
        regauge(rd)
    rd.run(500, randn=(randn_ii, randn_ij))
    return rd


@pytest.mark.parametrize("is_plusminus", [True, False])
def test_tio2_random_disp_plusminus(ph_tio2: Phonopy, is_plusminus: bool):
    """Test random plus-minus displacements of TiO2.

    Note
    ----
    Displacements of last 4 supercells are minus of those of first 4 supercells.

    """
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [72, 0.01, 0.0, 0.0],
        [72, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_ref, atol=1e-8)

    # Use an independent instance: generate_displacements replaces the
    # dataset, which also invalidates force constants, so the
    # session-scoped fixture must not be mutated.
    ph = ph_tio2.replicate()
    ph.nac_params = ph_tio2.nac_params
    ph.force_constants = ph_tio2.force_constants
    ph.generate_displacements(
        number_of_snapshots=4,
        distance=0.03,
        is_plusminus=is_plusminus,
        temperature=300,
    )
    d = ph.displacements
    if is_plusminus:
        assert len(d) == 8
        np.testing.assert_allclose(d[:4], -d[4:], atol=1e-8)
    else:
        assert len(d) == 4


def test_random_displacements_persist_after_temperature_generation(
    ph_tio2: Phonopy,
):
    """random_displacements must survive temperature-based generation.

    generate_displacements assigns a new dataset, which invalidates
    derived quantities. The finite-temperature sampler stored in
    random_displacements must not be cleared by that assignment, so its
    q-points, integrated modes, and frequencies remain available for
    reporting.

    """
    # Use an independent instance: generate_displacements replaces the
    # dataset, which also invalidates force constants, so the
    # session-scoped fixture must not be mutated.
    ph = ph_tio2.replicate()
    ph.nac_params = ph_tio2.nac_params
    ph.force_constants = ph_tio2.force_constants
    ph.generate_displacements(number_of_snapshots=4, temperature=300)

    assert ph.random_displacements is not None
    assert ph.random_displacements.qpoints is not None
    assert ph.random_displacements.integrated_modes is not None
    assert ph.random_displacements.frequencies is not None


def test_treat_imaginary_modes(ph_srtio3: Phonopy):
    """Test imaginary mode treatment of force constants.

    RandomDisplacements.treat_imaginary_modes method modified imaginary
    phonon frequenceis to have read-positive frequences.

    """
    ph = ph_srtio3
    rd = RandomDisplacements(ph.supercell, ph.primitive, ph.force_constants)
    # for freqs in (rd.frequencies[0], rd.frequencies[-1]):
    #     print(", ".join([f"{v:10.7f}" for v in freqs]))
    ref0 = [
        -2.3769150,
        -2.3769150,
        -2.3769150,
        -0.0000003,
        -0.0000003,
        -0.0000001,
        4.6902115,
        4.6902115,
        4.6902115,
        6.7590219,
        6.7590219,
        6.7590219,
        16.0075351,
        16.0075351,
        16.0075351,
    ]
    ref13 = [
        3.2707508,
        3.3132392,
        3.4395550,
        3.4395550,
        3.6676862,
        3.6676862,
        10.7490284,
        10.7970960,
        10.7970960,
        12.0900533,
        12.0900533,
        13.8508135,
        15.0638793,
        15.0638793,
        24.6446671,
    ]
    np.testing.assert_allclose(ref0, rd.frequencies[0], atol=1e-5)
    np.testing.assert_allclose(ref13, rd.frequencies[-1], atol=1e-5)

    rd.treat_imaginary_modes()
    # for freqs in (rd.frequencies[0], rd.frequencies[-1]):
    #     print(", ".join([f"{v:10.7f}" for v in freqs]))
    ref0 = [
        2.3769150,
        2.3769150,
        2.3769150,
        0.0000003,
        0.0000003,
        0.0000001,
        4.6902115,
        4.6902115,
        4.6902115,
        6.7590219,
        6.7590219,
        6.7590219,
        16.0075351,
        16.0075351,
        16.0075351,
    ]
    ref13 = [
        3.2707508,
        3.3132392,
        3.4395550,
        3.4395550,
        3.6676862,
        3.6676862,
        10.7490284,
        10.7970960,
        10.7970960,
        12.0900533,
        12.0900533,
        13.8508135,
        15.0638793,
        15.0638793,
        24.6446671,
    ]
    np.testing.assert_allclose(ref0, rd.frequencies[0], atol=1e-5)
    np.testing.assert_allclose(ref13, rd.frequencies[-1], atol=1e-5)

    # Test frequency shifts
    rd.treat_imaginary_modes(freq_to=3)
    # for freqs in (rd.frequencies[0], rd.frequencies[-1]):
    #     print(", ".join([f"{v:10.7f}" for v in freqs]))
    ref0 = [
        3.3769150,
        3.3769150,
        3.3769150,
        0.0000003,
        0.0000003,
        0.0000001,
        4.6902115,
        4.6902115,
        4.6902115,
        6.7590219,
        6.7590219,
        6.7590219,
        16.0075351,
        16.0075351,
        16.0075351,
    ]
    ref13 = [
        3.2707508,
        3.3132392,
        3.4395550,
        3.4395550,
        3.6676862,
        3.6676862,
        10.7490284,
        10.7970960,
        10.7970960,
        12.0900533,
        12.0900533,
        13.8508135,
        15.0638793,
        15.0638793,
        24.6446671,
    ]
    np.testing.assert_allclose(ref0, rd.frequencies[0], atol=1e-5)
    np.testing.assert_allclose(ref13, rd.frequencies[-1], atol=1e-5)


def _get_random_displacements_no_ij() -> RandomDisplacements:
    """Return a RandomDisplacements whose commensurate points have no ij pairs.

    A diagonal 2x2x2 supercell of a simple-cubic primitive cell places every
    commensurate point at a TRIM point (fractional coordinate 0 or 1/2), so all
    points are self-conjugate. This exercises the ``self._ij`` empty branch of
    the ``frequencies`` setter/getter.

    """
    cell = PhonopyAtoms(
        symbols=["H"], cell=np.eye(3) * 3.0, scaled_positions=[[0, 0, 0]]
    )
    ph = Phonopy(
        cell, supercell_matrix=np.eye(3, dtype=int) * 2, primitive_matrix="auto"
    )
    nsc = len(ph.supercell)
    fc = np.zeros((nsc, nsc, 3, 3))
    for i in range(nsc):
        fc[i, i] = np.eye(3) * 2.0
    return RandomDisplacements(ph.supercell, ph.primitive, fc)


def test_frequencies_setter_roundtrip_with_ij(ph_tipn3: Phonopy):
    """Setting frequencies back unchanged must preserve them (ij pairs present)."""
    ph = ph_tipn3
    rd = RandomDisplacements(ph.supercell, ph.primitive, ph.force_constants)
    assert len(rd._ij) > 0
    freqs = rd.frequencies.copy()
    rd.frequencies = freqs
    np.testing.assert_allclose(rd.frequencies, freqs, atol=1e-8)

    # Modified frequencies are reflected by the getter.
    shifted = freqs + 1.0
    rd.frequencies = shifted
    np.testing.assert_allclose(rd.frequencies, shifted, atol=1e-8)


def test_frequencies_setter_roundtrip_without_ij():
    """Setting frequencies back unchanged must preserve them (no ij pairs).

    Regression test: the setter used to assume ij pairs always exist and
    raised when ``self._ij`` was empty.

    """
    rd = _get_random_displacements_no_ij()
    assert len(rd._ij) == 0
    freqs = rd.frequencies.copy()
    rd.frequencies = freqs
    np.testing.assert_allclose(rd.frequencies, freqs, atol=1e-8)

    shifted = freqs + 1.0
    rd.frequencies = shifted
    np.testing.assert_allclose(rd.frequencies, shifted, atol=1e-8)


def test_frequencies_setter_wrong_dimension():
    """The setter rejects frequencies of the wrong length in both branches."""
    rd = _get_random_displacements_no_ij()
    with pytest.raises(RuntimeError):
        rd.frequencies = rd.frequencies.ravel()[:-1]


def _mass_sand(matrix, mass):
    return ((matrix * mass).T * mass).T
