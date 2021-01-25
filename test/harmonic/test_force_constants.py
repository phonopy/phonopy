import numpy as np


fc_1_10_ref = [-0.037549, 0.000000, 0.000000,
               0.000000, 0.002415, -0.001746,
               0.000000, -0.001746, 0.002415]

fc_1_10_nofcsym_ref = [-0.005051, 0.000000, 0.000000,
                       0.000000, 0.094457, 0.000000,
                       0.000000, 0.000000, -0.020424]

fc_1_10_compact_fcsym_ref = [-0.004481, 0.000000, 0.000000,
                             0.000000, 0.095230, 0.000000,
                             0.000000, 0.000000, -0.019893]


def test_fc(ph_nacl):
    fc_1_10 = ph_nacl.force_constants[1, 10]
    # print("".join(["%f, " % v for v in fc_1_10.ravel()]))
    np.testing.assert_allclose(fc_1_10.ravel(), fc_1_10_ref, atol=1e-5)


def test_fc_nofcsym(ph_nacl_nofcsym):
    fc_1_10 = ph_nacl_nofcsym.force_constants[1, 10]
    # print("".join(["%f, " % v for v in fc_1_10.ravel()]))
    np.testing.assert_allclose(fc_1_10.ravel(), fc_1_10_nofcsym_ref, atol=1e-5)


def test_fc_compact_fcsym(ph_nacl_compact_fcsym):
    fc_1_10 = ph_nacl_compact_fcsym.force_constants[1, 10]
    # print("".join(["%f, " % v for v in fc_1_10.ravel()]))
    np.testing.assert_allclose(
        fc_1_10.ravel(), fc_1_10_compact_fcsym_ref, atol=1e-5)
