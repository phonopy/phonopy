import cupy
import numpy as np

from phonopy.acc import vec3_numba as vec3
from phonopy.acc.numba_imports import cuda, exp, numba, rsqrt, sincospi, sqrt
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixNAC
from phonopy.structure.cells import sparse_to_dense_svecs

Q_ZERO_TOLERANCE = 1e-5
ZERO_TOLERANCE = 1e-12


def _extract_params(dm):
    """Port from dynamical_matrix.py to use device arrays."""
    svecs, multi = dm.primitive.get_smallest_vectors()
    if dm.primitive.store_dense_svecs:
        _svecs = svecs
        _multi = multi
    else:
        _svecs, _multi = sparse_to_dense_svecs(svecs, multi)

    masses = dm.primitive.masses
    rec_lattice = np.array(np.linalg.inv(dm.primitive.cell), dtype="double", order="C")
    positions = dm.primitive.positions
    if isinstance(dm, DynamicalMatrixNAC):
        born = dm.born
        nac_factor = float(dm.nac_factor)
        dielectric = dm.dielectric_constant
    else:
        born = np.zeros(9)  # dummy variable
        nac_factor = 0.0  # dummy variable
        dielectric = np.zeros(9)  # dummy variable

    return (
        cuda.to_device(_svecs),
        cuda.to_device(np.ascontiguousarray(_multi.transpose(1, 0, 2))),
        cuda.to_device(masses),
        cuda.to_device(rec_lattice),
        cuda.to_device(positions),
        cuda.to_device(born),
        nac_factor,
        cuda.to_device(dielectric),
    )


def _get_gonze_nac_params(dm):
    """Port from dynamical_matrix.py to use device arrays."""
    gonze_nac_dataset = dm.Gonze_nac_dataset
    if gonze_nac_dataset[0] is None:
        dm.make_Gonze_nac_dataset()
        gonze_nac_dataset = dm.Gonze_nac_dataset
    (
        gonze_fc,  # fc where the dipole-diple contribution is removed.
        dd_q0,  # second term of dipole-dipole expression.
        G_cutoff,  # Cutoff radius in reciprocal space. This will not be used.
        G_list,  # List of G points where d-d interactions are integrated.
        Lambda,
    ) = gonze_nac_dataset  # Convergence parameter
    return (
        cupy.asarray(gonze_fc),
        cupy.asarray(dd_q0),
        G_cutoff,
        cupy.asarray(G_list),
        Lambda,
    )


def _get_fc_elements_mapping(dm, fc):
    """Port from dynamical_matrix.py to use device arrays."""
    p2s_map = dm.primitive.p2s_map
    s2p_map = dm.primitive.s2p_map
    if fc.shape[0] == fc.shape[1]:  # full fc
        return cupy.array(p2s_map, dtype="int64"), cupy.array(s2p_map, dtype="int64")
    else:  # compact fc
        primitive = dm.primitive
        p2p_map = primitive.p2p_map
        s2pp_map = cupy.array(
            [p2p_map[s2p_map[i]] for i in range(len(s2p_map))], dtype="int64"
        )
        return cupy.arange(len(p2s_map), dtype="int64"), s2pp_map


@cuda.jit(device=True, inline=True)
def get_q_cart(q, rec_lat, q_cart):
    """Port from c/dynmat.c."""
    for i in range(3):
        q_cart[i] = 0.0
        for j in range(3):
            q_cart[i] += rec_lat[i, j] * q[j]
    return


@cuda.jit(device=True)
def dym_get_charge_sum(charge_sum, num_patom, factor, q_cart, born, q_born):
    """Port from c/dynmat.c."""
    for i in range(num_patom):
        for j in range(3):
            q_born[i, j] = 0.0
            for k in range(3):
                q_born[i, j] += q_cart[k] * born[i, k, j]

    for i in range(num_patom):
        for j in range(num_patom):
            for a in range(3):
                for b in range(3):
                    charge_sum[i, j, a, b] = q_born[i, a] * q_born[j, b] * factor
    return


@cuda.jit(device=True)
def get_dynmat_wang(
    dynmat,
    charge_sum,
    q_born,
    q,
    fc,
    svecs,
    multi,
    masses,
    p2s,
    s2p,
    born,
    dielectric,
    rec_lat,
    q_dir,
    q_dir_cart,
    nac_factor,
    hermitianize,
):
    """Port from c/dynmat.c."""
    n = len(s2p) / len(p2s)
    q_cart = cuda.local.array(3, dtype=numba.float64)
    get_q_cart(q, rec_lat, q_cart)
    if vec3.norm(q_cart) < Q_ZERO_TOLERANCE:
        if vec3.norm(q_dir) >= ZERO_TOLERANCE:
            dielectric_part = get_dielectric_part(q_dir_cart, dielectric)
            factor = nac_factor / n / dielectric_part
            dym_get_charge_sum(charge_sum, len(p2s), factor, q_dir_cart, born, q_born)
            get_dynamical_matrix_at_q(
                dynmat, fc, q, svecs, multi, masses, s2p, p2s, charge_sum, hermitianize
            )
        else:
            get_dynamical_matrix_at_q(
                dynmat, fc, q, svecs, multi, masses, s2p, p2s, None, hermitianize
            )
    else:
        dielectric_part = get_dielectric_part(q_cart, dielectric)
        factor = nac_factor / n / dielectric_part
        dym_get_charge_sum(charge_sum, len(p2s), factor, q_cart, born, q_born)
        get_dynamical_matrix_at_q(
            dynmat, fc, q, svecs, multi, masses, s2p, p2s, charge_sum, hermitianize
        )
    return


@cuda.jit(device=True)
def get_dm(dm, fc, q, svecs, multi, charge_sum, i, j):
    """Port from c/dynmat.c."""
    m_pair = multi[0]
    adrs = multi[1]
    csphase = complex(0)
    for l in range(m_pair):
        phase = 0.0
        for n in range(3):
            phase += q[n] * svecs[adrs + l, n]
        _sincos = sincospi(2 * phase)
        csphase += complex(_sincos[1], _sincos[0])
    csphase /= m_pair

    for l in range(3):
        for m in range(3):
            if charge_sum is not None:
                fc_elem = fc[l, m] + charge_sum[i, j, l, m]
            else:
                fc_elem = fc[l, m]
            dm[l, m] += fc_elem * csphase

    return


@cuda.jit(device=True)
def get_dynmat_ij(dynmat, fc, q, svecs, multi, masses, s2p, p2s, charge_sum, i, j):
    """Port from c/dynmat.c."""
    mass_rsqrt = rsqrt(masses[i] * masses[j])
    dm = cuda.local.array((3, 3), dtype=numba.complex128)
    for l in range(3):
        for m in range(3):
            dm[l, m] = 0.0

    for k in range(len(s2p)):
        if s2p[k] != p2s[j]:
            continue
        get_dm(dm, fc[p2s[i], k, :, :], q, svecs, multi[i, k, :], charge_sum, i, j)

    for k in range(3):
        for l in range(3):
            dynmat[i * 3 + k, j * 3 + l] = dm[k, l] * mass_rsqrt
    return


@cuda.jit(device=True, inline=True)
def make_Hermitian(mat, num_band):
    """Port from c/dynmat.c."""
    for i in range(num_band):
        for j in range(i, num_band):
            mat[i, j] += mat[j, i].conjugate()
            mat[i, j] /= 2
            mat[j, i] = mat[i, j].conjugate()
    return


@cuda.jit(device=True)
def get_dynamical_matrix_at_q(
    dynmat, fc, q, svecs, multi, masses, s2p, p2s, charge_sum, hermitianize
):
    """Port from c/dynmat.c."""
    num_patom = len(p2s)
    for i in range(num_patom):
        for j in range(num_patom):
            get_dynmat_ij(
                dynmat, fc, q, svecs, multi, masses, s2p, p2s, charge_sum, i, j
            )
    if hermitianize:
        make_Hermitian(dynmat, num_patom * 3)
    return


@cuda.jit(device=True, inline=True)
def get_dielectric_part(q_cart, dielectric):
    """Port from c/dynmat.c."""
    lsum = 0.0
    for i in range(3):
        for j in range(3):
            lsum += q_cart[i] * dielectric[i, j] * q_cart[j]
    return lsum


@cuda.jit(device=True, inline=True)
def get_dd_at_g(dd_part, i, j, G, num_patom, pos, KK):
    """Port from c/dynmat.c."""
    phase = 0.0
    for k in range(3):
        phase += (pos[i, k] - pos[j, k]) * G[k]
    _sincos = sincospi(2 * phase)
    cs_phase = complex(_sincos[1], _sincos[0])

    for k in range(3):
        for l in range(3):
            dd_part[i * 3 + k, j * 3 + l] += KK[k, l] * cs_phase
    return


@cuda.jit(device=True)
def get_dd(dd_part, G_list, num_patom, q_cart, q_dir_cart, dielectric, pos, Lambda):
    """Port from c/dynmat.c."""
    KK = cuda.local.array((3, 3), dtype=numba.float64)
    L2 = 4 * Lambda * Lambda
    q_K = cuda.local.array(3, dtype=numba.float64)
    for g in range(len(G_list)):
        norm = 0.0
        for i in range(3):
            for j in range(3):
                KK[i, j] = 0.0
        for i in range(3):
            q_K[i] = G_list[g, i] + q_cart[i]
            norm += q_K[i] * q_K[i]
        if sqrt(norm) < Q_ZERO_TOLERANCE:
            norm = 1
            if vec3.norm(q_dir_cart) >= ZERO_TOLERANCE:
                dielectric_part = get_dielectric_part(q_dir_cart, dielectric)
                for i in range(3):
                    for j in range(3):
                        KK[i, j] = q_dir_cart[i] * q_dir_cart[j] / dielectric_part
        else:
            dielectric_part = get_dielectric_part(q_K, dielectric)
            for i in range(3):
                for j in range(3):
                    KK[i, j] = (q_K[i] * q_K[j] / dielectric_part) * exp(
                        -dielectric_part / L2
                    )
        for i in range(num_patom):
            for j in range(num_patom):
                get_dd_at_g(dd_part, i, j, G_list[g], num_patom, pos, KK)

    return


@cuda.jit(device=True, inline=True)
def multiply_borns_at_ij(dd, i, j, dd_in, num_patom, born):
    """Port from c/dynmat.c."""
    for k in range(3):
        for l in range(3):
            for m in range(3):
                for n in range(3):
                    zz = born[i, m, k] * born[j, n, l]
                    dd[i * 3 + k, j * 3 + l] += dd_in[i * 3 + m, j * 3 + n] * zz


@cuda.jit(device=True, inline=True)
def multiply_borns(dd, dd_in, num_patom, born):
    """Port from c/dynmat.c."""
    for i in range(num_patom):
        for j in range(num_patom):
            multiply_borns_at_ij(dd, i, j, dd_in, num_patom, born)


@cuda.jit(device=True)
def dym_get_recip_dipole_dipole(
    dd,
    dd_tmp,
    dd_q0,
    G_list,
    num_patom,
    q_cart,
    q_dir_cart,
    born,
    dielectric,
    pos,
    factor,
    Lambda,
):
    """Port from c/dynmat.c."""
    get_dd(dd_tmp, G_list, num_patom, q_cart, q_dir_cart, dielectric, pos, Lambda)
    multiply_borns(dd, dd_tmp, num_patom, born)
    for i in range(num_patom):
        for k in range(3):
            for l in range(3):
                dd[i * 3 + k, i * 3 + l] -= dd_q0[i, k, l]
    for i in range(num_patom * 3):
        for j in range(num_patom * 3):
            dd[i, j] *= factor
    return


@cuda.jit(device=True)
def add_dynmat_dd_at_q(
    dynmat,
    dd,
    dd_tmp,
    q,
    fc,
    positions,
    num_patom,
    masses,
    born,
    dielectric,
    rec_lat,
    q_dir_cart,
    nac_factor,
    dd_q0,
    G_list,
    Lambda,
):
    """Port from c/dynmat.c."""
    q_cart = cuda.local.array(3, dtype=numba.float64)
    get_q_cart(q, rec_lat, q_cart)
    for k in range(num_patom * 3):
        for l in range(num_patom * 3):
            dd[k, l] = 0.0
            dd_tmp[k, l] = 0.0

    dym_get_recip_dipole_dipole(
        dd,
        dd_tmp,
        dd_q0,
        G_list,
        num_patom,
        q_cart,
        q_dir_cart,
        born,
        dielectric,
        positions,
        nac_factor,
        Lambda,
    )

    for i in range(num_patom):
        for j in range(num_patom):
            mm = rsqrt(masses[i] * masses[j])
            for k in range(3):
                for l in range(3):
                    dynmat[i * 3 + k, j * 3 + l] += dd[i * 3 + k, j * 3 + l] * mm
    return


@cuda.jit(device=True)
def _run_dynamical_matrix(dynmat, svecs, multi, masses, fc, p2s, s2p, qpoint):
    """Non-NAC version."""
    get_dynamical_matrix_at_q(
        dynmat, fc, qpoint, svecs, multi, masses, s2p, p2s, None, False
    )
    return


@cuda.jit
def _dm_kernel(dynmat, dm_svecs, dm_multi, dm_masses, dm_fc, p2s, s2p, qpoints):
    """Kernel function for non-NAC dynamical matrix."""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= len(qpoints):
        return
    q = qpoints[i]
    _run_dynamical_matrix(
        dynmat[i, :, :], dm_svecs, dm_multi, dm_masses, dm_fc, p2s, s2p, q
    )
    return


@cuda.jit(device=True)
def dynamical_matrices_with_dd_over_qpoints(
    dynmat,
    dd,
    dd_tmp,
    qpoint,
    fc,
    svecs,
    multi,
    positions,
    masses,
    s2p,
    p2s,
    nac_q_dir,
    q_dir_cart,
    born,
    dielectric,
    rec_lat,
    nac_factor,
    do_dd,
    dd_q0,
    G_list,
    Lambda,
    hermitianize,
):
    """Port of c/dynmat.c:dym_dynamical_matrices_with_dd_openmp_over_qpoints."""
    get_dynamical_matrix_at_q(
        dynmat, fc, qpoint, svecs, multi, masses, s2p, p2s, None, hermitianize
    )
    if do_dd:
        add_dynmat_dd_at_q(
            dynmat,
            dd,
            dd_tmp,
            qpoint,
            fc,
            positions,
            len(p2s),
            masses,
            born,
            dielectric,
            rec_lat,
            q_dir_cart,
            nac_factor,
            dd_q0,
            G_list,
            Lambda,
        )
    return


@cuda.jit(device=True)
def _run_dynamical_matrix_nac(
    i,
    dynmat,
    dd,
    dd_tmp,
    charge_sum,
    q_born,
    dm_svecs,
    dm_multi,
    dm_masses,
    dm_rec_lat,
    dm_positions,
    dm_born,
    dm_nac_factor,
    dm_dielectric,
    dm_fc,
    dd_q0,
    G_list,
    Lambda,
    p2s,
    s2p,
    qpoint,
    q_dir,
    hermitianize,
    use_Wang_NAC,
):
    """Port of NAC portion of dynamical_matrix.py:run_dynamical_matrix_solver_c."""
    fc = dm_fc
    q_dir_cart = cuda.local.array(3, dtype=numba.float64)
    get_q_cart(q_dir, dm_rec_lat, q_dir_cart)
    if use_Wang_NAC:
        get_dynmat_wang(
            dynmat,
            charge_sum[i],
            q_born[i],
            qpoint,
            fc,
            dm_svecs,
            dm_multi,
            dm_masses,
            p2s,
            s2p,
            dm_born,
            dm_dielectric,
            dm_rec_lat,
            q_dir,
            q_dir_cart,
            dm_nac_factor,
            hermitianize,
        )
    else:
        do_dd = True
        dynamical_matrices_with_dd_over_qpoints(
            dynmat,
            dd[i],
            dd_tmp[i],
            qpoint,
            fc,
            dm_svecs,
            dm_multi,
            dm_positions,
            dm_masses,
            s2p,
            p2s,
            q_dir,
            q_dir_cart,
            dm_born,
            dm_dielectric,
            dm_rec_lat,
            dm_nac_factor,
            do_dd,
            dd_q0,
            G_list,
            Lambda,
            hermitianize,
        )
    return


@cuda.jit
def _dm_kernel_nac(
    dynmat,
    dd,
    dd_tmp,
    charge_sum,
    q_born,
    dm_svecs,
    dm_multi,
    dm_masses,
    dm_rec_lat,
    dm_positions,
    dm_born,
    dm_nac_factor,
    dm_dielectric,
    dm_fc,
    dd_q0,
    G_list,
    Lambda,
    p2s,
    s2p,
    qpoints,
    q_dir,
    hermitianize,
    use_Wang_NAC,
):
    """Kernel function for NAC dynamical matrix."""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= len(qpoints):
        return
    q = qpoints[i]
    _run_dynamical_matrix_nac(
        i,
        dynmat[i, :, :],
        dd,
        dd_tmp,
        charge_sum,
        q_born,
        dm_svecs,
        dm_multi,
        dm_masses,
        dm_rec_lat,
        dm_positions,
        dm_born,
        dm_nac_factor,
        dm_dielectric,
        dm_fc,
        dd_q0,
        G_list,
        Lambda,
        p2s,
        s2p,
        q,
        q_dir,
        hermitianize,
        use_Wang_NAC,
    )
    return


class DynMatWorkspace:
    """Class to hold reusable data for dynamical matrix calculations."""

    def __init__(self, dynamical_matrix, with_eigenvectors=False):
        self.with_eigenvectors = with_eigenvectors
        self.dm = dynamical_matrix
        (
            self.svecs,
            self.multi,
            self.masses,
            self.rec_lattice,  # column vectors
            self.positions,  # primitive cell positions
            self.born,
            self.nac_factor,
            self.dielectric,
        ) = _extract_params(self.dm)

        _p2s, _s2p = _get_fc_elements_mapping(self.dm, self.dm.force_constants)
        self.p2s = _p2s
        self.s2p = _s2p

        self.dm_fc = cupy.asarray(self.dm.force_constants)

        self.dynmat = cupy.zeros(0, dtype=np.complex128)

        if isinstance(self.dm, DynamicalMatrixNAC):
            self.rec_lat = np.array(np.linalg.inv(self.dm.primitive.cell))

            if self.dm.nac_method == "gonze":
                (
                    # fc where the dipole-diple contribution is removed.
                    gonze_fc,
                    # second term of dipole-dipole expression.
                    self.dd_q0,
                    # Cutoff radius in reciprocal space. This will not be used.
                    G_cutoff,
                    # List of G points where d-d interactions are integrated.
                    self.G_list,
                    self.Lambda,
                ) = _get_gonze_nac_params(self.dm)
                self.dm_fc = gonze_fc
                self.use_Wang_NAC = False
            elif self.dm.nac_method == "wang":
                self.use_Wang_NAC = True
                self.dd_q0 = cupy.array((0, 0, 0))
                self.G_list = cupy.array((0, 0))
                self.Lambda = 0.0

            self.dd = cupy.zeros(0, dtype=np.complex128)
            self.dd_tmp = cupy.zeros(0, dtype=np.complex128)

    def get_empty_Wang_data(self):
        """Return placeholder values for variables used in Wang version."""
        return (cupy.empty((0, 0, 0, 0, 0)), cupy.empty((0, 0, 0)))

    def get_Wang_data(self, qpoints):
        """Allocate data for variables used in Wang version."""
        if self.charge_sum.shape[0] < len(qpoints):
            self.charge_sum = cupy.zeros(
                (len(qpoints), len(self._p2s), len(self._p2s), 3, 3)
            )
            self.q_born = cupy.zeros((len(qpoints), len(self._p2s), 3))
            charge_sum = self.charge_sum
            q_born = self.q_born
        else:
            charge_sum = self.charge_sum[0 : len(qpoints), :, :, :, :]
            q_born = self.q_born[0 : len(qpoints), :, :]
            charge_sum.fill(0.0)
            q_born.fill(0.0)
        return (charge_sum, q_born)

    def get_empty_Gonze_data(self):
        """Return placeholder values for variables used in Gonze version."""
        return (cupy.empty((0, 0, 0)), cupy.empty((0, 0, 0)))

    def get_Gonze_data(self, dynmat, qpoints):
        """Allocate data for variables used in Gonze version."""
        if self.dd.shape[0] < len(qpoints):
            self.dd = cupy.zeros_like(dynmat)
            self.dd_tmp = cupy.zeros_like(dynmat)
            dd = self.dd
            dd_tmp = self.dd_tmp
        else:
            dd = self.dd[0 : len(qpoints), :, :]
            dd_tmp = self.dd_tmp[0 : len(qpoints), :, :]
            dd.fill(0.0)
            dd_tmp.fill(0.0)
        return (dd, dd_tmp)

    def solve_dm_on_qpoints(self, qpoints, nac_q_direction=None, hermitianize=True):
        """Perform final preparation and launch appropriate kernel."""
        if self.dynmat.shape[0] < len(qpoints):
            self.dynmat = cupy.zeros(
                (len(qpoints), len(self.p2s) * 3, len(self.p2s) * 3),
                dtype=np.complex128,
            )
            dynmat = self.dynmat
        else:
            dynmat = self.dynmat[0 : len(qpoints), :, :]
            dynmat.fill(0.0)

        dynmat = cupy.zeros(
            (len(qpoints), len(self.p2s) * 3, len(self.p2s) * 3), dtype=np.complex128
        )

        qpoints_d = cupy.asarray(qpoints)

        threads = 256
        blocks = (len(qpoints) + (threads - 1)) // threads
        if isinstance(self.dm, DynamicalMatrixNAC):
            if nac_q_direction is None:
                q_dir = cupy.zeros(3)
            else:
                q_dir = cupy.asarray(nac_q_direction)

            if self.use_Wang_NAC:
                (charge_sum, q_born) = self.get_Wang_data(qpoints)
                (dd, dd_tmp) = self.get_empty_Gonze_data()
            else:
                (dd, dd_tmp) = self.get_Gonze_data(dynmat, qpoints)
                (charge_sum, q_born) = self.get_empty_Wang_data()

            _dm_kernel_nac[blocks, threads](
                dynmat,
                dd,
                dd_tmp,
                charge_sum,
                q_born,
                self.svecs,
                self.multi,
                self.masses,
                self.rec_lattice,
                self.positions,
                self.born,
                self.nac_factor,
                self.dielectric,
                self.dm_fc,
                self.dd_q0,
                self.G_list,
                self.Lambda,
                self.p2s,
                self.s2p,
                qpoints_d,
                q_dir,
                hermitianize,
                self.use_Wang_NAC,
            )
        else:
            _dm_kernel[blocks, threads](
                dynmat,
                self.svecs,
                self.multi,
                self.masses,
                self.dm_fc,
                self.p2s,
                self.s2p,
                qpoints_d,
            )

        if self.with_eigenvectors:
            eigvals_d, eigvecs_d = cupy.linalg.eigh(dynmat)
        else:
            eigvals_d = cupy.linalg.eigvalsh(dynmat)
            eigvecs_d = None

        return qpoints_d, eigvals_d, eigvecs_d, dynmat
