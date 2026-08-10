# SPDX-License-Identifier: BSD-3-Clause
"""Tests for reading electronic states and their k-mesh from vaspout.h5.

The k-mesh descriptions below are the four the campaign has actually met, and
their values are taken from real files: Gamma-centred divisions (HCP-Ti),
generating vectors for a centred lattice (anatase TiO2), and Monkhorst-Pack
with odd and with even divisions (Si). The last pair is the reason the reader
does not judge mappability itself; see _mesh_from_vaspout_kpoints.

"""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.interface.vasp import electronic_states_from_vaspout

pytest.importorskip("h5py")

# Anatase TiO2's generating vectors, whose inverse transpose rounds to
# [[0, 8, 8], [8, 0, 8], [3, 3, 0]]. Written with the few digits VASP prints.
ANATASE_BASIS_VECTORS = [
    [-0.0625, 0.0625, 0.0625],
    [0.0625, -0.0625, 0.0625],
    [0.16666667, 0.16666667, -0.16666667],
]
ANATASE_GRID_MATRIX = [[0, 8, 8], [8, 0, 8], [3, 3, 0]]


def _write_vaspout(
    path,
    kpoints_group: dict | None = None,
    kpoints_opt_group: dict | None = None,
    n_kpoints: int = 2,
    n_kpoints_opt: int = 3,
    efermi: float = 1.0,
    efermi_opt: float = 2.0,
) -> None:
    """Write a vaspout.h5 with as much as the reader needs and no more."""
    import h5py

    def write_kpoints(parent, name, spec):
        group = parent.create_group(name)
        for key, value in spec.items():
            if isinstance(value, str):
                group.create_dataset(key, data=np.bytes_(value))
            elif isinstance(value, int):
                group.create_dataset(key, data=np.int32(value))
            else:
                group.create_dataset(key, data=np.asarray(value, dtype="double"))

    with h5py.File(path, "w") as w:
        g = w.create_group("results/electron_eigenvalues")
        g.create_dataset("eigenvalues", data=np.zeros((1, n_kpoints, 3)))
        g.create_dataset(
            "kpoints_symmetry_weight", data=np.full(n_kpoints, 1.0 / n_kpoints)
        )
        g.create_dataset("kpoint_coords", data=np.zeros((n_kpoints, 3)))
        g.create_dataset("nelectrons", data=4.0)
        w.create_group("results/electron_dos").create_dataset("ncdij", data=np.int32(1))
        w["results/electron_dos"].create_dataset("efermi", data=efermi)

        poscar = w.create_group("input/poscar")
        poscar.create_dataset("lattice_vectors", data=np.eye(3) * 4.0)
        poscar.create_dataset("position_ions", data=np.zeros((1, 3)))
        poscar.create_dataset("number_ion_types", data=np.array([1], dtype="int32"))
        poscar.create_dataset("ion_types", data=np.array([b"Si"]))
        poscar.create_dataset("scale", data=1.0)
        poscar.create_dataset("direct_coordinates", data=np.int32(1))

        if kpoints_group is not None:
            write_kpoints(w["input"], "kpoints", kpoints_group)
        if kpoints_opt_group is not None:
            write_kpoints(w["input"], "kpoints_opt", kpoints_opt_group)
            o = w.create_group("results/electron_eigenvalues_kpoints_opt")
            o.create_dataset("eigenvalues", data=np.zeros((1, n_kpoints_opt, 3)))
            o.create_dataset(
                "kpoints_symmetry_weight",
                data=np.full(n_kpoints_opt, 1.0 / n_kpoints_opt),
            )
            o.create_dataset("kpoint_coords", data=np.zeros((n_kpoints_opt, 3)))
            w.create_group("results/electron_dos_kpoints_opt").create_dataset(
                "efermi", data=efermi_opt
            )


def _divisions(mode: str, nkp=(4, 4, 4)) -> dict:
    """Return a kpoints group given as numbers of divisions."""
    return {
        "mode": mode,
        "shift": [0.0, 0.0, 0.0],
        "number_kpoints": 0,
        "nkpx": int(nkp[0]),
        "nkpy": int(nkp[1]),
        "nkpz": int(nkp[2]),
    }


def _generating_vectors() -> dict:
    """Return a kpoints group given as generating vectors."""
    return {
        "mode": "b",
        "shift": [0.0, 0.0, 0.0],
        "number_kpoints": 0,
        "basis_vectors": ANATASE_BASIS_VECTORS,
    }


def test_mesh_from_gamma_centred_divisions(tmp_path):
    """Test the mesh of a Gamma-centred mesh given as divisions."""
    path = tmp_path / "vaspout.h5"
    _write_vaspout(path, kpoints_group=_divisions("g", (34, 34, 18)))
    states = electronic_states_from_vaspout(path)
    np.testing.assert_array_equal(states.mesh, [34, 34, 18])


def test_mesh_from_monkhorst_pack_divisions(tmp_path):
    """Test that a Monkhorst-Pack mesh still yields its divisions.

    Whether it can be mapped depends on whether the divisions are odd, which
    the input description does not say: odd and even Monkhorst-Pack meshes
    are written identically here, both with mode 'm' and zero shift, and only
    the k-point coordinates differ. So the reader reports the mesh and
    get_ir_kpoint_map decides.

    """
    path = tmp_path / "vaspout.h5"
    _write_vaspout(path, kpoints_group=_divisions("m", (19, 19, 19)))
    np.testing.assert_array_equal(
        electronic_states_from_vaspout(path).mesh, [19, 19, 19]
    )


def test_mesh_from_generating_vectors(tmp_path):
    """Test the grid matrix of a mesh given as generating vectors.

    Anatase TiO2's, which is a generalized regular grid in the primitive
    basis. basis_vectors is printed with few digits, so its inverse transpose
    has to be rounded, and this checks that the rounding lands on the integer
    matrix rather than near it.

    """
    path = tmp_path / "vaspout.h5"
    _write_vaspout(path, kpoints_group=_generating_vectors())
    states = electronic_states_from_vaspout(path)
    np.testing.assert_array_equal(states.mesh, ANATASE_GRID_MATRIX)


def test_explicit_kpoint_list_has_no_mesh(tmp_path):
    """Test that an explicit k-point list leaves the grid fields unset."""
    path = tmp_path / "vaspout.h5"
    spec = _divisions("g")
    spec["number_kpoints"] = 12
    _write_vaspout(path, kpoints_group=spec)
    states = electronic_states_from_vaspout(path)
    assert states.mesh is None
    assert states.kpoints is None
    assert states.cell is None


def test_missing_kpoints_group_has_no_mesh(tmp_path):
    """Test that a file without input/kpoints still reads as states."""
    path = tmp_path / "vaspout.h5"
    _write_vaspout(path, kpoints_group=None)
    states = electronic_states_from_vaspout(path)
    assert states.mesh is None
    assert states.n_electrons == 4.0


def test_kpoints_opt_is_preferred_by_default(tmp_path):
    """Test that the denser KPOINTS_OPT mesh is the one read.

    Matching what phonopy-vasp-efe already does on the vasprun.xml path, so
    that the two readers agree on the same calculation.

    """
    path = tmp_path / "vaspout.h5"
    _write_vaspout(
        path,
        kpoints_group=_divisions("g", (4, 4, 4)),
        kpoints_opt_group=_divisions("g", (8, 8, 8)),
    )
    states = electronic_states_from_vaspout(path)
    np.testing.assert_array_equal(states.mesh, [8, 8, 8])
    assert states.eigenvalues.shape[1] == 3


def test_everything_of_a_mesh_comes_from_that_mesh(tmp_path):
    """Test that the Fermi energy follows the mesh the eigenvalues came from.

    The two meshes have their own Fermi energies -- 1.850 against 1.887 eV on
    the anatase TiO2 file, a difference larger than k_B T at 300 K -- and the
    Fermi energy anchors the electron count when the free energy is computed
    from a density of states. Taking one mesh's eigenvalues with the other's
    Fermi energy would move that anchor.

    """
    path = tmp_path / "vaspout.h5"
    _write_vaspout(
        path,
        kpoints_group=_divisions("g", (4, 4, 4)),
        kpoints_opt_group=_divisions("g", (8, 8, 8)),
        efermi=1.849605,
        efermi_opt=1.886527,
    )
    scf = electronic_states_from_vaspout(path, kpoints_opt=False)
    opt = electronic_states_from_vaspout(path, kpoints_opt=True)

    assert scf.fermi_energy == pytest.approx(1.849605)
    assert opt.fermi_energy == pytest.approx(1.886527)
    np.testing.assert_array_equal(scf.mesh, [4, 4, 4])
    np.testing.assert_array_equal(opt.mesh, [8, 8, 8])
    # The electron count belongs to the calculation, not to a mesh, and VASP
    # writes it only on the SCF side.
    assert opt.n_electrons == scf.n_electrons


def test_kpoints_opt_can_be_declined(tmp_path):
    """Test that the SCF mesh can be asked for even when KPOINTS_OPT exists."""
    path = tmp_path / "vaspout.h5"
    _write_vaspout(
        path,
        kpoints_group=_divisions("g", (4, 4, 4)),
        kpoints_opt_group=_divisions("g", (8, 8, 8)),
    )
    states = electronic_states_from_vaspout(path, kpoints_opt=False)
    np.testing.assert_array_equal(states.mesh, [4, 4, 4])
    assert states.eigenvalues.shape[1] == 2


def test_demanding_a_missing_kpoints_opt_raises(tmp_path):
    """Test that asking for a KPOINTS_OPT mesh that is not there raises."""
    path = tmp_path / "vaspout.h5"
    _write_vaspout(path, kpoints_group=_divisions("g"))
    with pytest.raises(ValueError, match="no KPOINTS_OPT mesh"):
        electronic_states_from_vaspout(path, kpoints_opt=True)
