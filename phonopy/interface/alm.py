"""ALM force constants calculator interface."""

# Copyright (C) 2018 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive


def run_alm(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    displacements: np.ndarray,
    forces: np.ndarray,
    maxorder: int,
    is_compact_fc: bool = False,
    options: Optional[str] = None,
    log_level: int = 0,
) -> list[np.ndarray]:
    """Calculate force constants using ALM.

    The details of the parameters are found in the ALMFCSolver class.

    """
    if log_level:
        print(
            "---------------------------------"
            " ALM start "
            "--------------------------------"
        )
        print(
            "ALM is a non-trivial force constants calculator. " "Please cite the paper:"
        )
        print("T. Tadano and S. Tsuneyuki, " "J. Phys. Soc. Jpn. 87, 041015 (2018).")
        print("ALM is developed at https://github.com/ttadano/ALM by T. " "Tadano.")

    if log_level == 1:
        print("Increase log-level to watch detailed ALM log.")

    alm = ALMFCSolver(
        supercell,
        primitive,
        displacements,
        forces,
        maxorder,
        is_compact_fc=is_compact_fc,
        options=options,
        log_level=log_level,
    )

    if log_level:
        print(
            "----------------------------------"
            " ALM end "
            "---------------------------------"
        )

    return alm.force_constants


class ALMFCSolver:
    """ALM force constants calculator interface."""

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        displacements: np.ndarray,
        forces: np.ndarray,
        maxorder: int,
        is_compact_fc: bool = False,
        options: Optional[str] = None,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell structure.
        primitive : Primitive
            Primitive structure.
        displacements : np.ndarray
            Atomic displacements.
        forces : np.ndarray
            Forces.
        maxorder : int
            Maximum order of force constants. The integer values 1, 2, ...
            correspond to 2nd, 3rd, ... order force constants.
        is_compact_fc : bool, optional
            Compact force constants. Default is False.
        options : str, optional
            ALM options. Default is None.
        log_level : int, optional
            Log level. Default is 0.

        """
        fcs = self._run(
            supercell,
            primitive,
            displacements,
            forces,
            maxorder,
            is_compact_fc=is_compact_fc,
            options=options,
            log_level=log_level,
        )
        self._fc = {i + 2: fc for i, fc in enumerate(fcs)}

    @property
    def force_constants(self) -> dict[int, np.ndarray]:
        """Return force constants.

        Returns
        -------
        dict[int, np.ndarray]
            Force constants with order as key.

        """
        return self._fc

    def _run(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        displacements: np.ndarray,
        forces: np.ndarray,
        maxorder: int,
        is_compact_fc: bool = False,
        options: Optional[str] = None,
        log_level: int = 0,
    ) -> list[np.ndarray]:
        """Calculate force constants of arbitrary-order using ALM."""
        fcs = None  # This is returned.

        lattice = supercell.cell
        positions = supercell.scaled_positions
        numbers = supercell.numbers
        natom = len(numbers)
        p2s_map = primitive.p2s_map
        p2p_map = primitive.p2p_map

        alm_options = _update_options(options)
        num_elems = len(np.unique(numbers))

        if "norder" in alm_options:
            _maxorder = alm_options["norder"]
        elif "maxorder" in alm_options:
            _maxorder = alm_options["maxorder"]
        else:
            _maxorder = maxorder

        shape = (_maxorder, num_elems, num_elems)
        cutoff_radii = -np.ones(shape, dtype="double")
        if alm_options["cutoff"] is not None:
            if len(alm_options["cutoff"]) == 1:
                cutoff_radii[:] = alm_options["cutoff"][0]
            elif len(alm_options["cutoff"]) == _maxorder:
                for i, cutoff in enumerate(alm_options["cutoff"]):
                    cutoff_radii[i] = cutoff
            elif np.prod(shape) == len(alm_options["cutoff"]):
                cutoff_radii[:] = np.reshape(alm_options["cutoff"], shape)
            else:
                raise RuntimeError("Cutoff is not properly set.")

        _disps, _forces, df_msg = _slice_displacements_and_forces(
            displacements,
            forces,
            alm_options["ndata"],
            alm_options["nstart"],
            alm_options["nend"],
        )

        if log_level > 1:
            print("")
            print("  ndata: %d" % len(displacements))
            for key, val in alm_options.items():
                if val is not None:
                    print("  %s: %s" % (key, val))
            print("")
            print(" " + "-" * 67)

        if log_level > 0:
            log_level_alm = log_level - 1
        else:
            log_level_alm = 0

        try:
            from alm import ALM, optimizer_control_data_types
        except ImportError as exc:
            raise ModuleNotFoundError("ALM python module was not found.") from exc

        with ALM(lattice, positions, numbers) as alm:
            if log_level > 0:
                if alm_options["cutoff"] is not None:
                    for i in range(_maxorder):
                        if _maxorder > 1:
                            print("fc%d" % (i + 2))
                        print(
                            ("cutoff" + " %6s" * num_elems)
                            % tuple(alm.kind_names.values())
                        )
                        for r, kn in zip(cutoff_radii[i], alm.kind_names.values()):
                            print(
                                ("   %-3s" + " %6.2f" * num_elems) % ((kn,) + tuple(r))
                            )
                if df_msg is not None:
                    print(df_msg)
            if log_level > 1:
                print("")
            sys.stdout.flush()

            alm.output_filename_prefix = alm_options["output_filename_prefix"]
            alm.verbosity = log_level_alm

            alm.define(
                _maxorder,
                cutoff_radii=cutoff_radii,
                nbody=alm_options["nbody"],
                symmetrization_basis=alm_options["symmetrization_basis"],
            )
            alm.displacements = _disps
            alm.forces = _forces

            # Mainly for elastic net (or lasso) regression
            optcontrol = {}
            for key in optimizer_control_data_types:
                if key in alm_options:
                    optcontrol[key] = alm_options[key]
            if optcontrol:
                alm.optimizer_control = optcontrol
                if (
                    "cross_validation" in optcontrol
                    and optcontrol["cross_validation"] > 0
                ):
                    alm.optimize(solver=alm_options["solver"])
                    optcontrol["cross_validation"] = 0
                    optcontrol["l1_alpha"] = alm.cv_l1_alpha
                    alm.optimizer_control = optcontrol

            if alm_options["iconst"] == 0:
                alm.set_constraint(translation=False)

            alm.optimize(solver=alm_options["solver"])

            fcs = _extract_fc_from_alm(
                alm, natom, maxorder, is_compact_fc, p2s_map=p2s_map, p2p_map=p2p_map
            )

        return fcs


def _update_options(fc_calculator_options):
    """Set ALM options with appropriate data types.

    fc_calculator_options : str
        This string should be written such as follows:

            "solver = dense, cutoff = 5"

        This string is parsed as collection of settings that are separated by
        comma ','. Each setting has the format of 'option = value'. The value
        is cast to have its appropriate data type for ALM in this method.

    """
    try:
        from alm import optimizer_control_data_types
    except ImportError as exc:
        raise ModuleNotFoundError("ALM python module was not found.") from exc

    # Default settings.
    alm_options = {
        "solver": "dense",
        "ndata": None,
        "nstart": None,
        "nend": None,
        "nbody": None,
        "cutoff": None,
        "symmetrization_basis": "Lattice",
        "output_filename_prefix": None,
        "iconst": 11,
    }

    if fc_calculator_options is not None:
        alm_option_types = {
            "cutoff": np.double,
            "maxorder": int,
            "norder": int,
            "ndata": int,
            "nstart": int,
            "nend": int,
            "nbody": np.dtype("long"),
            "output_filename_prefix": str,
            "solver": str,
            "symmetrization_basis": str,
            "iconst": int,
        }
        alm_option_types.update(optimizer_control_data_types)
        for option_str in fc_calculator_options.split(","):
            key, val = [x.strip() for x in option_str.split("=")[:2]]
            if key.lower() in alm_option_types:
                if alm_option_types[key.lower()] is np.double:
                    option_value = np.array(
                        [float(x) for x in val.split()], dtype="double"
                    )
                elif alm_option_types[key.lower()] is np.dtype("long"):
                    option_value = np.array([int(x) for x in val.split()], dtype="long")
                else:
                    option_value = alm_option_types[key.lower()](val)
                alm_options[key] = option_value
    return alm_options


def _slice_displacements_and_forces(d, f, ndata, nstart, nend):
    msg = None
    if ndata is not None:
        _d = d[:ndata]
        _f = f[:ndata]
        msg = "Number of displacement supercells: %d" % ndata
    elif nstart is not None and nend is not None:
        _d = d[nstart - 1 : nend]
        _f = f[nstart - 1 : nend]
        msg = "Supercell index range: %d - %d" % (nstart, nend)
    else:
        return d, f, None

    return (
        np.array(_d, dtype="double", order="C"),
        np.array(_f, dtype="double", order="C"),
        msg,
    )


def _extract_fc_from_alm(
    alm, natom, maxorder, is_compact_fc, p2s_map=None, p2p_map=None
) -> list[np.ndarray]:
    # Harmonic: order=1, 3rd: order=2, ...
    fcs = []
    for order in range(1, maxorder + 1):
        fc = None
        p2s_map_alm = alm.getmap_primitive_to_supercell()[0]
        if (
            is_compact_fc
            and len(p2s_map_alm) == len(p2s_map)
            and (p2s_map_alm == p2s_map).all()
        ):
            fc_shape = (len(p2s_map),) + (natom,) * order + (3,) * (order + 1)
            fc = np.zeros(fc_shape, dtype="double", order="C")
            for fc_elem, indices in zip(
                *alm.get_fc(order, mode="origin", permutation=1)
            ):
                v = indices // 3
                c = indices % 3
                selection = (p2p_map[v[0]],) + tuple(v[1:]) + tuple(c)
                fc[selection] = fc_elem

        if fc is None:
            if is_compact_fc:
                atom_list = p2s_map
            else:
                atom_list = np.arange(natom, dtype=int)
            fc_shape = (len(atom_list),) + (natom,) * order + (3,) * (order + 1)
            fc = np.zeros(fc_shape, dtype="double", order="C")
            for fc_elem, indices in zip(*alm.get_fc(order, mode="all", permutation=1)):
                v = indices // 3
                idx = np.where(atom_list == v[0])[0]
                if len(idx) > 0:
                    c = indices % 3
                    selection = (idx[0],) + tuple(v[1:]) + tuple(c)
                    fc[selection] = fc_elem

        fcs.append(fc)

    return fcs
