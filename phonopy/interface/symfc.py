"""Symfc force constants calculator interface."""

# Copyright (C) 2024 Atsushi Togo
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

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry


def get_fc2(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    displacements: np.ndarray,
    forces: np.ndarray,
    atom_list: Optional[Union[Sequence[int], np.ndarray]] = None,
    symmetry: Optional[Symmetry] = None,
    options: Optional[Union[str, dict]] = None,
    log_level: int = 0,
):
    """Calculate fc2 using symfc."""
    p2s_map = primitive.p2s_map
    is_compact_fc = atom_list is not None and (atom_list == p2s_map).all()
    fc2 = run_symfc(
        supercell,
        primitive,
        displacements,
        forces,
        is_compact_fc=is_compact_fc,
        symmetry=symmetry,
        options=options,
        log_level=log_level,
    )[0]

    if not is_compact_fc and atom_list is not None:
        fc2 = np.array(fc2[atom_list], dtype="double", order="C")

    return fc2


def run_symfc(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    displacements: np.ndarray,
    forces: np.ndarray,
    orders: Optional[Sequence[int]] = None,
    is_compact_fc: bool = False,
    symmetry: Optional[Symmetry] = None,
    options: Optional[Union[str, dict]] = None,
    log_level: int = 0,
):
    """Calculate force constants."""
    try:
        from symfc import Symfc
        from symfc.utils.utils import SymfcAtoms
    except ImportError as exc:
        raise ModuleNotFoundError("Symfc python module was not found.") from exc

    if orders is None:
        _orders = [2]
    else:
        _orders = orders

    if options is None:
        options_dict = {}
    else:
        options_dict = parse_symfc_options(options)

    if log_level:
        print(
            "--------------------------------"
            " Symfc start "
            "-------------------------------"
        )
        print("Symfc is a force constants calculator. See the following paper:")
        print("A. Seko and A. Togo, arXiv:2403.03588.")
        print("Symfc is developed at https://github.com/symfc/symfc.")
        print(f"Computing {_orders} order force constants.", flush=True)
        if options_dict:
            print("Parameters:")
            for key, val in options_dict.items():
                print(f"  {key}: {val}", flush=True)

    if log_level == 1:
        print("Increase log-level to watch detailed symfc log.")

    symfc_supercell = SymfcAtoms(
        cell=supercell.cell,
        scaled_positions=supercell.scaled_positions,
        numbers=supercell.numbers,
    )
    spacegroup_operations = symmetry.symmetry_operations if symmetry else None
    symfc = Symfc(
        symfc_supercell,
        spacegroup_operations=spacegroup_operations,
        displacements=displacements,
        forces=forces,
        cutoff={int(max(_orders)): options_dict.get("cutoff", None)},
        use_mkl=options_dict.get("use_mkl", False),
        log_level=log_level - 1 and log_level,
    ).run(max_order=int(max(_orders)), is_compact_fc=is_compact_fc)

    if log_level:
        print(
            "---------------------------------"
            " Symfc end "
            "--------------------------------"
        )

    if is_compact_fc:
        assert (symfc.p2s_map == primitive.p2s_map).all()

    return [symfc.force_constants[n] for n in _orders]


def parse_symfc_options(options: Union[str, dict]):
    """Parse symfc options.

    Parameters
    ----------
    options : Union[str, dict]
        Options for symfc.

    Note
    ----
    When str, it should be written as follows:

        "cutoff = 10.0"

    """
    if isinstance(options, dict):
        return options
    elif isinstance(options, str):
        options_dict = {}
        for option in options.split(","):
            key_val = [v.strip().lower() for v in option.split("=")]
            if len(key_val) != 2:
                break
            key, val = key_val
            if key == "cutoff":
                options_dict[key] = float(val)
            if key == "use_mkl":
                if val.strip().lower() == "true":
                    options_dict[key] = True
        return options_dict
    else:
        raise TypeError(f"options must be str or dict, not {type(options)}.")
