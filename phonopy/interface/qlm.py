"""Questaal/LMTO calculator interface."""

# Initial version by GCGS, adaptation to phonopy v2 by DLP

import sys

import numpy as np

from phonopy.interface.vasp import (
    check_forces,
    get_drift_forces,
    sort_positions_by_symbols,
)
from phonopy.structure.atoms import PhonopyAtoms, symbol_map


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    is_parsed = True
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        qlm_forces = np.loadtxt(filename, dtype=float, skiprows=1)

        if check_forces(qlm_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                qlm_forces, filename=filename, verbose=verbose
            )
            force_sets.append(qlm_forces - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_qlm(filename):
    """Read crystal structure."""
    with open(filename, "r") as f:
        lines = f.readlines()

    header = lines[0]
    sites = lines[2:]

    qlm_in = QlmIn(header, sites)
    tags = qlm_in.get_variables(header, sites)

    plat = [tags["alat"] * np.array(tags["plat"][i]) for i in range(3)]
    positions = tags["atoms"]["positions"]
    symbols = tags["atoms"]["spfnames"]

    numbers = []
    for s in symbols:
        numbers.append(symbol_map[s])

    cell = PhonopyAtoms(numbers=numbers, cell=plat, scaled_positions=positions)

    return cell


def write_qlm(filename, cell):
    """Write cell to file."""
    with open(filename, "w") as f:
        f.write(get_qlm_structure(cell))


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    pre_filename="supercell",
    width=3,
    ext="lm",
):
    """Write supercells with displacements to files."""
    write_qlm(
        "{pre_filename}.{ext}".format(pre_filename=pre_filename, ext=ext), supercell
    )
    for i, cell in zip(ids, cells_with_displacements):
        write_qlm(
            "{pre_filename}-{0:0{width}}.{ext}".format(
                i, pre_filename=pre_filename, width=width, ext=ext
            ),
            cell,
        )


def get_qlm_structure(cell):
    """Write cell to string."""
    lattice = cell.cell
    (num_atoms, symbols, scaled_positions, sort_list) = sort_positions_by_symbols(
        cell.symbols, cell.scaled_positions
    )

    lines = "%% site-data vn=3.0 xpos fast io=62 nbas=%d" % sum(num_atoms)
    lines += " alat=1.0 plat=" + ("%.6f " * 9 + "\n") % tuple(lattice.ravel())
    lines += "#                        pos                                   "
    lines += "vel                                    eula                     "
    lines += "vshft  PL rlx\n"

    atom_species = []
    for i, j in zip(symbols, num_atoms):
        atom_species.append([i] * j)

    for x, y in zip(sum(atom_species, []), scaled_positions):
        lines += " %3s" % x
        lines += (" %12.7f" * 3) % tuple(y)
        lines += (" %12.7f" * 7) % tuple([0.0] * 7)
        lines += " 0 111\n"

    return lines


class QlmIn:
    """Class to read QLM structure file."""

    def __init__(self, header, sites):
        self._set_methods = {
            "atoms": self._set_atoms,
            "plat": self._set_plat,
            "alat": self._set_alat,
        }
        self._tags = {"atoms": None, "plat": None, "alat": 1.0}

    def _set_atoms(self, sites):
        spfnames = []
        positions = []
        for i in sites:
            spfnames.append(i.split()[0])
            positions.append([float(x) for x in i.split()[1:4]])
        self._tags["atoms"] = {"spfnames": spfnames, "positions": positions}

    def _set_plat(self, header):
        plat = []
        hlist = header.replace("=", "= ").split()
        index = hlist.index("plat=")
        for j in range(1, 10, 3):
            plat.append([float(x) for x in hlist[index + j : index + j + 3]])
        self._tags["plat"] = plat

    def _set_alat(self, header):
        hlist = header.split()
        for j in hlist:
            if j.startswith("alat"):
                alat = float(j.split("=")[1])
                break
        self._tags["alat"] = alat

    def _check_ord(self, header):
        if "xpos" not in header.split():
            raise RuntimeError(
                "Only site files with fractional coordinates are supported for now."
            )

    def get_variables(self, header, sites):
        """Obtain input variables in a dictionary."""
        self._check_ord(header)
        self._set_atoms(sites)
        self._set_plat(header)
        self._set_alat(header)
        return self._tags


if __name__ == "__main__":
    import sys

    from phonopy.structure.symmetry import Symmetry

    cell = read_qlm(sys.argv[1])
    symmetry = Symmetry(cell)
    print("# %s" % symmetry.get_international_table())
    print(get_qlm_structure(cell))
