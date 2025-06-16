"""Questaal/LMTO calculator interface."""

# Initial version by GCGS, adaptation to phonopy v2 and subsequent revision by DLP

import os
import sys

import numpy as np

from phonopy.interface.vasp import check_forces, get_drift_forces
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
    qlm_ctx = QlmFl()
    qlm_ctx.load_site(filename)

    plat = qlm_ctx.plat
    poss = qlm_ctx.positions

    scaled_poss = poss
    if not qlm_ctx.xpos:
        scaled_poss = poss @ np.linalg.inv(plat)

    plat *= qlm_ctx.alat

    cell = PhonopyAtoms(
        cell=plat, scaled_positions=scaled_poss, symbols=qlm_ctx.symbols_sfxn
    )

    return cell, (qlm_ctx,)


def to_site_str(cell, *extra_args):
    """Convert cell to site-formatted string."""
    qctx = extra_args[0] if extra_args else QlmFl()
    return qctx.to_site_str(cell)


def write_qlm(filename, cell, *extra_args):
    """Write site file."""
    qctx = extra_args[0] if extra_args else QlmFl()
    qctx.write_site(cell, filename=filename)


def write_supercells_with_displacements(
    supercell, cells_with_displacements, ids, qlm_ctx, width=3
):
    """Write supercells with displacements to files."""
    qlm_ctx.write_site(supercell, prefix="supercell")
    for i, cell in zip(ids, cells_with_displacements):
        qlm_ctx.write_site(cell, prefix="supercell", idx=i, width=width)


class QlmFl:
    """Class to read and write QLM site files."""

    def __init__(self):
        self._filename = "site.lm"
        self._ext = ".lm"
        self._content = ""

        self.natom = 0
        self.xpos = False
        self.alat = 1.0
        self.plat = np.eye(3)

        self.symbols = []
        self.symbols_sfxn = []
        self.positions = np.array([])

        self._lsorted_els = sorted(symbol_map.keys(), key=len, reverse=True)

        self._sq2sp = dict()
        self._sp2sq = dict()

    def split_symbol_element(self, s):
        """Split symbol to element and suffix."""
        el_sfx = "", s

        for e in self._lsorted_els:
            if s.startswith(e):
                el_sfx = e, s[len(e) :]
                break

        return el_sfx

    def load_site(self, filename):
        """Fill internal variables from site file."""
        self._filename = filename
        self._ext = os.path.splitext(filename)[1]
        self._content = open(filename, "r").read()

        lines = self._content.splitlines()
        header = lines[0]
        sites = lines[2:]

        hlist = header.replace("=", "= ").split()

        self.xpos = "xpos" in hlist

        idx = hlist.index("nbas=")
        self.natom = int(hlist[idx + 1])

        idx = hlist.index("alat=")
        self.alat = float(hlist[idx + 1])

        idx = hlist.index("plat=")
        self.plat = np.array(hlist[idx + 1 : idx + 10], dtype=float)
        self.plat.shape = 3, 3

        nl = 0
        positions = []
        for site in sites:
            if site.strip() != "":
                ls = site.split()
                self.symbols.append(ls[0])
                positions.append(ls[1:4])
                nl += 1
                if nl == self.natom:
                    break

        self.positions = np.array(positions, dtype=float)

        sfx = {self.split_symbol_element(symb)[1] for symb in self.symbols}
        sfx = {s for s in sfx if s != ""}
        sfxn = {s for s in sfx if s.isnumeric()}  # preserve numeric suffixes mapping
        mxn = max(map(int, sfxn)) if len(sfxn) != 0 else 0
        sfx = {s for s in sfx if not s.isnumeric()}
        qls2fns = dict(((s, str(n + mxn + 1)) for n, s in enumerate(sfx)))
        qls2fns |= dict((s, s) for s in sfxn)
        qls2fns[""] = ""

        self._sq2sp = dict()
        for symb in self.symbols:
            e, s = self.split_symbol_element(symb)
            fnsymb = e + qls2fns[s]
            if symb not in self._sq2sp:
                self._sq2sp[symb] = fnsymb

        self._sp2sq = dict((v, k) for k, v in self._sq2sp.items())

        self.symbols_sfxn = list(self._sq2sp[symb] for symb in self.symbols)

        assert self.natom == self.positions.shape[0]
        assert self.natom == len(self.symbols)

    def to_site_str(self, cell):
        """Write cell to site file formatted string."""
        lattice = cell.cell / self.alat
        natoms = len(cell.symbols)
        plat_str = " ".join(
            [("%.12f" % p).lstrip().rstrip("0").rstrip(".") for p in lattice.ravel()]
        )

        lines = [
            "% site-data vn=3.0"
            + (" xpos" if self.xpos else "")
            + " fast io=15"
            + " nbas=%d" % natoms
            + " alat="
            + ("%.12f" % self.alat).lstrip().rstrip("0").rstrip(".")
            + " plat="
            + plat_str
        ]
        lines.append("#                              pos")

        poss = cell.scaled_positions
        if not self.xpos:
            poss = cell.scaled_positions @ lattice

        symbs = cell.symbols
        if len(self._sp2sq) > 0:
            symbs = list(self._sp2sq[symb] for symb in cell.symbols)
        for symb, crd in zip(symbs, poss):
            lines.append(
                " %7s %17.12f %17.12f %17.12f" % (symb.ljust(7), crd[0], crd[1], crd[2])
            )

        return "\n".join(lines) + "\n"

    def write_site(self, cell, filename="", prefix="", idx=None, width=None):
        """Write optionally indexed cell to site file."""
        fname = filename

        if fname == "":
            fname = self._filename

        if prefix != "":
            ext = self._ext
            fname = f"{prefix}{ext}"
            if idx is not None:
                fname = f"{prefix}-{idx:0{width}}{ext}"

        with open(fname, "w") as f:
            f.write(self.to_site_str(cell))


if __name__ == "__main__":
    import sys

    from phonopy.structure.symmetry import Symmetry

    cell, (qlm_ctx,) = read_qlm(sys.argv[1])
    symmetry = Symmetry(cell)

    print("# %s" % symmetry.get_international_table())
    print(qlm_ctx.to_site_str(cell))

    print(to_site_str(cell))
    print(to_site_str(cell, qlm_ctx))
