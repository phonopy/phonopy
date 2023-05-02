"""Tests for QE calculater interface."""
import io

import yaml

from phonopy.interface.lammps import LammpsStructureParser
from phonopy.interface.phonopy_yaml import load_phonopy_yaml
from phonopy.structure.symmetry import Symmetry

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

lammps_Ti = """#

2 atoms
1 atom types

0.0 2.923479689273095 xlo xhi   # xx
0.0 2.531807678358337 ylo yhi   # yy
0.0 4.624022835916574 zlo zhi   # zz

-1.461739844636547 0.000000000000000 0.000000000000000 xy xz yz

Atoms

1 1 0.000000000000001 1.687871785572226 3.468017126937431
2 1 1.461739844636549 0.843935892786111 1.156005708979144
"""


phonopy_atoms_Ti = """lattice:
- [     2.923479689273095,     0.000000000000000,     0.000000000000000 ] # a
- [    -1.461739844636547,     2.531807678358337,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     4.624022835916574 ] # c
points:
- symbol: H  # 1
  coordinates: [  0.333333333333334,  0.666666666666667,  0.750000000000000 ]
  mass: 1.007940
- symbol: H  # 2
  coordinates: [  0.666666666666667,  0.333333333333333,  0.250000000000000 ]
  mass: 1.007940
"""


def test_LammpsStructure(helper_methods):
    """Test of LammpsParser."""
    lines = lammps_Ti.splitlines()
    lmps = LammpsStructureParser()
    lmps.parse(lines)
    phyml = load_phonopy_yaml(yaml.load(io.StringIO(phonopy_atoms_Ti), Loader=Loader))
    helper_methods.compare_cells_with_order(lmps.cell, phyml.unitcell)
    symmetry = Symmetry(phyml.unitcell)
    assert symmetry.dataset["number"] == 194
