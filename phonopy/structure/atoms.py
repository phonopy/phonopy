# Copyright (C) 2011 Atsushi Togo
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

import numpy as np

class Atoms:
    """Atoms class compatible with the ASE Atoms class
    Only the necessary stuffs to phonpy are implemented. """
    
    def __init__(self,
                 symbols=None,
                 positions=None,
                 numbers=None, 
                 masses=None,
                 magmoms=None,
                 scaled_positions=None,
                 cell=None,
                 pbc=None):

        # cell
        if cell == None:
            self.cell=None
        else:
            self.cell = np.array(cell, dtype='double', order='C')

        # position
        self.scaled_positions = None
        if (not self.cell == None) and  (not positions == None):
            self.set_positions(positions)
        if (not scaled_positions == None):
            self.set_scaled_positions(scaled_positions)

        # Atom symbols
        self.symbols = symbols

        # Atomic numbers
        if numbers==None:
            self.numbers = None
        else:
            self.numbers = np.array(numbers, dtype='intc')

        # masses
        self.set_masses(masses)

        # (initial) magnetic moments
        self.set_magnetic_moments(magmoms)

        # number --> symbol
        if not self.numbers == None:
            self.numbers_to_symbols()

        # symbol --> number
        elif not self.symbols == None:
            self.symbols_to_numbers()

        # symbol --> mass
        if self.symbols and (self.masses == None):
            self.symbols_to_masses()


    def set_cell(self, cell):
        self.cell = np.array(cell, dtype='double', order='C')

    def get_cell(self):
        return self.cell.copy()

    def set_positions(self, cart_positions):
        self.scaled_positions = np.dot(cart_positions,
                                        np.linalg.inv(self.cell))

    def get_positions(self):
        return np.dot(self.scaled_positions, self.cell)

    def set_scaled_positions(self, scaled_positions):
        self.scaled_positions = np.array(scaled_positions,
                                         dtype='double', order='C')

    def get_scaled_positions(self):
        return self.scaled_positions.copy()

    def set_masses(self, masses):
        if masses == None:
            self.masses = None
        else:
            self.masses = np.array(masses, dtype='double')

    def get_masses(self):
        return self.masses.copy()

    def set_magnetic_moments(self, magmoms):
        if magmoms == None:
            self.magmoms = None
        else:
            self.magmoms = np.array(magmoms, dtype='double')

    def get_magnetic_moments(self):
        if self.magmoms == None:
            return None
        else:
            return self.magmoms.copy()

    def set_chemical_symbols(self, symbols):
        self.symbols = symbols

    def get_chemical_symbols(self):
        return self.symbols[:]

    def get_number_of_atoms(self):
        return len(self.scaled_positions)

    def get_atomic_numbers(self):
        return self.numbers.copy()

    def numbers_to_symbols(self):
        self.symbols = [atom_data[n][1] for n in self.numbers]
        
    def symbols_to_numbers(self):
        self.numbers = np.array([symbol_map[s]
                                 for s in self.symbols])
        
    def symbols_to_masses(self):
        self.masses = np.array([atom_data[symbol_map[s]][3]
                                for s in self.symbols])

    def get_volume(self):
        return np.linalg.det(self.cell)

atom_data = [ 
    [  0, "X", "X", 0], # 0
    [  1, "H", "Hydrogen", 1.00794], # 1
    [  2, "He", "Helium", 4.002602], # 2
    [  3, "Li", "Lithium", 6.941], # 3
    [  4, "Be", "Beryllium", 9.012182], # 4
    [  5, "B", "Boron", 10.811], # 5
    [  6, "C", "Carbon", 12.0107], # 6
    [  7, "N", "Nitrogen", 14.0067], # 7
    [  8, "O", "Oxygen", 15.9994], # 8
    [  9, "F", "Fluorine", 18.9984032], # 9
    [ 10, "Ne", "Neon", 20.1797], # 10
    [ 11, "Na", "Sodium", 22.98976928], # 11
    [ 12, "Mg", "Magnesium", 24.3050], # 12
    [ 13, "Al", "Aluminium", 26.9815386], # 13
    [ 14, "Si", "Silicon", 28.0855], # 14
    [ 15, "P", "Phosphorus", 30.973762], # 15
    [ 16, "S", "Sulfur", 32.065], # 16
    [ 17, "Cl", "Chlorine", 35.453], # 17
    [ 18, "Ar", "Argon", 39.948], # 18
    [ 19, "K", "Potassium", 39.0983], # 19
    [ 20, "Ca", "Calcium", 40.078], # 20
    [ 21, "Sc", "Scandium", 44.955912], # 21
    [ 22, "Ti", "Titanium", 47.867], # 22
    [ 23, "V", "Vanadium", 50.9415], # 23
    [ 24, "Cr", "Chromium", 51.9961], # 24
    [ 25, "Mn", "Manganese", 54.938045], # 25
    [ 26, "Fe", "Iron", 55.845], # 26
    [ 27, "Co", "Cobalt", 58.933195], # 27
    [ 28, "Ni", "Nickel", 58.6934], # 28
    [ 29, "Cu", "Copper", 63.546], # 29
    [ 30, "Zn", "Zinc", 65.38], # 30
    [ 31, "Ga", "Gallium", 69.723], # 31
    [ 32, "Ge", "Germanium", 72.64], # 32
    [ 33, "As", "Arsenic", 74.92160], # 33
    [ 34, "Se", "Selenium", 78.96], # 34
    [ 35, "Br", "Bromine", 79.904], # 35
    [ 36, "Kr", "Krypton", 83.798], # 36
    [ 37, "Rb", "Rubidium", 85.4678], # 37
    [ 38, "Sr", "Strontium", 87.62], # 38
    [ 39, "Y", "Yttrium", 88.90585], # 39
    [ 40, "Zr", "Zirconium", 91.224], # 40
    [ 41, "Nb", "Niobium", 92.90638], # 41
    [ 42, "Mo", "Molybdenum", 95.96], # 42
    [ 43, "Tc", "Technetium", 0], # 43
    [ 44, "Ru", "Ruthenium", 101.07], # 44
    [ 45, "Rh", "Rhodium", 102.90550], # 45
    [ 46, "Pd", "Palladium", 106.42], # 46
    [ 47, "Ag", "Silver", 107.8682], # 47
    [ 48, "Cd", "Cadmium", 112.411], # 48
    [ 49, "In", "Indium", 114.818], # 49
    [ 50, "Sn", "Tin", 118.710], # 50
    [ 51, "Sb", "Antimony", 121.760], # 51
    [ 52, "Te", "Tellurium", 127.60], # 52
    [ 53, "I", "Iodine", 126.90447], # 53
    [ 54, "Xe", "Xenon", 131.293], # 54
    [ 55, "Cs", "Caesium", 132.9054519], # 55
    [ 56, "Ba", "Barium", 137.327], # 56
    [ 57, "La", "Lanthanum", 138.90547], # 57
    [ 58, "Ce", "Cerium", 140.116], # 58
    [ 59, "Pr", "Praseodymium", 140.90765], # 59
    [ 60, "Nd", "Neodymium", 144.242], # 60
    [ 61, "Pm", "Promethium", 0], # 61
    [ 62, "Sm", "Samarium", 150.36], # 62
    [ 63, "Eu", "Europium", 151.964], # 63
    [ 64, "Gd", "Gadolinium", 157.25], # 64
    [ 65, "Tb", "Terbium", 158.92535], # 65
    [ 66, "Dy", "Dysprosium", 162.500], # 66
    [ 67, "Ho", "Holmium", 164.93032], # 67
    [ 68, "Er", "Erbium", 167.259], # 68
    [ 69, "Tm", "Thulium", 168.93421], # 69
    [ 70, "Yb", "Ytterbium", 173.054], # 70
    [ 71, "Lu", "Lutetium", 174.9668], # 71
    [ 72, "Hf", "Hafnium", 178.49], # 72
    [ 73, "Ta", "Tantalum", 180.94788], # 73
    [ 74, "W", "Tungsten", 183.84], # 74
    [ 75, "Re", "Rhenium", 186.207], # 75
    [ 76, "Os", "Osmium", 190.23], # 76
    [ 77, "Ir", "Iridium", 192.217], # 77
    [ 78, "Pt", "Platinum", 195.084], # 78
    [ 79, "Au", "Gold", 196.966569], # 79
    [ 80, "Hg", "Mercury", 200.59], # 80
    [ 81, "Tl", "Thallium", 204.3833], # 81
    [ 82, "Pb", "Lead", 207.2], # 82
    [ 83, "Bi", "Bismuth", 208.98040], # 83
    [ 84, "Po", "Polonium", 0], # 84
    [ 85, "At", "Astatine", 0], # 85
    [ 86, "Rn", "Radon", 0], # 86
    [ 87, "Fr", "Francium", 0], # 87
    [ 88, "Ra", "Radium", 0], # 88
    [ 89, "Ac", "Actinium", 0], # 89
    [ 90, "Th", "Thorium", 232.03806], # 90
    [ 91, "Pa", "Protactinium", 231.03588], # 91
    [ 92, "U", "Uranium", 238.02891], # 92
    [ 93, "Np", "Neptunium", 0], # 93
    [ 94, "Pu", "Plutonium", 0], # 94
    [ 95, "Am", "Americium", 0], # 95
    [ 96, "Cm", "Curium", 0], # 96
    [ 97, "Bk", "Berkelium", 0], # 97
    [ 98, "Cf", "Californium", 0], # 98
    [ 99, "Es", "Einsteinium", 0], # 99
    [100, "Fm", "Fermium", 0], # 100
    [101, "Md", "Mendelevium", 0], # 101
    [102, "No", "Nobelium", 0], # 102
    [103, "Lr", "Lawrencium", 0], # 103
    [104, "Rf", "Rutherfordium", 0], # 104
    [105, "Db", "Dubnium", 0], # 105
    [106, "Sg", "Seaborgium", 0], # 106
    [107, "Bh", "Bohrium", 0], # 107
    [108, "Hs", "Hassium", 0], # 108
    [109, "Mt", "Meitnerium", 0], # 109
    [110, "Ds", "Darmstadtium", 0], # 110
    [111, "Rg", "Roentgenium", 0], # 111
    [112, "Cn", "Copernicium", 0], # 112
    [113, "Uut", "Ununtrium", 0], # 113
    [114, "Uuq", "Ununquadium", 0], # 114
    [115, "Uup", "Ununpentium", 0], # 115
    [116, "Uuh", "Ununhexium", 0], # 116
    [117, "Uus", "Ununseptium", 0], # 117
    [118, "Uuo", "Ununoctium", 0], # 118
    ]

symbol_map = {
    "H":1,
    "He":2,
    "Li":3,
    "Be":4,
    "B":5,
    "C":6,
    "N":7,
    "O":8,
    "F":9,
    "Ne":10,
    "Na":11,
    "Mg":12,
    "Al":13,
    "Si":14,
    "P":15,
    "S":16,
    "Cl":17,
    "Ar":18,
    "K":19,
    "Ca":20,
    "Sc":21,
    "Ti":22,
    "V":23,
    "Cr":24,
    "Mn":25,
    "Fe":26,
    "Co":27,
    "Ni":28,
    "Cu":29,
    "Zn":30,
    "Ga":31,
    "Ge":32,
    "As":33,
    "Se":34,
    "Br":35,
    "Kr":36,
    "Rb":37,
    "Sr":38,
    "Y":39,
    "Zr":40,
    "Nb":41,
    "Mo":42,
    "Tc":43,
    "Ru":44,
    "Rh":45,
    "Pd":46,
    "Ag":47,
    "Cd":48,
    "In":49,
    "Sn":50,
    "Sb":51,
    "Te":52,
    "I":53,
    "Xe":54,
    "Cs":55,
    "Ba":56,
    "La":57,
    "Ce":58,
    "Pr":59,
    "Nd":60,
    "Pm":61,
    "Sm":62,
    "Eu":63,
    "Gd":64,
    "Tb":65,
    "Dy":66,
    "Ho":67,
    "Er":68,
    "Tm":69,
    "Yb":70,
    "Lu":71,
    "Hf":72,
    "Ta":73,
    "W":74,
    "Re":75,
    "Os":76,
    "Ir":77,
    "Pt":78,
    "Au":79,
    "Hg":80,
    "Tl":81,
    "Pb":82,
    "Bi":83,
    "Po":84,
    "At":85,
    "Rn":86,
    "Fr":87,
    "Ra":88,
    "Ac":89,
    "Th":90,
    "Pa":91,
    "U":92,
    "Np":93,
    "Pu":94,
    "Am":95,
    "Cm":96,
    "Bk":97,
    "Cf":98,
    "Es":99,
    "Fm":100,
    "Md":101,
    "No":102,
    "Lr":103,
    "Rf":104,
    "Db":105,
    "Sg":106,
    "Bh":107,
    "Hs":108,
    "Mt":109,
    "Ds":110,
    "Rg":111,
    "Cn":112,
    "Uut":113,
    "Uuq":114,
    "Uup":115,
    "Uuh":116,
    "Uus":117,
    "Uuo":118,
    }

# This data are obtained from
# J. R. de Laeter, J. K. Böhlke, P. De Bièvre, H. Hidaka, H. S. Peiser,
# K. J. R. Rosman and P. D. P. Taylor (2003).
# "Atomic weights of the elements. Review 2000 (IUPAC Technical Report)"
isotope_data = {
    'H':  [[1, 0.999885], [2, 0.000115]],
    'He': [[3, 0.00000134], [4, 0.99999866]],
    'Li': [[6, 0.0759], [7, 0.9241]],
    'Be': [[9, 1.0000]],
    'B':  [[10, 0.199], [11, 0.801]],
    'C':  [[12, 0.9893], [13, 0.0107]],
    'N':  [[14, 0.99636], [15, 0.00364]],
    'O':  [[16, 0.99757], [17, 0.00038], [18, 0.00205]],
    'F':  [[19, 1.0000]],
    'Ne': [[20, 0.9048], [21, 0.0027], [22, 0.0925]],
    'Na': [[23, 1.0000]],
    'Mg': [[24, 0.7899], [25, 0.1000], [26, 0.1101]],
    'Al': [[27, 1.0000]],
    'Si': [[28, 0.92223], [29, 0.04685], [30, 0.03092]],
    'P':  [[31, 1.0000]],
    'S':  [[32, 0.9499], [33, 0.0075], [34, 0.0425], [36, 0.0001]],
    'Cl': [[35, 0.7576], [37, 0.2424]],
    'Ar': [[36, 0.003365], [38, 0.000632], [40, 0.996003]],
    'K':  [[39, 0.932581], [40, 0.000117], [41, 0.067302]],
    'Ca': [[40, 0.96941], [42, 0.00647], [43, 0.00135], [44, 0.02086],
           [46, 0.00004], [48, 0.00187]],
    'Sc': [[45, 1.0000]],
    'Ti': [[46, 0.0825], [47, 0.0744], [48, 0.7372], [49, 0.0541],
           [50, 0.0518]],
    'V':  [[50, 0.00250], [51, 0.99750]],
    'Cr': [[50, 0.04345], [52, 0.83789], [53, 0.09501], [54, 0.02365]],
    'Mn': [[55, 1.0000]],
    'Fe': [[54, 0.05845], [56, 0.91754], [57, 0.02119], [58, 0.00282]],
    'Co': [[59, 1.0000]],
    'Ni': [[58, 0.680769], [60, 0.262231], [61, 0.011399], [62, 0.036345],
           [64, 0.009256]],
    'Cu': [[63, 0.6915], [65, 0.3085]],
    'Zn': [[64, 0.48268], [66, 0.27975], [67, 0.04102], [68, 0.19024],
           [70, 0.00631]],
    'Ga': [[69, 0.60108], [71, 0.39892]],
    'Ge': [[70, 0.2038], [72, 0.2731], [73, 0.0776], [74, 0.3672],
           [76, 0.0783]],
    'As': [[75, 1.0000]],
    'Se': [[74, 0.0089], [76, 0.0937], [77, 0.0763], [78, 0.2377],
           [80, 0.4961], [82, 0.0873]],
    'Br': [[79, 0.5069], [81, 0.4931]],
    'Kr': [[78, 0.00355], [80, 0.02286], [82, 0.11593], [83, 0.11500],
           [84, 0.56987], [86, 0.17279]],
    'Rb': [[85, 0.7217], [87, 0.2783]],
    'Sr': [[84, 0.0056], [86, 0.0986], [87, 0.0700], [88, 0.8258]],
    'Y':  [[89, 1.0000]],
    'Zr': [[90, 0.5145], [91, 0.1122], [92, 0.1715], [94, 0.1738],
           [96, 0.0280]],
    'Nb': [[93, 1.0000]],
    'Mo': [[92, 0.1477], [94, 0.0923], [95, 0.1590], [96, 0.1668],
           [97, 0.0956], [98, 0.2419], [100, 0.0967]],
    'Tc': None,
    'Ru': [[96, 0.0554], [98, 0.0187], [99, 0.1276], [100, 0.1260],
           [101, 0.1706], [102, 0.3155], [104, 0.1862]],
    'Rh': [[103, 1.0000]],
    'Pd': [[102, 0.0102], [104, 0.1114], [105, 0.2233], [106, 0.2733],
           [108, 0.2646], [110, 0.1172]],
    'Ag': [[107, 0.51839], [109, 0.48161]],
    'Cd': [[106, 0.0125], [108, 0.0089], [110, 0.1249], [111, 0.1280],
           [112, 0.2413], [113, 0.1222], [114, 0.2873], [116, 0.0749]],
    'In': [[113, 0.0429], [115, 0.9571]],
    'Sn': [[112, 0.0097], [114, 0.0066], [115, 0.0034], [116, 0.1454],
           [117, 0.0768], [118, 0.2422], [119, 0.0859], [120, 0.3258],
           [112, 0.0463], [124, 0.0579]],
    'Sb': [[121, 0.5721], [123, 0.4279]],
    'Te': [[120, 0.0009], [122, 0.0255], [123, 0.0089], [124, 0.0474],
           [125, 0.0707], [126, 0.1884], [128, 0.3174], [130, 0.3408]],
    'I':  [[127, 1.0000]],
    'Xe': [[124, 0.000952], [126, 0.000890], [128, 0.019102], [129, 0.264006],
           [130, 0.040710], [131, 0.212324], [132, 0.269086], [134, 0.104357],
           [136, 0.088573]],
    'Cs': [[133, 1.0000]],
    'Ba': [[130, 0.00106], [132, 0.00101], [134, 0.02417], [135, 0.06592],
           [136, 0.07854], [137, 0.11232], [138, 0.71698]],
    'La': [[138, 0.00090], [139, 0.99910]],
    'Ce': [[136, 0.00185], [138, 0.00251], [140, 0.88450], [142, 0.11114]],
    'Pr': [[141, 1.0000]],
    'Nd': [[142, 0.272], [143, 0.122], [144, 0.238], [145, 0.083],
           [146, 0.172], [148, 0.057], [150, 0.056]],
    'Pm': None,
    'Sm': [[144, 0.0307], [147, 0.1499], [148, 0.1124], [149, 0.1382],
           [150, 0.0738], [152, 0.2675], [154, 0.2275]],
    'Eu': [[151, 0.4781], [153, 0.5219]],
    'Gd': [[152, 0.0020], [154, 0.0218], [155, 0.1480], [156, 0.2047],
           [157, 0.1565], [158, 0.2484], [160, 0.2186]],
    'Tb': [[159, 1.0000]],
    'Dy': [[156, 0.00056], [158, 0.00095], [160, 0.02329], [161, 0.18889],
           [162, 0.25475], [163, 0.24896], [164, 0.28260]],
    'Ho': [[165, 1.0000]],
    'Eu': [[162, 0.00139], [164, 0.01601], [166, 0.33503], [167, 0.22869],
           [168, 0.26978], [170, 0.14910]],
    'Tm': [[169, 1.0000]],
    'Yb': [[168, 0.0013], [170, 0.0304], [171, 0.1428], [172, 0.2183],
           [173, 0.1613], [174, 0.3183], [176, 0.1276]],
    'Lu': [[175, 0.9741], [176, 0.0259]],
    'Hf': [[174, 0.0016], [176, 0.0526], [177, 0.1860], [178, 0.2728],
           [179, 0.1362], [180, 0.3508]],
    'Ta': [[180, 0.00012], [181, 0.99988]],
    'W':  [[180, 0.0012], [182, 0.2650], [183, 0.1431], [184, 0.3064],
           [186, 0.2843]],
    'Re': [[185, 0.3740], [187, 0.6260]],
    'Os': [[184, 0.0002], [186, 0.0159], [187, 0.0196], [188, 0.1324],
           [189, 0.1615], [190, 0.2626], [192, 0.4078]],
    'Ir': [[191, 0.373], [193, 0.627]],
    'Pt': [[190, 0.00014], [192, 0.00782], [194, 0.32967], [195, 0.33832],
           [196, 0.25242], [198, 0.07163]],
    'Au': [[197, 1.0000]],
    'Hg': [[196, 0.0015], [198, 0.0997], [199, 0.1687], [200, 0.2310],
           [201, 0.1318], [202, 0.2986], [204, 0.0687]],
    'Tl': [[203, 0.2952], [205, 0.7048]],
    'Pb': [[204, 0.014], [206, 0.241], [207, 0.221], [208, 0.524]],
    'Bi': [[209, 1.0000]],
    'Po': None,
    'At': None,
    'Rn': None,
    'Fr': None,
    'Ra': None,
    'Ac': None,
    'Th': [[232, 1.0000]],
    'Pa': [[231, 1.0000]],
    'U':  [[234, 0.000054], [235, 0.007204], [238, 0.992742]]
}
