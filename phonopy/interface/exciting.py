"""exciting calculator interface."""

# Copyright (C) 2026 Martí Raya Moreno
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

import io
import os
import sys
import typing
import warnings
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy import Phonopy
from phonopy.file_IO import (
    iter_collect_forces,
    write_FORCE_CONSTANTS,
    write_force_constants_to_hdf5,
)
from phonopy.harmonic.force_constants import distribute_force_constants_by_translations
from phonopy.interface.vasp import (
    check_forces,
    get_drift_forces,
    get_scaled_positions_lines,
)
from phonopy.physical_units import get_physical_units
from phonopy.structure.atomic_data import get_atomic_data
from phonopy.structure.atoms import PhonopyAtoms, split_symbol_and_index
from phonopy.structure.cells import Primitive, get_primitive, get_supercell


def read_species_symbol(speciesfile: str | os.PathLike | typing.IO) -> str:
    """Extract chemical symbol from exciting species XML file.
    
    Args:
        speciesfile: Path to species XML file (e.g., "Li.xml") or file-like object
        
    Returns:
        Chemical symbol (e.g., "Li")
    """
    if isinstance(speciesfile, io.IOBase):
        xml_content = speciesfile.read()
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode('utf-8')
    else:
        with open(speciesfile, 'r') as f:
            xml_content = f.read()
    
    root = ET.fromstring(xml_content)
    sp_elem = root.find('.//sp')
    if sp_elem is None:
        raise ValueError(f"No 'sp' element found in species file: {speciesfile}")
    
    chemical_symbol = sp_elem.get('chemicalSymbol')
    if chemical_symbol is None:
        raise ValueError(f"No 'chemicalSymbol' attribute in species file: {speciesfile}")
    
    return chemical_symbol


def read_exciting(filename: str | os.PathLike | typing.IO) -> PhonopyAtoms:
    """Read exciting structure.
    
    Parses exciting XML format to extract lattice vectors and atomic positions.
    Reads chemical symbols from species files located in the speciespath directory.
    
    Args:
        filename: Path to exciting XML file or file-like object
        
    Returns:
        PhonopyAtoms object with parsed structure
    """

    # Read XML content
    if isinstance(filename, io.IOBase):
        xml_content = filename.read()
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode('utf-8')
        input_dir = "."
    else:
        input_dir = os.path.dirname(os.path.abspath(filename))
        with open(filename, 'r') as f:
            xml_content = f.read()
    
    # Parse XML
    root = ET.fromstring(xml_content)
    
    # Extract lattice vectors from crystal/basevect
    crystal = root.find('.//crystal')
    if crystal is None:
        raise ValueError("No 'crystal' element found in XML")
    
    # Get scale factor if present (default: 1.0)
    scale = float(crystal.get('scale', 1.0))
    
    basevects = []
    for basevect_elem in crystal.findall('basevect'):
        coords = [float(x) for x in basevect_elem.text.split()]
        basevects.append(coords)
    
    if len(basevects) != 3:
        raise ValueError(f"Expected 3 basis vectors, got {len(basevects)}")
    
    lattice         = np.array(basevects).T * scale  # Apply scale factor
    inverse_lattice = np.linalg.inv(lattice)
    
    # Get speciespath from structure element
    structure = root.find('.//structure')
    if structure is not None:
        structure_speciespath = structure.get('speciespath', '.')
        # Resolve path relative to input file directory
        if os.path.isabs(structure_speciespath):
            speciespath = structure_speciespath
        else:
            speciespath = os.path.normpath(os.path.join(input_dir, structure_speciespath))
    else:
        speciespath = input_dir
    
    # Check if coordinates are in Cartesian format (default: False)
    cartesian = structure.get('cartesian', 'false').lower() == 'true' if structure is not None else False
    
    # Extract atomic positions and species symbols
    symbols = []
    positions = []
    
    for species_elem in root.findall('.//species'):
        # Get species file path
        speciesfile = species_elem.get('speciesfile', '')
        if not speciesfile:
            raise ValueError("Species element missing 'speciesfile' attribute")
        
        # Try to read chemical symbol from species file
        try:
            species_file_path = os.path.join(speciespath, speciesfile)
            if os.path.exists(species_file_path):
                chemical_symbol = read_species_symbol(species_file_path)
            else:
                # Fallback: extract from filename. 
                # This is not safe, as in exciting the user can
                # rename its species file as it desires. But
                # I found it worth to allow Phonopy usage without 
                # species selection.
                chemical_symbol = speciesfile.replace('.xml', '').replace('.XML', '')
                warnings.warn(
                    f"Species file not found: {species_file_path}. "
                    f"Using fallback symbol: {chemical_symbol}",
                    UserWarning
                )
        except (ValueError, FileNotFoundError) as e:
            # Fallback: extract from filename.
            # This is not safe, as in exciting the user can
            # rename its species file as it desires.
            # I found it worth to allow Phonopy usage without 
            # species selection.
            chemical_symbol = speciesfile.replace('.xml', '').replace('.XML', '')
            warnings.warn(
                f"Could not read species file {species_file_path}: {e}. "
                f"Using fallback symbol: {chemical_symbol}",
                UserWarning
            )
        
        if not chemical_symbol:
            raise ValueError(f"Could not extract species symbol from {speciesfile}")
        
        # Extract all atoms for this species
        for atom_elem in species_elem.findall('atom'):
            coord_str = atom_elem.get('coord', '')
            if coord_str:
                # Parse coordinates
                coords = [float(x) for x in coord_str.split()]
                
                # Convert from Cartesian to fractional if needed
                if cartesian:
                    # coords are in Cartesian, convert to fractional
                    cart_coords = np.array(coords)
                    frac_coords = np.dot(inverse_lattice, cart_coords)
                else:
                    # coords are already fractional
                    frac_coords = coords
                
                symbols.append(chemical_symbol)
                positions.append(frac_coords)
    
    if not symbols:
        raise ValueError("No atoms found in structure")
    
    scaled_positions = np.array(positions)
    
    # Return PhonopyAtoms object
    return PhonopyAtoms(
        symbols=symbols,
        cell=lattice,
        scaled_positions=scaled_positions
    )


def get_exciting_structure(cell: PhonopyAtoms) -> str:
    """Return exciting structure in xml tree."""
    
    # Extract data from PhonopyAtoms object
    lattice = phonopy_atoms.cell.T
    scaled_positions = phonopy_atoms.scaled_positions
    chemical_symbols = phonopy_atoms.symbols
    
    # Create root element
    root = ET.Element('input')
    
    # Add title
    title_el#
# write vasp POSCAR
#
def write_vasp(
    filename: str | os.PathLike,
    cell: PhonopyAtoms,
    direct: bool = True,
    expand_mixtures: bool = False,
) -> None:
    """Write crystal structure to a VASP POSCAR style file.

    Parameters
    ----------
    filename : str
        Filename.
    cell : PhonopyAtoms
        Crystal structure.
    direct : bool, optional
        In 'Direct' or not in VASP POSCAR format. Default is True.
    expand_mixtures : bool, optional
        When True and ``cell`` has mixed-species sites, expand each mixture
        into one POSCAR row per constituent at the same fractional
        coordinates so the file is consumable by a VASP VCA calculation.
        Has no effect on cells without mixtures. Default is False.

    """
    lines = get_vasp_structure_lines(
        cell, direct=direct, expand_mixtures=expand_mixtures
    )
    with open(filename, "w") as w:
        w.write("\n".join(lines))em = ET.SubElement(root, 'title')
    title_elem.text = title
    
    # Create structure element with all attributes
    structure_attrib = {
        'speciespath': '.',
        'cartesian': 'false',
        'tshift': 'false'
    }
    
    structure = ET.SubElement(root, 'structure', attrib=structure_attrib)
    
    # Create crystal element
    crystal_attrib = {}
    crystal = ET.SubElement(structure, 'crystal')
    
    for i in range(3):
        basevect = ET.SubElement(crystal, 'basevect')
        basevect.text = '   {:16.10f}   {:16.10f}   {:16.10f}'.format(
            lattice[i, 0],
            lattice[i, 1],
            lattice[i, 2]
        )
    
    # Group atoms by species
    species_dict = OrderedDict()
    for i, symbol in enumerate(chemical_symbols):
        if symbol not in species_dict:
            species_dict[symbol] = []
        species_dict[symbol].append(positions[i])
    
    # Create species elements
    for symbol, atom_positions in species_dict.items():
        species = ET.SubElement(
            structure,
            'species',
            attrib={
                'speciesfile': f'{symbol}.xml'
            }
        )
        
        for atom_pos in atom_positions:
            atom = ET.SubElement(species, 'atom')
            atom.set(
                'coord',
                '   {:16.10f}   {:16.10f}   {:16.10f}'.format(
                    atom_pos[0],
                    atom_pos[1],
                    atom_pos[2]
                )
            )

    return ET.tostring(root, encoding="unicode")


def write_exciting(
    filename: str | os.PathLike,
    cell: PhonopyAtoms
) -> None:
    lines = get_exciting_structure(cell)
    with open(filename, "w") as w:
        w.write("\n".join(lines))

def write_supercells_with_displacements(
    supercell: PhonopyAtoms,
    cells_with_displacements: Sequence[PhonopyAtoms],
    ids: NDArray[np.int64] | Sequence[int],
    pre_filename: str | os.PathLike = "supercell",
    width: int = 3,
) -> None:
    """Write supercells with displacements to files."""
    write_exciting("%s.xml" % pre_filename, supercell)
    for i, cell in zip(ids, cells_with_displacements, strict=True):
        filename = "{pre_filename}-{0:0{width}}.in".format(
            i, pre_filename=pre_filename, width=width
        )
        write_exciting(filename, cell)

def parse_set_of_forces(
    num_atoms: int,
    forces_filenames: Sequence[str | os.PathLike],
    verbose: bool = True,
) -> list[NDArray[np.double]]:
    """Parse forces from output files."""
    hook = "Total atomic forces including IBS (cartesian) :"
    is_parsed = True
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        f = open(filename)
        exciting_forces = collect_forces(f, num_atoms, hook, [4, 5, 6])
        if check_forces(exciting_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                exciting_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(exciting_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


