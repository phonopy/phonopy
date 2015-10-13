#Commands to run this example:

head="
SystemName          silicon
SystemLabel         Si

NumberOfSpecies     1

%block ChemicalSpeciesLabel
 1  14  Si
%endblock ChemicalSpeciesLabel

PAO.BasisSize       sz

MeshCutoff         400.0 Ry

MaxSCFIterations    50
DM.MixingWeight      0.2
DM.NumberPulay       3
DM.Tolerance         1.d-4
DM.UseSaveDM

SolutionMethod       diagon

WriteForces          .true.

ElectronicTemperature  100 K

AtomicCoordinatesFormat  Fractional
"

kgrid_uc="
%block kgrid_Monkhorst_Pack
   8   0   0  0.0
   0   8   0  0.0
   0   0   8  0.0
%endblock Kgrid_Monkhorst_Pack
"

kgrid_sc="
%block kgrid_Monkhorst_Pack
   3   0   0  0.0
   0   3   0  0.0
   0   0   3  0.0
%endblock Kgrid_Monkhorst_Pack
"

atoms_uc="
NumberOfAtoms       2
LatticeConstant     5.430 Ang
%block LatticeVectors
  0.000  0.500  0.500
  0.500  0.000  0.500
  0.500  0.500  0.000
%endblock LatticeVectors

%block AtomicCoordinatesAndAtomicSpecies
    0.00    0.00    0.00   1 #  Si  1
    0.25    0.25    0.25   1 #  Si  2
%endblock AtomicCoordinatesAndAtomicSpecies
"

echo "$head" > Si.fdf
echo "$kgrid_uc" >> Si.fdf
echo "$atoms_uc" >> Si.fdf
phonopy --siesta -d --dim="3 3 3" -c Si.fdf --amplitude=0.04
mkdir disp-001
cp Si.psf supercell-001.fdf disp-001
cd disp-001
echo "$head" > Si.fdf
echo "$kgrid_sc" >> Si.fdf
echo "LatticeConstant 1.0 Bohr">> Si.fdf
echo "%include supercell-001.fdf" >> Si.fdf
siesta < Si.fdf
cd ..
phonopy --siesta -f disp-001/Si.FA -c Si.fdf 
cat > band.conf << EOF
ATOM_NAME = Si O
DIM =  3 3 3
BAND_POINTS = 100
BAND = 1/2 1/2 1/2  0.0 0.0 0.0  0.0 1/2 1/2  1.0 1.0 1.0
EOF
phonopy --siesta -p band.conf -c Si.fdf
