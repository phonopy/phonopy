# Commands to run this example

head="
PAO.EnergyShift 0.1 eV
DM.Tolerance 0.0001
DM.MixingWeight 0.2
Use.New.Diagk .true.
SystemLabel Gr
PAO.BasisSize sz
SolutionMethod diagon
MaxSCFIterations 120
SCF.MixAfterConvergence .false.
MeshCutoff 400 Ry
NumberOfSpecies 1
DM.UseSaveDM False
ElectronicTemperature 0.02 eV
%block ChemicalSpeciesLabel
1 6 C
%endblock ChemicalSpeciesLabel
"

kgrid_uc="
%block kgrid_Monkhorst_Pack
  30   0   0  0.0
   0  30   0  0.0
   0   0   1  0.0
%endblock Kgrid_Monkhorst_Pack
"

kgrid_sc="
%block kgrid_Monkhorst_Pack
  10   0   0  0.0
   0  10   0  0.0
   0   0   1  0.0
%endblock Kgrid_Monkhorst_Pack
"

atoms_uc="
NumberOfAtoms       2
LatticeConstant 1.000000 Ang
AtomicCoordinatesFormat NotScaledCartesianAng
%block LatticeVectors
 2.55000000000000 0.00000000000000 0.00000000000000
 -1.27500000000000 2.20836477965032 0.00000000000000
 0.00000000000000 0.00000000000000 30.00000000000000
%endblock LatticeVectors

AtomicCoordinatesFormat NotScaledCartesianAng

%block AtomicCoordinatesAndAtomicSpecies
0.00000000000000     0.00000000000000     0.00000000000000 1 12.011000
1.27500000000000     0.73612159321677     0.00000000000000 1 12.011000
%endblock AtomicCoordinatesAndAtomicSpecies
"

echo "$head" > Gr.fdf
echo "$kgrid_uc" >> Gr.fdf
echo "$atoms_uc" >> Gr.fdf
phonopy --siesta -d --dim="3 3 1" -c Gr.fdf --amplitude=0.02
mkdir disp-001
cp C.psf supercell-001.fdf disp-001
cd disp-001
echo "$head" > Gr.fdf
echo "$kgrid_sc" >> Gr.fdf
echo "LatticeConstant 1.0 Bohr">> Gr.fdf
echo "%include supercell-001.fdf" >> Gr.fdf
#siesta < Gr.fdf
cd ..
phonopy --siesta -f disp-001/Gr.FA -c Gr.fdf
cat > band.conf << EOF
ATOM_NAME = C
DIM =  3 3 1
BAND_POINTS = 50
BAND = 0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0
EOF
phonopy --siesta -p band.conf -c Gr.fdf
