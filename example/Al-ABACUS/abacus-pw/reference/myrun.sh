#!/bin/bash

phonopy setting.conf --abacus -d
echo 'stru_file ./STRU-001' >> INPUT
OMP_NUM_THREADS=1  mpirun -n 8 abacus > log
phonopy -f OUT.ABACUS/running_scf.log
phonopy band.conf --abacus
