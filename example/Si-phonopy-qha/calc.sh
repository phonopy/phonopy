#$ -S /bin/zsh
#$ -cwd
#$ -N Sivol-num
#$ -pe mpi* 4
#$ -e err.log
#$ -o std.log

cp $TMPDIR/machines machines
mpirun  vasp5212mpi >& ../progress-num
