#!/bin/bash

dim="2 2 2"
pa="0 1/2 1/2 1/2 0 1/2 1/2 1/2 0"
mp="41 41 41"
nunit=32

while getopts ':mrc' opt; do
  case $opt in
    m)
      for i in `/bin/ls QHAPOSCAR*`;do
	qhaname=`echo $i|sed s/POSCAR//`
	mkdir $qhaname
	mv $i $qhaname/POSCAR
	cd $qhaname
	pwd
	phonopy -d --dim="$dim"
	for j in `/bin/ls POSCAR-*`;do
	  dispname=`echo $j|sed s/POSCAR/disp/`
	  mkdir $dispname
	  mv $j $dispname/POSCAR
	  cp ../{INCAR,KPOINTS,POTCAR} $dispname
	done
	mkdir perfect
	cp ../{INCAR,KPOINTS,POTCAR} perfect
	cp SPOSCAR perfect/POSCAR
	cd ..
      done
      ;;
    r)
      for i in `/bin/ls -d QHA-*`;do
	cd $i
	pwd
	vol=`echo $i|sed s/QHA-//`
	# For displacements
  	for j in `/bin/ls -d disp-*`;do
  	  cd $j
  	  pwd
  	  num=`echo $j|sed s/disp-//`
  	  sed s/num/$num/ ../../calc.sh|sed s/vol/$vol/|qsub
  	  cd ..
  	done
	# For perfect
	cd perfect
	  sed s/num/per/ ../../calc.sh|sed s/vol/$vol/|qsub
	cd ..
	cd ..
      done
      ;;
    c)
      echo "#   cell volume        energy of cell other than phonon" > e-v.dat
      for i in `/bin/ls -d QHA-*`;do
	cd $i
 	phonopy -f disp-*/vasprun.xml
 	phonopy --mp="$mp" -t --dim="$dim" --pa="$pa" --tmax=2004 --tstep=2
	volume=`grep volume perfect/vasprun.xml|tail -n 1|awk -F'<|>' '{printf("%20.13f", $3)}'`
	energy=`grep e_wo_entrp perfect/vasprun.xml|tail -n 1|awk -F'<|>' '{printf("%20.13f", $3)}'`
	cd ..
	echo `echo "$volume/$nunit"|bc -ls` `echo "$energy/$nunit"|bc -ls` >> e-v.dat
      done
      ;;
  esac
done
