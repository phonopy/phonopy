#!/bin/bash

for i in supercell-*; do
    bn=${i%.*}
    num=${bn#*supercell-}
    if [[ -z ${num} ]]; then
        continue
    fi
    dir="DISP-"$num
    if [[ -d $dir ]]; then
        echo "Error. Directory $dir already exist."
        exit 1
    fi

    mkdir $dir

    isIDEAL=`echo $num | awk '{printf("%d", $0)}'`
    if [[ isIDEAL -eq "0" ]]; then
        cp polar.inp $dir
    fi

    cp $i $dir/positions.xyz
    cp force.inp $dir
done
