#!/bin/bash

PARAM=$1
TAIL=$2
CWD=`pwd`

if [[ ! -f $PARAM ]]; then
    echo "Error param file does not exist."
    echo "Usage $0 param_file tail.cell"
    exit 1
fi

if [[ ! -f $TAIL ]]; then
    echo "Error Tail file does not exist."
    echo "Usage $0 param_file tail.cell"
    exit 1
fi

for file in supercell-*; do
	fn=${file%.cell}
	num=${fn##*-}
	if [[ -d "displ-$num" ]]; then
	    echo "Error directory displ-$num already exist"
	    exit 1
	fi
	echo "Making displ-$num directory"
        mkdir displ-$num
	cp $file displ-$num/supercell.cell
        cp $PARAM displ-$num/supercell.param
	cat $TAIL >> "displ-$num"/supercell.cell
done
