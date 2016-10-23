#!/bin/bash

export LBL=$1
export TKN=$2
export GIT_BRANCH=$3

conda install conda-build anaconda-client --yes
conda build conda --no-anaconda-upload
TRG=`conda build conda --output |sed -e 's/--/-*-/'`
echo "Uploading: $TRG"
anaconda --token $TKN upload --label $LBL $TRG

