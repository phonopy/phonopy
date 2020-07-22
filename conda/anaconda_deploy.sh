#!/bin/bash

export LBL=$1
export TKN=$2
export GIT_BRANCH=$3

cd conda
sed s/version_from_shell/`git describe --tags --dirty | sed -e 's/-\(.*\)-g.*/+\1/' -e 's/^[vr]//g' -e 's/rc-/rc./' -e 's/-dirty//'`/ meta.yaml > meta.tmp.yaml
mv meta.tmp.yaml meta.yaml
echo "-----------------------"
echo "GIT_BRANCH: $GIT_BRANCH"
echo "-----------------------"
cd ..
conda install conda-build anaconda-client --yes
conda build conda -c https://conda.anaconda.org/conda-forge --no-anaconda-upload
TRG=`conda build conda -c https://conda.anaconda.org/conda-forge --output |sed -e 's/--/-*-/'`
echo "Uploading: $TRG"
anaconda --token $TKN upload --skip-existing --label $LBL $TRG
