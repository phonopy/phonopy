#!/bin/bash
set -e -x

yum install -y hdf5 freetype-devel

for PYBIN in /opt/python/*/bin; do
    if [[ $PYBIN == *"27"* ]] || [[ $PYBIN == *"35"* ]] || [[ $PYBIN == *"36"* ]] || [[ $PYBIN == *"37"* ]]; then
        "${PYBIN}/pip" install -r /io/dev-requirements.txt
        "${PYBIN}/pip" wheel /io/ -w wheelhouse/
    fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ $PYBIN == *"27"* ]] || [[ $PYBIN == *"35"* ]] || [[ $PYBIN == *"36"* ]] || [[ $PYBIN == *"37"* ]]; then
        "${PYBIN}/pip" install phonopy --no-index -f /io/wheelhouse
        (cd "$HOME"; "${PYBIN}/nosetests" phonopy)
    fi
done
