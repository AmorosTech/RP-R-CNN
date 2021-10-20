#!/usr/bin/env bash

# export CXXFLAGS="-std=c++11"
# export CFLAGS="-std=c99"

cd models/ops

echo "Building bbox op..."
python3 setup_ssd.py build_ext --inplace
rm -rf build

echo "Building rcnn op..."
if [ -d "build" ]; then
    rm -r build
fi
python3 setup_rcnn.py build_ext --inplace
rm -r build
